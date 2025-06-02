# -*- coding: utf-8 -*-
"""Main module."""
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch import nn as nn
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily, Normal
from torch.distributions import kl_divergence as kl

from ._constants import REGISTRY_KEYS

torch.backends.cudnn.benchmark = True


class DecoderVELOVI(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    linear_decoder
        Whether to use linear decoder for time
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        linear_decoder: bool = False,
        shared_time: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.n_output = n_output
        self.linear_decoder = linear_decoder
        self.shared_time = shared_time
        self.rho_first_decoder_up = FCLayers(
            n_in=n_input,
            n_out=n_hidden if not linear_decoder else n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers if not linear_decoder else 1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm if not linear_decoder else False,
            use_activation=not linear_decoder,
            bias=not linear_decoder,
            **kwargs,
        )

        self.rho_first_decoder_down = FCLayers(
            n_in=n_input,
            n_out=n_hidden if not linear_decoder else n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers if not linear_decoder else 1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm if not linear_decoder else False,
            use_activation=not linear_decoder,
            bias=not linear_decoder,
            **kwargs,
        )

        self.pi_first_decoder_up = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        self.pi_first_decoder_down = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # two categorical state assignments with two states each to model a gene allowing
        # either repression ss + upregulation or activation ss + downregulation
        self.px_pi_decoder_up = nn.Linear(n_hidden, 2 * n_output)
        self.px_pi_decoder_down = nn.Linear(n_hidden, 2 * n_output)

        self.px_rho_decoder_up = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())
        self.px_rho_decoder_down = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

    def forward(self, z: torch.Tensor, latent_dim: int = None):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        z_in = z
        if latent_dim is not None:
            mask = torch.zeros_like(z)
            mask[..., latent_dim] = 1
            z_in = z * mask

        if not self.linear_decoder:
            rho_first = self.rho_first_decoder_up(z_in)
            px_rho_up = self.px_rho_decoder_up(rho_first)

            rho_first = self.rho_first_decoder_down(z_in)
            px_rho_down = self.px_rho_decoder_up(rho_first)
        else:
            rho_first = self.rho_first_decoder(z_in)
            px_rho = nn.Sigmoid()(rho_first)

        if self.shared_time:
            px_rho_up = px_rho_up[:, 0].unsqueeze(1).repeat(1, px_rho_up.shape[1])
            px_rho_down = px_rho_up

        px_rho = torch.concat([px_rho_up.unsqueeze(-1), px_rho_down.unsqueeze(-1)], axis=-1)

        # two times cells by genes by 2
        pi_first_up = self.pi_first_decoder_up(z)
        pi_first_down = self.pi_first_decoder_down(z)

        px_pi_up = nn.Softplus()(
            torch.reshape(self.px_pi_decoder_up(pi_first_up), (z.shape[0], self.n_output, 2))
        )

        px_pi_down = nn.Softplus()(
            torch.reshape(self.px_pi_decoder_down(pi_first_down), (z.shape[0], self.n_output, 2))
        )

        return px_pi_up, px_pi_down, px_rho


class VELOVAE(BaseModuleClass):

    def __init__(
        self,
        n_input: int,
        n_dim_glue: int,
        upper_ss_prot,
        lower_ss_prot,
        upper_ss_rna,
        lower_ss_rna,
        dpt_start_rna=None,
        dpt_start_prot=None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        latent_distribution: str = "normal",
        use_batch_norm="both",
        use_layer_norm="both",
        var_activation: Optional[Callable] = torch.nn.Softplus(),
        model_steady_states: bool = True,
        penalty_scale: float = 0.2,
        param_loss_weight: float = 10000,
        kl_scaling: float = 1,
        kl_z_scaling: float = 1,
        kl_ss_scaling: float = 1,
        time_loss_weight: float = 0,
        dirichlet_concentration: float = 0.25,
        linear_decoder: bool = False,
        flexible_switch_time: bool = True,
        shared_time: bool = True,
    ):
        """
        Parameters
        ----------
        n_input
            Number of input genes.
        n_dim_glue
            Dimension of used representation for encoder input.
        upper_ss_prot
            Median expression of upper 1% percentile of proteins.
        lower_ss_prot
            Median expression of lower 1% percentile of proteins.
        upper_ss_rna
            Median expression of upper 1% percentile of rnas.
        lower_ss_rna
            Median expression of lower 1% percentile of rnas.
        dpt_start_rna
            (Optional) Median rna expression of cells within lower 1% percentile wrt pseudotime.
        dpt_start_prot
            (Optional) Median protein expression of cells within lower 1% percentile wrt pseudotime.
        n_hidden
            Number of nodes per hidden layer.
        n_latent
            Dimensionality of the latent space.
        n_layers
            Number of hidden layers used for encoder and decoder NNs.
        dropout_rate
            Dropout rate for neural networks.
        latent_distribution
            One of

            * ``'normal'`` - Isotropic normal
            * ``'ln'`` - Logistic normal with normal params N(0, 1)
        use_batch_norm
            Whether to use batch norm in layers.
        use_layer_norm
            Whether to use layer norm in layers.
        var_activation
            Callable used to ensure positivity of the variational distributions' variance.
            When `None`, defaults to `torch.exp`.
        model_steady_states
            If 'True', per cell a variable is learned to decide whether that cell is in steady state or dynamic state.
            Otherwise all cells are in dynamic state.
        penalty_scale
            Loss weighting factor to encourage initial states to be close to upper or lower 1% percentile of the data
            or start expression determined via pseudotime prior.
        param_loss_weight
            Loss weighting factor to enforce (!) restrictions on parameters.
        kl_scaling
            Loss weighting factor for KL loss on gene states (activation vs repression).
        kl_z_scaling
            Loss weighting factor for KL loss on latent space.
        kl_ss_scaling
            Loss weighting factor for KL loss on cell states (steady state vs dynamic state).
        time_loss_weight
            Loss weighting factor for time prior.
        dirichlet_concentration
            Dirichlet concentrations used as prior on gene and cell states.
        linear_decoder
            Whether to use a linear decoder or not.
        flexible_switch_time
            If 'True', per gene a time point is learned where cells switch from steady to dynamic state. In the repression
            case, this switch point can also be negative such that the unobserved switch point from activation to repression
            doesn't coincide with the observed initial state of repression. Important when time is shared between genes.
        shared_time
            Whether time is shared across genes or fitted individually for every gene.
        """
        super().__init__()
        self.n_latent = n_latent
        self.latent_distribution = latent_distribution
        self.n_input = n_input
        self.n_dim_glue = n_dim_glue
        self.use_time_prior = dpt_start_rna is not None and dpt_start_prot is not None
        self.model_steady_states = model_steady_states
        self.param_loss_weight = param_loss_weight
        self.kl_scaling = kl_scaling
        self.kl_z_scaling = kl_z_scaling
        self.kl_ss_scaling = kl_ss_scaling
        self.time_loss_weight = time_loss_weight
        self.penalty_scale = penalty_scale
        self.dirichlet_concentration = dirichlet_concentration
        self.flexible_switch_time = flexible_switch_time
        self.shared_time = shared_time

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_genes = n_input * 2

        self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_input, 2))
        self.kappa_mean_unconstr = torch.nn.Parameter(.5 * torch.ones(n_input, 2))

        self.beta_mean_unconstr = torch.nn.Parameter(.5 * torch.ones(n_input, 2))
        self.alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input, 3))  # up, up for down, down

        self.switch_time_unconstr = torch.nn.Parameter(15 * torch.ones(n_input))
        self.switch_time_ss_up_unconstr = torch.nn.Parameter(.2 * torch.ones(n_input))
        self.switch_time_ss_down_unconstr = torch.nn.Parameter(.2 * torch.ones(n_input))

        if self.use_time_prior:
            self.dpt_start_rna = torch.from_numpy(dpt_start_rna).to(device)
            self.dpt_start_prot = torch.from_numpy(dpt_start_prot).to(device)
        else:
            self.dpt_start_rna = None
            self.dpt_start_prot = None

        # defines whether gene does induction or repression
        self.px_pi_global = torch.nn.Parameter(self.dirichlet_concentration * torch.ones(n_input, 2))

        # likelihood dispersion
        # for now, with normal dist, this is just the variance
        self.scale_unconstr = torch.nn.Parameter(-1 * torch.ones(n_genes, 2))

        self.upper_ss_prot = torch.from_numpy(upper_ss_prot).to(device)
        self.lower_ss_prot = torch.from_numpy(lower_ss_prot).to(device)
        self.upper_ss_rna = torch.from_numpy(upper_ss_rna).to(device)
        self.lower_ss_rna = torch.from_numpy(lower_ss_rna).to(device)

        self.lower_start_rna = torch.nn.Parameter(self.lower_ss_rna.clone())
        self.lower_start_prot = torch.nn.Parameter(self.lower_ss_prot.clone())

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm_decoder

        self.z_encoder = Encoder(
            n_dim_glue,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = DecoderVELOVI(
            n_input_decoder,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            activation_fn=torch.nn.ReLU,
            linear_decoder=linear_decoder,
            shared_time=shared_time,
        )

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        gamma = inference_outputs["gamma"]
        kappa = inference_outputs["kappa"]
        beta = inference_outputs["beta"]
        alpha = inference_outputs["alpha"]
        pi_global = inference_outputs["pi_global"]

        input_dict = {
            "z": z,
            "gamma": gamma,
            "kappa": kappa,
            "beta": beta,
            "alpha": alpha,
            "pi_global": pi_global,
        }
        return input_dict

    def _get_rates(self):
        gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
        kappa = torch.clamp(F.softplus(self.kappa_mean_unconstr), 0, 50)
        beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)
        alpha = torch.clamp(F.softplus(self.alpha_unconstr), 0, 50)

        pi_global = F.softplus(self.px_pi_global)

        return gamma, kappa, beta, alpha, pi_global

    @auto_move_data
    def generative(self, z, gamma, kappa, beta, alpha, pi_global, latent_dim=None):
        """Runs the generative model."""
        decoder_input = z
        px_pi_alpha_up, px_pi_alpha_down, px_rho = self.decoder(decoder_input, latent_dim=latent_dim)
        px_pi_up = Dirichlet(px_pi_alpha_up).rsample()
        px_pi_down = Dirichlet(px_pi_alpha_down).rsample()
        pi_global = pi_global.unsqueeze(0).repeat(px_rho.shape[0], 1, 1)
        px_pi_global = Dirichlet(pi_global).rsample()

        scale_unconstr = self.scale_unconstr
        scale = F.softplus(scale_unconstr)

        mixture_dist_s, mixture_dist_u, end_penalty = self.get_px(
            px_pi_global,
            px_pi_up,
            px_pi_down,
            px_rho,
            scale,
            gamma=gamma,
            kappa=kappa,
            beta=beta,
            alpha=alpha,
        )

        return dict(
            px_pi_global=px_pi_global,  # Dirichlet output
            px_pi_up=px_pi_up,
            px_pi_down=px_pi_down,
            px_rho=px_rho,
            scale=scale,
            px_pi_alpha_up=px_pi_alpha_up,
            px_pi_alpha_down=px_pi_alpha_down,
            pi_global=pi_global,  # Dirichlet input
            mixture_dist_u=mixture_dist_u,
            mixture_dist_s=mixture_dist_s,
            end_penalty=end_penalty,
        )

    @auto_move_data
    def get_px(
        self,
        px_pi_global,
        px_pi_up,
        px_pi_down,
        px_rho,
        scale,
        gamma,
        kappa,
        beta,
        alpha,
    ) -> torch.Tensor:

        n_cells = px_rho.shape[0]
        n_genes = self.switch_time_unconstr.shape[0]

        # component dist
        # states are rep_ss, ind, act_ss, rep
        if self.model_steady_states:
            px_pi = torch.cat([px_pi_global[:, :, 0:1] * px_pi_up, px_pi_global[:, :, 1:2] * px_pi_down], dim=2)
        else:
            px_pi = px_pi_global

        comp_dist = Categorical(probs=px_pi)

        # induction
        t_offset = nn.Sigmoid()(self.switch_time_ss_up_unconstr) * self.flexible_switch_time
        mean_u_ind, mean_s_ind = self._get_induction_rna_protein(
            alpha=alpha[:, 0],
            beta=beta[:, 0],
            kappa=kappa[:, 0],
            gamma=gamma[:, 0],
            t=px_rho[:, :, 0],
            r_0=self.lower_start_rna,
            p_0=self.lower_start_prot,
            time_offset=t_offset,
        )

        mean_u_rep_steady = self.lower_start_rna.expand(n_cells, self.n_input)
        mean_s_rep_steady = self.lower_start_prot.expand(n_cells, self.n_input)

        # repression
        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), min=0)
        t_s = t_s.expand(1, -1).clone()
        zero_genes = torch.zeros(n_genes, device=t_s.device)
        r_0, p_0 = self._get_induction_rna_protein(
            alpha=alpha[:, 1],
            beta=beta[:, 1],
            kappa=kappa[:, 1],
            gamma=gamma[:, 1],
            t=t_s,
            r_0=zero_genes,
            p_0=zero_genes,
            time_offset=0,
        )
        t_offset = torch.clamp(self.switch_time_ss_down_unconstr, max=1) * self.flexible_switch_time
        mean_u_rep, mean_s_rep = self._get_induction_rna_protein(
            alpha=alpha[:, 2],
            beta=beta[:, 1],
            kappa=kappa[:, 1],
            gamma=gamma[:, 1],
            t=px_rho[:, :, 1],
            r_0=r_0,
            p_0=p_0,
            time_offset=t_offset,
        )

        mean_u_ind_steady = r_0.expand(n_cells, self.n_input)
        mean_s_ind_steady = p_0.expand(n_cells, self.n_input)

        scale_u = scale[: self.n_input, :].expand(n_cells, self.n_input, 2).sqrt()
        scale_s = scale[self.n_input:, :].expand(n_cells, self.n_input, 2).sqrt()

        if self.dpt_start_rna is not None:
            end_penalty_up = (self.lower_start_rna - self.dpt_start_rna).pow(2).mean() + \
                             (self.lower_start_prot - self.dpt_start_prot).pow(2).mean()
        else:
            end_penalty_up = (self.lower_start_rna - self.lower_ss_rna).pow(2).mean() \
                             + (self.lower_start_prot - self.lower_ss_prot).pow(2).mean()

        if self.dpt_start_rna is not None:
            end_penalty_down = (r_0 - self.dpt_start_rna).pow(2).mean() + \
                               (p_0 - self.dpt_start_prot).pow(2).mean()
        else:
            end_penalty_down = (r_0 - self.upper_ss_rna).pow(2).mean() \
            + (p_0 - self.upper_ss_prot).pow(2).mean()

        end_penalty = end_penalty_up + end_penalty_down

        if self.model_steady_states:
            mean_u = torch.stack([
                mean_u_rep_steady,
                mean_u_ind,
                mean_u_ind_steady,
                mean_u_rep,
            ], dim=2)
            scale_u = torch.stack([
                scale_u[..., 0],
                scale_u[..., 0],
                scale_u[..., 0],
                scale_u[..., 0],
            ], dim=2)
            mean_s = torch.stack([
                mean_s_rep_steady,
                mean_s_ind,
                mean_s_ind_steady,
                mean_s_rep,
            ], dim=2)
            scale_s = torch.stack([
                scale_s[..., 0],
                scale_s[..., 0],
                scale_s[..., 0],
                scale_s[..., 0],
            ], dim=2)
        else:
            mean_u = torch.stack([
                mean_u_ind,
                mean_u_rep,
            ], dim=2)
            scale_u = torch.stack([
                scale_u[..., 0],
                scale_u[..., 0],
            ], dim=2)
            mean_s = torch.stack([
                mean_s_ind,
                mean_s_rep,
            ], dim=2)
            scale_s = torch.stack([
                scale_s[..., 0],
                scale_s[..., 0],
            ], dim=2)

        dist_u = Normal(mean_u, scale_u)
        dist_s = Normal(mean_s, scale_s)

        mixture_dist_u = MixtureSameFamily(comp_dist, dist_u)
        mixture_dist_s = MixtureSameFamily(comp_dist, dist_s)

        return mixture_dist_s, mixture_dist_u, end_penalty

    def _get_induction_rna_protein(self, alpha, beta, kappa, gamma, t, r_0, p_0, time_offset=0):
        t_switch = time_offset
        t_orig = t
        t = t - t_switch
        rna = r_0 * torch.exp(-beta * t) + (alpha / beta) * (1 - torch.exp(-beta * t))

        protein = p_0 * torch.exp(-gamma * t) + alpha * kappa / beta / gamma * (
            1 - torch.exp(-gamma * t)
        ) + kappa * (r_0 * beta - alpha) / beta / (gamma - beta + 1e-6) * (
            torch.exp(-beta * t) - torch.exp(-gamma * t)
        )

        protein[:, gamma == beta] = 0

        n_cells = t.shape[0]
        rna_ss = r_0.expand(n_cells, -1).clone()
        protein_ss = p_0.expand(n_cells, -1).clone()

        mask_dyn = t_orig > t_switch
        mask_ss = t_orig <= t_switch

        rna[mask_ss] = 0
        rna_ss[mask_dyn] = 0

        rna = rna + rna_ss

        protein[mask_ss] = 0
        protein_ss[mask_dyn] = 0

        protein = protein + protein_ss

        return rna, protein


class VeloVAEPaired(VELOVAE):

    def _get_inference_input(self, tensors):
        glue = tensors['glue_embedding']

        input_dict = dict(
            glue=glue,
        )
        return input_dict

    @auto_move_data
    def inference(
        self,
        glue,
        n_samples=1,
    ):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """

        encoder_input = glue

        qz_m, qz_v, z = self.z_encoder(encoder_input)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        gamma, kappa, beta, alpha, pi_global = self._get_rates()

        outputs = dict(
            z=z, qz_m=qz_m, qz_v=qz_v, gamma=gamma, kappa=kappa, beta=beta, alpha=alpha, pi_global=pi_global
        )
        return outputs

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        protein = tensors[REGISTRY_KEYS.X_KEY]
        rna = tensors[REGISTRY_KEYS.U_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        alpha = inference_outputs['alpha']
        beta = inference_outputs['beta']
        kappa = inference_outputs['kappa']
        gamma = inference_outputs['gamma']

        # dirichlet parameters
        px_pi_alpha_up = generative_outputs["px_pi_alpha_up"]
        px_pi_alpha_down = generative_outputs["px_pi_alpha_down"]
        pi_global = generative_outputs["pi_global"]

        end_penalty = generative_outputs["end_penalty"]
        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]
        px_rho = generative_outputs["px_rho"]

        if self.use_time_prior and self.shared_time:
            time_prior = tensors['time_prior']
            time_loss = torch.nn.MSELoss()(time_prior, px_rho[:, :1, 0])
        else:
            time_loss = 0

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).mean(dim=1)

        # reconstruction loss
        reconst_loss_s = -mixture_dist_s.log_prob(protein)
        reconst_loss_u = -mixture_dist_u.log_prob(rna)
        reconst_loss = reconst_loss_u.mean(dim=-1) + reconst_loss_s.mean(dim=-1)

        # the default parameters of 0.25 give very strong incentive towards the 'corners'
        kl_pi_global = kl(
            Dirichlet(pi_global),
            Dirichlet(self.dirichlet_concentration * torch.ones_like(pi_global)),
        ).mean()

        prior_params = torch.from_numpy(
            np.array([self.dirichlet_concentration, self.dirichlet_concentration])[None, None, :]
        ).to(px_pi_alpha_up.device) * torch.ones_like(px_pi_alpha_up)

        # motivate clear assignment of a cell to either rep ss or activation (weighted equally)
        kl_pi_up = kl(
            Dirichlet(px_pi_alpha_up),
            Dirichlet(prior_params),
        ).sum(dim=-1)

        # motivate clear assignment of a cell to either act ss or repression (weighted equally)
        kl_pi_down = kl(
            Dirichlet(px_pi_alpha_down),
            Dirichlet(prior_params),
        ).sum(dim=-1)

        weighted_kl_local = self.kl_z_scaling * kl_weight * kl_divergence_z \
                            + kl_pi_global * self.kl_scaling  \
                            + (kl_pi_up + kl_pi_down) * self.kl_ss_scaling

        local_loss = torch.mean(reconst_loss + weighted_kl_local)

        r0 = self.lower_start_rna
        p0 = self.lower_start_prot

        param_loss = 0
        param_loss_rna = my_relu(r0 - alpha[:, 0] / beta[:, 0])
        mask = beta[:, 0] == 0
        param_loss_rna[mask] = 0
        param_loss += param_loss_rna.mean()

        param_loss_prot = my_relu(p0 - kappa[:, 0] / gamma[:, 0] * r0)   # will stay constantly positive in case of r0=0 and p0 > 0

        mask = gamma[:, 0] == 0
        param_loss_prot[mask] = 0
        param_loss += param_loss_prot.mean()

        n_genes = self.switch_time_unconstr.shape[0]
        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), min=0)
        t_s = t_s.expand(1, -1).clone()
        zero_genes = torch.zeros(n_genes, device=t_s.device)
        r_0_down, p_0_down = self._get_induction_rna_protein(
            alpha=alpha[:, 1],
            beta=beta[:, 1],
            kappa=kappa[:, 1],
            gamma=gamma[:, 1],
            t=t_s,
            r_0=zero_genes,
            p_0=zero_genes,
            time_offset=0,
        )

        param_loss_alpha = my_relu(alpha[:, 2] - beta[:, 1] * r_0_down)  # i want the reduced transcription rate to result in a downregulation of at least the rna
        param_loss += param_loss_alpha.mean()

        global_loss = 0
        loss = (
            local_loss
            + self.penalty_scale * end_penalty
            + self.param_loss_weight * (param_loss)
            + self.time_loss_weight * (1 - kl_weight) * time_loss
        )

        loss *= 100

        loss_recoder = LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=self.kl_z_scaling * kl_divergence_z,
            kl_global=torch.tensor(global_loss),
        )

        return loss_recoder


def my_relu(x):
    return torch.maximum(x, torch.zeros_like(x))
