import numpy as np


def add_fits_branches(adata, vae, n_samples=100):
    import torch.nn.functional as F

    states = vae.module.px_pi_global  # genes x state (up or down)
    states = F.softplus(states).cpu().detach().numpy()
    states_per_branch = states.argmax(axis=-1)
    adata.varm['pi_global'] = states
    adata.var['state'] = states_per_branch

    proteins, rna = vae.get_expression_fit(adata)
    adata.layers['protein_predicted'] = proteins
    adata.layers['rna_predicted'] = rna

    likelihoods = vae.get_gene_likelihood(
        adata=None,
        indices=None,
        gene_list=None,
        n_samples=n_samples,
        batch_size=None,
        return_mean=True,
    )
    adata.var['fit_likelihood'] = likelihoods.mean(axis=0)

    if vae.module.shared_time:
        adata.obs['latent_time'] = vae.get_latent_time(return_numpy=True)[:, 0, 0]
    else:
        latent_time = vae.get_latent_time(return_numpy=True)
        adata.layers['fit_t_up'] = latent_time[:, :, 0]
        adata.layers['fit_t_up'] = latent_time[:, :, 1]
        mask = adata.var['state'].values.reshape(1, latent_time.shape[1])
        mask = np.repeat(mask, latent_time.shape[0], axis=0)
        result = np.where(mask, latent_time[:, :, 1], latent_time[:, :, 0])
        adata.layers['fit_t'] = result
    adata.layers['velocity'] = vae.get_velocity(adata)
