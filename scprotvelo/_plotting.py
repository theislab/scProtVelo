import matplotlib.pyplot as plt
import seaborn as sns


def plot_phase_portraits(
        adata,
        genes,
        n_cols=4,
        figsize=(6, 6),
        rna_layer='rna',
        protein_layer='protein',
        hue='cell_type',
        axes_label=('mRNA', 'Protein'),
        save=None,
        start_at_0=False,
        show_linear_fit=False,
):
    """
    Plot phase portraits and model fits.

    Parameters
    ----------
    adata
        The adata containing model fits.
    genes
        Genes to plot in separate panels.
    n_cols
        Number columns to use for the subplots.
    figsize
        Figure size.
    rna_layer
        Layer to use for rna expression.
    protein_layer
        Layer to use for protein expression.
    hue
        Obs label to plot on the cells.
    axes_label
        X and Y axis labels.
    save
        Filename to save to, if 'None' the plot will not be saved.
    start_at_0
        Whether to start the x and y axis at 0.
    show_linear_fit
        Whether to include a linear fit.
    """
    if not isinstance(genes, list):
        genes = [genes]

    n_genes = len(genes)
    if n_genes <= n_cols:
        n_cols = n_genes
        n_rows = 1
    else:
        n_rows = ((n_genes - 1) // n_cols) + 1

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), dpi=100)
    plt.subplots_adjust(wspace=0, hspace=0)

    palette = dict(zip(adata.obs[hue].cat.categories, adata.uns[f'{hue}_colors']))

    for i, gene in enumerate(genes):
        ax_curr = ax[i // n_cols, i % n_cols]

        # plot data points
        rna = adata[:, gene].layers[rna_layer].squeeze()
        protein = adata[:, gene].layers[protein_layer].squeeze()
        hue_vec = adata.obs[hue].values
        sns.scatterplot(x=protein, y=rna, hue=hue_vec, s=10, ax=ax_curr, palette=palette)

        # plot linear fit in log-log space
        if show_linear_fit:
            import numpy as np
            loged_rna = adata[:, gene].layers['rna_log']
            rna = adata[:, gene].layers['rna']
            prot = np.exp(loged_rna * adata.var.loc[gene]['slope'] + adata.var.loc[gene]['intercept']) - 1
            ax_curr.scatter(prot, rna, color="black", s=.5)

        # plot fitted data
        rna = adata[:, gene].layers['rna_predicted']
        protein = adata[:, gene].layers['protein_predicted']
        ax_curr.scatter(protein, rna, color="black", s=3, alpha=1)
        ax_curr.scatter(protein, rna, color="white", s=.5, alpha=1)

        ax_curr.legend().remove()
        ax_curr.set_title(gene)

        if start_at_0:
            ax_curr.set_ylim(bottom=0)
            ax_curr.set_xlim(left=0)

        if i % n_cols != 0:
            ax_curr.set_ylabel('')
        else:
            ax_curr.set_ylabel(axes_label[1])
        if i // n_cols != n_rows - 1:
            ax_curr.set_xlabel('')
        else:
            ax_curr.set_xlabel(axes_label[0])

    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()
