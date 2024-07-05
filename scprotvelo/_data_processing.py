import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import gmean, ttest_ind
from statsmodels.stats.multitest import multipletests


def pair_data(
        data_path,
        data_path_out,
        combined,
        cluster_list,
        file_name,
        fractions=.3,
        n_cells=50,
        log_fc_prot=.1,
        log_fc_rna=.5,
):
    """
    Create computationally paired data by interpolating the missing modality over an integrated neighborhood graph.

    Parameters
    ----------
    data_path
        Path where protein and rna expression data are stored.
    data_path_out
        Path to save the paired adata objects.
    combined
        Adata object containing an integrated neighborhood graph of the protein and rna cells.
    cluster_list
        List of leiden clusters to which the cells of both modalities are subsetted.
    file_name
        Name for the resulting data files.
    fractions
        Fraction of top and bottom cells wrt pseudotime considered for differential expression testing.
    n_cells
        Number of cells to be drawn from the selected subsets of 'early' and 'late' cells wrt pseudotime for
        differential expression testing.
    log_fc_prot
        Log fold-change cutoff for protein differential testing.
    log_fc_rna
        Log fold-change cutoff for rna differential testing.
    """

    prot = sc.read_h5ad(f'{data_path}hBM_prot.h5ad')
    rna = sc.read_h5ad(f'{data_path}hBM_rna.h5ad')

    # subset all data to selected leiden clusters
    rna.obs['leiden'] = combined.obs[combined.obs['domain'] == 'rna']['leiden']
    prot.obs['leiden'] = combined.obs[combined.obs['domain'] == 'protein']['leiden']
    prot = prot[prot.obs['leiden'].isin(cluster_list)].copy()
    rna = rna[rna.obs['leiden'].isin(cluster_list)].copy()
    combined = combined[combined.obs['leiden'].isin(cluster_list)].copy()

    sc.pl.umap(combined, color=['cell_type', 'dpt_pseudotime', 'leiden'])

    # filter proteins to >= 30% completeness, rna to >= 2%, then take intersection of rnas and proteins
    protein_completeness = 0.3
    rna_completeness = 0.02
    prot = prot[:, np.isnan(prot.layers['batchcorr_norm_log2_nan']).mean(axis=0) < 1-protein_completeness].copy()
    rna = rna[:, np.array(np.mean(rna.layers['counts'] > 0, axis=0))[0] > rna_completeness].copy()
    intersection_genes = prot.var.index.intersection(rna.var.index)
    print(f'proteins after filtering for {int(protein_completeness * 100)}% completeness: {prot.n_vars}')
    print(f'mRNAs after filtering for {int(rna_completeness * 100)}% completeness: {rna.n_vars}')
    print(f'intersection: {len(intersection_genes)}\n')
    rna = rna[:, intersection_genes].copy()
    prot = prot[:, intersection_genes].copy()

    # prepare median ratios
    prot.layers['median_ratios'] = prot.layers['batchcorr']
    x = prot.layers['median_ratios'].copy()
    x[x == 0] = np.nan
    gmeans = gmean(x, axis=0, nan_policy='omit')  # feature means
    s_facs = np.nanmedian(x / gmeans, axis=1)  # scale features to have mean 1, then compute median deviation per cell
    prot.layers['median_ratios'] = x / s_facs[:, None]  # divide cells by median deviation, so half the features will have expression values above the average, half below

    rna.layers['median_ratios'] = np.array(rna.layers['counts'].todense()).copy()
    x = rna.layers['median_ratios'].copy()
    x[x == 0] = np.nan
    gmeans = gmean(x, axis=0, nan_policy='omit')
    s_facs = np.nanmedian(x / gmeans, axis=1)
    rna.layers['median_ratios'] = x / s_facs[:, None]
    rna.layers['median_ratios'][np.isnan(rna.layers['median_ratios'])] = 0

    # perform one smoothing step within the modalities
    prot.layers['smoothed'] = np.nan * prot.layers['median_ratios']
    X = prot.layers['median_ratios']
    X[X == 0] = np.nan
    cells_without_neighbors = []
    for i, cell in enumerate(prot.obs_names):
        n_inds = np.array(prot.obsp['distances'][i, :].todense() > 0).flatten()
        if np.sum(n_inds) < 1:
            cells_without_neighbors.append(cell)
        else:
            prot.layers['smoothed'][i, :] = np.nanmean(X[n_inds], axis=0)
    prot.layers['smoothed'][prot.layers['smoothed'] != prot.layers['smoothed']] = 0
    prot.layers['smoothed'] = prot.layers['smoothed'].copy()
    prot = prot[[o not in cells_without_neighbors for o in prot.obs_names]].copy()
    combined = combined[[o not in cells_without_neighbors for o in combined.obs_names]]

    rna.layers['smoothed'] = np.nan * rna.X
    X = rna.layers['median_ratios']
    cells_without_neighbors = []
    for i, cell in enumerate(rna.obs_names):
        n_inds = np.array(rna.obsp['distances'][i, :].todense() > 0).flatten()
        if np.sum(n_inds) < 1:
            cells_without_neighbors.append(cell)
        rna.layers['smoothed'][i, :] = np.mean(X[n_inds], axis=0)
    rna.layers['smoothed'] = rna.layers['smoothed'].copy()
    rna = rna[[o not in cells_without_neighbors for o in rna.obs_names]].copy()
    combined = combined[[o not in cells_without_neighbors for o in combined.obs_names]]

    # interpolate missing modality by 15 nearest neighbors in glue embedding
    nn = sc.Neighbors(combined, neighbors_key=None)
    nn.compute_neighbors(n_neighbors=300, use_rep='X_glue', write_knn_indices=True)

    n = 15

    prot_df = pd.DataFrame(data=prot.layers['smoothed'].T, index=intersection_genes, columns=prot.obs.index)
    rna_df = pd.DataFrame(data=rna.layers['smoothed'].T, index=intersection_genes, columns=rna.obs.index)

    mapping = []

    for i, cell_i in enumerate(combined.obs.index):
        domain = combined.obs['domain'].iloc[i]
        df = combined.obs.iloc[nn.knn_indices[i]]

        if domain == 'rna':
            prot_cells = df[df['domain'] == 'protein'].head(n).index
            rna_cells = df[df['domain'] == 'rna'].head(1).index.to_list()
        else:
            prot_cells = df[df['domain'] == 'protein'].head(1).index.to_list()
            rna_cells = df[df['domain'] == 'rna'].head(n).index

        mapping.extend([(cell_i, 'rna', x) for x in rna_cells])
        mapping.extend([(cell_i, 'protein', x) for x in prot_cells])

    mapped_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(mapping, names=['origin_idx', 'modality', 'nn_idx']))

    df_rna_map = rna_df.loc[intersection_genes].T.copy()
    df_rna_map.index.name = 'nn_idx'
    df_prot_map = prot_df.loc[intersection_genes].T.copy()
    df_prot_map.index.name = 'nn_idx'

    mapped_df = mapped_df.join(pd.concat([df_rna_map, df_prot_map]), how='inner')

    paired_df = mapped_df.groupby(level=[0, 1]).agg(np.nanmean)

    paired_adata = sc.AnnData(paired_df.xs('rna', level=1))
    paired_adata.layers['protein'] = paired_df.xs('protein', level=1).values
    paired_adata.layers['protein'][paired_adata.layers['protein'] != paired_adata.layers['protein']] = 0  # these are entries that had no not-nan neighbor
    paired_adata.obsm = {k: v for k, v in combined[paired_adata.obs.index].obsm.items() if k in ['X_glue', 'X_umap']}
    paired_adata.obs = combined[paired_adata.obs.index].obs[['dpt_pseudotime', 'cell_type', 'domain']].copy()
    paired_adata.uns = {k: v for k, v in combined.uns.items() if k in ['cell_type_colors', 'domain_colors', 'neighbors']}

    paired_adata.X = np.array(paired_adata.X)
    paired_adata.layers['rna'] = paired_adata.X
    paired_adata.layers['protein'] = np.array(paired_adata.layers['protein'])

    adata = paired_adata.copy()

    prot = prot[:, adata.var_names]
    rna = rna[:, adata.var_names]

    # scale all features by 1 / (75% quantile - 25% quantile) to give them the same weight in the likelihood function
    layer = 'rna'
    quantiles25 = np.quantile(adata.layers[layer], q=.25, axis=0)
    quantiles75 = np.quantile(adata.layers[layer], q=.75, axis=0)
    adata.layers[layer] = adata.layers[layer] / (quantiles75 - quantiles25)
    adata.var['scaling_rna'] = (quantiles75 - quantiles25)

    layer = 'protein'
    quantiles25 = np.quantile(adata.layers[layer], q=.25, axis=0)
    quantiles75 = np.quantile(adata.layers[layer], q=.75, axis=0)
    adata.layers[layer] = adata.layers[layer] / (quantiles75 - quantiles25)
    adata.var['scaling_prot'] = (quantiles75 - quantiles25)

    sc.pp.neighbors(adata, use_rep='X_glue')

    d, n = np.unique(adata.obs['domain'], return_counts=True)
    for domain, count in zip(d, n):
        print(f'{domain}: {count} cells')
    print()

    de_genes = _select_variable_genes(
        prot,
        rna,
        adata,
        fractions=fractions,
        n_cells=n_cells,
        log_fc_prot=log_fc_prot,
        log_fc_rna=log_fc_rna,
    )

    adata.write(f'{data_path_out}{file_name}_all_genes.h5ad')

    a = "'"
    print(f'{len(de_genes)} DE genes selected:')
    print(f"[{', '.join([f'{a}{p}{a}' for p in de_genes])}]")
    adata = adata[:, de_genes].copy()

    adata.write(f'{data_path_out}{file_name}_de_genes.h5ad')

    return adata, rna, prot


def _select_variable_genes(
        prot,
        rna,
        adata,
        fractions=.3,
        n_cells=50,
        log_fc_prot=.1,
        log_fc_rna=.5,
):
    p = prot.copy()
    p.X = p.layers['median_ratios']
    p.X[p.X == 0] = np.nan
    p.obs['dpt_pseudotime'] = adata.obs['dpt_pseudotime']

    r = rna.copy()
    r.X = r.layers['median_ratios']
    r.obs['dpt_pseudotime'] = adata.obs['dpt_pseudotime']

    selected_proteins, logfc_prot = _de_genes(adata=p, fractions=fractions, n_cells=n_cells, log_fc=log_fc_prot)
    selected_rnas, logfc_rna = _de_genes(adata=r, fractions=fractions, n_cells=n_cells, log_fc=log_fc_rna)

    adata.var['de_rna'] = ['up' if lfc > 0 else 'down' for lfc in logfc_rna]
    adata.var.loc[[g for g in adata.var_names if g not in selected_rnas]] = np.nan

    de_genes = [p for p in selected_proteins if p in selected_rnas]

    return de_genes


def _de_genes(adata, fractions, n_cells, log_fc):
    np.random.seed(0)

    dpt_low = adata.obs['dpt_pseudotime'].quantile(fractions)
    dpt_high = adata.obs['dpt_pseudotime'].quantile(1 - fractions)
    adata_low = adata[adata.obs['dpt_pseudotime'] < dpt_low]
    adata_high = adata[adata.obs['dpt_pseudotime'] > dpt_high]

    rand_ind = np.random.choice(np.arange(adata_high.shape[0]), size=n_cells, replace=True)
    adata_low = adata_low[rand_ind]
    adata_high = adata_high[rand_ind]

    p_vals = ttest_ind(adata_low.X, adata_high.X, nan_policy='omit')[1]

    p_vals_corrected = multipletests(p_vals, alpha=0.05, method="fdr_bh")[1]
    fc = np.nanmean(adata_high.X, axis=0) / np.nanmean(adata_low.X, axis=0)
    logfc = np.log(fc)

    gene_names = adata.var_names

    selected_genes = gene_names[(np.abs(logfc) > log_fc) & (p_vals_corrected < 0.05)]

    return selected_genes, logfc
