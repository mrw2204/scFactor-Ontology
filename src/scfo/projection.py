from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd

from .core import (
    as_dense,
    calc_enrichment,
    factor_weights_to_df,
    get_matrix_from_adata,
    gsea_enrichment,
    modality_feature_loadings_to_df,
    safe_factor_metadata,
    signature_to_df,
)


from __future__ import annotations

from typing import Optional, Tuple

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import scipy.sparse as sp

def project_ontology(
    adata: ad.AnnData,
    ontology: mu.MuData,
    annotation_key: Optional[str] = None,
    layer: Optional[str] = None,
    method: str = "dot",
    n_iter: int = 1000,
    seed: int = 0,
    inplace: bool = True,
    score_key_added: str = "scfo_scores",
    pval_key_added: str = "scfo_pvals",
    score_columns_uns_key: Optional[str] = None,
    store_sparse_scores: bool = True,
    show_progress=True,
) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """
    Project an ontology onto an AnnData object.

    If `annotation_key` is provided, cells are only projected onto ontology factors
    whose ontology Classification matches the cell annotation. If the AnnData and
    ontology annotations only partially overlap, only the overlapping subset is used
    for computation. If `inplace=True`, the result is expanded back to full AnnData
    dimensions before storage in `.obsm`.

    Parameters
    ----------
    adata
        AnnData object with cells in rows and genes in columns.
    ontology
        MuData ontology object with:
        - ontology.obsm["weights"] : factor x gene matrix
        - ontology.uns["gene_names"] : gene names for the weight matrix
        - ontology.obs["Classification"] : cell-type / lineage labels for factors
    annotation_key
        Column in `adata.obs` containing cell annotations to match against
        `ontology.obs["Classification"]`. If None, project all cells onto all factors.
    layer
        Optional layer from `adata.layers` to use instead of `adata.X`.
    method
        Either "dot" or "permutation".
    n_iter
        Number of permutations if `method="permutation"`.
    seed
        Random seed for permutation-based projection.
    inplace
        If True, write outputs to `adata.obsm` and return None.
        If False, return `(score_df, pval_df_or_none)`.
        In the annotated case, returned DataFrames are restricted to overlapping
        cells and overlapping ontology factors.
    score_key_added
        Key for projected scores in `adata.obsm`.
    pval_key_added
        Key for projected p-values in `adata.obsm` when `method="permutation"`.
    score_columns_uns_key
        If sparse scores are stored in `adata.obsm`, the corresponding column names
        are stored in `adata.uns[score_columns_uns_key]`. If None, defaults to
        `f"{score_key_added}_columns"`.
    store_sparse_scores
        If True and `inplace=True`, store projected scores in `adata.obsm` as a CSR
        sparse matrix after replacing NaN with 0. Column names are stored in `adata.uns`.
        If False, store the score DataFrame directly in `adata.obsm`.
    show_progress
        Whether to show progress bars for permutation-based projection.

    Returns
    -------
    None or (pandas.DataFrame, pandas.DataFrame or None)
        If `inplace=True`, returns None.
        Otherwise returns `(score_df, pval_df_or_none)`.
    """
    if method not in {"dot", "permutation"}:
        raise ValueError("method must be 'permutation' or 'dot'.")

    if "weights" not in ontology.obsm:
        raise KeyError("ontology.obsm['weights'] not found.")
    if "gene_names" not in ontology.uns:
        raise KeyError("ontology.uns['gene_names'] not found.")
    if "Classification" not in ontology.obs.columns:
        raise KeyError("ontology.obs['Classification'] not found.")

    if score_columns_uns_key is None:
        score_columns_uns_key = f"{score_key_added}_columns"

    X_df = get_matrix_from_adata(adata, layer=layer)
    weights = factor_weights_to_df(ontology)  # factor x gene
    factor_meta = ontology.obs.reindex(weights.index).copy()

    if factor_meta["Classification"].isna().any():
        factor_meta["Classification"] = factor_meta["Classification"].fillna("Unknown")

    # ------------------------------------------------------------------
    # Case 1: global projection (no annotation matching)
    # ------------------------------------------------------------------
    if annotation_key is None:
        common = X_df.columns.intersection(weights.columns)
        if len(common) == 0:
            raise ValueError("No overlapping genes between adata and ontology weights.")

        score_df = pd.DataFrame(
            np.nan,
            index=adata.obs_names,
            columns=weights.index,
            dtype=float,
        )

        pval_df = None
        if method == "permutation":
            pval_df = pd.DataFrame(
                np.nan,
                index=adata.obs_names,
                columns=weights.index,
                dtype=float,
            )

        if method == "dot":
            proj = X_df.loc[:, common] @ weights.loc[:, common].T
            score_df.loc[proj.index, proj.columns] = proj
        else:
            pair = calc_enrichment(
                X_df.loc[:, common],
                weights.loc[:, common].T,
                n_iter=n_iter,
                seed=seed,
                show_progress=show_progress,
                progress_message="Permuting globally...",
            )
            score_df.loc[pair.score.index, pair.score.columns] = pair.score
            assert pval_df is not None
            pval_df.loc[pair.pval.index, pair.pval.columns] = pair.pval

        pval_out = None if method == "dot" else pval_df

    # ------------------------------------------------------------------
    # Case 2: annotation-aware projection with partial overlap support
    # Compute on overlap subset first, then expand only if inplace=True
    # ------------------------------------------------------------------
    else:
        if annotation_key not in adata.obs.columns:
            raise KeyError(f"annotation_key '{annotation_key}' not found in adata.obs.")

        ann = adata.obs[annotation_key]
        ann_nonnull = ann.dropna().astype(str)

        adata_classes = pd.Index(ann_nonnull.unique())
        ontology_classes = pd.Index(
            factor_meta["Classification"].dropna().astype(str).unique()
        )
        cl_common = adata_classes.intersection(ontology_classes)

        if len(cl_common) == 0:
            raise ValueError(
                "No overlapping classifications between "
                f"adata.obs['{annotation_key}'] and ontology.obs['Classification']."
            )

        # Restrict computation to overlapping cells and overlapping factors only
        cell_mask = ann.notna() & ann.astype(str).isin(cl_common)
        cell_idx_use = adata.obs_names[cell_mask]

        factor_mask = factor_meta["Classification"].astype(str).isin(cl_common)
        factor_idx_use = factor_meta.index[factor_mask]

        score_sub = pd.DataFrame(
            np.nan,
            index=cell_idx_use,
            columns=factor_idx_use,
            dtype=float,
        )

        pval_sub = None
        if method == "permutation":
            pval_sub = pd.DataFrame(
                np.nan,
                index=cell_idx_use,
                columns=factor_idx_use,
                dtype=float,
            )

        X_sub = X_df.loc[cell_idx_use]
        ann_sub = ann.loc[cell_idx_use].astype(str)
        factor_classes_sub = factor_meta.loc[factor_idx_use, "Classification"].astype(str)

        for ct in cl_common:
            cell_idx = ann_sub.index[ann_sub == ct]
            if len(cell_idx) == 0:
                continue

            factor_idx = factor_classes_sub.index[factor_classes_sub == ct]
            if len(factor_idx) == 0:
                continue

            X_ct = X_sub.loc[cell_idx]
            W_ct = weights.loc[factor_idx]
            common = X_ct.columns.intersection(W_ct.columns)
            if len(common) == 0:
                continue

            if method == "dot":
                proj = X_ct.loc[:, common] @ W_ct.loc[:, common].T
                score_sub.loc[proj.index, proj.columns] = proj
            else:
                pair = calc_enrichment(
                    X_ct.loc[:, common],
                    W_ct.loc[:, common].T,
                    n_iter=n_iter,
                    seed=seed,
                    show_progress=show_progress,
                    progress_message=f"Permuting {ct}...",
                )
                score_sub.loc[pair.score.index, pair.score.columns] = pair.score
                assert pval_sub is not None
                pval_sub.loc[pair.pval.index, pair.pval.columns] = pair.pval

        # Returned object in non-inplace mode is the overlap-only result
        score_df = score_sub
        pval_out = None if method == "dot" else pval_sub

        # Expand back to full AnnData dimensions only for storage in .obsm
        if inplace:
            score_full = pd.DataFrame(
                np.nan,
                index=adata.obs_names,
                columns=weights.index,
                dtype=float,
            )
            score_full.loc[score_sub.index, score_sub.columns] = score_sub
            score_df = score_full

            if method == "permutation":
                pval_full = pd.DataFrame(
                    np.nan,
                    index=adata.obs_names,
                    columns=weights.index,
                    dtype=float,
                )
                assert pval_sub is not None
                pval_full.loc[pval_sub.index, pval_sub.columns] = pval_sub
                pval_out = pval_full

    # ------------------------------------------------------------------
    # Storage / return
    # ------------------------------------------------------------------
    if inplace:
        if store_sparse_scores:
            score_store = sp.csr_matrix(score_df.fillna(0.0).to_numpy(dtype=np.float32))
            adata.obsm[score_key_added] = score_store
            adata.uns[score_columns_uns_key] = list(map(str, score_df.columns))
        else:
            adata.obsm[score_key_added] = score_df
            if score_columns_uns_key in adata.uns:
                del adata.uns[score_columns_uns_key]

        if pval_out is not None:
            adata.obsm[pval_key_added] = pval_out
        elif pval_key_added in adata.obsm:
            del adata.obsm[pval_key_added]

        return None

    return score_df, pval_out

def collapse_projected_ontology_scores(
    adata: ad.AnnData,
    score_key: str = "scfo_scores",
    output_key: Optional[str] = None,
    columns_uns_key: Optional[str] = None,
    separator: str = "|",
    agg: str = "sum",
    store_sparse: bool = False,
    output_columns_uns_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Collapse a projected ontology score matrix from
    n_cells x (n_factors x n_celltypes) to n_cells x n_factors.

    This is most useful after annotation-aware projection, where for each cell only
    one lineage-specific block is typically populated and the rest are zero/empty.

    Parameters
    ----------
    adata
        AnnData object containing projected ontology scores in `.obsm`.
    score_key
        Key in `adata.obsm` containing the projected score matrix.
    output_key
        Optional key to write the collapsed matrix back into `adata.obsm`.
        If None, nothing is written.
    columns_uns_key
        Key in `adata.uns` containing column names for `adata.obsm[score_key]`
        when that matrix is stored sparsely. If None, defaults to
        `f"{score_key}_columns"`.
    separator
        Separator between factor name and cell type, e.g. `Factor0|Tumor`.
    agg
        Aggregation across cell types for the same factor. One of:
        `"sum"`, `"max"`, `"mean"`.
        For annotation-aware sparse projections, `"sum"` is usually appropriate.
    store_sparse
        If `output_key` is provided, whether to store the collapsed result as CSR
        sparse matrix in `.obsm`.
    output_columns_uns_key
        If storing sparse output, column names are written to this key in `.uns`.
        If None, defaults to `f"{output_key}_columns"`.

    Returns
    -------
    pandas.DataFrame
        Collapsed cell-by-factor DataFrame.
    """
    if score_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{score_key}'] not found.")

    obj = adata.obsm[score_key]

    if isinstance(obj, pd.DataFrame):
        score_df = obj.copy()
    else:
        if columns_uns_key is None:
            columns_uns_key = f"{score_key}_columns"
        if columns_uns_key not in adata.uns:
            raise KeyError(
                f"adata.uns['{columns_uns_key}'] not found. "
                "Column names are required when collapsing a sparse/array obsm matrix."
            )
        cols = pd.Index(list(map(str, adata.uns[columns_uns_key])), name="factor_celltype")
        score_df = pd.DataFrame(
            as_dense(obj),
            index=adata.obs_names,
            columns=cols,
        )

    base_factor_names = pd.Index(
        [str(c).split(separator, 1)[0] for c in score_df.columns],
        name="factor",
    )

    if agg == "sum":
        collapsed = score_df.T.groupby(base_factor_names).sum().T
    elif agg == "max":
        collapsed = score_df.T.groupby(base_factor_names).max().T
    elif agg == "mean":
        collapsed = score_df.T.groupby(base_factor_names).mean().T
    else:
        raise ValueError("agg must be one of {'sum', 'max', 'mean'}.")

    collapsed = collapsed.loc[adata.obs_names]

    if output_key is not None:
        if store_sparse:
            if output_columns_uns_key is None:
                output_columns_uns_key = f"{output_key}_columns"
            adata.obsm[output_key] = sp.csr_matrix(collapsed.to_numpy(dtype=np.float32))
            adata.uns[output_columns_uns_key] = list(map(str, collapsed.columns))
        else:
            adata.obsm[output_key] = collapsed

    return collapsed


def _normalize_cell_types(cell_types: Optional[Union[str, Sequence[str]]]) -> Optional[list]:
    if cell_types is None:
        return None
    if isinstance(cell_types, str):
        return [cell_types]
    return list(cell_types)


def _is_gene_set_query(signature: Union[Sequence[str], pd.Series, pd.DataFrame]) -> bool:
    return not isinstance(signature, (pd.Series, pd.DataFrame))


def _signature_genes(signature: Union[Sequence[str], pd.Series, pd.DataFrame]) -> pd.Index:
    if isinstance(signature, pd.DataFrame):
        if signature.shape[1] != 1:
            raise ValueError("signature DataFrame must have exactly one column.")
        genes = signature.index
    elif isinstance(signature, pd.Series):
        genes = signature.index
    else:
        genes = pd.Index(list(signature), dtype=str)
    return pd.Index([str(g) for g in genes]).drop_duplicates()


def _overlap_summary(query_genes: pd.Index, target_genes: pd.Index) -> dict:
    overlap = query_genes.intersection(target_genes)
    return {
        "query_n": int(len(query_genes)),
        "target_n": int(len(target_genes)),
        "overlap_n": int(len(overlap)),
        "overlap_frac_query": float(len(overlap) / len(query_genes)) if len(query_genes) else 0.0,
        "overlap_frac_target": float(len(overlap) / len(target_genes)) if len(target_genes) else 0.0,
        "overlap_genes": list(overlap),
    }


def _run_signature_query(signature, loadings: pd.DataFrame, pval_thresh=None, n_iter: int = 1000, seed: int = 0) -> pd.DataFrame:
    query_genes = _signature_genes(signature)
    overlap = query_genes.intersection(loadings.index)
    if len(overlap) == 0:
        raise ValueError("No overlap between the queried signature and the ontology target.")

    if _is_gene_set_query(signature):
        pair = gsea_enrichment(
            loadings,
            {"query": list(overlap)},
            permutation_num=n_iter,
            min_size=max(1, min(10, len(overlap))),
            seed=seed,
            processes=1,
        )
        df = pd.DataFrame({"score": pair.score["query"], "pval": pair.pval["query"]})
        df["method"] = "gsea"
    else:
        sig_df = signature_to_df(signature, loadings.index, name="query")
        pair = calc_enrichment(sig_df.T, loadings, n_iter=n_iter, seed=seed, show_progress=False)
        df = pd.DataFrame({"score": pair.score.iloc[0], "pval": pair.pval.iloc[0]})
        df["method"] = "permutation"

    if pval_thresh is not None:
        df = df.loc[df["pval"] < pval_thresh].copy()
    return df.sort_values("score", ascending=False)


def _resolve_modality_loading_keys(ontology: mu.MuData, modality: str, cell_types: Optional[Union[str, Sequence[str]]] = None):
    mod = ontology.mod[modality]
    varm_keys = list(mod.varm.keys())
    if len(varm_keys) == 0:
        return []
    ct_list = _normalize_cell_types(cell_types)
    if ct_list is None:
        return varm_keys
    selected = []
    for ct in ct_list:
        selected.extend([k for k in varm_keys if str(k).startswith(f"{ct}_")])
    if len(selected) == 0 and len(varm_keys) == 1:
        return varm_keys
    out = []
    seen = set()
    for k in selected:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _key_to_cell_type(varm_key: str) -> Optional[str]:
    s = str(varm_key)
    if "_" not in s:
        return None
    return s.split("_", 1)[0]


def signature_enrichment(
    signature: Union[Sequence[str], pd.Series, pd.DataFrame],
    ontology: mu.MuData,
    search_in: Optional[Sequence[str]] = None,
    pval_thresh: Optional[float] = None,
    n_iter: int = 1000,
    seed: int = 0,
    cell_types: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, Dict[str, object]]:
    """Query ontology weights and modality loadings with a gene list or weighted vector.

    Parameters
    ----------
    signature : sequence of str, pandas.Series, or one-column pandas.DataFrame
        Query signature. An unordered gene list is treated as a gene set and scored with
        preranked GSEA. A weighted series or one-column DataFrame is treated as a ranked
        vector and scored with the permutation framework.
    ontology : muon.MuData
        Ontology object.
    search_in : sequence of str
        Targets to query. May include ``"weights"`` and any ontology modality name.
    cell_types : str or sequence of str, optional
        Restrict the query to selected ontology lineages. For ``"weights"``, only factors
        from the selected lineages are searched. For modalities, only lineage-matched
        loading matrices are searched when available.
    pval_thresh : float, optional
        Optional p-value threshold applied to each result table.
    n_iter : int, default=1000
        Number of permutations or GSEA permutations.
    seed : int, default=0
        Random seed.

    Returns
    -------
    dict
        Mapping from each queried target to a dictionary with two entries:
        ``"results"`` (ranked enrichment table) and ``"overlap"`` (gene-overlap summary).
    """
    results: Dict[str, Dict[str, object]] = {}
    targets = ["weights"] + list(ontology.mod.keys()) if search_in is None else list(search_in)
    # deduplicate preserve order
    seen = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]
    query_genes = _signature_genes(signature)

    for target in targets:
        if target == "weights":
            loadings = factor_weights_to_df(ontology, transpose=True, cell_types=cell_types)
            overlap = _overlap_summary(query_genes, loadings.index)
            df = _run_signature_query(signature, loadings, pval_thresh=pval_thresh, n_iter=n_iter, seed=seed)
            results["weights"] = {"results": df, "overlap": overlap}
            continue

        if target not in ontology.mod:
            raise KeyError(
                f"Requested target '{target}' is not 'weights' and not found in ontology.mod. "
                f"Available modalities: {list(ontology.mod.keys())}"
            )

        keys = _resolve_modality_loading_keys(ontology, target, cell_types=cell_types)
        if len(keys) == 0:
            loadings = modality_feature_loadings_to_df(ontology, target, cell_types=cell_types)
            overlap = _overlap_summary(query_genes, loadings.index)
            df = _run_signature_query(signature, loadings, pval_thresh=pval_thresh, n_iter=n_iter, seed=seed)
            results[target] = {"results": df, "overlap": overlap}
            continue

        for key in keys:
            loadings = modality_feature_loadings_to_df(ontology, target, key=key)
            overlap = _overlap_summary(query_genes, loadings.index)
            df = _run_signature_query(signature, loadings, pval_thresh=pval_thresh, n_iter=n_iter, seed=seed)
            ct = _key_to_cell_type(key)
            result_key = f"{target}|{ct}" if ct is not None else target
            results[result_key] = {"results": df, "overlap": overlap}

    return results
