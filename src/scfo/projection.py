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


def project_ontology(
    adata: ad.AnnData,
    ontology: mu.MuData,
    layer: Optional[str] = None,
    annotation_key: Optional[str] = None,
    score_key_added: str = "ontology_scores",
    pval_key_added: str = "ontology_pvals",
    method: str = "permutation",
    n_iter: int = 1000,
    seed: int = 0,
    inplace: bool = True,
) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """Project ontology factor scores onto an external ``AnnData`` object.

    Parameters
    ----------
    adata : anndata.AnnData
        External dataset to score.
    ontology : muon.MuData
        Ontology object created with :func:`make_ontology`.
    layer : str, optional
        Layer to use for projection. If ``None``, ``adata.X`` is used.
    annotation_key : str, optional
        Column in ``adata.obs`` containing cell type labels. If provided, each cell is
        projected only against factors from the corresponding ontology lineage.
    score_key_added : str, default="ontology_scores"
        Key under which projected scores are stored in ``adata.obsm``.
    pval_key_added : str, default="ontology_pvals"
        Key under which permutation-derived p-values are stored in ``adata.obsm``.
    method : {"permutation", "dot"}, default="permutation"
        Scoring method. ``"permutation"`` returns z-scores and p-values, whereas
        ``"dot"`` performs a direct dot-product projection only.
    n_iter : int, default=1000
        Number of permutations for permutation-based projection.
    seed : int, default=0
        Random seed for permutation-based projection.
    inplace : bool, default=True
        If ``True``, store scores in ``adata.obsm``. Otherwise return them.

    Returns
    -------
    None or tuple of pandas.DataFrame
        If ``inplace=True``, returns ``None``. Otherwise returns ``(scores, pvals)``.
    """
    if "weights" not in ontology.obsm:
        raise KeyError("ontology.obsm['weights'] not found.")
    if "gene_names" not in ontology.uns:
        raise KeyError("ontology.uns['gene_names'] not found.")
    X_df = get_matrix_from_adata(adata, layer=layer)
    weights = factor_weights_to_df(ontology)
    factor_meta = ontology.obs.reindex(weights.index).copy()
    score_df = pd.DataFrame(np.nan, index=adata.obs_names, columns=weights.index, dtype=float)
    pval_df = pd.DataFrame(np.nan, index=adata.obs_names, columns=weights.index, dtype=float)
    if annotation_key is None:
        common = X_df.columns.intersection(weights.columns)
        if method == "dot":
            proj = X_df.loc[:, common] @ weights.loc[:, common].T
            score_df.loc[:, proj.columns] = proj
            pval_out = None
        elif method == "permutation":
            pair = calc_enrichment(X_df.loc[:, common], weights.loc[:, common].T, n_iter=n_iter, seed=seed, show_progress=False)
            score_df.loc[:, pair.score.columns] = pair.score
            pval_df.loc[:, pair.pval.columns] = pair.pval
            pval_out = pval_df
        else:
            raise ValueError("method must be 'permutation' or 'dot'.")
    else:
        if annotation_key not in adata.obs.columns:
            raise KeyError(f"annotation_key '{annotation_key}' not found in adata.obs.")
        for ct, cell_idx in adata.obs.groupby(annotation_key, sort=False).groups.items():
            factor_idx = factor_meta.index[factor_meta["Classification"].astype(str) == str(ct)]
            if len(factor_idx) == 0 or len(cell_idx) == 0:
                continue
            X_ct = X_df.loc[cell_idx]
            W_ct = weights.loc[factor_idx]
            common = X_ct.columns.intersection(W_ct.columns)
            if len(common) == 0:
                continue
            if method == "dot":
                proj = X_ct.loc[:, common] @ W_ct.loc[:, common].T
                score_df.loc[cell_idx, proj.columns] = proj
            elif method == "permutation":
                pair = calc_enrichment(X_ct.loc[:, common], W_ct.loc[:, common].T, n_iter=n_iter, seed=seed, show_progress=False)
                score_df.loc[cell_idx, pair.score.columns] = pair.score
                pval_df.loc[cell_idx, pair.pval.columns] = pair.pval
            else:
                raise ValueError("method must be 'permutation' or 'dot'.")
        pval_out = None if method == "dot" else pval_df
    if inplace:
        adata.obsm[score_key_added] = score_df
        if pval_out is not None:
            adata.obsm[pval_key_added] = pval_out
        elif pval_key_added in adata.obsm:
            del adata.obsm[pval_key_added]
        return None
    return score_df, pval_out


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
