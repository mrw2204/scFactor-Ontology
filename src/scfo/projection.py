from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import (
    as_dense,
    calc_enrichment,
    factor_weights_to_df,
    get_matrix_from_adata,
    gsea_enrichment,
    modality_feature_loadings_to_df,
    signature_to_df,
)


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
    show_progress: bool = True,
) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """Project ontology weights onto an AnnData object."""
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
    weights = factor_weights_to_df(ontology)
    factor_meta = ontology.obs.reindex(weights.index).copy()
    factor_meta["Classification"] = factor_meta["Classification"].fillna("Unknown").astype(str)

    if annotation_key is None:
        common = X_df.columns.intersection(weights.columns)
        if len(common) == 0:
            raise ValueError("No overlapping genes between adata and ontology weights.")

        if method == "dot":
            score_df = X_df.loc[:, common] @ weights.loc[:, common].T
            pval_out = None
        else:
            pair = calc_enrichment(
                X_df.loc[:, common],
                weights.loc[:, common].T,
                n_iter=n_iter,
                seed=seed,
                show_progress=show_progress,
                progress_message="Permuting globally...",
            )
            score_df = pair.score
            pval_out = pair.pval

        score_df = score_df.reindex(index=adata.obs_names, columns=weights.index)
        if pval_out is not None:
            pval_out = pval_out.reindex(index=adata.obs_names, columns=weights.index)

    else:
        if annotation_key not in adata.obs.columns:
            raise KeyError(f"annotation_key '{annotation_key}' not found in adata.obs.")

        ann = adata.obs[annotation_key]
        ann_nonnull = ann.dropna().astype(str)
        cl_common = pd.Index(ann_nonnull.unique()).intersection(
            pd.Index(factor_meta["Classification"].unique())
        )
        if len(cl_common) == 0:
            raise ValueError(
                f"No overlapping classifications between adata.obs['{annotation_key}'] and ontology.obs['Classification']."
            )

        cell_mask = ann.notna() & ann.astype(str).isin(cl_common)
        cell_idx_use = adata.obs_names[cell_mask]
        factor_idx_use = factor_meta.index[factor_meta["Classification"].isin(cl_common)]

        score_sub = pd.DataFrame(np.nan, index=cell_idx_use, columns=factor_idx_use, dtype=float)
        pval_sub = None if method == "dot" else pd.DataFrame(np.nan, index=cell_idx_use, columns=factor_idx_use, dtype=float)

        X_sub = X_df.loc[cell_idx_use]
        ann_sub = ann.loc[cell_idx_use].astype(str)
        factor_classes_sub = factor_meta.loc[factor_idx_use, "Classification"].astype(str)

        for ct in cl_common:
            cell_idx = ann_sub.index[ann_sub == ct]
            factor_idx = factor_classes_sub.index[factor_classes_sub == ct]
            if len(cell_idx) == 0 or len(factor_idx) == 0:
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

        if inplace:
            score_df = pd.DataFrame(np.nan, index=adata.obs_names, columns=weights.index, dtype=float)
            score_df.loc[score_sub.index, score_sub.columns] = score_sub
            if method == "dot":
                pval_out = None
            else:
                pval_out = pd.DataFrame(np.nan, index=adata.obs_names, columns=weights.index, dtype=float)
                assert pval_sub is not None
                pval_out.loc[pval_sub.index, pval_sub.columns] = pval_sub
        else:
            score_df = score_sub
            pval_out = None if method == "dot" else pval_sub

    if inplace:
        if store_sparse_scores:
            adata.obsm[score_key_added] = sp.csr_matrix(score_df.fillna(0.0).to_numpy(dtype=np.float32))
            adata.uns[score_columns_uns_key] = list(map(str, score_df.columns))
        else:
            adata.obsm[score_key_added] = score_df
            adata.uns.pop(score_columns_uns_key, None)

        if pval_out is not None:
            adata.obsm[pval_key_added] = pval_out
        else:
            adata.obsm.pop(pval_key_added, None)
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
    """Collapse cell-type-suffixed projected ontology scores to plain factor names."""
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
                f"adata.uns['{columns_uns_key}'] not found. Column names are required when collapsing a sparse/array obsm matrix."
            )
        score_df = pd.DataFrame(
            as_dense(obj),
            index=adata.obs_names,
            columns=pd.Index(list(map(str, adata.uns[columns_uns_key])), name="factor_celltype"),
        )

    base_factor_names = pd.Index([str(c).split(separator, 1)[0] for c in score_df.columns], name="factor")
    grouped = score_df.T.groupby(base_factor_names)
    if agg == "sum":
        collapsed = grouped.sum().T
    elif agg == "max":
        collapsed = grouped.max().T
    elif agg == "mean":
        collapsed = grouped.mean().T
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


def _resolve_modality_loading_keys(
    ontology: mu.MuData,
    modality: str,
    cell_types: Optional[Union[str, Sequence[str]]] = None,
):
    mod = ontology.mod[modality]
    varm_keys = list(mod.varm.keys())
    if len(varm_keys) == 0:
        return []

    row_axis = str(mod.uns.get("feature_loading_row_axis", "gene"))
    if len(varm_keys) == 1 or row_axis == "gene|cell_type":
        return []

    ct_list = _normalize_cell_types(cell_types)
    if ct_list is None:
        return varm_keys

    selected = []
    for ct in ct_list:
        selected.extend([k for k in varm_keys if str(k).startswith(f"{ct}_")])

    out = []
    seen = set()
    for key in selected:
        if key not in seen:
            out.append(key)
            seen.add(key)
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
    """Query ontology weights and modality loadings with a gene list or weighted vector."""
    results: Dict[str, Dict[str, object]] = {}
    targets = ["weights"] + list(ontology.mod.keys()) if search_in is None else list(search_in)
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
                f"Requested target '{target}' is not 'weights' and not found in ontology.mod. Available modalities: {list(ontology.mod.keys())}"
            )

        keys = _resolve_modality_loading_keys(ontology, target, cell_types=cell_types)
        if len(keys) == 0:
            loadings = modality_feature_loadings_to_df(
                ontology,
                target,
                cell_types=cell_types,
                strip_cell_type_suffix=len(_normalize_cell_types(cell_types) or []) == 1,
            )
            overlap = _overlap_summary(query_genes, loadings.index)
            df = _run_signature_query(signature, loadings, pval_thresh=pval_thresh, n_iter=n_iter, seed=seed)
            result_key = target if cell_types is None else f"{target}|{','.join(_normalize_cell_types(cell_types))}"
            results[result_key] = {"results": df, "overlap": overlap}
            continue

        for key in keys:
            loadings = modality_feature_loadings_to_df(ontology, target, key=key)
            overlap = _overlap_summary(query_genes, loadings.index)
            df = _run_signature_query(signature, loadings, pval_thresh=pval_thresh, n_iter=n_iter, seed=seed)
            ct = _key_to_cell_type(key)
            result_key = f"{target}|{ct}" if ct is not None else target
            results[result_key] = {"results": df, "overlap": overlap}

    return results
