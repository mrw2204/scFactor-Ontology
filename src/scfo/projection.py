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
    """
    Project ontology factor weights onto an AnnData expression matrix.

    This function scores each observation in ``adata`` against ontology factor
    weights, either globally or in a lineage-restricted manner. Projection can
    be performed by direct dot product or by permutation-based enrichment.

    Two projection modes are supported:

    - **global projection** (``annotation_key=None``):
      every cell is scored against every ontology factor using the shared
      overlapping gene set
    - **annotation-matched projection** (``annotation_key`` provided):
      each cell is scored only against ontology factors whose
      ``ontology.obs["Classification"]`` matches that cell's annotation

    Results can either be written back into ``adata.obsm`` or returned as
    DataFrames.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the expression matrix to score.
    ontology : muon.MuData
        Ontology object containing factor weights and factor metadata.
    annotation_key : str, optional
        Column in ``adata.obs`` containing lineage or cell-type annotations used
        to restrict projection to ontology factors with matching
        ``ontology.obs["Classification"]`` values. If omitted, projection is
        performed against all ontology factors.
    layer : str, optional
        Layer in ``adata.layers`` to use instead of ``adata.X``.
    method : {"dot", "permutation"}, default="dot"
        Scoring method. ``"dot"`` computes a simple matrix product between
        expression and factor weights. ``"permutation"`` computes
        permutation-based enrichment scores and empirical p-values.
    n_iter : int, default=1000
        Number of permutations used when ``method="permutation"``.
    seed : int, default=0
        Random seed for permutation-based scoring.
    inplace : bool, default=True
        If ``True``, store results in ``adata`` and return ``None``. If
        ``False``, return score and p-value DataFrames instead.
    score_key_added : str, default="scfo_scores"
        Key used to store projected scores in ``adata.obsm`` when
        ``inplace=True``.
    pval_key_added : str, default="scfo_pvals"
        Key used to store projected p-values in ``adata.obsm`` when
        permutation-based projection is used and ``inplace=True``.
    score_columns_uns_key : str, optional
        Key used to store factor column names in ``adata.uns`` when sparse or
        array-like scores are written to ``adata.obsm``. If omitted, defaults
        to ``f"{score_key_added}_columns"``.
    store_sparse_scores : bool, default=True
        If ``True`` and ``inplace=True``, projected scores are stored as a CSR
        sparse matrix in ``adata.obsm`` and factor names are stored separately
        in ``adata.uns[score_columns_uns_key]``. If ``False``, scores are
        stored directly as a DataFrame.
    show_progress : bool, default=True
        Whether to show progress bars during permutation-based scoring.

    Returns
    -------
    None or tuple of (pandas.DataFrame, pandas.DataFrame or None)
        If ``inplace=True``, returns ``None`` after writing results into
        ``adata``. If ``inplace=False``, returns:

        - ``score_df``: projected factor scores
        - ``pval_df``: projected p-values if ``method="permutation"``, else
          ``None``

        In global mode, returned matrices are shaped
        ``n_obs x n_factors``. In annotation-matched mode, returned matrices
        contain only the cells and factors actually evaluated unless
        ``inplace=True``, in which case full-size matrices with ``NaN`` outside
        matched lineage blocks are written back.

    Raises
    ------
    ValueError
        If ``method`` is invalid, if no genes overlap between ``adata`` and the
        ontology weights, or if no overlapping lineage labels are found between
        ``adata.obs[annotation_key]`` and ``ontology.obs["Classification"]``.
    KeyError
        If required ontology fields, required ``adata.obs`` annotations, or
        required storage keys are missing.

    Notes
    -----
    In annotation-matched mode, each lineage block is scored independently using
    only the genes overlapping between that subset of cells and the matching
    subset of ontology factors.
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
    """
    Collapse lineage-suffixed projected factor scores to plain factor names.

    This is useful when projected ontology scores are stored per lineage-aware
    factor, for example ``Factor3|Tumor`` and ``Factor3|Mg_TAM``, and the user
    wants to collapse these into lineage-agnostic factor summaries such as
    ``Factor3``.

    The function first reconstructs a score DataFrame from ``adata.obsm``. It
    then strips suffixes after ``separator`` from the score columns, groups
    columns by the resulting base factor name, and aggregates across grouped
    columns.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing projected ontology scores.
    score_key : str, default="scfo_scores"
        Key in ``adata.obsm`` containing projected scores.
    output_key : str, optional
        If provided, the collapsed matrix is also written to
        ``adata.obsm[output_key]``.
    columns_uns_key : str, optional
        Key in ``adata.uns`` containing column names for ``adata.obsm[score_key]``
        when the stored object is sparse or array-like rather than a DataFrame.
        If omitted, defaults to ``f"{score_key}_columns"``.
    separator : str, default="|"
        Delimiter used to split lineage-qualified factor names.
    agg : {"sum", "max", "mean"}, default="sum"
        Aggregation used to combine lineage-specific columns sharing the same
        base factor name.
    store_sparse : bool, default=False
        If ``True`` and ``output_key`` is provided, store the collapsed matrix
        as CSR sparse format in ``adata.obsm[output_key]``. Otherwise store it
        as a DataFrame.
    output_columns_uns_key : str, optional
        Key in ``adata.uns`` used to store collapsed column names when
        ``store_sparse=True``. If omitted, defaults to
        ``f"{output_key}_columns"``.

    Returns
    -------
    pandas.DataFrame
        Collapsed score matrix with rows aligned to ``adata.obs_names`` and
        columns representing base factor names without lineage suffixes.

    Raises
    ------
    KeyError
        If ``score_key`` is missing from ``adata.obsm``, or if a sparse/array
        score matrix is present but the corresponding column-name key is absent
        from ``adata.uns``.
    ValueError
        If ``agg`` is not one of ``{"sum", "max", "mean"}``.
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
    """
    Normalize a cell-type selector to a list.

    Parameters
    ----------
    cell_types : str, sequence of str, or None
        User-supplied lineage selection.

    Returns
    -------
    list or None
        Returns ``None`` if no selection was provided, a one-element list if a
        single string was provided, or ``list(cell_types)`` otherwise.
    """
    if cell_types is None:
        return None
    if isinstance(cell_types, str):
        return [cell_types]
    return list(cell_types)


def _is_gene_set_query(signature: Union[Sequence[str], pd.Series, pd.DataFrame]) -> bool:
    """
    Determine whether a query signature should be treated as an unweighted gene set.

    Parameters
    ----------
    signature : sequence of str, pandas.Series, or pandas.DataFrame
        Query signature supplied by the user.

    Returns
    -------
    bool
        ``True`` if ``signature`` is a plain sequence and should therefore be
        treated as an unweighted gene set for GSEA-style querying. ``False`` if
        it is a weighted ``Series`` or one-column ``DataFrame`` and should be
        treated as a scored vector for permutation-based enrichment.
    """
    return not isinstance(signature, (pd.Series, pd.DataFrame))


def _signature_genes(signature: Union[Sequence[str], pd.Series, pd.DataFrame]) -> pd.Index:
    """
    Extract the gene identifiers from a query signature.

    Parameters
    ----------
    signature : sequence of str, pandas.Series, or pandas.DataFrame
        Query signature. For weighted signatures, genes are taken from the
        index. For unweighted gene sets, genes are taken directly from the
        sequence values.

    Returns
    -------
    pandas.Index
        Unique gene names as strings, with duplicates removed while preserving
        pandas index semantics.

    Raises
    ------
    ValueError
        If ``signature`` is a DataFrame with more than one column.
    """
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
    """
    Summarize the overlap between a query gene set and a target gene universe.

    Parameters
    ----------
    query_genes : pandas.Index
        Genes present in the query signature.
    target_genes : pandas.Index
        Genes present in the ontology target being queried.

    Returns
    -------
    dict
        Dictionary containing:

        - ``query_n``: number of query genes
        - ``target_n``: number of target genes
        - ``overlap_n``: number of overlapping genes
        - ``overlap_frac_query``: fraction of query genes overlapping target
        - ``overlap_frac_target``: fraction of target genes overlapping query
        - ``overlap_genes``: list of overlapping genes
    """
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
    """
    Score a user-supplied signature against an ontology loading matrix.

    This helper dispatches to one of two methods depending on the type of
    ``signature``:

    - unweighted gene sets are queried using preranked GSEA
    - weighted signatures are queried using permutation-based enrichment

    In both cases, results are returned as a ranked DataFrame with one row per
    ontology feature, factor, or loading column.

    Parameters
    ----------
    signature : sequence of str, pandas.Series, or pandas.DataFrame
        Query signature. Plain sequences are treated as gene sets, while Series
        and one-column DataFrames are treated as weighted signatures.
    loadings : pandas.DataFrame
        Target loading matrix with genes in rows and ontology features in
        columns.
    pval_thresh : float, optional
        Optional p-value filter applied after scoring.
    n_iter : int, default=1000
        Number of permutations for either GSEA or permutation-based enrichment.
    seed : int, default=0
        Random seed for the scoring routine.

    Returns
    -------
    pandas.DataFrame
        Ranked result table with columns:

        - ``score``: enrichment score
        - ``pval``: p-value
        - ``method``: scoring method used, either ``"gsea"`` or
          ``"permutation"``

        Rows are sorted by descending score.

    Raises
    ------
    ValueError
        If the query signature shares no genes with ``loadings.index``.
    """
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
    """
    Determine which ``varm`` loading keys should be queried for a modality.

    This helper supports older or more complex modality storage layouts in which
    feature-loadings may be stored under multiple lineage-specific ``varm``
    keys. When a modality has only one key, or when its declared row axis is
    already ``"gene|cell_type"``, an empty list is returned to indicate that the
    default loading accessor should be used instead of key-specific retrieval.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    modality : str
        Modality name whose loading keys should be inspected.
    cell_types : str, sequence of str, or None, optional
        Optional lineage restriction used to select only lineage-specific keys.

    Returns
    -------
    list
        List of selected ``varm`` keys. An empty list indicates that either no
        special key resolution is needed or no key-based subdivision applies.
    """
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
    """
    Infer a cell-type label from a modality ``varm`` key.

    Parameters
    ----------
    varm_key : str
        ``varm`` key name, typically of the form ``celltype_something``.

    Returns
    -------
    str or None
        The substring before the first underscore, or ``None`` if no underscore
        is present.
    """
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
    """
    Query ontology weights and modality feature-loadings with a gene signature.

    This function provides a unified interface for searching an ontology with
    either:

    - an unweighted gene set
    - a weighted gene signature stored as a Series or one-column DataFrame

    The query can be run against the global factor weights and/or against any
    modality that stores feature-loadings. Each target returns both a ranked
    result table and a summary of gene overlap between the query and the target
    gene universe.

    Parameters
    ----------
    signature : sequence of str, pandas.Series, or pandas.DataFrame
        Query signature. Plain sequences are treated as gene sets and scored via
        GSEA. Series and one-column DataFrames are treated as weighted
        signatures and scored via permutation-based enrichment.
    ontology : muon.MuData
        Ontology object to search.
    search_in : sequence of str, optional
        Specific targets to search. Valid entries include ``"weights"`` and any
        modality present in ``ontology.mod``. If omitted, the search includes
        ``"weights"`` plus all ontology modalities.
    pval_thresh : float, optional
        Optional p-value threshold applied to each result table after scoring.
    n_iter : int, default=1000
        Number of permutations used in GSEA or permutation-based enrichment.
    seed : int, default=0
        Random seed.
    cell_types : str, sequence of str, or None, optional
        Optional lineage restriction applied when retrieving ontology weights or
        modality feature-loadings.

    Returns
    -------
    dict of {str: dict}
        Mapping from target name to a dictionary with two entries:

        - ``"results"``: ranked DataFrame of enrichment results
        - ``"overlap"``: overlap summary dictionary describing shared genes
          between the query and the target

        For modalities with lineage-specific loading keys, separate entries may
        be returned using labels such as ``"regulons|Tumor"``. For lineage-
        restricted queries without key-specific splitting, result keys may take
        forms such as ``"hallmark|Tumor"`` or
        ``"hallmark|Tumor,Mg_TAM"``.

    Raises
    ------
    KeyError
        If a requested target is neither ``"weights"`` nor a modality present in
        ``ontology.mod``.
    ValueError
        If a query cannot be evaluated because there is no gene overlap between
        the signature and a target loading matrix.

    Notes
    -----
    The function de-duplicates the target list while preserving order. The
    global ``"weights"`` target is always treated specially and queried using
    ``factor_weights_to_df``.
    """
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
