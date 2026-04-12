from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def collect_score_matrices(
    adata: ad.AnnData,
    ontology_keys: Union[str, Sequence[str]],
) -> pd.DataFrame:
    """
    Collect one or more ontology-derived score matrices or metadata columns into a single DataFrame.

    This helper pulls requested features from either ``adata.obsm`` or
    ``adata.obs`` and concatenates them column-wise into a single
    observation-by-feature matrix suitable for downstream testing.

    Retrieval rules are:

    - if a requested key is present in ``adata.obsm``, it must be stored as a
      pandas DataFrame and is appended directly
    - otherwise, if the key is present in ``adata.obs.columns``, that single
      metadata column is appended
    - otherwise, an error is raised

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing projected ontology score matrices in
        ``.obsm`` and/or scalar metadata in ``.obs``.
    ontology_keys : str or sequence of str
        Key or keys to retrieve. Each key may refer either to a DataFrame stored
        in ``adata.obsm`` or to a single column in ``adata.obs``.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with observations as rows and collected ontology
        score columns as columns. The row index matches ``adata.obs_names``.

    Raises
    ------
    KeyError
        If any requested key is found in neither ``adata.obsm`` nor
        ``adata.obs.columns``.
    TypeError
        If a requested key is present in ``adata.obsm`` but the stored object is
        not a pandas DataFrame.
    ValueError
        If the concatenated output contains duplicate column names.

    Notes
    -----
    This function currently requires projected score matrices in ``adata.obsm``
    to be stored as DataFrames rather than sparse arrays or NumPy matrices, in
    order to preserve column labels portably.
    """
    if isinstance(ontology_keys, str):
        ontology_keys = [ontology_keys]
    mats: List[pd.DataFrame] = []
    for key in ontology_keys:
        if key in adata.obsm:
            obj = adata.obsm[key]
            if isinstance(obj, pd.DataFrame):
                df = obj.copy()
                df.index = adata.obs_names
            else:
                raise TypeError(f"adata.obsm['{key}'] is not a DataFrame. Store projected ontology scores as DataFrames for portability.")
            mats.append(df)
        elif key in adata.obs.columns:
            mats.append(pd.DataFrame(adata.obs[[key]].copy()))
        else:
            raise KeyError(f"'{key}' not found in adata.obsm or adata.obs.")
    out = pd.concat(mats, axis=1)
    if out.columns.duplicated().any():
        raise ValueError("Collected ontology score columns contain duplicates.")
    return out


def diff_exp_ontology(
    adata_in: ad.AnnData,
    ontology_keys: Union[str, Sequence[str]],
    groupby: str,
    cell_type: Optional[str] = None,
    cell_type_key: str = "classification",
    reference: str = "rest",
    method: str = "wilcoxon",
    pseudo_bulk: bool = False,
    pseudo_bulk_by: Optional[Union[str, Sequence[str]]] = None,
    min_cells_pseudobulk: Optional[int] = None,
    min_cells_pseudo_bulk: Optional[int] = None,
    summary_metric: str = "mean",
    key_added: str = "rank_genes_groups",
    pval_thresh: Optional[float] = None,
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """
    Test projected ontology scores for differential abundance across groups.

    This function treats projected ontology scores as the variables of interest
    and compares them across categories defined by ``adata_in.obs[groupby]``
    using :func:`scanpy.tl.rank_genes_groups`. The input score space is built by
    collecting one or more matrices or columns via :func:`collect_score_matrices`.

    Two analysis modes are supported:

    - **single-cell mode**: each observation in ``adata_in`` is tested directly
    - **pseudobulk mode**: cells are aggregated across combinations of
      ``groupby`` and ``pseudo_bulk_by`` using the requested summary statistic,
      and testing is then performed on the aggregated samples

    Optional cell-type subsetting can be applied before either workflow.

    Parameters
    ----------
    adata_in : anndata.AnnData
        AnnData object containing projected ontology scores and grouping
        metadata.
    ontology_keys : str or sequence of str
        Keys identifying ontology-derived score matrices or scalar columns to
        test. These are passed to :func:`collect_score_matrices`.
    groupby : str
        Column in ``adata_in.obs`` defining the groups to compare.
    cell_type : str, optional
        If provided, subset to observations satisfying
        ``adata_in.obs[cell_type_key] == cell_type`` before testing.
    cell_type_key : str, default="classification"
        Column in ``adata_in.obs`` used for optional cell-type subsetting.
    reference : str, default="rest"
        Reference group passed to :func:`scanpy.tl.rank_genes_groups`. The
        default ``"rest"`` performs one-vs-all comparisons.
    method : str, default="wilcoxon"
        Statistical test used by :func:`scanpy.tl.rank_genes_groups`.
    pseudo_bulk : bool, default=False
        Whether to aggregate cells into pseudobulk units prior to testing.
    pseudo_bulk_by : str or sequence of str, optional
        One or more columns in ``adata_in.obs`` defining pseudobulk units.
        Required when ``pseudo_bulk=True``. Aggregation is performed across the
        unique combinations of ``groupby`` and ``pseudo_bulk_by``.
    min_cells_pseudobulk : int, optional
        Minimum number of cells required for a pseudobulk group to be retained.
        Only used when ``pseudo_bulk=True``.
    min_cells_pseudo_bulk : int, optional
        Deprecated alias for ``min_cells_pseudobulk`` retained for backward
        compatibility.
    summary_metric : {"mean", "median", "sum"}, default="mean"
        Summary statistic used to aggregate ontology scores within each
        pseudobulk group.
    key_added : str, default="rank_genes_groups"
        Key under which Scanpy stores differential testing results in
        ``test_adata.uns``.
    pval_thresh : float, optional
        If provided, filter the returned long-form results table to rows with
        ``pvals_adj < pval_thresh`` when adjusted p-values are available.

    Returns
    -------
    test_adata : anndata.AnnData
        AnnData object actually used for testing. In single-cell mode this
        contains the collected ontology score matrix for the selected cells. In
        pseudobulk mode this contains the aggregated pseudobulk score matrix,
        with pseudobulk metadata in ``.obs`` and an additional ``n_cells``
        column recording how many cells contributed to each pseudobulk sample.
    results_df : pandas.DataFrame
        Long-form differential testing results returned by
        :func:`scanpy.get.rank_genes_groups_df`.

    Raises
    ------
    KeyError
        If required columns such as ``groupby``, ``cell_type_key``, or
        ``pseudo_bulk_by`` columns are missing from ``adata_in.obs``.
    ValueError
        If inputs are inconsistent, if no observations remain after subsetting,
        if no ontology score columns are collected, if no pseudobulk samples
        remain after filtering/aggregation, if fewer than two groups remain for
        testing, or if a non-``"rest"`` reference is absent from the grouped
        categories.

    Notes
    -----
    In pseudobulk mode, aggregation is performed after joining collected
    ontology scores with the requested metadata columns. Pseudobulk samples are
    indexed by concatenating grouping values with ``"|"``.
    """
    if groupby not in adata_in.obs.columns:
        raise KeyError(f"groupby '{groupby}' not found in adata_in.obs.")

    # Backward-compatible handling of old parameter name
    if min_cells_pseudobulk is not None and min_cells_pseudo_bulk is not None:
        if min_cells_pseudobulk != min_cells_pseudo_bulk:
            raise ValueError(
                "Received conflicting values for 'min_cells_pseudobulk' and "
                "'min_cells_pseudo_bulk'. Please use only 'min_cells_pseudobulk'."
            )
    if min_cells_pseudobulk is None:
        min_cells_pseudobulk = min_cells_pseudo_bulk

    if min_cells_pseudobulk is not None:
        if not pseudo_bulk:
            raise ValueError(
                "'min_cells_pseudobulk' is only valid when pseudo_bulk=True."
            )
        if min_cells_pseudobulk < 1:
            raise ValueError("'min_cells_pseudobulk' must be >= 1.")

    # Optional cell-type subsetting
    if cell_type is None:
        adata = adata_in.copy()
    else:
        if cell_type_key not in adata_in.obs.columns:
            raise KeyError(f"cell_type_key '{cell_type_key}' not found in adata_in.obs.")
        adata = adata_in[adata_in.obs[cell_type_key] == cell_type].copy()

    if adata.n_obs == 0:
        raise ValueError("No observations remain after subsetting.")

    score_df = collect_score_matrices(adata, ontology_keys)
    if score_df.shape[1] == 0:
        raise ValueError("No ontology score columns were collected.")
    if score_df.shape[0] != adata.n_obs:
        raise ValueError(
            "collect_score_matrices returned a score matrix with a different number "
            "of rows than adata.n_obs."
        )

    if pseudo_bulk:
        if pseudo_bulk_by is None:
            raise ValueError("pseudo_bulk_by must be provided when pseudo_bulk=True.")
        if isinstance(pseudo_bulk_by, str):
            pseudo_bulk_by = [pseudo_bulk_by]

        agg_keys = list(dict.fromkeys([groupby] + list(pseudo_bulk_by)))
        missing = [k for k in agg_keys if k not in adata.obs.columns]
        if missing:
            raise KeyError(f"Missing pseudo_bulk_by/groupby columns in adata.obs: {missing}")

        meta_df = adata.obs[agg_keys].copy()
        score_df_join = score_df.join(meta_df)

        # Remove pseudobulk groups with too few cells before aggregation
        if min_cells_pseudobulk is not None:
            group_sizes = score_df_join.groupby(agg_keys, dropna=False).size()
            keep_groups = group_sizes[group_sizes >= min_cells_pseudobulk].index

            if len(keep_groups) == 0:
                raise ValueError(
                    "No pseudobulk groups remain after applying "
                    f"min_cells_pseudobulk={min_cells_pseudobulk}."
                )

            keep_mask = pd.MultiIndex.from_frame(meta_df).isin(keep_groups)
            score_df_join = score_df_join.loc[keep_mask].copy()

            if score_df_join.shape[0] == 0:
                raise ValueError(
                    "No cells remain after filtering pseudobulk groups by size."
                )

        grouped = score_df_join.groupby(agg_keys, dropna=False)

        if summary_metric == "mean":
            agg = grouped.mean(numeric_only=True)
        elif summary_metric == "median":
            agg = grouped.median(numeric_only=True)
        elif summary_metric == "sum":
            agg = grouped.sum(numeric_only=True)
        else:
            raise ValueError("summary_metric must be 'mean', 'median', or 'sum'.")

        agg = agg.dropna(how="all", axis=0)

        if agg.shape[0] == 0:
            raise ValueError("No pseudobulk observations remain after aggregation.")

        obs_pb = agg.index.to_frame(index=False)
        obs_pb.index = ["|".join(map(str, x)) for x in agg.index.to_list()]
        agg.index = obs_pb.index

        # Add n_cells per pseudobulk sample for transparency
        n_cells = grouped.size().reindex(agg.index if isinstance(agg.index, pd.Index) else grouped.size().index)
        # Easier/safer: recompute on filtered metadata to ensure exact alignment
        n_cells = score_df_join.groupby(agg_keys, dropna=False).size()
        n_cells.index = ["|".join(map(str, x)) for x in n_cells.index.to_list()]
        obs_pb["n_cells"] = n_cells.reindex(obs_pb.index).astype(int).values

        test_adata = ad.AnnData(
            X=agg.to_numpy(dtype=np.float32),
            obs=obs_pb,
            var=pd.DataFrame(index=pd.Index(map(str, agg.columns), name="score")),
        )
    else:
        obs_sc = adata.obs[[groupby]].copy()
        obs_sc.index = score_df.index

        test_adata = ad.AnnData(
            X=score_df.to_numpy(dtype=np.float32),
            obs=obs_sc,
            var=pd.DataFrame(index=pd.Index(map(str, score_df.columns), name="score")),
        )

    # Ensure grouping column is categorical and valid for Scanpy
    test_adata.obs[groupby] = pd.Categorical(test_adata.obs[groupby])
    test_adata.obs[groupby] = test_adata.obs[groupby].cat.remove_unused_categories()

    n_groups = test_adata.obs[groupby].nunique(dropna=True)
    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 groups in '{groupby}' for differential testing; found {n_groups}."
        )

    if reference != "rest" and reference not in test_adata.obs[groupby].cat.categories:
        raise ValueError(
            f"reference '{reference}' not found among groups in '{groupby}'."
        )

    sc.tl.rank_genes_groups(
        test_adata,
        groupby=groupby,
        method=method,
        key_added=key_added,
        reference=reference,
    )

    res = sc.get.rank_genes_groups_df(test_adata, group=None, key=key_added)

    if pval_thresh is not None and "pvals_adj" in res.columns:
        res = res.loc[res["pvals_adj"] < pval_thresh].copy()

    return test_adata, res
