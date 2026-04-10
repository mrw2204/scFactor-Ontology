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
    adata: ad.AnnData,
    ontology_keys: Union[str, Sequence[str]],
    groupby: str,
    reference: str = "rest",
    method: str = "wilcoxon",
    pseudo_bulk: bool = False,
    pseudo_bulk_by: Optional[Union[str, Sequence[str]]] = None,
    summary_metric: str = "mean",
    key_added: str = "rank_genes_groups",
    pval_thresh: Optional[float] = None,
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Test projected ontology scores between categories in ``adata.obs``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing projected ontology scores.
    ontology_keys : str or sequence of str
        Keys in ``adata.obsm`` (preferred) or ``adata.obs`` containing ontology-derived
        score matrices or columns.
    groupby : str
        Column in ``adata.obs`` defining the groups to compare.
    method : str, default="wilcoxon"
        Method passed to :func:`scanpy.tl.rank_genes_groups`.
    reference : str, default="rest"
        Group to compare to in :func:`scanpy.tl.rank_genes_groups`. Default "rest" does
        one-vs-all comparison.
    pseudo_bulk : bool, default=False
        Whether to aggregate scores before testing.
    pseudo_bulk_by : str or sequence of str, optional
        Columns defining pseudobulk units. Required if ``pseudo_bulk=True``.
    summary_metric : {"mean", "median", "sum"}, default="mean"
        Aggregation statistic used when pseudobulking.
    key_added : str, default="rank_genes_groups"
        Key under which Scanpy stores test results.
    pval_thresh : float, optional
        If provided, filter the returned result table to adjusted p-values below this
        threshold.

    Returns
    -------
    tuple
        ``(test_adata, results_df)`` where ``test_adata`` is the score-space AnnData used
        for testing and ``results_df`` is the long-form result table.
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"groupby '{groupby}' not found in adata.obs.")
    score_df = collect_score_matrices(adata, ontology_keys)
    if pseudo_bulk:
        if pseudo_bulk_by is None:
            raise ValueError("pseudo_bulk_by must be provided when pseudo_bulk=True.")
        if isinstance(pseudo_bulk_by, str):
            pseudo_bulk_by = [pseudo_bulk_by]
        agg_keys = list(dict.fromkeys([groupby] + list(pseudo_bulk_by)))
        missing = [k for k in agg_keys if k not in adata.obs.columns]
        if missing:
            raise KeyError(f"Missing pseudo_bulk_by/groupby columns in adata.obs: {missing}")
        grouped = score_df.join(adata.obs[agg_keys]).groupby(agg_keys, dropna=False)
        if summary_metric == "mean":
            agg = grouped.mean(numeric_only=True).dropna(how = 'all', axis = 0)
        elif summary_metric == "median":
            agg = grouped.median(numeric_only=True).dropna(how = 'all', axis = 0)
        elif summary_metric == "sum":
            agg = grouped.sum(numeric_only=True).dropna(how = 'all', axis = 0)
        else:
            raise ValueError("summary_metric must be 'mean', 'median', or 'sum'.")
        obs_pb = agg.index.to_frame(index=False)
        obs_pb.index = ["|".join(map(str, x)) for x in agg.index.to_list()]
        agg.index = obs_pb.index
        test_adata = ad.AnnData(X=agg.to_numpy(dtype=np.float32), obs=obs_pb, var=pd.DataFrame(index=agg.columns))
    else:
        obs_sc = adata.obs[[groupby]].copy()
        test_adata = ad.AnnData(X=score_df.to_numpy(dtype=np.float32), obs=obs_sc, var=pd.DataFrame(index=score_df.columns))
    sc.tl.rank_genes_groups(test_adata, groupby=groupby, method=method, key_added=key_added, reference=reference)
    res = sc.get.rank_genes_groups_df(test_adata, group=None, key=key_added)
    if pval_thresh is not None and "pvals_adj" in res.columns:
        res = res.loc[res["pvals_adj"] < pval_thresh].copy()
    return test_adata, res
