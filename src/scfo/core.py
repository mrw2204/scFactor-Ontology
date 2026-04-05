
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, Any

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import re
import scipy.sparse as sp
from tqdm.auto import tqdm

try:
    import gseapy as gp
except Exception:  # pragma: no cover - optional dependency
    gp = None

DEFAULT_CELL_TYPES: List[str] = [
    "Tumor",
    "Mo_TAM",
    "Mg_TAM",
    "T cell",
    "Neuron",
    "Oligo",
    "Pericyte",
    "Endothelial",
    "Astrocyte",
    "Non_T_Lymphocyte",
    "Non_Mac_Myelocyte",
]


@dataclass(frozen=True)
class MatrixPair:
    score: pd.DataFrame
    pval: pd.DataFrame


def require_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Validate that an object is a well-formed pandas DataFrame.

    Parameters
    ----------
    df
        Object to validate.
    name
        Human-readable argument name used in error messages.

    Returns
    -------
    pandas.DataFrame
        The validated input DataFrame.

    Raises
    ------
    ValueError
        If ``df`` is ``None`` or if its index or columns contain duplicates.
    TypeError
        If ``df`` is not a pandas DataFrame.
    """
    if df is None:
        raise ValueError(f"{name} is required.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if df.index.has_duplicates:
        raise ValueError(f"{name}.index contains duplicates.")
    if df.columns.has_duplicates:
        raise ValueError(f"{name}.columns contains duplicates.")
    return df


def ensure_no_nan(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensure that a DataFrame contains no missing values.

    Parameters
    ----------
    df
        DataFrame to validate.
    name
        Human-readable argument name used in error messages.

    Returns
    -------
    pandas.DataFrame
        The validated DataFrame.

    Raises
    ------
    ValueError
        If any entry in the DataFrame is ``NaN``.
    """
    if df.isna().any().any():
        raise ValueError(
            f"{name} contains NaN values. Fill or remove missing values before using ontology_tools "
            f"(for example, factor_loadings = factor_loadings.fillna(0))."
        )
    return df


def as_dense(x):
    """
    Convert a dense or sparse matrix-like object to a NumPy array.

    Parameters
    ----------
    x
        Dense or sparse array-like object.

    Returns
    -------
    numpy.ndarray
        Dense representation of ``x``.
    """
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def as_csr(x: Union[np.ndarray, sp.spmatrix]) -> sp.csr_matrix:
    """
    Convert a dense or sparse matrix-like object to CSR format.

    Parameters
    ----------
    x
        Dense or sparse array-like object.

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR representation of ``x``.
    """
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def row_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each row of a DataFrame independently.

    Parameters
    ----------
    df
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Row-wise z-scored matrix. Rows with zero variance are returned as zeros.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    out = df.sub(mean, axis=0).div(std, axis=0)
    return out.fillna(0.0)


def safe_factor_metadata(factor_names: Sequence[str], separator: str = "|") -> pd.DataFrame:
    """
    Construct standard factor metadata from factor names.

    Parameters
    ----------
    factor_names
        Sequence of factor names, typically in the form ``FactorN|CellType``.
    separator
        Separator used between the base factor name and the classification label.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by factor name with columns ``FactorName``,
        ``Classification``, and ``Number``.

    Notes
    -----
    If a factor name does not contain ``separator``, its classification is set
    to ``"Unknown"``.
    """
    idx = pd.Index([str(x) for x in factor_names], name="factor")
    meta = pd.DataFrame(index=idx)
    left = idx.astype(str).str.split(separator).str[0]
    right = idx.astype(str).str.split(separator).str[1]
    meta["FactorName"] = idx
    meta["Classification"] = right.where(right.notna(), "Unknown")
    num = left.str.extract(r"Factor(\d+)", expand=False)
    meta["Number"] = pd.to_numeric(num, errors="coerce")
    return meta


def subset_factor_loadings_by_cell_type(
    factor_loadings: pd.DataFrame,
    cell_type: str,
    separator: str = "|",
) -> pd.DataFrame:
    """
    Restrict a gene-by-factor loading matrix to one factor lineage.

    Parameters
    ----------
    factor_loadings
        Gene-by-factor matrix with factor names in the columns.
    cell_type
        Cell-type / lineage label to retain.
    separator
        Separator used in factor names.

    Returns
    -------
    pandas.DataFrame
        Copy of ``factor_loadings`` restricted to factors whose parsed
        classification equals ``cell_type``.
    """
    meta = safe_factor_metadata(factor_loadings.columns, separator=separator)
    keep = meta.index[meta["Classification"] == cell_type]
    return factor_loadings.loc[:, keep].copy()


def build_modality_adata(
    score_df: pd.DataFrame,
    pval_df: Optional[pd.DataFrame] = None,
    feature_loadings: Optional[pd.DataFrame] = None,
    factor_metadata: Optional[pd.DataFrame] = None,
    feature_loading_key: str = "feature_loadings",
    cell_type_separator: str = "|",
) -> ad.AnnData:
    """
    Build an AnnData container for a single ontology modality.

    Parameters
    ----------
    score_df
        Factor-by-feature score matrix. Rows correspond to ontology factors and
        columns correspond to modality features.
    pval_df
        Optional factor-by-feature p-value matrix aligned to ``score_df``.
    feature_loadings
        Optional loading matrix stored alongside the modality. Expected shape is
        ``loading_row x feature``, where ``loading_row`` is either plain genes
        or ``gene|cell_type`` labels.
    factor_metadata
        Optional metadata table aligned to factor rows. When provided, it is
        stored in ``.obs``.
    feature_loading_key
        Key under which the transposed loading matrix is stored in ``.varm``.
    cell_type_separator
        Separator used to detect whether ``feature_loadings.index`` is tagged as
        ``gene|cell_type``.

    Returns
    -------
    anndata.AnnData
        Modality container with scores in ``.X``, optional p-values in
        ``layers["pval"]``, and optional feature loadings in
        ``varm[feature_loading_key]``.
    """
    require_dataframe(score_df, "score_df")
    score_df = score_df.copy()
    score_df.index = score_df.index.astype(str)
    score_df.columns = score_df.columns.astype(str)

    if pval_df is not None:
        require_dataframe(pval_df, "pval_df")
        pval_df = pval_df.reindex(index=score_df.index, columns=score_df.columns)

    if feature_loadings is not None:
        require_dataframe(feature_loadings, "feature_loadings")
        feature_loadings = feature_loadings.copy()
        feature_loadings.index = feature_loadings.index.astype(str)
        feature_loadings.columns = feature_loadings.columns.astype(str)
        feature_loadings = feature_loadings.reindex(columns=score_df.columns).fillna(0.0)

    obs = (
        factor_metadata.reindex(score_df.index).copy()
        if factor_metadata is not None
        else pd.DataFrame(index=score_df.index)
    )
    var = pd.DataFrame(index=score_df.columns)

    adata = ad.AnnData(
        X=score_df.to_numpy(dtype=np.float32),
        obs=obs,
        var=var,
    )
    adata.obs_names = score_df.index.astype(str)
    adata.var_names = score_df.columns.astype(str)

    if pval_df is not None:
        adata.layers["pval"] = pval_df.to_numpy(dtype=np.float32)

    if feature_loadings is not None:
        adata.varm[feature_loading_key] = as_csr(feature_loadings.T.to_numpy(dtype=np.float32))
        adata.uns["gene_names"] = list(feature_loadings.index.astype(str))
        adata.uns["feature_loading_row_axis"] = (
            "gene|cell_type"
            if _index_has_cell_type_tags(feature_loadings.index, separator=cell_type_separator)
            else "gene"
        )

    return adata


def signature_to_df(
    signature: Union[Sequence[str], pd.Series, pd.DataFrame],
    gene_index: Sequence[str],
    name: str = "signature",
) -> pd.DataFrame:
    """
    Convert a gene list or weighted gene signature to a one-column DataFrame.

    Parameters
    ----------
    signature
        Sequence of genes, weighted Series, or one-column DataFrame.
    gene_index
        Reference gene universe used for reindexing.
    name
        Column name for the returned signature.

    Returns
    -------
    pandas.DataFrame
        One-column gene-by-weight DataFrame aligned to ``gene_index``.
    """
    genes = pd.Index(gene_index, name="gene")
    if isinstance(signature, pd.DataFrame):
        if signature.shape[1] != 1:
            raise ValueError("signature DataFrame must have exactly one column.")
        ser = signature.iloc[:, 0].copy()
        ser.name = name
    elif isinstance(signature, pd.Series):
        ser = signature.copy()
        ser.name = name
    else:
        ser = pd.Series(1.0, index=pd.Index(list(signature), dtype=str), name=name)
    ser.index = ser.index.astype(str)
    ser = ser.groupby(level=0).mean()
    ser = ser.reindex(genes).fillna(0.0)
    return ser.to_frame()


def get_matrix_from_adata(adata: ad.AnnData, layer: Optional[str] = None) -> pd.DataFrame:
    """
    Extract an AnnData matrix as a labeled DataFrame.

    Parameters
    ----------
    adata
        AnnData object to extract from.
    layer
        Optional layer name. If omitted, ``adata.X`` is used.

    Returns
    -------
    pandas.DataFrame
        Dense DataFrame with ``adata.obs_names`` as rows and ``adata.var_names``
        as columns.
    """
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        X = adata.layers[layer]
    return pd.DataFrame(as_dense(X), index=adata.obs_names, columns=adata.var_names)


def calc_enrichment(
    samples_by_genes: pd.DataFrame,
    signatures: pd.DataFrame,
    n_iter: int = 1000,
    seed: int = 0,
    show_progress: bool = True,
    progress_message: str = 'Permuting...',
) -> MatrixPair:
    X_df = ensure_no_nan(require_dataframe(samples_by_genes, "samples_by_genes"), "samples_by_genes")
    W_df = ensure_no_nan(require_dataframe(signatures, "signatures"), "signatures")
    common_genes = X_df.columns.intersection(W_df.index)
    if len(common_genes) == 0:
        raise ValueError("No overlap between samples_by_genes.columns and signatures.index.")
    X_df = X_df.loc[:, common_genes]
    W_df = W_df.loc[common_genes, :]
    X = X_df.to_numpy(dtype=np.float32, copy=False)
    W = W_df.to_numpy(dtype=np.float32, copy=False)
    n_samples, n_genes = X.shape
    obs = X @ W
    rng = np.random.default_rng(seed)
    null_mean = np.zeros_like(obs, dtype=np.float64)
    null_M2 = np.zeros_like(obs, dtype=np.float64)
    count_ge = np.ones_like(obs, dtype=np.int32)
    count_le = np.ones_like(obs, dtype=np.int32)
    iterator = range(1, n_iter + 1)
    if show_progress:
        iterator = tqdm(iterator, desc=progress_message, leave=True)
    for t in iterator:
        perm = rng.permutation(n_genes)
        null_t = X @ W[perm, :]
        delta = null_t - null_mean
        null_mean += delta / t
        delta2 = null_t - null_mean
        null_M2 += delta * delta2
        count_ge += (null_t >= obs)
        count_le += (null_t <= obs)
    null_var = null_M2 / (n_iter - 1) if n_iter > 1 else np.zeros_like(null_M2)
    null_std = np.sqrt(null_var)
    null_std[null_std == 0] = np.nan
    z = (obs - null_mean) / null_std
    p_ge = count_ge / (n_iter + 1)
    p_le = count_le / (n_iter + 1)
    p = np.minimum(1.0, 2.0 * np.minimum(p_ge, p_le))
    return MatrixPair(
        score=pd.DataFrame(z, index=X_df.index, columns=W_df.columns),
        pval=pd.DataFrame(p, index=X_df.index, columns=W_df.columns),
    )


def gsea_enrichment(
    factor_loadings: pd.DataFrame,
    gene_sets: Union[Mapping[str, Sequence[str]], str],
    permutation_num: int = 1000,
    min_size: int = 10,
    max_size: int = 5000,
    seed: int = 42,
    processes: int = 1,
    show_progress: bool = True,
    progress_message: str = 'Permuting...',
) -> MatrixPair:
    """Robust preranked GSEA for sparse loading matrices.

    Adds a tiny deterministic jitter to break massive ties, adapts ``min_size``
    for small query overlaps, and fails gracefully for pathological ranking
    vectors by returning neutral scores (NES=0, p=1).
    """
    if gp is None:
        raise ImportError("gseapy is required for gsea_enrichment().")

    fl = ensure_no_nan(require_dataframe(factor_loadings, "factor_loadings"), "factor_loadings")

    if isinstance(gene_sets, Mapping):
        gene_sets_clean = {
            str(k): [str(g) for g in v if str(g) in fl.index]
            for k, v in gene_sets.items()
        }
    else:
        gene_sets_clean = gene_sets

    nes_ls = []
    p_ls = []
    iterator = fl.columns
    if show_progress:
        iterator = tqdm(iterator, desc=progress_message, leave=True)
    for factor in iterator:
        rnk = fl[factor].astype(float).copy()

        # break extreme ties deterministically
        scale = max(float(np.nanmax(np.abs(rnk.values))), 1.0)
        eps = scale * 1e-12
        jitter = pd.Series(np.linspace(0, eps, len(rnk), endpoint=False), index=rnk.index)
        rnk = (rnk + jitter).sort_values(ascending=False)

        min_size_eff = min_size
        if isinstance(gene_sets_clean, Mapping):
            overlaps = [len(set(gs).intersection(rnk.index)) for gs in gene_sets_clean.values()]
            if len(overlaps) == 0 or max(overlaps) == 0:
                terms = list(gene_sets_clean.keys())
                nes = pd.Series({k: 0.0 for k in terms}, name=factor)
                pval = pd.Series({k: 1.0 for k in terms}, name=factor)
                nes_ls.append(nes)
                p_ls.append(pval)
                continue
            min_size_eff = max(1, min(min_size, max(overlaps)))

        try:
            res = gp.prerank(
                rnk=rnk,
                gene_sets=gene_sets_clean,
                permutation_num=permutation_num,
                min_size=min_size_eff,
                max_size=max_size,
                seed=seed,
                processes=processes,
                outdir=None,
                no_plot=True,
                verbose=False,
            )
            res2d = res.res2d.copy()
            if res2d.empty:
                terms = list(gene_sets_clean.keys()) if isinstance(gene_sets_clean, Mapping) else []
                nes = pd.Series({k: 0.0 for k in terms}, name=factor)
                pval = pd.Series({k: 1.0 for k in terms}, name=factor)
            else:
                nes = res2d.set_index("Term")["NES"]
                pval = res2d.set_index("Term")["NOM p-val"]
                nes.name = factor
                pval.name = factor
        except Exception:
            terms = list(gene_sets_clean.keys()) if isinstance(gene_sets_clean, Mapping) else []
            nes = pd.Series({k: 0.0 for k in terms}, name=factor)
            pval = pd.Series({k: 1.0 for k in terms}, name=factor)

        nes_ls.append(nes)
        p_ls.append(pval)

    nes_df = pd.concat(nes_ls, axis=1).T.fillna(0.0)
    p_df = pd.concat(p_ls, axis=1).T.reindex(index=nes_df.index, columns=nes_df.columns).fillna(1.0)
    return MatrixPair(score=nes_df, pval=p_df)


def make_factor_gene_salience(
    factor_loadings: pd.DataFrame,
    cell_types: Sequence[str],
    separator: str = "|",
) -> pd.DataFrame:
    """Replicate original behavior: factor_loadings / factor_loadings.std(), then abs-sum per lineage."""
    fl = ensure_no_nan(require_dataframe(factor_loadings, "factor_loadings"), "factor_loadings")
    fl_norm = fl.div(fl.std(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    out: Dict[str, pd.Series] = {}
    for ct in cell_types:
        cols = [c for c in fl_norm.columns if ct in str(c)]
        out[ct] = fl_norm.loc[:, cols].abs().sum(axis=1) if len(cols) > 0 else pd.Series(0.0, index=fl.index)
    return pd.DataFrame(out)


def make_filtered_lr_signatures(
    liana: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    cell_types: Sequence[str] = DEFAULT_CELL_TYPES,
    plot_scores: bool = True,
    liana_z_threshold: float = 0.0,
    factor_z_threshold: float = 0.0,
    plot_size: float = 0.1,
):
    """Preserve original LIANA-signature construction logic exactly.

    Returns
    -------
    all_send_signatures
        Rows are ``sender_cell_type|ligand_gene`` and columns are ``rec by|receiver_cell_type``.
    all_rec_signatures
        Rows are ``receiver_cell_type|receptor_gene`` and columns are ``sent by|sender_cell_type``.
    filtered
        Long-form filtered LIANA table.
    all_biopsy_factors_mag
        Pivoted magnitude matrix.
    all_biopsy_factors_spec
        Pivoted specificity matrix.
    """
    required = {"source", "target", "ligand_complex", "receptor_complex", "magnitude_rank", "specificity_rank"}
    missing = required.difference(liana.columns)
    if missing:
        raise ValueError(f"liana is missing required columns: {sorted(missing)}")

    fl = ensure_no_nan(require_dataframe(factor_loadings, "factor_loadings"), "factor_loadings")
    liana_sub = liana[["source", "target", "ligand_complex", "receptor_complex", "magnitude_rank", "specificity_rank"]].copy()
    liana_sub["source_target"] = liana_sub["source"].astype(str) + "->" + liana_sub["target"].astype(str)
    liana_sub["ligand_receptor"] = liana_sub["ligand_complex"].astype(str) + "->" + liana_sub["receptor_complex"].astype(str)

    all_piv = []
    for cl in liana_sub.source.unique():
        sub = liana_sub[liana_sub.source == cl].drop(columns=["receptor_complex", "target", "ligand_complex", "source", "specificity_rank"])[["source_target", "ligand_receptor", "magnitude_rank"]]
        piv = sub.pivot_table(index="ligand_receptor", columns="source_target", values="magnitude_rank", aggfunc="first").fillna(0)
        all_piv.append(piv)
    all_biopsy_factors_mag = pd.concat(all_piv, axis=1)

    all_piv = []
    for cl in liana_sub.source.unique():
        sub = liana_sub[liana_sub.source == cl].drop(columns=["receptor_complex", "target", "ligand_complex", "source", "magnitude_rank"])[["source_target", "ligand_receptor", "specificity_rank"]]
        piv = sub.pivot_table(index="ligand_receptor", columns="source_target", values="specificity_rank", aggfunc="first").fillna(0)
        all_piv.append(piv)
    all_biopsy_factors_spec = pd.concat(all_piv, axis=1)

    liana_sum = all_biopsy_factors_mag * all_biopsy_factors_spec
    liana_sum_melted = liana_sum.melt(ignore_index=False).reset_index()
    liana_sum_melted.columns = ["ligand->receptor", "sender->receiver", "score"]
    liana_sum_melted["ligand"] = liana_sum_melted["ligand->receptor"].astype(str).str.split("->").str[0]
    liana_sum_melted["receptor"] = liana_sum_melted["ligand->receptor"].astype(str).str.split("->").str[1]
    liana_sum_melted["sender"] = liana_sum_melted["sender->receiver"].astype(str).str.split("->").str[0]
    liana_sum_melted["receiver"] = liana_sum_melted["sender->receiver"].astype(str).str.split("->").str[1]

    all_weights_norm = fl / fl.std()
    all_total_scores_ls = []
    for ann in cell_types:
        total_scores_ct = all_weights_norm[all_weights_norm.columns[all_weights_norm.columns.astype(str).str.contains(ann)]].abs().sum(axis=1)
        all_total_scores_ls.append(total_scores_ct)
    all_total_scores = pd.concat(all_total_scores_ls, axis=1)
    all_total_scores.columns = list(cell_types)
    score_lookup = all_total_scores.stack()

    liana_sum_melted["sender_ligand_mofa_score"] = (
        liana_sum_melted.set_index(["ligand", "sender"]).index.map(score_lookup)
    )
    liana_sum_melted["receiver_receptor_mofa_score"] = (
        liana_sum_melted.set_index(["receptor", "receiver"]).index.map(score_lookup)
    )
    liana_sum_melted[["sender_ligand_mofa_score", "receiver_receptor_mofa_score"]] = liana_sum_melted[["sender_ligand_mofa_score", "receiver_receptor_mofa_score"]].fillna(0.0)
    liana_sum_melted["total_mofa_score"] = liana_sum_melted["sender_ligand_mofa_score"] + liana_sum_melted["receiver_receptor_mofa_score"]

    all_scores_z_ls = []
    for send in liana_sum_melted.sender.unique():
        for rec in liana_sum_melted.receiver.unique():
            sub = liana_sum_melted[(liana_sum_melted.sender == send) & (liana_sum_melted.receiver == rec)]
            scores = sub[["score", "total_mofa_score"]]
            scores_z = (scores - scores.mean()) / scores.std()
            all_scores_z_ls.append(scores_z)
    all_scores_z = pd.concat(all_scores_z_ls)
    all_scores_z.columns = ["liana_score_z", "mofa_score_z"]
    liana_sum_melted[["liana_score_z", "mofa_score_z"]] = all_scores_z

    if plot_scores:
        import matplotlib.pyplot as plt
        plt.scatter(liana_sum_melted.liana_score_z, liana_sum_melted.mofa_score_z, s=plot_size)
        plt.axhline(factor_z_threshold, c="red")
        plt.axvline(liana_z_threshold, c="red")
        plt.xlabel("Liana z-score")
        plt.ylabel("Factor z-score")

    filtered = liana_sum_melted[(liana_sum_melted.liana_score_z > liana_z_threshold) & (liana_sum_melted.mofa_score_z > factor_z_threshold)].copy()

    all_signatures_send = {}
    for send in filtered.sender.unique():
        send_ls = []
        for rec in filtered.receiver.unique():
            sub = filtered[(filtered.sender == send) & (filtered.receiver == rec)]
            send_sig = sub[["ligand", "score"]]
            send_final = send_sig.groupby("ligand").sum().score
            sig = pd.DataFrame(send_final)
            sig.columns = [f"rec by|{rec}"]
            send_ls.append(sig)
        all_signatures_send[send] = pd.concat(send_ls, axis=1).fillna(0)

    all_sig_send_ls = []
    for ct in all_signatures_send:
        sig_send = all_signatures_send[ct].copy()
        sig_send.index = ct + "|" + sig_send.index.astype(str)
        all_sig_send_ls.append(sig_send)

    all_signatures_rec = {}
    for rec in filtered.receiver.unique():
        rec_ls = []
        for send in filtered.sender.unique():
            sub = filtered[(filtered.sender == send) & (filtered.receiver == rec)]
            rec_sig = sub[["receptor", "score"]]
            rec_final = rec_sig.groupby("receptor").sum().score
            sig = pd.DataFrame(rec_final)
            sig.columns = [f"sent by|{send}"]
            rec_ls.append(sig)
        all_signatures_rec[rec] = pd.concat(rec_ls, axis=1).fillna(0)

    all_sig_rec_ls = []
    for ct in all_signatures_rec:
        sig_rec = all_signatures_rec[ct].copy()
        sig_rec.index = ct + "|" + sig_rec.index.astype(str)
        all_sig_rec_ls.append(sig_rec)

    all_send_signatures = pd.concat(all_sig_send_ls)
    all_rec_signatures = pd.concat(all_sig_rec_ls)

    parts_rec = []
    receiver_types = all_rec_signatures.index.to_series().str.split("|").str[0].unique()
    for ct in receiver_types:
        sub = all_rec_signatures[all_rec_signatures.index.to_series().str.startswith(ct + "|")].copy()
        sub.index = sub.index.to_series().str.split("|").str[1] 
        sub.columns = [f"{c}|{ct}" for c in sub.columns]          
        parts_rec.append(sub)
    all_rec_signatures_wide = pd.concat(parts_rec, axis=1, join="outer").fillna(0)

    parts_send = []
    receiver_types = all_send_signatures.index.to_series().str.split("|").str[0].unique()
    for ct in receiver_types:
        sub = all_send_signatures[all_send_signatures.index.to_series().str.startswith(ct + "|")].copy()
        sub.index = sub.index.to_series().str.split("|").str[1] 
        sub.columns = [f"{c}|{ct}" for c in sub.columns]          
        parts_send.append(sub)
    all_send_signatures_wide = pd.concat(parts_send, axis=1, join="outer").fillna(0)
    
    return all_send_signatures_wide, all_rec_signatures_wide, filtered, all_biopsy_factors_mag, all_biopsy_factors_spec



def lr_enrichment(
    factor_loadings: pd.DataFrame,
    lr_loadings: pd.DataFrame,
    cell_types: Sequence[str] = DEFAULT_CELL_TYPES,
    cell_type_separator: str = "|",
    direction: str = "ligand",
    n_iter: int = 1000,
    seed: int = 0,
    show_progress: bool = True,
) -> MatrixPair:
    """
    Cell-type-aware LR enrichment using row-paired intermediate signature matrices.

    Expected lr_loadings structure
    ------------------------------
    direction='ligand'
        rows: <sender_cell_type>|<ligand_gene>
        cols: rec by|<receiver_cell_type>

    direction='receptor'
        rows: <receiver_cell_type>|<receptor_gene>
        cols: sent by|<sender_cell_type>

    For each factor lineage, only the corresponding row block is used.
    The final modality keeps the shared 11 paired-context columns.
    """
    fl = ensure_no_nan(require_dataframe(factor_loadings, "factor_loadings"), "factor_loadings")
    lr = ensure_no_nan(require_dataframe(lr_loadings, "lr_loadings"), "lr_loadings")

    if direction not in {"ligand", "receptor"}:
        raise ValueError("direction must be 'ligand' or 'receptor'")

    shared_cols = [f"rec by|{ct}" for ct in cell_types] if direction == "ligand" else [f"sent by|{ct}" for ct in cell_types]

    all_score = []
    all_pval = []

    for ct in cell_types:
        factor_ct = subset_factor_loadings_by_cell_type(fl, ct, separator=cell_type_separator)
        if factor_ct.shape[1] == 0:
            continue

        row_mask = lr.index.astype(str).str.startswith(f"{ct}{cell_type_separator}")
        if row_mask.sum() == 0:
            continue

        lr_ct = lr.loc[row_mask].copy()
        lr_ct.index = lr_ct.index.astype(str).str.split(cell_type_separator, n=1).str[1]
        lr_ct = lr_ct.groupby(level=0).sum()
        lr_ct = lr_ct.reindex(columns=shared_cols).fillna(0.0)

        pair = calc_enrichment(
            factor_ct.T,
            lr_ct,
            n_iter=n_iter,
            seed=seed,
            show_progress=show_progress,
        )
        score_ct = pair.score.reindex(columns=shared_cols)
        pval_ct = pair.pval.reindex(columns=shared_cols)
        all_score.append(score_ct)
        all_pval.append(pval_ct)

    if len(all_score) == 0:
        raise ValueError(f"No {direction} LR enrichments were produced. Check LR row naming and cell types.")

    score_df = pd.concat(all_score, axis=0).reindex(index=fl.columns, columns=shared_cols)
    p_df = pd.concat(all_pval, axis=0).reindex(index=fl.columns, columns=shared_cols)
    return MatrixPair(score=score_df, pval=p_df)



def regulon_enrichment(
    factor_loadings: pd.DataFrame,
    regulon_loadings: pd.DataFrame,
    cell_types: Sequence[str] = DEFAULT_CELL_TYPES,
    cell_type_separator: str = "|",
    cell_type_regulons: bool = True,
    n_iter: int = 1000,
    seed: int = 0,
    show_progress: bool = True,
    progress_message: str = 'Permuting...',
) -> MatrixPair:
    fl = ensure_no_nan(require_dataframe(factor_loadings, "factor_loadings"), "factor_loadings")
    rl = ensure_no_nan(require_dataframe(regulon_loadings, "regulon_loadings"), "regulon_loadings")
    if not cell_type_regulons:
        return calc_enrichment(fl.T, rl, n_iter=n_iter, seed=seed, show_progress=False)

    score_blocks: List[pd.DataFrame] = []
    pval_blocks: List[pd.DataFrame] = []
    for ct in cell_types:
        factor_ct = subset_factor_loadings_by_cell_type(fl, ct, separator=cell_type_separator)
        if factor_ct.shape[1] == 0:
            continue
        reg_cols = [c for c in rl.columns if str(c).endswith(f"{cell_type_separator}{ct}")]
        if len(reg_cols) == 0:
            continue
        reg_ct = rl.loc[:, reg_cols].copy()
        pair = calc_enrichment(factor_ct.T, reg_ct, n_iter=n_iter, seed=seed, show_progress=show_progress, progress_message=progress_message)
        stripped_cols = [str(c).rsplit(f"{cell_type_separator}{ct}", 1)[0] if str(c).endswith(f"{cell_type_separator}{ct}") else str(c) for c in pair.score.columns]
        pair.score.columns = stripped_cols
        pair.pval.columns = stripped_cols
        score_blocks.append(pair.score)
        pval_blocks.append(pair.pval)

    if len(score_blocks) == 0:
        raise ValueError("No regulon enrichments were produced. Check cell type naming and separator.")
    all_reg_names = sorted(set().union(*[df.columns for df in score_blocks]))
    score_df = pd.concat([df.reindex(columns=all_reg_names) for df in score_blocks], axis=0)
    p_df = pd.concat([df.reindex(columns=all_reg_names) for df in pval_blocks], axis=0)
    score_df = score_df.reindex(index=fl.columns, columns=all_reg_names)
    p_df = p_df.reindex(index=fl.columns, columns=all_reg_names)
    return MatrixPair(score=score_df, pval=p_df)

def _split_base_and_cell_type(labels: Sequence[str], separator: str = "|") -> Tuple[pd.Index, pd.Index]:
    """
    Split labels of the form ``base|cell_type`` into base names and suffixes.

    Parameters
    ----------
    labels
        Sequence of labels to split.
    separator
        Separator between the base token and the cell-type suffix.

    Returns
    -------
    tuple of pandas.Index
        Base labels and cell-type suffixes. Labels without ``separator`` receive
        ``None`` as their suffix.
    """
    labels = pd.Index([str(x) for x in labels])
    base = pd.Index(
        [x.rsplit(separator, 1)[0] if separator in x else x for x in labels],
        name=labels.name,
    )
    ct = pd.Index(
        [x.rsplit(separator, 1)[1] if separator in x else None for x in labels],
        name="cell_type",
    )
    return base, ct


def _index_has_cell_type_tags(index: Sequence[str], separator: str = "|") -> bool:
    """
    Return whether any label in an index contains the requested separator.

    Parameters
    ----------
    index
        Sequence of labels to inspect.
    separator
        Separator used to mark cell-type-tagged labels.

    Returns
    -------
    bool
        True if at least one label contains ``separator``.
    """
    idx = pd.Index([str(x) for x in index])
    return bool(idx.str.contains(re.escape(separator), regex=True).any())


def _unique_preserve_order(values: Sequence[str]) -> pd.Index:
    """
    Return unique string values in first-seen order.

    Parameters
    ----------
    values
        Sequence of values to deduplicate.

    Returns
    -------
    pandas.Index
        Unique values preserving original order.
    """
    seen = set()
    out: List[str] = []
    for value in values:
        s = str(value)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return pd.Index(out)


def _collapse_column_cell_type_tagged_matrix(
    df: pd.DataFrame,
    separator: str = "|",
    allowed_cell_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Collapse a gene-by-(feature|cell_type) matrix to gene-by-feature form.

    Rows are plain genes.
    Columns are expected to be labeled like ``<feature>|<cell_type>``.
    Only columns whose cell type is in ``allowed_cell_types`` are retained when
    that argument is provided. After stripping suffixes, duplicate feature names
    are summed.
    """
    require_dataframe(df, "df")
    out = df.copy()
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)

    base_cols = pd.Index(
        [c.rsplit(separator, 1)[0] if separator in c else c for c in out.columns],
        name=out.columns.name,
    )
    col_ct = pd.Index(
        [c.rsplit(separator, 1)[1] if separator in c else None for c in out.columns],
        name="cell_type",
    )

    if allowed_cell_types is not None:
        allowed = pd.Index([str(x) for x in allowed_cell_types])
        keep = col_ct.isin(allowed)
        out = out.loc[:, keep]
        base_cols = base_cols[keep]

    out.columns = base_cols
    out = out.T.groupby(level=0, sort=False).sum().T
    return out


def _collapse_column_cell_type_tagged_matrix_preserve_rows(
    df: pd.DataFrame,
    separator: str = "|",
    allowed_cell_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Collapse a gene-by-(feature|cell_type) matrix to a row-tagged matrix.

    Input
    -----
    rows
        <gene>
    columns
        <feature>|<cell_type>

    Output
    ------
    rows
        <gene>|<cell_type>
    columns
        <feature>

    This preserves cell-type-specific gene loadings while still collapsing the
    feature axis to the generic feature names used by the modality score matrix.
    """
    require_dataframe(df, "df")
    out = df.copy()
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)

    col_base, col_ct = _split_base_and_cell_type(out.columns, separator=separator)
    col_ct_arr = np.asarray(col_ct, dtype=object)
    cts = pd.Index([x for x in col_ct if x is not None]).unique()
    if allowed_cell_types is not None:
        cts = cts.intersection(pd.Index([str(x) for x in allowed_cell_types]))

    if len(cts) == 0:
        raise ValueError(
            "No cell-type-tagged columns were found in the supplied matrix."
        )

    blocks: List[pd.DataFrame] = []
    for ct in cts:
        col_mask = col_ct_arr == ct
        if col_mask.sum() == 0:
            continue

        sub = out.loc[:, col_mask].copy()
        sub.columns = col_base[col_mask]
        if sub.columns.has_duplicates:
            sub = sub.T.groupby(level=0, sort=False).sum().T
        sub.index = pd.Index([f"{g}{separator}{ct}" for g in sub.index], name=out.index.name)
        if sub.index.has_duplicates:
            sub = sub.groupby(level=0, sort=False).sum()
        blocks.append(sub)

    if len(blocks) == 0:
        raise ValueError("No cell-type-aware blocks could be constructed from the supplied matrix.")

    out = pd.concat(blocks, axis=0, join="outer", sort=False).fillna(0.0)
    if out.index.has_duplicates:
        out = out.groupby(level=0, sort=False).sum()
    if out.columns.has_duplicates:
        out = out.T.groupby(level=0, sort=False).sum().T
    return out


def _collapse_cell_type_tagged_matrix(
    df: pd.DataFrame,
    separator: str = "|",
    allowed_cell_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Collapse a fully cell-type-tagged matrix while preserving row cell-type labels.

    Expected structure
    ------------------
    rows
        <gene>|<cell_type>
    columns
        <feature>|<cell_type>

    Only blocks where the row and column cell type match are retained. Within each
    cell type, column suffixes are stripped and duplicate feature names are summed.
    Row suffixes are preserved, so the output retains ``<gene>|<cell_type>`` rows.
    """
    require_dataframe(df, "df")
    df = df.copy()
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    row_base, row_ct = _split_base_and_cell_type(df.index, separator=separator)
    col_base, col_ct = _split_base_and_cell_type(df.columns, separator=separator)

    row_cts = pd.Index([x for x in row_ct if x is not None]).unique()
    col_cts = pd.Index([x for x in col_ct if x is not None]).unique()
    common_cts = row_cts.intersection(col_cts)

    if allowed_cell_types is not None:
        common_cts = common_cts.intersection(pd.Index([str(x) for x in allowed_cell_types]))

    if len(common_cts) == 0:
        raise ValueError(
            "No overlapping cell-type tags were found between the rows and columns "
            "of the supplied cell-type-aware matrix."
        )

    blocks: List[pd.DataFrame] = []
    row_ct_arr = np.asarray(row_ct, dtype=object)
    col_ct_arr = np.asarray(col_ct, dtype=object)

    for ct in common_cts:
        row_mask = row_ct_arr == ct
        col_mask = col_ct_arr == ct
        if row_mask.sum() == 0 or col_mask.sum() == 0:
            continue

        sub = df.iloc[row_mask, col_mask].copy()
        sub.index = pd.Index([f"{base}{separator}{ct}" for base in row_base[row_mask]], name=df.index.name)
        sub.columns = col_base[col_mask]

        if sub.index.has_duplicates:
            sub = sub.groupby(level=0, sort=False).sum()
        if sub.columns.has_duplicates:
            sub = sub.T.groupby(level=0, sort=False).sum().T
        blocks.append(sub)

    if len(blocks) == 0:
        raise ValueError("No cell-type-aware blocks could be constructed from the supplied matrix.")

    out = pd.concat(blocks, axis=0, join="outer", sort=False).fillna(0.0)
    if out.index.has_duplicates:
        out = out.groupby(level=0, sort=False).sum()
    if out.columns.has_duplicates:
        out = out.T.groupby(level=0, sort=False).sum().T
    return out


def _scored_enrichment_cell_type_aware(
    factor_weights: pd.DataFrame,
    factor_meta: pd.DataFrame,
    modality_df: pd.DataFrame,
    n_iter: int = 1000,
    seed: int = 0,
    show_progress: bool = True,
    progress_message: str = "Permuting...",
    cell_type_separator: str = "|",
    collapse_output: bool = True,
) -> Tuple[MatrixPair, pd.DataFrame]:
    """
    Compute cell-type-aware scored enrichment and build storage-ready feature loadings.

    Supported input structures
    --------------------------
    1. rows = <gene>,          columns = <feature>|<cell_type>
    2. rows = <gene>|<cell_type>, columns = <feature>|<cell_type>

    In both cases, enrichment is computed only between ontology factors and
    modality feature columns whose cell-type tags match. When ``collapse_output``
    is True, score columns are collapsed to plain feature names while the
    returned feature-loading matrix preserves row-level cell-type identity as
    ``<gene>|<cell_type>``.
    """
    factor_weights = require_dataframe(factor_weights, "factor_weights").copy()
    modality_df = require_dataframe(modality_df, "modality_df").copy()
    factor_meta = require_dataframe(factor_meta, "factor_meta").copy()

    factor_weights.index = factor_weights.index.astype(str)
    factor_weights.columns = factor_weights.columns.astype(str)
    modality_df.index = modality_df.index.astype(str)
    modality_df.columns = modality_df.columns.astype(str)
    factor_meta.index = factor_meta.index.astype(str)

    col_base, col_ct = _split_base_and_cell_type(modality_df.columns, separator=cell_type_separator)
    col_ct_arr = np.asarray(col_ct, dtype=object)

    row_is_tagged = _index_has_cell_type_tags(modality_df.index, separator=cell_type_separator)
    if row_is_tagged:
        row_base, row_ct = _split_base_and_cell_type(modality_df.index, separator=cell_type_separator)
        row_ct_arr = np.asarray(row_ct, dtype=object)
        row_cts = pd.Index([x for x in row_ct if x is not None]).unique()
    else:
        row_base = pd.Index(modality_df.index.astype(str), name=modality_df.index.name)
        row_ct = pd.Index([None] * modality_df.shape[0], name="cell_type")
        row_ct_arr = np.asarray(row_ct, dtype=object)
        row_cts = pd.Index([], dtype=object)

    factor_cts = pd.Index(factor_meta["Classification"].dropna().astype(str).unique())
    col_cts = pd.Index([x for x in col_ct if x is not None]).unique()
    common_cts = factor_cts.intersection(col_cts)
    if row_is_tagged:
        common_cts = common_cts.intersection(row_cts)

    if len(common_cts) == 0:
        raise ValueError(
            "No overlapping cell types were found across ontology factor metadata "
            "and the supplied cell-type-aware modality labels."
        )

    factor_classes = factor_meta["Classification"].astype(str)
    score_blocks: List[pd.DataFrame] = []
    pval_blocks: List[pd.DataFrame] = []
    feature_blocks: List[pd.DataFrame] = []

    for ct in common_cts:
        factor_idx = factor_meta.index[factor_classes == ct]
        if len(factor_idx) == 0:
            continue

        col_mask = col_ct_arr == ct
        if col_mask.sum() == 0:
            continue

        if row_is_tagged:
            row_mask = row_ct_arr == ct
            if row_mask.sum() == 0:
                continue
            sub_raw = modality_df.iloc[row_mask, col_mask].copy()
            sig = sub_raw.copy()
            sig.index = row_base[row_mask]
        else:
            sub_raw = modality_df.loc[:, col_mask].copy()
            sig = sub_raw.copy()

        sig.columns = col_base[col_mask]
        if sig.index.has_duplicates:
            sig = sig.groupby(level=0, sort=False).sum()
        if sig.columns.has_duplicates:
            sig = sig.T.groupby(level=0, sort=False).sum().T

        pair = calc_enrichment(
            samples_by_genes=factor_weights.loc[factor_idx],
            signatures=sig,
            n_iter=n_iter,
            seed=seed,
            show_progress=show_progress,
            progress_message=f"{progress_message} [{ct}]",
        )

        if collapse_output:
            pair.score.columns = pair.score.columns.astype(str)
            pair.pval.columns = pair.pval.columns.astype(str)
        else:
            pair.score.columns = [f"{c}{cell_type_separator}{ct}" for c in pair.score.columns]
            pair.pval.columns = [f"{c}{cell_type_separator}{ct}" for c in pair.pval.columns]

        score_blocks.append(pair.score)
        pval_blocks.append(pair.pval)

        if collapse_output:
            if row_is_tagged:
                feat = _collapse_cell_type_tagged_matrix(
                    sub_raw,
                    separator=cell_type_separator,
                    allowed_cell_types=[ct],
                )
            else:
                feat = _collapse_column_cell_type_tagged_matrix_preserve_rows(
                    sub_raw,
                    separator=cell_type_separator,
                    allowed_cell_types=[ct],
                )
        else:
            feat = sub_raw.copy()
            if row_is_tagged:
                feat.index = pd.Index([f"{g}{cell_type_separator}{ct}" for g in row_base[row_mask]], name=feat.index.name)
            else:
                feat.index = pd.Index([f"{g}{cell_type_separator}{ct}" for g in feat.index], name=feat.index.name)
            feat.columns = pd.Index([f"{c}{cell_type_separator}{ct}" for c in col_base[col_mask]], name=feat.columns.name)
            if feat.index.has_duplicates:
                feat = feat.groupby(level=0, sort=False).sum()
            if feat.columns.has_duplicates:
                feat = feat.T.groupby(level=0, sort=False).sum().T

        feature_blocks.append(feat)

    if len(score_blocks) == 0:
        raise ValueError("No scored cell-type-aware enrichments were produced.")

    all_features = _unique_preserve_order([c for df in score_blocks for c in df.columns])
    score_df = pd.concat([df.reindex(columns=all_features) for df in score_blocks], axis=0)
    pval_df = pd.concat([df.reindex(columns=all_features) for df in pval_blocks], axis=0)

    score_df = score_df.reindex(index=factor_weights.index, columns=all_features)
    pval_df = pval_df.reindex(index=factor_weights.index, columns=all_features)

    feature_loadings = pd.concat(feature_blocks, axis=0, join="outer", sort=False).fillna(0.0)
    if feature_loadings.index.has_duplicates:
        feature_loadings = feature_loadings.groupby(level=0, sort=False).sum()
    if feature_loadings.columns.has_duplicates:
        feature_loadings = feature_loadings.T.groupby(level=0, sort=False).sum().T
    feature_loadings = feature_loadings.reindex(columns=score_df.columns).fillna(0.0)

    return MatrixPair(score=score_df, pval=pval_df), feature_loadings


def add_modality(
    ontology: mu.MuData,
    modality_name: str,
    score_df: Optional[pd.DataFrame] = None,
    pval_df: Optional[pd.DataFrame] = None,
    modality_data: Optional[Union[pd.DataFrame, Mapping[str, Sequence[str]], str]] = None,
    modality_type: str = "auto",
    feature_loadings: Optional[pd.DataFrame] = None,
    feature_loading_key: str = "feature_loadings",
    modality_uns: Optional[Mapping[str, Any]] = None,
    n_iter: int = 1000,
    seed: int = 0,
    show_progress: bool = True,
    permutation_kwargs: Optional[Mapping[str, Any]] = None,
    gsea_kwargs: Optional[Mapping[str, Any]] = None,
    store_input_as_feature_loadings: bool = True,
    scored_cell_type_aware: bool = False,
    cell_type_separator: str = "|",
    collapse_cell_type_aware_scores: bool = True,
    inplace: bool = True,
) -> Optional[mu.MuData]:
    """
    Add a modality to an existing ontology object.

    This function supports precomputed factor-by-feature score matrices,
    scored gene-by-feature priors that require permutation enrichment, and
    gene-set modalities scored by preranked GSEA. For cell-type-aware scored
    modalities, columns are expected to carry a suffix after
    ``cell_type_separator`` and are matched to ontology factor lineages.

    Returns
    -------
    None or muon.MuData
        ``None`` when ``inplace=True``; otherwise an updated ontology copy.
    """
    permutation_kwargs = dict(permutation_kwargs or {})
    gsea_kwargs = dict(gsea_kwargs or {})
    modality_uns = dict(modality_uns or {})

    if modality_type not in {"auto", "precomputed", "scored", "gene_set"}:
        raise ValueError("modality_type must be one of {'auto', 'precomputed', 'scored', 'gene_set'}.")

    if ontology.obs is None or ontology.obs.shape[0] == 0:
        raise ValueError("ontology.obs is empty. Cannot align modality rows to ontology factors.")
    if "weights" not in ontology.obsm:
        raise KeyError("ontology.obsm['weights'] not found.")
    if "gene_names" not in ontology.uns:
        raise KeyError("ontology.uns['gene_names'] not found.")

    factor_index = pd.Index(ontology.obs.index.astype(str), name="factor")
    factor_meta = ontology.obs.reindex(factor_index).copy()
    factor_weights = factor_weights_to_df(ontology).reindex(factor_index)

    inferred_type = modality_type
    if modality_type == "auto":
        if score_df is not None:
            inferred_type = "precomputed"
        elif isinstance(modality_data, pd.DataFrame):
            inferred_type = "scored"
        elif isinstance(modality_data, Mapping) or isinstance(modality_data, str):
            inferred_type = "gene_set"
        else:
            raise ValueError(
                "Could not infer modality_type. Provide either `score_df`, a DataFrame `modality_data`, or a gene-set mapping/string."
            )

    if inferred_type == "precomputed":
        if score_df is None:
            raise ValueError("modality_type='precomputed' requires `score_df`.")
    else:
        if score_df is not None:
            raise ValueError(
                "`score_df` was provided together with enrichment mode. Use either precomputed mode (`score_df`) or raw `modality_data`, not both."
            )
        if modality_data is None:
            raise ValueError(f"modality_type='{inferred_type}' requires `modality_data`.")

    if inferred_type == "precomputed":
        require_dataframe(score_df, "score_df")
        score_df = score_df.copy()
        score_df.index = score_df.index.astype(str)
        score_df.columns = score_df.columns.astype(str)
        if pval_df is not None:
            require_dataframe(pval_df, "pval_df")
            pval_df = pval_df.copy()
            pval_df.index = pval_df.index.astype(str)
            pval_df.columns = pval_df.columns.astype(str)

    elif inferred_type == "scored":
        require_dataframe(modality_data, "modality_data")
        modality_df = ensure_no_nan(modality_data.copy(), "modality_data")
        modality_df.index = modality_df.index.astype(str)
        modality_df.columns = modality_df.columns.astype(str)

        perm_kwargs = {
            "n_iter": n_iter,
            "seed": seed,
            "show_progress": show_progress,
        }
        perm_kwargs.update(permutation_kwargs)
        perm_progress_message = perm_kwargs.pop("progress_message", f"Permuting for '{modality_name}'...")

        inferred_feature_loadings: Optional[pd.DataFrame] = None
        if scored_cell_type_aware:
            pair, inferred_feature_loadings = _scored_enrichment_cell_type_aware(
                factor_weights=factor_weights,
                factor_meta=factor_meta,
                modality_df=modality_df,
                progress_message=perm_progress_message,
                cell_type_separator=cell_type_separator,
                collapse_output=collapse_cell_type_aware_scores,
                **perm_kwargs,
            )
            score_df = pair.score
            pval_df = pair.pval
        else:
            pair = calc_enrichment(
                samples_by_genes=factor_weights,
                signatures=modality_df,
                progress_message=perm_progress_message,
                **perm_kwargs,
            )
            score_df = pair.score
            pval_df = pair.pval
            if store_input_as_feature_loadings:
                inferred_feature_loadings = modality_df.copy()

        if feature_loadings is None and store_input_as_feature_loadings:
            feature_loadings = inferred_feature_loadings

    elif inferred_type == "gene_set":
        if not (isinstance(modality_data, Mapping) or isinstance(modality_data, str)):
            raise TypeError(
                "For gene_set mode, `modality_data` must be either a mapping {set_name: genes} or a string gene-set library name."
            )

        ggsea_kwargs = {
            "permutation_num": n_iter,
            "seed": seed,
            "show_progress": show_progress,
            "progress_message": f"Calculating GSEA for '{modality_name}'...",
        }
        ggsea_kwargs.update(gsea_kwargs)
        pair = gsea_enrichment(
            factor_loadings=factor_weights.T,
            gene_sets=modality_data,
            **ggsea_kwargs,
        )
        score_df = pair.score
        pval_df = pair.pval

    extra_factors = score_df.index.difference(factor_index)
    if len(extra_factors) > 0:
        raise ValueError(
            f"score_df contains factors not present in ontology.obs: {list(extra_factors[:10])}" + (" ..." if len(extra_factors) > 10 else "")
        )

    score_df = score_df.reindex(factor_index)
    score_df.columns = score_df.columns.astype(str)

    if pval_df is not None:
        extra_pval_factors = pval_df.index.difference(factor_index)
        if len(extra_pval_factors) > 0:
            raise ValueError(
                f"pval_df contains factors not present in ontology.obs: {list(extra_pval_factors[:10])}" + (" ..." if len(extra_pval_factors) > 10 else "")
            )
        pval_df = pval_df.reindex(index=factor_index, columns=score_df.columns)
        pval_df.columns = pval_df.columns.astype(str)

    if feature_loadings is not None:
        require_dataframe(feature_loadings, "feature_loadings")
        feature_loadings = feature_loadings.copy()
        feature_loadings.index = feature_loadings.index.astype(str)
        feature_loadings.columns = feature_loadings.columns.astype(str)

        if inferred_type == "scored" and scored_cell_type_aware and collapse_cell_type_aware_scores:
            if _index_has_cell_type_tags(feature_loadings.columns, separator=cell_type_separator):
                if _index_has_cell_type_tags(feature_loadings.index, separator=cell_type_separator):
                    feature_loadings = _collapse_cell_type_tagged_matrix(
                        feature_loadings,
                        separator=cell_type_separator,
                        allowed_cell_types=factor_meta["Classification"].dropna().astype(str).unique(),
                    )
                else:
                    feature_loadings = _collapse_column_cell_type_tagged_matrix_preserve_rows(
                        feature_loadings,
                        separator=cell_type_separator,
                        allowed_cell_types=factor_meta["Classification"].dropna().astype(str).unique(),
                    )

        feature_loadings = feature_loadings.reindex(columns=score_df.columns).fillna(0.0)

    mod_adata = build_modality_adata(
        score_df=score_df,
        pval_df=pval_df,
        feature_loadings=feature_loadings,
        factor_metadata=ontology.obs.reindex(factor_index).copy(),
        feature_loading_key=feature_loading_key,
        cell_type_separator=cell_type_separator,
    )

    mod_adata.uns["modality_type"] = inferred_type
    mod_adata.uns["feature_loading_key_default"] = feature_loading_key
    mod_adata.uns["scored_cell_type_aware"] = bool(inferred_type == "scored" and scored_cell_type_aware)
    mod_adata.uns["cell_type_separator"] = cell_type_separator
    mod_adata.uns["collapse_cell_type_aware_scores"] = bool(
        inferred_type == "scored" and scored_cell_type_aware and collapse_cell_type_aware_scores
    )
    if feature_loadings is not None:
        mod_adata.uns["feature_loading_row_axis"] = (
            "gene|cell_type" if _index_has_cell_type_tags(feature_loadings.index, separator=cell_type_separator) else "gene"
        )

    if inferred_type == "gene_set":
        if isinstance(modality_data, Mapping):
            mod_adata.uns["gene_sets"] = {
                str(k): [str(g) for g in v]
                for k, v in modality_data.items()
            }
        elif isinstance(modality_data, str):
            mod_adata.uns["gene_set_library"] = str(modality_data)

    for k, v in modality_uns.items():
        mod_adata.uns[k] = v

    target = ontology if inplace else ontology.copy()
    target.mod[modality_name] = mod_adata

    if "modality_names" not in target.uns:
        target.uns["modality_names"] = []
    target.uns["modality_names"] = sorted(set(list(target.uns["modality_names"]) + [modality_name]))

    if inplace:
        return None
    return target


def make_ontology(
    factor_loadings: pd.DataFrame,
    factor_type: str = "MOFA",
    factor_metadata: Optional[pd.DataFrame] = None,
    cell_type_separator: str = "|",
    scored_modalities: Optional[Mapping[str, pd.DataFrame]] = None,
    scored_modalities_ct_aware: Optional[Mapping[str, pd.DataFrame]] = None,
    gene_set_modalities: Optional[Mapping[str, Union[Mapping[str, Sequence[str]], str]]] = None,
    scored_feature_loadings: Optional[Mapping[str, Optional[pd.DataFrame]]] = None,
    feature_loading_key_map: Optional[Mapping[str, str]] = None,
    scored_modality_uns: Optional[Mapping[str, Mapping[str, Any]]] = None,
    gene_set_modality_uns: Optional[Mapping[str, Mapping[str, Any]]] = None,
    n_iter: int = 1000,
    seed: int = 0,
    show_progress = True,
) -> mu.MuData:
    """
    Construct a factor-centric ontology object from factor loadings and a set of
    generic annotation modalities.

    This function initializes ontology-wide factor metadata and global factor
    weights from a gene-by-factor loading matrix, computes enrichment for each
    supplied modality, stores each modality as an ``AnnData`` object in
    ``ontology.mod``, and returns the final ontology as a ``MuData`` object.

    Two modality classes are supported. Scored modalities are weighted
    gene-by-feature matrices and are processed with permutation-based
    enrichment using :func:`calc_enrichment`. Gene-set modalities are unranked
    gene-set collections or gene-set library names and are processed with
    :func:`gsea_enrichment`.

    Scored modalities added through :func:`make_ontology` are handled in a
    cell-type-agnostic manner by default. That is, each feature column is
    evaluated against every ontology factor using the shared gene space,
    without matching row or column labels by lineage. For cell-type-aware
    scored enrichment, modalities should instead be added after ontology
    construction using :func:`add_modality` with
    ``scored_cell_type_aware=True``.

    The resulting ontology stores factor metadata in ``ontology.obs``, global
    factor weights in ``ontology.obsm["weights"]``, the corresponding gene
    names in ``ontology.uns["gene_names"]``, and each annotation modality as
    an ``AnnData`` object in ``ontology.mod``.

    Parameters
    ----------
    factor_loadings
        Gene-by-factor loading matrix. Rows must correspond to genes and
        columns must correspond to ontology factors.
    factor_type
        Label describing the origin or type of the factor loadings. This value
        is stored in ``ontology.uns["factor_type"]`` and also added to factor
        metadata as the ``FactorType`` column.
    factor_metadata
        Optional factor metadata DataFrame indexed by factor name. If not
        provided, metadata are inferred from factor names using
        :func:`safe_factor_metadata`. If provided but incomplete, missing
        metadata values are filled from the inferred defaults where possible.
    cell_type_separator
        Separator used when parsing factor names for inferred metadata.
    scored_modalities
        Optional mapping from modality name to a scored modality matrix. Each
        value must be a DataFrame with rows corresponding to genes and columns
        corresponding to modality features or signatures. Each scored modality
        is processed in a cell-type-agnostic fashion by default.
    scored_modalities_ct_aware
        Optional mapping from modality name to a scored modality matrix in cell 
        type-aware mode. Each value must be a DataFrame with rows corresponding 
        to genes and columns corresponding to modality features or signatures, 
        with cell type as a suffix matching factor name cell type labels. 
    gene_set_modalities
        Optional mapping from modality name to an unranked gene-set collection.
        Each value may be either a mapping of gene-set names to gene lists or a
        string naming a gene-set library supported by ``gseapy``.
    scored_feature_loadings
        Optional mapping from scored modality name to an explicit
        gene-by-feature loading matrix to store in the modality ``varm`` slot.
        If omitted for a scored modality, the scored modality input matrix
        itself may be used as the default feature-loading representation.
    feature_loading_key_map
        Optional mapping from modality name to the key used when storing that
        modality's feature loadings in ``varm``.
    scored_modality_uns
        Optional mapping from scored modality name to a metadata dictionary to
        store in the modality ``uns`` slot.
    gene_set_modality_uns
        Optional mapping from gene-set modality name to a metadata dictionary
        to store in the modality ``uns`` slot.
    n_iter
        Default number of permutations for scored modalities and default
        ``permutation_num`` passed to GSEA unless overridden downstream.
    seed
        Default random seed passed to modality enrichment routines.
    show_progress
        Whether to display progress output during modality enrichment.

    Returns
    -------
    muon.MuData
        Newly constructed ontology object containing factor metadata, global
        factor weights, and all requested modalities.

    Raises
    ------
    ValueError
        Raised if factor loadings are invalid or if no modalities are supplied
        in environments where an empty ``MuData`` object cannot be constructed.

    Notes
    -----
    This function no longer performs modality-specific preprocessing for
    ligand-receptor, regulon, or other specialized inputs. Such preprocessing
    should be performed upstream. For example, LIANA outputs can be converted
    into ligand and receptor signature matrices using
    :func:`make_filtered_lr_signatures`, and the resulting matrices can then be
    passed through ``scored_modalities``. Additional scored or gene-set
    modalities can be appended after ontology construction using
    :func:`add_modality`.
    """
    fl = ensure_no_nan(require_dataframe(factor_loadings, "factor_loadings"), "factor_loadings").copy()
    fl.index = fl.index.astype(str)
    fl.columns = fl.columns.astype(str)

    if factor_metadata is None:
        factor_meta = safe_factor_metadata(fl.columns, separator=cell_type_separator)
    else:
        factor_meta = require_dataframe(factor_metadata, "factor_metadata").copy()
        factor_meta.index = factor_meta.index.astype(str)

        default_meta = safe_factor_metadata(fl.columns, separator=cell_type_separator)
        factor_meta = factor_meta.reindex(default_meta.index)

        for col in default_meta.columns:
            if col not in factor_meta.columns:
                factor_meta[col] = default_meta[col]
            else:
                factor_meta[col] = factor_meta[col].where(
                    factor_meta[col].notna(),
                    default_meta[col],
                )

    factor_meta["FactorType"] = factor_type

    scored_feature_loadings = dict(scored_feature_loadings or {})
    feature_loading_key_map = dict(feature_loading_key_map or {})
    scored_modality_uns = dict(scored_modality_uns or {})
    gene_set_modality_uns = dict(gene_set_modality_uns or {})

    modalities: Dict[str, ad.AnnData] = {}
    modalities['weights'] = ad.AnnData(fl.T.copy())
    # -----------------------------
    # Scored modalities
    # -----------------------------
    for modality_name, modality_df in dict(scored_modalities or {}).items():
        modality_df = ensure_no_nan(
            require_dataframe(modality_df, f"scored_modalities['{modality_name}']"),
            f"scored_modalities['{modality_name}']",
        ).copy()
        modality_df.index = modality_df.index.astype(str)
        modality_df.columns = modality_df.columns.astype(str)

        pair = calc_enrichment(
            samples_by_genes=fl.T,
            signatures=modality_df,
            n_iter=n_iter,
            seed=seed,
            show_progress=show_progress,
            progress_message=f"Permuting for '{modality_name}'...",
        )

        feat = scored_feature_loadings.get(modality_name)
        if feat is None:
            feat = modality_df
        else:
            feat = ensure_no_nan(
                require_dataframe(feat, f"scored_feature_loadings['{modality_name}']"),
                f"scored_feature_loadings['{modality_name}']",
            ).copy()
            feat.index = feat.index.astype(str)
            feat.columns = feat.columns.astype(str)
            feat = feat.reindex(columns=pair.score.columns).fillna(0.0)

        mod_adata = build_modality_adata(
            score_df=pair.score,
            pval_df=pair.pval,
            feature_loadings=feat,
            factor_metadata=factor_meta,
            feature_loading_key=feature_loading_key_map.get(modality_name, "feature_loadings"),
            cell_type_separator=cell_type_separator,
        )
        mod_adata.uns["modality_type"] = "scored"

        for k, v in scored_modality_uns.get(modality_name, {}).items():
            mod_adata.uns[k] = v

        modalities[modality_name] = mod_adata

    # -----------------------------
    # Gene-set modalities
    # -----------------------------
    for modality_name, gene_sets in dict(gene_set_modalities or {}).items():
        pair = gsea_enrichment(
            factor_loadings=fl,
            gene_sets=gene_sets,
            permutation_num=n_iter,
            seed=seed,
            show_progress=show_progress,
            progress_message=f"Calculating GSEA for '{modality_name}'...",
        )

        mod_adata = build_modality_adata(
            score_df=pair.score,
            pval_df=pair.pval,
            feature_loadings=None,
            factor_metadata=factor_meta,
            feature_loading_key=feature_loading_key_map.get(modality_name, "feature_loadings"),
            cell_type_separator=cell_type_separator,
        )
        mod_adata.uns["modality_type"] = "gene_set"

        if isinstance(gene_sets, Mapping):
            mod_adata.uns["gene_sets"] = {
                str(k): [str(g) for g in v]
                for k, v in gene_sets.items()
            }
        else:
            mod_adata.uns["gene_set_library"] = str(gene_sets)

        for k, v in gene_set_modality_uns.get(modality_name, {}).items():
            mod_adata.uns[k] = v

        modalities[modality_name] = mod_adata

    if len(modalities) == 0:
        raise ValueError(
            "make_ontology requires at least one modality in `scored_modalities` "
            "or `gene_set_modalities` for this mudata version."
        )

    ontology = mu.MuData(modalities)
    ontology.obs = factor_meta.copy()
    ontology.obsm["weights"] = as_csr(fl.T.to_numpy(dtype=np.float32))
    ontology.uns["gene_names"] = list(fl.index.astype(str))
    ontology.uns["factor_type"] = factor_type
    ontology.uns["modality_names"] = list(modalities.keys())

    for key, value in dict(scored_modalities_ct_aware or {}).items():
        add_modality(
            ontology,
            modality_name=key,
            modality_data=value,
            modality_type="scored",
            scored_cell_type_aware=True,
            cell_type_separator=cell_type_separator,
            n_iter=n_iter,
            seed=seed,
            show_progress=show_progress,
        )
    
    return ontology


def _normalize_cell_types(cell_types: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    """
    Normalize an optional cell-type selector to a list of strings.

    Parameters
    ----------
    cell_types
        Single cell type, sequence of cell types, or ``None``.

    Returns
    -------
    list of str or None
        Normalized cell-type list, or ``None`` if no filtering was requested.
    """
    if cell_types is None:
        return None
    if isinstance(cell_types, str):
        return [cell_types]
    return list(cell_types)


def _filter_factor_index_by_cell_types(index: pd.Index, ontology: mu.MuData, cell_types: Optional[Union[str, Sequence[str]]]) -> pd.Index:
    """
    Restrict a factor index to selected ontology classifications.

    Parameters
    ----------
    index
        Factor index to filter.
    ontology
        Ontology object containing factor metadata in ``ontology.obs``.
    cell_types
        Optional lineage selector.

    Returns
    -------
    pandas.Index
        Filtered factor index.
    """
    ct_list = _normalize_cell_types(cell_types)
    if ct_list is None:
        return index
    meta = ontology.obs.reindex(index)
    keep = meta["Classification"].astype(str).isin(ct_list)
    return index[keep.values]


def modality_scores_to_df(ontology: mu.MuData, modality: str, cell_types: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
    """Return a modality score matrix as a DataFrame.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    modality : str
        Name of the modality to extract.
    cell_types : str or sequence of str, optional
        Restrict returned factor rows to selected ontology lineages.

    Returns
    -------
    pandas.DataFrame
        Factor-by-feature score matrix for the requested modality.
    """
    mod = ontology.mod[modality]
    df = pd.DataFrame(as_dense(mod.X), index=mod.obs_names, columns=mod.var_names)
    keep = _filter_factor_index_by_cell_types(df.index, ontology, cell_types)
    return df.loc[keep].copy()


def modality_pvals_to_df(ontology: mu.MuData, modality: str, cell_types: Optional[Union[str, Sequence[str]]] = None) -> Optional[pd.DataFrame]:
    """Return modality p-values as a DataFrame.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    modality : str
        Name of the modality to extract.
    cell_types : str or sequence of str, optional
        Restrict returned factor rows to selected ontology lineages.

    Returns
    -------
    pandas.DataFrame or None
        Factor-by-feature p-value matrix if available, otherwise ``None``.
    """
    mod = ontology.mod[modality]
    if "pval" not in mod.layers:
        return None
    df = pd.DataFrame(as_dense(mod.layers["pval"]), index=mod.obs_names, columns=mod.var_names)
    keep = _filter_factor_index_by_cell_types(df.index, ontology, cell_types)
    return df.loc[keep].copy()


def factor_weights_to_df(ontology: mu.MuData, transpose: bool = False, cell_types: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
    """Return global factor weights as a pandas DataFrame.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    cell_types : str or sequence of str, optional
        Restrict returned factors to selected ontology lineages.
    transpose : bool, default=False
        If ``False``, return ``factor x gene``. If ``True``, return ``gene x factor``.

    Returns
    -------
    pandas.DataFrame
        DataFrame representation of ``ontology.obsm['weights']``.
    """
    if "weights" not in ontology.obsm:
        raise KeyError("ontology.obsm['weights'] not found.")
    if "gene_names" not in ontology.uns:
        raise KeyError("ontology.uns['gene_names'] not found.")

    weights = pd.DataFrame(
        as_dense(ontology.obsm["weights"]),
        index=ontology.obs_names,
        columns=pd.Index(list(map(str, ontology.uns["gene_names"])), name="gene"),
    )
    weights.index.name = "factor"
    keep = _filter_factor_index_by_cell_types(weights.index, ontology, cell_types)
    weights = weights.loc[keep].copy()
    return weights.T if transpose else weights


def modality_feature_loadings_to_df(
    ontology: mu.MuData,
    modality: str,
    key: Optional[str] = None,
    cell_types: Optional[Union[str, Sequence[str]]] = None,
    strip_cell_type_suffix: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Reconstruct a modality feature-loading matrix from stored ``.varm`` values.

    Parameters
    ----------
    ontology
        Ontology object.
    modality
        Name of the modality to extract.
    key
        Specific ``varm`` key to use. If omitted, the default key stored in
        ``uns["feature_loading_key_default"]`` is preferred when available; if
        there is exactly one ``varm`` matrix, that key is used automatically.
    cell_types
        Optional cell type or list of cell types used to filter row-tagged
        loading matrices.
    strip_cell_type_suffix
        Whether to strip the ``|cell_type`` suffix from returned row names when
        a single cell type is requested. If omitted, suffix stripping is applied
        automatically only when exactly one cell type is requested.

    Returns
    -------
    pandas.DataFrame
        Loading matrix with rows equal to loading-row labels and columns equal
        to modality features.

    Notes
    -----
    This helper supports both legacy per-cell-type ``varm`` storage and the
    newer single-key storage scheme in which loading rows themselves can be
    tagged as ``gene|cell_type``.
    """
    mod = ontology.mod[modality]
    varm_keys = list(mod.varm.keys())
    if len(varm_keys) == 0:
        raise KeyError(f"No varm matrices found for modality '{modality}'.")

    default_key = mod.uns.get("feature_loading_key_default")
    if key is None and default_key in varm_keys:
        key = str(default_key)
    if key is None and len(varm_keys) == 1:
        key = varm_keys[0]

    ct_list = _normalize_cell_types(cell_types)
    separator = str(mod.uns.get("cell_type_separator", "|"))

    def _matrix_to_df(varm_key: str) -> pd.DataFrame:
        gene_names = mod.uns.get("gene_names", ontology.uns.get("gene_names"))
        if gene_names is None:
            raise KeyError(f"No gene names stored for modality '{modality}'.")
        arr = as_dense(mod.varm[varm_key])
        return pd.DataFrame(
            arr.T,
            index=pd.Index(list(map(str, gene_names)), name="gene"),
            columns=mod.var_names,
        )

    if key is not None:
        if key not in varm_keys:
            raise KeyError(f"varm key '{key}' not found for modality '{modality}'.")
        df = _matrix_to_df(key)
    else:
        selected_keys: List[str] = []
        if ct_list is None:
            selected_keys = list(varm_keys)
        else:
            for ct in ct_list:
                selected_keys.extend([k for k in varm_keys if str(k).startswith(f"{ct}_")])
            if len(selected_keys) == 0:
                selected_keys = list(varm_keys)

        selected_keys = list(_unique_preserve_order(selected_keys))
        if len(selected_keys) == 1:
            df = _matrix_to_df(selected_keys[0])
        else:
            dfs = []
            for varm_key in selected_keys:
                part = _matrix_to_df(varm_key)
                suffix = str(varm_key).split("_", 1)[0] if "_" in str(varm_key) else str(varm_key)
                part.columns = [f"{c}|{suffix}" for c in part.columns]
                dfs.append(part)
            df = pd.concat(dfs, axis=1)

    row_axis = str(mod.uns.get("feature_loading_row_axis", "gene"))
    row_is_tagged = row_axis == "gene|cell_type" or _index_has_cell_type_tags(df.index, separator=separator)

    if ct_list is not None and row_is_tagged:
        row_base, row_ct = _split_base_and_cell_type(df.index, separator=separator)
        mask = pd.Index([str(x) if x is not None else None for x in row_ct]).isin(ct_list)
        df = df.loc[mask].copy()
        if strip_cell_type_suffix is None:
            strip_cell_type_suffix = len(ct_list) == 1
        if strip_cell_type_suffix and len(ct_list) == 1:
            df.index = row_base[mask]
            if df.index.has_duplicates:
                df = df.groupby(level=0, sort=False).sum()

    return df


def get_factor_scores(
    ontology: mu.MuData,
    factors: Optional[Sequence[str]] = None,
    modalities: Optional[Sequence[str]] = None,
    cell_types: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Collect score matrices from one or more ontology modalities.

    Parameters
    ----------
    ontology
        Ontology object.
    factors
        Optional subset of factor names to retain.
    modalities
        Optional subset of modality names to query. If omitted, all modalities
        are returned.
    cell_types
        Optional lineage selector applied to factor rows.

    Returns
    -------
    dict
        Mapping from modality name to factor-by-feature score DataFrame.
    """
    modalities = list(ontology.mod.keys()) if modalities is None else list(modalities)
    out: Dict[str, pd.DataFrame] = {}
    for modality in modalities:
        df = modality_scores_to_df(ontology, modality, cell_types=cell_types)
        if factors is not None:
            df = df.loc[list(factors)]
        out[modality] = df
    return out


def query_gene_set(ontology: mu.MuData, gene_set: Sequence[str], modalities: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Subset modality loading matrices to a supplied gene set.

    Parameters
    ----------
    ontology
        Ontology object.
    gene_set
        Sequence of genes to retrieve.
    modalities
        Optional subset of modalities to query. If omitted, all modalities are
        searched.

    Returns
    -------
    dict
        Mapping from modality name to the subset of its loading matrix
        overlapping the requested genes.
    """
    modalities = list(ontology.mod.keys()) if modalities is None else list(modalities)
    gene_set = list(dict.fromkeys(map(str, gene_set)))
    out: Dict[str, pd.DataFrame] = {}
    for modality in modalities:
        loadings = modality_feature_loadings_to_df(ontology, modality)
        common = [g for g in gene_set if g in loadings.index]
        out[modality] = loadings.loc[common].copy()
    return out


def top_features_for_factor(ontology: mu.MuData, factor: str, modality: str, n_pos: int = 10, n_neg: int = 10, alpha: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return top positive and negative features for a factor.

    Parameters
    ----------
    ontology
        Factor-centric ontology MuData.
    factor
        Factor name, e.g. ``Factor0|Tumor``.
    modality
        Ontology modality name, or ``"weights"`` to query the global factor
        weight matrix stored in ``ontology.obsm['weights']``.
    n_pos, n_neg
        Number of positive / negative features to return.
    alpha
        Optional p-value threshold. Only applies to ontology modalities that
        store p-values; ignored for ``modality='weights'``.
    """
    if modality == "weights":
        if "weights" not in ontology.obsm:
            raise KeyError("ontology.obsm['weights'] not found.")
        if "gene_names" not in ontology.uns:
            raise KeyError("ontology.uns['gene_names'] not found.")
        weights = pd.DataFrame(
            as_dense(ontology.obsm["weights"]),
            index=ontology.obs_names,
            columns=pd.Index(ontology.uns["gene_names"], name="gene"),
        )
        if factor not in weights.index:
            raise KeyError(f"Factor '{factor}' not found in ontology weights.")
        row = pd.DataFrame({"feature": weights.columns, "score": weights.loc[factor].values})
        row["pval"] = np.nan
    else:
        scores = modality_scores_to_df(ontology, modality)
        if factor not in scores.index:
            raise KeyError(f"Factor '{factor}' not found in modality '{modality}'.")
        pvals = modality_pvals_to_df(ontology, modality)
        row = pd.DataFrame({"feature": scores.columns, "score": scores.loc[factor].values})
        row["pval"] = pvals.loc[factor].values if pvals is not None else np.nan
        if alpha is not None and row["pval"].notna().any():
            row = row.loc[row["pval"] < alpha].copy()

    pos = row.sort_values("score", ascending=False).head(n_pos).reset_index(drop=True)
    neg = row.sort_values("score", ascending=True).head(n_neg).reset_index(drop=True)
    return pos, neg