
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
    """
    Container for enrichment outputs.

    Attributes
    ----------
    score : pandas.DataFrame
        Primary score matrix, typically a factor-by-feature matrix of
        z-scores, normalized enrichment scores, or other signed effect sizes.
    pval : pandas.DataFrame
        Matrix of p-values aligned exactly to ``score``.
    """
    score: pd.DataFrame
    pval: pd.DataFrame


def require_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Validate that an input is a pandas DataFrame with unique row and column labels.

    This helper is used throughout the module to fail early when callers pass
    ``None``, non-DataFrame objects, or DataFrames whose index or columns are
    duplicated. It returns the original object unchanged so it can be used
    inline during argument normalization.

    Parameters
    ----------
    df : pandas.DataFrame
        Object to validate.
    name : str
        Human-readable argument name used in error messages.

    Returns
    -------
    pandas.DataFrame
        The validated DataFrame.

    Raises
    ------
    ValueError
        If ``df`` is ``None``, or if ``df.index`` or ``df.columns`` contains
        duplicate values.
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
    Validate that a DataFrame contains no missing values.

    Many downstream enrichment routines in this module assume dense numerical
    matrices with no ``NaN`` entries. This helper is therefore used to enforce
    that requirement before matrix multiplication, permutation testing, or
    storage inside ontology containers.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check.
    name : str
        Human-readable argument name used in error messages.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame, returned unchanged when validation succeeds.

    Raises
    ------
    ValueError
        If any entry in ``df`` is missing.
    """
    if df.isna().any().any():
        raise ValueError(
            f"{name} contains NaN values. Fill or remove missing values before using ontology_tools "
            f"(for example, factor_loadings = factor_loadings.fillna(0))."
        )
    return df


def as_dense(x):
    """
    Convert a dense or sparse matrix-like object to a NumPy ndarray.

    Parameters
    ----------
    x : array-like or scipy.sparse.spmatrix
        Input matrix or array.

    Returns
    -------
    numpy.ndarray
        Dense representation of ``x``. Sparse inputs are converted with
        ``toarray()``; dense inputs are passed through ``numpy.asarray``.
    """
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def as_csr(x: Union[np.ndarray, sp.spmatrix]) -> sp.csr_matrix:
    """
    Convert a dense or sparse matrix-like object to CSR sparse format.

    Parameters
    ----------
    x : numpy.ndarray or scipy.sparse.spmatrix
        Input matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR-formatted sparse matrix. Existing sparse inputs are converted with
        ``tocsr()``; dense inputs are wrapped with ``scipy.sparse.csr_matrix``.
    """
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def row_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each row of a DataFrame independently.

    For each row, values are centered by the row mean and scaled by the row
    standard deviation. Rows with zero variance are returned as all zeros.

    Parameters
    ----------
    df : pandas.DataFrame
        Input matrix whose rows should be standardized independently.

    Returns
    -------
    pandas.DataFrame
        Row-wise z-scored matrix with the same index and columns as ``df``.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    out = df.sub(mean, axis=0).div(std, axis=0)
    return out.fillna(0.0)


def safe_factor_metadata(factor_names: Sequence[str], separator: str = "|") -> pd.DataFrame:
    """
    Infer standard ontology factor metadata from factor names.

    Factor names are assumed to follow a convention such as
    ``Factor3|Tumor``. The function extracts the full factor name, the
    classification suffix after ``separator``, and the numeric factor index
    when the left-hand portion contains a token of the form ``FactorN``.

    Parameters
    ----------
    factor_names : sequence of str
        Factor identifiers.
    separator : str, default="|"
        Delimiter separating the base factor token from the lineage or
        classification label.

    Returns
    -------
    pandas.DataFrame
        Metadata table indexed by factor name with columns:

        - ``FactorName``: original factor identifier
        - ``Classification``: suffix after ``separator`` or ``"Unknown"``
        - ``Number``: parsed integer factor number when available

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
    Subset a gene-by-factor loading matrix to one ontology lineage.

    Parameters
    ----------
    factor_loadings : pandas.DataFrame
        Gene-by-factor loading matrix with factors in columns.
    cell_type : str
        Classification label to retain, for example ``"Tumor"`` or
        ``"Mg_TAM"``.
    separator : str, default="|"
        Delimiter used when parsing factor names.

    Returns
    -------
    pandas.DataFrame
        Copy of ``factor_loadings`` containing only factor columns whose
        inferred classification matches ``cell_type``.
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
    Build an ``AnnData`` object representing a single ontology modality.

    The returned object stores factor-by-feature scores in ``.X``, optional
    p-values in ``layers["pval"]``, optional factor metadata in ``.obs``, and
    optional feature-loading matrices in ``.varm``. When feature loadings are
    provided, their row labels are additionally recorded in ``uns["gene_names"]``
    and the row-axis style is tracked in ``uns["feature_loading_row_axis"]``.

    Parameters
    ----------
    score_df : pandas.DataFrame
        Factor-by-feature score matrix. Rows correspond to ontology factors and
        columns correspond to modality features.
    pval_df : pandas.DataFrame, optional
        Factor-by-feature p-value matrix aligned to ``score_df``.
    feature_loadings : pandas.DataFrame, optional
        Loading matrix to store for downstream interpretation. Expected shape is
        ``loading_row x feature``, where loading rows are either plain genes or
        labels such as ``gene|cell_type``.
    factor_metadata : pandas.DataFrame, optional
        Metadata table aligned to factor rows. Stored in ``.obs`` after
        reindexing to ``score_df.index``.
    feature_loading_key : str, default="feature_loadings"
        Key used when storing the transposed loading matrix in ``.varm``.
    cell_type_separator : str, default="|"
        Delimiter used to determine whether feature-loading row labels are
        cell-type tagged.

    Returns
    -------
    anndata.AnnData
        Modality container with aligned score, p-value, metadata, and optional
        loading information.

    Notes
    -----
    Feature loadings are reindexed to ``score_df.columns`` before storage, so
    missing features are filled with zeros and extra columns are dropped.
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
    Convert a gene list or weighted signature into a one-column DataFrame.

    This helper standardizes several user-facing signature formats to a common
    gene-by-weight table aligned to a chosen gene universe.

    Parameters
    ----------
    signature : sequence of str, pandas.Series, or pandas.DataFrame
        Signature to convert. Accepted forms are:

        - sequence of gene names, interpreted as a binary signature with weight 1
        - weighted Series indexed by gene
        - one-column DataFrame indexed by gene

    gene_index : sequence of str
        Reference gene universe used for reindexing and zero-filling.
    name : str, default="signature"
        Column name for the output.

    Returns
    -------
    pandas.DataFrame
        One-column gene-by-weight DataFrame indexed by ``gene_index``.

    Raises
    ------
    ValueError
        If ``signature`` is a DataFrame with more than one column.

    Notes
    -----
    Duplicate gene entries are averaged before reindexing.
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
    Extract ``adata.X`` or one of ``adata.layers`` as a labeled DataFrame.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to read from.
    layer : str, optional
        Layer name to extract. If omitted, ``adata.X`` is used.

    Returns
    -------
    pandas.DataFrame
        Dense DataFrame with ``adata.obs_names`` as rows and
        ``adata.var_names`` as columns.

    Raises
    ------
    KeyError
        If ``layer`` is specified but is not present in ``adata.layers``.
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
    """
    Compute permutation-based enrichment of weighted signatures in sample profiles.

    The input ``samples_by_genes`` matrix is multiplied by the
    gene-by-signature matrix ``signatures`` to obtain observed sample-by-feature
    scores. A null distribution is then estimated by repeatedly permuting gene
    order in the signature matrix, preserving the marginal distribution of both
    inputs while breaking gene-level correspondence. Final scores are reported
    as z-scores relative to the permutation null, together with two-sided
    empirical p-values.

    Parameters
    ----------
    samples_by_genes : pandas.DataFrame
        Matrix with samples, factors, or observations in rows and genes in
        columns.
    signatures : pandas.DataFrame
        Gene-by-feature matrix of weighted signatures. Rows must be genes and
        columns correspond to features to score.
    n_iter : int, default=1000
        Number of gene-label permutations used to estimate the null
        distribution.
    seed : int, default=0
        Random seed for permutation reproducibility.
    show_progress : bool, default=True
        Whether to display a progress bar.
    progress_message : str, default="Permuting..."
        Message displayed by the progress bar.

    Returns
    -------
    MatrixPair
        Object containing:

        - ``score``: observed-vs-null z-scores
        - ``pval``: two-sided empirical p-values

    Raises
    ------
    ValueError
        If either input contains missing values, is malformed, or if the two
        matrices share no genes.

    Notes
    -----
    Only the intersection of ``samples_by_genes.columns`` and
    ``signatures.index`` is used.
    """
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
    """
    Run robust preranked GSEA on a factor loading matrix.

    Each factor column is treated as a ranked gene list and scored against the
    supplied gene sets using ``gseapy.prerank``. The implementation adds a tiny
    deterministic jitter to break extreme ties, adapts ``min_size`` to small
    overlap cases, and falls back to neutral outputs when GSEA fails or when a
    factor has no overlap with the supplied signatures.

    Parameters
    ----------
    factor_loadings : pandas.DataFrame
        Gene-by-factor loading matrix.
    gene_sets : mapping or str
        Either a mapping ``{set_name: genes}`` or a gene-set library name
        understood by ``gseapy``.
    permutation_num : int, default=1000
        Number of permutations used by GSEA.
    min_size : int, default=10
        Minimum gene-set size passed to GSEA, subject to automatic reduction in
        low-overlap settings.
    max_size : int, default=5000
        Maximum gene-set size passed to GSEA.
    seed : int, default=42
        Random seed for GSEA.
    processes : int, default=1
        Number of worker processes used by ``gseapy``.
    show_progress : bool, default=True
        Whether to display a progress bar over factors.
    progress_message : str, default="Permuting..."
        Message displayed by the progress bar.

    Returns
    -------
    MatrixPair
        Object containing:

        - ``score``: factor-by-gene-set normalized enrichment scores
        - ``pval``: aligned nominal p-values

    Raises
    ------
    ImportError
        If ``gseapy`` is not installed.
    ValueError
        If ``factor_loadings`` is invalid or contains missing values.

    Notes
    -----
    When GSEA cannot be run for a factor, the function returns NES=0 and p=1
    for the affected terms rather than failing the entire modality.
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
    """
    Summarize per-gene salience across factors within each lineage.

    Factor loadings are first normalized by the standard deviation of each
    factor column. For each requested cell type, the absolute normalized
    loadings are then summed across all factor columns whose names contain that
    lineage label.

    Parameters
    ----------
    factor_loadings : pandas.DataFrame
        Gene-by-factor loading matrix.
    cell_types : sequence of str
        Cell types or lineage labels to summarize.
    separator : str, default="|"
        Reserved for compatibility with factor naming conventions.

    Returns
    -------
    pandas.DataFrame
        Gene-by-cell-type salience matrix in which larger values indicate genes
        with larger aggregate absolute loading magnitude across factors in that
        lineage.
    """
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
    """
    Convert LIANA results into filtered ligand and receptor signature matrices.

    This helper reproduces the original package logic for transforming LIANA
    interaction tables into sender-ligand and receiver-receptor signature
    matrices that can be used as scored ontology modalities. Interactions are
    first summarized across sender/receiver contexts using LIANA magnitude and
    specificity ranks, then filtered using z-scored LIANA interaction strength
    together with a lineage-aware gene salience score derived from factor
    loadings.

    Parameters
    ----------
    liana : pandas.DataFrame
        LIANA interaction table containing at least the columns
        ``source``, ``target``, ``ligand_complex``, ``receptor_complex``,
        ``magnitude_rank``, and ``specificity_rank``.
    factor_loadings : pandas.DataFrame
        Gene-by-factor loading matrix used to compute MOFA or factor-derived
        salience of ligands and receptors within each cell type.
    cell_types : sequence of str, default=DEFAULT_CELL_TYPES
        Cell types used when computing per-lineage gene salience and when
        constructing the paired LR output matrices.
    plot_scores : bool, default=True
        Whether to show a diagnostic scatter plot of LIANA z-score versus
        factor-derived salience z-score.
    liana_z_threshold : float, default=0.0
        Minimum standardized LIANA interaction score required to retain an
        interaction.
    factor_z_threshold : float, default=0.0
        Minimum standardized factor-derived salience score required to retain an
        interaction.
    plot_size : float, default=0.1
        Point size for the optional diagnostic scatter plot.

    Returns
    -------
    tuple
        Five objects are returned:

        1. ``all_send_signatures_wide`` :
           gene-by-feature sender-ligand signature matrix
        2. ``all_rec_signatures_wide`` :
           gene-by-feature receiver-receptor signature matrix
        3. ``filtered`` :
           long-format filtered interaction table
        4. ``all_biopsy_factors_mag`` :
           pivoted LIANA magnitude matrix
        5. ``all_biopsy_factors_spec`` :
           pivoted LIANA specificity matrix

    Raises
    ------
    ValueError
        If required LIANA columns are missing.

    Notes
    -----
    Output columns are context labels such as ``rec by|Tumor`` or
    ``sent by|Mg_TAM`` and are designed to plug into downstream scored
    enrichment functions.
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
    Compute lineage-matched ligand or receptor enrichment for ontology factors.

    This function scores ontology factors against intermediate ligand or
    receptor signature matrices whose rows encode lineage-qualified genes and
    whose columns encode paired communication contexts. For each lineage, only
    the matching factor subset and matching row block of ``lr_loadings`` are
    compared.

    Parameters
    ----------
    factor_loadings : pandas.DataFrame
        Gene-by-factor ontology loading matrix.
    lr_loadings : pandas.DataFrame
        Intermediate LR signature matrix. Expected structure depends on
        ``direction``:

        - ``direction="ligand"``:
          rows = ``<sender_cell_type>|<ligand_gene>``,
          columns = ``rec by|<receiver_cell_type>``
        - ``direction="receptor"``:
          rows = ``<receiver_cell_type>|<receptor_gene>``,
          columns = ``sent by|<sender_cell_type>``

    cell_types : sequence of str, default=DEFAULT_CELL_TYPES
        Ordered list of ontology lineages to evaluate.
    cell_type_separator : str, default="|"
        Delimiter used in factor names and LR row labels.
    direction : {"ligand", "receptor"}, default="ligand"
        Which LR representation is being scored.
    n_iter : int, default=1000
        Number of permutations used by the enrichment routine.
    seed : int, default=0
        Random seed.
    show_progress : bool, default=True
        Whether to display progress bars.

    Returns
    -------
    MatrixPair
        Factor-by-context z-scores and p-values, aligned to the full factor set.

    Raises
    ------
    ValueError
        If ``direction`` is invalid or if no lineage-matched enrichments can be
        produced.
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
    """
    Score ontology factors against regulon loading matrices.

    In cell-type-aware mode, regulon columns are expected to be suffixed with a
    lineage label such as ``TF|Tumor``. Factors are then compared only against
    regulons from the matching lineage, and suffixes are stripped from the final
    feature names before lineage blocks are concatenated. In agnostic mode, all
    factors are scored against the full regulon matrix at once.

    Parameters
    ----------
    factor_loadings : pandas.DataFrame
        Gene-by-factor ontology loading matrix.
    regulon_loadings : pandas.DataFrame
        Gene-by-regulon loading matrix.
    cell_types : sequence of str, default=DEFAULT_CELL_TYPES
        Ordered list of ontology lineages to evaluate.
    cell_type_separator : str, default="|"
        Delimiter used in factor names and regulon column labels.
    cell_type_regulons : bool, default=True
        If ``True``, require lineage matching between factors and regulons. If
        ``False``, run a single cell-type-agnostic enrichment.
    n_iter : int, default=1000
        Number of permutations used in scored enrichment.
    seed : int, default=0
        Random seed.
    show_progress : bool, default=True
        Whether to show progress bars in cell-type-aware mode.
    progress_message : str, default="Permuting..."
        Message prefix used for the progress bar.

    Returns
    -------
    MatrixPair
        Factor-by-regulon z-score and p-value matrices.

    Raises
    ------
    ValueError
        If lineage-aware mode is requested but no matching lineage blocks are
        found.
    """
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
    labels : sequence of str
        Labels to parse.
    separator : str, default="|"
        Delimiter between the base token and the cell-type suffix.

    Returns
    -------
    tuple of pandas.Index
        Two aligned indices:

        - base labels with the suffix removed
        - cell-type suffixes, or ``None`` where no suffix is present
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
    Return whether any label in an index contains a cell-type suffix.

    Parameters
    ----------
    index : sequence of str
        Labels to inspect.
    separator : str, default="|"
        Delimiter used to mark cell-type-tagged labels.

    Returns
    -------
    bool
        ``True`` if at least one label contains ``separator``.
    """
    idx = pd.Index([str(x) for x in index])
    return bool(idx.str.contains(re.escape(separator), regex=True).any())


def _unique_preserve_order(values: Sequence[str]) -> pd.Index:
    """
    Return unique values in order of first appearance.

    Parameters
    ----------
    values : sequence of str
        Input values.

    Returns
    -------
    pandas.Index
        De-duplicated values, preserving original order.
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
    Collapse a gene-by-``(feature|cell_type)`` matrix to gene-by-feature form.

    Column suffixes are stripped, duplicate feature names are summed, and an
    optional lineage filter can be applied before collapsing.

    Parameters
    ----------
    df : pandas.DataFrame
        Matrix with plain gene rows and cell-type-tagged feature columns.
    separator : str, default="|"
        Delimiter separating feature names from cell-type labels.
    allowed_cell_types : sequence of str, optional
        If provided, only columns whose suffix is in this set are retained.

    Returns
    -------
    pandas.DataFrame
        Collapsed gene-by-feature matrix with plain gene rows and plain feature
        columns.
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
    Collapse a matrix with tagged columns while preserving lineage on the row axis.

    Input rows are plain genes and input columns are
    ``<feature>|<cell_type>``. The output is reorganized so that each lineage
    becomes a separate row block labeled ``<gene>|<cell_type>``, while columns
    are collapsed to plain feature names.

    Parameters
    ----------
    df : pandas.DataFrame
        Gene-by-``(feature|cell_type)`` matrix.
    separator : str, default="|"
        Delimiter separating tokens from cell-type labels.
    allowed_cell_types : sequence of str, optional
        Optional subset of lineages to keep.

    Returns
    -------
    pandas.DataFrame
        Row-tagged loading matrix with rows of the form ``gene|cell_type`` and
        plain feature columns.

    Raises
    ------
    ValueError
        If no cell-type-tagged columns are present after filtering.
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
    Collapse a fully cell-type-tagged matrix while preserving row lineage labels.

    Both rows and columns are expected to carry suffixes of the form
    ``|cell_type``. Only row/column blocks with matching cell types are
    retained. Within each lineage block, column suffixes are stripped and
    duplicate labels are summed.

    Parameters
    ----------
    df : pandas.DataFrame
        Matrix with rows ``<gene>|<cell_type>`` and columns
        ``<feature>|<cell_type>``.
    separator : str, default="|"
        Delimiter separating tokens from cell-type labels.
    allowed_cell_types : sequence of str, optional
        Optional subset of lineages to keep.

    Returns
    -------
    pandas.DataFrame
        Matrix with rows still labeled ``gene|cell_type`` and columns collapsed
        to plain feature names.

    Raises
    ------
    ValueError
        If no overlapping cell-type tags are found across rows and columns.
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
    Compute lineage-aware scored enrichment for a cell-type-tagged modality.

    The function supports two modality layouts:

    1. rows = ``<gene>``, columns = ``<feature>|<cell_type>``
    2. rows = ``<gene>|<cell_type>``, columns = ``<feature>|<cell_type>``

    In both cases, ontology factors are grouped by
    ``factor_meta["Classification"]`` and scored only against modality features
    from the matching lineage. A storage-ready feature-loading matrix is built
    in parallel so the resulting modality can later expose interpretable
    per-feature gene weights.

    Parameters
    ----------
    factor_weights : pandas.DataFrame
        Factor-by-gene ontology weight matrix.
    factor_meta : pandas.DataFrame
        Factor metadata indexed by factor name and containing a
        ``Classification`` column.
    modality_df : pandas.DataFrame
        Cell-type-aware scored modality matrix.
    n_iter : int, default=1000
        Number of permutations used for enrichment.
    seed : int, default=0
        Random seed.
    show_progress : bool, default=True
        Whether to display progress bars.
    progress_message : str, default="Permuting..."
        Base message for the progress bar; lineage labels are appended
        internally.
    cell_type_separator : str, default="|"
        Delimiter used to parse cell-type tags.
    collapse_output : bool, default=True
        If ``True``, score columns are collapsed to plain feature names across
        lineages. If ``False``, feature columns retain explicit
        ``feature|cell_type`` suffixes.

    Returns
    -------
    tuple
        Two objects are returned:

        1. ``MatrixPair`` of factor-by-feature enrichment scores and p-values
        2. feature-loading matrix aligned to the final feature axis

    Raises
    ------
    ValueError
        If no overlapping lineages are present between ontology factors and the
        modality labels, or if no lineage-specific enrichments can be produced.
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
    Add a new modality to an existing ontology object.

    The function supports three workflows:

    - **precomputed**: provide ``score_df`` (and optionally ``pval_df``)
    - **scored**: provide a gene-by-feature DataFrame in ``modality_data`` and
      compute permutation-based enrichment
    - **gene_set**: provide either a gene-set mapping or a gene-set library
      name in ``modality_data`` and compute preranked GSEA

    When ``modality_type="auto"``, the mode is inferred from the supplied
    inputs. For scored modalities, both agnostic and cell-type-aware enrichment
    are supported. Resulting modality scores, p-values, and optional feature
    loadings are stored as an ``AnnData`` object in ``ontology.mod[modality_name]``.

    Parameters
    ----------
    ontology : muon.MuData
        Existing ontology object.
    modality_name : str
        Name under which the new modality will be stored.
    score_df : pandas.DataFrame, optional
        Precomputed factor-by-feature score matrix. Required in precomputed
        mode.
    pval_df : pandas.DataFrame, optional
        Optional precomputed p-value matrix aligned to ``score_df``.
    modality_data : pandas.DataFrame or mapping or str, optional
        Raw modality input used in scored or gene-set mode.
    modality_type : {"auto", "precomputed", "scored", "gene_set"}, default="auto"
        How the new modality should be interpreted.
    feature_loadings : pandas.DataFrame, optional
        Explicit loading matrix to store for downstream interpretation.
    feature_loading_key : str, default="feature_loadings"
        Key used for storage in the modality ``.varm`` slot.
    modality_uns : mapping, optional
        Additional metadata copied into ``modality.uns``.
    n_iter : int, default=1000
        Default number of permutations for enrichment routines.
    seed : int, default=0
        Random seed passed to enrichment routines.
    show_progress : bool, default=True
        Whether to display progress output.
    permutation_kwargs : mapping, optional
        Additional keyword arguments forwarded to ``calc_enrichment`` or the
        lineage-aware scored enrichment helper.
    gsea_kwargs : mapping, optional
        Additional keyword arguments forwarded to ``gsea_enrichment``.
    store_input_as_feature_loadings : bool, default=True
        In scored mode, whether the supplied or inferred input matrix should
        also be stored as feature loadings when no explicit
        ``feature_loadings`` are provided.
    scored_cell_type_aware : bool, default=False
        Whether scored modality columns should be interpreted as
        lineage-qualified features and matched only to factors of the same
        lineage.
    cell_type_separator : str, default="|"
        Delimiter used to parse lineage labels in factor names and modality
        labels.
    collapse_cell_type_aware_scores : bool, default=True
        Whether to collapse lineage-qualified score columns to plain feature
        names in cell-type-aware scored mode.
    inplace : bool, default=True
        If ``True``, modify ``ontology`` in place and return ``None``.
        Otherwise return an updated copy.

    Returns
    -------
    muon.MuData or None
        ``None`` when ``inplace=True``; otherwise the updated ontology object.

    Raises
    ------
    ValueError
        If inputs are inconsistent with the requested mode, if score matrices
        contain factors absent from the ontology, or if ontology metadata
        needed for alignment are missing.
    KeyError
        If required ontology components such as ``ontology.uns["gene_names"]``
        or the ``"weights"`` modality are absent.

    Notes
    -----
    The function attempts to reconcile cell-type-tagged feature-loading matrices
    with collapsed score axes when necessary, so that stored loadings remain
    aligned to the final modality feature names.
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
    
        # If the provided feature loadings do not already match the score_df feature axis,
        # try to reconcile cell-type-tagged feature columns to the collapsed score columns.
        if not pd.Index(score_df.columns).isin(feature_loadings.columns).all():
            if _index_has_cell_type_tags(feature_loadings.columns, separator=cell_type_separator):
                allowed_cts = factor_meta["Classification"].dropna().astype(str).unique()
    
                # rows already tagged: (gene|ct) x (feature|ct)  ->  (gene|ct) x feature
                if _index_has_cell_type_tags(feature_loadings.index, separator=cell_type_separator):
                    feature_loadings = _collapse_cell_type_tagged_matrix(
                        feature_loadings,
                        separator=cell_type_separator,
                        allowed_cell_types=allowed_cts,
                    )
    
                # plain gene rows: gene x (feature|ct)  ->  (gene|ct) x feature
                else:
                    feature_loadings = _collapse_column_cell_type_tagged_matrix_preserve_rows(
                        feature_loadings,
                        separator=cell_type_separator,
                        allowed_cell_types=allowed_cts,
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
    Construct a factor-centric ontology ``MuData`` object from factor loadings.

    The ontology is initialized from a gene-by-factor loading matrix and stores:

    - factor metadata in ``ontology.obs``
    - the global factor weight matrix as a modality named ``"weights"``
    - gene names in ``ontology.uns["gene_names"]``
    - additional modalities in ``ontology.mod``

    Scored modalities are processed with permutation-based enrichment and
    gene-set modalities are processed with preranked GSEA. Cell-type-aware
    scored modalities can also be supplied and are appended after initial object
    creation using the same lineage-aware logic exposed by ``add_modality``.

    Parameters
    ----------
    factor_loadings : pandas.DataFrame
        Gene-by-factor loading matrix. Rows are genes and columns are factors.
    factor_type : str, default="MOFA"
        Label describing the source or type of factors. Stored in
        ``ontology.uns["factor_type"]`` and copied into the ``FactorType``
        column of factor metadata.
    factor_metadata : pandas.DataFrame, optional
        Optional metadata indexed by factor name. Missing standard fields are
        inferred from factor names where possible.
    cell_type_separator : str, default="|"
        Delimiter used when parsing factor names and lineage-aware modality
        labels.
    scored_modalities : mapping, optional
        Mapping from modality name to gene-by-feature scored matrices to be
        evaluated in a cell-type-agnostic manner.
    scored_modalities_ct_aware : mapping, optional
        Mapping from modality name to lineage-aware scored matrices whose
        columns carry ``|cell_type`` suffixes and should be matched to ontology
        factor lineages.
    gene_set_modalities : mapping, optional
        Mapping from modality name to gene-set collections or gene-set library
        names.
    scored_feature_loadings : mapping, optional
        Optional explicit loading matrices to store for scored modalities.
    feature_loading_key_map : mapping, optional
        Per-modality override for the key used in ``.varm`` to store feature
        loadings.
    scored_modality_uns : mapping, optional
        Per-modality metadata dictionaries for scored modalities.
    gene_set_modality_uns : mapping, optional
        Per-modality metadata dictionaries for gene-set modalities.
    n_iter : int, default=1000
        Default number of permutations used for scored modalities and default
        GSEA permutation count.
    seed : int, default=0
        Random seed passed to enrichment routines.
    show_progress : bool, default=True
        Whether to display progress output during ontology construction.

    Returns
    -------
    muon.MuData
        Newly constructed ontology object.

    Raises
    ------
    ValueError
        If ``factor_loadings`` are malformed, contain missing values, or if no
        user-facing modalities are supplied in an environment where an empty
        ontology cannot be constructed.

    Notes
    -----
    Specialized modality preprocessing is intentionally externalized. For
    example, LIANA outputs should first be converted into scored ligand and
    receptor signature matrices before being passed into ontology construction.
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
    #ontology.obsm["weights"] = as_csr(fl.T.to_numpy(dtype=np.float32))
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
    Normalize a cell-type selector to a list of strings.

    Parameters
    ----------
    cell_types : str, sequence of str, or None
        User-supplied lineage selector.

    Returns
    -------
    list of str or None
        Normalized list of cell types, or ``None`` if no filtering was
        requested.
    """
    if cell_types is None:
        return None
    if isinstance(cell_types, str):
        return [cell_types]
    return list(cell_types)


def _filter_factor_index_by_cell_types(index: pd.Index, ontology: mu.MuData, cell_types: Optional[Union[str, Sequence[str]]]) -> pd.Index:
    """
    Filter a factor index by ontology lineage metadata.

    Parameters
    ----------
    index : pandas.Index
        Factor names to filter.
    ontology : muon.MuData
        Ontology object whose ``obs`` table contains a ``Classification``
        column.
    cell_types : str, sequence of str, or None
        Requested lineage subset.

    Returns
    -------
    pandas.Index
        Subset of ``index`` whose factors belong to the requested lineages.
    """
    ct_list = _normalize_cell_types(cell_types)
    if ct_list is None:
        return index
    meta = ontology.obs.reindex(index)
    keep = meta["Classification"].astype(str).isin(ct_list)
    return index[keep.values]


def modality_scores_to_df(ontology: mu.MuData, modality: str, cell_types: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
    """
    Return a modality score matrix as a pandas DataFrame.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    modality : str
        Name of the modality to extract.
    cell_types : str or sequence of str, optional
        Restrict factor rows to selected ontology lineages.

    Returns
    -------
    pandas.DataFrame
        Factor-by-feature score matrix for ``modality``.
    """
    mod = ontology.mod[modality]
    df = pd.DataFrame(as_dense(mod.X), index=mod.obs_names, columns=mod.var_names)
    keep = _filter_factor_index_by_cell_types(df.index, ontology, cell_types)
    return df.loc[keep].copy()


def modality_pvals_to_df(ontology: mu.MuData, modality: str, cell_types: Optional[Union[str, Sequence[str]]] = None) -> Optional[pd.DataFrame]:
    """
    Return a modality p-value matrix as a pandas DataFrame.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    modality : str
        Name of the modality to extract.
    cell_types : str or sequence of str, optional
        Restrict factor rows to selected ontology lineages.

    Returns
    -------
    pandas.DataFrame or None
        Factor-by-feature p-value matrix if ``modality`` stores a ``"pval"``
        layer; otherwise ``None``.
    """
    mod = ontology.mod[modality]
    if "pval" not in mod.layers:
        return None
    df = pd.DataFrame(as_dense(mod.layers["pval"]), index=mod.obs_names, columns=mod.var_names)
    keep = _filter_factor_index_by_cell_types(df.index, ontology, cell_types)
    return df.loc[keep].copy()


def factor_weights_to_df(ontology: mu.MuData, transpose: bool = False, cell_types: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
    """
    Return the global ontology factor weight matrix as a DataFrame.

    In the current storage scheme, global weights are kept as the modality
    ``ontology.mod["weights"]`` rather than in ``ontology.obsm``.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    transpose : bool, default=False
        If ``False``, return ``factor x gene``. If ``True``, return
        ``gene x factor``.
    cell_types : str or sequence of str, optional
        Restrict factor rows to selected ontology lineages before optional
        transposition.

    Returns
    -------
    pandas.DataFrame
        Global factor weight matrix.

    Raises
    ------
    KeyError
        If the ``"weights"`` modality or ``ontology.uns["gene_names"]`` is
        missing.
    """
    if "weights" not in list(ontology.mod.keys()):
        raise KeyError("ontology.obsm['weights'] not found.")
    if "gene_names" not in ontology.uns:
        raise KeyError("ontology.uns['gene_names'] not found.")

    weights = ontology.mod['weights'].to_df()
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
    Reconstruct a modality feature-loading matrix from stored ``.varm`` arrays.

    This helper supports both the current single-key storage scheme and older
    per-lineage ``varm`` layouts. When row labels are stored as
    ``gene|cell_type``, optional lineage filtering can be applied and suffixes
    can be stripped automatically when only one lineage is requested.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    modality : str
        Modality from which to retrieve feature loadings.
    key : str, optional
        Specific ``varm`` key to use. If omitted, the function prefers the
        modality's stored default key and otherwise falls back to automatic
        selection.
    cell_types : str or sequence of str, optional
        Optional lineage subset used to filter row-tagged loading matrices.
    strip_cell_type_suffix : bool, optional
        Whether to remove ``|cell_type`` suffixes from row names after
        filtering. If omitted, suffixes are stripped automatically when exactly
        one cell type is requested.

    Returns
    -------
    pandas.DataFrame
        Loading matrix with loading rows as index and modality features as
        columns.

    Raises
    ------
    KeyError
        If no stored feature-loadings are available or if the requested key does
        not exist.
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
    ontology : muon.MuData
        Ontology object.
    factors : sequence of str, optional
        Optional subset of factor names to retain.
    modalities : sequence of str, optional
        Optional subset of modality names. If omitted, all modalities are
        queried.
    cell_types : str or sequence of str, optional
        Optional lineage filter applied before factor subsetting.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Mapping from modality name to factor-by-feature score matrix.
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
    Retrieve feature-loadings overlapping a supplied gene set.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    gene_set : sequence of str
        Genes to query.
    modalities : sequence of str, optional
        Optional subset of modalities to search. If omitted, all modalities are
        queried.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Mapping from modality name to the subset of its loading matrix whose
        row labels overlap ``gene_set``.

    Notes
    -----
    Genes are de-duplicated while preserving input order.
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
    Return the top positive and negative features for a factor.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    factor : str
        Factor identifier, for example ``"Factor0|Tumor"``.
    modality : str
        Modality to query, or ``"weights"`` to query the global ontology weight
        matrix stored in ``ontology.mod["weights"]``.
    n_pos : int, default=10
        Number of highest-scoring features to return.
    n_neg : int, default=10
        Number of lowest-scoring features to return.
    alpha : float, optional
        Optional p-value threshold used to pre-filter modality features before
        ranking. Ignored for ``modality="weights"``.

    Returns
    -------
    tuple of pandas.DataFrame
        Two DataFrames with columns ``feature``, ``score``, and ``pval``:

        - positive features sorted descending by score
        - negative features sorted ascending by score

    Raises
    ------
    KeyError
        If the requested factor or modality is not present.
    """
    if modality == "weights":
        if "weights" not in list(ontology.mod.keys()):
            raise KeyError("ontology.obsm['weights'] not found.")
        if "gene_names" not in ontology.uns:
            raise KeyError("ontology.uns['gene_names'] not found.")
        weights = ontology.mod['weights'].to_df()
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
