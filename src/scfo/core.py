
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
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
    if df.isna().any().any():
        raise ValueError(
            f"{name} contains NaN values. Fill or remove missing values before using ontology_tools "
            f"(for example, factor_loadings = factor_loadings.fillna(0))."
        )
    return df


def as_dense(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def as_csr(x: Union[np.ndarray, sp.spmatrix]) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def row_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    out = df.sub(mean, axis=0).div(std, axis=0)
    return out.fillna(0.0)


def safe_factor_metadata(factor_names: Sequence[str], separator: str = "|") -> pd.DataFrame:
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
    meta = safe_factor_metadata(factor_loadings.columns, separator=separator)
    keep = meta.index[meta["Classification"] == cell_type]
    return factor_loadings.loc[:, keep].copy()


def build_modality_adata(
    score_df: pd.DataFrame,
    pval_df: Optional[pd.DataFrame] = None,
    feature_loadings: Optional[pd.DataFrame] = None,
    factor_metadata: Optional[pd.DataFrame] = None,
    feature_loading_key: str = "feature_loadings",
) -> ad.AnnData:
    require_dataframe(score_df, "score_df")
    if pval_df is not None:
        require_dataframe(pval_df, "pval_df")
        pval_df = pval_df.reindex(index=score_df.index, columns=score_df.columns)
    if feature_loadings is not None:
        require_dataframe(feature_loadings, "feature_loadings")
        feature_loadings = feature_loadings.loc[:, score_df.columns]

    obs = factor_metadata.reindex(score_df.index).copy() if factor_metadata is not None else pd.DataFrame(index=score_df.index)
    var = pd.DataFrame(index=score_df.columns)
    adata = ad.AnnData(X=score_df.to_numpy(dtype=np.float32), obs=obs, var=var)
    adata.obs_names = score_df.index.astype(str)
    adata.var_names = score_df.columns.astype(str)
    if pval_df is not None:
        adata.layers["pval"] = pval_df.to_numpy(dtype=np.float32)
    if feature_loadings is not None:
        adata.varm[feature_loading_key] = as_csr(feature_loadings.T.to_numpy(dtype=np.float32))
        adata.uns["gene_names"] = list(feature_loadings.index.astype(str))
    return adata


def signature_to_df(
    signature: Union[Sequence[str], pd.Series, pd.DataFrame],
    gene_index: Sequence[str],
    name: str = "signature",
) -> pd.DataFrame:
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
        iterator = tqdm(iterator, desc="Permuting", leave=False)
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

    for factor in fl.columns:
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
    return all_send_signatures, all_rec_signatures, filtered, all_biopsy_factors_mag, all_biopsy_factors_spec



def lr_enrichment(
    factor_loadings: pd.DataFrame,
    lr_loadings: pd.DataFrame,
    cell_types: Sequence[str] = DEFAULT_CELL_TYPES,
    cell_type_separator: str = "|",
    direction: str = "ligand",
    n_iter: int = 1000,
    seed: int = 0,
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
            show_progress=False,
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
        pair = calc_enrichment(factor_ct.T, reg_ct, n_iter=n_iter, seed=seed, show_progress=False)
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


def make_ontology(
    factor_loadings: pd.DataFrame,
    factor_type: str = "MOFA",
    n_iter: int = 1000,
    cell_types: Sequence[str] = DEFAULT_CELL_TYPES,
    hallmark_lib: Optional[Union[Mapping[str, Sequence[str]], str]] = None,
    gene_sets: Optional[Union[Mapping[str, Sequence[str]], str]] = None,
    sender_loadings: Optional[pd.DataFrame] = None,
    receiver_loadings: Optional[pd.DataFrame] = None,
    liana: Optional[pd.DataFrame] = None,
    liana_z_threshold: float = 0.0,
    factor_z_threshold: float = 0.0,
    regulon_loadings: Optional[pd.DataFrame] = None,
    cell_type_regulons: bool = True,
    cell_type_separator: str = "|",
    feature_loading_key_map: Optional[Mapping[str, str]] = None,
    plot_liana_scores: bool = True,
    liana_plot_size: float = 0.1,
) -> mu.MuData:
    """Build a factor-centric ontology object from factor loadings and optional annotation modalities.

    Parameters
    ----------
    
    factor_loadings : pandas.DataFrame
        Gene-by-factor loading matrix. Rows must be genes and columns must be factors.
        Factor names are expected to follow the convention ``FactorN|CellType`` so that
        lineage information can be parsed automatically.
    factor_type : str, default="MOFA"
        Label describing the source of the factor loadings.
    n_iter : int, default=1000
        Number of permutations to use for permutation-based enrichment steps.
    cell_types : sequence of str, optional
        Ordered list of ontology cell types to use when matching lineage-specific
        regulon and ligand–receptor annotations.
    hallmark_lib : mapping or str, optional
        Gene set collection for hallmark-style enrichment.
    gene_sets : mapping or str, optional
        Additional gene sets for factor annotation.
    sender_loadings : pandas.DataFrame, optional
        Precomputed ligand-side paired signature matrix.
    receiver_loadings : pandas.DataFrame, optional
        Precomputed receptor-side paired signature matrix.
    liana : pandas.DataFrame, optional
        Raw LIANA results table. If provided and sender/receptor loadings are not given,
        paired ligand and receptor signatures will be constructed automatically.
    liana_z_threshold : float, default=0.0
        Minimum LIANA z-score for retaining ligand–receptor pairs.
    factor_z_threshold : float, default=0.0
        Minimum factor-support z-score for retaining ligand–receptor pairs.
    regulon_loadings : pandas.DataFrame, optional
        Gene-by-regulon loading matrix. Column names should follow the convention
        ``TF(+)|CellType`` when lineage-specific regulons are used.
    cell_type_regulons : bool, default=True
        Whether regulon columns are lineage-specific.
    cell_type_separator : str, default="|"
        Separator used in factor and regulon names.
    feature_loading_key_map : mapping, optional
        Optional mapping to override the names of modality-specific loading matrices
        stored in ``varm``.
    plot_liana_scores : bool, default=False
        Whether to display the LIANA filtering scatterplot during ontology construction.
    liana_plot_size : float, default=2.0
        Marker size for the LIANA filtering scatterplot.

    Returns
    -------
    muon.MuData
        Factor-centric ontology object with factors in ``obs``, global weights in
        ``obsm['weights']``, and annotation modalities in ``mod``.
    """
    fl = ensure_no_nan(require_dataframe(factor_loadings, "factor_loadings"), "factor_loadings")
    if fl.empty:
        raise ValueError("factor_loadings is empty.")
    feature_loading_key_map = dict(feature_loading_key_map or {})
    factor_meta = safe_factor_metadata(fl.columns, separator=cell_type_separator)
    modalities: Dict[str, ad.AnnData] = {}

    if hallmark_lib is not None:
        pair = gsea_enrichment(fl, hallmark_lib)
        modalities["hallmark"] = build_modality_adata(pair.score, pair.pval, None, factor_meta, feature_loading_key=feature_loading_key_map.get("hallmark", "feature_loadings"))

    if gene_sets is not None:
        pair = gsea_enrichment(fl, gene_sets)
        modalities["gene_sets"] = build_modality_adata(pair.score, pair.pval, None, factor_meta, feature_loading_key=feature_loading_key_map.get("gene_sets", "feature_loadings"))

    if liana is not None and sender_loadings is None and receiver_loadings is None:
        sender_loadings, receiver_loadings, liana_filtered, liana_mag, liana_spec = make_filtered_lr_signatures(
            liana=liana,
            factor_loadings=fl,
            cell_types=cell_types,
            plot_scores=plot_liana_scores,
            liana_z_threshold=liana_z_threshold,
            factor_z_threshold=factor_z_threshold,
            plot_size=liana_plot_size,
        )
    else:
        liana_filtered = None
        liana_mag = None
        liana_spec = None

    if sender_loadings is not None:
        sender_loadings = ensure_no_nan(require_dataframe(sender_loadings, "sender_loadings"), "sender_loadings")
        pair = lr_enrichment(fl, sender_loadings, cell_types=cell_types, cell_type_separator=cell_type_separator, direction="ligand", n_iter=n_iter)
        lig_adata = build_modality_adata(pair.score, pair.pval, feature_loadings=None, factor_metadata=factor_meta)
        lig_adata.uns["gene_names"] = list(pd.Index(sender_loadings.index.astype(str).str.split(cell_type_separator, n=1).str[1]).unique())
        if liana_filtered is not None:
            lig_adata.uns["liana_filtered"] = liana_filtered
        if liana_mag is not None:
            lig_adata.uns["liana_magnitude_matrix"] = liana_mag
        if liana_spec is not None:
            lig_adata.uns["liana_specificity_matrix"] = liana_spec
        for ct in cell_types:
            row_mask = sender_loadings.index.astype(str).str.startswith(f"{ct}{cell_type_separator}")
            if row_mask.sum() == 0:
                continue
            sub = sender_loadings.loc[row_mask].copy()
            sub.index = sub.index.astype(str).str.split(cell_type_separator, n=1).str[1]
            sub = sub.groupby(level=0).sum()
            sub = sub.reindex(index=lig_adata.uns["gene_names"], columns=lig_adata.var_names).fillna(0.0)
            lig_adata.varm[f"{ct}_ligand_signatures"] = as_csr(sub.T.to_numpy(dtype=np.float32))
        modalities["liana_ligand"] = lig_adata

    if receiver_loadings is not None:
        receiver_loadings = ensure_no_nan(require_dataframe(receiver_loadings, "receiver_loadings"), "receiver_loadings")
        pair = lr_enrichment(fl, receiver_loadings, cell_types=cell_types, cell_type_separator=cell_type_separator, direction="receptor", n_iter=n_iter)
        rec_adata = build_modality_adata(pair.score, pair.pval, feature_loadings=None, factor_metadata=factor_meta)
        rec_adata.uns["gene_names"] = list(pd.Index(receiver_loadings.index.astype(str).str.split(cell_type_separator, n=1).str[1]).unique())
        if liana_filtered is not None:
            rec_adata.uns["liana_filtered"] = liana_filtered
        if liana_mag is not None:
            rec_adata.uns["liana_magnitude_matrix"] = liana_mag
        if liana_spec is not None:
            rec_adata.uns["liana_specificity_matrix"] = liana_spec
        for ct in cell_types:
            row_mask = receiver_loadings.index.astype(str).str.startswith(f"{ct}{cell_type_separator}")
            if row_mask.sum() == 0:
                continue
            sub = receiver_loadings.loc[row_mask].copy()
            sub.index = sub.index.astype(str).str.split(cell_type_separator, n=1).str[1]
            sub = sub.groupby(level=0).sum()
            sub = sub.reindex(index=rec_adata.uns["gene_names"], columns=rec_adata.var_names).fillna(0.0)
            rec_adata.varm[f"{ct}_receptor_signatures"] = as_csr(sub.T.to_numpy(dtype=np.float32))
        modalities["liana_receptor"] = rec_adata

    if regulon_loadings is not None:
        regulon_loadings = ensure_no_nan(require_dataframe(regulon_loadings, "regulon_loadings"), "regulon_loadings")
        pair = regulon_enrichment(
            factor_loadings=fl,
            regulon_loadings=regulon_loadings,
            cell_types=cell_types,
            cell_type_separator=cell_type_separator,
            cell_type_regulons=cell_type_regulons,
            n_iter=n_iter,
        )
        reg_adata = build_modality_adata(pair.score, pair.pval, feature_loadings=None, factor_metadata=factor_meta)
        if cell_type_regulons:
            reg_adata.uns["gene_names"] = list(regulon_loadings.index.astype(str))
            for ct in cell_types:
                reg_cols = [c for c in regulon_loadings.columns if str(c).endswith(f"{cell_type_separator}{ct}")]
                if len(reg_cols) == 0:
                    continue
                reg_ct = regulon_loadings.loc[:, reg_cols].copy()
                reg_ct.columns = [str(c).rsplit(f"{cell_type_separator}{ct}", 1)[0] if str(c).endswith(f"{cell_type_separator}{ct}") else str(c) for c in reg_ct.columns]
                reg_ct = reg_ct.groupby(reg_ct.columns, axis=1).mean()
                reg_ct = reg_ct.reindex(index=regulon_loadings.index, columns=reg_adata.var_names).fillna(0.0)
                reg_adata.varm[f"{ct}_regulons"] = as_csr(reg_ct.T.to_numpy(dtype=np.float32))
        else:
            shared_regs = regulon_loadings.reindex(index=regulon_loadings.index, columns=reg_adata.var_names).fillna(0.0)
            reg_adata.varm[feature_loading_key_map.get("regulons", "regulon_loadings")] = as_csr(shared_regs.T.to_numpy(dtype=np.float32))
            reg_adata.uns["gene_names"] = list(regulon_loadings.index.astype(str))
        modalities["regulons"] = reg_adata

    if len(modalities) == 0:
        raise ValueError("No ontology modalities were created. Provide at least one annotation modality.")
    mdata = mu.MuData(modalities)
    mdata.obs = factor_meta.reindex(mdata.obs_names).copy()
    mdata.obsm["weights"] = as_csr(fl.T.to_numpy(dtype=np.float32))
    mdata.uns["gene_names"] = list(fl.index.astype(str))
    mdata.uns["factor_type"] = factor_type
    mdata.uns["cell_type_separator"] = cell_type_separator
    return mdata


def _normalize_cell_types(cell_types: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if cell_types is None:
        return None
    if isinstance(cell_types, str):
        return [cell_types]
    return list(cell_types)


def _filter_factor_index_by_cell_types(index: pd.Index, ontology: mu.MuData, cell_types: Optional[Union[str, Sequence[str]]]) -> pd.Index:
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
) -> pd.DataFrame:
    """Return a modality loading matrix as a DataFrame.

    Parameters
    ----------
    
    ontology : muon.MuData
        Ontology object.
    modality : str
        Name of the modality to extract.
    key : str, optional
        Specific ``varm`` key to extract. If omitted, the first available loading matrix
        is used.
    cell_types : str or sequence of str, optional
        Restrict returned features to lineage-matched loading matrices when available.

    Returns
    -------
    pandas.DataFrame
        Gene-by-feature loading matrix.
    """
    mod = ontology.mod[modality]
    varm_keys = list(mod.varm.keys())
    if len(varm_keys) == 0:
        raise KeyError(f"No varm matrices found for modality '{modality}'.")

    gene_names = mod.uns.get("gene_names", ontology.uns.get("gene_names"))
    if gene_names is None:
        raise KeyError(f"No gene names stored for modality '{modality}'.")
    gene_index = pd.Index(gene_names, name="gene")

    if key is not None:
        arr = as_dense(mod.varm[key])
        return pd.DataFrame(arr.T, index=gene_index, columns=mod.var_names)

    ct_list = _normalize_cell_types(cell_types)
    selected_keys = []
    if ct_list is None:
        selected_keys = varm_keys
    else:
        for ct in ct_list:
            matches = [k for k in varm_keys if str(k).startswith(f"{ct}_")]
            selected_keys.extend(matches)
        if len(selected_keys) == 0 and len(varm_keys) == 1:
            selected_keys = varm_keys

    # deduplicate preserve order
    out_keys = []
    seen = set()
    for k in selected_keys:
        if k not in seen:
            out_keys.append(k)
            seen.add(k)

    if len(out_keys) == 1:
        arr = as_dense(mod.varm[out_keys[0]])
        return pd.DataFrame(arr.T, index=gene_index, columns=mod.var_names)

    dfs = []
    for k in out_keys:
        arr = as_dense(mod.varm[k])
        df = pd.DataFrame(arr.T, index=gene_index, columns=mod.var_names)
        suffix = str(k).split("_", 1)[0] if "_" in str(k) else str(k)
        df.columns = [f"{c}|{suffix}" for c in df.columns]
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def get_factor_scores(ontology: mu.MuData, factors: Optional[Sequence[str]] = None, modalities: Optional[Sequence[str]] = None, cell_types: Optional[Union[str, Sequence[str]]] = None) -> Dict[str, pd.DataFrame]:
    modalities = list(ontology.mod.keys()) if modalities is None else list(modalities)
    out: Dict[str, pd.DataFrame] = {}
    for modality in modalities:
        df = modality_scores_to_df(ontology, modality, cell_types=cell_types)
        if factors is not None:
            df = df.loc[list(factors)]
        out[modality] = df
    return out


def query_gene_set(ontology: mu.MuData, gene_set: Sequence[str], modalities: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
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
