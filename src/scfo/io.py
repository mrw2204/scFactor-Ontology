from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union
import re

import anndata as ad
import muon as mu
import pandas as pd

from .core import as_csr, as_dense, build_modality_adata, modality_feature_loadings_to_df, modality_pvals_to_df, modality_scores_to_df


def sanitize_sheet_name(name: str, used: Optional[set] = None, max_len: int = 31) -> str:
    if used is None:
        used = set()
    clean = re.sub(r"[:\\/*?\[\]]", "_", str(name))[:max_len]
    base = clean
    i = 1
    while clean in used:
        suffix = f"_{i}"
        clean = base[: max_len - len(suffix)] + suffix
        i += 1
    used.add(clean)
    return clean


def export_ontology_excel(
    ontology: mu.MuData,
    outdir: Union[str, Path] = ".",
    prefix: str = "ontology",
) -> Dict[str, str]:
    """Export an ontology object to standardized Excel workbooks.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    outdir : str or pathlib.Path, default='.'
        Output directory.
    prefix : str, default='ontology'
        Filename prefix.

    Returns
    -------
    dict
        Mapping with paths to the scores and features workbooks.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scores_path = outdir / f"{prefix}_scores.xlsx"
    features_path = outdir / f"{prefix}_features.xlsx"
    used = set()
    manifest_scores = []
    with pd.ExcelWriter(scores_path, engine="openpyxl") as writer:
        obs_df = ontology.obs.copy()
        obs_df.index.name = "obs_id"
        sheet = sanitize_sheet_name("obs", used)
        obs_df.reset_index().to_excel(writer, sheet_name=sheet, index=False)
        manifest_scores.append({"sheet": sheet, "role": "obs", "modality": "", "key": ""})
        if "weights" in ontology.obsm:
            weights_arr = as_dense(ontology.obsm["weights"])
            weights = pd.DataFrame(
                weights_arr,
                index=ontology.obs_names,
                columns=ontology.uns.get("gene_names", [f"gene_{i}" for i in range(weights_arr.shape[1])]),
            )
            weights.index.name = "obs_id"
            sheet = sanitize_sheet_name("global_weights", used)
            weights.reset_index().to_excel(writer, sheet_name=sheet, index=False)
            manifest_scores.append({"sheet": sheet, "role": "global_weights", "modality": "", "key": "weights"})
        for modality, mod in ontology.mod.items():
            scores = modality_scores_to_df(ontology, modality)
            scores.index.name = "obs_id"
            sheet = sanitize_sheet_name(f"{modality}__scores", used)
            scores.reset_index().to_excel(writer, sheet_name=sheet, index=False)
            manifest_scores.append({"sheet": sheet, "role": "scores", "modality": modality, "key": "X"})
            pvals = modality_pvals_to_df(ontology, modality)
            if pvals is not None:
                pvals.index.name = "obs_id"
                sheet = sanitize_sheet_name(f"{modality}__pvals", used)
                pvals.reset_index().to_excel(writer, sheet_name=sheet, index=False)
                manifest_scores.append({"sheet": sheet, "role": "pvals", "modality": modality, "key": "pval"})
        pd.DataFrame(manifest_scores).to_excel(writer, sheet_name=sanitize_sheet_name("MANIFEST", used), index=False)

    used = set()
    manifest_features = []
    with pd.ExcelWriter(features_path, engine="openpyxl") as writer:
        for modality, mod in ontology.mod.items():
            var_df = mod.var.copy() if mod.var.shape[1] > 0 else pd.DataFrame(index=mod.var_names)
            var_df.index.name = "feature"
            sheet = sanitize_sheet_name(f"{modality}__features", used)
            var_df.reset_index().to_excel(writer, sheet_name=sheet, index=False)
            manifest_features.append({"sheet": sheet, "role": "var", "modality": modality, "key": "var"})
            for key in mod.varm.keys():
                df = modality_feature_loadings_to_df(ontology, modality, key)
                df.index.name = "gene"
                sheet = sanitize_sheet_name(f"{modality}__{key}", used)
                df.reset_index().to_excel(writer, sheet_name=sheet, index=False)
                manifest_features.append({"sheet": sheet, "role": "varm", "modality": modality, "key": key})
        pd.DataFrame(manifest_features).to_excel(writer, sheet_name=sanitize_sheet_name("MANIFEST", used), index=False)
    return {"scores": str(scores_path), "features": str(features_path)}


def load_ontology_excel(
    scores_xlsx: Union[str, Path],
    features_xlsx: Union[str, Path],
    factor_type: str = "unknown",
    cell_type_separator: str = "|",
) -> mu.MuData:
    """Reconstruct an ontology object from standardized Excel exports.

    Parameters
    ----------
    scores_xlsx : str or pathlib.Path
        Path to the workbook containing scores, p-values, and factor metadata.
    features_xlsx : str or pathlib.Path
        Path to the workbook containing feature metadata and loading matrices.
    factor_type : str, default='unknown'
        Label stored in the reconstructed ontology metadata.
    cell_type_separator : str, default='|'
        Separator used in factor naming.

    Returns
    -------
    muon.MuData
        Reconstructed ontology object.
    """
    scores_xlsx = Path(scores_xlsx)
    features_xlsx = Path(features_xlsx)
    scores_manifest = pd.read_excel(scores_xlsx, sheet_name="MANIFEST")
    features_manifest = pd.read_excel(features_xlsx, sheet_name="MANIFEST")
    obs_sheet = scores_manifest.loc[scores_manifest["role"] == "obs", "sheet"]
    if len(obs_sheet) != 1:
        raise ValueError("Scores workbook must contain exactly one obs sheet.")
    obs = pd.read_excel(scores_xlsx, sheet_name=obs_sheet.iloc[0]).set_index("obs_id")
    weights = None
    gw = scores_manifest.loc[scores_manifest["role"] == "global_weights"]
    if len(gw) == 1:
        weights = pd.read_excel(scores_xlsx, sheet_name=gw.iloc[0]["sheet"]).set_index("obs_id")
    modalities: Dict[str, ad.AnnData] = {}
    score_rows = scores_manifest.loc[scores_manifest["role"] == "scores"]
    for _, row in score_rows.iterrows():
        modality = row["modality"]
        score_df = pd.read_excel(scores_xlsx, sheet_name=row["sheet"]).set_index("obs_id")
        p_rows = scores_manifest.loc[(scores_manifest["role"] == "pvals") & (scores_manifest["modality"] == modality)]
        p_df = None
        if len(p_rows) == 1:
            p_df = pd.read_excel(scores_xlsx, sheet_name=p_rows.iloc[0]["sheet"]).set_index("obs_id")
        modalities[modality] = build_modality_adata(score_df, p_df, factor_metadata=obs)
    for modality in list(modalities.keys()):
        var_rows = features_manifest.loc[(features_manifest["role"] == "var") & (features_manifest["modality"] == modality)]
        if len(var_rows) == 1:
            var_df = pd.read_excel(features_xlsx, sheet_name=var_rows.iloc[0]["sheet"]).set_index("feature")
            modalities[modality].var = var_df.reindex(modalities[modality].var_names).copy()
        varm_rows = features_manifest.loc[(features_manifest["role"] == "varm") & (features_manifest["modality"] == modality)]
        for _, vrow in varm_rows.iterrows():
            key = vrow["key"]
            load_df = pd.read_excel(features_xlsx, sheet_name=vrow["sheet"]).set_index("gene")
            load_df = load_df.loc[:, modalities[modality].var_names]
            modalities[modality].varm[key] = as_csr(load_df.T.to_numpy(dtype=np.float32))
            modalities[modality].uns["gene_names"] = list(load_df.index.astype(str))
    ontology = mu.MuData(modalities)
    ontology.obs = obs.reindex(ontology.obs_names).copy()
    ontology.uns["factor_type"] = factor_type
    ontology.uns["cell_type_separator"] = cell_type_separator
    if weights is not None:
        ontology.obsm["weights"] = as_csr(weights.loc[ontology.obs_names].to_numpy(dtype=np.float32))
        ontology.uns["gene_names"] = list(weights.columns.astype(str))
    return ontology
