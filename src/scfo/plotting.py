from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .core import top_features_for_factor

plt.rcParams["svg.fonttype"] = "none"


def plot_factor_top_features(
    ontology,
    factor: str,
    modality: str,
    n_pos: int = 10,
    n_neg: int = 10,
    alpha: Optional[float] = None,
    figsize: Tuple[float, float] = (7, 6),
    path: Optional[str] = None,
):
    """
    Horizontal bar plot of the top positive and negative features for a factor
    within a given ontology modality, or from the global factor weight matrix
    when ``modality="weights"``.
    """
    pos, neg = top_features_for_factor(
        ontology, factor, modality, n_pos=n_pos, n_neg=n_neg, alpha=alpha
    )
    df = pd.concat([neg, pos], axis=0).copy()

    if df.empty:
        raise ValueError(f"No features available for factor={factor}, modality={modality}")

    df["label"] = df["feature"]
    df = df.sort_values("score")

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4575b4" if v < 0 else "#d73027" for v in df["score"]]
    ax.barh(df["label"], df["score"], color=colors)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Feature score")
    ax.set_ylabel("")
    title = f"{factor} | {modality}"
    if alpha is not None:
        title += f" | p < {alpha}"
    ax.set_title(title)

    if "pval" in df.columns and df["pval"].notna().any():
        for i, (_, row) in enumerate(df.iterrows()):
            if pd.notna(row["pval"]):
                ax.text(
                    row["score"],
                    i,
                    f"  p={row['pval']:.1e}",
                    va="center",
                    ha="left" if row["score"] >= 0 else "right",
                    fontsize=8,
                )

    plt.tight_layout()
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig, ax



def _resolve_feature_column(df: pd.DataFrame, feature: str, cell_type: Optional[str] = None) -> str:
    """Resolve a feature column name, allowing optional lineage-suffixed matches."""
    if feature in df.columns:
        return feature
    matches = [c for c in df.columns if str(c) == str(feature)]
    if len(matches) == 1:
        return matches[0]
    if cell_type is not None:
        matches = [c for c in df.columns if str(c) == f"{feature}|{cell_type}"]
        if len(matches) == 1:
            return matches[0]
    base_matches = [c for c in df.columns if str(c).split("|", 1)[0] == str(feature)]
    if len(base_matches) == 1:
        return base_matches[0]
    if len(base_matches) > 1:
        raise KeyError(
            f"Feature '{feature}' matched multiple columns: {base_matches}. "
            "Provide cell_type to disambiguate."
        )
    raise KeyError(f"Feature '{feature}' not found.")



def plot_modality_feature_top_items(
    ontology,
    modality: str,
    feature: str,
    cell_type: Optional[str] = None,
    what: str = "genes",
    n_pos: int = 10,
    n_neg: int = 10,
    alpha: Optional[float] = None,
    figsize: Tuple[float, float] = (7, 6),
    path: Optional[str] = None,
):
    """
    Plot top genes or top factors for a given modality feature and cell type.

    Parameters
    ----------
    ontology
        Factor-centric ontology MuData.
    modality
        Modality name, e.g. ``'regulons'`` or ``'liana_ligand'``.
    feature
        Feature/context/regulon name within the modality.
    cell_type
        Optional ontology lineage to restrict both the feature loadings and factor
        score view. Recommended for lineage-specific modalities.
    what
        ``'genes'`` to plot the top positive/negative genes making up the feature
        loadings, or ``'factors'`` to plot the top positive/negative factors in
        which the feature is enriched.
    n_pos, n_neg
        Number of positive / negative items to show.
    alpha
        Optional p-value threshold. Only applies when ``what='factors'`` and the
        modality stores p-values.
    """
    if what not in {"genes", "factors"}:
        raise ValueError("what must be 'genes' or 'factors'.")

    if what == "genes":
        from .core import modality_feature_loadings_to_df

        df = modality_feature_loadings_to_df(ontology, modality, cell_types=cell_type)
        col = _resolve_feature_column(df, feature, cell_type=cell_type)
        row = pd.DataFrame({"feature": df.index.astype(str), "score": df[col].values})
        row["pval"] = pd.NA
        label = col
    else:
        from .core import modality_scores_to_df, modality_pvals_to_df

        scores = modality_scores_to_df(ontology, modality, cell_types=cell_type)
        col = _resolve_feature_column(scores, feature, cell_type=cell_type)
        row = pd.DataFrame({"feature": scores.index.astype(str), "score": scores[col].values})
        pvals = modality_pvals_to_df(ontology, modality, cell_types=cell_type)
        row["pval"] = pvals[col].values if pvals is not None and col in pvals.columns else pd.NA
        if alpha is not None and pd.notna(row["pval"]).any():
            row = row.loc[row["pval"] < alpha].copy()
        label = col

    pos = row.sort_values("score", ascending=False).head(n_pos).reset_index(drop=True)
    neg = row.sort_values("score", ascending=True).head(n_neg).reset_index(drop=True)
    df_plot = pd.concat([neg, pos], axis=0).copy()

    if df_plot.empty:
        raise ValueError(
            f"No items available for modality={modality}, feature={feature}, cell_type={cell_type}, what={what}."
        )

    df_plot["label"] = df_plot["feature"]
    df_plot = df_plot.sort_values("score")

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4575b4" if v < 0 else "#d73027" for v in df_plot["score"]]
    ax.barh(df_plot["label"], df_plot["score"], color=colors)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Feature score")
    ax.set_ylabel("")
    title = f"{modality} | {label}"
    if cell_type is not None:
        title += f" | {cell_type}"
    title += f" | top {what}"
    if alpha is not None and what == "factors":
        title += f" | p < {alpha}"
    ax.set_title(title)

    if "pval" in df_plot.columns and pd.notna(df_plot["pval"]).any():
        for i, (_, row_) in enumerate(df_plot.iterrows()):
            if pd.notna(row_["pval"]):
                ax.text(
                    row_["score"],
                    i,
                    f"  p={row_['pval']:.1e}",
                    va="center",
                    ha="left" if row_["score"] >= 0 else "right",
                    fontsize=8,
                )

    plt.tight_layout()
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig, ax
