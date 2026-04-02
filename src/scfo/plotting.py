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
    """Plot the top positive and negative features for a factor.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    factor : str
        Factor to plot.
    modality : str
        Modality to query. Use ``"weights"`` to plot global factor weights.
    n_pos : int, default=10
        Number of top positive features to display.
    n_neg : int, default=10
        Number of top negative features to display.
    alpha : float, optional
        Optional p-value filter for modality features.
    figsize : tuple, default=(7, 6)
        Figure size.
    path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    tuple
        ``(fig, ax)`` matplotlib objects.
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
    """Plot the top genes or top factors for a modality feature.

    Parameters
    ----------
    ontology : muon.MuData
        Ontology object.
    modality : str
        Modality containing the feature of interest.
    feature : str
        Feature name within the modality.
    cell_type : str, optional
        Lineage to use when selecting lineage-specific loading matrices.
    what : {"genes", "factors"}, default="genes"
        Whether to plot genes contributing to the feature or factors enriched in the
        feature.
    n_pos : int, default=10
        Number of top positive items.
    n_neg : int, default=10
        Number of top negative items.
    alpha : float, optional
        Optional p-value filter when ``what='factors'``.
    figsize : tuple, default=(7, 6)
        Figure size.
    path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    tuple
        ``(fig, ax)`` matplotlib objects.
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
