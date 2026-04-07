from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .core import (
    modality_feature_loadings_to_df,
    modality_pvals_to_df,
    modality_scores_to_df,
    top_features_for_factor,
)

plt.rcParams["svg.fonttype"] = "none"


def _make_fixed_panel_figure(
    figsize: Tuple[float, float],
    panel_margins: Tuple[float, float, float, float] = (2.8, 0.35, 0.45, 0.55),
):
    """
    Create a figure where `figsize` specifies the main plotting panel size only.

    Parameters
    ----------
    figsize
        Width, height of the plotting panel in inches.
    panel_margins
        Margins around the panel in inches: (left, right, bottom, top).
    """
    panel_w, panel_h = figsize
    left, right, bottom, top = panel_margins

    fig_w = panel_w + left + right
    fig_h = panel_h + bottom + top

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(
        [
            left / fig_w,
            bottom / fig_h,
            panel_w / fig_w,
            panel_h / fig_h,
        ]
    )
    return fig, ax


def _resolve_xlim(
    scores: pd.Series,
    xlim: Optional[Tuple[float, float]] = None,
    xpad_frac: float = 0.06,
) -> Tuple[float, float]:
    """Resolve x-limits, ensuring some padding and inclusion of zero by default."""
    if xlim is not None:
        return xlim

    scores = pd.Series(scores, dtype=float)
    xmin = float(scores.min()) if len(scores) else -1.0
    xmax = float(scores.max()) if len(scores) else 1.0

    xmin = min(xmin, 0.0)
    xmax = max(xmax, 0.0)

    span = xmax - xmin
    if span == 0:
        span = max(abs(xmin), abs(xmax), 1.0)

    pad = span * xpad_frac
    return xmin - pad, xmax + pad


def _add_pvalue_labels_at_bar_base(
    ax,
    df_plot: pd.DataFrame,
    y_positions,
    pval_col: str = "pval",
    pval_fmt: str = "p={:.1e}",
    pval_fontsize: float = 8,
    pval_offset_frac: float = 0.015,
):
    """
    Place p-value labels just across the zero line, at the base of each bar.

    Positive bars get labels just left of zero (right-justified).
    Negative bars get labels just right of zero (left-justified).
    """
    if pval_col not in df_plot.columns or not df_plot[pval_col].notna().any():
        return

    xmin, xmax = ax.get_xlim()
    dx = (xmax - xmin) * pval_offset_frac

    for y, (_, row) in zip(y_positions, df_plot.iterrows()):
        if pd.isna(row[pval_col]):
            continue

        score = float(row["score"])
        ptxt = pval_fmt.format(float(row[pval_col]))

        if score >= 0:
            x = 0 - dx
            ha = "right"
        else:
            x = 0 + dx
            ha = "left"

        ax.text(
            x,
            y,
            ptxt,
            va="center",
            ha=ha,
            fontsize=pval_fontsize,
            clip_on=False,
        )


def _resolve_feature_column(df: pd.DataFrame, feature: str, cell_type: Optional[str] = None) -> str:
    """Resolve a feature column name, allowing optional lineage-suffixed matches."""
    if feature in df.columns:
        return str(feature)

    exact = [c for c in df.columns if str(c) == str(feature)]
    if len(exact) == 1:
        return exact[0]

    if cell_type is not None:
        tagged = [c for c in df.columns if str(c) == f"{feature}|{cell_type}"]
        if len(tagged) == 1:
            return tagged[0]

    base_matches = [c for c in df.columns if str(c).split("|", 1)[0] == str(feature)]
    if len(base_matches) == 1:
        return base_matches[0]
    if len(base_matches) > 1:
        raise KeyError(
            f"Feature '{feature}' matched multiple columns: {base_matches}. Provide cell_type to disambiguate."
        )
    raise KeyError(f"Feature '{feature}' not found.")


def plot_factor_top_features(
    ontology,
    factor: str,
    modality: str,
    n_pos: int = 10,
    n_neg: int = 10,
    alpha: Optional[float] = None,
    figsize: Tuple[float, float] = (7, 6),
    panel_margins: Tuple[float, float, float, float] = (2.8, 0.35, 0.45, 0.55),
    xlim: Optional[Tuple[float, float]] = None,
    xpad_frac: float = 0.06,
    pval_fontsize: float = 8,
    pval_fmt: str = "p={:.1e}",
    pval_offset_frac: float = 0.015,
    path: Optional[str] = None,
):
    """Plot the top positive and negative features for a factor."""
    pos, neg = top_features_for_factor(
        ontology,
        factor,
        modality,
        n_pos=n_pos,
        n_neg=n_neg,
        alpha=alpha,
    )
    df = pd.concat([neg, pos], axis=0).copy()
    if df.empty:
        raise ValueError(f"No features available for factor={factor}, modality={modality}.")

    df["label"] = df["feature"].astype(str)
    df = df.sort_values("score").reset_index(drop=True)

    fig, ax = _make_fixed_panel_figure(figsize=figsize, panel_margins=panel_margins)

    y = list(range(len(df)))
    colors = ["#4575b4" if value < 0 else "#d73027" for value in df["score"]]
    ax.barh(y, df["score"], color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_ylabel("")
    ax.set_xlabel("Feature score")

    resolved_xlim = _resolve_xlim(df["score"], xlim=xlim, xpad_frac=xpad_frac)
    ax.set_xlim(*resolved_xlim)
    ax.axvline(0, color="k", lw=1)

    title = f"{factor} | {modality}"
    if alpha is not None:
        title += f" | p < {alpha}"
    ax.set_title(title)

    _add_pvalue_labels_at_bar_base(
        ax,
        df_plot=df,
        y_positions=y,
        pval_col="pval",
        pval_fmt=pval_fmt,
        pval_fontsize=pval_fontsize,
        pval_offset_frac=pval_offset_frac,
    )

    if path is not None:
        fig.savefig(path)
    return fig, ax


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
    panel_margins: Tuple[float, float, float, float] = (2.8, 0.35, 0.45, 0.55),
    xlim: Optional[Tuple[float, float]] = None,
    xpad_frac: float = 0.06,
    pval_fontsize: float = 8,
    pval_fmt: str = "p={:.1e}",
    pval_offset_frac: float = 0.015,
    path: Optional[str] = None,
):
    """Plot the top genes or top factors for a modality feature."""
    if what not in {"genes", "factors"}:
        raise ValueError("what must be 'genes' or 'factors'.")

    if what == "genes":
        df = modality_feature_loadings_to_df(
            ontology,
            modality,
            cell_types=cell_type,
            strip_cell_type_suffix=True,
        )
        col = _resolve_feature_column(df, feature, cell_type=cell_type)
        row = pd.DataFrame({"feature": df.index.astype(str), "score": df[col].values})
        row["pval"] = pd.NA
        label = col
    else:
        scores = modality_scores_to_df(ontology, modality, cell_types=cell_type)
        col = _resolve_feature_column(scores, feature, cell_type=cell_type)
        row = pd.DataFrame({"feature": scores.index.astype(str), "score": scores[col].values})
        pvals = modality_pvals_to_df(ontology, modality, cell_types=cell_type)
        row["pval"] = pvals[col].values if pvals is not None and col in pvals.columns else pd.NA
        if alpha is not None and row["pval"].notna().any():
            row = row.loc[row["pval"] < alpha].copy()
        label = col

    pos = row.sort_values("score", ascending=False).head(n_pos).reset_index(drop=True)
    neg = row.sort_values("score", ascending=True).head(n_neg).reset_index(drop=True)
    df_plot = pd.concat([neg, pos], axis=0).copy()

    if df_plot.empty:
        raise ValueError(
            f"No items available for modality={modality}, feature={feature}, cell_type={cell_type}, what={what}."
        )

    df_plot["label"] = df_plot["feature"].astype(str)
    df_plot = df_plot.sort_values("score").reset_index(drop=True)

    fig, ax = _make_fixed_panel_figure(figsize=figsize, panel_margins=panel_margins)

    y = list(range(len(df_plot)))
    colors = ["#4575b4" if value < 0 else "#d73027" for value in df_plot["score"]]
    ax.barh(y, df_plot["score"], color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["label"])
    ax.set_ylabel("")
    ax.set_xlabel("Feature score")

    resolved_xlim = _resolve_xlim(df_plot["score"], xlim=xlim, xpad_frac=xpad_frac)
    ax.set_xlim(*resolved_xlim)
    ax.axvline(0, color="k", lw=1)

    title = f"{modality} | {label}"
    title += " | factors" if what == "factors" else " | genes"
    if alpha is not None and what == "factors":
        title += f" | p < {alpha}"
    ax.set_title(title)

    _add_pvalue_labels_at_bar_base(
        ax,
        df_plot=df_plot,
        y_positions=y,
        pval_col="pval",
        pval_fmt=pval_fmt,
        pval_fontsize=pval_fontsize,
        pval_offset_frac=pval_offset_frac,
    )

    if path is not None:
        fig.savefig(path)
    return fig, ax
