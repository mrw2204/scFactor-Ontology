from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

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
    Create a matplotlib figure whose requested size refers only to the plotting panel.

    This helper is useful when consistent inner panel dimensions are desired
    across multiple figures, regardless of differences in axis labels, tick
    labels, or titles. The total figure size is calculated by adding the
    requested panel margins to the requested panel width and height, and the
    main axes are positioned explicitly with ``fig.add_axes``.

    Parameters
    ----------
    figsize : tuple of float
        Width and height of the main plotting panel, in inches.
    panel_margins : tuple of float, default=(2.8, 0.35, 0.45, 0.55)
        Margins around the panel, in inches, ordered as
        ``(left, right, bottom, top)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The main axes occupying the requested panel area.
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
    """
    Determine x-axis limits for a one-dimensional score vector.

    When explicit limits are not provided, the function computes limits from the
    observed score range, forces inclusion of zero, and adds symmetric padding
    proportional to the total span. This is primarily used for bar plots where
    both positive and negative values should be visually comparable around a
    central zero line.

    Parameters
    ----------
    scores : pandas.Series
        Numeric values used to determine the plotting range.
    xlim : tuple of float, optional
        Explicit x-axis limits. If provided, these are returned unchanged.
    xpad_frac : float, default=0.06
        Fraction of the score span added as padding on each side when limits are
        inferred automatically.

    Returns
    -------
    tuple of float
        Resolved ``(xmin, xmax)`` limits for plotting.

    Notes
    -----
    If all values are identical, a fallback span based on the absolute value of
    that constant score, or 1.0 if necessary, is used to avoid zero-width axes.
    """
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
    Add p-value labels near the zero line for horizontal bar plots.

    For each row in ``df_plot``, the function places a formatted p-value label
    just to the left or right of the zero line depending on the sign of the
    corresponding score. This keeps the p-value annotation visually anchored at
    the base of the bar rather than at the bar tip.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the horizontal bars.
    df_plot : pandas.DataFrame
        Plotting DataFrame containing at least a ``"score"`` column and, if
        available, a p-value column.
    y_positions : sequence
        Y coordinates corresponding to the rows of ``df_plot``.
    pval_col : str, default="pval"
        Column in ``df_plot`` containing p-values.
    pval_fmt : str, default="p={:.1e}"
        Format string used to render p-values.
    pval_fontsize : float, default=8
        Font size for p-value labels.
    pval_offset_frac : float, default=0.015
        Fraction of the current x-axis width used to offset labels from zero.

    Returns
    -------
    None

    Notes
    -----
    Positive bars receive labels just left of zero with right alignment, while
    negative bars receive labels just right of zero with left alignment.
    Rows with missing p-values are skipped.
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
    """
    Resolve a feature name to a unique column in a DataFrame.

    This helper supports several matching strategies to accommodate ontology
    modalities whose feature columns may be stored either as plain feature names
    or as lineage-qualified labels such as ``feature|cell_type``.

    Matching is attempted in the following order:

    1. direct membership in ``df.columns``
    2. exact string equality
    3. exact lineage-qualified match using ``cell_type`` if provided
    4. base-name match after splitting each column at the first ``"|"``

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose columns represent modality features.
    feature : str
        Requested feature name.
    cell_type : str, optional
        Optional lineage used to disambiguate feature names stored as
        ``feature|cell_type``.

    Returns
    -------
    str
        The resolved column name in ``df``.

    Raises
    ------
    KeyError
        If the feature cannot be found, or if multiple base-name matches are
        found and ``cell_type`` is insufficient to disambiguate.
    """
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
    """
    Plot the top positive and negative features for one factor within one modality.

    The function first retrieves ranked feature scores for the requested factor
    using ``top_features_for_factor``. It then combines the top negative and top
    positive features into a single horizontal bar plot, colors negative scores
    blue and positive scores red, draws a vertical zero line, and optionally
    annotates bars with p-values near their bases.

    Parameters
    ----------
    ontology : MuData-like
        Ontology object passed to ``top_features_for_factor``.
    factor : str
        Factor identifier to plot.
    modality : str
        Modality name to query.
    n_pos : int, default=10
        Number of highest-scoring positive features to include.
    n_neg : int, default=10
        Number of lowest-scoring negative features to include.
    alpha : float, optional
        Optional p-value threshold forwarded to ``top_features_for_factor``.
    figsize : tuple of float, default=(7, 6)
        Width and height of the plotting panel in inches.
    panel_margins : tuple of float, default=(2.8, 0.35, 0.45, 0.55)
        Margins around the panel in inches, ordered as
        ``(left, right, bottom, top)``.
    xlim : tuple of float, optional
        Explicit x-axis limits. If omitted, limits are inferred from the plotted
        scores and padded automatically.
    xpad_frac : float, default=0.06
        Fractional padding used when resolving x-axis limits automatically.
    pval_fontsize : float, default=8
        Font size for p-value labels.
    pval_fmt : str, default="p={:.1e}"
        Format string used for p-value annotations.
    pval_offset_frac : float, default=0.015
        Fraction of x-axis width used to offset p-value labels from zero.
    path : str, optional
        Optional output path for saving the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The bar-plot axes.

    Raises
    ------
    ValueError
        If no features are available for the requested factor and modality.
    """
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
    """
    Plot the top genes or top factors associated with a modality feature.

    This function supports two complementary views of a modality feature:

    - ``what="genes"``: rank genes by the feature-loading column for the
      requested feature
    - ``what="factors"``: rank ontology factors by their modality score for the
      requested feature

    In both modes, the function plots the top positive and top negative items as
    a horizontal bar chart and optionally annotates p-values when factor-level
    modality p-values are available.

    Parameters
    ----------
    ontology : MuData-like
        Ontology object.
    modality : str
        Name of the modality to query.
    feature : str
        Feature name to resolve within the modality.
    cell_type : str, optional
        Optional lineage used to filter or disambiguate lineage-qualified
        features and factor subsets.
    what : {"genes", "factors"}, default="genes"
        Type of item to rank and plot.
    n_pos : int, default=10
        Number of top positive items to include.
    n_neg : int, default=10
        Number of top negative items to include.
    alpha : float, optional
        Optional p-value threshold used only when ``what="factors"``.
    figsize : tuple of float, default=(7, 6)
        Width and height of the plotting panel in inches.
    panel_margins : tuple of float, default=(2.8, 0.35, 0.45, 0.55)
        Margins around the panel in inches, ordered as
        ``(left, right, bottom, top)``.
    xlim : tuple of float, optional
        Explicit x-axis limits.
    xpad_frac : float, default=0.06
        Fractional padding added when x-axis limits are inferred automatically.
    pval_fontsize : float, default=8
        Font size for p-value labels.
    pval_fmt : str, default="p={:.1e}"
        Format string used for p-value annotations.
    pval_offset_frac : float, default=0.015
        Fraction of x-axis width used to offset p-value labels from zero.
    path : str, optional
        Optional output path for saving the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The bar-plot axes.

    Raises
    ------
    ValueError
        If ``what`` is invalid or if no items remain after filtering.
    KeyError
        If the requested feature cannot be resolved to a unique modality column.
    """
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


def _matrix_to_dataframe(
    adata_like: Any,
    row_names: Optional[Sequence[str]] = None,
    col_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Convert an AnnData-like matrix container to a dense DataFrame.

    The function extracts ``adata_like.X``, converts it to a dense NumPy array,
    and wraps it in a pandas DataFrame using either explicitly supplied row and
    column names or, by default, ``adata_like.obs_names`` and
    ``adata_like.var_names``.

    Parameters
    ----------
    adata_like : Any
        Object with an ``X`` matrix and, optionally, ``obs_names`` and
        ``var_names`` attributes.
    row_names : sequence of str, optional
        Row labels to use instead of ``adata_like.obs_names``.
    col_names : sequence of str, optional
        Column labels to use instead of ``adata_like.var_names``.

    Returns
    -------
    pandas.DataFrame
        Dense DataFrame representation of the input matrix.
    """
    X = adata_like.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.asarray(X)

    if row_names is None:
        row_names = getattr(adata_like, "obs_names", None)
    if col_names is None:
        col_names = getattr(adata_like, "var_names", None)

    return pd.DataFrame(X, index=pd.Index(row_names), columns=pd.Index(col_names))


def _get_factor_names(
    ontology: Any,
    factor_name_key: str = "FactorName",
) -> pd.Index:
    """
    Retrieve factor names from ontology metadata.

    The function prefers the ``factor_name_key`` column in ``ontology.obs`` when
    present, and otherwise falls back to ``ontology.obs_names``.

    Parameters
    ----------
    ontology : Any
        Ontology-like object with ``obs`` and ``obs_names`` attributes.
    factor_name_key : str, default="FactorName"
        Column in ``ontology.obs`` containing factor names.

    Returns
    -------
    pandas.Index
        Index of factor names labeled with name ``"factor"``.
    """
    if factor_name_key in ontology.obs.columns:
        return pd.Index(ontology.obs[factor_name_key].astype(str), name="factor")
    return pd.Index(ontology.obs_names.astype(str), name="factor")


def _get_weights_df_from_ontology(
    ontology: Any,
    weights_modality: str = "weights",
    factor_name_key: str = "FactorName",
) -> pd.DataFrame:
    """
    Extract the ontology factor-weight matrix as a genes-by-factors DataFrame.

    The function reads the requested weights modality from ``ontology.mod``,
    resolves factor names from ``ontology.obs``, resolves gene names from
    ``weights.var_names`` or, if necessary, ``ontology.uns["gene_names"]``,
    converts the modality matrix to a DataFrame, and finally transposes it so
    that genes are rows and factors are columns.

    Parameters
    ----------
    ontology : MuData-like
        Ontology object containing modalities in ``.mod``.
    weights_modality : str, default="weights"
        Name of the modality containing factor-gene weights.
    factor_name_key : str, default="FactorName"
        Column in ``ontology.obs`` containing factor names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with genes as rows and factors as columns.

    Raises
    ------
    KeyError
        If ``weights_modality`` is not present in ``ontology.mod``.
    ValueError
        If gene names cannot be determined from the modality or ontology-level
        metadata.
    """
    if weights_modality not in ontology.mod:
        raise KeyError(
            f"weights_modality '{weights_modality}' not found in ontology.mod."
        )

    factor_names = _get_factor_names(ontology, factor_name_key=factor_name_key)
    w_mod = ontology.mod[weights_modality]

    if hasattr(w_mod, "var_names") and len(w_mod.var_names) > 0:
        gene_names = pd.Index(w_mod.var_names.astype(str), name="gene")
    elif "gene_names" in ontology.uns:
        gene_names = pd.Index(pd.Index(ontology.uns["gene_names"]).astype(str), name="gene")
    else:
        raise ValueError(
            "Could not determine gene names for the weights modality. "
            "Expected weights.var_names or ontology.uns['gene_names']."
        )

    weights_df = _matrix_to_dataframe(
        w_mod,
        row_names=factor_names,
        col_names=gene_names,
    )

    # Return genes x factors to match plotting logic
    return weights_df.T


def _get_modality_df_from_ontology(
    ontology: Any,
    modality: str,
    factor_name_key: str = "FactorName",
) -> pd.DataFrame:
    """
    Extract a modality matrix from an ontology object as a factor-by-feature DataFrame.

    Parameters
    ----------
    ontology : MuData-like
        Ontology object containing modalities in ``.mod``.
    modality : str
        Name of the modality to extract.
    factor_name_key : str, default="FactorName"
        Column in ``ontology.obs`` containing factor names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with factors as rows and modality features as columns.

    Raises
    ------
    KeyError
        If ``modality`` is not present in ``ontology.mod``.
    """
    if modality not in ontology.mod:
        raise KeyError(f"modality '{modality}' not found in ontology.mod.")

    factor_names = _get_factor_names(ontology, factor_name_key=factor_name_key)
    mod = ontology.mod[modality]

    return _matrix_to_dataframe(
        mod,
        row_names=factor_names,
        col_names=pd.Index(mod.var_names.astype(str), name="feature"),
    )


def plot_factor_effect_sizes(
    ontology: Any,
    modality: str,
    group: str,
    cell_type: str,
    *,
    weights_modality: str = "weights",
    cell_type_key: str = "Classification",
    factor_name_key: str = "FactorName",
    factors: Optional[Sequence[str]] = None,
    n_factors: int = 5,
    n_label: int = 10,
    sort_by_abs: bool = True,
    figsize: Tuple[float, float] = (5, 8),
    cmap: str = "coolwarm",
    scatter_xlim: Optional[Tuple[float, float]] = None,
    color_scale_fraction: float = 0.5,
    point_size: float = 50,
    point_alpha: float = 1.0,
    bar_alpha: float = 0.3,
    bar_color: str = "black",
    label_fontsize: float = 9,
    connector_color: str = "gray",
    connector_lw: float = 0.8,
    connector_alpha: float = 0.6,
    xlabel: str = "Gene weight",
    effect_xlabel: Optional[str] = None,
    ylabel: str = "Factor",
    title: Optional[str] = None,
    strip_cell_type_from_labels: bool = True,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    transparent: bool = False,
    rasterize_points: bool = False,
    show: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes], pd.Series]:
    """
    Plot factor gene weights together with modality-derived effect sizes for one lineage.

    This visualization combines two views of the selected ontology factors:

    1. a scatter plot of per-gene factor weights for each factor
    2. a mirrored horizontal bar plot showing the effect size of each factor for
       one modality feature or group

    Factors are first restricted to the requested ``cell_type`` using
    ``ontology.obs[cell_type_key]``. If explicit factors are not supplied, the
    top ``n_factors`` are selected from the requested effect-size vector,
    either by absolute magnitude or by raw signed value. For each selected
    factor, the function plots all gene weights, colors points by weight value,
    and annotates the top positive and top negative genes with connector lines.

    Parameters
    ----------
    ontology : MuData-like
        Ontology object containing factor metadata in ``.obs`` and modality data
        in ``.mod``.
    modality : str
        Name of the modality containing the effect-size vector to plot.
    group : str
        Column within the selected modality to use as the factor-level effect
        size, for example a treatment, condition, or comparison.
    cell_type : str
        Cell type or lineage whose factors should be displayed.
    weights_modality : str, default="weights"
        Name of the modality containing factor-gene weights.
    cell_type_key : str, default="Classification"
        Column in ``ontology.obs`` used to assign factors to lineages.
    factor_name_key : str, default="FactorName"
        Column in ``ontology.obs`` containing factor names.
    factors : sequence of str, optional
        Explicit factor names to plot. If omitted, factors are selected from the
        requested effect-size vector.
    n_factors : int, default=5
        Number of factors to select when ``factors`` is not provided.
    n_label : int, default=10
        Number of top positive and top negative genes to label for each factor.
    sort_by_abs : bool, default=True
        Whether automatic factor selection should use absolute effect size.
    figsize : tuple of float, default=(5, 8)
        Figure size in inches.
    cmap : str, default="coolwarm"
        Colormap used for gene-weight points.
    scatter_xlim : tuple of float, optional
        Explicit x-axis limits for the gene-weight scatter plot. If omitted,
        limits are inferred from the selected factors.
    color_scale_fraction : float, default=0.5
        Fraction of the maximum absolute selected weight used to define the
        symmetric color scale range.
    point_size : float, default=50
        Scatter point size.
    point_alpha : float, default=1.0
        Scatter point opacity.
    bar_alpha : float, default=0.3
        Opacity of the effect-size bars.
    bar_color : str, default="black"
        Color of the effect-size bars.
    label_fontsize : float, default=9
        Font size for gene labels.
    connector_color : str, default="gray"
        Color of connector lines linking gene labels to points.
    connector_lw : float, default=0.8
        Line width of connector lines.
    connector_alpha : float, default=0.6
        Opacity of connector lines.
    xlabel : str, default="Gene weight"
        X-axis label for the scatter axis.
    effect_xlabel : str, optional
        X-axis label for the effect-size bar axis. If omitted, defaults to
        ``"{modality}: {group}"``.
    ylabel : str, default="Factor"
        Y-axis label for the main scatter axis.
    title : str, optional
        Figure title. If omitted, a default title based on ``cell_type`` and
        ``group`` is used.
    strip_cell_type_from_labels : bool, default=True
        Whether to remove the trailing ``"|{cell_type}"`` suffix from factor
        labels on the y-axis.
    save : str or pathlib.Path, optional
        Optional output path for saving the figure.
    dpi : int, default=300
        Resolution used when saving raster formats.
    transparent : bool, default=False
        Whether to save with a transparent background.
    rasterize_points : bool, default=False
        Whether to rasterize the scatter points, which can help reduce file size
        in vector outputs.
    show : bool, default=True
        Whether to display the figure via ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : tuple of matplotlib.axes.Axes
        Tuple ``(ax, ax2)`` containing the gene-weight scatter axis and the
        effect-size bar axis.
    effect_sizes : pandas.Series
        Effect-size values used in the plot, indexed by the selected factor
        names.

    Raises
    ------
    KeyError
        If required ontology metadata columns are missing, if the requested
        modality or weights modality is absent, or if ``group`` is not a column
        in the selected modality.
    ValueError
        If no factors exist for the requested cell type, if no valid effect
        sizes remain after filtering, or if the selected factors have no usable
        weights.
    """
    if cell_type_key not in ontology.obs.columns:
        raise KeyError(
            f"cell_type_key '{cell_type_key}' not found in ontology.obs."
        )

    if factor_name_key not in ontology.obs.columns:
        raise KeyError(
            f"factor_name_key '{factor_name_key}' not found in ontology.obs."
        )

    weights_df = _get_weights_df_from_ontology(
        ontology=ontology,
        weights_modality=weights_modality,
        factor_name_key=factor_name_key,
    )
    modality_df = _get_modality_df_from_ontology(
        ontology=ontology,
        modality=modality,
        factor_name_key=factor_name_key,
    )

    if group not in modality_df.columns:
        raise KeyError(
            f"group '{group}' not found in modality '{modality}'. "
            f"Available columns include: {list(modality_df.columns[:10])}"
            + ("..." if modality_df.shape[1] > 10 else "")
        )

    factor_meta = ontology.obs[[factor_name_key, cell_type_key]].copy()
    factor_meta[factor_name_key] = factor_meta[factor_name_key].astype(str)
    factor_meta[cell_type_key] = factor_meta[cell_type_key].astype(str)

    cell_type_factors = factor_meta.loc[
        factor_meta[cell_type_key] == str(cell_type), factor_name_key
    ].tolist()

    if len(cell_type_factors) == 0:
        raise ValueError(f"No factors found for cell_type '{cell_type}'.")

    effect_sizes_all = modality_df[group].copy()
    effect_sizes_all.index = effect_sizes_all.index.astype(str)
    effect_sizes_all = effect_sizes_all.loc[
        effect_sizes_all.index.intersection(cell_type_factors)
    ].dropna()

    if effect_sizes_all.empty:
        raise ValueError(
            f"No non-null effect sizes found for cell_type '{cell_type}' "
            f"in modality '{modality}' and group '{group}'."
        )

    if factors is None:
        if sort_by_abs:
            selected_factors = (
                effect_sizes_all.abs()
                .sort_values(ascending=False)
                .head(n_factors)
                .index.tolist()
            )
        else:
            selected_factors = (
                effect_sizes_all.sort_values(ascending=False)
                .head(n_factors)
                .index.tolist()
            )
    else:
        selected_factors = [str(f) for f in factors]

    if len(selected_factors) == 0:
        raise ValueError("No factors selected for plotting.")

    missing_in_cell_type = [f for f in selected_factors if f not in cell_type_factors]
    if missing_in_cell_type:
        raise ValueError(
            f"The following requested factors are not assigned to cell_type "
            f"'{cell_type}': {missing_in_cell_type}"
        )

    missing_in_weights = [f for f in selected_factors if f not in weights_df.columns]
    if missing_in_weights:
        raise KeyError(
            f"The following factors are missing from weights modality "
            f"'{weights_modality}': {missing_in_weights}"
        )

    effect_sizes = effect_sizes_all.loc[selected_factors]
    weights_sub = weights_df.loc[:, selected_factors]

    arrays = [weights_sub[f].dropna().to_numpy() for f in selected_factors]
    arrays = [a for a in arrays if a.size > 0]
    if len(arrays) == 0:
        raise ValueError("Selected factors contain no non-null gene weights.")

    all_weights_sub = np.concatenate(arrays)
    max_abs_weight = np.nanmax(np.abs(all_weights_sub))
    if not np.isfinite(max_abs_weight) or max_abs_weight == 0:
        max_abs_weight = 1.0

    if scatter_xlim is None:
        pad = 0.05 * max_abs_weight
        scatter_xlim = (-(max_abs_weight + pad), max_abs_weight + pad)

    color_vmax = color_scale_fraction * max_abs_weight
    if not np.isfinite(color_vmax) or color_vmax <= 0:
        color_vmax = 1.0

    norm = colors.TwoSlopeNorm(vmin=-color_vmax, vcenter=0, vmax=color_vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)

    for i, factor in enumerate(selected_factors, start=1):
        s = weights_sub[factor].dropna()

        x = s.to_numpy()
        y = np.full(len(x), i)

        ax.scatter(
            x,
            y,
            c=x,
            cmap=cmap_obj,
            norm=norm,
            s=point_size,
            alpha=point_alpha,
            linewidths=0,
            rasterized=rasterize_points,
        )

        if n_label > 0 and len(s) > 0:
            top_pos = s.sort_values(ascending=False).head(min(n_label, len(s)))
            top_neg = s.sort_values(ascending=True).head(min(n_label, len(s)))

            pos_offsets = (
                np.linspace(-0.35, 0.35, len(top_pos))
                if len(top_pos) > 1 else np.array([0.0])
            )
            neg_offsets = (
                np.linspace(-0.35, 0.35, len(top_neg))
                if len(top_neg) > 1 else np.array([0.0])
            )

            for (gene, val), dy in zip(top_pos.items(), pos_offsets):
                ax.annotate(
                    str(gene),
                    xy=(val, i),
                    xytext=(
                        scatter_xlim[1] - 0.025 * (scatter_xlim[1] - scatter_xlim[0]),
                        i + dy,
                    ),
                    ha="right",
                    va="center",
                    fontsize=label_fontsize,
                    arrowprops=dict(
                        arrowstyle="-",
                        color=connector_color,
                        lw=connector_lw,
                        alpha=connector_alpha,
                    ),
                    clip_on=False,
                )

            for (gene, val), dy in zip(top_neg.items(), neg_offsets):
                ax.annotate(
                    str(gene),
                    xy=(val, i),
                    xytext=(
                        scatter_xlim[0] + 0.025 * (scatter_xlim[1] - scatter_xlim[0]),
                        i + dy,
                    ),
                    ha="left",
                    va="center",
                    fontsize=label_fontsize,
                    arrowprops=dict(
                        arrowstyle="-",
                        color=connector_color,
                        lw=connector_lw,
                        alpha=connector_alpha,
                    ),
                    clip_on=False,
                )

    if strip_cell_type_from_labels:
        suffix = f"|{cell_type}"
        yticklabels = [
            f[:-len(suffix)] if f.endswith(suffix) else f
            for f in selected_factors
        ]
    else:
        yticklabels = selected_factors

    ax.set_xlim(*scatter_xlim)
    ax.set_yticks(range(1, len(selected_factors) + 1))
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)

    ax2 = ax.twiny()
    ypos = np.arange(1, len(selected_factors) + 1)
    ax2.barh(
        ypos,
        effect_sizes.to_numpy(),
        height=0.45,
        alpha=bar_alpha,
        color=bar_color,
    )

    if effect_xlabel is None:
        effect_xlabel = f"{modality}: {group}"
    ax2.set_xlabel(effect_xlabel)

    max_abs_effect = np.nanmax(np.abs(effect_sizes.to_numpy()))
    if not np.isfinite(max_abs_effect) or max_abs_effect == 0:
        max_abs_effect = 1.0
    ax2.set_xlim(-1.2 * max_abs_effect, 1.2 * max_abs_effect)

    ax.invert_yaxis()

    if title is None:
        title = f"{cell_type} factors: {group}"
    ax.set_title(title)

    fig.tight_layout()

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save,
            bbox_inches="tight",
            pad_inches=0.25,
            dpi=dpi,
            transparent=transparent,
        )

    if show:
        plt.show()

    return fig, (ax, ax2), effect_sizes
