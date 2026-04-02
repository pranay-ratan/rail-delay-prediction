"""
visualizations.py
=================
Publication-quality static (matplotlib/seaborn) and interactive (Plotly)
visualizations for the Canadian Rail Incident & Delay Prediction project.

All figures saved to outputs/figures/.

Author : Pranay Ratan | SFU Data Science
Project: Canadian Rail Incident & Delay Prediction
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import linregress
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
FIGURES_DIR: Path = Path(__file__).resolve().parent.parent / "outputs" / "figures"

CPKC_RED: str = "#C8102E"
CPKC_GREY: str = "#4A4A4A"
CPKC_LIGHT: str = "#F5F5F5"
PALETTE_MULTI: list[str] = [
    "#C8102E", "#1A3A5C", "#2E8B57", "#D4A017",
    "#6A3D9A", "#FF6B35", "#00B4D8", "#8B0000"
]

SEASON_ORDER: list[str] = ["Winter", "Spring", "Summer", "Fall"]
DPI: int = 150
FIG_SIZE: tuple[int, int] = (14, 7)

# Publication-quality style
plt.rcParams.update({
    "figure.dpi": DPI,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
})


def _save(fig: plt.Figure, filename: str) -> Path:
    """Save a matplotlib figure to the figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved figure → %s", path)
    return path


# ---------------------------------------------------------------------------
# CHART 1 — Incidents per year (line + trend)
# ---------------------------------------------------------------------------

def plot_incidents_per_year(df: pd.DataFrame) -> Path:
    """
    Line chart: total rail incidents per year (2000–present) with OLS trend.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched incident DataFrame with 'year' column.

    Returns
    -------
    Path
        Saved figure path.
    """
    annual = df.groupby("year").size().reset_index(name="incidents")
    annual = annual[annual["year"].between(2000, 2025)]

    slope, intercept, *_ = linregress(annual["year"], annual["incidents"])
    trend = intercept + slope * annual["year"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.fill_between(annual["year"], annual["incidents"], alpha=0.15, color=CPKC_RED)
    ax.plot(annual["year"], annual["incidents"], color=CPKC_RED,
            linewidth=2.5, marker="o", markersize=5, label="Annual Incidents")
    ax.plot(annual["year"], trend, color=CPKC_GREY,
            linewidth=2, linestyle="--", label=f"Trend (slope={slope:+.1f}/yr)")

    ax.set_title("Canadian Rail Incidents Per Year (2000–2024)", pad=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Incidents")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for yr, cnt in zip(annual["year"], annual["incidents"]):
        if yr % 5 == 0:
            ax.annotate(str(cnt), (yr, cnt), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color=CPKC_GREY)

    return _save(fig, "01_incidents_per_year.png")


# ---------------------------------------------------------------------------
# CHART 2 — Province × Incident Type Heatmap
# ---------------------------------------------------------------------------

def plot_province_type_heatmap(df: pd.DataFrame) -> Path:
    """
    Heatmap: incident frequency by province and incident type.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Path
    """
    pivot = df.pivot_table(
        index="province", columns="incident_type",
        values="year", aggfunc="count", fill_value=0
    )
    # Normalize per row for easier visual comparison
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        pivot_norm, annot=pivot.values, fmt="d",
        cmap="RdYlGn_r", linewidths=0.5, linecolor="#E0E0E0",
        ax=ax, cbar_kws={"label": "Proportion of Province Incidents"},
        annot_kws={"size": 8},
    )
    ax.set_title("Rail Incident Frequency by Province & Type\n(Count annotated, colour = row proportion)",
                 pad=15)
    ax.set_xlabel("Incident Type")
    ax.set_ylabel("Province")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    return _save(fig, "02_province_type_heatmap.png")


# ---------------------------------------------------------------------------
# CHART 3 — Top 10 provinces bar chart
# ---------------------------------------------------------------------------

def plot_top_provinces(df: pd.DataFrame) -> Path:
    """
    Horizontal bar chart: top 10 provinces/regions by incident count.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Path
    """
    top = df["province"].value_counts().head(10).reset_index()
    top.columns = ["province", "incidents"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top["province"][::-1], top["incidents"][::-1],
                   color=CPKC_RED, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, top["incidents"][::-1]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=10, color=CPKC_GREY)

    ax.set_title("Top 10 Provinces by Total Rail Incidents", pad=15)
    ax.set_xlabel("Total Incidents")
    ax.set_ylabel("")
    ax.set_xlim(0, top["incidents"].max() * 1.15)
    return _save(fig, "03_top_provinces.png")


# ---------------------------------------------------------------------------
# CHART 4 — Seasonal decomposition
# ---------------------------------------------------------------------------

def plot_seasonal_decomposition(df: pd.DataFrame) -> Path:
    """
    Seasonal decomposition of monthly rail incident counts.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Path
    """
    monthly = (
        df.groupby(["year", "month"]).size()
        .reset_index(name="incidents")
        .sort_values(["year", "month"])
    )
    ts = pd.Series(
        monthly["incidents"].values,
        index=pd.to_datetime(
            monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01"
        )
    )
    ts = ts[ts.index >= "2000-01-01"]
    ts = ts.asfreq("MS").fillna(ts.mean())

    if len(ts) < 24:
        logger.warning("Not enough data for seasonal decomposition — skipping.")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Insufficient data for decomposition",
                ha="center", va="center", transform=ax.transAxes)
        return _save(fig, "04_seasonal_decomposition.png")

    result = seasonal_decompose(ts, model="additive", period=12, extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    components = [
        (ts, "Observed", CPKC_RED),
        (result.trend, "Trend", "#1A3A5C"),
        (result.seasonal, "Seasonal", "#2E8B57"),
        (result.resid, "Residual", CPKC_GREY),
    ]
    for ax, (data, label, color) in zip(axes, components):
        ax.plot(data, color=color, linewidth=1.5)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.5)

    axes[0].set_title("Seasonal Decomposition of Monthly Rail Incidents", pad=15)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    return _save(fig, "04_seasonal_decomposition.png")


# ---------------------------------------------------------------------------
# CHART 5 — Correlation matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, feature_cols: list[str]) -> Path:
    """
    Correlation heatmap of all engineered numeric features.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list of str
        Numeric feature column names to include.

    Returns
    -------
    Path
    """
    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f",
        linewidths=0.5, linecolor="#E0E0E0",
        ax=ax, annot_kws={"size": 8},
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    return _save(fig, "05_correlation_matrix.png")


# ---------------------------------------------------------------------------
# CHART 6 — Severity score distribution
# ---------------------------------------------------------------------------

def plot_severity_distribution(df: pd.DataFrame) -> Path:
    """
    Bar chart: distribution of incident severity (HIGH vs LOW).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Path
    """
    counts = df["incident_severity"].value_counts().sort_index()
    labels = ["LOW Risk", "HIGH Risk"]
    colors = ["#2E8B57", CPKC_RED]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)],
                  color=colors, edgecolor="white", width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{bar.get_height():,}",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=CPKC_GREY)

    total = counts.sum()
    ax.set_title("Incident Severity Class Distribution", pad=15)
    ax.set_ylabel("Number of Incidents")
    ax.set_ylim(0, max(counts.values) * 1.2)
    pct = counts.get(1, 0) / total * 100
    ax.text(0.98, 0.95, f"HIGH: {pct:.1f}% of all incidents",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=CPKC_RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor=CPKC_RED))
    return _save(fig, "06_severity_distribution.png")


# ---------------------------------------------------------------------------
# CHART 7 — Rolling 12m incidents by province (multi-line)
# ---------------------------------------------------------------------------

def plot_rolling_incidents_by_province(df: pd.DataFrame, top_n: int = 6) -> Path:
    """
    Multi-line chart: rolling 12-month incident trend for top N provinces.

    Parameters
    ----------
    df : pd.DataFrame
    top_n : int
        Number of provinces to show.

    Returns
    -------
    Path
    """
    top_provinces = df["province"].value_counts().head(top_n).index.tolist()
    df_top = df[df["province"].isin(top_provinces)].copy()

    monthly = (
        df_top.groupby(["province", "year", "month"])
        .size().reset_index(name="incidents")
        .sort_values(["province", "year", "month"])
    )
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01"
    )
    monthly = monthly.set_index("date")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for i, prov in enumerate(top_provinces):
        ts = monthly[monthly["province"] == prov]["incidents"]
        ts = ts.resample("MS").sum().rolling(12, min_periods=1).mean()
        ax.plot(ts, linewidth=2, color=PALETTE_MULTI[i % len(PALETTE_MULTI)],
                label=prov, alpha=0.9)

    ax.set_title(f"Rolling 12-Month Rail Incidents — Top {top_n} Provinces", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling 12-Month Incident Count")
    ax.legend(loc="upper right", ncol=2)
    return _save(fig, "07_rolling_incidents_by_province.png")


# ---------------------------------------------------------------------------
# CHART 8 — ROC curves (all models)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
) -> Path:
    """
    Combined ROC curve plot for all models.

    Parameters
    ----------
    roc_data : dict
        model_name → (fpr, tpr, auc) from models.compute_roc_curves.

    Returns
    -------
    Path
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, linewidth=2.5, color=PALETTE_MULTI[i],
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
    ax.set_title("ROC Curves — All Models Compared", pad=15)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return _save(fig, "08_roc_curves.png")


# ---------------------------------------------------------------------------
# CHART 9 — Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray, model_name: str
) -> Path:
    """
    Heatmap confusion matrix for the best model.

    Parameters
    ----------
    cm : np.ndarray
        2×2 confusion matrix.
    model_name : str
        Label for title.

    Returns
    -------
    Path
    """
    labels = ["LOW Risk", "HIGH Risk"]
    annot = np.array([
        [f"TN\n{cm[0,0]:,}", f"FP\n{cm[0,1]:,}"],
        [f"FN\n{cm[1,0]:,}", f"TP\n{cm[1,1]:,}"],
    ])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=annot, fmt="s", cmap="Reds",
                xticklabels=labels, yticklabels=labels,
                linewidths=2, linecolor="white", ax=ax,
                cbar_kws={"label": "Count"})
    ax.set_title(f"Confusion Matrix — {model_name}", pad=15)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return _save(fig, "09_confusion_matrix.png")


# ---------------------------------------------------------------------------
# CHART 10 — Feature importance (RF + XGB side by side)
# ---------------------------------------------------------------------------

def plot_feature_importance_comparison(
    imp_rf: pd.DataFrame,
    imp_xgb: pd.DataFrame,
) -> Path:
    """
    Side-by-side horizontal bar charts of Random Forest and XGBoost feature importances.

    Parameters
    ----------
    imp_rf : pd.DataFrame
        Feature importance from Random Forest.
    imp_xgb : pd.DataFrame
        Feature importance from XGBoost.

    Returns
    -------
    Path
    """
    top_n = 15
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, df_imp, color, label in [
        (axes[0], imp_rf.head(top_n), "#1A3A5C", "Random Forest"),
        (axes[1], imp_xgb.head(top_n), CPKC_RED, "XGBoost"),
    ]:
        df_imp = df_imp.sort_values("importance")
        ax.barh(df_imp["feature"], df_imp["importance"],
                color=color, edgecolor="white", alpha=0.9)
        ax.set_title(f"{label} — Top {top_n} Features", pad=12)
        ax.set_xlabel("Importance Score")
        for val, feat in zip(df_imp["importance"], df_imp["feature"]):
            ax.text(val + 0.001, feat, f"{val:.4f}",
                    va="center", ha="left", fontsize=8, color=CPKC_GREY)

    fig.suptitle("Feature Importance Comparison: Random Forest vs XGBoost", fontsize=15,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "10_feature_importance_comparison.png")


# ---------------------------------------------------------------------------
# PLOTLY — Interactive versions (for Streamlit embedding)
# ---------------------------------------------------------------------------

def plotly_incidents_per_year(df: pd.DataFrame) -> go.Figure:
    """
    Interactive Plotly line chart: incidents per year with trend.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    go.Figure
    """
    annual = df.groupby("year").size().reset_index(name="incidents")
    annual = annual[annual["year"].between(2000, 2025)]

    slope, intercept, *_ = linregress(annual["year"], annual["incidents"])
    trend = intercept + slope * annual["year"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual["year"], y=annual["incidents"],
        mode="lines+markers", name="Annual Incidents",
        line=dict(color=CPKC_RED, width=3),
        marker=dict(size=7, color=CPKC_RED),
        fill="tozeroy", fillcolor="rgba(200,16,46,0.1)",
        hovertemplate="<b>%{x}</b>: %{y} incidents<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=annual["year"], y=trend,
        mode="lines", name=f"Trend ({slope:+.1f}/yr)",
        line=dict(color=CPKC_GREY, width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Canadian Rail Incidents Per Year (2000–2024)",
        xaxis_title="Year", yaxis_title="Total Incidents",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        font=dict(family="Arial", size=13),
    )
    return fig


def plotly_province_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Interactive Plotly heatmap: incidents by province and type.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    go.Figure
    """
    pivot = df.pivot_table(
        index="province", columns="incident_type",
        values="year", aggfunc="count", fill_value=0
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        title="Rail Incident Frequency by Province & Type",
        labels=dict(x="Incident Type", y="Province", color="Count"),
        text_auto=True,
        aspect="auto",
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12),
        coloraxis_colorbar=dict(title="Count"),
    )
    return fig


def plotly_rolling_by_province(df: pd.DataFrame, top_n: int = 6) -> go.Figure:
    """
    Interactive Plotly multi-line chart: rolling 12-month incidents by province.

    Parameters
    ----------
    df : pd.DataFrame
    top_n : int
        Number of top provinces.

    Returns
    -------
    go.Figure
    """
    top_provinces = df["province"].value_counts().head(top_n).index.tolist()
    df_top = df[df["province"].isin(top_provinces)].copy()

    monthly = (
        df_top.groupby(["province", "year", "month"])
        .size().reset_index(name="incidents")
        .sort_values(["province", "year", "month"])
    )
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    fig = go.Figure()
    for i, prov in enumerate(top_provinces):
        ts = monthly[monthly["province"] == prov].set_index("date")["incidents"]
        ts = ts.resample("MS").sum().rolling(12, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.values,
            mode="lines", name=prov,
            line=dict(color=PALETTE_MULTI[i % len(PALETTE_MULTI)], width=2.5),
            hovertemplate=f"<b>{prov}</b><br>%{{x|%b %Y}}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Rolling 12-Month Incidents — Top {top_n} Provinces",
        xaxis_title="Date", yaxis_title="Rolling 12-Month Count",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Arial", size=13),
    )
    return fig


def plotly_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
) -> go.Figure:
    """
    Interactive Plotly ROC curve chart.

    Parameters
    ----------
    roc_data : dict
        model_name → (fpr, tpr, auc).

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name}  (AUC={auc:.4f})",
            line=dict(color=PALETTE_MULTI[i % len(PALETTE_MULTI)], width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="grey", dash="dash", width=1),
        name="Random Classifier", showlegend=True,
    ))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        shapes=[dict(type="rect", x0=0, y0=0, x1=1, y1=1,
                     line=dict(color="#E0E0E0"))],
        font=dict(family="Arial", size=13),
    )
    return fig


def plotly_feature_importance(imp_df: pd.DataFrame, title: str = "Feature Importance") -> go.Figure:
    """
    Interactive Plotly horizontal bar chart of feature importances.

    Parameters
    ----------
    imp_df : pd.DataFrame
        DataFrame with columns: feature, importance.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    df_plot = imp_df.sort_values("importance").tail(15)
    fig = go.Figure(go.Bar(
        x=df_plot["importance"], y=df_plot["feature"],
        orientation="h",
        marker_color=CPKC_RED,
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template="plotly_white",
        font=dict(family="Arial", size=12),
        bargap=0.3,
    )
    return fig


def plotly_gauge(probability: float, label: str = "Incident Risk Probability") -> go.Figure:
    """
    Plotly gauge chart for live predictor risk probability.

    Parameters
    ----------
    probability : float
        Predicted probability 0.0–1.0.
    label : str
        Chart label.

    Returns
    -------
    go.Figure
    """
    pct = round(probability * 100, 1)
    if probability < 0.4:
        bar_color = "#2E8B57"
        risk_text = "LOW"
    elif probability < 0.7:
        bar_color = "#D4A017"
        risk_text = "MEDIUM"
    else:
        bar_color = CPKC_RED
        risk_text = "HIGH"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"{label}<br><span style='font-size:1.2em;color:{bar_color}'><b>{risk_text} RISK</b></span>"},
        delta={"reference": 50, "increasing": {"color": CPKC_RED}},
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#E0E0E0",
            "steps": [
                {"range": [0, 40], "color": "rgba(46,139,87,0.15)"},
                {"range": [40, 70], "color": "rgba(212,160,23,0.15)"},
                {"range": [70, 100], "color": "rgba(200,16,46,0.15)"},
            ],
            "threshold": {
                "line": {"color": CPKC_GREY, "width": 4},
                "thickness": 0.75,
                "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=320,
        font=dict(family="Arial", size=14),
        paper_bgcolor="white",
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig
