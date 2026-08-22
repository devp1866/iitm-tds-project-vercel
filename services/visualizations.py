"""
services/visualizations.py
Smart Plotly chart generation for Autolysis v2.
Auto-detects data types and generates appropriate charts.
Returns Plotly JSON dicts (not images).
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import combinations
import structlog

logger = structlog.get_logger()

# Brand color palette
PALETTE = {
    "peach": "#FFDBBB",
    "sky": "#BADFFF",
    "mint": "#BAFFF5",
    "slate": "#496580",
    "bg_deep": "#0d1e2c",
    "bg_mid": "#162535",
    "text": "#e8f4f8",
    "text_muted": "#8fb3c8",
    "anomaly": "#FF6B6B",
    "grid": "rgba(73,101,128,0.3)",
}

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(22,37,53,0.0)",
    plot_bgcolor="rgba(22,37,53,0.0)",
    font=dict(family="Inter, sans-serif", color=PALETTE["text"], size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"]),
    yaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"]),
)

COLOR_SEQ = [PALETTE["sky"], PALETTE["mint"], PALETTE["peach"], "#a8d8ea", "#b8e4bc", "#ffc6a5"]


def generate_smart_charts(df: pd.DataFrame, anomaly_data: dict = None) -> list[dict]:
    """
    Generate a list of Plotly chart JSON dicts based on data characteristics.
    Each dict: {"title": str, "type": str, "figure": plotly_json_dict}
    """
    charts = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if df[c].nunique() <= 30
    ]
    datetime_cols = _detect_datetime_cols(df)

    # 1. Correlation Heatmap (always if ≥2 numeric cols)
    if len(numeric_cols) >= 2:
        charts.append(_correlation_heatmap(df, numeric_cols))

    # 2. Distribution histograms (up to 4 numeric cols)
    for col in numeric_cols[:4]:
        charts.append(_distribution_chart(df, col))

    # 3. Time series (if datetime detected)
    if datetime_cols:
        for dt_col in datetime_cols[:1]:
            for num_col in numeric_cols[:3]:
                charts.append(_time_series_chart(df, dt_col, num_col))

    # 4. Category bar charts
    for col in cat_cols[:2]:
        charts.append(_category_bar_chart(df, col))

    # 5. Scatter plot for top-correlated pair
    if len(numeric_cols) >= 2:
        top_pair = _find_top_correlated_pair(df, numeric_cols)
        if top_pair:
            charts.append(_scatter_chart(df, top_pair[0], top_pair[1]))

    # 6. Box plots for numeric cols (up to 5)
    if len(numeric_cols) >= 1:
        charts.append(_box_plot(df, numeric_cols[:5]))

    # 7. Anomaly scatter (if anomaly data provided)
    if anomaly_data and "anomaly_indices" in anomaly_data and len(numeric_cols) >= 2:
        charts.append(_anomaly_scatter(df, numeric_cols, anomaly_data))

    # 8. Cluster scatter (if ≥2 numeric cols, use first 2)
    if len(numeric_cols) >= 2:
        charts.append(_cluster_scatter(df, numeric_cols[:2]))

    # Filter out None results
    charts = [c for c in charts if c is not None]
    logger.info("charts_generated", count=len(charts))
    return charts


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL CHART GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> dict:
    try:
        corr = df[numeric_cols].corr().round(3)
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            text=corr.values.round(2),
            texttemplate="%{text}",
            colorscale=[[0, PALETTE["peach"]], [0.5, PALETTE["slate"]], [1, PALETTE["sky"]]],
            zmid=0,
            showscale=True,
            colorbar=dict(thickness=12, outlinewidth=0, tickfont=dict(color=PALETTE["text_muted"])),
        ))
        fig.update_layout(**_BASE_LAYOUT, title=dict(text="Correlation Matrix", font=dict(size=14)))
        return {"title": "Correlation Matrix", "type": "heatmap", "figure": json.loads(fig.to_json())}
    except Exception as e:
        logger.warning("chart_failed", chart="heatmap", error=str(e))
        return None


def _distribution_chart(df: pd.DataFrame, col: str) -> dict:
    try:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[col].dropna(),
            nbinsx=40,
            marker_color=PALETTE["sky"],
            opacity=0.8,
            name=col,
        ))
        # KDE overlay
        try:
            from scipy import stats as scipy_stats
            vals = df[col].dropna()
            kde_x = np.linspace(vals.min(), vals.max(), 200)
            kde_y = scipy_stats.gaussian_kde(vals)(kde_x) * len(vals) * (vals.max() - vals.min()) / 40
            fig.add_trace(go.Scatter(
                x=kde_x, y=kde_y,
                mode="lines",
                line=dict(color=PALETTE["mint"], width=2),
                name="KDE",
            ))
        except ImportError:
            pass
        fig.update_layout(**_BASE_LAYOUT, title=dict(text=f"Distribution — {col}", font=dict(size=14)),
                          showlegend=False, bargap=0.05)
        return {"title": f"Distribution — {col}", "type": "histogram", "figure": json.loads(fig.to_json())}
    except Exception as e:
        logger.warning("chart_failed", chart="distribution", col=col, error=str(e))
        return None


def _time_series_chart(df: pd.DataFrame, dt_col: str, num_col: str) -> dict:
    try:
        plot_df = df[[dt_col, num_col]].dropna().sort_values(dt_col)
        fig = go.Figure(go.Scatter(
            x=plot_df[dt_col],
            y=plot_df[num_col],
            mode="lines",
            line=dict(color=PALETTE["mint"], width=2),
            fill="tozeroy",
            fillcolor="rgba(186,255,245,0.1)",
        ))
        fig.update_layout(**_BASE_LAYOUT, title=dict(text=f"{num_col} Over Time", font=dict(size=14)))
        return {"title": f"{num_col} Over Time", "type": "timeseries", "figure": json.loads(fig.to_json())}
    except Exception as e:
        logger.warning("chart_failed", chart="timeseries", error=str(e))
        return None


def _category_bar_chart(df: pd.DataFrame, col: str) -> dict:
    try:
        counts = df[col].value_counts().head(15)
        fig = go.Figure(go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker=dict(
                color=counts.values,
                colorscale=[[0, PALETTE["slate"]], [1, PALETTE["sky"]]],
                showscale=False,
            ),
            text=counts.values,
            textposition="outside",
        ))
        fig.update_layout(**_BASE_LAYOUT, title=dict(text=f"Top Values — {col}", font=dict(size=14)))
        fig.update_xaxes(tickangle=-30)
        return {"title": f"Top Values — {col}", "type": "bar", "figure": json.loads(fig.to_json())}
    except Exception as e:
        logger.warning("chart_failed", chart="bar", col=col, error=str(e))
        return None


def _scatter_chart(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    try:
        plot_df = df[[col_x, col_y]].dropna().sample(min(2000, len(df)), random_state=42)
        fig = go.Figure(go.Scatter(
            x=plot_df[col_x],
            y=plot_df[col_y],
            mode="markers",
            marker=dict(
                color=PALETTE["sky"],
                opacity=0.6,
                size=5,
                line=dict(width=0),
            ),
        ))
        # Trend line
        try:
            z = np.polyfit(plot_df[col_x], plot_df[col_y], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_df[col_x].min(), plot_df[col_x].max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line, y=p(x_line),
                mode="lines",
                line=dict(color=PALETTE["peach"], width=2, dash="dash"),
                name="Trend",
            ))
        except Exception:
            pass
        fig.update_layout(**_BASE_LAYOUT,
                          title=dict(text=f"{col_x} vs {col_y}", font=dict(size=14)),
                          xaxis_title=col_x, yaxis_title=col_y)
        return {"title": f"{col_x} vs {col_y}", "type": "scatter", "figure": json.loads(fig.to_json())}
    except Exception as e:
        logger.warning("chart_failed", chart="scatter", error=str(e))
        return None


def _box_plot(df: pd.DataFrame, cols: list) -> dict:
    try:
        fig = go.Figure()
        for i, col in enumerate(cols):
            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                marker_color=COLOR_SEQ[i % len(COLOR_SEQ)],
                line_color=COLOR_SEQ[i % len(COLOR_SEQ)],
                fillcolor=f"rgba{tuple(int(COLOR_SEQ[i % len(COLOR_SEQ)].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + (0.2,)}",
                boxmean=True,
            ))
        fig.update_layout(**_BASE_LAYOUT,
                          title=dict(text="Box Plots — Numeric Columns", font=dict(size=14)),
                          showlegend=False)
        return {"title": "Box Plots — Numeric Columns", "type": "boxplot", "figure": json.loads(fig.to_json())}
    except Exception as e:
        logger.warning("chart_failed", chart="boxplot", error=str(e))
        return None


def _anomaly_scatter(df: pd.DataFrame, numeric_cols: list, anomaly_data: dict) -> dict:
    try:
        col_x, col_y = numeric_cols[0], numeric_cols[1]
        plot_df = df[[col_x, col_y]].copy()
        plot_df["is_anomaly"] = False
        anomaly_idx = anomaly_data.get("anomaly_indices", [])
        valid_idx = [i for i in anomaly_idx if i in plot_df.index]
        plot_df.loc[valid_idx, "is_anomaly"] = True

        normal = plot_df[~plot_df["is_anomaly"]]
        anomalies = plot_df[plot_df["is_anomaly"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=normal[col_x], y=normal[col_y],
            mode="markers",
            marker=dict(color=PALETTE["sky"], size=5, opacity=0.5),
            name="Normal",
        ))
        fig.add_trace(go.Scatter(
            x=anomalies[col_x], y=anomalies[col_y],
            mode="markers",
            marker=dict(color=PALETTE["anomaly"], size=8, symbol="diamond",
                        line=dict(color="white", width=1)),
            name=f"Anomalies ({len(anomalies)})",
        ))
        fig.update_layout(**_BASE_LAYOUT,
                          title=dict(text="Anomaly Detection (Isolation Forest)", font=dict(size=14)),
                          xaxis_title=col_x, yaxis_title=col_y)
        return {"title": "Anomaly Detection", "type": "anomaly_scatter", "figure": json.loads(fig.to_json())}
    except Exception as e:
        logger.warning("chart_failed", chart="anomaly_scatter", error=str(e))
        return None


def _cluster_scatter(df: pd.DataFrame, cols: list) -> dict:
    """Quick 2D cluster scatter using KMeans labels."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        col_x, col_y = cols[0], cols[1]
        plot_df = df[[col_x, col_y]].dropna().sample(min(2000, len(df)), random_state=42)
        X = StandardScaler().fit_transform(plot_df)
        k = min(4, max(2, len(plot_df) // 50))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

        fig = go.Figure()
        cluster_colors = [PALETTE["sky"], PALETTE["mint"], PALETTE["peach"], "#a8d8ea"]
        for i in range(k):
            mask = labels == i
            fig.add_trace(go.Scatter(
                x=plot_df[col_x].values[mask],
                y=plot_df[col_y].values[mask],
                mode="markers",
                marker=dict(color=cluster_colors[i % len(cluster_colors)], size=5, opacity=0.7),
                name=f"Cluster {i + 1}",
            ))
        fig.update_layout(**_BASE_LAYOUT,
                          title=dict(text=f"Cluster Analysis — {col_x} vs {col_y}", font=dict(size=14)),
                          xaxis_title=col_x, yaxis_title=col_y)
        return {"title": "Cluster Analysis", "type": "cluster_scatter", "figure": json.loads(fig.to_json())}
    except Exception as e:
        logger.warning("chart_failed", chart="cluster_scatter", error=str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────

def generate_column_charts(df: pd.DataFrame, col: str) -> list[dict]:
    """
    Generate 2–3 charts focused on a single column for the drill-down modal.
    """
    charts = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if col in numeric_cols:
        charts.append(_distribution_chart(df, col))
        # Correlation bar chart vs other numeric cols
        if len(numeric_cols) > 1:
            other_cols = [c for c in numeric_cols if c != col]
            corr_vals = [df[col].corr(df[c]) for c in other_cols]
            sorted_pairs = sorted(zip(other_cols, corr_vals), key=lambda x: abs(x[1]), reverse=True)[:10]
            corr_cols, corr_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])
            fig = go.Figure(go.Bar(
                x=list(corr_cols), y=list(corr_scores),
                marker=dict(
                    color=[PALETTE["sky"] if v >= 0 else PALETTE["peach"] for v in corr_scores],
                ),
                text=[f"{v:.2f}" for v in corr_scores],
                textposition="outside",
            ))
            fig.update_layout(**_BASE_LAYOUT,
                              title=dict(text=f"Correlation with {col}", font=dict(size=14)),
                              yaxis=dict(range=[-1.1, 1.1]))
            charts.append({"title": f"Correlation with {col}", "type": "corr_bar", "figure": json.loads(fig.to_json())})
    else:
        # Categorical column
        counts = df[col].value_counts().head(15)
        fig = go.Figure(go.Bar(
            x=counts.index.tolist(), y=counts.values.tolist(),
            marker_color=PALETTE["sky"],
            text=counts.values.tolist(),
            textposition="outside",
        ))
        fig.update_layout(**_BASE_LAYOUT,
                          title=dict(text=f"Value Counts — {col}", font=dict(size=14)),
                          xaxis=dict(tickangle=-30))
        charts.append({"title": f"Value Counts — {col}", "type": "bar", "figure": json.loads(fig.to_json())})

        # Pie chart if ≤10 unique values
        if df[col].nunique() <= 10:
            fig2 = go.Figure(go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                marker=dict(colors=COLOR_SEQ),
                hole=0.4,
            ))
            fig2.update_layout(**_BASE_LAYOUT, title=dict(text=f"Proportion — {col}", font=dict(size=14)))
            charts.append({"title": f"Proportion — {col}", "type": "pie", "figure": json.loads(fig2.to_json())})

    return [c for c in charts if c is not None]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_datetime_cols(df: pd.DataFrame) -> list:
    """Return columns that look like datetimes."""
    result = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result.append(col)
            continue
        if df[col].dtype == object:
            try:
                sample = df[col].dropna().head(20)
                pd.to_datetime(sample, infer_datetime_format=True)
                result.append(col)
            except Exception:
                pass
    return result


def _find_top_correlated_pair(df: pd.DataFrame, numeric_cols: list):
    """Find the pair of columns with the highest absolute correlation."""
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr().abs()
    best, best_val = None, 0
    for a, b in combinations(numeric_cols, 2):
        val = corr.loc[a, b]
        if not np.isnan(val) and val > best_val:
            best_val = val
            best = (a, b)
    return best
