"""
services/analysis.py
Core data analysis service for Autolysis v2.
Handles multi-format reading, statistical analysis, ML anomaly detection, clustering.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from charset_normalizer import detect
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import structlog

logger = structlog.get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# FILE READING
# ─────────────────────────────────────────────────────────────────────────────

MAX_ROWS = 100_000  # Safety limit


def read_dataset(file_path: str, file_type: str = None) -> pd.DataFrame:
    """
    Read a dataset from file. Supports CSV, Excel (.xlsx/.xls), and JSON.
    Returns a DataFrame or raises ValueError on failure.
    """
    ext = file_type or os.path.splitext(file_path)[1].lower().lstrip(".")

    try:
        if ext == "csv":
            encoding = _detect_encoding(file_path)
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(file_path, engine="openpyxl")
        elif ext == "json":
            df = pd.read_json(file_path)
            if not isinstance(df, pd.DataFrame):
                raise ValueError("JSON did not parse to a tabular structure.")
        else:
            raise ValueError(f"Unsupported file format: .{ext}. Supported: CSV, Excel, JSON.")
    except Exception as e:
        logger.error("read_dataset_failed", ext=ext, error=str(e))
        raise ValueError(f"Failed to read file: {e}") from e

    # Row limit
    if len(df) > MAX_ROWS:
        logger.warning("dataset_truncated", original=len(df), limit=MAX_ROWS)
        df = df.head(MAX_ROWS)

    logger.info("dataset_read", rows=len(df), cols=len(df.columns), format=ext)
    return df


def _detect_encoding(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read(100_000)  # Read first 100KB for detection
    result = detect(raw)
    encoding = result.get("encoding") or "utf-8"
    # Fallback chain
    for enc in [encoding, "utf-8", "latin-1", "cp1252"]:
        try:
            with open(file_path, encoding=enc) as f:
                f.read(1000)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8"


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def get_column_info(df: pd.DataFrame) -> dict:
    """Return rich metadata about each column."""
    info = {}
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        is_datetime = pd.api.types.is_datetime64_any_dtype(col_data)
        n_unique = col_data.nunique()
        n_missing = int(col_data.isna().sum())

        entry = {
            "dtype": dtype,
            "is_numeric": is_numeric,
            "is_datetime": is_datetime,
            "n_unique": int(n_unique),
            "n_missing": n_missing,
            "pct_missing": round(n_missing / len(df) * 100, 2),
        }

        if is_numeric:
            entry.update({
                "mean": _safe_float(col_data.mean()),
                "std": _safe_float(col_data.std()),
                "min": _safe_float(col_data.min()),
                "max": _safe_float(col_data.max()),
                "q25": _safe_float(col_data.quantile(0.25)),
                "median": _safe_float(col_data.median()),
                "q75": _safe_float(col_data.quantile(0.75)),
            })
        elif not is_datetime and n_unique <= 50:
            top_vals = col_data.value_counts().head(10).to_dict()
            entry["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

        info[col] = entry
    return info


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """IQR-based outlier detection per numerical column."""
    numeric = df.select_dtypes(include=["number"])
    summary = {}
    for col in numeric.columns:
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        mask = (numeric[col] < lo) | (numeric[col] > hi)
        count = int(mask.sum())
        if count > 0:
            summary[col] = {
                "count": count,
                "percentage": round(count / len(df) * 100, 2),
                "lower_bound": _safe_float(lo),
                "upper_bound": _safe_float(hi),
                "min_outlier": _safe_float(numeric[col][mask].min()),
                "max_outlier": _safe_float(numeric[col][mask].max()),
            }
    return summary


def detect_anomalies_ml(df: pd.DataFrame) -> dict:
    """
    Isolation Forest anomaly detection on all numeric columns.
    Returns: {
        'anomaly_indices': [...],   # row indices flagged as anomalies
        'anomaly_scores': [...],    # decision function scores
        'anomaly_count': int,
        'anomaly_pct': float,
        'feature_columns': [...]    # which columns were used
    }
    """
    numeric = df.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    if numeric.shape[1] < 1:
        return {"error": "No numeric columns for anomaly detection."}

    clean = numeric.dropna()
    if len(clean) < 10:
        return {"error": "Insufficient data rows for anomaly detection (need ≥10)."}

    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(clean)

        # Auto contamination based on data size
        contamination = min(0.05, max(0.01, 10 / len(clean)))
        iso = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        labels = iso.fit_predict(X)
        scores = iso.decision_function(X)

        anomaly_mask = labels == -1
        anomaly_indices = clean.index[anomaly_mask].tolist()
        anomaly_count = int(anomaly_mask.sum())

        return {
            "anomaly_indices": anomaly_indices,
            "anomaly_scores": scores.tolist(),
            "anomaly_count": anomaly_count,
            "anomaly_pct": round(anomaly_count / len(clean) * 100, 2),
            "feature_columns": list(numeric.columns),
            "contamination_used": round(contamination, 4),
        }
    except Exception as e:
        logger.error("anomaly_detection_failed", error=str(e))
        return {"error": f"Anomaly detection failed: {e}"}


def perform_clustering(df: pd.DataFrame) -> dict:
    """
    KMeans clustering with elbow method to auto-select k (2–6).
    Returns cluster summary dict.
    """
    numeric = df.select_dtypes(include=["number"]).dropna()
    if len(numeric.columns) < 2:
        return {"error": "Need ≥2 numerical columns for clustering."}
    if len(numeric) < 10:
        return {"error": "Insufficient rows for clustering (need ≥10)."}

    try:
        X = StandardScaler().fit_transform(numeric.values)

        # Elbow method: find optimal k (2–6)
        max_k = min(6, len(numeric) // 3)
        if max_k < 2:
            return {"error": "Too few data points for meaningful clustering."}

        best_k = 3
        best_inertia_drop = 0
        centroids_list = []
        inertias = []

        for k in range(2, max_k + 1):
            centroids, labels, inertia = _kmeans(X, k)
            inertias.append(inertia)
            centroids_list.append((k, centroids, labels))

        # Pick elbow: largest relative drop
        for i in range(1, len(inertias)):
            drop = (inertias[i - 1] - inertias[i]) / (inertias[i - 1] + 1e-8)
            if drop > best_inertia_drop:
                best_inertia_drop = drop
                best_k = i + 2  # k starts from 2

        # Use best k
        _, best_centroids, best_labels = centroids_list[best_k - 2]
        cluster_col = pd.Series(best_labels, index=numeric.index, name="Cluster")
        summary_df = numeric.copy()
        summary_df["Cluster"] = cluster_col
        cluster_means = summary_df.groupby("Cluster").mean().round(3)

        return {
            "k": best_k,
            "cluster_sizes": {int(k): int(v) for k, v in cluster_col.value_counts().items()},
            "cluster_means": json.loads(cluster_means.to_json()),
            "inertias": inertias,
        }
    except Exception as e:
        logger.error("clustering_failed", error=str(e))
        return {"error": f"Clustering failed: {e}"}


def _kmeans(X: np.ndarray, k: int, max_iter: int = 50):
    """Lightweight NumPy KMeans. Returns (centroids, labels, inertia)."""
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), k, replace=False)
    centroids = X[idx].copy()

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    inertia = float(np.min(dists, axis=1).sum())
    return centroids, labels.tolist(), inertia


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY CONTEXT (for LLM)
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_context(
    df: "pd.DataFrame",
    column_info: dict,
    outliers: dict,
    anomalies: dict,
    clustering: dict,
    filename: str,
) -> dict:
    """
    Build a rich, LLM-queryable context dict with ACTUAL computed values.
    This lets the AI answer specific questions like 'what are total sales?'
    without saying 'I cannot access the data'.
    """
    # ── Numeric aggregates: real computed values per column ──────────────────
    numeric_aggregates = {}
    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        numeric_aggregates[col] = {
            "sum": _safe_float(series.sum()),
            "mean": _safe_float(series.mean()),
            "median": _safe_float(series.median()),
            "std": _safe_float(series.std()),
            "min": _safe_float(series.min()),
            "max": _safe_float(series.max()),
            "count": int(series.count()),
            "null_count": int(df[col].isnull().sum()),
            "q25": _safe_float(series.quantile(0.25)),
            "q75": _safe_float(series.quantile(0.75)),
            "skewness": _safe_float(series.skew()),
        }

    # ── Categorical aggregates: top values per column ──────────────────────
    cat_aggregates = {}
    cat_cols = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    for col in cat_cols[:15]:
        vc = df[col].value_counts(dropna=True)
        cat_aggregates[col] = {
            "unique_count": int(df[col].nunique()),
            "null_count": int(df[col].isnull().sum()),
            "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
            "mode": str(vc.index[0]) if len(vc) > 0 else None,
        }

    # ── Pairwise correlations (top 15 strongest pairs) ────────────────────
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    top_correlations = []
    if len(numeric_cols) >= 2:
        corr_mat = df[numeric_cols].corr()
        pairs = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                val = _safe_float(corr_mat.loc[c1, c2])
                if val is not None:
                    pairs.append((abs(val), val, c1, c2))
        pairs.sort(reverse=True)
        top_correlations = [
            {"col1": c1, "col2": c2, "correlation": v}
            for _, v, c1, c2 in pairs[:15]
        ]

    return {
        "filename": filename,
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample_rows": df.head(5).fillna("").astype(str).to_dict(orient="records"),
        "numeric_aggregates": numeric_aggregates,
        "cat_aggregates": cat_aggregates,
        "top_correlations": top_correlations,
        "column_info": column_info,
        "outliers": outliers,
        "anomaly_summary": {
            k: v for k, v in anomalies.items()
            if k not in ("anomaly_indices", "anomaly_scores")
        },
        "clustering": clustering,
    }



# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val):
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 6)
    except Exception:
        return None


def json_serializer(obj):
    """Custom JSON serializer for numpy types. NaN/Inf → None (JSON null)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        # JSON has no NaN/Infinity — return null instead
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")
