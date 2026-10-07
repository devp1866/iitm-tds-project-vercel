"""
jobs/job_queue.py
Async job queue for Autolysis v2.

Local/VPS: Uses threading (in-memory job tracking).
Vercel: Uses Upstash Redis for cross-invocation state persistence.

The job runner:
1. Reads the uploaded file
2. Runs statistical analysis
3. Generates Plotly charts
4. Calls LLM for story
5. Saves report to Firestore
6. Updates job status
"""

import os
import json
import uuid
import threading
import pandas as pd
import structlog

from services.analysis import (
    read_dataset, get_column_info, detect_outliers_iqr,
    detect_anomalies_ml, perform_clustering, build_analysis_context,
    json_serializer,
)
from services.visualizations import generate_smart_charts
from services.llm import generate_story
from models.db import create_job, update_job_status, save_report

logger = structlog.get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# UPSTASH REDIS (optional, for Vercel)
# ─────────────────────────────────────────────────────────────────────────────

_redis_client = None


def _get_redis():
    """Lazy-init Upstash Redis client. Returns None if not configured."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    url = os.getenv("UPSTASH_REDIS_URL")
    token = os.getenv("UPSTASH_REDIS_TOKEN")
    if not url or not token:
        return None

    try:
        from upstash_redis import Redis
        _redis_client = Redis(url=url, token=token)
        logger.info("upstash_redis_connected")
        return _redis_client
    except Exception as e:
        logger.warning("upstash_redis_failed", error=str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# JOB SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────

def submit_job(file_path: str, filename: str, file_ext: str) -> str:
    """
    Submit a new analysis job. Returns job_id immediately.
    Processing runs in a background thread.
    """
    job_id = str(uuid.uuid4())
    create_job(job_id, filename, file_ext)

    thread = threading.Thread(
        target=_run_analysis_job,
        args=(job_id, file_path, filename, file_ext),
        daemon=True,
    )
    thread.start()
    logger.info("job_submitted", job_id=job_id, filename=filename)
    return job_id


# ─────────────────────────────────────────────────────────────────────────────
# JOB RUNNER (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_analysis_job(job_id: str, file_path: str, filename: str, file_ext: str):
    """Full analysis pipeline. Runs in a background thread."""
    try:
        # Stage 1: Read dataset
        _progress(job_id, 5, "Reading dataset...")
        df = read_dataset(file_path, file_ext)

        # Stage 2: Column info
        _progress(job_id, 20, "Computing column statistics...")
        column_info = get_column_info(df)

        # Stage 3: Outlier detection
        _progress(job_id, 35, "Detecting outliers...")
        outliers = detect_outliers_iqr(df)

        # Stage 4: ML Anomaly detection
        _progress(job_id, 50, "Running anomaly detection...")
        anomaly_data = detect_anomalies_ml(df)

        # Stage 5: Clustering
        _progress(job_id, 60, "Clustering data...")
        clustering = perform_clustering(df)

        # Stage 6: Build context
        _progress(job_id, 65, "Building analysis context...")
        context = build_analysis_context(df, column_info, outliers, anomaly_data, clustering, filename)

        # Stage 7: Generate charts
        _progress(job_id, 75, "Generating interactive charts...")
        charts = generate_smart_charts(df, anomaly_data)

        # Stage 8: LLM story
        _progress(job_id, 85, "Generating AI narrative...")
        readme = generate_story(context)

        # Build markdown report with stats table
        from services.analysis import json_serializer
        readme_full = _build_full_readme(readme, df, filename, anomaly_data)

        # Stage 9: Save to Firestore
        _progress(job_id, 95, "Saving report...")
        report_data = {
            "filename": filename,
            "readme": readme_full,
            "charts": charts,
            "context": context,
            "column_info": column_info,
            "anomaly_data": anomaly_data,
        }
        share_token = save_report(job_id, report_data)

        # Done!
        update_job_status(job_id, "done", progress=100, progress_label="Analysis complete!")
        logger.info("job_completed", job_id=job_id, share_token=share_token)

    except Exception as e:
        logger.error("job_failed", job_id=job_id, error=str(e))
        update_job_status(job_id, "error", error_msg=str(e))

    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


def _progress(job_id: str, pct: int, label: str):
    update_job_status(job_id, "running", progress=pct, progress_label=label)
    logger.debug("job_progress", job_id=job_id, pct=pct, label=label)


def _df_to_markdown(df) -> str:
    """Robust custom markdown table generator without needing tabulate."""
    if df.empty:
        return "No data available."
    
    # Handle Series
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Get headers
    cols = [""] + list(df.columns) if df.index.name is None else [df.index.name or "Index"] + list(df.columns)
    header = "| " + " | ".join(str(c).replace("|", "-").strip() for c in cols) + " |"
    separator = "|" + "|".join(["---"] * len(cols)) + "|"
    
    # Get rows
    rows = []
    for idx, row in df.iterrows():
        row_vals = [str(idx)] + [str(v).replace("\n", " ").replace("|", "-").strip() for v in row.values]
        # truncate very long strings to prevent breaking layout
        row_vals = [v[:50] + "..." if len(v) > 50 else v for v in row_vals]
        rows.append("| " + " | ".join(row_vals) + " |")
        
    return "\n".join([header, separator] + rows)


def _build_full_readme(narrative: str, df, filename: str, anomaly_data: dict = None) -> str:
    """Combine LLM narrative + stats table into final markdown."""
    summary = df.describe(include="all").transpose()
    missing = df.isnull().sum()
    missing_df = missing[missing > 0].rename("Missing Count")

    md = f"# Analysis Report — `{filename}`\n\n"
    md += narrative + "\n\n"
    
    if anomaly_data and "top_anomalies" in anomaly_data and anomaly_data["top_anomalies"]:
        md += "---\n\n## 🚨 Top Anomalies Detected\n\n"
        md += f"Isolation Forest detected **{anomaly_data['anomaly_count']}** anomalies. Here are the most extreme cases:\n\n"
        top_anom_df = pd.DataFrame(anomaly_data["top_anomalies"])
        md += _df_to_markdown(top_anom_df) + "\n\n"

    md += "---\n\n## 📊 Detailed Statistics\n\n"
    md += "### Summary Statistics\n"
    md += _df_to_markdown(summary) + "\n\n"

    if not missing_df.empty:
        md += "### Missing Values\n"
        md += _df_to_markdown(missing_df) + "\n\n"

    return md
