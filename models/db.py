"""
models/db.py
Firebase Firestore integration for Autolysis v2.
Stores analysis jobs and reports for async processing and share links.

Collections:
  - analysis_jobs: job status tracking
  - analysis_reports: full results for share links

Setup:
  1. Create a Firebase project at https://console.firebase.google.com
  2. Go to Project Settings > Service Accounts > Generate new private key
  3. Save the JSON file as firebase-credentials.json in project root
  4. Set FIREBASE_CREDENTIALS_PATH=firebase-credentials.json in .env
"""

import os
import json
import uuid
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger()

_db = None  # Lazy-initialized Firestore client
_mock_store = {}  # In-memory fallback if Firebase is not configured


def _resolve_credentials_path() -> str | None:
    """
    Resolve Firebase credentials. Supports two modes:
    1. FIREBASE_CREDENTIALS_BASE64 — base64 encoded JSON (for Vercel/cloud deployments).
       Decodes to /tmp/firebase-credentials.json at runtime.
    2. FIREBASE_CREDENTIALS_PATH — local file path (for local development).
    Returns the path to the credentials file, or None if not configured.
    """
    # Mode 1: Base64 env var (Vercel-friendly)
    b64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")
    if b64:
        import base64
        tmp_path = "/tmp/firebase-credentials.json"
        try:
            decoded = base64.b64decode(b64).decode("utf-8")
            with open(tmp_path, "w") as f:
                f.write(decoded)
            logger.info("firebase_credentials_decoded", path=tmp_path)
            return tmp_path
        except Exception as e:
            logger.error("firebase_credentials_decode_failed", error=str(e))
            return None

    # Mode 2: Local file path
    creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
    if os.path.exists(creds_path):
        return creds_path

    logger.warning("firebase_credentials_missing",
                   note="Using in-memory fallback — data won't persist across restarts")
    return None


def _get_db():
    """Lazy-initialize Firestore client. Falls back to in-memory store."""
    global _db
    if _db is not None:
        return _db

    creds_path = _resolve_credentials_path()
    if not creds_path:
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_path)
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
        logger.info("firestore_connected")
        return _db
    except Exception as e:
        logger.error("firestore_init_failed", error=str(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# JOB OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def create_job(job_id: str, filename: str, file_format: str) -> dict:
    """Create a new analysis job record."""
    job = {
        "id": job_id,
        "status": "pending",
        "filename": filename,
        "file_format": file_format,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error_msg": None,
        "progress": 0,
        "progress_label": "Queued",
    }
    _write_job(job_id, job)
    return job


def update_job_status(job_id: str, status: str, progress: int = None,
                      progress_label: str = None, error_msg: str = None):
    """Update a job's status and progress."""
    updates = {"status": status}
    if progress is not None:
        updates["progress"] = progress
    if progress_label is not None:
        updates["progress_label"] = progress_label
    if error_msg is not None:
        updates["error_msg"] = error_msg
    if status in ("done", "error"):
        updates["completed_at"] = datetime.now(timezone.utc).isoformat()

    _update_job(job_id, updates)


def get_job(job_id: str) -> dict | None:
    """Fetch a job record by ID."""
    return _read_job(job_id)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_report(job_id: str, report_data: dict) -> str:
    """
    Save a completed analysis report. Returns the share_token.
    report_data keys: readme, charts_json, summary_context, column_info, anomaly_data
    """
    share_token = _generate_share_token()
    doc = {
        "id": job_id,
        "share_token": share_token,
        "filename": report_data.get("filename", ""),
        "readme": report_data.get("readme", ""),
        "charts_json": json.dumps(report_data.get("charts", [])),
        "summary_context": json.dumps(report_data.get("context", {})),
        "column_info": json.dumps(report_data.get("column_info", {})),
        "anomaly_data": json.dumps(report_data.get("anomaly_data", {})),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_report(job_id, share_token, doc)
    return share_token


def get_report_by_job(job_id: str) -> dict | None:
    """Fetch report by job ID."""
    return _read_report_by_job(job_id)


def get_report_by_token(share_token: str) -> dict | None:
    """Fetch report by share token (for public links)."""
    return _read_report_by_token(share_token)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: Firestore / In-Memory ops
# ─────────────────────────────────────────────────────────────────────────────

def _write_job(job_id: str, data: dict):
    db = _get_db()
    if db:
        db.collection("analysis_jobs").document(job_id).set(data)
    else:
        _mock_store.setdefault("jobs", {})[job_id] = data


def _update_job(job_id: str, updates: dict):
    db = _get_db()
    if db:
        db.collection("analysis_jobs").document(job_id).update(updates)
    else:
        if job_id in _mock_store.get("jobs", {}):
            _mock_store["jobs"][job_id].update(updates)


def _read_job(job_id: str) -> dict | None:
    db = _get_db()
    if db:
        doc = db.collection("analysis_jobs").document(job_id).get()
        return doc.to_dict() if doc.exists else None
    else:
        return _mock_store.get("jobs", {}).get(job_id)


def _write_report(job_id: str, share_token: str, data: dict):
    db = _get_db()
    if db:
        db.collection("analysis_reports").document(job_id).set(data)
        # Also index by share_token for fast lookup
        db.collection("report_tokens").document(share_token).set({"job_id": job_id})
    else:
        _mock_store.setdefault("reports", {})[job_id] = data
        _mock_store.setdefault("tokens", {})[share_token] = job_id


def _read_report_by_job(job_id: str) -> dict | None:
    db = _get_db()
    if db:
        doc = db.collection("analysis_reports").document(job_id).get()
        return doc.to_dict() if doc.exists else None
    else:
        return _mock_store.get("reports", {}).get(job_id)


def _read_report_by_token(share_token: str) -> dict | None:
    db = _get_db()
    if db:
        token_doc = db.collection("report_tokens").document(share_token).get()
        if not token_doc.exists:
            return None
        job_id = token_doc.to_dict().get("job_id")
        return _read_report_by_job(job_id) if job_id else None
    else:
        job_id = _mock_store.get("tokens", {}).get(share_token)
        if job_id:
            return _mock_store.get("reports", {}).get(job_id)
        return None


def _generate_share_token(length: int = 8) -> str:
    """Generate a short, URL-safe share token."""
    import secrets
    import string
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))
