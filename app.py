"""
app.py — Autolysis v2
Flask application: route definitions only.
All business logic lives in services/ and jobs/.
"""

import os
import json
import uuid
import structlog
from flask import Flask, render_template, request, jsonify, abort, send_from_directory
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

from jobs.job_queue import submit_job
from models.db import get_job, get_report_by_job, get_report_by_token
from services.llm import chat_with_data, analyze_column
from services.email_service import send_contact_email
from services.visualizations import generate_column_charts
from services.analysis import read_dataset, get_column_info, json_serializer

logger = structlog.get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")
app.config["UPLOAD_FOLDER"] = "/tmp"
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "60 per hour"],
    storage_uri="memory://",
)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "parquet"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/report/<share_token>")
def public_report(share_token: str):
    """Public share link — read-only view of analysis results."""
    report = get_report_by_token(share_token)
    if not report:
        abort(404)
    return render_template("report.html", share_token=share_token)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "2.0.0"})


@app.route("/robots.txt")
def robots_txt():
    """Serve robots.txt at canonical root URL for search engines."""
    return send_from_directory(app.static_folder, "robots.txt",
                               mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap_xml():
    """Serve sitemap.xml at canonical root URL for search engines."""
    return send_from_directory(app.static_folder, "sitemap.xml",
                               mimetype="application/xml")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per hour")
def submit():
    """
    Upload a file and start an async analysis job.
    Returns job_id immediately — client polls /status/<job_id>.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    filename = secure_filename(file.filename)
    ext = _get_ext(filename)
    job_id = str(uuid.uuid4())
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_{filename}")

    try:
        file.save(file_path)
    except Exception as e:
        logger.error("file_save_failed", error=str(e))
        return jsonify({"error": "Failed to save uploaded file."}), 500

    # Submit async job
    try:
        submitted_job_id = submit_job(file_path, filename, ext)
        logger.info("analysis_submitted", job_id=submitted_job_id, filename=filename)
        return jsonify({"job_id": submitted_job_id}), 202
    except Exception as e:
        logger.error("job_submit_failed", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/status/<job_id>", methods=["GET"])
def job_status(job_id: str):
    """Poll job status. Returns status, progress (0-100), and label."""
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    response = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress": job.get("progress", 0),
        "progress_label": job.get("progress_label", ""),
        "error_msg": job.get("error_msg"),
    }
    return jsonify(response)


@app.route("/result/<job_id>", methods=["GET"])
def get_result(job_id: str):
    """
    Fetch complete analysis results once job is done.
    Returns the full report including charts JSON and share token.
    """
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    if job.get("status") != "done":
        return jsonify({"error": "Analysis not yet complete.", "status": job.get("status")}), 202

    report = get_report_by_job(job_id)
    if not report:
        return jsonify({"error": "Report not found."}), 404

    # Parse JSON fields
    try:
        charts = json.loads(report.get("charts_json", "[]"))
    except Exception:
        charts = []

    try:
        column_info = json.loads(report.get("column_info", "{}"))
    except Exception:
        column_info = {}

    try:
        anomaly_data = json.loads(report.get("anomaly_data", "{}"))
    except Exception:
        anomaly_data = {}

    return jsonify({
        "job_id": job_id,
        "filename": report.get("filename", ""),
        "readme": report.get("readme", ""),
        "charts": charts,
        "column_info": column_info,
        "anomaly_data": anomaly_data,
        "share_token": report.get("share_token", ""),
        "created_at": report.get("created_at", ""),
    })


# ─────────────────────────────────────────────────────────────────────────────
# AI CHAT API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
@limiter.limit("30 per hour")
def chat():
    """
    AI chat endpoint.
    Body: { "job_id": str, "messages": [{"role": "user"|"assistant", "content": str}] }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    job_id = data.get("job_id")
    messages = data.get("messages", [])

    if not job_id:
        return jsonify({"error": "job_id is required."}), 400
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "messages array is required."}), 400

    report = get_report_by_job(job_id)
    if not report:
        return jsonify({"error": "Analysis report not found. Run analysis first."}), 404

    try:
        context = json.loads(report.get("summary_context", "{}"))
    except Exception:
        context = {}

    try:
        reply = chat_with_data(context, messages)
        return jsonify({"reply": reply})
    except TimeoutError as e:
        return jsonify({"error": str(e)}), 504
    except Exception as e:
        logger.error("chat_failed", error=str(e))
        return jsonify({"error": "Chat failed. Please try again."}), 500


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN ANALYSIS API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/column-analyze", methods=["POST"])
@limiter.limit("20 per hour")
def column_analyze():
    """
    Deep-dive analysis for a single column.
    Body: { "job_id": str, "column": str }
    Returns: { "insight": str (markdown), "charts": [...] }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400

    job_id = data.get("job_id")
    column = data.get("column")

    if not job_id or not column:
        return jsonify({"error": "job_id and column are required."}), 400

    report = get_report_by_job(job_id)
    if not report:
        return jsonify({"error": "Report not found."}), 404

    try:
        context = json.loads(report.get("summary_context", "{}"))
    except Exception:
        context = {}

    # Validate column exists
    if column not in context.get("columns", []):
        return jsonify({"error": f"Column '{column}' not found in dataset."}), 400

    try:
        insight = analyze_column(context, column)
    except Exception as e:
        logger.error("column_analyze_failed", error=str(e))
        return jsonify({"error": "Column analysis failed."}), 500

    # Return column charts too (requires rebuilding from context — we use summary stats)
    # For column charts we rely on the client-side Plotly data already loaded
    return jsonify({
        "column": column,
        "insight": insight,
    })


# ─────────────────────────────────────────────────────────────────────────────
# REPORT API (for share link)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/report/<share_token>", methods=["GET"])
def api_report(share_token: str):
    """Return report data by share token (used by report.html)."""
    report = get_report_by_token(share_token)
    if not report:
        return jsonify({"error": "Report not found."}), 404

    try:
        charts = json.loads(report.get("charts_json", "[]"))
    except Exception:
        charts = []

    try:
        column_info = json.loads(report.get("column_info", "{}"))
    except Exception:
        column_info = {}

    try:
        anomaly_data = json.loads(report.get("anomaly_data", "{}"))
    except Exception:
        anomaly_data = {}

    return jsonify({
        "filename": report.get("filename", ""),
        "readme": report.get("readme", ""),
        "charts": charts,
        "column_info": column_info,
        "anomaly_data": anomaly_data,
        "created_at": report.get("created_at", ""),
    })


# ─────────────────────────────────────────────────────────────────────────────
# CONTACT
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/contact", methods=["POST"])
@limiter.limit("5 per hour")
def contact():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    message = request.form.get("message", "").strip()

    if not name or not email or not message:
        return render_template("about.html", error="All fields are required.")

    success, msg = send_contact_email(name, email, message)
    if success:
        return render_template("about.html", success=True)
    else:
        return render_template("about.html", error=msg)


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404


@app.errorhandler(413)
def file_too_large(e):
    max_mb = app.config.get("MAX_CONTENT_LENGTH", 50 * 1024 * 1024) // (1024 * 1024)
    return jsonify({"error": f"File too large. Maximum size is {max_mb}MB."}), 413


@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
