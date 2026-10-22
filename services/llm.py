"""
services/llm.py — Autolysis v2
All LLM interactions powered by Google Gemini 2.0 Flash via the official google-genai SDK v2.x.
- Analysis story generation
- Streaming conversational chat (Server-Sent Events)
- Column deep-dive analysis
"""

import os
import json
import structlog
from google import genai
from google.genai import types

logger = structlog.get_logger()

# Using gemini-2.0-flash: fast, free-tier, 1M context, best for data analysis
MODEL = "gemini-2.0-flash"


def _get_client() -> genai.Client:
    """Lazy-initialize Gemini client using GOOGLE_API_KEY."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def _fmt_num(v):
    """Format a number nicely for the system prompt."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) >= 1_000_000:
            return f"{v:,.0f}"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:.4g}"
    return str(v)


def _build_system_context(context: dict) -> str:
    """
    Convert analysis context into a rich system prompt containing
    ACTUAL computed values so the AI can answer specific questions directly.
    """
    filename = context.get("filename", "dataset")
    shape = context.get("shape", {})
    cols = context.get("columns", [])
    sample = context.get("sample_rows", [])
    outlier_cols = list(context.get("outliers", {}).keys())
    anomaly = context.get("anomaly_summary", {})
    clustering = context.get("clustering", {})
    numeric_agg = context.get("numeric_aggregates", {})
    cat_agg = context.get("cat_aggregates", {})
    top_corr = context.get("top_correlations", [])
    cleaning_report = context.get("cleaning_report", {})

    lines = [
        f"You are an expert data analyst. The user uploaded a dataset called '{filename}'.",
        f"Shape: {shape.get('rows', '?')} rows × {shape.get('cols', '?')} columns.",
        f"Columns: {', '.join(cols)}",
        "",
        "=== IMPORTANT: You have FULL access to pre-computed statistics below. ===",
        "Always answer specific questions using these exact values. NEVER say you cannot access the data.",
        "",
    ]

    if cleaning_report and cleaning_report.get("actions"):
        lines.append("--- DATA CLEANING APPLIED ---")
        for action in cleaning_report["actions"]:
            lines.append(f"  • {action}")
        lines.append("")

    if numeric_agg:
        lines.append("--- NUMERIC COLUMN STATISTICS ---")
        for col, stats in numeric_agg.items():
            parts = [
                f"sum={_fmt_num(stats.get('sum'))}",
                f"mean={_fmt_num(stats.get('mean'))}",
                f"median={_fmt_num(stats.get('median'))}",
                f"min={_fmt_num(stats.get('min'))}",
                f"max={_fmt_num(stats.get('max'))}",
                f"std={_fmt_num(stats.get('std'))}",
                f"nulls={stats.get('null_count', 0)}",
            ]
            lines.append(f"  [{col}]: {', '.join(parts)}")
        lines.append("")

    if cat_agg:
        lines.append("--- CATEGORICAL COLUMN STATISTICS ---")
        for col, stats in cat_agg.items():
            top = stats.get("top_values", {})
            top_str = ", ".join(f"'{k}':{v}" for k, v in list(top.items())[:5])
            lines.append(
                f"  [{col}]: {stats.get('unique_count')} unique, "
                f"mode='{stats.get('mode')}', top: {top_str}, nulls={stats.get('null_count', 0)}"
            )
        lines.append("")

    if top_corr:
        lines.append("--- TOP CORRELATIONS ---")
        for pair in top_corr[:8]:
            lines.append(
                f"  {pair['col1']} ↔ {pair['col2']}: r={_fmt_num(pair['correlation'])}"
            )
        lines.append("")

    if sample:
        lines.append(f"--- SAMPLE ROWS (first {len(sample)}) ---")
        lines.append(json.dumps(sample, default=str))
        lines.append("")

    if outlier_cols:
        lines.append(f"IQR outliers detected in: {', '.join(outlier_cols)}")
    if anomaly.get("anomaly_count"):
        lines.append(
            f"Isolation Forest found {anomaly['anomaly_count']} anomalies "
            f"({anomaly.get('anomaly_pct', 0)}% of rows)."
        )
    if isinstance(clustering, dict) and "k" in clustering:
        lines.append(
            f"KMeans clustering: {clustering['k']} clusters, "
            f"sizes: {clustering.get('cluster_sizes', {})}"
        )

    lines.extend([
        "",
        "Rules: Answer ONLY from the statistics above. Give exact numbers when asked. "
        "Be concise and direct. Format in markdown. Never say the data is inaccessible.",
    ])

    return "\n".join(lines)


def _build_chat_contents(system_prompt: str, messages: list) -> list:
    """Build contents list for multi-turn chat (google-genai v2.x format)."""
    contents = []
    # Inject system context as first user + model turn so it's part of history
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=f"[SYSTEM CONTEXT]\n{system_prompt}\n[/SYSTEM CONTEXT]\n\nAcknowledge you have the context.")]
    ))
    contents.append(types.Content(
        role="model",
        parts=[types.Part(text="Understood. I have full access to the dataset statistics and will answer based on those exact values.")]
    ))
    # Add conversation history (last 10 turns)
    for msg in messages[-10:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=msg["content"])]
        ))
    return contents


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_story(context: dict) -> str:
    """
    Generate a comprehensive, structured analysis narrative from the context dict.
    Returns a Markdown string.
    """
    client = _get_client()
    filename = context.get("filename", "dataset")
    cols = context.get("columns", [])
    sample = context.get("sample_rows", [])
    numeric_agg = context.get("numeric_aggregates", {})
    cat_agg = context.get("cat_aggregates", {})
    outliers = context.get("outliers", {})
    anomaly_summary = context.get("anomaly_summary", {})
    clustering = context.get("clustering", {})
    top_corr = context.get("top_correlations", [])
    shape = context.get("shape", {})
    cleaning_report = context.get("cleaning_report", {})

    stats_block = ""
    if numeric_agg:
        rows = []
        for col, s in numeric_agg.items():
            rows.append(
                f"| {col} | {_fmt_num(s.get('sum'))} | {_fmt_num(s.get('mean'))} | "
                f"{_fmt_num(s.get('median'))} | {_fmt_num(s.get('min'))} | {_fmt_num(s.get('max'))} |"
            )
        stats_block = (
            "| Column | Sum | Mean | Median | Min | Max |\n"
            "|--------|-----|------|--------|-----|-----|\n"
            + "\n".join(rows)
        )

    cleaning_note = ""
    if cleaning_report.get("actions"):
        cleaning_note = (
            f"\n**Auto-cleaning applied:** {len(cleaning_report['actions'])} operations "
            f"({', '.join(cleaning_report['actions'][:3])})\n"
        )

    prompt = (
        f"You are a senior data analyst writing a professional executive report.\n\n"
        f"**Dataset:** `{filename}` — {shape.get('rows','?')} rows × {shape.get('cols','?')} columns\n"
        f"**Columns:** {cols}\n"
        f"**Sample rows:** {json.dumps(sample[:3], default=str)}\n"
        f"{cleaning_note}\n"
        f"**Computed Statistics:**\n{stats_block}\n\n"
        f"**Categorical columns:** {json.dumps({k: v['top_values'] for k, v in cat_agg.items()}, default=str)}\n"
        f"**Top correlations:** {json.dumps(top_corr[:8], default=str)}\n"
        f"**IQR Outliers:** {json.dumps(outliers, default=str)}\n"
        f"**ML Anomaly Detection:** {json.dumps(anomaly_summary, default=str)}\n"
        f"**Clustering:** {json.dumps(clustering, default=str)}\n\n"
        "Write a comprehensive professional analysis report with EXACTLY this structure:\n"
        "# Dataset Analysis Report\n"
        "## 1. Executive Summary\n"
        "## 2. Dataset Overview\n"
        "## 3. Key Metrics & Statistics\n"
        "## 4. Distributions & Patterns\n"
        "## 5. Correlations & Relationships\n"
        "## 6. Anomalies & Outliers\n"
        "## 7. Cluster Analysis\n"
        "## 8. Recommendations\n\n"
        "Rules:\n"
        "- Use EXACT numbers from the statistics above.\n"
        "- Use markdown tables where helpful.\n"
        "- Be specific and data-driven, no vague statements.\n"
        "- Do not invent data or guess column semantics beyond what's visible.\n"
    )

    logger.info("generating_story", filename=filename, model=MODEL)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=3000,
        ),
    )
    return response.text or ""


def chat_with_data(context: dict, messages: list) -> str:
    """
    Conversational AI chat grounded in the full analysis context.
    Non-streaming version (for /chat endpoint).
    """
    client = _get_client()
    system_prompt = _build_system_context(context)
    contents = _build_chat_contents(system_prompt, messages)

    logger.info("chat_request", message_count=len(messages))
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=1200,
        ),
    )
    return response.text or ""


def chat_with_data_stream(context: dict, messages: list):
    """
    Streaming chat. Yields text tokens as they arrive (for SSE /chat/stream endpoint).
    Usage: for token in chat_with_data_stream(context, messages): yield token
    """
    client = _get_client()
    system_prompt = _build_system_context(context)
    contents = _build_chat_contents(system_prompt, messages)

    logger.info("chat_stream_request", message_count=len(messages))
    for chunk in client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=1200,
        ),
    ):
        if chunk.text:
            yield chunk.text


def analyze_column(context: dict, column_name: str) -> str:
    """
    Generate a deep-dive analysis for a single column using actual computed stats.
    Returns a Markdown string.
    """
    client = _get_client()
    col_info = context.get("column_info", {}).get(column_name, {})
    outlier_info = context.get("outliers", {}).get(column_name)
    num_stats = context.get("numeric_aggregates", {}).get(column_name)
    cat_stats = context.get("cat_aggregates", {}).get(column_name)
    filename = context.get("filename", "dataset")

    stats_section = ""
    if num_stats:
        stats_section = (
            f"Computed stats: sum={_fmt_num(num_stats.get('sum'))}, "
            f"mean={_fmt_num(num_stats.get('mean'))}, "
            f"median={_fmt_num(num_stats.get('median'))}, "
            f"std={_fmt_num(num_stats.get('std'))}, "
            f"min={_fmt_num(num_stats.get('min'))}, "
            f"max={_fmt_num(num_stats.get('max'))}, "
            f"skewness={_fmt_num(num_stats.get('skewness'))}, "
            f"nulls={num_stats.get('null_count', 0)}"
        )
    elif cat_stats:
        top = cat_stats.get("top_values", {})
        top_str = ", ".join(f"'{k}': {v}" for k, v in list(top.items())[:8])
        stats_section = (
            f"Unique values: {cat_stats.get('unique_count')}, "
            f"mode: '{cat_stats.get('mode')}', "
            f"nulls: {cat_stats.get('null_count', 0)}, "
            f"top values: {top_str}"
        )

    prompt = (
        f"You are a data analyst. Give a focused deep-dive on one column from '{filename}'.\n\n"
        f"**Column:** `{column_name}`\n"
        f"**Type info:** {json.dumps(col_info, default=str)}\n"
        f"**{stats_section}**\n"
        f"**Outlier info:** {json.dumps(outlier_info, default=str)}\n\n"
        "Cover:\n"
        "1. **Distribution** — shape, skewness, spread, what the numbers reveal\n"
        "2. **Data Quality** — missing values, outliers, anomalies\n"
        "3. **Business Insight** — what this column tells us practically\n"
        "4. **Action Items** — specific recommendations\n\n"
        "Use EXACT numbers. Be direct and analytical. 150–250 words. Markdown format."
    )

    logger.info("column_analysis", column=column_name)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=700,
        ),
    )
    return response.text or ""
