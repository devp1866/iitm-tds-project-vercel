"""
services/llm.py
All LLM interactions for Autolysis v2.
- Analysis story generation
- Conversational AI chat (with REAL data values in context)
- Column deep-dive analysis
"""

import os
import json
import requests
import structlog

logger = structlog.get_logger()

API_URL = "https://aipipe.org/openrouter/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"


def _get_token() -> str:
    token = os.getenv("AIPROXY_TOKEN")
    if not token:
        raise EnvironmentError("AIPROXY_TOKEN environment variable is not set.")
    return token


def _call_llm(messages: list, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Make a chat completions API call. Returns the assistant's message text."""
    token = _get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Title": "Autolysis",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        raise TimeoutError("LLM API request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "unknown"
        raise RuntimeError(f"LLM API HTTP error {status}: {e}") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected LLM API response format: {e}") from e


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
    Convert analysis context into a rich system prompt that contains
    ACTUAL computed values so the AI can answer specific questions
    (e.g. 'total sales', 'average score') directly.
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

    lines = [
        f"You are an expert data analyst. The user uploaded a dataset called '{filename}'.",
        f"Shape: {shape.get('rows', '?')} rows × {shape.get('cols', '?')} columns.",
        f"Columns: {', '.join(cols)}",
        "",
        "=== IMPORTANT: You have FULL access to pre-computed statistics below. ===",
        "Always answer specific questions using these exact values. NEVER say you cannot access the data.",
        "",
    ]

    # Numeric aggregates — the most important section for answering specific questions
    if numeric_agg:
        lines.append("--- NUMERIC COLUMN STATISTICS (use these for specific questions) ---")
        for col, stats in numeric_agg.items():
            parts = [
                f"  sum={_fmt_num(stats.get('sum'))}",
                f"mean={_fmt_num(stats.get('mean'))}",
                f"median={_fmt_num(stats.get('median'))}",
                f"min={_fmt_num(stats.get('min'))}",
                f"max={_fmt_num(stats.get('max'))}",
                f"std={_fmt_num(stats.get('std'))}",
                f"count={stats.get('count', 'N/A')}",
                f"nulls={stats.get('null_count', 0)}",
            ]
            lines.append(f"  [{col}]: {', '.join(parts)}")
        lines.append("")

    # Categorical columns
    if cat_agg:
        lines.append("--- CATEGORICAL COLUMN STATISTICS ---")
        for col, stats in cat_agg.items():
            top = stats.get("top_values", {})
            top_str = ", ".join(f"'{k}':{v}" for k, v in list(top.items())[:5])
            lines.append(
                f"  [{col}]: {stats.get('unique_count')} unique values, "
                f"most common: {top_str}, "
                f"mode='{stats.get('mode')}', nulls={stats.get('null_count', 0)}"
            )
        lines.append("")

    # Correlations
    if top_corr:
        lines.append("--- TOP CORRELATIONS ---")
        for pair in top_corr[:8]:
            lines.append(
                f"  {pair['col1']} ↔ {pair['col2']}: r={_fmt_num(pair['correlation'])}"
            )
        lines.append("")

    # Sample data
    if sample:
        lines.append(f"--- SAMPLE ROWS (first {len(sample)}) ---")
        lines.append(json.dumps(sample, default=str))
        lines.append("")

    # Anomaly / outliers
    if outlier_cols:
        lines.append(f"IQR outliers detected in: {', '.join(outlier_cols)}")
    if anomaly.get("anomaly_count"):
        lines.append(
            f"Isolation Forest found {anomaly['anomaly_count']} anomalies "
            f"({anomaly.get('anomaly_pct', 0)}% of rows)."
        )

    # Clustering
    if isinstance(clustering, dict) and "k" in clustering:
        lines.append(
            f"KMeans clustering: {clustering['k']} clusters, "
            f"sizes: {clustering.get('cluster_sizes', {})}"
        )

    lines.append("")
    lines.append(
        "Rules: Answer ONLY from the statistics above. Give exact numbers when asked. "
        "Be concise and direct. Format in markdown. Never say the data is inaccessible."
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_story(context: dict) -> str:
    """
    Generate a comprehensive, structured analysis narrative from the context dict.
    Returns a Markdown string.
    """
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

    # Build a rich stats block for the story prompt
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

    prompt = (
        f"You are a senior data analyst writing a professional executive report.\n\n"
        f"**Dataset:** `{filename}` — {shape.get('rows','?')} rows × {shape.get('cols','?')} columns\n"
        f"**Columns:** {cols}\n"
        f"**Sample rows:** {json.dumps(sample[:3], default=str)}\n\n"
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
        "- Use EXACT numbers from the statistics provided above.\n"
        "- Use markdown tables where helpful.\n"
        "- Be specific and data-driven, no vague statements.\n"
        "- Do not invent data or guess column semantics beyond what's visible.\n"
    )

    messages = [{"role": "user", "content": prompt}]
    logger.info("generating_story", filename=filename)
    return _call_llm(messages, temperature=0.5, max_tokens=3000)


def chat_with_data(context: dict, messages: list) -> str:
    """
    Conversational AI chat grounded in the full analysis context with real computed values.

    Args:
        context: The analysis context dict (from Firestore/in-memory store)
        messages: List of {"role": "user"|"assistant", "content": str}

    Returns:
        Assistant's reply string (Markdown)
    """
    system_prompt = _build_system_context(context)

    llm_messages = [{"role": "system", "content": system_prompt}]
    # Include last 12 messages for multi-turn context
    llm_messages.extend(messages[-12:])

    logger.info("chat_request", message_count=len(messages))
    return _call_llm(llm_messages, temperature=0.5, max_tokens=1200)


def analyze_column(context: dict, column_name: str) -> str:
    """
    Generate a deep-dive analysis for a single column using actual computed stats.
    Returns a Markdown string.
    """
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

    messages = [{"role": "user", "content": prompt}]
    logger.info("column_analysis", column=column_name)
    return _call_llm(messages, temperature=0.4, max_tokens=700)
