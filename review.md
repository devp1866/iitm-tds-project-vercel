# Autolysis v2 — 2-Phase International Upgrade Plan

> **Goal:** Transform Autolysis from a smart prototype into a production-grade, internationally-competitive
> Data Science SaaS platform. Every decision here is justified, free-tier compatible, and implementable
> within weeks. Evaluated at the end of each phase before proceeding.

---

## Answers to Your Tweaks

| Your Question | Decision | Reason |
|---|---|---|
| **LLM Model** | Stay on `google/gemini-flash-1.5` via aipipe.org | Gemini Flash is fastest, cheapest (essentially free via aipipe), and outperforms GPT-4o-mini on structured data analysis tasks as of 2024. |
| **Backend Platform** | **Render.com** (Free tier) | Free 512MB RAM instance, no sleep on hobby tier now, stays online, Docker-compatible. Since Firebase is already the DB, Render only runs the Flask app — no migration of data needed. |
| **LangChain Chatbot** | Yes, but **Pandas DataFrame Agent** not standard RAG | We give the agent the actual `DataFrame` object, it generates Python code, executes it in a sandbox, and returns real computed answers. Far more accurate than RAG for tabular data. |
| **Data Cleaning** | **Optional toggle in UI** | User sees a "✨ Clean Data Before Analysis" checkbox before submitting. Defaults to OFF. |
| **UI/UX** | Targeted improvements listed in Phase 1 | Focus on micro-interactions, better loading states, error feedback. |

---

## 🔵 PHASE 1 — Foundation & Critical Fixes

**Objective:** Make the app actually production-safe, fix the Vercel threading bug, modernize the LLM,
and significantly improve the user experience. No feature bloat, just making current features work
**perfectly and robustly**.

**Estimated Time:** 3-4 days of focused work.

### 1.1 ✅ Migrate Backend to Render.com (Critical Bug Fix)

**The Problem:** `threading.Thread` on Vercel is silently killed. Users see the progress bar stuck.

**The Fix:**
- Containerize the Flask app with a minimal `Dockerfile`
- Deploy as a **long-running web service on Render** (free tier)
- Keep Vercel **only for the frontend** static assets (or remove Vercel entirely)
- Firebase Firestore stays as the database — no changes there

**Architecture after this fix:**
```
User Browser → Render Flask App → Firebase Firestore (DB)
                    └── Background Thread (now safe, runs for minutes)
                    └── aipipe.org LLM API
```

**Files to create/change:**
- `Dockerfile` (new)
- `vercel.json` (simplify or remove)
- `render.yaml` (new — Render deploy config)

---

### 1.2 ✅ Upgrade LLM to Gemini Flash

**Change `services/llm.py`:**
```python
# FROM:
MODEL = "openai/gpt-4o-mini"

# TO:
MODEL = "google/gemini-flash-1.5"
```

**Why:** Gemini Flash is 3x faster than GPT-4o-mini, free via aipipe, has a 1M token context window
(fits far more statistics), and scores higher on coding + data analysis benchmarks.

---

### 1.3 ✅ Optional Data Cleaning Pipeline

**Frontend (index.html):** Add a toggle checkbox before file submission:
```html
<label class="toggle-row">
  <input type="checkbox" id="clean-toggle">
  <span>✨ Auto-clean data before analysis</span>
  <small>(Remove near-empty columns, impute missing values, parse dates)</small>
</label>
```

**Backend (services/cleaning.py — new file):**
```python
def auto_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns (cleaned_df, report_dict).
    Steps applied:
    1. Drop columns with >85% missing values
    2. Impute numeric NaNs with column median
    3. Parse columns that look like dates into datetime
    4. Strip whitespace from string columns
    """
```
The cleaning report dict (what was changed and why) gets passed to the LLM narrative for richer output.

---

### 1.4 ✅ Streaming LLM Responses (Server-Sent Events)

**The Problem:** The "Generating AI narrative..." stage hangs for 10-15 seconds with no visible feedback.

**The Fix:** Use `stream=True` in the OpenAI-compatible API call. The Flask route yields tokens
via `text/event-stream`. The frontend `chat.js` receives tokens and appends them character by character
(like ChatGPT). The `generate_story` call in the job queue still runs non-streaming (for the report),
but the **AI Chat drawer** will become fully streaming.

**Files changed:** `services/llm.py`, `static/js/chat.js`, `app.py` (new `/chat/stream` route)

---

### 1.5 ✅ UI/UX Enhancements

Targeted improvements that take the UI from "good" to "polished":

| Area | Change |
|---|---|
| **Upload Zone** | Show file name + size preview *before* submit with a "Remove" button |
| **Progress Bar** | Replace generic spinner with a **step-by-step stage indicator** (Reading → ML → Charts → AI) |
| **Error States** | Full-screen friendly error cards with actionable messages (not just a toast) |
| **Stats Table** | Add column sorting (click header to sort by value) |
| **Anomaly Section** | Highlight anomalous rows in a heatmap-style color gradient |
| **Mobile** | Fix horizontal scroll on the stats table for mobile screens |

---

### Phase 1 Evaluation Checklist
Before moving to Phase 2, verify:
- [ ] App runs continuously on Render without crashes for 30+ minutes
- [ ] Upload → Analysis → Result works for at least 3 different CSVs (small/medium/large)
- [ ] Streaming chat responds token-by-token
- [ ] Optional cleaning toggle works on/off
- [ ] Share links persist correctly in Firebase

---

## 🟣 PHASE 2 — Advanced AI & Performance

**Objective:** Add the features that truly make this an internationally stand-out project.
These are enterprise-grade capabilities that most portfolios simply do not have.

**Estimated Time:** 5-7 days of focused work.

### 2.1 ✅ Polars Migration (Performance Upgrade)

Replace Pandas with **Polars** in `services/analysis.py` for all data loading and aggregation.

**Why Polars:**
- Written in Rust — 5-50x faster than Pandas
- Multi-threaded by default — uses all CPU cores
- Lazy evaluation — computes only what's needed
- 50-70% lower memory usage

**Migration Strategy:**
```python
# services/analysis.py — only the read/aggregate layer changes
import polars as pl

def read_dataset(file_path, file_type="csv") -> pl.DataFrame:
    return pl.read_csv(file_path, infer_schema_length=1000)

def get_column_info(df: pl.DataFrame) -> dict:
    # Use Polars native describe() and schema
```
ML (Isolation Forest, KMeans) still uses scikit-learn, so we convert to NumPy at the boundary:
`X = df.select(numeric_cols).to_numpy()` — zero pandas needed.

---

### 2.2 ✅ LangChain Pandas DataFrame Agent (Intelligent Chat)

**This replaces the current stats-injection approach for the chatbot.**

**How it works:**
1. When the user sends a chat message, the backend loads the DataFrame from Firebase Storage (or `/tmp`)
2. A **LangChain Pandas DataFrame Agent** is initialized with the DataFrame
3. The agent's LLM (Gemini Flash) writes Python code to answer the query
4. The code is executed in a secure sandbox (using `exec()` with restricted globals)
5. The result (a number, a table, a list) is returned as the response

**Example interaction:**
- User: *"What is the average revenue for customers in India?"*
- Agent writes: `df[df['country'] == 'India']['revenue'].mean()`
- Returns: `₹42,350.33`

**This is not basic RAG.** This is actual agentic reasoning.

**New dependency:** `langchain`, `langchain-community`, `langchain-google-genai`

---

### 2.3 ✅ Production PDF Report Export

**What is generated:** A full, branded, multi-page PDF including:
- Title page (filename, date, key stats)
- AI-generated narrative (Section 1)
- Interactive charts rendered as static images (Section 2)
- Anomaly detection table (Section 3)
- Summary statistics table (Section 4)

**Implementation:**
- Use `WeasyPrint` (HTML-to-PDF converter) — free and open source
- Create a `templates/pdf_report.html` — a print-optimized version of `report.html`
- Add a `/export/pdf/<job_id>` Flask route that renders the template and returns a binary PDF
- Frontend adds a **"⬇ Download PDF"** button on the results page

---

### 2.4 ✅ Upstash Redis for Rate Limiting & Caching

**Current issue:** `flask-limiter` uses in-memory storage — on Render, every deploy resets all rate
limit counters.

**The Fix:** Connect `flask-limiter` to Upstash Redis (already in requirements.txt):
```python
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=os.getenv("UPSTASH_REDIS_URL"),  # persistent across deploys
)
```
Additionally, cache the LLM story for a given file hash, so the same CSV never pays an LLM cost twice.

---

### Phase 2 Evaluation Checklist
Before calling the project complete:
- [ ] Polars reads a 20MB CSV in under 2 seconds
- [ ] LangChain agent correctly answers 5 diverse questions about a real CSV
- [ ] PDF downloads cleanly with all charts and tables rendered correctly
- [ ] Rate limits persist across Render server restarts

---

## 📋 Final Architecture (After Phase 2)

```
┌─────────────────────────────────────────────────────────┐
│                     User Browser                        │
│  Upload → Optional Clean → Progress → Results → PDF     │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────────┐
│                  Render.com (Flask App)                  │
│   routes: /submit, /status, /result, /chat, /pdf        │
│   services: Polars, Isolation Forest, LangChain Agent   │
└─────────┬───────────────────────────┬───────────────────┘
          │                           │
┌─────────▼──────────┐   ┌───────────▼────────────────────┐
│  Firebase Firestore │   │  Upstash Redis                 │
│  (Jobs + Reports)  │   │  (Rate limits + LLM cache)     │
└────────────────────┘   └────────────────────────────────┘
          │
┌─────────▼──────────┐
│   aipipe.org       │
│  Gemini Flash 1.5  │
└────────────────────┘
```

---

## 🚀 Ready to Start?

Reply with **"Start Phase 1"** and I will begin implementation immediately, file by file,
with tests and commits at every step.
