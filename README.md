# Autolysis v2 | Intelligent Data Analysis Platform

**Autolysis** is a production-ready, AI-powered automated data analysis platform. Upload a CSV, Excel, JSON, or Parquet file and get instant interactive charts, ML anomaly detection, statistical insights, and AI-narrated reports — with a conversational AI you can chat with about your data.

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-format Upload** | CSV, Excel (.xlsx/.xls), JSON, Parquet (up to 50MB) |
| **Interactive Plotly Charts** | Zoomable, hoverable, exportable charts — auto-selected by data type |
| **AI Report** | GPT-4o-mini generated narrative with structured sections |
| **AI Chat** | Ask questions about your data after analysis |
| **ML Anomaly Detection** | Isolation Forest with auto-contamination rate |
| **Smart Clustering** | KMeans with elbow method to auto-select optimal k |
| **Column Deep Dive** | Click any column for AI insights + focused charts |
| **Async Processing** | Non-blocking analysis with real-time progress stages |
| **Public Share Links** | Share `/report/<token>` with teammates |
| **Working Contact Form** | Web3Forms powered email delivery |
| **Firebase Persistence** | Reports stored in Firestore (with in-memory fallback) |

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
git clone https://github.com/devp1866/iitm-tds-project-vercel.git
cd iitm-tds-project-vercel

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `AIPROXY_TOKEN` | ✅ | From [aipipe.org](https://aipipe.org/login) |
| `WEB3FORMS_KEY` | ✅ | From [web3forms.com](https://web3forms.com) (free) |
| `FIREBASE_CREDENTIALS_PATH` | ⚠️ Optional | Path to Firebase service account JSON — enables persistent share links |
| `UPSTASH_REDIS_URL` | ⚠️ Optional | From [upstash.com](https://upstash.com) — for Vercel async jobs |
| `UPSTASH_REDIS_TOKEN` | ⚠️ Optional | Upstash token |
| `SECRET_KEY` | ✅ | Random string for Flask session security |

> **Without Firebase**: App works fully but share links are in-memory only (lost on restart).
> **Without Upstash**: App works locally with threading. Vercel async jobs need Upstash.

### Firebase Setup (for persistent share links)

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Create a project → Enable Firestore Database
3. Project Settings → Service Accounts → Generate new private key
4. Save JSON as `firebase-credentials.json` in project root
5. Set `FIREBASE_CREDENTIALS_PATH=firebase-credentials.json` in `.env`

### Running Locally

```bash
python app.py
```

App runs at `http://127.0.0.1:5000`

## 🏗️ Architecture

```
autolysis/
├── app.py                   # Flask routes only
├── services/
│   ├── analysis.py          # Multi-format reading, IQR outliers, Isolation Forest, KMeans
│   ├── visualizations.py    # Plotly chart generation (smart auto-selection)
│   ├── llm.py               # GPT-4o-mini — story, chat, column analysis
│   └── email_service.py     # Web3Forms contact form
├── models/
│   └── db.py                # Firebase Firestore (with in-memory fallback)
├── jobs/
│   └── job_queue.py         # Async analysis pipeline (threading + Upstash)
├── static/
│   ├── css/style.css        # Full design system (Peach/Sky/Mint palette)
│   └── js/
│       ├── app.js           # Upload, polling, result rendering
│       ├── charts.js        # Plotly rendering + column modal
│       └── chat.js          # AI chat drawer
├── templates/
│   ├── base.html            # Shared layout
│   ├── index.html           # Main app
│   ├── report.html          # Public share link view
│   └── about.html           # About + contact
└── tests/
    └── test_analysis.py     # Unit tests (7 passing)
```

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

## 🚀 Deploy to Vercel

1. Push to GitHub
2. Import project in [Vercel](https://vercel.com)
3. Add environment variables in Vercel dashboard
4. Set `UPSTASH_REDIS_URL` and `UPSTASH_REDIS_TOKEN` for async job support on Vercel

## 👨‍💻 Developer

**DEVKUMAR PATEL** — Junior Data Scientist · Web Developer · AI Enthusiast

[Portfolio](https://devkumarpatel.vercel.app/) · [GitHub](https://github.com/devp1866)
