# Autolysis | Intelligent Data Analysis

**Autolysis** is an intelligent, automated data analysis tool designed to simplify the process of deriving insights from CSV datasets. Powered by LLMs (Large Language Models), it automatically analyzes your data, detects trends, outliers, and patterns, and generates comprehensive reports with visualizations—all without requiring any manual coding or setup.

## 🚀 Features

- **Automated Analysis**: Upload a CSV and get a full analysis report instantly.
- **Intelligent Insights**: Uses AI to narrate the story behind your data.
- **Visualizations**: Automatically generates correlation heatmaps, distribution plots, and clustering analysis.
- **Stateless & Secure**: Processes data in-memory for security and is optimized for serverless deployment (Vercel).
- **PDF Reports**: Download a professional PDF report of your analysis.

## 🛠️ Setup & Installation

Follow these steps to run Autolysis locally on your machine.

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/devp1866/iitm-tds-project-vercel.git
    cd iitm-tds-project-vercel
    ```

2.  **Create a Virtual Environment** (Recommended)

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in the root directory and add your AI Proxy Token:

    ```bash
    AIPROXY_TOKEN=your_api_token_here
    ```

5.  **Run the Application**
    ```bash
    python app.py
    ```
    The app will start at `http://127.0.0.1:5000`.

## 👨‍💻 Developer

**DEVKUMAR PATEL**
_Junior Data Scientist | Web Developer | AI Enthusiast_

[Portfolio / GitHub Profile](https://devp1866.framer.website/)
