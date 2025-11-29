import openai
import pandas as pd
from charset_normalizer import detect
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import json
import sys
from dotenv import load_dotenv
import numpy as np
import io
import base64

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

load_dotenv()


def save_plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)  # High DPI for quality
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def read_dataset(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = detect(raw_data)
        encoding = result["encoding"]
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print("File successfully read.")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return df


def generate_visualizations(data):
    images = {}

    # Correlation heatmap for numerical columns
    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_mat = data[numerical_cols].corr()
        sns.heatmap(correlation_mat, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        images["heatmap.png"] = save_plot_to_base64(plt.gcf())

    # Distribution plot for the first numerical column
    if len(numerical_cols) > 0:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[numerical_cols[0]], kde=True, color="blue")
        plt.title(f"Distribution of {numerical_cols[0]}")
        images[f"distribution_{numerical_cols[0]}.png"] = save_plot_to_base64(plt.gcf())

    # Pairplot for the first few numerical columns (if more than 2 exist)
    if len(numerical_cols) > 2:
        pairplot_data = data[numerical_cols[:4]]  # Limit to the first 4 columns
        sns.pairplot(pairplot_data, diag_kind="kde")
        images["pairplot.png"] = save_plot_to_base64(plt.gcf())

    return images


def detect_outliers(data):
    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    outliers_summary = {}
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        if not outliers.empty:
            outliers_summary[col] = {
                "count": len(outliers),
                "percentage": (len(outliers) / len(data)) * 100,
                "min_outlier": outliers[col].min(),
                "max_outlier": outliers[col].max(),
            }
    return outliers_summary


def perform_clustering(data):
    if not SKLEARN_AVAILABLE:
        return "Clustering skipped (scikit-learn not available)."

    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    if len(numerical_cols) < 2:
        return "Not enough numerical columns for clustering."

    # Drop rows with NaNs for clustering
    cluster_data = data[numerical_cols].dropna()
    if cluster_data.empty:
        return "No data available for clustering after dropping NaNs."

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    # Simple K-Means with k=3 (can be optimized)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)

    # Add labels to a copy of the data to analyze centers
    cluster_data["Cluster"] = labels
    cluster_summary = cluster_data.groupby("Cluster").mean().to_dict()

    return cluster_summary


def json_serializer(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def create_README_content(data: pd.DataFrame, narrated_story, dataset_name):
    summary = data.describe(include="all").transpose()
    missing_values = data.isnull().sum()

    readme_content = "# Automated Analysis Report\n\n"
    readme_content += f"**Dataset:** `{dataset_name}`\n\n"
    readme_content += narrated_story + "\n\n"
    readme_content += "## Detailed Metrics\n"
    readme_content += "### Summary Statistics\n"
    readme_content += summary.to_markdown(tablefmt="github") + "\n\n"
    readme_content += "### Missing Values\n"
    readme_content += missing_values.to_markdown(tablefmt="github")

    return readme_content


def generate_story(data, summary_stats):
    token = os.getenv("AIPROXY_TOKEN")
    if token is None:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

    # Perform advanced analysis
    outliers = detect_outliers(data)
    clustering = perform_clustering(data)

    # Generate a summary of the dataset
    columns = data.columns.tolist()
    example_rows = data.head(3).to_dict(orient="records")

    prompt = (
        f"You are a data analyst. I am providing you with a dataset. \n"
        f"Columns: {columns}\n"
        f"The first three rows are: {example_rows}\n"
        f"Summary Statistics:\n{summary_stats}\n"
        f"Outlier Analysis:\n{json.dumps(outliers, indent=2, default=json_serializer)}\n"
        f"Cluster Analysis (Mean values per cluster):\n{json.dumps(clustering, indent=2, default=json_serializer) if isinstance(clustering, dict) else clustering}\n\n"
        f"Please provide a comprehensive analysis report with the following structure:\n"
        f"1. **Dataset Overview**: A concise 4-5 line to summarize the dataset.\n"
        f"2. **Key Metrics**\n"
        f"3. **Trends and Patterns**\n"
        f"4. **Notable Anomalies**\n"
        f"5. **Data Analysis**\n"
        f"6. **Implications**\n"
        f"7. **Recommendations**\n\n"
        f"We do not have to describe the meaning of the names of the columns used in the dataset, describe everything statistically."
    )

    # Define the request payload
    api_url = "https://aipipe.org/openrouter/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        story = result["choices"][0]["message"]["content"]
        return story

    except requests.exceptions.RequestException as e:
        print(f"HTTP request error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error parsing response: {e}")
        sys.exit(1)


def generate_(file_path):
    data = read_dataset(file_path)
    if data is None:
        return None, None

    images = generate_visualizations(data)
    summary_stats = data.describe(include="all").transpose().to_string()
    narrated_story = generate_story(data, summary_stats)
    readme_content = create_README_content(
        data, narrated_story, os.path.basename(file_path)
    )

    return readme_content, images


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <csv_file>")
        sys.exit(1)
    csv_file = sys.argv[1]
    readme, images = generate_(csv_file)
    if readme:
        print(readme)
        print("\nGenerated Images:", list(images.keys()))
