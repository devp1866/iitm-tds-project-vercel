import openai
import pandas as pd
from charset_normalizer import detect
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import json
import sys
from dotenv import load_dotenv
import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

load_dotenv()

if len(sys.argv) != 2:
    print("Usage: python autolysis.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]


def save_plot(fig, file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)


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


def generate_visualizations_in_dir(data, directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Correlation heatmap for numerical columns
    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_mat = data[numerical_cols].corr()
        sns.heatmap(correlation_mat, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(directory, "heatmap.png"))
        plt.close()

    # Distribution plot for the first numerical column
    if len(numerical_cols) > 0:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[numerical_cols[0]], kde=True, color="blue")
        plt.title(f"Distribution of {numerical_cols[0]}")
        plt.savefig(os.path.join(directory, f"distribution_{numerical_cols[0]}.png"))
        plt.close()

    # Pairplot for the first few numerical columns (if more than 2 exist)
    if len(numerical_cols) > 2:
        pairplot_data = data[numerical_cols[:4]]  # Limit to the first 4 columns
        sns.pairplot(pairplot_data, diag_kind="kde")
        plt.savefig(os.path.join(directory, "pairplot.png"))
        plt.close()


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


def create_README(data: pd.DataFrame, directory, narrated_story, dataset_path):
    summary = data.describe(include="all").transpose()
    missing_values = data.isnull().sum()

    # Find generated images
    images = [f for f in os.listdir(directory) if f.endswith(".png")]

    with open(f"{directory}/README.md", "w", encoding="utf-8") as f:
        f.write("# Automated Analysis Report\n\n")
        f.write(f"**Dataset:** `{os.path.basename(dataset_path)}`\n\n")

        f.write(narrated_story)
        f.write("\n\n")

        f.write("## Visualizations\n")
        for img in images:
            f.write(f"![{img}]({img})\n")
        f.write("\n\n")

        f.write("## Detailed Metrics\n")
        f.write("### Summary Statistics\n")
        f.write(summary.to_markdown(tablefmt="github"))
        f.write("\n\n")
        f.write("### Missing Values\n")
        f.write(missing_values.to_markdown(tablefmt="github"))


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
        # f"Please provide a comprehensive analysis report with the following structure:\n"
        # f"1. **Dataset Overview**: A concise 4-5 line to summarize the dataset.\n"
        # f"2. **Key Metrics**: A bulleted list of the most critical numbers and insights.\n"
        # f"3. **Trends and Patterns**: Identify and describe any significant trends or recurring patterns in the data.\n"
        # f"4. **Notable Anomalies**: Highlight any specific outliers or unusual data points and their potential significance.\n"
        # f"5. **Data Analysis**: Detailed analysis of distributions, correlations, and clusters.\n"
        # f"6. **Implications**: What do these findings mean for the business or domain?\n"
        # f"7. **Recommendations**: Actionable next steps based on the data.\n\n"
        # f"We do not have to describe the meaning of the names of the columns used in the dataset, describe everything statistically."
    
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
    directory = os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    data = read_dataset(file_path)
    if data is None:
        return
    generate_visualizations_in_dir(data, directory)
    summary_stats = data.describe(include="all").transpose().to_string()
    narrated_story = generate_story(data, summary_stats)
    create_README(data, directory, narrated_story, file_path)


if __name__ == "__main__":
    generate_(csv_file)
