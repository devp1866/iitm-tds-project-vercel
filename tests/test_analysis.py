"""
tests/test_analysis.py
Unit tests for the analysis service.
"""

import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.analysis import (
    detect_outliers_iqr,
    detect_anomalies_ml,
    perform_clustering,
    get_column_info,
    build_analysis_context,
)


@pytest.fixture
def sample_df():
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'age': rng.randint(18, 80, 200).astype(float),
        'salary': rng.normal(60000, 15000, 200),
        'score': rng.uniform(0, 100, 200),
        'category': rng.choice(['A', 'B', 'C'], 200),
        'name': [f'Person_{i}' for i in range(200)],
    })
    # Inject outliers
    df.loc[0, 'salary'] = 999_999
    df.loc[1, 'salary'] = -5_000
    return df


def test_detect_outliers_iqr(sample_df):
    result = detect_outliers_iqr(sample_df)
    assert 'salary' in result
    assert result['salary']['count'] >= 1
    assert 0 < result['salary']['percentage'] < 10


def test_detect_anomalies_ml(sample_df):
    result = detect_anomalies_ml(sample_df)
    assert 'anomaly_count' in result
    assert result['anomaly_count'] > 0
    assert 'anomaly_indices' in result
    assert isinstance(result['anomaly_pct'], float)


def test_perform_clustering(sample_df):
    result = perform_clustering(sample_df)
    assert 'k' in result
    assert 2 <= result['k'] <= 6
    assert 'cluster_sizes' in result
    assert sum(result['cluster_sizes'].values()) <= len(sample_df)


def test_get_column_info(sample_df):
    info = get_column_info(sample_df)
    assert 'age' in info
    assert info['age']['is_numeric'] is True
    assert info['category']['is_numeric'] is False
    assert 'top_values' in info['category']
    assert info['age']['n_missing'] == 0


def test_clustering_insufficient_cols():
    df = pd.DataFrame({'a': range(20)})
    result = perform_clustering(df)
    assert 'error' in result


def test_anomaly_insufficient_data():
    df = pd.DataFrame({'a': range(5), 'b': range(5)})
    result = detect_anomalies_ml(df)
    assert 'error' in result


def test_build_analysis_context(sample_df):
    col_info = get_column_info(sample_df)
    outliers = detect_outliers_iqr(sample_df)
    anomalies = detect_anomalies_ml(sample_df)
    clustering = perform_clustering(sample_df)
    ctx = build_analysis_context(sample_df, col_info, outliers, anomalies, clustering, 'test.csv')

    assert ctx['filename'] == 'test.csv'
    assert ctx['shape']['rows'] == 200
    assert ctx['shape']['cols'] == 5
    assert 'age' in ctx['columns']
