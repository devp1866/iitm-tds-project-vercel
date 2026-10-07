import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from jobs.job_queue import submit_job, _run_analysis_job


@patch("jobs.job_queue.threading.Thread")
@patch("jobs.job_queue.create_job")
def test_submit_job(mock_create_job, mock_thread):
    """Test job submission creates a job and starts a thread."""
    mock_thread_instance = MagicMock()
    mock_thread.return_value = mock_thread_instance

    job_id = submit_job("fake_path.csv", "fake_path.csv", "csv")
    
    assert job_id is not None
    assert type(job_id) == str
    mock_create_job.assert_called_once_with(job_id, "fake_path.csv", "csv")
    mock_thread.assert_called_once()
    mock_thread_instance.start.assert_called_once()


@patch("jobs.job_queue.update_job_status")
@patch("jobs.job_queue.save_report")
@patch("jobs.job_queue.generate_story")
@patch("jobs.job_queue.generate_smart_charts")
@patch("jobs.job_queue.read_dataset")
@patch("os.remove")
@patch("os.path.exists")
def test_run_analysis_job_success(
    mock_exists, mock_remove, mock_read, mock_charts, mock_story, mock_save, mock_update
):
    """Test the full background analysis pipeline runs sequentially."""
    mock_exists.return_value = True
    
    # Mock dataset
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    mock_read.return_value = df
    
    # Mock LLM and Charts
    mock_charts.return_value = [{"data": [], "layout": {}}]
    mock_story.return_value = "This is a great dataset."
    mock_save.return_value = "fake_share_token"

    _run_analysis_job("job_123", "fake.csv", "fake.csv", "csv")

    # Assertions
    mock_read.assert_called_once_with("fake.csv", "csv")
    mock_story.assert_called_once()
    mock_save.assert_called_once()
    
    # Final status should be 'done'
    mock_update.assert_called_with("job_123", "done", progress=100, progress_label="Analysis complete!")
    
    # Temp file should be deleted
    mock_remove.assert_called_once_with("fake.csv")
