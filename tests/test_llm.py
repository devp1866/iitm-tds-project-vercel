import os
import pytest
from unittest.mock import patch, MagicMock
from services.llm import _get_client, _fmt_num, chat_with_data, chat_with_data_stream


def test_get_client_missing_key():
    """Ensure client creation raises EnvironmentError when key missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
            _get_client()


def test_get_client_with_key():
    """Ensure _get_client returns a client object when GOOGLE_API_KEY is set."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key-for-test"}):
        with patch("services.llm.genai.Client") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            client = _get_client()
            mock_client_cls.assert_called_once_with(api_key="fake-key-for-test")
            assert client is not None


def test_fmt_num():
    """Test number formatting utility."""
    assert _fmt_num(None) == "N/A"
    assert _fmt_num(10) == "10"
    assert _fmt_num(1500000.0) == "1,500,000"
    assert _fmt_num(2500.55) == "2,500.55"
    assert _fmt_num(0.000123) == "0.000123"


def test_chat_with_data_calls_gemini():
    """Test chat_with_data calls generate_content and returns text."""
    mock_response = MagicMock()
    mock_response.text = "This is a Gemini response"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    context = {
        "filename": "test.csv",
        "shape": {"rows": 100, "cols": 5},
        "columns": ["a", "b", "c"],
        "numeric_aggregates": {},
        "cat_aggregates": {},
    }
    messages = [{"role": "user", "content": "What is this dataset?"}]

    with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key-for-test"}):
        with patch("services.llm._get_client", return_value=mock_client):
            response = chat_with_data(context, messages)
            assert response == "This is a Gemini response"
            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args[1]
            assert call_kwargs["model"] == "gemini-2.0-flash"


def test_chat_with_data_stream():
    """Test streaming chat yields tokens from generate_content_stream."""
    def _fake_stream(*args, **kwargs):
        chunks = []
        for text in ["Hello", " world", "!"]:
            chunk = MagicMock()
            chunk.text = text
            chunks.append(chunk)
        return iter(chunks)

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.side_effect = _fake_stream

    context = {
        "filename": "test.csv",
        "columns": [],
        "numeric_aggregates": {},
        "cat_aggregates": {},
    }
    messages = [{"role": "user", "content": "Hi"}]

    with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key-for-test"}):
        with patch("services.llm._get_client", return_value=mock_client):
            tokens = list(chat_with_data_stream(context, messages))
            assert tokens == ["Hello", " world", "!"]
            mock_client.models.generate_content_stream.assert_called_once()
