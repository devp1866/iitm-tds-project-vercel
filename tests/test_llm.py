import os
import pytest
from unittest.mock import patch, MagicMock
from services.llm import _get_token, _call_llm, _fmt_num, chat_with_data


def test_get_token():
    """Ensure token retrieval works or raises appropriate error."""
    with patch.dict(os.environ, {"AIPROXY_TOKEN": "test_token"}):
        assert _get_token() == "test_token"
        
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError):
            _get_token()


def test_fmt_num():
    """Test number formatting utility."""
    assert _fmt_num(None) == "N/A"
    assert _fmt_num(10) == "10"
    assert _fmt_num(1500000.0) == "1,500,000"
    assert _fmt_num(2500.55) == "2,500.55"
    assert _fmt_num(0.000123) == "0.000123"


@patch("services.llm.requests.post")
def test_call_llm_success(mock_post):
    """Test successful LLM call parsing."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Hello, data!"}}]
    }
    mock_post.return_value = mock_resp

    with patch.dict(os.environ, {"AIPROXY_TOKEN": "test_token"}):
        result = _call_llm([{"role": "user", "content": "Hi"}])
        assert result == "Hello, data!"
        mock_post.assert_called_once()


@patch("services.llm._call_llm")
def test_chat_with_data(mock_call):
    """Test public chat function."""
    mock_call.return_value = "This is a chat response"
    context = {"filename": "test.csv"}
    messages = [{"role": "user", "content": "What is this?"}]
    
    with patch.dict(os.environ, {"AIPROXY_TOKEN": "test_token"}):
        response = chat_with_data(context, messages)
        assert response == "This is a chat response"
        # Verify call arguments
        call_msgs = mock_call.call_args[0][0]
        assert call_msgs[0]["role"] == "system"
        assert "test.csv" in call_msgs[0]["content"]
        assert call_msgs[1]["content"] == "What is this?"
