"""Tests for common.api — SecClient and base ApiClient behaviour."""

import time
from unittest.mock import MagicMock, patch

import requests

from idi_corporate_structure.common.api import SecClient


class TestSecClientQueryEndpoint:
    """Tests for SecClient.query_endpoint()."""

    def test_returns_json_by_default(self):
        client = SecClient(rate_limit=0.0, user_agent="Test Agent test@example.com")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://www.sec.gov/test"
        mock_response.json.return_value = {"directory": {"item": []}}

        with patch.object(client.session, "get", return_value=mock_response):
            result = client.query_endpoint("https://www.sec.gov/test")

        assert result["status_code"] == 200
        assert result["data"] == {"directory": {"item": []}}

    def test_returns_text_when_return_json_false(self):
        client = SecClient(rate_limit=0.0, user_agent="Test Agent test@example.com")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://www.sec.gov/ex21.htm"
        mock_response.text = "<html>Subsidiaries</html>"

        with patch.object(client.session, "get", return_value=mock_response):
            result = client.query_endpoint("https://www.sec.gov/ex21.htm", return_json=False)

        assert result["status_code"] == 200
        assert result["data"] == "<html>Subsidiaries</html>"

    def test_returns_error_key_on_request_exception(self):
        client = SecClient(rate_limit=0.0, user_agent="Test Agent test@example.com")

        with patch.object(
            client.session,
            "get",
            side_effect=requests.exceptions.ConnectionError("unreachable"),
        ):
            result = client.query_endpoint("https://www.sec.gov/bad-url")

        assert "error" in result
        assert "unreachable" in result["error"]
        assert "data" not in result

    def test_returns_error_on_404(self):
        client = SecClient(rate_limit=0.0, user_agent="Test Agent test@example.com")
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")

        with patch.object(client.session, "get", return_value=mock_response):
            result = client.query_endpoint("https://www.sec.gov/missing")

        assert "error" in result

    def test_updates_last_request_timestamp_after_rate_limit(self):
        client = SecClient(rate_limit=0.0, user_agent="Test Agent test@example.com")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://www.sec.gov/test"
        mock_response.json.return_value = {}

        with patch.object(client.session, "get", return_value=mock_response):
            client.query_endpoint("https://www.sec.gov/test")

        before = time.time()
        client.rate_limit()

        assert client._last_request >= before

    def test_returns_bytes_when_return_bytes_true(self):
        client = SecClient(rate_limit=0.0, user_agent="Test Agent test@example.com")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://www.sec.gov/ex21.pdf"
        mock_response.content = b"%PDF binary content"

        with patch.object(client.session, "get", return_value=mock_response):
            result = client.query_endpoint("https://www.sec.gov/ex21.pdf", return_bytes=True)

        assert result["data"] == b"%PDF binary content"
        assert isinstance(result["data"], bytes)


class TestSecClientRateLimit:
    """Tests for SecClient.rate_limit()."""

    def test_sleeps_when_not_enough_time_elapsed(self):
        client = SecClient(rate_limit=0.2, user_agent="Test Agent test@example.com")
        client._last_request = time.time()  # just now

        with patch("time.sleep") as mock_sleep:
            client.rate_limit()

        mock_sleep.assert_called_once()
        sleep_duration = mock_sleep.call_args[0][0]
        assert 0 < sleep_duration <= 0.2

    def test_no_sleep_when_enough_time_elapsed(self):
        client = SecClient(rate_limit=0.1, user_agent="Test Agent test@example.com")
        client._last_request = time.time() - 1.0  # 1 second ago — well past 0.1s limit

        with patch("time.sleep") as mock_sleep:
            client.rate_limit()

        mock_sleep.assert_not_called()

    def test_updates_last_request_after_sleep(self):
        client = SecClient(rate_limit=0.1, user_agent="Test Agent test@example.com")
        client._last_request = time.time()
        before = time.time()

        with patch("time.sleep"):
            client.rate_limit()

        assert client._last_request >= before
