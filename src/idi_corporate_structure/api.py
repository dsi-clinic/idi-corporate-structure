"""Provides API utilities for use across the application."""

# Standard library imports
import json
from typing import Any

# Third-party imports
from idi_ftm2j_shared.api import ApiClient


class SecClient(ApiClient):
    """API client for the SEC EDGAR archive, with built-in rate limiting."""

    SEC_URL = "https://www.sec.gov/Archives/edgar/data"

    def __init__(self, rate_limit: float = 0.2, user_agent: str = "") -> None:
        """Initializes the SEC API.

        Args:
            rate_limit: How long to wait in between requests.
            user_agent: Value for the SEC-required ``User-Agent`` header.
        """
        super().__init__(rate_limit=rate_limit)
        self._sec_headers = {"User-Agent": user_agent}

    @property
    def sec_headers(self) -> dict:
        """Return the SEC header for querying."""
        return self._sec_headers

    def query_endpoint(
        self, sec_url: str, return_json: bool = True, return_bytes: bool = False
    ) -> dict[str, Any]:
        """Query a SEC EDGAR endpoint with the required User-Agent header.

        Args:
            sec_url: Full SEC EDGAR URL to query.
            return_json: If True, parse response as JSON; otherwise return raw text.
            return_bytes: If True, return raw response bytes (overrides ``return_json``).

        Returns:
            Dict with ``status_code``, ``url``, and ``data`` on success, plus ``error``
            on failure.
        """
        return self._query_with_error_handling(
            url=sec_url,
            headers=self._sec_headers,
            method="get",
            return_json=return_json,
            return_bytes=return_bytes,
        )


class OpenAiClient(ApiClient):
    """API client for the OpenAI API."""

    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
    REQUEST_TIMEOUT: tuple[int, int] = (10, 90)

    def __init__(self, api_key: str, rate_limit: float = 0.5) -> None:
        """Initializes the OpenAI API.

        Args:
            api_key: The API key.
            rate_limit: Minimum seconds between requests.
        """
        super().__init__(api_key=api_key, rate_limit=rate_limit)

    def query_endpoint(
        self,
        data: str | dict | None = None,
        return_json: bool = True,
    ) -> dict[str, Any]:
        """Query the OpenAI chat completions endpoint.

        If ``data`` is a dict it is serialized to JSON before being sent.

        Args:
            data: Request payload as a JSON string or a dict. Pass ``None`` to send
                an empty body.
            return_json: If True, parse response as JSON; otherwise return raw text.

        Returns:
            Dict with ``status_code``, ``url``, and ``data`` on success, plus ``error``
            on failure.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if isinstance(data, dict):
            data = json.dumps(data)

        response = self._query_with_error_handling(
            url=self.OPENAI_API_URL,
            headers=headers,
            method="post",
            data=data,
            return_json=return_json,
        )
        return response
