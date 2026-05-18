"""Provides API utilities for use across the application."""

# Standard library imports
import json
from typing import Any

# Third-party imports
from idi_ftm2j_shared.api import ApiClient


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

        self.rate_limit()
        response = self._query_with_error_handling(
            url=self.OPENAI_API_URL,
            headers=headers,
            method="post",
            data=data,
            return_json=return_json,
        )
        return response
