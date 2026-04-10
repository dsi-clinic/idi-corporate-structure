"""Provides API utilities for use across the application."""

# Standard library imports
import contextlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Literal

# Third party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Application imports
from .logs import get_logger


class ApiClient(ABC):
    """Base class for API clients."""

    DEFAULT_MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: tuple[int, int] = (10, 30)
    RETRY_BACKOFF_FACTOR: int = 2  # Wait 1, 2, 4 seconds between retries
    RETRY_STATUS_FORCELIST: list[int] = [429, 500, 502, 503, 504]
    USER_AGENT: str = "idi-company-info"

    def __init__(
        self,
        api_key: str = "",
        max_retries: int = DEFAULT_MAX_RETRIES,
        rate_limit: float | None = None,
    ) -> None:
        """Initialize the ApiClient.

        Args:
            api_key: The API key.
            max_retries: The maximum number of retries.
            rate_limit: Minimum seconds between requests. None disables rate limiting.
        """
        self.api_key: str = api_key
        self.max_retries: int = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        self.logger: logging.Logger = get_logger("ApiClient")
        self._rate_limit = rate_limit
        self._last_request = time.time()
        self._lock: threading.Lock | contextlib.AbstractContextManager = (
            threading.Lock() if rate_limit is not None else contextlib.nullcontext()
        )

    def rate_limit(self) -> None:
        """Enforce rate limit between requests.

        No-op when rate_limit was not set at construction time.
        Thread-safe: serializes callers when rate_limit is configured.
        """
        if self._rate_limit is None:
            return
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._rate_limit:
                time.sleep(self._rate_limit - elapsed)
            self._last_request = time.time()

    @cached_property
    def session(self) -> requests.Session:
        """Create a requests Session with retry strategy.

        Returns:
            Configured requests.Session with retry logic
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.RETRY_BACKOFF_FACTOR,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=self.RETRY_STATUS_FORCELIST,
            allowed_methods=["GET", "POST"],
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def get(
        self, url: str, params: dict | None = None, headers: dict | None = None, **kwargs: object
    ) -> requests.Response:
        """Get a resource from the API.

        Args:
            url: The URL to get from.
            params: The parameters to pass to the API.
            headers: The headers to pass to the API.
            kwargs: Additional keyword arguments to pass to the API.

        Returns:
            The response from the API.
        """
        kwargs.setdefault("timeout", self.REQUEST_TIMEOUT)
        response = self.session.get(url, params=params, headers=headers, **kwargs)
        response.raise_for_status()
        return response

    def post(
        self,
        url: str,
        data: str | dict | None = None,
        headers: dict | None = None,
        **kwargs: object,
    ) -> requests.Response:
        """Post a resource to the API.

        Args:
            url: The URL to post to.
            data: The data to post to the API.
            headers: The headers to post to the API.
            kwargs: Additional keyword arguments to pass to the API.

        Returns:
            The response from the API.
        """
        kwargs.setdefault("timeout", self.REQUEST_TIMEOUT)
        response = self.session.post(url, headers=headers, data=data, **kwargs)
        response.raise_for_status()
        return response

    def _query_with_error_handling(
        self,
        url: str,
        data: str | dict | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        method: Literal["get", "post"] = "get",
        return_json: bool = True,
        return_bytes: bool = False,
    ) -> dict[str, Any]:
        """Query an endpoint with error handling, capturing errors in the return value.

        On success the returned dict contains ``status_code``, ``url``, and ``data`` keys.
        On failure an ``error`` key is added; HTTP errors also include ``status_code``.
        Exceptions are never re-raised — callers should check for the ``error`` key.

        Args:
            url: The URL to query.
            data: The data to post to the API. Only used when ``method`` is ``"post"``.
            params: Query-string parameters to pass to the API.
            headers: HTTP headers to pass to the API.
            method: HTTP verb to use — ``"get"`` or ``"post"``.
            return_json: If True, parse the response body as JSON; otherwise return raw text.
            return_bytes: If True, return raw response bytes (overrides ``return_json``).

        Returns:
            Dict with ``status_code``, ``url``, and ``data`` keys on success.
            On error, ``error`` is added and ``data`` may be absent.
        """
        response, error, error_exc = None, None, None
        try:
            response = (
                self.get(url=url, params=params, headers=headers)
                if method == "get"
                else self.post(url=url, data=data, headers=headers)
            )

        except requests.exceptions.RequestException as e:
            error = f"Error querying {url}: {e}"
            error_exc = e
            self.logger.error(error)

        response_data = {}
        if isinstance(error_exc, requests.exceptions.HTTPError) and error_exc.response is not None:
            response_data["status_code"] = error_exc.response.status_code

        if response is not None:
            try:
                if return_bytes:
                    r_data = response.content
                elif return_json:
                    r_data = response.json()
                else:
                    r_data = response.text

                response_data.update(
                    {
                        "status_code": response.status_code,
                        "url": response.url,
                        "data": r_data,
                    }
                )
            except ValueError:
                self.logger.error(f"Error parsing JSON response from {url}: {response.text}")

        if error is not None:
            response_data.update({"error": error})

        return response_data

    @abstractmethod
    def query_endpoint(self, **kwargs: object) -> dict[str, Any]:
        """Query the API endpoint specific to this client.

        Subclasses define the exact positional/keyword parameters relevant to their
        endpoint. The return dict follows the ``_query_with_error_handling`` contract:
        ``status_code``, ``url``, and ``data`` on success; ``error`` on failure.

        Args:
            **kwargs: Endpoint-specific arguments defined by each subclass.

        Returns:
            Dict with ``status_code``, ``url``, and ``data`` on success, plus ``error``
            on failure.
        """
        ...


class LsegEntitySearch(ApiClient):
    """API client for the LSEG Entity Search API."""

    ENTITY_SEARCH_URL = "https://api-eit.refinitiv.com/permid/search"

    def query_endpoint(self, params: dict) -> dict[str, Any]:
        """Query the LSEG Entity Search API.

        Args:
            params: Query parameters to pass to the entity search endpoint.

        Returns:
            Dict with ``status_code``, ``url``, and ``data`` on success, plus ``error``
            on failure.
        """
        headers = {
            "X-AG-Access-Token": self.api_key,
            "Accept": "application/json",
            "User-Agent": self.USER_AGENT,
        }
        return self._query_with_error_handling(
            url=self.ENTITY_SEARCH_URL, params=params, headers=headers, method="get"
        )


class LsegRecordMatch(ApiClient):
    """API client for the LSEG Record Match API."""

    RECORD_MATCH_URL = "https://api-eit.refinitiv.com/permid/match"

    def query_endpoint(self, csv_data: str) -> dict[str, Any]:
        """Query the LSEG Record Match API.

        Args:
            csv_data: CSV-formatted string of records to match against LSEG entity data.

        Returns:
            Dict with ``status_code``, ``url``, and ``data`` on success, plus ``error``
            on failure.
        """
        headers = {
            "accept": "application/json",
            "Content-Type": "text/plain",
            "x-ag-access-token": self.api_key,
            "x-openmatch-numberOfMatchesPerRecord": "1",
            "x-openmatch-dataType": "Organization",
            "User-Agent": self.USER_AGENT,
        }
        return self._query_with_error_handling(
            url=self.RECORD_MATCH_URL, data=csv_data, headers=headers, method="post"
        )


class LSEGEntityLookup(ApiClient):
    """API client for the LSEG Entity Lookup API."""

    def query_endpoint(self, permid_url: str) -> dict[str, Any]:
        """Query the LSEG Entity Lookup API.

        Args:
            permid_url: Full PermID URL to look up (e.g. ``https://permid.org/1-...``).

        Returns:
            Dict with ``status_code``, ``url``, and ``data`` on success, plus ``error``
            on failure.
        """
        headers = {
            "X-AG-Access-Token": self.api_key,
            "Accept": "application/ld+json",
        }
        params = {"format": "json-ld"}
        return self._query_with_error_handling(
            url=permid_url, params=params, headers=headers, method="get"
        )


class GeonamesApi(ApiClient):
    """API client for the Geonames API."""

    GEONAMES_API_URL = "http://api.geonames.org/getJSON"

    def __init__(self, api_key: str, geonames_user: str) -> None:
        """Initialize the GeonamesApi.

        Args:
            api_key: The API key.
            geonames_user: The Geonames user.
        """
        super().__init__(api_key=api_key)
        self.geonames_user = geonames_user

    def query_endpoint(self, geoname_url: str) -> dict[str, Any]:
        """Query the Geonames API.

        Extracts the numeric geoname ID from ``geoname_url`` and fetches the
        corresponding record from the Geonames REST service.

        Args:
            geoname_url: Geonames linked-data URL containing the numeric ID
                (e.g. ``http://sws.geonames.org/6252001/``).

        Returns:
            Dict with ``status_code``, ``url``, and ``data`` on success, plus ``error``
            on failure.
        """
        # Extract geoname ID from URL (e.g., http://sws.geonames.org/6252001/)
        geoname_id = geoname_url.rstrip("/").split("/")[-1]

        # Query Geonames API with credentials (per https://www.geonames.org/export/web-services.html)
        params = {"geonameId": geoname_id, "username": self.geonames_user}
        return self._query_with_error_handling(
            url=self.GEONAMES_API_URL, params=params, method="get"
        )


class SecClient(ApiClient):
    """API client for the SEC EDGAR archive, with built-in rate limiting."""

    SEC_HEADERS = {"User-Agent": "Nicole Tebaldi ntebaldi@uchicago.edu"}
    SEC_URL = "https://www.sec.gov/Archives/edgar/data"

    def __init__(self, rate_limit: float = 0.2) -> None:
        """Initializes the SEC API.

        Args:
            rate_limit: How long to wait in between requests
        """
        super().__init__(rate_limit=rate_limit)

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
            headers=self.SEC_HEADERS,
            method="get",
            return_json=return_json,
            return_bytes=return_bytes,
        )


class OpenAiClient(ApiClient):
    """API client for the OpenAI API."""

    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

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
