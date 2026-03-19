"""Provides API utilities for use across the application."""

# Standard library imports
import logging
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

    def __init__(self, api_key: str = "", max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        """Initialize the ApiClient.

        Args:
            api_key: The API key.
            max_retries: The maximum number of retries.
        """
        self.api_key: str = api_key
        self.max_retries: int = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        self.logger: logging.Logger = get_logger("ApiClient")

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
        data: str | dict = None,
        params: dict = None,
        headers: dict = None,
        method: Literal["get", "post"] = "get",
    ) -> dict[str, Any]:
        """Query an endpoint with error handling.

        Args:
            url: The URL to query.
            data: The data to post to the API.
            params: The parameters to pass to the API.
            headers: The headers to pass to the API.
            method: The method to use to query the API.

        Returns:
            The data from the API.
        """
        response, error = None, None
        try:
            response = (
                self.get(url=url, params=params, headers=headers)
                if method == "get"
                else self.post(url=url, data=data, headers=headers)
            )

        except requests.exceptions.RequestException as e:
            error = f"Error querying {url}: {e}"
            self.logger.error(error)

        response_data = {}
        if response is not None:
            try:
                response_data.update(
                    {
                        "status_code": response.status_code,
                        "url": response.url,
                        "data": response.json(),
                    }
                )
            except ValueError:
                self.logger.error(f"Error parsing JSON response from {url}: {response.text}")

        if error is not None:
            response_data.update({"error": error})

        return response_data

    @abstractmethod
    def query_endpoint(self, **kwargs) -> dict[str, Any]:
        """Query an endpoint."""
        ...


class LsegEntitySearch(ApiClient):
    """API client for the LSEG Entity Search API."""

    ENTITY_SEARCH_URL = "https://api-eit.refinitiv.com/permid/search"

    def query_endpoint(self, params: dict) -> dict:
        """Query the LSEG Entity Search API.

        Args:
            params: The parameters to pass to the API.

        Returns:
            The data from the API.
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

    def query_endpoint(self, csv_data: str) -> dict:
        """Query the LSEG Record Match API.

        Args:
            csv_data: The CSV data to search for.

        Returns:
            The data from the API.
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

    def query_endpoint(self, permid_url: str) -> dict:
        """Query the LSEG Entity Lookup API.

        Args:
            permid_url: The PermID URL to lookup.

        Returns:
            The data from the API.
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

    def query_endpoint(self, geoname_url: str) -> dict:
        """Query the Geonames API.

        Args:
            geoname_url: The Geonames URL to look up (e.g. http://sws.geonames.org/6252001/).
        """
        # Extract geoname ID from URL (e.g., http://sws.geonames.org/6252001/)
        geoname_id = geoname_url.rstrip("/").split("/")[-1]

        # Query Geonames API with credentials (per https://www.geonames.org/export/web-services.html)
        params = {"geonameId": geoname_id, "username": self.geonames_user}
        return self._query_with_error_handling(
            url=self.GEONAMES_API_URL, params=params, method="get"
        )


class SecClient(ApiClient):

    def __init__(self):
        """Initializes the SEC API."""
        super().__init__()

    def query_endpoint(self, sec_url, headers={}):
        return self._query_with_error_handling(
            url=sec_url, headers=headers, method="get"
        )
