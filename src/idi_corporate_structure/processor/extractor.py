"""Extractors for parsing subsidiary data from SEC exhibit documents."""

# Standard imports
import json
from abc import ABC, abstractmethod

# Application imports
from idi_corporate_structure.common.api import OpenAiClient
from idi_corporate_structure.processor.types import Filing, Subsidiary


class DocumentError(Exception):
    """Exception raised for document-specific errors."""

    pass


class Extractor(ABC):
    """Interface for extracting subsidiaries from a single exhibit document."""

    @abstractmethod
    def extract(self, filing: Filing, document: dict) -> list[Subsidiary]:
        """Extract subsidiaries from a single exhibit document.

        Args:
            filing: The filing the document belongs to.
            document: Dict with 'url' and 'data' keys for the exhibit content.

        Returns:
            List of extracted Subsidiary objects.
        """
        ...


class GptExtractor(Extractor):
    """Extracts subsidiaries from a single exhibit document using GPT."""

    _DOCUMENT_ERROR_STATUS_CODE = 400
    _SYSTEM_PROMPT = """
    Given a table of a company's subsidiaries (in Markdown or raw text, previously converted from PDF), format them as a JSON, like

    ```json
    {
    "subsidiaries": [
        {"name": "XXX", "in": YYY}
    ]
    }
    ```

    objects, where `"XXX"` is the name of the subsidiary and `YYY` is the place of incorporation or other location, or `null` if not provided.

    Include all of the subsidiaries, but ignore any nested structure and ignore any data unrelated to subsidiaries.
    """.strip()

    def __init__(self, openai_api_key: str) -> None:
        """Initialize the GPT extractor with the OpenAI API key."""
        self._openai_client = OpenAiClient(api_key=openai_api_key)

    def _get_request_data_json(self, document: str) -> dict:
        """Build the OpenAI chat completions request payload for subsidiary extraction.

        Constructs a structured-output request that instructs the model to parse
        a subsidiary table from ``document`` and return a JSON object conforming
        to the ``list_of_subsidiaries`` schema.

        Args:
            document: Raw exhibit text (Markdown or plain text) to send as the user
                message.

        Returns:
            Dict ready to be serialized and posted to the OpenAI API, containing
            ``model``, ``messages``, and ``response_format`` keys.
        """
        return {
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": self._SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": document,
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "list_of_subsidiaries",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "subsidiaries": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "in": {"type": ["string", "null"]},
                                    },
                                    "required": ["name", "in"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
        }

    def _summarize(self, document: str) -> dict:
        """Send the document to GPT and return the parsed subsidiary data.

        Args:
            document: Raw exhibit text to pass to the model.

        Returns:
            Parsed JSON response from the model as a dict with a ``"subsidiaries"``
            key containing a list of ``{"name": ..., "in": ...}`` objects.

        Raises:
            DocumentError: If the API returns a 400-level status code, indicating
                the document itself is malformed or too long.
            RuntimeError: If the API returns any other error.
        """
        post_data = self._get_request_data_json(document)
        response = self._openai_client.query_endpoint(post_data)

        if "error" in response:
            if response.get("status_code") == self._DOCUMENT_ERROR_STATUS_CODE:
                raise DocumentError(response["error"])
            raise RuntimeError(response["error"])

        content = response["data"]["choices"][0]["message"]["content"]
        return json.loads(content)

    def extract(self, filing: Filing, document: dict) -> list[Subsidiary]:
        """Extract subsidiaries from an exhibit document using GPT.

        Sends the document text to the OpenAI API and maps each item in the
        returned ``"subsidiaries"`` list to a :class:`Subsidiary` dataclass,
        inheriting parent metadata from ``filing``.

        Args:
            filing: The SEC filing the exhibit belongs to. Provides parent company
                metadata (CIK, name, location, dates).
            document: Dict with ``"url"`` (exhibit URL) and ``"data"`` (exhibit text)
                keys.

        Returns:
            List of :class:`Subsidiary` objects extracted from the document.

        Raises:
            DocumentError: If GPT rejects the document (e.g. content too long or
                malformed).
            RuntimeError: If the OpenAI API returns any other error.
        """
        summary = self._summarize(document["data"])
        return [
            Subsidiary(
                parent_cik=filing.cik,
                parent_name=filing.company_name,
                parent_location=filing.location,
                filing_date=filing.filing_date,
                form_type=filing.form_type,
                accession_number=filing.accession_number,
                exhibit_url=document["url"],
                name=sub["name"],
                location=sub.get("in") or "",
            )
            for sub in summary["subsidiaries"]
        ]
