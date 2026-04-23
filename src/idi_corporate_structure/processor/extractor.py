"""Extractors for parsing subsidiary data from SEC exhibit documents."""

# Standard imports
import html as _html
import importlib.resources
import json
from abc import ABC, abstractmethod

# Application imports
from idi_corporate_structure.common.api import OpenAiClient
from idi_corporate_structure.common.logs import get_logger
from idi_corporate_structure.processor.types import Filing, Subsidiary

_PROMPTS = importlib.resources.files("idi_corporate_structure.processor.prompts")


class DocumentError(Exception):
    """Exception raised for document-specific errors."""

    pass


class Extractor(ABC):
    """Interface for extracting subsidiaries from a single exhibit document."""

    @abstractmethod
    def extract(self, filing: Filing, document: dict) -> tuple[list[Subsidiary], int]:
        """Extract subsidiaries from a single exhibit document.

        Args:
            filing: The filing the document belongs to.
            document: Dict with 'url' and 'data' keys for the exhibit content.

        Returns:
            Tuple of (extracted Subsidiary objects, count of subsidiaries dropped
            during grounding checks).
        """
        ...


class GptExtractor(Extractor):
    """Extracts subsidiaries from a single exhibit document using GPT."""

    _DOCUMENT_ERROR_STATUS_CODE = 400
    _SYSTEM_PROMPT: str = (
        _PROMPTS.joinpath("gpt_extractor_system.txt").read_text(encoding="utf-8").strip()
    )

    def __init__(self, openai_api_key: str) -> None:
        """Initialize the GPT extractor with the OpenAI API key."""
        self._openai_client = OpenAiClient(api_key=openai_api_key)
        self._logger = get_logger(__name__)

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
                                        "location": {"type": ["string", "null"]},
                                        "source_quote": {"type": "string"},
                                    },
                                    "required": ["name", "location", "source_quote"],
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
            key containing a list of ``{"name": ..., "location": ..., "source_quote": ...}`` objects.

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

    @staticmethod
    def _is_grounded(quote: str, document: str) -> bool:
        """Check if a source quote is grounded in the document.

        Args:
            quote: The source quote to check.
            document: The document to check the quote in.

        Returns:
            True if the quote is grounded in the document, False otherwise.
        """
        if not quote:
            return False

        return " ".join(quote.split()) in " ".join(document.split())

    @staticmethod
    def _is_name_grounded(name: str, quote: str) -> bool:
        """Check if a name is grounded in a source quote.

        Args:
            name: The name to check.
            quote: The source quote to check.

        Returns:
            True if the name is grounded in the quote, False otherwise.
        """
        if not name or not quote:
            return False

        return _normalize(name) in _normalize(quote)

    def _locate_grounded_subsidiaries(
        self, subsidiaries: list[dict], document: dict
    ) -> tuple[list[dict], int]:
        """Locate grounded subsidiaries in a document.

        Args:
            subsidiaries: The subsidiaries to check.
            document: The document to check the subsidiaries in.

        Returns:
            List of grounded subsidiaries.
        """
        grounded_subsidiaries = []
        dropped_count = 0
        for sub in subsidiaries:
            name = sub.get("name", "")
            quote = sub.get("source_quote", "")
            if not self._is_grounded(quote, document.get("data", "")):
                self._logger.warning(
                    "Dropped %r from %s (ungrounded quote)", name, document.get("url", "")
                )
                dropped_count += 1
                continue
            if not self._is_name_grounded(name, quote):
                self._logger.warning(
                    "Dropped %r from %s (ungrounded name)", name, document.get("url", "")
                )
                dropped_count += 1
                continue
            grounded_subsidiaries.append(sub)

        if dropped_count:
            self._logger.warning(
                "Dropped %d ungrounded subsidiaries from %s", dropped_count, document.get("url", "")
            )

        return grounded_subsidiaries, dropped_count

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

        grounded_subsidiaries, dropped = self._locate_grounded_subsidiaries(
            summary["subsidiaries"], document
        )

        subsidiaries = [
            Subsidiary(
                parent_cik=filing.cik,
                parent_name=filing.company_name,
                parent_location=filing.location,
                filing_date=filing.filing_date,
                form_type=filing.form_type,
                exhibit_type=filing.exhibit_type,
                accession_number=filing.accession_number,
                exhibit_url=document["url"],
                name=_html.unescape(sub["name"]),
                location=sub.get("location") or "",
                source_quote=sub.get("source_quote") or "",
            )
            for sub in grounded_subsidiaries
        ]
        return subsidiaries, dropped


def _normalize(s: str) -> str:
    """Decode HTML entities, normalize apostrophes and whitespace, and lowercase."""
    s = _html.unescape(s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(s.split()).lower()
