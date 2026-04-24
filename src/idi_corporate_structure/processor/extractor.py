"""Extractors for parsing subsidiary data from SEC exhibit documents."""

# Standard imports
import html as _html
import importlib.resources
import json
import re
from abc import ABC, abstractmethod

# Third-party imports
from bs4 import BeautifulSoup, Comment

# Application imports
from idi_corporate_structure.common.api import OpenAiClient
from idi_corporate_structure.common.logs import get_logger
from idi_corporate_structure.processor.types import Filing, Subsidiary

_PROMPTS = importlib.resources.files("idi_corporate_structure.processor.prompts")

_BLOCK_TAGS = frozenset(
    {
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ol",
        "p",
        "table",
        "tr",
        "ul",
    }
)
_CELL_TAGS = frozenset({"td", "th"})
_INLINE_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


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
    def _is_name_in_document(name: str, document: str) -> bool:
        """Check if a name appears in the document (the source of truth).

        Args:
            name: The name to check.
            document: The document text to check for the name.

        Returns:
            True if the normalized name is a substring of the normalized
            document, False otherwise.
        """
        if not name:
            return False

        return _normalize(name) in _normalize(document)

    def _is_location_grounded(
        self, name: str, location: str, doc_text_normalized: str, doc_url: str
    ) -> int:
        """Check if a location is near a name in the document and logs a warning if not.

        Args:
            name: The name to check.
            location: The location to check.
            doc_text_normalized: The normalized document text to check for the name and location.
            doc_url: The URL of the document.

        Returns:
            The number of ungrounded locations.
        """
        ungrounded_location = 0
        if location:
            name_pos = doc_text_normalized.find(_normalize(name))
            name_len = len(_normalize(name))
            window = doc_text_normalized[max(0, name_pos - 200) : name_pos + name_len + 200]

            if _normalize(location) not in window:
                self._logger.debug("Location %r not near name %r @ %s", location, name, doc_url)
                ungrounded_location += 1
        return ungrounded_location

    def _locate_grounded_subsidiaries(
        self, subsidiaries: list[dict], document: dict
    ) -> tuple[list[dict], int, int]:
        """Locate subsidiaries whose names are grounded in the document.

        The document is the source of truth. A subsidiary is kept if its
        ``name`` appears in the document. The model's ``source_quote`` is
        advisory: a mismatch between quote and document is logged at DEBUG
        level but does not drop the row.

        Args:
            subsidiaries: The subsidiaries returned by the model.
            document: Dict with ``"url"`` and ``"data"`` (text) keys.

        Returns:
            Tuple of (kept subsidiaries, count dropped for missing name, count dropped for missing location).
        """
        grounded_subsidiaries = []
        ungrounded_name = 0
        ungrounded_location = 0

        doc_text = document.get("data", "")
        doc_url = document.get("url", "")
        doc_text_normalized = _normalize(doc_text)

        for sub in subsidiaries:
            name = sub.get("name", "")
            if not self._is_name_in_document(name, doc_text):
                self._logger.warning("Dropped %r from %s (name not in document)", name, doc_url)
                ungrounded_name += 1
                continue

            quote = sub.get("source_quote", "")
            if quote and _normalize(quote) not in doc_text_normalized:
                self._logger.debug("Quote not in document for %r @ %s", name, doc_url)

            ungrounded_location += self._is_location_grounded(
                name, sub.get("location") or "", doc_text_normalized, doc_url
            )
            grounded_subsidiaries.append(sub)

        if ungrounded_name:
            self._logger.warning(
                "Dropped %d ungrounded subsidiaries from %s", ungrounded_name, doc_url
            )

        return grounded_subsidiaries, ungrounded_name, ungrounded_location

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

        grounded_subsidiaries, ungrounded_name, ungrounded_location = (
            self._locate_grounded_subsidiaries(summary["subsidiaries"], document)
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
        return subsidiaries, ungrounded_name, ungrounded_location


def _html_to_text(raw_html: str) -> str:
    """Convert HTML to plain text, preserving table row/cell boundaries.

    Args:
        raw_html: Raw HTML string.

    Returns:
        Plain text with block tags rendered as newlines and table cells
        separated by spaces. HTML entities are decoded. If the input is
        already plain text (no tags), it passes through with whitespace
        normalized.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    for tag in soup.find_all(True):
        if tag.name in _CELL_TAGS:
            tag.insert_before(" ")
            tag.insert_after(" ")
        elif tag.name in _BLOCK_TAGS:
            tag.insert_before("\n")
            tag.insert_after("\n")

    text = _html.unescape(soup.get_text())

    lines = [_INLINE_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    collapsed = _MULTINEWLINE_RE.sub("\n\n", "\n".join(lines))
    return collapsed.strip()


def _normalize(s: str) -> str:
    """Decode HTML entities, normalize apostrophes and whitespace, and lowercase."""
    s = _html.unescape(s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(s.split()).lower()
