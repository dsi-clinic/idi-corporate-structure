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

# Chunking thresholds — empirically tuned to keep gpt-4.1-nano output under
# ~2K tokens per call, well below its self-imposed laziness ceiling of ~5K.
_CHUNK_THRESHOLD_CHARS = 4_000
_CHUNK_MAX_CHARS = 4_000
_CHUNK_OVERLAP_CHARS = 400

_INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")


class DocumentError(Exception):
    """Exception raised for document-specific errors."""

    pass


class ExtractionTimeoutError(RuntimeError):
    """Exception raised when the OpenAI API times out during extraction."""

    pass


class ExtractionTruncatedError(RuntimeError):
    """Raised when the model's extraction response was cut off by the output token limit."""

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
    # Maximum output tokens. Set explicitly so that truncation surfaces as
    # finish_reason="length" rather than depending on API defaults.
    _MAX_COMPLETION_TOKENS = 32768
    _DEFAULT_MODEL = "gpt-4.1-nano"
    _SYSTEM_PROMPT: str = (
        _PROMPTS.joinpath("gpt_extractor_system.txt").read_text(encoding="utf-8").strip()
    )

    def __init__(self, openai_api_key: str, model: str = "") -> None:
        """Initialize the GPT extractor with the OpenAI API key.

        Args:
            openai_api_key: OpenAI API key.
            model: OpenAI model ID to use for extraction.  Defaults to
                ``_DEFAULT_MODEL`` when omitted or empty.
        """
        self._openai_client = OpenAiClient(api_key=openai_api_key)
        self._model = model or self._DEFAULT_MODEL
        self._logger = get_logger(__name__)

    def _extract_with_chunking(self, doc_text: str, company_name: str) -> tuple[list[dict], int]:
        """Run extraction one-shot, falling back to chunked extraction if needed.

        Two triggers cause chunking:
          * Preemptive: ``len(doc_text) > _CHUNK_THRESHOLD_CHARS`` (catches the
            ``finish_reason="stop"`` laziness case where the model gives up
            mid-extraction without surfacing as truncation).
          * Reactive: a one-shot call hits the explicit output cap and raises
            :class:`ExtractionTruncatedError`.

        Args:
            doc_text: Full plain-text exhibit content.
            company_name: String name of the filing company

        Returns:
            Tuple of (raw subsidiary dicts from the model, num chunks used).
            ``num_chunks == 1`` means no chunking was performed.
        """
        if len(doc_text) > _CHUNK_THRESHOLD_CHARS:
            return self._summarize_chunks(doc_text, company_name)

        try:
            return self._summarize(doc_text).get("subsidiaries", []), 1
        except ExtractionTruncatedError:
            self._logger.info("One-shot extraction truncated; retrying with chunking")
            return self._summarize_chunks(doc_text, company_name)

    def _summarize_chunks(self, doc_text: str, company_name: str) -> tuple[list[dict], int]:
        """Chunk ``doc_text`` and run a separate summarize call per chunk.

        Args:
            doc_text: Full plain-text exhibit content
            company_name: String name of the filing company

        Returns:
            Tuple of (concatenated raw subsidiaries from all chunks, chunk count)

        Raises:
            ExtractionTruncatedError: If any individual chunk truncates — the
                chunk-size constants need tuning down.
        """
        chunks = _chunk_document(doc_text, _CHUNK_MAX_CHARS, _CHUNK_OVERLAP_CHARS)
        self._logger.info("%s chunked extraction: %d chunks", company_name, len(chunks))
        all_subs: list[dict] = []
        for i, chunk in enumerate(chunks, 1):
            try:
                result = self._summarize(chunk)
            except ExtractionTruncatedError:
                self._logger.error(
                    "Chunk %d/%d truncated — chunk size may be too large", i, len(chunks)
                )
                raise
            all_subs.extend(result.get("subsidiaries", []))
        return all_subs, len(chunks)

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
            "model": self._model,
            "max_completion_tokens": self._MAX_COMPLETION_TOKENS,
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
            ExtractionTimeoutError: If the API call timed out.
            ExtractionTruncatedError: If the model response was cut off by the
                output token limit (``finish_reason == "length"``).
            RuntimeError: If the API returns any other error.
        """
        post_data = self._get_request_data_json(document)
        response = self._openai_client.query_endpoint(post_data)

        if "error" in response:
            if response.get("status_code") == self._DOCUMENT_ERROR_STATUS_CODE:
                raise DocumentError(response["error"])
            if response.get("timeout"):
                raise ExtractionTimeoutError(response["error"])
            raise RuntimeError(response["error"])

        choice = response["data"]["choices"][0]
        finish_reason = choice.get("finish_reason")
        usage = response["data"].get("usage", {})
        self._logger.debug(
            "OpenAI extraction input_chars=%d | finish_reason=%s | usage=%s",
            len(document),
            finish_reason,
            usage,
        )

        if finish_reason == "length":
            raise ExtractionTruncatedError(
                f"Model response truncated at output token limit "
                f"(max_completion_tokens={self._MAX_COMPLETION_TOKENS}, usage={usage})"
            )

        content = choice["message"]["content"]
        return json.loads(content)

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
            if not _is_name_in_document(name, doc_text):
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

    def extract(self, filing: Filing, document: dict) -> tuple[list[Subsidiary], int, int, int]:
        """Extract subsidiaries from an exhibit document using GPT.

        Sends the document text to the OpenAI API (one-shot or chunked) and maps
        each returned item to a :class:`Subsidiary` dataclass, inheriting parent
        metadata from ``filing``.

        Args:
            filing: The SEC filing the exhibit belongs to. Provides parent company
                metadata (CIK, name, location, dates).
            document: Dict with ``"url"`` (exhibit URL) and ``"data"`` (exhibit text)
                keys.

        Returns:
            Tuple of (extracted Subsidiary objects, ungrounded-name count,
            ungrounded-location count, num chunks used). ``num_chunks > 1``
            indicates chunked extraction was used.

        Raises:
            DocumentError: If GPT rejects the document (e.g. content too long or
                malformed).
            ExtractionTruncatedError: If a chunked extraction still truncates
                (chunk size needs tuning).
            RuntimeError: If the OpenAI API returns any other error.
        """
        # Summarize the subsidiaries in the exhibit
        raw_subs, num_chunks = self._extract_with_chunking(document["data"], filing.company_name)

        # Dedupe by normalized name
        deduped = dedup_by_name(raw_subs=raw_subs)

        # Double check the results of the summarize
        grounded_subsidiaries, ungrounded_name, ungrounded_location = (
            self._locate_grounded_subsidiaries(deduped, document)
        )

        # Create subsidiaries
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
                name=_clean_name(sub["name"]),
                location=sub.get("location") or "",
                source_quote=sub.get("source_quote") or "",
            )
            for sub in grounded_subsidiaries
        ]
        return subsidiaries, ungrounded_name, ungrounded_location, num_chunks


def html_to_text(raw_html: str) -> str:
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


def _take_overlap(chunk_text: str, overlap_chars: int) -> str:
    r"""Return the trailing whole paragraphs of ``chunk_text`` fitting in ``overlap_chars``.

    Used to build the overlap region between adjacent chunks. By taking only
    whole paragraphs, we guarantee the overlap text never bisects an entity row
    — so the next chunk never begins with a partial name like ``"wer Eight
    Project LLC"`` that the model would mis-extract as ``"Eight Project LLC"``.

    Always returns at least the final paragraph, even if it exceeds
    ``overlap_chars``, so the boundary entity is always carried forward.

    Args:
        chunk_text: The just-emitted chunk's full text.
        overlap_chars: Soft budget for the overlap region.

    Returns:
        Joined paragraphs (with ``\n\n`` separators) or empty string when
        ``overlap_chars <= 0``.
    """
    if overlap_chars <= 0:
        return ""

    paragraphs = chunk_text.split("\n\n")
    overlap_paras: list[str] = []
    overlap_len = 0

    for para in reversed(paragraphs):
        # +2 accounts for the "\n\n" separator we'd add when joining
        if overlap_paras and overlap_len + len(para) + 2 > overlap_chars:
            break
        overlap_paras.insert(0, para)
        overlap_len += len(para) + 2

    return "\n\n".join(overlap_paras)


def _chunk_document(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    r"""Split text into overlapping chunks at paragraph boundaries.

    Walks paragraphs (split on ``\\n\\n``), greedily packing them into chunks of
    up to ``max_chars``. Each chunk after the first carries ``overlap_chars`` of
    trailing text from the previous chunk to catch entries near the boundary;
    the merge step deduplicates by name.

    Oversized paragraphs (greater than ``max_chars``) are emitted as their own
    chunk and a warning is logged — better one possibly-truncated chunk than a
    corrupt entry split mid-row.

    Args:
        text: Plain-text exhibit content to chunk.
        max_chars: Soft maximum size per chunk.
        overlap_chars: Trailing characters carried into the next chunk.

    Returns:
        Ordered list of chunk strings. A short ``text`` returns ``[text]``.
    """
    if len(text) <= max_chars:
        return [text]

    logger = get_logger(__name__)
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # account for "\n\n" separator

        if para_len > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            logger.warning(
                "Paragraph exceeds chunk size (%d > %d chars); emitting as single oversized chunk",
                len(para),
                max_chars,
            )
            chunks.append(para)
            continue

        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            tail = _take_overlap(chunks[-1], overlap_chars)
            current = [tail] if tail else []
            current_len = len(tail)

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


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


def _normalize(s: str) -> str:
    """Decode HTML entities, normalize apostrophes and whitespace, and lowercase."""
    s = _html.unescape(s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(s.split()).lower()


def dedup_by_name(raw_subs: list[dict]) -> list[dict]:
    """Dedupe by normalized name (collisions come from the chunk overlap region)

    Args:
        raw_subs: List of dictionaries that contain extracted subsidiaries

    Returns:
        De-duplicated subsidiaries for exhibit
    """
    seen: set[str] = set()
    deduped: list[dict] = []
    for sub in raw_subs:
        key = _normalize(sub.get("name", ""))
        # First occurrence wins for location and source_quote
        if key and key not in seen:
            seen.add(key)
            deduped.append(sub)
    return deduped


def _clean_name(name: str) -> str:
    """Clean up subsidiary name.

    Args:
        name: String extracted for subsidiary name

    Returns:
        cleaned name string
    """
    name = _html.unescape(name)
    name = _INVISIBLE_CHARS_RE.sub("", name)
    name = name.replace("\xa0", " ")  # ← this line fixes most of the J&J diff
    return " ".join(name.split())
