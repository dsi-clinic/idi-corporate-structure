"""Data types for the corporate structure pipeline."""

# Standard application imports
import pathlib
import re
import threading
from dataclasses import dataclass

_REMOTE_SCHEMES = ("s3://", "https://", "http://", "gs://")


def _is_local(path: str) -> bool:
    """Return True if the path refers to a local filesystem location.

    A path is considered remote when it begins with one of the known URI
    schemes in ``_REMOTE_SCHEMES`` (``s3://``, ``https://``, ``http://``,
    ``gs://``).

    Args:
        path: File path or URI string to test.

    Returns:
        True if the path does not start with a known remote scheme,
        False otherwise.
    """
    return not path.startswith(_REMOTE_SCHEMES)


@dataclass
class Filing:
    """Represents a single SEC 10-K filing with its metadata and document URLs."""

    cik: str
    filing_date: str
    form_type: str
    accession_number: str
    directory: str
    primary_document: str
    company_name: str = ""
    location: str = ""
    filename: str = ""

    @property
    def exhibit_type(self) -> str:
        """Retrun the exhibit number for the filing's subsidiary list.

        10-K filers use Exhibit 21; 20-F filers use Exhibit 8.
        """
        return "8" if re.match(r"20-?F", self.form_type) else "21"


@dataclass
class PipelineConfig:
    """Configuration for the subsidiary pipeline."""

    input_file: str
    failure_file: str
    output_file: str
    openai_api_key: str = ""
    failure_flush_every: int = 50
    rate_limit: float = 0.2
    num_workers: int = 10

    def __post_init__(self) -> None:
        """Validate existence of local files."""
        if _is_local(self.input_file) and not pathlib.Path(self.input_file).exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        if _is_local(self.failure_file) and not pathlib.Path(self.failure_file).parent.exists():
            pathlib.Path(self.failure_file).parent.mkdir(parents=True, exist_ok=True)
        if _is_local(self.output_file) and not pathlib.Path(self.output_file).parent.exists():
            pathlib.Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineStats:
    """Thread-safe counters tracking pipeline progress and failures."""

    total_filing: int = 0
    failed_filings: int = 0
    skipped_filings: int = 0
    total_subsidiaries: int = 0
    failed_subsidiaries: int = 0

    def __post_init__(self) -> None:
        """Initialize the pipeline stats."""
        self._lock = threading.Lock()

    def increment(self, field: str, n: int = 1) -> None:
        """Increment the pipeline stats by a given amount.

        Args:
            field: The field to increment
            n: The amount to increment the field by
        """
        with self._lock:
            setattr(self, field, getattr(self, field) + n)


@dataclass
class Subsidiary:
    """A single subsidiary entity extracted from an Exhibit 21 document."""

    parent_cik: str
    filing_date: str
    form_type: str
    exhibit_type: str
    accession_number: str
    exhibit_url: str
    name: str
    location: str
    parent_name: str = ""
    parent_location: str = ""
