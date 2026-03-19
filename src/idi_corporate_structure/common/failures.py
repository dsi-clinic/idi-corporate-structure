"""Generic failure classification and registry for pipeline processors."""

# Standard library imports
import json
import pathlib
from abc import ABC, abstractmethod
from enum import StrEnum

# Application imports
from idi_corporate_structure.common.logs import get_logger
from idi_corporate_structure.common.storage import load_json, save_json

_MIN_ENTRY_LEN = 2


class FailureClassifier(ABC):
    """Base class for failure classification. Subclasses implement domain-specific logic."""

    @property
    @abstractmethod
    def do_not_retry(self) -> frozenset:
        """Return the set of failure types that should not be retried."""
        ...

    def is_retryable(self, failure_type: StrEnum) -> bool:
        """Check if a failure type should be retried.

        Args:
            failure_type: The type of failure.

        Returns:
            True if the failure is transient and should be retried.
        """
        return failure_type not in self.do_not_retry

    @abstractmethod
    def classify_from_response(self, response: dict, **kwargs) -> StrEnum:
        """Classify a failure from an API response.

        Args:
            response: API response dict with status_code and optional error.

        Returns:
            The classified failure type.
        """
        ...


class FailureRegistry:
    """Persists permanent failures to avoid retrying entities that will always fail."""

    def __init__(self, file_path: str, classifier: FailureClassifier, flush_every: int = 10) -> None:
        """Initialize the FailureRegistry.

        Args:
            file_path: Path to the JSON file for persistence.
            flush_every: Number of new failures to buffer before writing to disk.
        """
        self.file_path = file_path
        self._classifier = classifier
        self._flush_every = flush_every
        self._pending = 0
        self._entries: set[tuple[str, str]] = set()
        self._reasons: dict[tuple[str, str], str] = {}
        self.logger = get_logger("FailureRegistry")
        self.load()

    def load(self) -> None:
        """Load entries from the persistence file."""
        if not self.file_path or (
            not self.file_path.startswith("s3://") and not pathlib.Path(self.file_path).exists()
        ):
            self._entries = set()
            self._reasons = {}
            return

        try:
            data = load_json(self.file_path, return_type="dict")
        except json.JSONDecodeError:
            self._entries = set()
            self._reasons = {}
            return

        if not isinstance(data, dict):
            self._entries = set()
            self._reasons = {}
            return

        entries_data = data.get("entries", [])
        reasons_data = data.get("reasons", {})

        self._entries = {tuple(e) for e in entries_data if len(e) >= _MIN_ENTRY_LEN}
        self._reasons = {}
        for entry in self._entries:
            key = f"{entry[0]} {entry[1]}"
            if key in reasons_data:
                self._reasons[entry] = reasons_data[key]

    def save(self) -> None:
        """Persist entries to the file."""
        if not self.file_path:
            return

        entries_list = [list(e) for e in self._entries]
        reasons_dict = {f"{e[0]} {e[1]}": self._reasons.get(e, "") for e in self._entries}
        save_json(self.file_path, {"entries": entries_list, "reasons": reasons_dict})

    def add(self, cik: str, accession_number: str, failure_type: StrEnum) -> None:
        """Add a permanent failure entry.

        Args:
            cik: The CIK of the filing entity.
            accession_number: The accession number of the filing.
            reason: Optional reason for debugging.
        """
        if self._classifier.is_retryable(failure_type):
            return

        key = (cik, accession_number)
        if key in self._entries:
            return

        self._entries.add(key)
        self._reasons[key] = str(failure_type)

        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        """Write all buffered failures to disk and reset the pending counter."""
        self.save()
        self._pending = 0

    def __contains__(self, key: tuple[str, str]) -> bool:
        """Set-like membership check.

        Args:
            key: Tuple of (cik, accession_number).

        Returns:
            True if the filing should not be retried.
        """
        return key in self._entries
