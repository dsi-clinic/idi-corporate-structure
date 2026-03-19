# Standard application imports
import pathlib
from dataclasses import dataclass


_REMOTE_SCHEMES = ("s3://", "https://", "http://", "gs://")


def _is_local(path: str) -> bool:
    return not path.startswith(_REMOTE_SCHEMES)


@dataclass
class Filing:
    cik: str
    filing_date: str
    form_type: str
    accession_number: str
    directory: str
    primary_document: str


@dataclass
class PipelineConfig:
    input_file: str
    failure_file: str
    failure_flush_every: int = 50

    def __post_init__(self) -> None:
        "Validate existence of local files."
        if _is_local(self.input_file) and not pathlib.Path(self.input_file).exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        if _is_local(self.failure_file) and not pathlib.Path(self.failure_file).parent.exists():
            raise FileNotFoundError(
                f"Failure file directory does not exist: {pathlib.Path(self.failure_file).parent}"
            )


@dataclass
class PipelineStats:
    total_filing: int = 0
    failed_filings: int = 0
    skipped_filings: int = 0
    total_subsidiaries: int = 0
    failed_subsidiaries: int = 0


@dataclass
class Subsidiary:
    parent_cik: str
    name: str
    location: str
    filing_date: str
