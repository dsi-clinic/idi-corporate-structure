"""Shared fixtures for the test suite."""

import zipfile
from unittest.mock import MagicMock

import pytest

from idi_corporate_structure.api import SecClient
from idi_corporate_structure.extractor import GptExtractor
from idi_corporate_structure.pipeline import SubsidiaryPipeline
from idi_corporate_structure.types import Filing, PipelineConfig

# ── Data helpers ──────────────────────────────────────────────────────────────


def make_cik_json(
    forms: list | None = None,
    accession_numbers: list | None = None,
    primary_documents: list | None = None,
    filing_dates: list | None = None,
    cik: str = "",
) -> dict:
    """Build a minimal CIK JSON payload matching the SEC submissions.zip format."""
    payload: dict = {
        "filings": {
            "recent": {
                "form": forms or [],
                "accessionNumber": accession_numbers or [],
                "primaryDocument": primary_documents or [],
                "filingDate": filing_dates or [],
            }
        }
    }
    if cik:
        payload["cik"] = cik
    return payload


def make_directory_response(items: list | None = None) -> dict:
    """Build a minimal SEC index.json response."""
    return {
        "status_code": 200,
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/index.json",
        "data": {"directory": {"item": items or [], "name": "000032019324000123"}},
    }


def make_exhibit_response(
    content: str = (
        "<html><body>\nApple Operations LLC (Delaware)\nApple Europe Ltd (Ireland)\n</body></html>"
    ),
) -> dict:
    """Build a minimal SEC exhibit HTTP response dict."""
    return {
        "status_code": 200,
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/ex21.htm",
        "data": content,
    }


# ── Core fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_20f_filing() -> Filing:
    """A realistic 20-F Filing dataclass instance (Coincheck Group)."""
    return Filing(
        cik="0001913847",
        filing_date="2025-07-30",
        form_type="20-F",
        accession_number="0001628280-25-036727",
        directory="https://www.sec.gov/Archives/edgar/data/1913847/000162828025036727/index.json",
        primary_document="",
        company_name="Coincheck Group",
    )


@pytest.fixture
def sample_filing() -> Filing:
    """A realistic Filing dataclass instance."""
    return Filing(
        cik="0000320193",
        filing_date="2024-09-28",
        form_type="10-K",
        accession_number="0000320193-24-000123",
        directory="https://www.sec.gov/Archives/edgar/data/0000320193/000032019324000123/index.json",
        primary_document="https://www.sec.gov/Archives/edgar/data/0000320193/000032019324000123/aapl-20240928.htm",
        company_name="APPLE INC",
        location="CA",
    )


@pytest.fixture
def mock_sec_client() -> MagicMock:
    """A MagicMock that satisfies the SecClient interface."""
    client = MagicMock(spec=SecClient)
    client.SEC_HEADERS = {"User-Agent": "test test@test.com"}
    client.SEC_URL = "https://www.sec.gov/Archives/edgar/data"
    client.rate_limit.return_value = None
    return client


@pytest.fixture
def mock_extractor() -> MagicMock:
    """A MagicMock GptExtractor that returns an empty subsidiary list by default."""
    extractor = MagicMock(spec=GptExtractor)
    extractor.extract.return_value = ([], 0, 0, 1)
    return extractor


@pytest.fixture
def pipeline(tmp_path, mock_sec_client, mock_extractor) -> SubsidiaryPipeline:
    """A SubsidiaryPipeline wired with a temp zip, temp failure file, and mocked dependencies."""
    input_zip = tmp_path / "submissions.zip"
    with zipfile.ZipFile(input_zip, "w"):
        pass

    config = PipelineConfig(
        input_file=str(input_zip),
        failure_file=str(tmp_path / "failures.json"),
        output_file=str(tmp_path / "subsidiaries.parquet"),
        rate_limit=0.0,
        num_workers=2,
        failure_flush_every=100,
    )

    return SubsidiaryPipeline(
        config=config,
        sec_client=mock_sec_client,
        extractor=mock_extractor,
    )
