"""Tests for processor.pipeline — SubsidiaryPipeline."""

import io
import json
import queue
import threading
import zipfile
from unittest.mock import MagicMock

import pandas as pd

from idi_corporate_structure.processor.extractor import (
    ExtractionTimeoutError,
    ExtractionTruncatedError,
    _html_to_text as html_to_text,
)
from idi_corporate_structure.processor.failures import FailureType
from idi_corporate_structure.processor.types import Filing, Subsidiary
from tests.conftest import make_cik_json, make_directory_response, make_exhibit_response

# ── _zip_file_data ────────────────────────────────────────────────────────────


class TestZipFileData:
    """Tests for SubsidiaryPipeline._zip_file_data()."""

    def test_returns_zip_for_valid_equal_length_data(self, pipeline):
        data = make_cik_json(
            forms=["10-K", "10-Q"],
            accession_numbers=["0001-24-000001", "0001-24-000002"],
            primary_documents=["doc1.htm", "doc2.htm"],
            filing_dates=["2024-09-28", "2024-06-30"],
        )
        result = pipeline._zip_file_data(
            data, "CIK0000000001.json", "0000000001", MagicMock(spec=zipfile.ZipFile)
        )

        assert result is not None
        rows = list(result)
        assert len(rows) == 2
        assert rows[0][0] == "10-K"

    def test_returns_none_for_empty_data(self, pipeline):
        data = make_cik_json()
        result = pipeline._zip_file_data(
            data, "CIK0000000001.json", "0000000001", MagicMock(spec=zipfile.ZipFile)
        )

        assert result is None

    def test_returns_none_for_mismatched_lengths(self, pipeline):
        data = make_cik_json(
            forms=["10-K", "10-Q"],
            accession_numbers=["0001-24-000001"],  # mismatched — only 1
            primary_documents=["doc1.htm", "doc2.htm"],
            filing_dates=["2024-09-28", "2024-06-30"],
        )
        result = pipeline._zip_file_data(
            data, "CIK0000000001.json", "0000000001", MagicMock(spec=zipfile.ZipFile)
        )

        assert result is None

    def test_records_mismatched_failure(self, pipeline):
        data = make_cik_json(
            forms=["10-K"],
            accession_numbers=["ACC1", "ACC2"],  # mismatched
            primary_documents=["doc1.htm"],
            filing_dates=["2024-01-01"],
        )
        pipeline._zip_file_data(
            data, "CIK0000000001.json", "0000000001", MagicMock(spec=zipfile.ZipFile)
        )

        assert pipeline.stats.failed_filings == 1

    def test_returns_none_for_missing_filings_key(self, pipeline):
        result = pipeline._zip_file_data(
            {}, "CIK0000000001.json", "0000000001", MagicMock(spec=zipfile.ZipFile)
        )
        assert result is None


# ── _create_filing ────────────────────────────────────────────────────────────


class TestCreateFiling:
    """Tests for SubsidiaryPipeline._create_filing()."""

    _COMPANY_DATA = {
        "cik": "0000320193",
        "name": "APPLE INC",
        "location": "CA",
        "filename": "CIK0000320193.json",
    }

    def test_builds_directory_url(self, pipeline):
        filing = pipeline._create_filing(
            accession_number="0000320193-24-000123",
            primary_document="aapl-20240928.htm",
            filing_date="2024-09-28",
            form="10-K",
            company_data=self._COMPANY_DATA,
        )
        assert (
            filing.directory
            == "https://www.sec.gov/Archives/edgar/data/0000320193/000032019324000123/index.json"
        )

    def test_strips_dashes_from_accession_for_url(self, pipeline):
        filing = pipeline._create_filing(
            accession_number="0000320193-24-000123",
            primary_document="doc.htm",
            filing_date="2024-09-28",
            form="10-K",
            company_data=self._COMPANY_DATA,
        )
        assert "0000320193-24-000123" not in filing.directory
        assert "000032019324000123" in filing.directory

    def test_builds_primary_document_url_for_htm(self, pipeline):
        filing = pipeline._create_filing(
            accession_number="0000320193-24-000123",
            primary_document="aapl-20240928.htm",
            filing_date="2024-09-28",
            form="10-K",
            company_data=self._COMPANY_DATA,
        )
        assert "aapl-20240928.htm" in filing.primary_document

    def test_sets_empty_primary_for_non_htm(self, pipeline):
        filing = pipeline._create_filing(
            accession_number="0000320193-24-000123",
            primary_document="aapl-20240928.xyz",
            filing_date="2024-09-28",
            form="10-K",
            company_data=self._COMPANY_DATA,
        )
        assert filing.primary_document == ""

    def test_sets_empty_primary_for_blank_document(self, pipeline):
        filing = pipeline._create_filing(
            accession_number="0000320193-24-000123",
            primary_document="",
            filing_date="2024-09-28",
            form="10-K",
            company_data=self._COMPANY_DATA,
        )
        assert filing.primary_document == ""

    def test_filing_metadata_is_preserved(self, pipeline):
        filing = pipeline._create_filing(
            accession_number="0000320193-24-000123",
            primary_document="doc.htm",
            filing_date="2024-09-28",
            form="10-K",
            company_data=self._COMPANY_DATA,
        )
        assert filing.cik == "0000320193"
        assert filing.filing_date == "2024-09-28"
        assert filing.form_type == "10-K"
        assert filing.accession_number == "0000320193-24-000123"
        assert filing.company_name == "APPLE INC"
        assert filing.location == "CA"


# ── _parse_file ───────────────────────────────────────────────────────────────


class TestParseFile:
    """Tests for SubsidiaryPipeline._parse_file()."""

    def _make_zip_with_cik(self, cik_filename: str, payload: dict) -> MagicMock:
        """Return a mock ZipFile whose open() yields a BytesIO of the JSON payload."""
        raw = json.dumps(payload).encode()
        mock_zf = MagicMock(spec=zipfile.ZipFile)
        mock_zf.open.return_value.__enter__ = lambda s: io.BytesIO(raw)
        mock_zf.open.return_value.__exit__ = MagicMock(return_value=False)
        return mock_zf

    def test_returns_10k_filings(self, pipeline):
        data = make_cik_json(
            forms=["10-K", "10-Q", "8-K"],
            accession_numbers=["ACC-10K", "ACC-10Q", "ACC-8K"],
            primary_documents=["10k.htm", "10q.htm", "8k.htm"],
            filing_dates=["2024-09-28", "2024-06-30", "2024-03-01"],
            cik="0000000001",
        )
        mock_zf = self._make_zip_with_cik("CIK0000000001.json", data)

        filings = pipeline._parse_file(mock_zf, "CIK0000000001.json")

        assert len(filings) == 1
        assert filings[0].form_type == "10-K"
        assert filings[0].cik == "0000000001"

    def test_matches_10k_and_10_k_variants(self, pipeline):
        data = make_cik_json(
            forms=["10-K", "10-K/A", "10K"],
            accession_numbers=["ACC1", "ACC2", "ACC3"],
            primary_documents=["d1.htm", "d2.htm", "d3.htm"],
            filing_dates=["2024-01-01", "2024-02-01", "2024-03-01"],
        )
        mock_zf = self._make_zip_with_cik("CIK0000000001.json", data)

        filings = pipeline._parse_file(mock_zf, "CIK0000000001.json")

        # All three match IS_10K pattern
        assert len(filings) == 3

    def test_returns_20f_filings(self, pipeline):
        data = make_cik_json(
            forms=["20-F", "10-Q"],
            accession_numbers=["ACC-20F", "ACC-10Q"],
            primary_documents=["20f.htm", "10q.htm"],
            filing_dates=["2025-07-30", "2025-04-30"],
            cik="0001913847",
        )
        mock_zf = self._make_zip_with_cik("CIK0001913847.json", data)

        filings = pipeline._parse_file(mock_zf, "CIK0001913847.json")

        assert len(filings) == 1
        assert filings[0].form_type == "20-F"
        assert filings[0].exhibit_type == "8"

    def test_ignores_non_cik_files(self, pipeline):
        mock_zf = MagicMock(spec=zipfile.ZipFile)

        filings = pipeline._parse_file(mock_zf, "company-tickers.json")

        assert filings == []
        mock_zf.open.assert_not_called()

    def test_returns_empty_for_cik_with_no_10k(self, pipeline):
        data = make_cik_json(
            forms=["8-K", "DEF 14A"],
            accession_numbers=["ACC1", "ACC2"],
            primary_documents=["doc1.htm", "doc2.htm"],
            filing_dates=["2024-01-01", "2024-02-01"],
        )
        mock_zf = self._make_zip_with_cik("CIK0000000001.json", data)

        filings = pipeline._parse_file(mock_zf, "CIK0000000001.json")

        assert filings == []

    def test_returns_empty_for_cik_with_empty_filings(self, pipeline):
        data = make_cik_json()
        mock_zf = self._make_zip_with_cik("CIK0000000001.json", data)

        filings = pipeline._parse_file(mock_zf, "CIK0000000001.json")

        assert filings == []

    def test_increments_stats_for_10k(self, pipeline):
        data = make_cik_json(
            forms=["10-K"],
            accession_numbers=["ACC1"],
            primary_documents=["doc.htm"],
            filing_dates=["2024-01-01"],
        )
        mock_zf = self._make_zip_with_cik("CIK0000000001.json", data)

        pipeline._parse_file(mock_zf, "CIK0000000001.json")

        assert pipeline.stats.total_filing == 1

    def test_reads_cik_from_json_data(self, pipeline):
        data = make_cik_json(
            forms=["10-K"],
            accession_numbers=["ACC1"],
            primary_documents=["doc.htm"],
            filing_dates=["2024-01-01"],
            cik="0000320193",
        )
        mock_zf = self._make_zip_with_cik("CIK0000320193.json", data)

        filings = pipeline._parse_file(mock_zf, "CIK0000320193.json")

        assert filings[0].cik == "0000320193"

    def test_parses_company_name_and_location_from_json(self, pipeline):
        data = make_cik_json(
            forms=["10-K"],
            accession_numbers=["ACC1"],
            primary_documents=["doc.htm"],
            filing_dates=["2024-01-01"],
        )
        data["name"] = "APPLE INC"
        data["stateOfIncorporation"] = "CA"
        mock_zf = self._make_zip_with_cik("CIK0000320193.json", data)

        filings = pipeline._parse_file(mock_zf, "CIK0000320193.json")

        assert filings[0].company_name == "APPLE INC"
        assert filings[0].location == "CA"

    def test_company_name_and_location_default_to_empty_string(self, pipeline):
        data = make_cik_json(
            forms=["10-K"],
            accession_numbers=["ACC1"],
            primary_documents=["doc.htm"],
            filing_dates=["2024-01-01"],
        )
        mock_zf = self._make_zip_with_cik("CIK0000320193.json", data)

        filings = pipeline._parse_file(mock_zf, "CIK0000320193.json")

        assert filings[0].company_name == ""
        assert filings[0].location == ""


# ── _fetch_directory ──────────────────────────────────────────────────────────


class TestFetchDirectory:
    """Tests for SubsidiaryPipeline._fetch_directory()."""

    def test_returns_items_on_success(self, pipeline, sample_filing):
        items = [
            {"name": "ex21.htm", "type": "text.gif"},
            {"name": "aapl-20240928.htm", "type": "text.gif"},
        ]
        pipeline.sec_client.query_endpoint.return_value = make_directory_response(items)

        result = pipeline._fetch_directory(sample_filing)

        assert result == items

    def test_returns_empty_list_when_no_directory_key(self, pipeline, sample_filing):
        pipeline.sec_client.query_endpoint.return_value = {
            "status_code": 200,
            "url": "https://...",
            "data": {},  # missing "directory" key
        }

        result = pipeline._fetch_directory(sample_filing)

        assert result == []

    def test_records_failure_when_no_directory(self, pipeline, sample_filing):
        pipeline.sec_client.query_endpoint.return_value = {
            "status_code": 200,
            "url": "https://...",
            "data": {},
        }

        pipeline._fetch_directory(sample_filing)

        assert pipeline.stats.failed_subsidiaries == 1

    def test_returns_empty_list_for_empty_directory(self, pipeline, sample_filing):
        pipeline.sec_client.query_endpoint.return_value = make_directory_response([])

        result = pipeline._fetch_directory(sample_filing)

        assert result == []


# ── _fetch_exhibit_content ────────────────────────────────────────────────────


class TestFetchExhibitContent:
    """Tests for SubsidiaryPipeline._fetch_exhibit_content()."""

    def test_fetches_exhibit_21_by_name(self, pipeline, sample_filing):
        # The regex is r"\BEX" — EX must NOT be at a word boundary (i.e. must
        # be preceded by a word character).  "d12345ex21.htm" → "D12345EX21.HTM"
        # has "EX" preceded by "5" (\w), so \B matches.
        item = {"name": "d12345ex21.htm", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = make_exhibit_response()

        result = pipeline._fetch_exhibit_content(sample_filing, item)

        assert result["url"] is not None
        assert result["data"] == html_to_text(make_exhibit_response()["data"])
        pipeline.sec_client.query_endpoint.assert_called_once()

    def test_fetches_exhibit_named_21_prefix(self, pipeline, sample_filing):
        item = {"name": "21subsidiaries.htm", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = make_exhibit_response()

        result = pipeline._fetch_exhibit_content(sample_filing, item)

        assert result != {}

    def test_skips_non_exhibit_21_files(self, pipeline, sample_filing):
        item = {"name": "aapl-20240928.htm", "type": "text.gif"}

        result = pipeline._fetch_exhibit_content(sample_filing, item)

        assert result == {}
        pipeline.sec_client.query_endpoint.assert_not_called()

    def test_skips_unsupported_file_type(self, pipeline, sample_filing):
        item = {"name": "d12345ex21.xml", "type": "text.gif"}

        result = pipeline._fetch_exhibit_content(sample_filing, item)

        assert result == {}
        pipeline.sec_client.query_endpoint.assert_not_called()

    def test_returns_empty_when_no_content_returned(self, pipeline, sample_filing):
        item = {"name": "d12345ex21.htm", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = {}  # no data key

        result = pipeline._fetch_exhibit_content(sample_filing, item)

        assert result == {}
        assert pipeline.stats.failed_subsidiaries == 1

    def test_does_not_fetch_for_skipped_item(self, pipeline, sample_filing):
        # "primarydoc.htm" contains neither EX (preceded by \w) nor SUB
        item = {"name": "primarydoc.htm", "type": "text.gif"}

        pipeline._fetch_exhibit_content(sample_filing, item)

        pipeline.sec_client.query_endpoint.assert_not_called()

    def test_builds_correct_url(self, pipeline, sample_filing):
        item = {"name": "d12345ex21.htm", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = make_exhibit_response()

        pipeline._fetch_exhibit_content(sample_filing, item)

        call_kwargs = pipeline.sec_client.query_endpoint.call_args
        called_url = call_kwargs.kwargs.get("sec_url") or call_kwargs.args[0]
        assert "0000320193" in called_url
        assert "d12345ex21.htm" in called_url
        assert "-" not in called_url.split("/")[-2]  # accession number has no dashes

    def test_increments_htm_counter(self, pipeline, sample_filing):
        item = {"name": "d12345ex21.htm", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = make_exhibit_response()

        pipeline._fetch_exhibit_content(sample_filing, item)

        assert pipeline.stats.htm_exhibits == 1
        assert pipeline.stats.html_exhibits == 0
        assert pipeline.stats.txt_exhibits == 0
        assert pipeline.stats.pdf_exhibits == 0

    def test_increments_html_counter(self, pipeline, sample_filing):
        item = {"name": "d12345ex21.html", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = make_exhibit_response()

        pipeline._fetch_exhibit_content(sample_filing, item)

        assert pipeline.stats.html_exhibits == 1

    def test_increments_txt_counter(self, pipeline, sample_filing):
        item = {"name": "d12345ex21.txt", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = make_exhibit_response()

        pipeline._fetch_exhibit_content(sample_filing, item)

        assert pipeline.stats.txt_exhibits == 1

    def test_fetches_pdf_exhibit_and_extracts_text(self, pipeline, sample_filing, mocker):
        item = {"name": "d12345ex21.pdf", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = {"data": b"%PDF content"}

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Subsidiary A — Delaware"
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]
        mocker.patch(
            "idi_corporate_structure.processor.pipeline.pdfplumber.open", return_value=mock_pdf
        )

        result = pipeline._fetch_exhibit_content(sample_filing, item)

        assert result["data"] == "Subsidiary A — Delaware"
        assert "d12345ex21.pdf" in result["url"]

    def test_logs_warning_for_pdf_exhibit(self, pipeline, sample_filing, mocker):
        item = {"name": "d12345ex21.pdf", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = {"data": b"%PDF content"}

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "text"
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]
        mocker.patch(
            "idi_corporate_structure.processor.pipeline.pdfplumber.open", return_value=mock_pdf
        )

        mock_warn = mocker.patch.object(pipeline.logger, "warning")
        pipeline._fetch_exhibit_content(sample_filing, item)

        mock_warn.assert_called_once()

    def test_records_failure_on_pdf_extraction_error(self, pipeline, sample_filing, mocker):
        item = {"name": "d12345ex21.pdf", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = {"data": b"%PDF content"}
        mocker.patch(
            "idi_corporate_structure.processor.pipeline.pdfplumber.open",
            side_effect=Exception("corrupt PDF"),
        )

        result = pipeline._fetch_exhibit_content(sample_filing, item)

        assert result == {}
        assert pipeline.stats.failed_subsidiaries == 1

    # ── 20-F / Exhibit 8 ──────────────────────────────────────────────────────

    def test_fetches_exhibit_8_for_20f_embedded_ex(self, pipeline, sample_20f_filing):
        """Filename with EX embedded (not at start) and exhibit 8 number."""
        item = {"name": "d12345ex8.htm", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = make_exhibit_response()

        result = pipeline._fetch_exhibit_content(sample_20f_filing, item)

        assert result != {}
        pipeline.sec_client.query_endpoint.assert_called_once()

    def test_fetches_real_coincheck_exhibit_8_filename(self, pipeline, sample_20f_filing):
        """Real filename from Coincheck 20-F: passes via 'SUB' in 'SUBSIDIARIES'."""
        item = {"name": "ex81_listofsubsidiariesofc.htm", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = make_exhibit_response()

        result = pipeline._fetch_exhibit_content(sample_20f_filing, item)

        assert result != {}

    def test_skips_exhibit_21_for_20f_filing(self, pipeline, sample_20f_filing):
        """20-F filers use Exhibit 8 — Exhibit 21 files should be ignored."""
        item = {"name": "d12345ex21.htm", "type": "text.gif"}

        result = pipeline._fetch_exhibit_content(sample_20f_filing, item)

        assert result == {}
        pipeline.sec_client.query_endpoint.assert_not_called()

    def test_skips_exhibit_8_for_10k_filing(self, pipeline, sample_filing):
        """10-K filers use Exhibit 21 — Exhibit 8 (tax opinion) should be ignored."""
        item = {"name": "d12345ex8.htm", "type": "text.gif"}

        result = pipeline._fetch_exhibit_content(sample_filing, item)

        assert result == {}
        pipeline.sec_client.query_endpoint.assert_not_called()

    def test_sub_fallback_matches_for_20f(self, pipeline, sample_20f_filing):
        """Files containing 'SUB' pass the filter for 20-F filings too."""
        item = {"name": "subsidiaries.htm", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = make_exhibit_response()

        result = pipeline._fetch_exhibit_content(sample_20f_filing, item)

        assert result != {}

    def test_builds_correct_url_for_20f_exhibit_8(self, pipeline, sample_20f_filing):
        item = {"name": "d12345ex8.htm", "type": "text.gif"}
        pipeline.sec_client.query_endpoint.return_value = make_exhibit_response()

        pipeline._fetch_exhibit_content(sample_20f_filing, item)

        call_kwargs = pipeline.sec_client.query_endpoint.call_args
        called_url = call_kwargs.kwargs.get("sec_url") or call_kwargs.args[0]
        assert "0001913847" in called_url
        assert "d12345ex8.htm" in called_url
        assert "-" not in called_url.split("/")[-2]  # accession number has no dashes


# ── _fetch_exhibit ────────────────────────────────────────────────────────────


class TestFetchExhibit:
    """Tests for SubsidiaryPipeline._fetch_exhibit()."""

    def test_returns_only_non_empty_contents(self, pipeline, sample_filing, mocker):
        directory_items = [
            {"name": "ex21.htm", "type": "text.gif"},  # matches → has content
            {"name": "primarydoc.htm", "type": "text.gif"},  # no match → skipped
        ]
        mocker.patch.object(pipeline, "_fetch_directory", return_value=directory_items)
        mocker.patch.object(
            pipeline,
            "_fetch_exhibit_content",
            side_effect=[make_exhibit_response(), {}],
        )

        result = pipeline._fetch_exhibit(sample_filing)

        assert len(result) == 1
        assert result[0]["url"] is not None

    def test_returns_empty_list_when_directory_empty(self, pipeline, sample_filing, mocker):
        mocker.patch.object(pipeline, "_fetch_directory", return_value=[])

        result = pipeline._fetch_exhibit(sample_filing)

        assert result == []

    def test_returns_empty_list_when_no_matching_exhibits(self, pipeline, sample_filing, mocker):
        directory_items = [{"name": "primarydoc.htm", "type": "text.gif"}]
        mocker.patch.object(pipeline, "_fetch_directory", return_value=directory_items)
        mocker.patch.object(pipeline, "_fetch_exhibit_content", return_value={})

        result = pipeline._fetch_exhibit(sample_filing)

        assert result == []


# ── process ───────────────────────────────────────────────────────────────────


class TestProcess:
    """Tests for SubsidiaryPipeline.process()."""

    def _make_filing(self, n: int) -> Filing:
        return Filing(
            cik=f"CIK{n:010d}",
            filing_date="2024-09-28",
            form_type="10-K",
            accession_number=f"{n:010d}-24-{n:06d}",
            directory=f"https://www.sec.gov/Archives/edgar/data/{n}/000001/index.json",
            primary_document="",
        )

    def _make_subsidiary(self, parent_cik: str) -> Subsidiary:
        return Subsidiary(
            parent_cik=parent_cik,
            name="Test Sub LLC",
            location="Delaware",
            filing_date="2024-09-28",
            form_type="10-K",
            exhibit_type="21",
            accession_number="ACC001",
            exhibit_url="https://example.com/ex21.htm",
        )

    def test_returns_subsidiaries_from_extractor(self, pipeline, mocker):
        filings = [self._make_filing(i) for i in range(3)]
        exhibit_content = [make_exhibit_response()]

        mocker.patch.object(pipeline, "_fetch_exhibit", return_value=exhibit_content)
        pipeline.extractor.extract.side_effect = [
            ([self._make_subsidiary(f.cik)], 0, 0) for f in filings
        ]

        results = pipeline.process(filings)

        assert len(results) == 3
        assert all(isinstance(r, Subsidiary) for r in results)

    def test_returns_empty_list_for_empty_input(self, pipeline):
        results = pipeline.process([])
        assert results == []

    def test_calls_fetch_exhibit_for_each_filing(self, pipeline, mocker):
        filings = [self._make_filing(i) for i in range(4)]
        mock_fetch = mocker.patch.object(pipeline, "_fetch_exhibit", return_value=[])
        pipeline.extractor.extract.return_value = []

        pipeline.process(filings)

        assert mock_fetch.call_count == 4

    def test_handles_extractor_exception_gracefully(self, pipeline, mocker):
        """A failed extraction should not crash the pipeline — other filings still processed."""
        filings = [self._make_filing(i) for i in range(3)]
        mocker.patch.object(pipeline, "_fetch_exhibit", return_value=[make_exhibit_response()])

        subsidiary = self._make_subsidiary("CIK0000000000")
        pipeline.extractor.extract.side_effect = [
            RuntimeError("GPT error"),
            ([subsidiary], 0, 0),
            ([subsidiary], 0, 0),
        ]

        results = pipeline.process(filings)

        # One failure + two successes
        assert len(results) == 2
        assert pipeline.stats.failed_subsidiaries >= 1

    def test_increments_total_subsidiaries(self, pipeline, mocker):
        filings = [self._make_filing(0)]
        mocker.patch.object(pipeline, "_fetch_exhibit", return_value=[make_exhibit_response()])
        pipeline.extractor.extract.return_value = (
            [self._make_subsidiary("CIK0000000000"), self._make_subsidiary("CIK0000000000")],
            0,
            0,
        )

        pipeline.process(filings)

        assert pipeline.stats.total_subsidiaries == 2


# ── _extract_worker ───────────────────────────────────────────────────────────


class TestExtractWorker:
    """Tests for SubsidiaryPipeline._extract_worker()."""

    def _start_worker(self, pipeline, work_queue, results_queue):
        threading.Thread(
            target=pipeline._extract_worker,
            args=(work_queue, results_queue),
            daemon=True,
        ).start()

    def test_calls_extractor_with_filing_and_contents(self, pipeline, sample_filing):
        exhibit = make_exhibit_response()
        work_queue, results_queue = queue.Queue(), queue.Queue()
        self._start_worker(pipeline, work_queue, results_queue)

        work_queue.put((sample_filing, exhibit))
        work_queue.join()

        pipeline.extractor.extract.assert_called_once_with(sample_filing, exhibit)

    def test_puts_result_on_results_queue(self, pipeline, sample_filing):
        subsidiary = Subsidiary(
            parent_cik=sample_filing.cik,
            name="",
            location="",
            filing_date=sample_filing.filing_date,
            form_type=sample_filing.form_type,
            exhibit_type=sample_filing.exhibit_type,
            accession_number=sample_filing.accession_number,
            exhibit_url="",
        )
        pipeline.extractor.extract.return_value = ([subsidiary], 0, 0)

        work_queue, results_queue = queue.Queue(), queue.Queue()
        self._start_worker(pipeline, work_queue, results_queue)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        assert results_queue.get_nowait() == [subsidiary]

    def test_marks_work_task_done_on_success(self, pipeline, sample_filing):
        work_queue, results_queue = queue.Queue(), queue.Queue()
        self._start_worker(pipeline, work_queue, results_queue)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()  # completes only if task_done() was called

    def test_marks_work_task_done_on_exception(self, pipeline, sample_filing):
        pipeline.extractor.extract.side_effect = RuntimeError("GPT error")

        work_queue, results_queue = queue.Queue(), queue.Queue()
        self._start_worker(pipeline, work_queue, results_queue)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()  # completes only if task_done() is called in finally

    def test_increments_failed_subsidiaries_on_exception(self, pipeline, sample_filing):
        pipeline.extractor.extract.side_effect = RuntimeError("GPT error")

        work_queue, results_queue = queue.Queue(), queue.Queue()
        self._start_worker(pipeline, work_queue, results_queue)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        assert pipeline.stats.failed_subsidiaries == 1

    def test_records_failure_on_exception(self, pipeline, sample_filing, mocker):
        pipeline.extractor.extract.side_effect = RuntimeError("GPT error")
        spy = mocker.spy(pipeline.failure_registry, "add")

        work_queue, results_queue = queue.Queue(), queue.Queue()
        self._start_worker(pipeline, work_queue, results_queue)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        spy.assert_called_once_with(
            (sample_filing.cik, sample_filing.filename), FailureType.EXTRACTION_FAILED
        )

    def test_truncated_extraction_increments_truncated_and_failed(
        self, pipeline, sample_filing, mocker
    ):
        pipeline.extractor.extract.side_effect = ExtractionTruncatedError("output cut off")
        spy = mocker.spy(pipeline.failure_registry, "add")

        work_queue, results_queue = queue.Queue(), queue.Queue()
        self._start_worker(pipeline, work_queue, results_queue)

        work_queue.put((sample_filing, make_exhibit_response()))
        work_queue.join()

        assert pipeline.stats.truncated_extractions == 1
        assert pipeline.stats.failed_subsidiaries == 1
        spy.assert_called_once_with(
            (sample_filing.cik, sample_filing.filename), FailureType.TRUNCATED_ERROR
        )


# ── _results_worker ───────────────────────────────────────────────────────────


class TestResultsWorker:
    """Tests for SubsidiaryPipeline._results_worker()."""

    def _make_subsidiary(self, filing: Filing) -> Subsidiary:
        return Subsidiary(
            parent_cik=filing.cik,
            name="Sub Inc",
            location="Delaware",
            filing_date=filing.filing_date,
            form_type=filing.form_type,
            exhibit_type=filing.exhibit_type,
            accession_number=filing.accession_number,
            exhibit_url="",
        )

    def _start_worker(self, pipeline, results_queue, subsidiaries):
        threading.Thread(
            target=pipeline._results_worker,
            args=(results_queue, subsidiaries),
            daemon=True,
        ).start()

    def test_extends_subsidiaries_list(self, pipeline, sample_filing):
        subsidiary = self._make_subsidiary(sample_filing)
        results_queue = queue.Queue()
        subsidiaries = []
        self._start_worker(pipeline, results_queue, subsidiaries)

        results_queue.put([subsidiary])
        results_queue.join()

        assert subsidiaries == [subsidiary]

    def test_increments_total_subsidiaries(self, pipeline, sample_filing):
        batch = [self._make_subsidiary(sample_filing) for _ in range(3)]
        results_queue = queue.Queue()
        subsidiaries = []
        self._start_worker(pipeline, results_queue, subsidiaries)

        results_queue.put(batch)
        results_queue.join()

        assert pipeline.stats.total_subsidiaries == 3

    def test_marks_results_task_done(self, pipeline):
        results_queue = queue.Queue()
        self._start_worker(pipeline, results_queue, [])

        results_queue.put([])
        results_queue.join()  # completes only if task_done() was called


# ── save_output ───────────────────────────────────────────────────────────────


class TestSaveOutput:
    """Tests for SubsidiaryPipeline.save_output()."""

    def _make_subsidiary(self, name: str, accession: str = "0000320193-24-000123") -> Subsidiary:
        return Subsidiary(
            parent_cik="0000320193",
            parent_name="APPLE INC",
            parent_location="CA",
            name=name,
            location="Ireland",
            filing_date="2024-09-28",
            form_type="10-K",
            exhibit_type="21",
            accession_number=accession,
            exhibit_url="https://www.sec.gov/Archives/edgar/data/320193/ex21.htm",
        )

    def test_writes_parquet_file(self, pipeline):
        subsidiaries = [self._make_subsidiary("Apple Operations International")]

        pipeline.save_output(subsidiaries)

        assert pipeline.config.output_file
        result_df = pd.read_parquet(pipeline.config.output_file)
        assert len(result_df) == 1

    def test_output_contains_all_subsidiary_fields(self, pipeline):
        subsidiaries = [self._make_subsidiary("Apple Sales International")]

        pipeline.save_output(subsidiaries)

        result_df = pd.read_parquet(pipeline.config.output_file)
        assert result_df.iloc[0]["name"] == "Apple Sales International"
        assert result_df.iloc[0]["parent_cik"] == "0000320193"
        assert result_df.iloc[0]["location"] == "Ireland"

    def test_adds_date_added_column(self, pipeline):
        subsidiaries = [self._make_subsidiary("Apple Operations International")]

        pipeline.save_output(subsidiaries)

        result_df = pd.read_parquet(pipeline.config.output_file)
        assert "date_added" in result_df.columns
        assert result_df.iloc[0]["date_added"] is not None

    def test_deduplicates_within_filing(self, pipeline):
        """Same parent_cik + accession_number + name should be written once."""
        subsidiaries = [
            self._make_subsidiary("Apple Operations International"),
            self._make_subsidiary("Apple Operations International"),
        ]

        pipeline.save_output(subsidiaries)

        result_df = pd.read_parquet(pipeline.config.output_file)
        assert len(result_df) == 1

    def test_keeps_same_name_across_different_filings(self, pipeline):
        """Same subsidiary name in two different filings should produce two rows."""
        subsidiaries = [
            self._make_subsidiary(
                "Apple Operations International", accession="0000320193-23-000001"
            ),
            self._make_subsidiary(
                "Apple Operations International", accession="0000320193-24-000002"
            ),
        ]

        pipeline.save_output(subsidiaries)

        result_df = pd.read_parquet(pipeline.config.output_file)
        assert len(result_df) == 2

    def test_normalizes_location_strings_on_write(self, pipeline):
        """Footnote markers, sentinels, and country aliases should be canonicalized."""
        subs = [
            Subsidiary(
                parent_cik="0000000001",
                parent_name="ACME",
                parent_location="DE",
                name="Acme China Sub",
                location="PRC",
                filing_date="2024-01-01",
                form_type="10-K",
                exhibit_type="21",
                accession_number="0000000001-24-000001",
                exhibit_url="https://example.com/ex21.htm",
            ),
            Subsidiary(
                parent_cik="0000000001",
                parent_name="ACME",
                parent_location="E9",
                name="Acme Mexico Sub",
                location="Mexico(2)",
                filing_date="2024-01-01",
                form_type="10-K",
                exhibit_type="21",
                accession_number="0000000001-24-000001",
                exhibit_url="https://example.com/ex21.htm",
            ),
            Subsidiary(
                parent_cik="0000000001",
                parent_name="ACME",
                parent_location="L2",
                name="Acme Mystery Sub",
                location="Unknown",
                filing_date="2024-01-01",
                form_type="10-K",
                exhibit_type="21",
                accession_number="0000000001-24-000001",
                exhibit_url="https://example.com/ex21.htm",
            ),
        ]

        pipeline.save_output(subs)

        result_df = pd.read_parquet(pipeline.config.output_file).set_index("name")
        assert result_df.loc["Acme China Sub", "location"] == "China"
        assert result_df.loc["Acme China Sub", "parent_location"] == "Delaware"
        assert result_df.loc["Acme Mexico Sub", "location"] == "Mexico"
        assert result_df.loc["Acme Mexico Sub", "parent_location"] == "Cayman Islands"
        assert result_df.loc["Acme Mystery Sub", "location"] == ""
        assert result_df.loc["Acme Mystery Sub", "parent_location"] == "Ireland"


# ── display_stats ─────────────────────────────────────────────────────────────


class TestDisplayStats:
    """Tests for SubsidiaryPipeline.display_stats()."""

    def test_logs_filing_counts(self, pipeline):
        pipeline.stats.increment("total_filing", 10)
        pipeline.stats.increment("failed_filings", 2)

        with MagicMock() as mock_logger:
            pipeline.logger = mock_logger
            pipeline.display_stats()

        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "10" in logged
        assert "2" in logged

    def test_logs_subsidiary_counts(self, pipeline):
        pipeline.stats.increment("total_subsidiaries", 50)
        pipeline.stats.increment("failed_subsidiaries", 3)

        with MagicMock() as mock_logger:
            pipeline.logger = mock_logger
            pipeline.display_stats()

        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        assert "50" in logged
        assert "3" in logged

    def test_logs_section_headers(self, pipeline):
        with MagicMock() as mock_logger:
            pipeline.logger = mock_logger
            pipeline.display_stats()

        logged_args = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Filings" in arg for arg in logged_args)
        assert any("Subsidiaries" in arg for arg in logged_args)
        assert any("=" in arg for arg in logged_args)


# ── _filter_already_processed ─────────────────────────────────────────────────


class TestFilterAlreadyProcessed:
    """Tests for SubsidiaryPipeline._filter_already_processed()."""

    def _make_filing(
        self,
        cik: str = "0000320193",
        accession: str = "0000320193-24-000001",
        filename: str = "CIK0000320193.json",
    ) -> Filing:
        return Filing(
            cik=cik,
            filing_date="2024-09-28",
            form_type="10-K",
            accession_number=accession,
            directory="https://www.sec.gov/Archives/edgar/data/0000320193/index.json",
            primary_document="",
            filename=filename,
        )

    def _write_parquet(self, path: str, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_parquet(path)

    def test_returns_all_filings_when_no_parquet(self, pipeline):
        filings = [self._make_filing()]

        result = pipeline._filter_already_processed(filings)

        assert result == filings

    def test_returns_all_when_parquet_is_empty(self, pipeline):
        pd.DataFrame(columns=["parent_cik", "accession_number"]).to_parquet(
            pipeline.config.output_file
        )
        filing = self._make_filing()

        result = pipeline._filter_already_processed([filing])

        assert filing in result

    def test_returns_all_when_parquet_has_no_expected_columns(self, pipeline):
        pd.DataFrame().to_parquet(pipeline.config.output_file)
        filing = self._make_filing()

        result = pipeline._filter_already_processed([filing])

        assert filing in result

    def test_skips_already_processed_filing(self, pipeline):
        filing = self._make_filing()
        self._write_parquet(
            pipeline.config.output_file,
            [
                {
                    "parent_cik": filing.cik,
                    "accession_number": filing.accession_number,
                    "name": "Sub A",
                }
            ],
        )

        result = pipeline._filter_already_processed([filing])

        assert result == []

    def test_includes_new_filing_not_in_parquet(self, pipeline):
        self._write_parquet(
            pipeline.config.output_file,
            [{"parent_cik": "OTHER_CIK", "accession_number": "OTHER_ACC", "name": "Sub A"}],
        )
        filing = self._make_filing()

        result = pipeline._filter_already_processed([filing])

        assert filing in result

    def test_skips_filing_in_failure_registry(self, pipeline):
        filing = self._make_filing()
        self._write_parquet(
            pipeline.config.output_file,
            [{"parent_cik": "OTHER", "accession_number": "OTHER", "name": "X"}],
        )
        pipeline.failure_registry._entries.add((filing.cik, filing.filename))

        result = pipeline._filter_already_processed([filing])

        assert filing not in result

    def test_mixed_already_processed_and_new(self, pipeline):
        """Already-processed filing skipped, new included, failure-registry excluded."""
        processed_filing = self._make_filing(accession="ACC-PROCESSED")
        new_filing = self._make_filing(accession="ACC-NEW")
        failed_filing = self._make_filing(
            cik="0000111111", accession="ACC-FAILED", filename="CIK0000111111.json"
        )

        self._write_parquet(
            pipeline.config.output_file,
            [{"parent_cik": "0000320193", "accession_number": "ACC-PROCESSED", "name": "Sub A"}],
        )
        pipeline.failure_registry._entries.add(("0000111111", "CIK0000111111.json"))

        result = pipeline._filter_already_processed([processed_filing, new_filing, failed_filing])

        assert processed_filing not in result
        assert new_filing in result
        assert failed_filing not in result


# ── run (early exit) ──────────────────────────────────────────────────────────


class TestRunEarlyExit:
    """Tests for Pipeline.run() early-exit when load_input returns nothing."""

    def test_skips_process_when_nothing_to_process(self, pipeline, mocker):
        mocker.patch.object(pipeline, "load_input", return_value=[])
        mock_process = mocker.patch.object(pipeline, "process")

        pipeline.run()

        mock_process.assert_not_called()

    def test_skips_save_output_when_nothing_to_process(self, pipeline, mocker):
        mocker.patch.object(pipeline, "load_input", return_value=[])
        mock_save = mocker.patch.object(pipeline, "save_output")

        pipeline.run()

        mock_save.assert_not_called()

    def test_flushes_failure_registry_on_early_exit(self, pipeline, mocker):
        mocker.patch.object(pipeline, "load_input", return_value=[])
        mock_flush = mocker.patch.object(pipeline.failure_registry, "flush")

        pipeline.run()

        mock_flush.assert_called_once()
