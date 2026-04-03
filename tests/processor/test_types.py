"""Tests for processor.types — dataclasses and PipelineStats thread safety."""

import threading
import zipfile

import pytest

from idi_corporate_structure.processor.types import (
    Filing,
    PipelineConfig,
    PipelineStats,
    Subsidiary,
)


class TestPipelineStats:
    """Tests for PipelineStats thread-safe counters."""

    def test_increment_default_by_one(self):
        stats = PipelineStats()
        stats.increment("total_filing")
        assert stats.total_filing == 1

    def test_increment_by_n(self):
        stats = PipelineStats()
        stats.increment("total_subsidiaries", 5)
        assert stats.total_subsidiaries == 5

    def test_increment_multiple_fields(self):
        stats = PipelineStats()
        stats.increment("total_filing")
        stats.increment("failed_filings", 3)
        stats.increment("failed_subsidiaries", 2)

        assert stats.total_filing == 1
        assert stats.failed_filings == 3
        assert stats.failed_subsidiaries == 2

    def test_thread_safe_concurrent_increments(self):
        """Many threads incrementing the same field should produce an exact count."""
        stats = PipelineStats()
        n_threads = 20
        increments_per_thread = 100

        def increment_many() -> None:
            for _ in range(increments_per_thread):
                stats.increment("total_filing")

        threads = [threading.Thread(target=increment_many) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.total_filing == n_threads * increments_per_thread

    def test_starts_at_zero(self):
        stats = PipelineStats()
        assert stats.total_filing == 0
        assert stats.failed_filings == 0
        assert stats.total_subsidiaries == 0
        assert stats.failed_subsidiaries == 0
        assert stats.skipped_filings == 0


class TestPipelineConfig:
    """Tests for PipelineConfig validation."""

    def test_valid_config_local_files(self, tmp_path):
        input_zip = tmp_path / "submissions.zip"
        with zipfile.ZipFile(input_zip, "w"):
            pass

        config = PipelineConfig(
            input_file=str(input_zip),
            failure_file=str(tmp_path / "failures.json"),
        )

        assert config.input_file == str(input_zip)
        assert config.num_workers == 10  # default
        assert config.rate_limit == 0.1  # default

    def test_raises_when_input_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            PipelineConfig(
                input_file=str(tmp_path / "nonexistent.zip"),
                failure_file=str(tmp_path / "failures.json"),
            )

    def test_raises_when_failure_dir_missing(self, tmp_path):
        input_zip = tmp_path / "submissions.zip"
        with zipfile.ZipFile(input_zip, "w"):
            pass

        with pytest.raises(FileNotFoundError, match="Failure file directory does not exist"):
            PipelineConfig(
                input_file=str(input_zip),
                failure_file=str(tmp_path / "nonexistent_dir" / "failures.json"),
            )

    def test_skips_validation_for_s3_paths(self):
        """S3 paths should not trigger local file existence checks."""
        config = PipelineConfig(
            input_file="s3://my-bucket/submissions.zip",
            failure_file="s3://my-bucket/failures.json",
        )
        assert config.input_file == "s3://my-bucket/submissions.zip"

    def test_custom_num_workers(self, tmp_path):
        input_zip = tmp_path / "submissions.zip"
        with zipfile.ZipFile(input_zip, "w"):
            pass

        config = PipelineConfig(
            input_file=str(input_zip),
            failure_file=str(tmp_path / "failures.json"),
            num_workers=4,
        )
        assert config.num_workers == 4


class TestFilingDataclass:
    """Tests for the Filing dataclass."""

    def test_filing_fields(self, sample_filing):
        assert sample_filing.cik == "0000320193"
        assert sample_filing.form_type == "10-K"
        assert "index.json" in sample_filing.directory
        assert sample_filing.company_name == "APPLE INC"
        assert sample_filing.location == "CA"

    def test_filing_equality(self):
        f1 = Filing(
            cik="001",
            filing_date="2024-01-01",
            form_type="10-K",
            accession_number="001-24-000001",
            directory="https://example.com/index.json",
            primary_document="",
        )
        f2 = Filing(
            cik="001",
            filing_date="2024-01-01",
            form_type="10-K",
            accession_number="001-24-000001",
            directory="https://example.com/index.json",
            primary_document="",
        )
        assert f1 == f2


class TestSubsidiaryDataclass:
    """Tests for the Subsidiary dataclass."""

    def test_subsidiary_fields(self):
        sub = Subsidiary(
            parent_cik="0000320193",
            name="Apple Operations International",
            location="Ireland",
            filing_date="2024-09-28",
            form_type="10-K",
            accession_number="0000320193-24-000123",
            exhibit_url="https://www.sec.gov/Archives/edgar/data/320193/ex21.htm",
        )
        assert sub.parent_cik == "0000320193"
        assert sub.name == "Apple Operations International"
        assert sub.location == "Ireland"
