"""Tests for processor.extractor — GptExtractor."""

from idi_corporate_structure.processor.extractor import GptExtractor
from idi_corporate_structure.processor.types import Subsidiary
from tests.conftest import make_exhibit_response


class TestGptExtractor:
    """Tests for GptExtractor.extract() stub behavior."""

    def test_returns_empty_list_for_no_documents(self, sample_filing):
        extractor = GptExtractor()
        assert extractor.extract(sample_filing, []) == []

    def test_returns_one_subsidiary_per_document(self, sample_filing):
        extractor = GptExtractor()
        result = extractor.extract(
            sample_filing, [make_exhibit_response(), make_exhibit_response()]
        )
        assert len(result) == 2
        assert all(isinstance(s, Subsidiary) for s in result)

    def test_subsidiary_fields_from_filing(self, sample_filing):
        extractor = GptExtractor()
        result = extractor.extract(sample_filing, [make_exhibit_response()])
        s = result[0]
        assert s.parent_cik == sample_filing.cik
        assert s.filing_date == sample_filing.filing_date
        assert s.form_type == sample_filing.form_type
        assert s.accession_number == sample_filing.accession_number

    def test_subsidiary_exhibit_url_from_document(self, sample_filing):
        extractor = GptExtractor()
        document = make_exhibit_response()
        result = extractor.extract(sample_filing, [document])
        assert result[0].exhibit_url == document["url"]

    def test_subsidiary_name_and_location_are_empty(self, sample_filing):
        extractor = GptExtractor()
        result = extractor.extract(sample_filing, [make_exhibit_response()])
        assert result[0].name == ""
        assert result[0].location == ""
