"""Tests for processor.extractor — GptExtractor."""

import json

import pytest

from idi_corporate_structure.processor.extractor import GptExtractor
from idi_corporate_structure.processor.types import Subsidiary
from tests.conftest import make_exhibit_response


def _make_openai_response(subsidiaries: list[dict]) -> dict:
    """Build a fake OpenAI chat completions response dict."""
    return {
        "status_code": 200,
        "url": "https://api.openai.com/v1/chat/completions",
        "data": {"choices": [{"message": {"content": json.dumps({"subsidiaries": subsidiaries})}}]},
    }


class TestGptExtractor:
    """Tests for GptExtractor.extract()."""

    def test_returns_subsidiaries_from_gpt_response(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {"name": "Apple Operations LLC", "in": "Delaware"},
                    {"name": "Apple Europe Ltd", "in": "Ireland"},
                ]
            ),
        )
        result = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 2
        assert all(isinstance(s, Subsidiary) for s in result)
        assert result[0].name == "Apple Operations LLC"
        assert result[0].location == "Delaware"
        assert result[1].name == "Apple Europe Ltd"
        assert result[1].location == "Ireland"

    def test_subsidiary_filing_fields_preserved(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response([{"name": "Sub LLC", "in": "Delaware"}]),
        )
        result = extractor.extract(sample_filing, make_exhibit_response())
        s = result[0]

        assert s.parent_cik == sample_filing.cik
        assert s.parent_name == sample_filing.company_name
        assert s.filing_date == sample_filing.filing_date
        assert s.form_type == sample_filing.form_type
        assert s.accession_number == sample_filing.accession_number

    def test_exhibit_url_from_document(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response([{"name": "Sub LLC", "in": "Delaware"}]),
        )
        document = make_exhibit_response()
        result = extractor.extract(sample_filing, document)

        assert result[0].exhibit_url == document["url"]

    def test_null_location_becomes_empty_string(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response([{"name": "Sub LLC", "in": None}]),
        )
        result = extractor.extract(sample_filing, make_exhibit_response())

        assert result[0].location == ""

    def test_returns_empty_list_for_no_subsidiaries(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response([]),
        )
        result = extractor.extract(sample_filing, make_exhibit_response())

        assert result == []

    def test_raises_on_api_error(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value={"error": "connection timeout"},
        )

        with pytest.raises(RuntimeError):
            extractor.extract(sample_filing, make_exhibit_response())
