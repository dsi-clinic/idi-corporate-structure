"""Tests for processor.extractor — GptExtractor."""

import json

import pytest

from idi_corporate_structure.processor.extractor import GptExtractor
from idi_corporate_structure.processor.types import Subsidiary
from tests.conftest import make_exhibit_response

# Convenience quote constants matching make_exhibit_response() default content.
_QUOTE_APPLE_OPS = "Apple Operations LLC (Delaware)"
_QUOTE_APPLE_EU = "Apple Europe Ltd (Ireland)"


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
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    },
                    {
                        "name": "Apple Europe Ltd",
                        "location": "Ireland",
                        "source_quote": _QUOTE_APPLE_EU,
                    },
                ]
            ),
        )
        result = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 2
        assert all(isinstance(s, Subsidiary) for s in result)
        assert result[0].name == "Apple Operations LLC"
        assert result[0].location == "Delaware"
        assert result[0].source_quote == _QUOTE_APPLE_OPS
        assert result[1].name == "Apple Europe Ltd"
        assert result[1].location == "Ireland"
        assert result[1].source_quote == _QUOTE_APPLE_EU

    def test_subsidiary_filing_fields_preserved(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
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
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        document = make_exhibit_response()
        result = extractor.extract(sample_filing, document)

        assert result[0].exhibit_url == document["url"]

    def test_null_location_becomes_empty_string(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": None,
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        result = extractor.extract(sample_filing, make_exhibit_response())

        assert result[0].location == ""
        assert result[0].source_quote == _QUOTE_APPLE_OPS

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

    # ── Grounding tests ───────────────────────────────────────────────────────

    def test_source_quote_preserved(self, sample_filing, mocker):
        """source_quote from GPT response is mapped onto the Subsidiary."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        result = extractor.extract(sample_filing, make_exhibit_response())

        assert result[0].source_quote == _QUOTE_APPLE_OPS

    def test_ungrounded_quote_is_dropped(self, sample_filing, mocker):
        """Subsidiaries whose source_quote doesn't appear in the exhibit are dropped."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Made Up Corp",
                        "location": "Delaware",
                        "source_quote": "Made Up Corp (Delaware)",
                    }
                ]
            ),
        )
        result = extractor.extract(sample_filing, make_exhibit_response())

        assert result == []

    def test_grounded_rows_kept_ungrounded_dropped(self, sample_filing, mocker):
        """Only grounded rows survive when some quotes are fabricated."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    },
                    {
                        "name": "Ghost Corp",
                        "location": "Nevada",
                        "source_quote": "Ghost Corp (Nevada)",
                    },
                ]
            ),
        )
        result = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_empty_source_quote_is_dropped(self, sample_filing, mocker):
        """An empty source_quote is treated as ungrounded and the row is dropped."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [{"name": "Apple Operations LLC", "location": "Delaware", "source_quote": ""}]
            ),
        )
        result = extractor.extract(sample_filing, make_exhibit_response())

        assert result == []

    def test_is_grounded_normalizes_whitespace(self):
        """_is_grounded matches despite minor whitespace differences from PDF extraction."""
        extractor = GptExtractor(openai_api_key="fake-key")
        doc = "Apple  Operations   LLC\n(Delaware)"
        assert extractor._is_grounded("Apple Operations LLC (Delaware)", doc)

    def test_is_grounded_rejects_fabricated_quote(self):
        """_is_grounded returns False when the quote does not appear in the document."""
        extractor = GptExtractor(openai_api_key="fake-key")
        assert not extractor._is_grounded("Ghost Corp", "Apple Operations LLC (Delaware)")
