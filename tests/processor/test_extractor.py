"""Tests for processor.extractor — GptExtractor."""

import json

import pytest

from idi_corporate_structure.processor.extractor import ExtractionTimeoutError, GptExtractor
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
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

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
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())
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
        result, *_ = extractor.extract(sample_filing, document)

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
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert result[0].location == ""
        assert result[0].source_quote == _QUOTE_APPLE_OPS

    def test_returns_empty_list_for_no_subsidiaries(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response([]),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert result == []

    def test_raises_on_api_error(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value={"error": "connection refused"},
        )

        with pytest.raises(RuntimeError):
            extractor.extract(sample_filing, make_exhibit_response())

    def test_raises_extraction_timeout_error_on_timeout(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value={"error": "Timeout querying https://api.openai.com/...", "timeout": True},
        )

        with pytest.raises(ExtractionTimeoutError):
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
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert result[0].source_quote == _QUOTE_APPLE_OPS

    def test_name_missing_from_doc_is_dropped(self, sample_filing, mocker):
        """Subsidiaries whose name does not appear in the document are dropped."""
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
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert result == []

    def test_grounded_rows_kept_ungrounded_dropped(self, sample_filing, mocker):
        """Rows whose name appears in the doc are kept; missing names are dropped."""
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
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_empty_source_quote_kept_when_name_in_doc(self, sample_filing, mocker):
        """An empty source_quote is not a drop reason when the name is grounded."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [{"name": "Apple Operations LLC", "location": "Delaware", "source_quote": ""}]
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_html_entity_name_matches_decoded_doc(self, sample_filing, mocker):
        """HTML entities in the model's name still match a decoded plain-text doc."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "&#8220;American Eagle&#8221; Holdings",
                        "location": "Delaware",
                        "source_quote": "&#8220;American Eagle&#8221; Holdings",
                    }
                ]
            ),
        )
        doc = make_exhibit_response(content='"American Eagle" Holdings, a Delaware Corporation')
        result, *_ = extractor.extract(sample_filing, doc)

        assert len(result) == 1
        assert result[0].name == "\u201cAmerican Eagle\u201d Holdings"

    def test_smart_quote_name_matches_ascii_doc(self, sample_filing, mocker):
        """A smart-quoted name in the model output matches an ASCII doc."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "O\u2019Reilly Holdings",
                        "location": "Delaware",
                        "source_quote": "O\u2019Reilly Holdings",
                    }
                ]
            ),
        )
        doc = make_exhibit_response(content="O'Reilly Holdings, a Delaware Corporation")
        result, *_ = extractor.extract(sample_filing, doc)

        assert len(result) == 1

    def test_name_surrounding_whitespace_matches_doc(self, sample_filing, mocker):
        """Normalization tolerates runs of whitespace between tokens."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": "Apple Operations LLC",
                    }
                ]
            ),
        )
        doc = make_exhibit_response(content="Apple  Operations   LLC\n(Delaware)")
        result, *_ = extractor.extract(sample_filing, doc)

        assert len(result) == 1

    def test_quote_mismatch_does_not_drop_but_logs_debug(self, sample_filing, mocker):
        """A non-matching source_quote logs DEBUG but does not drop the row."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": "fabricated quote that is not in doc",
                    }
                ]
            ),
        )
        debug_spy = mocker.spy(extractor._logger, "debug")
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert any(
            "Quote not in document" in str(call.args[0]) for call in debug_spy.call_args_list
        )

    def test_location_not_near_name_logs_debug(self, sample_filing, mocker):
        """A location absent from the window around the name logs DEBUG but keeps the row."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor._openai_client,
            "query_endpoint",
            return_value=_make_openai_response(
                [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Narnia",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            ),
        )
        debug_spy = mocker.spy(extractor._logger, "debug")
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert any(
            "Location" in str(call.args[0]) and "not near name" in str(call.args[0])
            for call in debug_spy.call_args_list
        )

    def test_location_near_name_does_not_log_debug(self, sample_filing, mocker):
        """A location present in the window around the name emits no location debug log."""
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
        debug_spy = mocker.spy(extractor._logger, "debug")
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert not any("Location" in str(call.args[0]) for call in debug_spy.call_args_list)

    def test_is_name_in_document_normalizes_whitespace(self):
        """_is_name_in_document matches despite whitespace differences."""
        extractor = GptExtractor(openai_api_key="fake-key")
        doc = "Apple  Operations   LLC\n(Delaware)"
        assert extractor._is_name_in_document("Apple Operations LLC", doc)

    def test_is_name_in_document_rejects_missing_name(self):
        """_is_name_in_document returns False when the name is absent."""
        extractor = GptExtractor(openai_api_key="fake-key")
        assert not extractor._is_name_in_document("Ghost Corp", "Apple Operations LLC (Delaware)")
