"""Tests for processor.extractor — GptExtractor."""

import json
from pathlib import Path

import pytest

from idi_corporate_structure.processor.extractor import (
    ExtractionTimeoutError,
    ExtractionTruncatedError,
    GptExtractor,
    html_to_text,
)
from idi_corporate_structure.processor.types import Subsidiary
from tests.conftest import make_exhibit_response

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_JNJ_FIXTURE = _FIXTURES_DIR / "jnj_ex21_subsidiaries.htm"

# Convenience quote constants matching make_exhibit_response() default content.
_QUOTE_APPLE_OPS = "Apple Operations LLC (Delaware)"
_QUOTE_APPLE_EU = "Apple Europe Ltd (Ireland)"


def _make_openai_response(subsidiaries: list[dict], finish_reason: str = "stop") -> dict:
    """Build a fake OpenAI chat completions response dict."""
    return {
        "status_code": 200,
        "url": "https://api.openai.com/v1/chat/completions",
        "data": {
            "choices": [
                {
                    "message": {"content": json.dumps({"subsidiaries": subsidiaries})},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        },
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

    # ── Truncation tests ──────────────────────────────────────────────────────

    def test_raises_truncated_error_on_finish_reason_length(self, sample_filing, mocker):
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
                ],
                finish_reason="length",
            ),
        )

        with pytest.raises(ExtractionTruncatedError):
            extractor.extract(sample_filing, make_exhibit_response())

    def test_finish_reason_stop_parses_normally(self, sample_filing, mocker):
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
                ],
                finish_reason="stop",
            ),
        )
        result, *_ = extractor.extract(sample_filing, make_exhibit_response())

        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_request_payload_sets_max_completion_tokens(self):
        extractor = GptExtractor(openai_api_key="fake-key")
        payload = extractor._get_request_data_json("some document text")

        assert "max_completion_tokens" in payload
        assert payload["max_completion_tokens"] == GptExtractor._MAX_COMPLETION_TOKENS

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
        from idi_corporate_structure.processor.extractor import _is_name_in_document

        doc = "Apple  Operations   LLC\n(Delaware)"
        assert _is_name_in_document("Apple Operations LLC", doc)

    def test_is_name_in_document_rejects_missing_name(self):
        """_is_name_in_document returns False when the name is absent."""
        from idi_corporate_structure.processor.extractor import _is_name_in_document

        assert not _is_name_in_document("Ghost Corp", "Apple Operations LLC (Delaware)")


class TestChunkDocument:
    """Tests for the _chunk_document splitter."""

    def test_small_doc_returns_single_chunk(self):
        from idi_corporate_structure.processor.extractor import _chunk_document

        text = "small doc"
        chunks = _chunk_document(text, max_chars=1000, overlap_chars=100)

        assert chunks == [text]

    def test_large_doc_returns_multiple_chunks(self):
        from idi_corporate_structure.processor.extractor import _chunk_document

        paragraphs = [f"para{i} " + "x" * 100 for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = _chunk_document(text, max_chars=500, overlap_chars=50)

        assert len(chunks) > 1
        # Every chunk respects the size limit (allowing for overlap headroom)
        for c in chunks:
            assert len(c) <= 500 + 50

    def test_chunks_split_on_paragraph_boundaries(self):
        from idi_corporate_structure.processor.extractor import _chunk_document

        paragraphs = [f"Paragraph {i}" for i in range(50)]
        text = "\n\n".join(paragraphs)
        chunks = _chunk_document(text, max_chars=80, overlap_chars=0)

        # Every chunk should consist of complete paragraphs joined by \n\n
        for chunk in chunks:
            for line in chunk.split("\n\n"):
                stripped = line.strip()
                # Each line is either a "Paragraph N" entry or empty
                assert stripped == "" or stripped.startswith("Paragraph ")

    def test_chunks_have_overlap(self):
        from idi_corporate_structure.processor.extractor import _chunk_document

        paragraphs = [f"para{i}_" + "x" * 100 for i in range(15)]
        text = "\n\n".join(paragraphs)
        chunks = _chunk_document(text, max_chars=400, overlap_chars=50)

        assert len(chunks) > 1
        # Each non-first chunk's start should match the previous chunk's tail
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-50:]
            assert chunks[i].startswith(tail)

    def test_oversized_paragraph_emitted_as_own_chunk(self):
        from idi_corporate_structure.processor.extractor import _chunk_document

        text = "small\n\n" + "x" * 5000 + "\n\nsmall"
        chunks = _chunk_document(text, max_chars=1000, overlap_chars=100)

        # The oversized middle paragraph must appear as its own chunk
        assert any(len(c) > 1000 for c in chunks)


class TestExtractWithChunking:
    """Tests for chunked extraction path in GptExtractor.extract()."""

    def test_no_chunking_for_small_doc(self, sample_filing, mocker):
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
        result, _, _, num_chunks = extractor.extract(sample_filing, make_exhibit_response())

        assert num_chunks == 1
        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_preemptive_chunking_for_large_doc(self, sample_filing, mocker):
        """A document over the threshold is chunked preemptively (before any LLM call)."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor,
            "_summarize",
            return_value={
                "subsidiaries": [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            },
        )

        # Document well over the 4_000-char threshold, with paragraph breaks
        big_content = "Apple Operations LLC (Delaware)\n\n" + ("filler line\n\n" * 500)
        doc = make_exhibit_response(content=big_content)
        result, _, _, num_chunks = extractor.extract(sample_filing, doc)

        assert num_chunks > 1
        # Even though every chunk returned the same sub, dedup keeps it once
        assert len(result) == 1
        assert result[0].name == "Apple Operations LLC"

    def test_chunking_triggered_on_truncation(self, sample_filing, mocker):
        """A doc under the preemptive threshold but that one-shot-truncates falls back to chunking."""
        # Push the preemptive threshold high so this doc skips preemptive chunking,
        # but keep the per-chunk max smaller so the retry path produces multiple chunks.
        mocker.patch("idi_corporate_structure.processor.extractor._CHUNK_THRESHOLD_CHARS", 100_000)
        mocker.patch("idi_corporate_structure.processor.extractor._CHUNK_MAX_CHARS", 2_000)

        extractor = GptExtractor(openai_api_key="fake-key")
        call_log = []

        def fake_summarize(doc):
            call_log.append(len(doc))
            # First call (one-shot on full doc) raises truncated; chunked calls succeed.
            if len(call_log) == 1:
                raise ExtractionTruncatedError("test truncation")
            return {
                "subsidiaries": [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    }
                ]
            }

        mocker.patch.object(extractor, "_summarize", side_effect=fake_summarize)

        # ~6K chars — under the patched 100K threshold but over the patched 2K chunk max.
        content = "Apple Operations LLC (Delaware)\n\n" + ("padding paragraph\n\n" * 400)
        doc = make_exhibit_response(content=content)
        result, _, _, num_chunks = extractor.extract(sample_filing, doc)

        assert len(call_log) >= 2  # one-shot + at least one chunked call
        assert num_chunks > 1
        assert any(s.name == "Apple Operations LLC" for s in result)

    def test_chunked_truncation_bubbles_up(self, sample_filing, mocker):
        """If a chunk itself truncates, the error bubbles up — chunk size needs tuning."""
        extractor = GptExtractor(openai_api_key="fake-key")
        mocker.patch.object(
            extractor,
            "_summarize",
            side_effect=ExtractionTruncatedError("chunk too big"),
        )

        big_content = "x" * 10_000
        doc = make_exhibit_response(content=big_content)

        with pytest.raises(ExtractionTruncatedError):
            extractor.extract(sample_filing, doc)

    def test_chunked_dedupes_same_name_across_chunks(self, sample_filing, mocker):
        """Subsidiaries appearing in multiple chunks (via overlap) are deduped by name."""
        extractor = GptExtractor(openai_api_key="fake-key")
        # Every chunk returns the same subsidiary
        mocker.patch.object(
            extractor,
            "_summarize",
            return_value={
                "subsidiaries": [
                    {
                        "name": "Apple Operations LLC",
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    },
                    {
                        "name": "APPLE OPERATIONS LLC",  # different case → still same after _normalize
                        "location": "Delaware",
                        "source_quote": _QUOTE_APPLE_OPS,
                    },
                ]
            },
        )

        big_content = "Apple Operations LLC (Delaware)\n\n" + ("filler\n\n" * 500)
        doc = make_exhibit_response(content=big_content)
        result, _, _, num_chunks = extractor.extract(sample_filing, doc)

        assert num_chunks > 1
        assert len([s for s in result if s.name.lower() == "apple operations llc"]) == 1


class TestJnjChunkingFixture:
    """End-to-end chunking test against the real Johnson & Johnson Exhibit 21.

    The fixture (``tests/processor/fixtures/jnj_ex21_subsidiaries.htm``) is the
    same exhibit that empirically caused gpt-4.1-nano to give up at ~70 of 410
    subsidiaries in production. This test mocks the OpenAI call so no API
    request is made; it verifies that:
      * the exhibit triggers preemptive chunking (>1 chunk),
      * one ``_summarize`` call is made per chunk,
      * subsidiaries returned by every chunk make it into the final result,
      * dedup collapses overlap-region duplicates.
    """

    def _load_jnj_document(self) -> dict:
        """Build a document dict from the J&J fixture (HTML → plain text)."""
        raw_html = _JNJ_FIXTURE.read_text(encoding="utf-8")
        return {
            "url": "https://www.sec.gov/Archives/edgar/data/200406/000020040625000038/ex21-subsidiariesxform10xk.htm",
            "data": html_to_text(raw_html),
        }

    def test_jnj_exhibit_is_chunked(self, sample_filing, mocker):
        """The J&J exhibit is well over the chunking threshold and should be split."""
        extractor = GptExtractor(openai_api_key="fake-key")
        # No-op model: returns nothing per chunk. We only care about the chunk count here.
        mocker.patch.object(extractor, "_summarize", return_value={"subsidiaries": []})

        _, _, _, num_chunks = extractor.extract(sample_filing, self._load_jnj_document())

        assert num_chunks > 1, "J&J exhibit should trigger chunking"

    def test_jnj_chunking_calls_summarize_once_per_chunk(self, sample_filing, mocker):
        extractor = GptExtractor(openai_api_key="fake-key")
        spy_summarize = mocker.patch.object(
            extractor, "_summarize", return_value={"subsidiaries": []}
        )

        _, _, _, num_chunks = extractor.extract(sample_filing, self._load_jnj_document())

        assert spy_summarize.call_count == num_chunks

    def test_jnj_chunking_makes_no_real_api_call(self, sample_filing, mocker):
        """Mocking _summarize means the underlying HTTP client is never touched."""
        extractor = GptExtractor(openai_api_key="fake-key")
        spy_query = mocker.spy(extractor._openai_client, "query_endpoint")
        mocker.patch.object(extractor, "_summarize", return_value={"subsidiaries": []})

        extractor.extract(sample_filing, self._load_jnj_document())

        assert spy_query.call_count == 0

    def test_jnj_chunking_aggregates_results_across_chunks(self, sample_filing, mocker):
        """Each chunk returns its own subsidiary; merged result includes all."""
        # Real subsidiary names from the J&J Exhibit 21 — chosen so they ground.
        names_per_chunk = [
            "ABD Holding Company, Inc.",
            "DePuy Synthes, Inc.",
            "Janssen Biotech, Inc.",
            "Mentor Worldwide LLC",
            "Synthes USA, LLC",
        ]
        extractor = GptExtractor(openai_api_key="fake-key")

        call_idx = {"i": 0}

        def fake_summarize(_chunk: str) -> dict:
            i = call_idx["i"]
            call_idx["i"] += 1
            if i < len(names_per_chunk):
                return {
                    "subsidiaries": [
                        {
                            "name": names_per_chunk[i],
                            "location": "Delaware",
                            "source_quote": names_per_chunk[i],
                        }
                    ]
                }
            return {"subsidiaries": []}

        mocker.patch.object(extractor, "_summarize", side_effect=fake_summarize)

        result, ungrounded_name, _, num_chunks = extractor.extract(
            sample_filing, self._load_jnj_document()
        )

        assert num_chunks >= len(names_per_chunk)
        assert ungrounded_name == 0, "All sample names should ground against the real exhibit"
        result_names = {s.name for s in result}
        for name in names_per_chunk:
            assert name in result_names

    def test_jnj_chunking_dedupes_overlap_duplicates(self, sample_filing, mocker):
        """A subsidiary returned by every chunk (overlap simulation) appears once."""
        extractor = GptExtractor(openai_api_key="fake-key")
        # Same sub returned by every call — dedup-by-name should collapse to one row.
        mocker.patch.object(
            extractor,
            "_summarize",
            return_value={
                "subsidiaries": [
                    {
                        "name": "ABD Holding Company, Inc.",
                        "location": "Delaware",
                        "source_quote": "ABD Holding Company, Inc.",
                    }
                ]
            },
        )

        result, _, _, num_chunks = extractor.extract(sample_filing, self._load_jnj_document())

        assert num_chunks > 1
        assert len([s for s in result if s.name == "ABD Holding Company, Inc."]) == 1
