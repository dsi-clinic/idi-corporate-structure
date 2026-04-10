# IDI Corporate Structure Pipeline

Automated pipeline for extracting subsidiary information from SEC 10-K filings (Exhibit 21) and building hierarchical corporate structure trees.

## Pipeline Overview

Each run performs three stages:

1. **Collection** — parse `submissions.zip` from SEC EDGAR bulk data; extract all 10-K filing metadata (CIK, accession number, filing date, exhibit URLs); output `Filing` records
2. **Retrieval** — fetch each filing's directory index from SEC EDGAR; locate and download Exhibit 21 (Subsidiaries of the Registrant)
3. **Extraction** — pass exhibit content to GPT to parse subsidiary names and incorporation locations; output structured `Subsidiary` records

Processing tracks permanent failures to disk so interrupted runs do not re-attempt filings that will always fail.

### Output Layout

```
output/
  # Parquet — columns: parent_cik, name, location, filing_date, form_type, accession_number, exhibit_url, date_added
  {output_file}
failures/
  # permanent failures keyed by (cik, accession_number)
  failures.json
```

Paths support local directories or S3 URLs (`s3://bucket/path`).

---

## Quick Start

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
uv sync              # Production
uv sync --all-groups # Development (includes tests and linting tools)
```

### Credentials

```bash
export OPENAI_API_KEY='your-key'
```

| Credential | Source |
|---|---|
| OpenAI API key | [platform.openai.com](https://platform.openai.com/api-keys) |

AWS credentials are required only if using S3 paths for input or output.

### Run

```bash
uv run python3 -m src.idi_corporate_structure.processor.orchestrator \
    --input-file "/local/input/submissions.zip" \
    --output-file "/local/output/subsidiaries.parquet" \
    --failure-file "/local/failures/failures.json" \
    --openai-api-key "sk-proj-xxxxxxxxxxxxx" \
    --rate-limit 0.2 \
    --num-workers 10
```

- To read the `submissions.zip` file in via HTTP from SEC EDGAR, pass the following URL into the `--input-file` argument:
    - `https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip`

### Configuration Reference

| Field | Default | Description |
|---|---|---|
| `input_file` | — | Required. Path to `submissions.zip` (local, `s3://`, or `https://`) |
| `output_file` | — | Required. Path for Parquet output (local or `s3://`) |
| `failure_file` | — | Required. Path to failures JSON; parent directory created if missing |
| `failure_flush_every` | `50` | Write failures to disk after every N new entries |
| `rate_limit` | `0.1` | Seconds between SEC HTTP requests (SEC limit: 10 req/s) |
| `num_workers` | `10` | Number of concurrent GPT extraction worker threads |

---

## Data Flow

The pipeline walks the SEC EDGAR data hierarchy one level at a time, from the bulk submissions archive down to individual exhibit documents:

```
submissions.zip  (SEC EDGAR bulk data — one JSON blob per CIK)
  │
  │  parse per-CIK JSON, filter to 10-K form type
  ▼
10-K Filing  (CIK, accession number, filing date)
  │
  │  GET https://www.sec.gov/Archives/edgar/data/{CIK}/{accession_number}/index.json
  ▼
10-K Directory Index  (list of all documents in the filing)
  │
  │  locate Exhibit 21 entry by document type / filename pattern
  ▼
Exhibit 21 Document  (HTML, plain text, or PDF)
  │
  │  download and extract text (PDF → pdfplumber)
  ▼
Exhibit Content  (raw text listing subsidiaries)
  │
  │  POST to OpenAI with structured JSON schema
  ▼
GPT Extraction  (gpt-4.1-nano, structured output)
  │
  │  parse response into Subsidiary records
  ▼
Subsidiaries List  (name, location, parent CIK, accession number, …)
  │
  │  deduplicate, append metadata, write via pandas
  ▼
Parquet Output  ({output_file})
```

Each filing that cannot be processed (missing exhibit, empty content, document too long, etc.) is recorded in `failures.json` as a permanent failure and skipped on subsequent runs.

---

## Architecture

```
pipeline.py
  └── SubsidiaryPipeline.run()
        ├── load_input()       — parse submissions.zip → list[Filing]
        └── process()
              ├── Main thread (producer)
              │     for each Filing:
              │       SecClient.query()  ← fetch directory index (rate-limited)
              │       SecClient.query()  ← fetch Exhibit 21 content
              │       work_queue.put()   ← blocks if all worker slots full
              │
              ├── N extract workers (daemon threads)
              │     work_queue.get()     ← blocks until work available
              │     GptExtractor.extract() → list[Subsidiary]
              │     results_queue.put()
              │
              └── 1 results worker (daemon thread)
                    results_queue.get()  ← accumulates Subsidiary records
```

**Producer-consumer design**:

- The SEC fetcher runs serially on the main thread (respecting EDGAR's 10 req/s rate limit).
- Exhibit content is pushed onto a bounded `queue.Queue(maxsize=num_workers * 2)`, which blocks the producer if workers fall behind.
- GPT extraction runs concurrently across `num_workers` daemon threads, draining the queue as fast as OpenAI responds.
- The two stages overlap — SEC fetching continues while earlier exhibits are being summarized.

**Resumability**:

-`FailureRegistry` persists non-retryable failures to disk after every `failure_flush_every` entries. On re-run, filings whose failures are classified as permanent are skipped without making network requests.

### Modules

**Common** (`src/idi_corporate_structure/common/`):

| Module | Purpose |
|---|---|
| `api.py` | `ApiClient` base class (retries, rate limiting); `SecClient` for SEC EDGAR; `OpenAiClient` for GPT extraction |
| `failures.py` | `FailureRegistry` — persists permanent failures to JSON on disk; `FailureClassifier` abstract base |
| `logs.py` | Structured logging setup with optional CloudWatch (`watchtower`) integration |
| `storage.py` | `load_json`/`save_json` and `open_zip` supporting local, `s3://`, and `https://` paths |

**Processor** (`src/idi_corporate_structure/processor/`):

| Module | Purpose |
|---|---|
| `types.py` | `Filing`, `Subsidiary`, `PipelineConfig`, and `PipelineStats` dataclasses |
| `extractor.py` | `GptExtractor` — calls OpenAI with a structured JSON schema to parse subsidiary names and locations from exhibit text |
| `failures.py` | `FailureType` enum and `CorporateStructureFailureClassifier` (maps HTTP responses to retryable vs permanent failures) |
| `pipeline.py` | `SubsidiaryPipeline` — orchestrates collection, retrieval, and extraction; deduplicates and writes Parquet output |

### Failure Types

| Type | Retryable | Description |
|---|---|---|
| `mismatched_lengths` | No | Parallel filing arrays have unequal lengths |
| `no_form_data` | No | Filing arrays are empty |
| `no_10k_filings` | No | CIK has no 10-K forms |
| `no_overflow_filings` | No | CIK has no overflow filing entries |
| `no_filing_directory` | No | SEC returned no directory listing for the filing |
| `no_exhibit_content` | No | Exhibit URL returned no content |
| `document_error` | No | Exhibit document is too long to process |
| `extraction_failed` | Yes | GPT returned no subsidiary data |
| `api_error` | Yes | Transient HTTP failure |
| `rate_limit` | Yes | SEC 429 — retried with backoff |

---

## Development & Contributing

Install all dependency groups (includes `dev` tools: pytest, ruff):

```bash
uv sync --all-groups
```

### Tests

```bash
uv run pytest
```

### Linting & Formatting

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
```

### Code Style

| Rule | Value |
|---|---|
| Line length | 100 characters |
| Docstring convention | Google (`pydocstyle`) |
| Type annotations | Required on all public functions and classes |
| String quotes | Double-quoted (ruff `Q` ruleset) |
