# IDI Corporate Structure Pipeline

> [DRAFT in progress]

Automated pipeline for extracting subsidiary information from SEC 10-K filings (Exhibit 21) and building hierarchical corporate structure trees.

## Pipeline Overview

Each run performs three stages:

1. **Collection** — parse `submissions.zip` from SEC EDGAR bulk data; extract all 10-K filing metadata (CIK, accession number, filing date, exhibit URLs)
2. **Retrieval** — fetch each filing's directory index from SEC EDGAR; locate and download Exhibit 21 (Subsidiaries of the Registrant)
3. **Extraction** — pass exhibit content to GPT-4 to parse subsidiary names and incorporation locations; output structured `Subsidiary` records

Processing tracks permanent failures to disk so interrupted runs do not re-attempt filings that will always fail.

### Output Layout

```
{output_dir}/
  subsidiaries.csv     parent_cik, name, location, filing_date, form_type, accession_number, exhibit_url # This is TBD
  failures/
    failures.json      permanent failures keyed by (cik, accession_number)
```

Paths support local directories or S3 URLs (`s3://bucket/path`).

---

## Quick Start

### Installation

```bash
uv sync              # Production
uv sync --all-groups # Development (includes tests)
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
uv run python -m src.idi_corporate_structure.processor.pipeline
```

Configure the run by editing the `PipelineConfig` block at the bottom of `pipeline.py`:

```python
config = PipelineConfig(
    input_file  = "path/to/submissions.zip",   # local path or s3:// or https://
    failure_file= "path/to/failures.json",
    rate_limit  = 0.12,                        # seconds between SEC requests (~8 req/s)
    num_workers = 10,                          # concurrent GPT extraction threads
)
```

The `submissions.zip` bulk data file is available from SEC EDGAR:

```
https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip
```

### Configuration Reference

| Field | Default | Description |
|---|---|---|
| `input_file` | — | Required. Path to `submissions.zip` (local, `s3://`, or `https://`) |
| `failure_file` | — | Required. Path to failures JSON; parent directory must exist |
| `failure_flush_every` | `50` | Write failures to disk after every N new entries |
| `rate_limit` | `0.1` | Seconds between SEC HTTP requests (SEC limit: 10 req/s) |
| `num_workers` | `10` | Number of concurrent GPT extraction worker threads |

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

**Producer-consumer design**: The SEC fetcher runs serially on the main thread (respecting EDGAR's 10 req/s rate limit). Exhibit content is pushed onto a bounded `queue.Queue(maxsize=num_workers * 2)`, which blocks the producer if workers fall behind. GPT extraction runs concurrently across `num_workers` daemon threads, draining the queue as fast as OpenAI responds. The two stages overlap — SEC fetching continues while earlier exhibits are being summarized.

**Resumability**: `FailureRegistry` persists non-retryable failures to disk after every `failure_flush_every` entries. On re-run, filings whose failures are classified as permanent (e.g. `no_10k_filings`, `no_exhibit_content`) are skipped without making network requests.

**Common modules** (`src/idi_corporate_structure/common/`):

| Module | Purpose |
|---|---|
| `api.py` | SEC, LSEG, and Geonames API clients with retry logic |
| `failures.py` | Permanent-failure registry (do-not-retry) |
| `logs.py` | Structured logging with optional CloudWatch integration |
| `storage.py` | JSON load/save (local + S3 + HTTPS) |

### Failure Types

| Type | Retryable | Description |
|---|---|---|
| `mismatched_lengths` | No | Filing arrays have unequal lengths |
| `no_form_data` | No | Filing arrays are empty |
| `no_10k_filings` | No | CIK has no 10-K forms |
| `no_filing_directory` | No | SEC returned no directory listing for the filing |
| `no_exhibit_content` | No | Exhibit URL returned no content |
| `extraction_failed` | Yes | GPT returned no subsidiary data |
| `api_error` | Yes | Transient HTTP failure |
| `rate_limit` | Yes | SEC 429 — retried with backoff |
