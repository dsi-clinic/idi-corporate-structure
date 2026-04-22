# IDI Corporate Structure Pipeline

Automated pipeline for extracting subsidiary information from SEC 10-K filings (Exhibit 21) and building hierarchical corporate structure trees.

## Pipeline Overview

Each run performs three stages:

1. **Collection** — parse `submissions.zip` from SEC EDGAR bulk data; extract all 10-K filing metadata (CIK, accession number, filing date, exhibit URLs); output `Filing` records
2. **Retrieval** — fetch each filing's directory index from SEC EDGAR; locate and download Exhibit 21 (Subsidiaries of the Registrant)
3. **Extraction** — pass exhibit content to `gpt-4.1-nano` using structured output to parse subsidiary names and incorporation locations. Each result includes a `source_quote`: a verbatim snippet from the exhibit that contains the subsidiary's name. Rows whose quote cannot be matched in the source text are dropped to reduce hallucinations. Output is structured `Subsidiary` records written to Parquet.

Processing tracks permanent failures to disk so interrupted runs do not re-attempt filings that will always fail.

### Output Layout

```
output/
  # Parquet — columns: parent_cik, name, location, source_quote, filing_date, form_type, accession_number, exhibit_url, date_added
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
| `rate_limit` | `0.2` | Seconds between SEC HTTP requests (SEC limit: 10 req/s) |
| `num_workers` | `10` | Number of concurrent GPT extraction worker threads |

---

## Container Usage

The pipeline ships with a multi-stage Dockerfile and Docker Compose files for running the orchestrator in a container.

### Files

| File | Purpose |
|---|---|
| `dockerfiles/Dockerfile.orchestrator` | Multi-stage `python:3.13-slim` image; non-root `pipeline` user |
| `compose.yml` | Service definition; pulls image from registry |
| `compose.override.yml` | Adds `build:` block for local development; merged automatically by `docker compose` |

### Environment Variables

All required variables must be set before running. Optional variables fall back to the listed defaults.

#### Required

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for GPT extraction |
| `INPUT_MOUNT_SOURCE` | Host directory containing the input file (e.g. `/data/input`) |
| `INPUT_FILE` | Container-side path to the input file (e.g. `/data/input/submissions.zip`); can be an `https://` URL instead of a local path, in which case `INPUT_MOUNT_SOURCE` is unused |
| `OUTPUT_MOUNT_SOURCE` | Host directory for Parquet output (e.g. `/data/output`) |
| `FAILURE_MOUNT_SOURCE` | Host directory for failures JSON (e.g. `/data/failures`) |
| `LOG_DIR` | Host directory for log files (e.g. `/data/logs`) |

#### Optional

| Variable | Default | Description |
|---|---|---|
| `OUTPUT_FILE` | `/data/output/output.parquet` | Container-side path for Parquet output |
| `FAILURE_FILE` | `/data/failures/failures.json` | Container-side path for failures JSON |
| `RATE_LIMIT` | `0.2` | Seconds between SEC HTTP requests |
| `NUM_WORKERS` | `10` | Number of concurrent GPT extraction worker threads |
| `INPUT_SAMPLE_SIZE` | `0` | Limit input to N files for testing (`0` = no limit) |
| `AWS_REGION` | `us-east-2` | AWS region for S3 and CloudWatch |
| `CLOUDWATCH_LOGS_ENABLED` | `false` | Enable CloudWatch log shipping |
| `ORCHESTRATOR_IMAGE` | `ghcr.io/dsi-clinic/idi-corporate-structure-orchestrator:latest` | Image to pull on EC2 (ignored when building locally) |

### Run

**Local — build from source and run:**

```bash
export OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxx"
export INPUT_MOUNT_SOURCE="/path/to/input"       # must contain submissions.zip
export OUTPUT_MOUNT_SOURCE="/path/to/output"
export FAILURE_MOUNT_SOURCE="/path/to/failures"
export LOG_DIR="/path/to/logs"

docker compose up --build orchestrator
```

`compose.override.yml` is merged automatically when running locally, which adds the `build:` block so the image is built from source rather than pulled.

**Detached (background):**

```bash
docker compose up -d --build orchestrator
docker compose logs -f orchestrator   # tail logs
docker compose down                   # stop
```

---

## AWS ECS Architecture

The pipeline runs as an **ECS Fargate task** scheduled by **EventBridge Scheduler**. Infrastructure is defined in `pulumi/` using Pulumi (Python).

### Design Decisions

| Decision | Rationale |
|---|---|
| **Fargate** (not EC2) | No instance management — container runs and exits; portable image |
| **EventBridge Scheduler** (not Step Functions) | Processors are independent; no workflow orchestration needed |
| **Public subnet** (no NAT Gateway) | Task needs outbound internet for SEC EDGAR and OpenAI |
| **`awslogs` driver only** | Captures all stdout/stderr; linked directly to the task in the ECS console; app-level CloudWatch handler disabled (`CLOUDWATCH_LOGS_ENABLED=false`) |
| **ECR** | No pull credential configuration required for Fargate |

### Resources

| Module | Resources |
|---|---|
| `config.py` | Shared name prefix (`{project}-{stack}-{app}`), tags, AWS caller identity |
| `networking.py` | Default VPC, single-AZ public subnet, egress-only security group |
| `iam.py` | Task execution role (ECR pull, CloudWatch Logs, Secrets Manager) + task role (S3, ECS Exec) |
| `ecr.py` | ECR repository + lifecycle policy (retains last 5 images) |
| `ecs.py` | ECS cluster (Fargate, Container Insights), CloudWatch log group (30-day retention), task definition (1 vCPU / 4 GB) |
| `secrets.py` | Secrets Manager secret for OpenAI API key; injected as `OPENAI_API_KEY` env var at task startup |
| `scheduling.py` | EventBridge Scheduler (cron, starts disabled), SQS dead-letter queue for failed invocations, scheduler IAM role |

### S3 File Layout

All pipeline files live in a single externally-managed S3 bucket:

```
{bucket}/
  {app}/
    input/submissions.zip       ← input (or pass an HTTPS URL via config)
    output/subsidiaries.parquet ← output
    failures/failures.json      ← permanent failure registry
```

### Deployment

```bash
cd pulumi/

# First-time setup
uv run --group pulumi pulumi stack init dev
uv run --group pulumi pulumi config set aws:region us-east-2
uv run --group pulumi pulumi config set idi:bucket_name <bucket>
uv run --group pulumi pulumi config set --secret idi:openai_api_key <key>

# Deploy
uv run --group pulumi pulumi up
```

#### Configuration Reference

| Config | Default | Description |
|---|---|---|
| `aws:region` | `us-east-2` | AWS region |
| `idi:app_name` | `corporate-structure` | Application name used in resource naming |
| `idi:bucket_name` | — | S3 bucket for input, output, and failures (created externally) |
| `idi:openai_api_key` | — | OpenAI API key (secret; stored in Secrets Manager) |
| `idi:input_file` | SEC EDGAR HTTPS URL | Input file path (S3 URI or HTTPS URL) |
| `idi:cron_corporate_structure` | `cron(0 2 * * ? *)` | EventBridge schedule expression |
| `idi:schedule_enabled` | `false` | Enable the EventBridge schedule |
| `idi:cpu` | `1024` | Fargate task CPU units |
| `idi:memory` | `4096` | Fargate task memory (MiB) |
| `idi:rate_limit` | `0.2` | Seconds between SEC API requests |
| `idi:num_workers` | `10` | GPT extraction worker threads |
| `idi:input_sample_size` | `0` | Filings to process (`0` = all; set `>0` for testing) |

### Manual Task Execution

```bash
aws ecs run-task \
    --cluster <cluster-name> \
    --task-definition <task-definition> \
    --launch-type FARGATE \
    --propagate-tags TASK_DEFINITION \
    --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}" \
    --overrides '{
        "containerOverrides": [{
            "name": "corporate-structure-orchestrator",
            "environment": [{"name": "INPUT_SAMPLE_SIZE", "value": "5"}]
        }]
    }'
```

Use `pulumi stack output` to retrieve cluster name, subnet ID, and security group ID.

### Monitoring

- **Logs**: CloudWatch → Log groups → `/ecs/{name_prefix}` → stream per task run
- **ECS console**: Tasks tab shows stopped tasks for up to 1 hour after completion
- **Scheduling failures**: Check the SQS dead-letter queue (`pulumi stack output dlq_url`)
- **ECS Exec** (interactive debug into running task):
  ```bash
  aws ecs execute-command \
    --cluster <cluster> \
    --task <task-id> \
    --container corporate-structure-orchestrator \
    --interactive \
    --command "/bin/sh"
  ```

### Building and Pushing the Container Image

```bash
# Set ECR repo URL
ECR_REPO=$(cd pulumi && uv run --group pulumi pulumi stack output ecr_repo_url)

# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-2 | \
  docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-2.amazonaws.com

# Build for linux/amd64 (required on Apple Silicon) and push
docker buildx build --platform linux/amd64 \
  -f dockerfiles/Dockerfile.orchestrator \
  -t $ECR_REPO \
  --push .
```

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
