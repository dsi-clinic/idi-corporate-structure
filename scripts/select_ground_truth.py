r"""Emit a CSV of ground-truth candidates from a curated submissions zip.

Scans a year-filtered submissions zip (see ``curate_submissions.py``) and
surfaces **10-K and 20-F** filings (including ``/A`` amendments) across
several buckets that together stress the pipeline's subsidiary-extraction
paths:

  - known_giant       : CIKs hardcoded in scripts/verify_output.py
  - large_filer_seed  : hand-seeded well-known large 10-K filers
  - amendment         : any 10-K/A or 20-F/A
  - foreign_incorp    : non-US stateOfIncorporation (captures most 20-F
                        foreign private issuers)
  - small_filer       : company whose total recent.form list is short
                        (proxy for infrequent filers; stresses the
                        zero-subsidiary / single-subsidiary edge case)

One row per (company, filing). A single filing can match multiple buckets
— the ``buckets`` column is a comma-separated list.

Each row includes a ``directory_url`` pointing at the filing's EDGAR
index (e.g. ``.../index.json``). Actual exhibit filenames aren't
deterministic — they vary per filing — so this script does not try to
resolve the exact exhibit URL. One click from the directory URL gets a
human reviewer to the exhibit.

Usage:
    uv run python scripts/select_ground_truth.py \\
        --input submissions_2025.zip \\
        --output ground_truth_candidates.csv
"""

import argparse
import csv
import json
import logging
import pathlib
import re
import sys
import zipfile
from collections import Counter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mirrored from src/idi_corporate_structure/processor/pipeline.py so this
# script has no runtime dependency on the processor package.
# ---------------------------------------------------------------------------
_IS_PRIMARY_CIK = re.compile(r"^CIK\d+\.json$")
_IS_10K = re.compile(r"10-?K")
_IS_20F = re.compile(r"20-?F")
_SEC_URL = "https://www.sec.gov/Archives/edgar/data"

# From scripts/verify_output.py lines 44-48 — pipeline's smoke-test anchors.
_KNOWN_GIANT_CIKS: set[str] = {"320193", "789019", "97476"}

# Hand-seeded well-known large US filers likely to have many subsidiaries.
# Documented explicitly so the seed list is auditable / editable.
_LARGE_FILER_SEEDS: set[str] = {
    "1067983",  # Berkshire Hathaway
    "1652044",  # Alphabet (Google)
    "19617",  # JPMorgan Chase
    "40545",  # General Electric
    "34088",  # Exxon Mobil
    "1018724",  # Amazon
    "104169",  # Walmart
    "200406",  # Johnson & Johnson
    "80424",  # Procter & Gamble
    "78003",  # Pfizer
    "732717",  # AT&T
    "732712",  # Verizon (approx.)
    "831001",  # Citigroup
    "886982",  # Goldman Sachs
    "895421",  # Morgan Stanley
    "37996",  # Ford
    "1467858",  # General Motors
    "886158",  # Schlumberger
    "21344",  # Coca-Cola
    "1090872",  # Arconic / older industrials
}

# Common US-state / DC codes. Anything outside this set counts as "foreign".
_US_STATES: set[str] = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
    "PR",
}

_SMALL_FILER_THRESHOLD = 20  # companies with <=N total recent filings


def _normalize_cik(raw: str | int) -> str:
    """Strip leading zeros — matches how _KNOWN_CIKS is keyed in verify_output.py."""
    return str(raw).lstrip("0") or "0"


def _is_target_form(form: str) -> bool:
    """Keep only 10-K, 10-K/A, 20-F, 20-F/A (same test the pipeline applies)."""
    return bool(_IS_10K.match(form) or _IS_20F.match(form))


def _exhibit_type(form: str) -> str:
    """Return the exhibit number — 8 for 20-F, 21 for 10-K. Mirrors Filing.exhibit_type."""
    return "8" if _IS_20F.match(form) else "21"


def _buckets_for(
    cik_norm: str,
    form: str,
    state: str,
    total_recent: int,
) -> list[str]:
    """Return the list of bucket tags that match this filing."""
    buckets: list[str] = []
    if cik_norm in _KNOWN_GIANT_CIKS:
        buckets.append("known_giant")
    if cik_norm in _LARGE_FILER_SEEDS:
        buckets.append("large_filer_seed")
    if form.endswith("/A"):
        buckets.append("amendment")
    if state and state.upper() not in _US_STATES:
        buckets.append("foreign_incorp")
    if total_recent <= _SMALL_FILER_THRESHOLD:
        buckets.append("small_filer")
    return buckets


def _filing_index_url(cik: str, accession_number: str) -> str:
    """Return the human-readable EDGAR filing index page URL (deterministic, no network).

    The HTML index lists every document in the filing with its type (EX-21,
    EX-8, 10-K, etc.) and a direct link, making it easy to locate the
    subsidiary exhibit by eye.
    """
    accession_no_dashes = accession_number.replace("-", "")
    return f"{_SEC_URL}/{cik}/{accession_no_dashes}/{accession_number}-index.htm"


def select(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Scan the curated zip and write a CSV of candidate filings."""
    rows: list[dict] = []

    with zipfile.ZipFile(input_path, "r") as zf:
        for name in zf.namelist():
            if not _IS_PRIMARY_CIK.match(name):
                continue

            try:
                with zf.open(name) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue

            cik_raw = str(data.get("cik", ""))
            cik_norm = _normalize_cik(cik_raw)
            company_name = data.get("name", "")
            state = data.get("stateOfIncorporation", "") or ""

            recent = data.get("filings", {}).get("recent", {}) or {}
            forms = recent.get("form", []) or []
            accs = recent.get("accessionNumber", []) or []
            docs = recent.get("primaryDocument", []) or []
            dates = recent.get("filingDate", []) or []
            total_recent = len(forms)

            for form, acc, doc, date in zip(forms, accs, docs, dates):
                if not _is_target_form(form):
                    continue
                buckets = _buckets_for(cik_norm, form, state, total_recent)
                if not buckets:
                    continue
                rows.append(
                    {
                        "cik": cik_raw,
                        "company_name": company_name,
                        "state_of_incorporation": state,
                        "form": form,
                        "accession_number": acc,
                        "filing_date": date,
                        "primary_document": doc,
                        "exhibit_type": _exhibit_type(form),
                        "filing_index_url": _filing_index_url(cik_raw, acc),
                        "buckets": ",".join(buckets),
                        "expected_subsidiary_count": "",  # for manual fill-in
                        "sample_expected_names": "",  # for manual fill-in
                        "notes": "",  # for manual fill-in
                    }
                )

    # Sort: known_giant first, then by bucket priority, then by CIK.
    priority = {
        "known_giant": 0,
        "large_filer_seed": 1,
        "amendment": 2,
        "foreign_incorp": 3,
        "small_filer": 4,
    }
    rows.sort(
        key=lambda r: (
            min((priority.get(b, 99) for b in r["buckets"].split(",")), default=99),
            r["cik"],
        )
    )

    fieldnames = [
        "cik",
        "company_name",
        "state_of_incorporation",
        "form",
        "accession_number",
        "filing_date",
        "primary_document",
        "exhibit_type",
        "filing_index_url",
        "buckets",
        "expected_subsidiary_count",
        "sample_expected_names",
        "notes",
    ]
    with output_path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    bucket_counts: Counter[str] = Counter()
    form_counts: Counter[str] = Counter()
    for r in rows:
        bucket_counts.update(r["buckets"].split(","))
        form_counts[r["form"]] += 1
    log.info("=" * 60)
    log.info("Ground-truth candidate summary")
    log.info("=" * 60)
    log.info("Total candidate rows (10-K / 20-F only): %d", len(rows))
    log.info("By bucket:")
    for bucket, count in bucket_counts.most_common():
        log.info("  %-20s %d", bucket, count)
    log.info("By form:")
    for form, count in form_counts.most_common():
        log.info("  %-20s %d", form, count)
    log.info("Output written to: %s", output_path)


def main() -> None:
    """Parse args and run selector."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, type=pathlib.Path, help="Path to curated submissions zip"
    )
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path to output CSV")
    args = parser.parse_args()

    if not args.input.exists():
        log.error("Input zip not found: %s", args.input)
        sys.exit(1)

    select(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
