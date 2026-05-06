r"""Curate a year-filtered subset of SEC EDGAR submissions.zip.

Produces a drop-in replacement zip that contains only companies with at
least one filing in the target year(s), with each company's CIK JSON rewritten
so that ``filings.recent`` holds *only* the matching filings (across all
forms) and ``filings.files`` is empty. Overflow sibling files
(``CIK*-submissions-*.json``) are not copied.

The output zip is structurally identical to the original — same flat layout
of ``CIK*.json`` entries, same per-company schema — just smaller. It drops
straight into the pipeline via ``config.input_file``; no pipeline code
changes are needed.

Usage (single year):
    uv run python scripts/curate_submissions.py \\
        --input submissions.zip \\
        --output submissions_2025.zip \\
        --year 2025

Usage (multiple years — combined into one zip):
    uv run python scripts/curate_submissions.py \\
        --input submissions.zip \\
        --output submissions_2023_2024_2025.zip \\
        --year 2023 2024 2025
"""

import argparse
import json
import logging
import pathlib
import re
import sys
import zipfile
from collections import Counter

_IS_OVERFLOW = re.compile(r"-submissions-\d+\.json$")
_IS_PRIMARY_CIK = re.compile(r"^CIK\d+\.json$")

log = logging.getLogger(__name__)


def _filter_recent(recent: dict, year_prefixes: set[str]) -> tuple[dict, list[str]]:
    """Return a new ``recent`` dict keeping only indices where filingDate matches any prefix.

    Every parallel array in ``recent`` is masked with the same index set so
    the equal-length invariant the pipeline validates (pipeline.py:238-245)
    is preserved.

    Args:
        recent: Original ``filings.recent`` mapping.
        year_prefixes: E.g. ``{"2023-", "2024-", "2025-"}``; a row is kept if its
            filingDate starts with any of these.

    Returns:
        Tuple of (filtered recent dict, list of kept form strings).
    """
    filing_dates = recent.get("filingDate", [])
    keep_idx = [
        i
        for i, d in enumerate(filing_dates)
        if isinstance(d, str) and any(d.startswith(p) for p in year_prefixes)
    ]

    filtered: dict = {}
    for key, value in recent.items():
        if isinstance(value, list) and len(value) == len(filing_dates):
            filtered[key] = [value[i] for i in keep_idx]
        else:
            # Preserve any non-parallel-array fields verbatim (defensive; SEC
            # submissions.recent is all parallel arrays in practice).
            filtered[key] = value

    kept_forms = [recent.get("form", [])[i] for i in keep_idx] if recent.get("form") else []
    return filtered, kept_forms


def _merge_overflow(
    data: dict, zf: zipfile.ZipFile, year_prefixes: set[str]
) -> tuple[dict, list[str]]:
    """Append any year-matching rows from overflow files into a filtered recent.

    Overflow files store the same parallel-array keys at top level (not
    nested under ``filings.recent``) — see ``_retrieve_overflow_filings`` in
    pipeline.py.

    Returns:
        Tuple of (combined filtered recent dict, list of kept forms).
    """
    recent = data.get("filings", {}).get("recent", {}) or {}
    filtered_recent, kept_forms = _filter_recent(recent, year_prefixes)

    for entry in data.get("filings", {}).get("files", []) or []:
        name = entry.get("name", "")
        if not name:
            continue
        try:
            with zf.open(name) as of:
                overflow = json.load(of)
        except KeyError:
            log.warning("Overflow file referenced but not in zip: %s", name)
            continue

        o_dates = overflow.get("filingDate", [])
        if not o_dates:
            continue
        keep_idx = [
            i
            for i, d in enumerate(o_dates)
            if isinstance(d, str) and any(d.startswith(p) for p in year_prefixes)
        ]
        if not keep_idx:
            continue

        # Append matching overflow rows to each parallel array already in filtered_recent.
        for key, value in overflow.items():
            if not isinstance(value, list) or len(value) != len(o_dates):
                continue
            picked = [value[i] for i in keep_idx]
            if key in filtered_recent and isinstance(filtered_recent[key], list):
                filtered_recent[key].extend(picked)
            else:
                # Key exists in overflow but not in recent — seed with empty then extend.
                filtered_recent.setdefault(key, []).extend(picked)

        kept_forms.extend(
            [overflow.get("form", [])[i] for i in keep_idx] if overflow.get("form") else []
        )

    return filtered_recent, kept_forms


def curate(input_path: pathlib.Path, output_path: pathlib.Path, years: list[int]) -> None:
    """Read input submissions zip, write year-filtered output zip."""
    year_prefixes = {f"{y}-" for y in years}
    companies_scanned = 0
    companies_kept = 0
    total_filings = 0
    form_counter: Counter[str] = Counter()

    with (
        zipfile.ZipFile(input_path, "r") as zin,
        zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout,
    ):
        for name in zin.namelist():
            if _IS_OVERFLOW.search(name):
                continue  # overflow files handled via their primary
            if not _IS_PRIMARY_CIK.match(name):
                continue

            companies_scanned += 1
            if companies_scanned % 5000 == 0:
                log.info("Scanned %d companies (%d kept)...", companies_scanned, companies_kept)

            try:
                with zin.open(name) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Could not read %s: %s", name, exc)
                continue

            filtered_recent, kept_forms = _merge_overflow(data, zin, year_prefixes)

            if not kept_forms:
                continue

            data.setdefault("filings", {})["recent"] = filtered_recent
            data["filings"]["files"] = []

            zout.writestr(name, json.dumps(data))
            companies_kept += 1
            total_filings += len(kept_forms)
            form_counter.update(kept_forms)

    years_str = ", ".join(str(y) for y in sorted(years))
    log.info("=" * 60)
    log.info("Curation summary (years=%s)", years_str)
    log.info("=" * 60)
    log.info("Companies scanned: %d", companies_scanned)
    log.info("Companies kept:    %d", companies_kept)
    log.info("Total filings:     %d", total_filings)
    log.info("Form breakdown (top 20):")
    for form, count in form_counter.most_common(20):
        log.info("  %-20s %d", form, count)
    log.info("Output written to: %s", output_path)


def main() -> None:
    """Parse args and run the curator."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, type=pathlib.Path, help="Path to input submissions.zip"
    )
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path to output zip")
    parser.add_argument(
        "--year",
        type=int,
        nargs="+",
        default=[2025],
        help="Filing year(s) to keep (default: 2025). Pass multiple values to combine years.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error("Input zip not found: %s", args.input)
        sys.exit(1)

    curate(args.input, args.output, args.year)


if __name__ == "__main__":
    main()
