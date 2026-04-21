"""Verification script for corporate structure pipeline output.

Each check is independently selectable. Run all checks by default, or pass
--checks to run a specific subset. Use --list-checks to see what is available.

Usage:
    uv run python scripts/verify_output.py \
        --parquet output/subsidiaries.parquet \
        --failures failures/failures.json \
        --user-agent "Your Name your@email.com"

    # Run specific checks only:
    uv run python scripts/verify_output.py \
        --parquet output/subsidiaries.parquet \
        --failures failures/failures.json \
        --user-agent "Your Name your@email.com" \
        --checks structural location failures

    # List available checks:
    uv run python scripts/verify_output.py --list-checks
"""

import argparse
import dataclasses
import html as _html
import json
import logging
import pathlib
import re
import sys
from collections import Counter
from collections.abc import Callable

import pandas as pd
import requests

_DEFAULT_SAMPLE_SIZE = 30
_LOCATION_EMPTY_RATE_THRESHOLD = 0.15
_EXTRACTION_FAILED_RATE_THRESHOLD = 0.05
_GROUNDING_FAIL_THRESHOLD = 0.05

# Conservative lower bounds on subsidiary count for the most recent filing.
# Actual counts are higher; these catch silent extraction failures.
_KNOWN_CIKS: dict[str, tuple[str, int]] = {
    "320193": ("Apple Inc. (10-K, Exhibit 21)", 500),
    "789019": ("Microsoft Corporation (10-K, Exhibit 21)", 400),
    "97476": ("Toyota Motor Corporation (20-F, Exhibit 8)", 200),
}

_PASS = "[PASS]"  # noqa: S105
_FAIL = "[FAIL]"
_WARN = "[WARN]"
_INFO = "[INFO]"

log = logging.getLogger(__name__)


@dataclasses.dataclass
class VerifyContext:
    """Shared inputs passed to every check function."""

    df: pd.DataFrame
    failures_path: str
    sample_size: int
    sec_headers: dict[str, str]


@dataclasses.dataclass(frozen=True)
class Check:
    """A single named verification check."""

    name: str
    description: str
    fn: Callable[[VerifyContext], list[str]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    """Log a titled section separator."""
    log.info("\n%s", "=" * 60)
    log.info("  %s", title)
    log.info("%s", "=" * 60)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_structural(ctx: VerifyContext) -> list[str]:
    """Assert exhibit_url is non-null and exhibit_type matches form_type.

    Catches:
      - Rows where no exhibit URL was recorded (exhibit was never fetched).
      - Rows where the wrong exhibit number is assigned — 10-K must be exhibit
        21, 20-F must be exhibit 8.
    """
    _section("Structural integrity")
    failures: list[str] = []
    df = ctx.df

    blank_url = df["exhibit_url"].isna() | (df["exhibit_url"].astype(str) == "")
    if blank_url.any():
        n = int(blank_url.sum())
        log.error("%s %d row(s) have a blank exhibit_url", _FAIL, n)
        log.error(
            "%s",
            df[blank_url][["parent_cik", "accession_number", "name"]].to_string(index=False),
        )
        failures.append(f"{n} rows with blank exhibit_url")
    else:
        log.info("%s All %d rows have a non-empty exhibit_url", _PASS, len(df))

    is_10k = df["form_type"].str.match(r"10-?K", na=False)
    is_20f = df["form_type"].str.match(r"20-?F", na=False)
    bad_10k = is_10k & (df["exhibit_type"] != "21")
    bad_20f = is_20f & (df["exhibit_type"] != "8")

    if bad_10k.any():
        n = int(bad_10k.sum())
        log.error("%s %d 10-K row(s) have exhibit_type != '21'", _FAIL, n)
        failures.append(f"{n} 10-K rows with wrong exhibit_type")
    else:
        log.info("%s All 10-K rows have exhibit_type='21'", _PASS)

    if bad_20f.any():
        n = int(bad_20f.sum())
        log.error("%s %d 20-F row(s) have exhibit_type != '8'", _FAIL, n)
        failures.append(f"{n} 20-F rows with wrong exhibit_type")
    else:
        log.info("%s All 20-F rows have exhibit_type='8'", _PASS)

    other = ~is_10k & ~is_20f
    if other.any():
        log.warning("%s %d row(s) have an unexpected form_type:", _WARN, int(other.sum()))
        log.warning("%s", df[other]["form_type"].value_counts().to_string())

    return failures


def check_grounding(ctx: VerifyContext) -> list[str]:
    """Fetch each unique exhibit URL once and verify all subsidiary names appear in it.

    Catches:
      - Hallucinated names (GPT invented a name not present in the source text).
      - Wrong exhibit fetched (URL points to a different document than expected).

    Groups rows by exhibit_url so each document is fetched exactly once, then checks
    every name in that group against the response text. --sample-size limits the
    number of unique exhibits fetched, not the number of rows — all names within a
    fetched exhibit are always checked. Omit --sample-size (or set to 0) to check
    every exhibit in the output.

    Location is not checked here — jurisdiction strings (e.g. "Delaware") appear
    many times throughout a typical exhibit, so a substring match would pass even
    for an incorrectly attributed location. Location completeness and distribution
    are covered by the location check instead.
    """
    eligible = ctx.df[ctx.df["exhibit_url"].astype(str).str.startswith("http")]
    groups = eligible.groupby("exhibit_url")
    urls = list(groups.groups.keys())

    if ctx.sample_size:
        urls = urls[: ctx.sample_size]

    _section(
        f"Name grounding — {len(urls)} exhibit(s), {eligible['exhibit_url'].isin(urls).sum()} rows"
    )
    failures: list[str] = []

    if not urls:
        log.warning("%s No rows with fetchable exhibit URLs — skipping", _WARN)
        return failures

    total_names = 0
    passed = 0
    fetch_errors: list[tuple[str, str]] = []

    for url in urls:
        try:
            resp = requests.get(url, headers=ctx.sec_headers, timeout=15)
            resp.raise_for_status()
            # Decode HTML entities (&amp; → &, &lt; → <, etc.) then collapse all
            # whitespace to a single space — older EDGAR exhibits both encode
            # special characters as entities and split names across HTML lines.
            text = re.sub(r"\s+", " ", _html.unescape(resp.text)).lower()

            for _, row in groups.get_group(url).iterrows():
                total_names += 1
                if row["name"].lower() in text:
                    passed += 1
                else:
                    log.warning("%s '%s' not found in %s", _WARN, row["name"], url)

        except requests.RequestException as exc:
            fetch_errors.append((str(url), str(exc)))

    if fetch_errors:
        log.warning("%s %d exhibit(s) could not be fetched:", _WARN, len(fetch_errors))
        for url, err in fetch_errors[:5]:
            log.warning("    %s: %s", url, err)

    ungrounded = total_names - passed
    ungrounded_rate = ungrounded / total_names if total_names else 0
    rate_status = _FAIL if ungrounded_rate > _GROUNDING_FAIL_THRESHOLD else _PASS
    log.info(
        "%s %d/%d names found across %d exhibit(s) — %.1f%% ungrounded (threshold: %.0f%%)",
        rate_status,
        passed,
        total_names,
        len(urls),
        ungrounded_rate * 100,
        _GROUNDING_FAIL_THRESHOLD * 100,
    )
    if ungrounded_rate > _GROUNDING_FAIL_THRESHOLD:
        failures.append(
            f"Ungrounded name rate {ungrounded_rate:.1%} exceeds {_GROUNDING_FAIL_THRESHOLD:.0%}"
            f" ({ungrounded}/{total_names} names across {len(urls)} exhibit(s))"
        )

    return failures


def check_location(ctx: VerifyContext) -> list[str]:
    """Report empty subsidiary location rate and parent_location distribution.

    Catches:
      - GPT systematically failing to extract jurisdiction strings from exhibits.
      - Exhibits where the location column is absent (expect a higher empty rate
        for those companies specifically).

    Two fields are reported separately:
      - location: GPT-extracted subsidiary jurisdiction (free text, may be blank
        legitimately for some exhibits).
      - parent_location: raw SEC stateOfIncorporation code (not GPT-extracted);
        logged for manual inspection of domestic/foreign composition.
    """
    _section("Location quality")
    failures: list[str] = []
    df = ctx.df

    empty_loc = df["location"].isna() | (df["location"].astype(str) == "")
    rate = float(empty_loc.mean())
    status = _FAIL if rate > _LOCATION_EMPTY_RATE_THRESHOLD else _PASS
    log.info(
        "%s Subsidiary location empty rate: %.1f%% (threshold: %.0f%%)",
        status,
        rate * 100,
        _LOCATION_EMPTY_RATE_THRESHOLD * 100,
    )
    if rate > _LOCATION_EMPTY_RATE_THRESHOLD:
        failures.append(
            f"Subsidiary location empty rate {rate:.1%} exceeds "
            f"{_LOCATION_EMPTY_RATE_THRESHOLD:.0%}"
        )

    log.info("\n%s parent_location top 20 (raw SEC stateOfIncorporation codes):", _INFO)
    log.info("%s", df["parent_location"].value_counts().head(20).to_string())

    return failures


def check_failures(ctx: VerifyContext) -> list[str]:
    """Load failures.json and report the failure type distribution.

    Catches:
      - Elevated extraction_failed rates indicating GPT is not returning
        structured data (prompt issue, model change, or document size).
      - Unexpected failure patterns after a code or configuration change.

    failures.json schema (written by FailureRegistry.save):
      {"entries": [[cik, filename], ...], "reasons": {"cik filename": "type"}}
    """
    _section("Failure distribution")
    issues: list[str] = []

    try:
        with pathlib.Path(ctx.failures_path).open() as f:
            data = json.load(f)
    except FileNotFoundError:
        log.warning("%s failures.json not found at %s — skipping", _WARN, ctx.failures_path)
        return issues
    except json.JSONDecodeError as exc:
        log.error("%s Could not parse failures.json: %s", _FAIL, exc)
        return [f"failures.json parse error: {exc}"]

    reasons = data.get("reasons", {})
    total = len(reasons)
    log.info("%s Total permanent failures recorded: %d", _INFO, total)

    if not reasons:
        log.info("%s No failures recorded", _PASS)
        return issues

    counts = Counter(reasons.values())
    log.info("\n%s Failure type breakdown:", _INFO)
    for failure_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        log.info("  %-40s %8d  (%.1f%%)", failure_type, count, pct)

    extraction_failed = counts.get("extraction_failed", 0)
    rate = extraction_failed / total
    status = _FAIL if rate > _EXTRACTION_FAILED_RATE_THRESHOLD else _PASS
    log.info(
        "\n%s extraction_failed rate: %.1f%% (threshold: %.0f%%)",
        status,
        rate * 100,
        _EXTRACTION_FAILED_RATE_THRESHOLD * 100,
    )
    if rate > _EXTRACTION_FAILED_RATE_THRESHOLD:
        issues.append(f"High extraction_failed rate: {rate:.1%} of all failures")

    return issues


# ---------------------------------------------------------------------------
# Check registry — controls ordering and --list-checks output
# ---------------------------------------------------------------------------

CHECKS: list[Check] = [
    Check(
        name="structural",
        description="Assert exhibit_url is non-null and exhibit_type matches form_type",
        fn=check_structural,
    ),
    Check(
        name="grounding",
        description="Fetch each unique exhibit URL once and verify all subsidiary names appear in it",
        fn=check_grounding,
    ),
    Check(
        name="location",
        description="Report empty subsidiary location rate and parent_location distribution",
        fn=check_location,
    ),
    Check(
        name="failures",
        description="Load failures.json and report failure type distribution",
        fn=check_failures,
    ),
]

_CHECK_NAMES: list[str] = [c.name for c in CHECKS]


# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------


def run_checks(ctx: VerifyContext, checks: list[Check]) -> list[str]:
    """Run each check in sequence and return all collected failure messages.

    Args:
        ctx: Shared pipeline output and configuration for all checks.
        checks: Ordered list of checks to execute.

    Returns:
        Flat list of failure message strings across all checks.
    """
    all_failures: list[str] = []
    for check in checks:
        all_failures += check.fn(ctx)
    return all_failures


def print_summary(all_failures: list[str]) -> None:
    """Print the summary of the checks."""
    _section("Summary")
    if all_failures:
        log.error("%s %d check(s) failed:\n", _FAIL, len(all_failures))
        for item in all_failures:
            log.error("  - %s", item)
        sys.exit(1)
    else:
        log.info("%s All checks passed", _PASS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def create_args() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Verify SubsidiaryPipeline output (Parquet + failures.json).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--parquet", required=True, help="Path to subsidiaries.parquet")
    parser.add_argument("--failures", required=True, help="Path to failures.json")
    parser.add_argument(
        "--user-agent",
        required=True,
        metavar="STRING",
        help="User-Agent header sent with SEC EDGAR requests (required by EDGAR policy)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=_DEFAULT_SAMPLE_SIZE,
        help=(
            f"Max unique exhibits to fetch for the grounding check (default: {_DEFAULT_SAMPLE_SIZE}). "
            "All names within each fetched exhibit are checked. Set to 0 to fetch every exhibit."
        ),
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=_CHECK_NAMES,
        metavar="CHECK",
        default=_CHECK_NAMES,
        help=(
            f"One or more checks to run. Choices: {', '.join(_CHECK_NAMES)}. "
            "Default: all. Use --list-checks to see descriptions."
        ),
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="Print available checks with descriptions and exit",
    )
    return parser


def main() -> None:
    """Parse arguments, load data, run selected checks, and exit with status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = create_args()
    args = parser.parse_args()

    if args.list_checks:
        print("\nAvailable checks:\n")
        for check in CHECKS:
            print(f"  {check.name:<15}  {check.description}")
        print()
        sys.exit(0)

    # Load the results parquet file
    log.info("\nLoading %s ...", args.parquet)
    try:
        df = pd.read_parquet(args.parquet)
    except FileNotFoundError:
        log.error("%s Parquet file not found: %s", _FAIL, args.parquet)
        sys.exit(1)

    log.info("%s %d rows across %d unique parent CIKs", _INFO, len(df), df["parent_cik"].nunique())

    # Create an object to hold the context for the checks
    ctx = VerifyContext(
        df=df,
        failures_path=args.failures,
        sample_size=args.sample_size,
        sec_headers={"User-Agent": args.user_agent},
    )

    # Run selected checks
    selected = [c for c in CHECKS if c.name in set(args.checks)]
    all_failures = run_checks(ctx, selected)

    # Print the summary
    print_summary(all_failures)


if __name__ == "__main__":
    main()
