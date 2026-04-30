r"""Curate a submissions.zip subset containing only specific (CIK, accession_number) pairs.

Companion to ``curate_submissions.py``. Where that script filters by year, this
one filters down to a specific list of filings — useful for building tiny
integration-test inputs that target known problem cases (e.g. filings whose
GPT extraction was truncated).

For each requested CIK, ``filings.recent`` is rewritten to contain only the
single matching row, and ``filings.files`` is emptied. The output zip is
structurally identical to the input, just smaller.

Usage:
    uv run python scripts/curate_submissions_by_accession.py \\
        --input data/corporate-struct/input/submissions.zip \\
        --output data/corporate-struct/integration-tests/submissions_truncated.zip \\
        --filing 0000040545:0000040545-25-000015 \\
        --filing 0000099250:0000107263-25-000031 \\
        --filing 0000200406:0000200406-25-000038
"""

import argparse
import json
import logging
import pathlib
import sys
import zipfile

log = logging.getLogger(__name__)


def _find_filing(data: dict, accession: str, zf: zipfile.ZipFile) -> tuple[dict, str] | None:
    """Search ``filings.recent`` and overflow files for a row with ``accession``.

    Args:
        data: Parsed CIK*.json content.
        accession: Accession number to find (e.g. ``0000040545-25-000015``).
        zf: Open zipfile for resolving overflow file references.

    Returns:
        Tuple of (single-row recent dict, source description) if found, else None.
    """
    recent = data.get("filings", {}).get("recent", {}) or {}
    row = _extract_row(recent, accession)
    if row is not None:
        return row, "recent"

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
        row = _extract_row(overflow, accession)
        if row is not None:
            return row, f"overflow:{name}"

    return None


def _extract_row(recent: dict, accession: str) -> dict | None:
    """Return a single-row recent dict for ``accession``, or None if absent."""
    accs = recent.get("accessionNumber", [])
    if accession not in accs:
        return None
    i = accs.index(accession)
    n = len(accs)
    one_row: dict = {}
    for key, value in recent.items():
        if isinstance(value, list) and len(value) == n:
            one_row[key] = [value[i]]
        else:
            one_row[key] = value
    return one_row


def curate(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    targets: dict[str, str],
) -> None:
    """Read input zip, write a zip containing only the requested filings."""
    written = 0
    missing: list[tuple[str, str]] = []

    with (
        zipfile.ZipFile(input_path, "r") as zin,
        zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout,
    ):
        for cik, accession in targets.items():
            name = f"CIK{cik}.json"
            try:
                with zin.open(name) as f:
                    data = json.load(f)
            except KeyError:
                log.warning("CIK file not in zip: %s", name)
                missing.append((cik, accession))
                continue

            found = _find_filing(data, accession, zin)
            if found is None:
                log.warning("Accession %s not found for CIK %s", accession, cik)
                missing.append((cik, accession))
                continue

            one_row, source = found
            data.setdefault("filings", {})["recent"] = one_row
            data["filings"]["files"] = []
            zout.writestr(name, json.dumps(data))
            written += 1
            log.info(
                "Kept %s acc=%s form=%s date=%s doc=%s (from %s)",
                cik,
                accession,
                one_row.get("form", ["?"])[0],
                one_row.get("filingDate", ["?"])[0],
                one_row.get("primaryDocument", ["?"])[0],
                source,
            )

    log.info("=" * 60)
    log.info("Wrote %d / %d requested filings to %s", written, len(targets), output_path)
    if missing:
        log.warning("Missing:")
        for cik, acc in missing:
            log.warning("  %s  %s", cik, acc)


def _parse_filing(spec: str) -> tuple[str, str]:
    """Parse a ``CIK:ACCESSION`` argument."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"Filing spec must be CIK:ACCESSION, got {spec!r}")
    cik, acc = spec.split(":", 1)
    return cik.strip(), acc.strip()


def main() -> None:
    """Parse args and run the curator."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--filing",
        required=True,
        action="append",
        type=_parse_filing,
        help="CIK:ACCESSION pair. Repeat for each filing to keep.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error("Input zip not found: %s", args.input)
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    targets = dict(args.filing)
    curate(args.input, args.output, targets)


if __name__ == "__main__":
    main()
