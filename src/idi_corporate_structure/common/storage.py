"""Provides storage utilities for use across the application."""

# Standard library imports
import json
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager

# Third party imports
import smart_open
from botocore.exceptions import ClientError


def _empty_for_return_type(return_type: str) -> dict | list:
    """Return empty dict or list per return_type."""
    if return_type == "dict":
        return {}
    if return_type == "list":
        return []
    raise ValueError(f"Invalid return type: {return_type}")


def load_json(file_path: str, return_type: str = "dict") -> dict | list:
    """Load a JSON file from a local path or S3 URL.

    Supports any path scheme understood by ``smart_open`` (local, ``s3://``).
    Missing files — locally absent or absent on S3 — are treated as empty and
    return the appropriate empty container instead of raising.

    Args:
        file_path: Local filesystem path or ``s3://bucket/key`` URL of the JSON file.
        return_type: Expected top-level type of the JSON document — ``"dict"`` or
            ``"list"``. Controls the empty value returned when the file is absent.
            Raises ``ValueError`` for any other value.

    Returns:
        Parsed JSON content as a ``dict`` or ``list``. Returns an empty ``dict`` or
        ``list`` (per ``return_type``) when the file does not exist.

    Raises:
        ValueError: If ``return_type`` is not ``"dict"`` or ``"list"``.
        botocore.exceptions.ClientError: If an S3 error other than ``NoSuchKey`` occurs.
        json.JSONDecodeError: If the file exists but contains invalid JSON.
    """
    try:
        with smart_open.open(file_path) as f:
            return json.load(f)

    except (FileNotFoundError, OSError):
        return _empty_for_return_type(return_type)

    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return _empty_for_return_type(return_type)
        raise


def save_json(file_path: str, data: dict | list, mode: str = "w") -> None:
    """Save a JSON file to the given path.

    Efficient writing: https://github.com/piskvorky/smart_open/blob/develop/howto.md#how-to-write-to-s3-efficiently

    Can write in append mode for local files, S3 files are always overwritten.

    Args:
        file_path: The path to the JSON file.
        data: The JSON data to save to the file as a dictionary or list.
        mode: File open mode ("w" to overwrite, "a" to append). S3 paths always overwrite.
    """
    if "s3://" in file_path:
        with tempfile.NamedTemporaryFile() as tmp:
            tp = {"writebuffer": tmp}
            with smart_open.open(file_path, "w", transport_params=tp) as fout:
                json.dump(data, fout, indent=2)
    else:
        with smart_open.open(file_path, mode) as fout:
            json.dump(data, fout, indent=2)


@contextmanager
def open_zip(file_path: str, headers: dict | None = None) -> Iterator[zipfile.ZipFile]:
    """Open a zip file from a local path, S3, or HTTPS URL.

    Supports any path scheme handled by smart_open (local, s3://, https://).
    HTTPS requires the server to support range requests (Accept-Ranges: bytes).

    Args:
        file_path: Path to the ZIP file — local filesystem path, ``s3://`` URL, or
            ``https://`` URL. HTTPS requires the server to support range requests
            (``Accept-Ranges: bytes``).
        headers: Optional HTTP headers passed as transport params (e.g. ``User-Agent``
            for SEC EDGAR). Ignored for local and S3 paths.

    Yields:
        An open ``zipfile.ZipFile`` object. The underlying stream is closed
        automatically when the context manager exits.

    Raises:
        zipfile.BadZipFile: If the file is not a valid ZIP archive.
        OSError: If the file cannot be opened or read.
    """
    tp = {"headers": headers} if headers else {}
    with smart_open.open(file_path, "rb", transport_params=tp) as f:
        with zipfile.ZipFile(f) as zf:
            yield zf
