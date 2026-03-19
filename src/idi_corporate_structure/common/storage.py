"""Provides storage utilities for use across the application."""

# Standard library imports
import json
import tempfile
import zipfile
from contextlib import contextmanager
from typing import Iterator

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
    """Load a JSON file from the given path.

    Supports local paths and s3:// URLs.
    Returns empty dict/list if file does not exist; raises on other errors.
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
        file_path: The path to the JSON file
    """

    tp = { "headers": headers } if headers else {}
    with smart_open.open(file_path, "rb", transport_params=tp) as f:
        with zipfile.ZipFile(f) as zf:
            yield zf
