"""Shared file I/O utilities."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from labeille.logging import get_logger

log = get_logger("io_utils")

T = TypeVar("T")


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write content to a file atomically using a temp file and os.replace.

    Creates a temporary file in the same directory as *path*, writes *content*,
    then atomically replaces *path*.  On failure the temp file is cleaned up.
    """
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:  # Cleanup temp file on any failure, including KeyboardInterrupt.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_meta_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON metadata file atomically.

    Uses ``atomic_write_text`` to prevent corruption if the process
    is interrupted mid-write.
    """
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def iter_jsonl(
    path: Path,
    deserialize: Callable[[dict[str, Any]], T],
) -> Iterator[T]:
    """Stream JSONL records from *path*, skipping malformed lines.

    Yields deserialized objects using *deserialize*.  Malformed lines
    (truncated JSON, missing keys) are logged and skipped so that a
    single corrupt trailing line does not crash the entire load.
    """
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield deserialize(data)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                log.debug("Skipping malformed JSONL line in %s: %s", path.name, exc)
                skipped += 1
    if skipped:
        log.warning("Skipped %d malformed line(s) in %s", skipped, path)


def load_jsonl(
    path: Path,
    deserialize: Callable[[dict[str, Any]], T],
) -> list[T]:
    """Load all records from a JSONL file, tolerating malformed lines."""
    return list(iter_jsonl(path, deserialize))


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    """Append a single JSON object as a line to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
