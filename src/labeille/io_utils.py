"""Shared file I/O and process utilities."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

import yaml

from labeille.logging import get_logger

log = get_logger("io_utils")

T = TypeVar("T")


def generate_run_id(prefix: str) -> str:
    """Generate a UTC-timestamped run ID with a consistent format.

    Format: ``{prefix}_{YYYYMMDD}_{HHMMSS}`` (always UTC).
    """
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


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


def safe_load_yaml(path: Path) -> dict[str, Any] | None:
    """Load a YAML file, returning ``None`` on parse errors.

    Catches ``yaml.YAMLError`` and logs a warning so that a single
    malformed file does not crash batch operations.  Returns ``None``
    if the file cannot be parsed or does not contain a YAML mapping.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        log.warning("Skipping %s: malformed YAML: %s", path.name, exc)
        return None
    if not isinstance(data, dict):
        log.warning("Skipping %s: expected YAML mapping, got %s", path.name, type(data).__name__)
        return None
    return data


def write_meta_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON metadata file atomically.

    Uses ``atomic_write_text`` to prevent corruption if the process
    is interrupted mid-write.
    """
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def load_json_file(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file.

    Raises:
        ValueError: If the file contains invalid JSON or is not a dict.
        OSError: If the file cannot be read.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path.name}, got {type(data).__name__}")
    return data


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


def kill_process_group(pid: int) -> None:
    """Kill an entire process group by PID.

    Resolves the process group ID via ``os.getpgid`` and sends SIGKILL to
    the entire group.  Silently ignores errors if the process has already
    exited or cannot be killed (e.g. permission denied).

    Use with subprocesses started via ``start_new_session=True``.
    """
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except (PermissionError, OSError) as exc:
        log.warning("Could not kill process group for PID %d: %s", pid, exc)


def run_in_process_group(
    cmd: str | list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    """Run a command in its own process group with timeout enforcement.

    On timeout, kills the entire process group (not just the immediate
    child) to prevent orphaned grandchild processes from accumulating.

    Args:
        cmd: Shell command string or argument list.
        cwd: Working directory.
        env: Environment variables.
        timeout: Timeout in seconds.

    Returns:
        A :class:`~subprocess.CompletedProcess` with stdout, stderr, and returncode.

    Raises:
        subprocess.TimeoutExpired: If the command exceeds the timeout.
            The exception's ``stdout`` and ``stderr`` attributes contain any
            partial output captured before the timeout.
    """
    shell = isinstance(cmd, str)
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        kill_process_group(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
