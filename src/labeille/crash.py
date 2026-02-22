"""Crash detection and signature extraction.

This module handles detecting crashes from test suite runs (segfaults, aborts,
assertion failures) and extracting crash signatures for deduplication. It also
provides utilities for interpreting process exit codes and signals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrashInfo:
    """Information about a detected crash."""

    signal_number: int
    signal_name: str
    signature: str
    stderr_snippet: str
    is_assertion: bool = False


def extract_crash_info(return_code: int, stderr: str) -> CrashInfo | None:
    """Extract crash information from a process exit code and stderr output.

    Detects segfaults (signal 11), aborts (signal 6), and assertion failures.

    Args:
        return_code: The process return code (negative values indicate signals).
        stderr: The captured stderr output.

    Returns:
        A ``CrashInfo`` if a crash was detected, otherwise ``None``.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def compute_crash_signature(crash: CrashInfo) -> str:
    """Compute a deduplication signature for a crash.

    The signature is used to group identical crashes across different
    test runs and packages.

    Args:
        crash: The crash information.

    Returns:
        A string signature for the crash.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def signal_name(signal_number: int) -> str:
    """Convert a signal number to its human-readable name.

    Args:
        signal_number: The signal number (e.g. 11 for SIGSEGV).

    Returns:
        The signal name (e.g. ``"SIGSEGV"``).

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError
