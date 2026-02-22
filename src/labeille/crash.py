"""Crash detection and signature extraction.

This module handles detecting crashes from test suite runs (segfaults, aborts,
assertion failures) and extracting crash signatures for deduplication. It also
provides utilities for interpreting process exit codes and signals.
"""

from __future__ import annotations

import re
import signal as _signal
from dataclasses import dataclass

from labeille.logging import get_logger

log = get_logger("crash")

# Exit codes that map to well-known signals on systems that don't use negative
# return codes.
_SIGNAL_EXIT_CODES: dict[int, int] = {
    134: _signal.SIGABRT,
    139: _signal.SIGSEGV,
}

# Stderr patterns that indicate a crash.  Each tuple is
# (compiled_regex, is_assertion_flag).  Order matters: assertion is checked
# before the generic "Aborted" pattern so that assertion messages containing
# "Aborted" are classified correctly.
_CRASH_PATTERNS: list[tuple[re.Pattern[str], bool]] = [
    (re.compile(r"Segmentation fault", re.IGNORECASE), False),
    (re.compile(r"Fatal Python error:", re.IGNORECASE), False),
    (re.compile(r"Assertion .+ failed", re.IGNORECASE), True),
    # Match "Aborted" only at the start of a line (e.g. "Aborted (core dumped)").
    # This avoids false positives from "AbortError" or "was aborted" in normal text.
    (re.compile(r"^Aborted", re.MULTILINE), False),
]


@dataclass
class CrashInfo:
    """Information about a detected crash."""

    signal_number: int
    signal_name: str
    signature: str
    stderr_snippet: str
    is_assertion: bool = False


def detect_crash(return_code: int, stderr: str) -> CrashInfo | None:
    """Detect whether a process result represents a crash.

    A result is classified as a crash if any of these conditions hold:

    * *return_code* < 0  (killed by signal)
    * *return_code* is 134 (SIGABRT) or 139 (SIGSEGV)
    * *stderr* matches one of the known crash patterns

    Args:
        return_code: The process return code.
        stderr: The captured stderr output.

    Returns:
        A :class:`CrashInfo` if a crash was detected, otherwise ``None``.
    """
    sig_num = _signal_from_exit_code(return_code)
    is_assertion = False
    matched_line = ""

    # Check stderr patterns.
    for pattern, assertion_flag in _CRASH_PATTERNS:
        match = pattern.search(stderr)
        if match is not None:
            # Extract the line containing the match.
            start = stderr.rfind("\n", 0, match.start()) + 1
            end = stderr.find("\n", match.end())
            if end == -1:
                end = len(stderr)
            matched_line = stderr[start:end].strip()
            if assertion_flag:
                is_assertion = True
            # If we haven't identified a signal from exit code, try from pattern.
            if sig_num == 0:
                if "segmentation fault" in matched_line.lower():
                    sig_num = _signal.SIGSEGV
                elif assertion_flag or "abort" in matched_line.lower():
                    sig_num = _signal.SIGABRT
                elif "fatal python error" in matched_line.lower():
                    # Fatal Python errors typically result in SIGABRT.
                    sig_num = _signal.SIGABRT
            break  # Use the first matching pattern.

    if sig_num == 0 and not matched_line:
        return None

    # Fall back to SIGABRT for stderr-only detections without a clear signal.
    if sig_num == 0:
        sig_num = _signal.SIGABRT

    sig_name = signal_name(sig_num)
    snippet = _extract_snippet(stderr)
    signature = extract_crash_signature(sig_name, matched_line)

    return CrashInfo(
        signal_number=sig_num,
        signal_name=sig_name,
        signature=signature,
        stderr_snippet=snippet,
        is_assertion=is_assertion,
    )


def extract_crash_signature(sig_name: str, matched_line: str) -> str:
    """Build a human-readable crash signature from signal name and stderr line.

    Args:
        sig_name: The signal name (e.g. ``"SIGSEGV"``).
        matched_line: The relevant stderr line, if any.

    Returns:
        A string signature for the crash.
    """
    if matched_line:
        return f"{sig_name}: {matched_line}"
    return sig_name


def signal_name(signal_number: int) -> str:
    """Convert a signal number to its human-readable name.

    Args:
        signal_number: The signal number (e.g. 11 for SIGSEGV).

    Returns:
        The signal name (e.g. ``"SIGSEGV"``), or ``"SIG<N>"`` if unknown.
    """
    try:
        return _signal.Signals(signal_number).name
    except ValueError:
        return f"SIG{signal_number}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _signal_from_exit_code(return_code: int) -> int:
    """Determine the signal number from a process return code.

    Returns 0 if the return code does not indicate a signal-based crash.
    """
    if return_code < 0:
        return -return_code
    return _SIGNAL_EXIT_CODES.get(return_code, 0)


def _extract_snippet(stderr: str, max_lines: int = 50) -> str:
    """Return the last *max_lines* lines of *stderr* as a snippet."""
    lines = stderr.splitlines()
    if len(lines) <= max_lines:
        return stderr
    return "\n".join(lines[-max_lines:])
