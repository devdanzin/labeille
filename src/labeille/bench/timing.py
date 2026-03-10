"""Timing capture for benchmark iterations.

Measures wall-clock time, user CPU time, system CPU time, and peak
resident set size for subprocess executions.  Uses resource.getrusage
for CPU times and ``/usr/bin/time -v`` for per-process peak RSS.
"""

from __future__ import annotations

import os
import re
import resource
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labeille.io_utils import kill_process_group
from labeille.logging import get_logger

log = get_logger("bench.timing")


# ---------------------------------------------------------------------------
# TimedResult
# ---------------------------------------------------------------------------


@dataclass
class TimedResult:
    """Result of a timed subprocess execution."""

    wall_time_s: float
    user_time_s: float
    sys_time_s: float
    peak_rss_mb: float
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def cpu_time_s(self) -> float:
        """Total CPU time (user + system)."""
        return self.user_time_s + self.sys_time_s


# ---------------------------------------------------------------------------
# Core timing implementation
# ---------------------------------------------------------------------------


def run_timed(
    command: str | list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
    use_time_wrapper: bool = True,
) -> TimedResult:
    """Execute a command and capture timing and resource usage.

    Args:
        command: Shell command string or argument list.
        cwd: Working directory for the subprocess.
        env: Environment variables for the subprocess.
        timeout: Maximum execution time in seconds.
        use_time_wrapper: If True and ``/usr/bin/time`` exists, wrap
            the command to get accurate per-invocation peak RSS.

    Returns:
        TimedResult with timing data and process output.
    """
    from labeille.runner import clean_env

    run_env = clean_env()
    if env:
        run_env.update(env)

    # Determine if we can use /usr/bin/time for RSS measurement.
    time_output_file: str | None = None
    if use_time_wrapper and Path("/usr/bin/time").exists():
        import tempfile

        fd, time_output_file = tempfile.mkstemp(
            prefix="labeille-time-",
            suffix=".txt",
        )
        os.close(fd)
        # GNU time with verbose output to a file.
        if isinstance(command, str):
            cmd_str = command
        else:
            cmd_str = " ".join(shlex.quote(c) for c in command)
        command = (
            f"/usr/bin/time -v -o {shlex.quote(time_output_file)} sh -c {shlex.quote(cmd_str)}"
        )

    # Snapshot children's resource usage before.
    pre_rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_start = time.monotonic()

    timed_out = False
    proc = subprocess.Popen(
        command if isinstance(command, list) else command,
        shell=isinstance(command, str),
        cwd=str(cwd) if cwd else None,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_group(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        exit_code = -1

    wall_time = time.monotonic() - wall_start

    # Snapshot children's resource usage after.
    post_rusage = resource.getrusage(resource.RUSAGE_CHILDREN)

    user_time = post_rusage.ru_utime - pre_rusage.ru_utime
    sys_time = post_rusage.ru_stime - pre_rusage.ru_stime

    # Peak RSS.
    peak_rss_mb = _extract_peak_rss(time_output_file, pre_rusage, post_rusage)

    return TimedResult(
        wall_time_s=round(wall_time, 6),
        user_time_s=round(max(user_time, 0.0), 6),
        sys_time_s=round(max(sys_time, 0.0), 6),
        peak_rss_mb=round(peak_rss_mb, 1),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


def _extract_peak_rss(
    time_output_file: str | None,
    pre_rusage: resource.struct_rusage,
    post_rusage: resource.struct_rusage,
) -> float:
    """Extract peak RSS from GNU time output or rusage fallback.

    Returns peak RSS in megabytes.
    """
    if time_output_file is not None:
        rss = _parse_gnu_time_rss(time_output_file)
        try:
            os.unlink(time_output_file)
        except OSError:
            pass
        if rss > 0:
            return rss

    # Fallback: delta in ru_maxrss.  On Linux, ru_maxrss is in KB.
    # On macOS, ru_maxrss is in bytes.
    import sys

    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    rss_delta = post_rusage.ru_maxrss - pre_rusage.ru_maxrss
    if rss_delta > 0:
        return rss_delta / divisor
    # Can't determine per-iteration RSS; report the cumulative.
    return post_rusage.ru_maxrss / divisor


def _parse_gnu_time_rss(time_output_file: str) -> float:
    """Parse peak RSS from GNU time verbose output.

    GNU ``time -v`` outputs a line like::

        Maximum resident set size (kbytes): 123456

    Returns peak RSS in MB, or 0.0 if parsing fails.
    """
    try:
        content = Path(time_output_file).read_text(encoding="utf-8")
        for line in content.splitlines():
            if "Maximum resident set size" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    kb = int(parts[-1].strip())
                    return kb / 1024
    except (OSError, ValueError) as exc:
        log.debug("Could not parse GNU time RSS from %s: %s", time_output_file, exc)
    return 0.0


# ---------------------------------------------------------------------------
# Venv-aware execution helper
# ---------------------------------------------------------------------------


def run_timed_in_venv(
    venv_path: Path,
    test_command: str,
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> TimedResult:
    """Execute a test command inside a virtual environment.

    Sets up ``PATH`` and ``VIRTUAL_ENV`` so the venv's Python and
    installed tools are used.

    Args:
        venv_path: Path to the virtual environment root.
        test_command: The test command to run (e.g. ``python -m pytest``).
        cwd: Working directory (typically the package repo).
        env: Additional environment variables.
        timeout: Maximum execution time in seconds.
    """
    venv_bin = venv_path / "bin"
    run_env: dict[str, str] = {
        "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
        "VIRTUAL_ENV": str(venv_path),
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        # Ensure Python doesn't pick up host site-packages.
        "PYTHONNOUSERSITE": "1",
    }
    if env:
        run_env.update(env)

    return run_timed(
        test_command,
        cwd=cwd,
        env=run_env,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Per-test timing capture
# ---------------------------------------------------------------------------


@dataclass
class TestTiming:
    """Timing for a single test phase from pytest --durations output."""

    test_id: str  # e.g. "tests/test_foo.py::test_heavy_computation"
    phase: str  # "setup", "call", "teardown"
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "test_id": self.test_id,
            "phase": self.phase,
            "duration_s": round(self.duration_s, 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestTiming:
        """Deserialize from a dict, ignoring unknown fields."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PerTestTimings:
    """All test timings parsed from a single pytest run's output."""

    timings: list[TestTiming] = field(default_factory=list)
    parse_success: bool = True
    raw_output: str = ""  # Preserved for debugging (not serialized by default)

    @property
    def by_test(self) -> dict[str, dict[str, float]]:
        """Group by test_id -> {phase: duration_s}."""
        result: dict[str, dict[str, float]] = {}
        for t in self.timings:
            result.setdefault(t.test_id, {})[t.phase] = t.duration_s
        return result

    @property
    def slowest_tests(self) -> list[tuple[str, float]]:
        """Top tests by call duration, sorted descending."""
        calls = [(t.test_id, t.duration_s) for t in self.timings if t.phase == "call"]
        return sorted(calls, key=lambda x: x[1], reverse=True)

    @property
    def total_test_time_s(self) -> float:
        """Sum of all call durations."""
        return sum(t.duration_s for t in self.timings if t.phase == "call")

    @property
    def test_count(self) -> int:
        """Number of unique test IDs."""
        return len({t.test_id for t in self.timings})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "timings": [t.to_dict() for t in self.timings],
            "parse_success": self.parse_success,
            # raw_output intentionally omitted to save space
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerTestTimings:
        """Deserialize from a dict."""
        return cls(
            timings=[TestTiming.from_dict(t) for t in data.get("timings", [])],
            parse_success=data.get("parse_success", True),
        )


# Header pattern for pytest --durations output.
_DURATIONS_HEADER_RE = re.compile(
    r"={3,}\s*slowest\s+(test\s+)?durations\s*={3,}",
    re.IGNORECASE,
)

# Line pattern: "1.23s call     tests/test_foo.py::test_heavy"
_TIMING_LINE_RE = re.compile(
    r"^\s*(\d+\.\d+)s\s+(setup|call|teardown)\s+(.+)$",
)


def parse_pytest_durations(output: str) -> PerTestTimings:
    """Parse pytest --durations=0 output into structured test timings.

    Expects the standard pytest format::

        ===== slowest durations =====
        1.23s call     tests/test_foo.py::test_heavy
        0.45s call     tests/test_bar.py::test_network
        0.12s setup    tests/test_foo.py::test_heavy
        0.01s teardown tests/test_foo.py::test_heavy

    The parser is lenient: if the durations section cannot be found or
    individual lines don't match the expected format, parse_success is
    set to False and whatever was successfully parsed is returned.
    Never raises -- a failed parse should not fail a benchmark.

    Args:
        output: The combined stdout from a pytest run.

    Returns:
        PerTestTimings with parsed data and success flag.
    """
    timings: list[TestTiming] = []
    parse_success = True

    # Find the durations header.
    lines = output.splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        if _DURATIONS_HEADER_RE.search(line):
            header_idx = i
            break

    if header_idx < 0:
        return PerTestTimings(
            timings=[],
            parse_success=False,
            raw_output=output,
        )

    # Parse lines after the header until the next === section or end.
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("==="):
            break
        match = _TIMING_LINE_RE.match(stripped)
        if match:
            duration = float(match.group(1))
            phase = match.group(2)
            test_id = match.group(3).strip()
            timings.append(TestTiming(test_id=test_id, phase=phase, duration_s=duration))
        # Non-matching lines are silently skipped.

    return PerTestTimings(
        timings=timings,
        parse_success=parse_success,
        raw_output=output,
    )


def prepare_per_test_command(
    test_command: str,
    test_framework: str,
) -> tuple[str, bool]:
    """Prepare a test command for per-test timing capture.

    Appends --durations=0 to pytest commands. Returns the original
    command unchanged for non-pytest frameworks.

    Args:
        test_command: The original test command.
        test_framework: From the registry ("pytest" or "unittest").

    Returns:
        Tuple of (modified_command, per_test_enabled). per_test_enabled
        is True only if --durations=0 was successfully added.
    """
    if test_framework != "pytest":
        return (test_command, False)
    if "--durations" in test_command:
        return (test_command, False)
    return (f"{test_command} --durations=0", True)
