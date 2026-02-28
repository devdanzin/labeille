"""Timing capture for benchmark iterations.

Measures wall-clock time, user CPU time, system CPU time, and peak
resident set size for subprocess executions.  Uses resource.getrusage
for CPU times and ``/usr/bin/time -v`` for per-process peak RSS.
"""

from __future__ import annotations

import logging
import os
import resource
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("labeille")


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
    run_env = dict(os.environ)
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
        _kill_process_group(proc.pid)
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


def _kill_process_group(pid: int) -> None:
    """Attempt to kill the entire process group on timeout."""
    import signal

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


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
        content = Path(time_output_file).read_text()
        for line in content.splitlines():
            if "Maximum resident set size" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    kb = int(parts[-1].strip())
                    return kb / 1024
    except Exception:  # noqa: BLE001
        pass
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
