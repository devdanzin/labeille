"""Run summary formatting for labeille.

Provides functions to format test run results into human-readable summaries
with per-package tables, timing statistics, and crash details.
"""

from __future__ import annotations

import signal as signal_module
from pathlib import Path

from labeille.runner import PackageResult, RunnerConfig, RunSummary

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATUS_ORDER: dict[str, int] = {
    "crash": 0,
    "timeout": 1,
    "fail": 2,
    "install_error": 3,
    "clone_error": 4,
    "error": 5,
    "pass": 6,
}

_STATUS_DISPLAY: dict[str, str] = {
    "crash": "\u2717 CRASH",
    "timeout": "\u23f1 TIMEOUT",
    "fail": "\u2717 FAIL",
    "install_error": "\u26a0 ERROR",
    "clone_error": "\u26a0 ERROR",
    "error": "\u26a0 ERROR",
    "pass": "\u2713 PASS",
}

_SEPARATOR = "\u2500" * 84


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Returns strings like ``"3s"``, ``"2m  5s"``, or ``"1h 12m 34s"``.
    """
    total = int(seconds)
    if total >= 3600:
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h}h {m:2d}m {s:2d}s"
    if total >= 60:
        m = total // 60
        s = total % 60
        return f"{m}m {s:2d}s"
    return f"{total}s"


def _signal_name(sig: int | None) -> str:
    """Convert a signal number to its name (e.g. 11 -> ``"SIGSEGV"``)."""
    if sig is None:
        return ""
    try:
        return signal_module.Signals(sig).name
    except (ValueError, AttributeError):
        return f"SIG{sig}"


def _detail(result: PackageResult) -> str:
    """Build a brief detail string for a single result."""
    if result.status == "crash":
        return (result.crash_signature or "")[:60]
    if result.status == "timeout":
        return f"(timeout: {int(result.duration_seconds)}s)"
    if result.status == "fail":
        return f"exit code {result.exit_code}"
    if result.status in ("install_error", "clone_error", "error"):
        return (result.error_message or "")[:60]
    return ""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _format_run_header(
    run_id: str,
    python_version: str,
    jit_enabled: bool,
    total_duration: float,
) -> str:
    """Format the run header section."""
    return "\n".join(
        [
            f"Run ID: {run_id}",
            f"Target Python: {python_version}",
            f"JIT enabled: {'yes' if jit_enabled else 'no'}",
            f"Duration: {format_duration(total_duration)}",
        ]
    )


def _format_package_table(
    results: list[PackageResult],
    *,
    show_passing: bool = True,
) -> str:
    """Format the per-package results table.

    When *show_passing* is ``False``, rows with status ``"pass"`` are omitted.
    """
    if not results:
        return ""

    rows = results if show_passing else [r for r in results if r.status != "pass"]
    if not rows:
        return ""

    rows = sorted(rows, key=lambda r: (_STATUS_ORDER.get(r.status, 99), r.package))

    header = f"  {'Package':<17s}{'Status':<13s}{'Duration':>10s}  {'Signal':<8s} {'Detail'}"
    lines: list[str] = [
        f"\u2500\u2500\u2500 Results {'\u2500' * 75}",
        header,
    ]

    for r in rows:
        pkg_name = r.package[:16]
        status = _STATUS_DISPLAY.get(r.status, r.status)
        dur = format_duration(r.duration_seconds)
        sig = _signal_name(r.signal)
        detail = _detail(r)
        if len(detail) > 50:
            detail = detail[:49] + "\u2026"
        lines.append(f"  {pkg_name:<17s}{status:<13s}{dur:>10s}  {sig:<8s} {detail}")

    lines.append(_SEPARATOR)
    return "\n".join(lines)


def _format_aggregate(
    results: list[PackageResult],
    summary: RunSummary,
    total_duration: float,
) -> str:
    """Format the aggregate summary with timing stats."""
    durations = [r.duration_seconds for r in results]
    avg_dur = sum(durations) / len(durations) if durations else 0.0

    fastest = min(results, key=lambda r: r.duration_seconds) if results else None
    slowest = max(results, key=lambda r: r.duration_seconds) if results else None

    def pct(n: int) -> str:
        if summary.tested == 0:
            return "0.0%"
        return f"{n / summary.tested * 100:.1f}%"

    left = [
        f"Packages tested: {summary.tested} / {summary.total}",
        f"  Passed:        {summary.passed:3d} ({pct(summary.passed)})",
        f"  Failed:        {summary.failed:3d} ({pct(summary.failed)})",
        f"  Crashed:       {summary.crashed:3d} ({pct(summary.crashed)})",
        f"  Timed out:     {summary.timed_out:3d}",
        f"  Install errors:{summary.install_errors:3d}",
        f"  Clone errors:  {summary.clone_errors:3d}",
        f"  Other errors:  {summary.errors:3d}",
        f"Skipped:         {summary.skipped:3d}",
    ]

    right: list[str] = [
        f"Total time: {format_duration(total_duration)}",
        f"Avg per package: {format_duration(avg_dur)}",
    ]
    if fastest:
        right.append(f"Fastest: {fastest.package} ({format_duration(fastest.duration_seconds)})")
    if slowest:
        right.append(f"Slowest: {slowest.package} ({format_duration(slowest.duration_seconds)})")

    pad = max(len(line) for line in left) + 4
    lines: list[str] = []
    for i in range(max(len(left), len(right))):
        l_text = left[i] if i < len(left) else ""
        r_text = right[i] if i < len(right) else ""
        lines.append(f"{l_text:<{pad}s}{r_text}")

    return "\n".join(lines)


def _format_crash_detail(
    results: list[PackageResult],
    run_dir: Path | None = None,
) -> str:
    """Format the detailed crash section."""
    crashes = [r for r in results if r.status == "crash"]
    if not crashes:
        return ""

    lines: list[str] = [f"\u2500\u2500\u2500 Crashes {'\u2500' * 75}"]
    for r in crashes:
        sig = _signal_name(r.signal)
        signature = r.crash_signature or "unknown"
        lines.append(f"  {r.package}: {sig}: {signature}")
        if run_dir:
            lines.append(f"    Stderr: {run_dir / 'crashes' / f'{r.package}.stderr'}")
        lines.append(f"    Test command: {r.test_command}")
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()
    lines.append(_SEPARATOR)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_summary(
    results: list[PackageResult],
    summary: RunSummary,
    config: RunnerConfig,
    python_version: str,
    jit_enabled: bool,
    total_duration: float,
    run_dir: Path | None = None,
    mode: str = "default",
) -> str:
    """Format a complete run summary.

    Args:
        results: List of per-package results.
        summary: Aggregate summary statistics.
        config: Runner configuration (used for run_id).
        python_version: Target Python version string.
        jit_enabled: Whether the JIT was enabled.
        total_duration: Total wall-clock duration in seconds.
        run_dir: Path to the run directory (for crash stderr paths).
        mode: Output mode -- ``"verbose"``, ``"default"``, or ``"quiet"``.

    Returns:
        The formatted summary string.
    """
    if mode == "quiet":
        if summary.crashed == 0:
            return ""
        parts: list[str] = []
        crash_detail = _format_crash_detail(results, run_dir)
        if crash_detail:
            parts.append(crash_detail)
        crash_word = "crash" if summary.crashed == 1 else "crashes"
        parts.append(
            f"{summary.crashed} {crash_word} found "
            f"in {summary.tested} packages tested ({format_duration(total_duration)})"
        )
        return "\n".join(parts)

    parts = []

    # Run header
    parts.append("")
    parts.append(_format_run_header(config.run_id, python_version, jit_enabled, total_duration))
    parts.append("")

    # Package table
    show_passing = mode == "verbose"
    table = _format_package_table(results, show_passing=show_passing)
    if table:
        parts.append(table)
        parts.append("")

    # Aggregate summary
    parts.append(_format_aggregate(results, summary, total_duration))

    # Crash detail
    crash_detail = _format_crash_detail(results, run_dir)
    if crash_detail:
        parts.append("")
        parts.append(crash_detail)

    return "\n".join(parts)
