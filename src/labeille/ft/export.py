"""Export free-threading results for external analysis and sharing.

Supports CSV (for pandas/R/Excel), JSON (for programmatic use),
and markdown (for reports and documentation).
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

log = logging.getLogger("labeille")


def export_csv(
    results: list[Any],
) -> str:
    """Export results as CSV with one row per package.

    Columns include all key metrics: category, pass rate, crash
    count, deadlock count, extension status, TSAN warnings, etc.

    Returns the CSV as a string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header.
    writer.writerow(
        [
            "package",
            "category",
            "pass_rate",
            "iterations_completed",
            "pass_count",
            "fail_count",
            "crash_count",
            "deadlock_count",
            "timeout_count",
            "tsan_warning_iterations",
            "mean_duration_s",
            "is_pure_python",
            "gil_fallback_active",
            "install_ok",
            "import_ok",
            "flaky_test_count",
            "failure_signatures",
            "tsan_warning_types",
            "commit",
        ]
    )

    for r in sorted(results, key=lambda r: r.package):
        ext = r.extension_compat or {}
        writer.writerow(
            [
                r.package,
                r.category.value,
                round(r.pass_rate, 4),
                r.iterations_completed,
                r.pass_count,
                r.fail_count,
                r.crash_count,
                r.deadlock_count,
                r.timeout_count,
                r.tsan_warning_iterations,
                round(r.mean_duration_s, 2),
                ext.get("is_pure_python", True),
                ext.get("gil_fallback_active", False),
                r.install_ok,
                r.import_ok,
                len(r.flaky_tests),
                "; ".join(r.failure_signatures),
                "; ".join(r.tsan_warning_types),
                r.commit or "",
            ]
        )

    return output.getvalue()


def export_json(
    results: list[Any],
    *,
    indent: int = 2,
) -> str:
    """Export results as JSON array of package objects.

    Includes full iteration detail for programmatic analysis.
    """
    return json.dumps(
        [r.to_dict() for r in results],
        indent=indent,
    )


def generate_report(
    meta: Any,
    results: list[Any],
    *,
    format: str = "markdown",
) -> str:
    """Generate a comprehensive compatibility report.

    Produces a full report suitable for sharing with the CPython
    core team or the free-threading compatibility tracker.

    Args:
        meta: Run metadata.
        results: Package results.
        format: "markdown" or "text".

    Returns:
        The formatted report as a string.
    """
    from labeille.ft.analysis import analyze_ft_run
    from labeille.ft.results import FTRunSummary

    summary = FTRunSummary.compute(results)
    analysis = analyze_ft_run(results)

    lines: list[str] = []

    if format == "markdown":
        lines.extend(_report_header_md(meta, summary))
        lines.extend(_report_summary_table_md(summary))
        lines.extend(_report_crashes_md(results))
        lines.extend(_report_deadlocks_md(results))
        lines.extend(_report_intermittent_md(results, analysis))
        lines.extend(_report_tsan_md(results))
        lines.extend(_report_extensions_md(results))
        lines.extend(_report_footer_md(meta))
    else:
        # Plain text falls back to the display module formatters.
        from labeille.ft.display import (
            format_compatibility_summary,
            format_package_table,
            format_triage_list,
        )

        lines.append(format_compatibility_summary(summary.to_dict()))
        lines.append("")
        lines.append(format_package_table(results))
        if analysis.triage:
            lines.append("")
            lines.append(format_triage_list(analysis.triage))

    return "\n".join(lines)


def _report_header_md(meta: Any, summary: Any) -> list[str]:
    lines = [
        "# Free-Threading Compatibility Report",
        "",
    ]

    py = meta.python_profile or {}
    sys_p = meta.system_profile or {}

    lines.append(f"**Date:** {meta.timestamp}")
    lines.append(f"**Python:** {py.get('version', '?')} ({py.get('implementation', '?')})")
    if py.get("gil_disabled"):
        lines.append("**Mode:** Free-threaded (GIL disabled)")
    lines.append(
        f"**System:** {sys_p.get('cpu_model', '?')}, "
        f"{sys_p.get('ram_total_gb', 0):.0f}GB RAM, "
        f"{sys_p.get('os_distro', '?')}"
    )

    config = meta.config or {}
    lines.append(f"**Iterations:** {config.get('iterations', '?')} per package")
    lines.append(f"**Packages tested:** {summary.total_packages}")
    lines.append("")

    return lines


def _report_summary_table_md(summary: Any) -> list[str]:
    lines = [
        "## Summary",
        "",
        "| Category | Count | Percentage |",
        "|----------|------:|------------|",
    ]

    categories = summary.categories
    total = summary.total_packages or 1

    display_order = [
        ("compatible", "Compatible"),
        ("compatible_gil_fallback", "Compatible (GIL fallback)"),
        ("tsan_warnings", "TSAN warnings"),
        ("intermittent", "Intermittent"),
        ("incompatible", "Incompatible"),
        ("crash", "Crash"),
        ("deadlock", "Deadlock"),
        ("install_failure", "Install failure"),
        ("import_failure", "Import failure"),
    ]

    for cat_value, label in display_order:
        count = categories.get(cat_value, 0)
        if count > 0:
            pct = f"{count / total * 100:.1f}%"
            lines.append(f"| {label} | {count} | {pct} |")

    lines.append("")
    return lines


def _report_crashes_md(results: list[Any]) -> list[str]:
    from labeille.ft.results import FailureCategory

    crash_pkgs = [r for r in results if r.category == FailureCategory.CRASH]
    if not crash_pkgs:
        return []

    lines = [
        "## Packages with crashes",
        "",
        "These packages trigger crashes under free-threading, "
        "likely indicating thread-safety bugs in C extensions.",
        "",
        "| Package | Pass rate | Crash count | Signature |",
        "|---------|-----------|-------------|-----------|",
    ]

    for r in sorted(crash_pkgs, key=lambda r: r.crash_count, reverse=True):
        sig = r.failure_signatures[0][:50] if r.failure_signatures else ""
        rate = f"{r.pass_count}/{r.iterations_completed}"
        lines.append(f"| {r.package} | {rate} | {r.crash_count} | {sig} |")

    lines.append("")
    return lines


def _report_deadlocks_md(results: list[Any]) -> list[str]:
    from labeille.ft.results import FailureCategory

    deadlock_pkgs = [r for r in results if r.category == FailureCategory.DEADLOCK]
    if not deadlock_pkgs:
        return []

    lines = [
        "## Packages with deadlocks",
        "",
        "These packages hang (no output progress) under free-threading.",
        "",
        "| Package | Deadlock count | Last output |",
        "|---------|----------------|-------------|",
    ]

    for r in sorted(deadlock_pkgs, key=lambda r: r.package):
        last = ""
        for it in reversed(r.iterations):
            if it.last_output_line:
                last = it.last_output_line[:50]
                break
        lines.append(f"| {r.package} | {r.deadlock_count} | {last} |")

    lines.append("")
    return lines


def _report_intermittent_md(results: list[Any], analysis: Any) -> list[str]:
    from labeille.ft.results import FailureCategory

    intermittent = [r for r in results if r.category == FailureCategory.INTERMITTENT]
    if not intermittent:
        return []

    lines = [
        "## Packages with intermittent failures",
        "",
        "These packages have non-deterministic test failures, likely due to race conditions.",
        "",
        "| Package | Pass rate | Flaky tests | Pattern |",
        "|---------|-----------|-------------|---------|",
    ]

    for r in sorted(intermittent, key=lambda r: r.pass_rate):
        rate = f"{r.pass_count}/{r.iterations_completed}"
        flaky_count = len(r.flaky_tests)
        # Find matching profile.
        pattern = "?"
        for p in analysis.flaky_profiles:
            if p.package == r.package:
                pattern = p.pattern
                break
        lines.append(f"| {r.package} | {rate} | {flaky_count} | {pattern} |")

    lines.append("")
    return lines


def _report_tsan_md(results: list[Any]) -> list[str]:
    tsan_pkgs = [r for r in results if r.tsan_warning_iterations > 0]
    if not tsan_pkgs:
        return []

    lines = [
        "## TSAN warnings",
        "",
        "These packages produce ThreadSanitizer warnings, "
        "indicating data races even when tests pass.",
        "",
        "| Package | Warning types | Iterations affected |",
        "|---------|---------------|---------------------|",
    ]

    for r in sorted(tsan_pkgs, key=lambda r: r.package):
        types = ", ".join(r.tsan_warning_types[:3])
        lines.append(
            f"| {r.package} | {types} | {r.tsan_warning_iterations}/{r.iterations_completed} |"
        )

    lines.append("")
    return lines


def _report_extensions_md(results: list[Any]) -> list[str]:
    ext_pkgs = [
        r
        for r in results
        if r.extension_compat and not r.extension_compat.get("is_pure_python", True)
    ]
    if not ext_pkgs:
        return []

    lines = [
        "## C extension compatibility",
        "",
        "| Package | GIL fallback | Source Py_mod_gil | Status |",
        "|---------|-------------|-------------------|--------|",
    ]

    for r in sorted(ext_pkgs, key=lambda r: r.package):
        ext = r.extension_compat
        fallback = "Yes" if ext.get("gil_fallback_active") else "No"
        scan = ext.get("source_scan", {})
        if scan and scan.get("declarations"):
            all_not_used = all(d.get("is_not_used", False) for d in scan["declarations"])
            source = "Declared (NOT_USED)" if all_not_used else "Declared (USED)"
        else:
            source = "Not declared"
        lines.append(f"| {r.package} | {fallback} | {source} | {r.category.value} |")

    lines.append("")
    return lines


def _report_footer_md(meta: Any) -> list[str]:
    lines = [
        "---",
        "",
        f"*Generated by labeille {meta.timestamp}*",
        "",
        "*Report format: [py-free-threading.github.io]"
        "(https://py-free-threading.github.io/) compatible*",
    ]
    return lines
