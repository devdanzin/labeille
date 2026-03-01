"""Terminal display formatting for free-threading results.

Provides functions to format compatibility summaries, flakiness
profiles, triage lists, GIL comparison tables, and progress output
for terminal display.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("labeille")


def format_compatibility_summary(
    summary: dict[str, Any],
    *,
    python_info: str = "",
    system_info: str = "",
) -> str:
    """Format the top-level compatibility summary.

    Output::

        Free-Threading Compatibility Report
        ════════════════════════════════════

        Python: 3.14.0b2 (free-threaded, --disable-gil)
        System: AMD Ryzen 9 7950X, 64GB RAM, Ubuntu 24.04

        Compatibility Summary
        ─────────────────────
          ✓ Compatible:              245  (70.6%)
          ⚠ Compatible (GIL fallback): 28  ( 8.1%)
          ...
    """
    lines = [
        "Free-Threading Compatibility Report",
        "════════════════════════════════════",
        "",
    ]

    if python_info:
        lines.append(python_info)
    if system_info:
        lines.append(system_info)
    if python_info or system_info:
        lines.append("")

    total = summary.get("total_packages", 0)
    categories = summary.get("categories", {})

    lines.append("Compatibility Summary")
    lines.append("─────────────────────")

    # Display categories in severity order.
    display_order = [
        ("compatible", "✓", "Compatible"),
        ("compatible_gil_fallback", "⚠", "Compatible (GIL fallback)"),
        ("tsan_warnings", "⚡", "TSAN warnings (tests pass)"),
        ("intermittent", "~", "Intermittent"),
        ("incompatible", "✗", "Incompatible"),
        ("crash", "💥", "Crash"),
        ("deadlock", "🔒", "Deadlock"),
        ("install_failure", "⚙", "Install failure"),
        ("import_failure", "📦", "Import failure"),
        ("unknown", "?", "Unknown"),
    ]

    for cat_value, symbol, label in display_order:
        count = categories.get(cat_value, 0)
        if count == 0:
            continue
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"  {symbol} {label + ':':<35s} {count:3d}  ({pct:5.1f}%)")

    lines.append(f"  {'':37s} {'─' * 3}")
    lines.append(f"  {'Total:':<37s} {total:3d}")

    # Pure Python vs extension breakdown.
    pp_count = summary.get("pure_python_count", 0)
    ext_count = summary.get("extension_count", 0)
    pp_pct = summary.get("pure_python_compatible_pct", 0)
    ext_pct = summary.get("extension_compatible_pct", 0)

    if pp_count > 0 or ext_count > 0:
        lines.append("")
        lines.append("By package type:")
        if pp_count > 0:
            lines.append(f"  Pure Python:    {pp_count:3d} packages, {pp_pct:.1f}% compatible")
        if ext_count > 0:
            lines.append(f"  C extensions:   {ext_count:3d} packages, {ext_pct:.1f}% compatible")

    return "\n".join(lines)


def format_package_table(
    results: list[Any],
    *,
    sort_by: str = "category",
    max_rows: int | None = None,
    show_iterations: bool = False,
) -> str:
    """Format a table of per-package results.

    Output::

        Package          Category        Pass Rate  Iters  Crashes  Details
        ─────────────────────────────────────────────────────────────────────
        requests         ✓ compatible     10/10      10       0
        numpy            💥 crash           7/10      10       3     SIGSEGV
        aiohttp          ~ intermittent    7/10      10       0     3 flaky tests
        ...
    """
    if sort_by == "category":
        results = sorted(
            results,
            key=lambda r: (r.category.severity, -r.pass_rate, r.package),
        )
    elif sort_by == "pass_rate":
        results = sorted(results, key=lambda r: (r.pass_rate, r.package))
    else:
        results = sorted(results, key=lambda r: r.package)

    if max_rows and len(results) > max_rows:
        total = len(results)
        results = results[:max_rows]
        truncated = True
    else:
        total = len(results)
        truncated = False

    # Header.
    header = (
        f"{'Package':<20s} {'Category':<25s} {'Pass Rate':>10s} "
        f"{'Iters':>6s} {'Crashes':>8s}  {'Details'}"
    )
    separator = "─" * min(len(header) + 20, 100)

    lines = [header, separator]

    for r in results:
        cat_str = f"{r.category.symbol} {r.category.value}"

        if r.iterations_completed > 0:
            rate_str = f"{r.pass_count}/{r.iterations_completed}"
        else:
            rate_str = "N/A"

        # Details column.
        details: list[str] = []
        if r.failure_signatures:
            details.append(r.failure_signatures[0][:30])
        if r.flaky_tests:
            details.append(f"{len(r.flaky_tests)} flaky tests")
        if r.tsan_warning_types:
            details.append(f"TSAN: {r.tsan_warning_types[0]}")
        if r.deadlock_count > 0:
            details.append(f"{r.deadlock_count} deadlocks")
        if not r.install_ok:
            details.append("install failed")
        if not r.import_ok:
            details.append("import failed")

        detail_str = ", ".join(details)[:50]

        lines.append(
            f"{r.package:<20s} {cat_str:<25s} {rate_str:>10s} "
            f"{r.iterations_completed:>6d} {r.crash_count:>8d}  "
            f"{detail_str}"
        )

    if truncated:
        lines.append(f"  ... ({total - max_rows} more)")  # type: ignore[operator]

    return "\n".join(lines)


def format_triage_list(
    triage: list[Any],
    *,
    max_entries: int = 20,
) -> str:
    """Format the triage priority list.

    Output::

        Investigation Priority
        ──────────────────────
         1. numpy          💥 crash       (score: 70)  3 crashes; C extension
         2. gevent         🔒 deadlock    (score: 60)  2 deadlocks
         3. aiohttp        ~ intermittent (score: 35)  severity:intermittent
         ...
    """
    from labeille.ft.results import FailureCategory

    lines = ["Investigation Priority", "──────────────────────"]

    for i, entry in enumerate(triage[:max_entries], 1):
        cat = FailureCategory(entry.category)
        lines.append(
            f"  {i:2d}. {entry.package:<18s} {cat.symbol} "
            f"{entry.category:<14s} (score: {entry.priority_score:.0f})  "
            f"{entry.reason[:60]}"
        )

    if len(triage) > max_entries:
        lines.append(f"  ... and {len(triage) - max_entries} more")

    return "\n".join(lines)


def format_flakiness_profile(profile: Any) -> str:
    """Format a single flakiness profile for display.

    Output::

        Flakiness Profile: aiohttp
        ──────────────────────────
        Pass rate:          7/10 (70.0%)
        Pattern:            consistent_test
        Failure modes:      fail: 3
        Consecutive passes: 4
        Consecutive fails:  2

        Flaky tests (fail intermittently):
          test_connector::test_close     3/5 (60.0%)
          test_client::test_timeout      1/5 (20.0%)
    """
    lines = [
        f"Flakiness Profile: {profile.package}",
        "─" * (20 + len(profile.package)),
    ]

    passes = int(profile.pass_rate * profile.total_iterations)
    lines.append(
        f"Pass rate:          {passes}/{profile.total_iterations} ({profile.pass_rate * 100:.1f}%)"
    )
    lines.append(f"Pattern:            {profile.pattern}")

    if profile.failure_modes:
        modes = ", ".join(f"{k}: {v}" for k, v in sorted(profile.failure_modes.items()))
        lines.append(f"Failure modes:      {modes}")

    lines.append(f"Consecutive passes: {profile.max_consecutive_passes}")
    lines.append(f"Consecutive fails:  {profile.max_consecutive_failures}")

    if profile.flaky_tests:
        lines.append("")
        lines.append("Flaky tests (fail intermittently):")
        for t in profile.flaky_tests[:15]:
            short_id = t.test_id
            if len(short_id) > 50:
                short_id = "..." + short_id[-47:]
            lines.append(
                f"  {short_id:<55s} {t.fail_count}/{t.total_seen} ({t.fail_rate * 100:.1f}%)"
            )
        if len(profile.flaky_tests) > 15:
            lines.append(f"  ... and {len(profile.flaky_tests) - 15} more")

    return "\n".join(lines)


def format_gil_comparison(
    comparisons: list[Any],
) -> str:
    """Format GIL-enabled vs GIL-disabled comparison table.

    Output::

        GIL Comparison (free-threaded vs GIL-enabled)
        ──────────────────────────────────────────────
        Package          GIL=0 Rate  GIL=1 Rate  Classification
        ─────────────────────────────────────────────────────────
        requests            100%        100%      ft_compatible
        numpy                70%        100%      ft_intermittent  ← FT-specific
        sqlalchemy           60%         80%      ft_exacerbated
        celery               50%         50%      pre_existing
        ...

        Summary:
          FT-specific failures: 12 packages
          Pre-existing failures: 5 packages
    """
    lines = [
        "GIL Comparison (free-threaded vs GIL-enabled)",
        "──────────────────────────────────────────────",
        f"{'Package':<20s} {'GIL=0 Rate':>11s} {'GIL=1 Rate':>11s}  {'Classification':<20s}",
        "─" * 75,
    ]

    ft_specific = 0
    pre_existing = 0

    for c in comparisons:
        ft_rate = f"{c.gil_disabled_pass_rate * 100:.0f}%"
        gil_rate = f"{c.gil_enabled_pass_rate * 100:.0f}%"
        marker = ""
        if c.free_threading_specific:
            marker = " ← FT-specific"
            ft_specific += 1
        if c.classification == "pre_existing":
            pre_existing += 1

        lines.append(
            f"{c.package:<20s} {ft_rate:>11s} {gil_rate:>11s}  {c.classification:<20s}{marker}"
        )

    lines.append("")
    lines.append("Summary:")
    lines.append(f"  FT-specific failures: {ft_specific} packages")
    lines.append(f"  Pre-existing failures: {pre_existing} packages")

    return "\n".join(lines)


def format_ft_comparison(
    comparison: Any,
    *,
    label_a: str = "run_a",
    label_b: str = "run_b",
) -> str:
    """Format a comparison between two free-threading runs.

    This is a stub that will be fully implemented in prompt 29
    when ft/compare.py is available.
    """
    lines = [
        f"Free-Threading Run Comparison: {label_a} vs {label_b}",
        "═" * 60,
    ]
    if hasattr(comparison, "improved") and hasattr(comparison, "regressed"):
        lines.append(f"  Improved:   {len(comparison.improved)} packages")
        lines.append(f"  Regressed:  {len(comparison.regressed)} packages")
        lines.append(f"  Unchanged:  {len(comparison.unchanged)} packages")
    else:
        lines.append("  (comparison data not available)")
    return "\n".join(lines)


def format_progress(
    package: str,
    iteration: int,
    total_iterations: int,
    status: str,
    duration: float,
    packages_done: int,
    packages_total: int,
) -> str:
    """Format a single-line progress update.

    Output: ``(15/350) requests iter 3/10: pass (12.3s)``
    """
    return (
        f"({packages_done}/{packages_total}) {package} "
        f"iter {iteration}/{total_iterations}: "
        f"{status} ({duration:.1f}s)"
    )
