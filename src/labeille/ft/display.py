"""Terminal display formatting for free-threading results.

Provides functions to format compatibility summaries, flakiness
profiles, triage lists, GIL comparison tables, and progress output
for terminal display.
"""

from __future__ import annotations

from typing import Any

from labeille.ft.results import FailureCategory


def format_compatibility_summary(
    summary: dict[str, Any],
    *,
    python_info: str = "",
    system_info: str = "",
    install_from: str = "",
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
    if install_from == "sdist":
        lines.append("Install source: sdist")
    if python_info or system_info or install_from == "sdist":
        lines.append("")

    total = summary.get("total_packages", 0)
    categories = summary.get("categories", {})

    lines.append("Compatibility Summary")
    lines.append("─────────────────────")

    # Display categories in severity order.
    display_order = [
        ("compatible", "✓", "Compatible"),
        ("compatible_by_wheel", "⊕", "Compatible (FT wheel)"),
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
        if r.category == FailureCategory.COMPATIBLE_BY_WHEEL:
            details.append(f"FT wheel ({r.ft_wheel_version or '?'})")
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

    if truncated and max_rows is not None:
        lines.append(f"  ... ({total - max_rows} more)")

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


def format_ft_comparison(
    comp: Any,
    *,
    label_a: str = "",
    label_b: str = "",
) -> str:
    """Format a cross-run comparison for terminal display.

    Output::

        Free-Threading Compatibility Changes: 3.14a1 -> 3.14b2
        ══════════════════════════════════════════════════════

        Improvements (15):
          numpy:       crash -> compatible (pass rate 70% -> 100%)
          ...

        Regressions (3):
          cryptography: compatible -> crash (new crashes)
          ...

        Unchanged: 240 packages

        Aggregate:
          Compatible: 245 -> 260 (+15)
          Crashes: 7 -> 4 (-3)
    """
    a_label = label_a or comp.label_a
    b_label = label_b or comp.label_b

    lines = [
        f"Free-Threading Compatibility Changes: {a_label} → {b_label}",
        "═" * 60,
        "",
    ]

    # Improvements.
    if comp.improvements:
        lines.append(f"Improvements ({len(comp.improvements)}):")
        for t in comp.improvements:
            lines.append(f"  {t.package:<20s} {t.detail}")
        lines.append("")

    # Regressions.
    if comp.regressions:
        lines.append(f"Regressions ({len(comp.regressions)}):")
        for t in comp.regressions:
            lines.append(f"  {t.package:<20s} {t.detail}")
        lines.append("")

    lines.append(f"Unchanged: {comp.unchanged} packages")

    # New / removed packages.
    if comp.packages_only_in_b:
        lines.append(f"New packages in {b_label}: {len(comp.packages_only_in_b)}")
    if comp.packages_only_in_a:
        lines.append(f"Removed from {b_label}: {len(comp.packages_only_in_a)}")

    # Aggregate.
    lines.append("")
    lines.append("Aggregate:")
    net = comp.net_improvement
    net_str = f"+{net}" if net > 0 else str(net)
    lines.append(
        f"  Compatible: {comp.compatible_count_a} → {comp.compatible_count_b} ({net_str})"
    )
    crash_delta = comp.crash_count_b - comp.crash_count_a
    crash_str = f"+{crash_delta}" if crash_delta > 0 else str(crash_delta)
    lines.append(f"  Crashes:    {comp.crash_count_a} → {comp.crash_count_b} ({crash_str})")

    return "\n".join(lines)
