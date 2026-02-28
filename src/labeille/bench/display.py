"""Terminal display formatting for benchmark results.

Produces clean, aligned tables and summaries for benchmark data.
Uses Unicode box-drawing characters for visual structure.
No external dependencies.
"""

from __future__ import annotations

import math
import statistics as _stats

from labeille.bench.compare import ComparisonReport, compare_conditions
from labeille.bench.results import (
    BenchMeta,
    BenchPackageResult,
)
from labeille.bench.stats import (
    compute_overhead,
)


# ---------------------------------------------------------------------------
# Table formatting utilities
# ---------------------------------------------------------------------------


def _format_time(seconds: float, precision: int = 2) -> str:
    """Format a time value with adaptive units."""
    if math.isnan(seconds):
        return "N/A"
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}\u00b5s"
    if seconds < 1:
        return f"{seconds * 1000:.{precision}f}ms"
    if seconds < 60:
        return f"{seconds:.{precision}f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:.0f}s"


def _format_pct(value: float, precision: int = 1) -> str:
    """Format a percentage with sign."""
    if math.isnan(value):
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{precision}f}%"


def _format_ci(lower: float, upper: float, precision: int = 1) -> str:
    """Format a confidence interval."""
    if math.isnan(lower) or math.isnan(upper):
        return "[N/A]"
    return f"[{_format_pct(lower * 100, precision)}, {_format_pct(upper * 100, precision)}]"


def _significance_marker(stars: str) -> str:
    """Format significance for display."""
    return stars if stars != "ns" else "ns"


# ---------------------------------------------------------------------------
# Single benchmark display
# ---------------------------------------------------------------------------


def format_bench_show(
    meta: BenchMeta,
    results: list[BenchPackageResult],
) -> str:
    """Format a complete benchmark run for display.

    Shows system info, conditions, per-package summaries, and
    measurement quality.

    Args:
        meta: BenchMeta instance.
        results: List of BenchPackageResult.

    Returns:
        Formatted string for terminal output.
    """
    from labeille.bench.system import (
        PythonProfile,
        format_python_profile,
        format_system_profile,
    )

    lines: list[str] = []

    # Header.
    title = meta.name or meta.bench_id
    lines.append(title)
    lines.append("\u2500" * len(title))
    if meta.description:
        lines.append(meta.description)
    lines.append("")

    # System info.
    lines.append(format_system_profile(meta.system))
    lines.append("")

    # Python profiles.
    for name, pp in meta.python_profiles.items():
        if isinstance(pp, dict):
            pp = PythonProfile.from_dict(pp)
        lines.append(f"Condition '{name}':")
        lines.append(f"  {format_python_profile(pp).replace(chr(10), chr(10) + '  ')}")
    lines.append("")

    # Config.
    cfg = meta.config
    lines.append(
        f"Iterations: {cfg.get('iterations', '?')} measured + {cfg.get('warmup', '?')} warmup"
    )
    strategy = "alternating" if cfg.get("alternate") else "block"
    if cfg.get("interleave"):
        strategy = "interleaved"
    lines.append(f"Strategy: {strategy}")
    lines.append(f"Packages: {meta.packages_completed} completed, {meta.packages_skipped} skipped")
    if meta.start_time and meta.end_time:
        lines.append(f"Time: {meta.start_time} \u2192 {meta.end_time}")
    lines.append("")

    # Condition names.
    condition_names = list(meta.conditions.keys())

    if len(condition_names) == 1:
        lines.append(_format_single_condition_table(results, condition_names[0]))
    else:
        lines.append(_format_multi_condition_table(results, condition_names))

    # Measurement quality.
    lines.append("")
    lines.append(_format_quality_summary(results, condition_names))

    return "\n".join(lines)


def _format_single_condition_table(
    results: list[BenchPackageResult],
    condition_name: str,
) -> str:
    """Format a table for a single-condition benchmark."""
    lines: list[str] = []

    lines.append(
        f"{'Package':<30s} {'Wall (s)':>10s} {'\u00b1':>8s} "
        f"{'CPU (s)':>10s} {'RSS (MB)':>10s} {'CV':>6s} {'Status':>8s}"
    )
    lines.append("\u2500" * 88)

    active = [r for r in results if not r.skipped]
    active.sort(key=lambda r: _get_median_wall(r, condition_name), reverse=True)

    for r in active:
        cond = r.conditions.get(condition_name)
        if not cond or not cond.wall_time_stats:
            continue

        ws = cond.wall_time_stats
        us = cond.user_time_stats
        rs = cond.peak_rss_stats
        cv_str = f"{ws.cv:.3f}" if ws.cv < 1 else f"{ws.cv:.1f}"

        # Determine predominant status.
        statuses = [it.status for it in cond.measured_iterations]
        status = max(set(statuses), key=statuses.count) if statuses else "N/A"

        lines.append(
            f"{r.package:<30s} {ws.median:>10.2f} {ws.stdev:>7.2f}s "
            f"{us.mean if us else 0:>10.2f} "
            f"{rs.median if rs else 0:>10.1f} "
            f"{cv_str:>6s} {status:>8s}"
        )

    return "\n".join(lines)


def _format_multi_condition_table(
    results: list[BenchPackageResult],
    condition_names: list[str],
) -> str:
    """Format a comparison table for multi-condition benchmarks."""
    lines: list[str] = []

    baseline_name = condition_names[0]
    treatment_names = condition_names[1:]

    # Build header.
    header = f"{'Package':<25s} {baseline_name + ' (s)':>12s}"
    for tn in treatment_names:
        header += f" {tn + ' (s)':>12s} {'Overhead':>10s} {'CI (95%)':>20s} {'Sig.':>5s}"
    lines.append(header)
    lines.append("\u2500" * len(header))

    active = [r for r in results if not r.skipped]
    active.sort(key=lambda r: _get_median_wall(r, baseline_name), reverse=True)

    for r in active:
        base_cond = r.conditions.get(baseline_name)
        if not base_cond or not base_cond.wall_time_stats:
            continue

        row = f"{r.package:<25s} {base_cond.wall_time_stats.median:>11.2f}s"

        for tn in treatment_names:
            treat_cond = r.conditions.get(tn)
            if not treat_cond or not treat_cond.wall_time_stats:
                row += f" {'N/A':>12s} {'':>10s} {'':>20s} {'':>5s}"
                continue

            overhead = compute_overhead(
                base_cond.wall_times,
                treat_cond.wall_times,
                ci_seed=42,
            )

            base_median = base_cond.wall_time_stats.median
            if base_median > 0:
                ci_lower_frac = overhead.ci.lower / base_median
                ci_upper_frac = overhead.ci.upper / base_median
            else:
                ci_lower_frac = 0.0
                ci_upper_frac = 0.0

            row += (
                f" {treat_cond.wall_time_stats.median:>11.2f}s"
                f" {_format_pct(overhead.overhead_pct):>10s}"
                f" {_format_ci(ci_lower_frac, ci_upper_frac):>20s}"
                f" {_significance_marker(overhead.ttest.significance_stars):>5s}"
            )

        lines.append(row)

    return "\n".join(lines)


def _get_median_wall(
    result: BenchPackageResult,
    condition: str,
) -> float:
    """Get median wall time for sorting. Returns 0 if unavailable."""
    cond = result.conditions.get(condition)
    if cond and cond.wall_time_stats:
        return cond.wall_time_stats.median
    return 0.0


# ---------------------------------------------------------------------------
# Quality summary
# ---------------------------------------------------------------------------


def _format_quality_summary(
    results: list[BenchPackageResult],
    condition_names: list[str],
) -> str:
    """Format measurement quality metrics."""
    lines = [
        "Measurement Quality",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    ]

    for cond_name in condition_names:
        cvs: list[float] = []
        outlier_count = 0
        total_iterations = 0
        max_load = 0.0

        for r in results:
            if r.skipped:
                continue
            cond = r.conditions.get(cond_name)
            if not cond or not cond.wall_time_stats:
                continue
            cvs.append(cond.wall_time_stats.cv)
            outlier_count += cond.n_outliers
            total_iterations += len(cond.measured_iterations)
            for it in cond.measured_iterations:
                max_load = max(max_load, it.load_avg_start, it.load_avg_end)

        if not cvs:
            continue

        median_cv = _stats.median(cvs)
        high_cv = sum(1 for cv in cvs if cv > 0.10)

        lines.append(f"  {cond_name}:")
        lines.append(f"    Median CV:         {median_cv:.3f}")
        if high_cv:
            lines.append(f"    Packages with CV > 10%: {high_cv}")
        lines.append(f"    Outliers:          {outlier_count} / {total_iterations} iterations")
        lines.append(f"    Max system load:   {max_load:.1f}")

    # Overall quality assessment.
    all_cvs: list[float] = []
    for r in results:
        for cond in r.conditions.values():
            if cond.wall_time_stats:
                all_cvs.append(cond.wall_time_stats.cv)
    if all_cvs:
        overall_cv = _stats.median(all_cvs)
        if overall_cv < 0.03:
            quality = "Excellent (CV < 3%)"
        elif overall_cv < 0.05:
            quality = "Good (CV < 5%)"
        elif overall_cv < 0.10:
            quality = "Acceptable (CV < 10%)"
        else:
            quality = "Poor (CV \u2265 10%) \u2014 results may be unreliable"
        lines.append(f"  Overall: {quality}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------


def format_comparison_summary(
    results: list[BenchPackageResult],
    baseline_name: str,
    treatment_name: str,
) -> str:
    """Format an aggregate comparison summary.

    Delegates to compare_conditions for analysis, then formats
    the resulting ComparisonReport.

    Args:
        results: Package results containing both conditions.
        baseline_name: Name of the baseline condition.
        treatment_name: Name of the treatment condition.

    Returns:
        Formatted summary string.
    """
    report = compare_conditions(results, baseline_name, treatment_name)
    return format_comparison_report(report)


def format_comparison_report(report: ComparisonReport) -> str:
    """Format a ComparisonReport for terminal display."""
    if report.total_packages == 0:
        return "No comparable packages found."

    lines: list[str] = []
    lines.append("Aggregate Summary")
    lines.append("\u2500" * 17)
    lines.append(f"  Packages compared:       {report.total_packages}")
    lines.append(f"    Reliable measurements: {report.reliable_packages}")
    if report.unreliable_packages:
        lines.append(f"    Unreliable (high CV):  {report.unreliable_packages}")
    lines.append(f"  Median overhead:         {_format_pct(report.median_overhead_pct)}")
    lines.append(f"  Mean overhead:           {_format_pct(report.mean_overhead_pct)}")
    lines.append(
        f"  Range:                   "
        f"{_format_pct(report.min_overhead_pct)} to "
        f"{_format_pct(report.max_overhead_pct)}"
    )
    lines.append(f"  Statistically significant: {report.significant_packages}")
    lines.append(f"  Practically significant:   {report.practically_significant}")
    lines.append(f"  Faster under treatment:    {report.faster_packages}")
    lines.append(f"  Slower under treatment:    {report.slower_packages}")

    if report.most_affected:
        lines.append("")
        lines.append("  Most affected (highest overhead):")
        for p in report.most_affected:
            lines.append(f"    {p.package:<25s} {_format_pct(p.overhead.overhead_pct):>8s}")

    if report.most_improved:
        lines.append("")
        lines.append("  Most improved (fastest under treatment):")
        for p in report.most_improved:
            lines.append(f"    {p.package:<25s} {_format_pct(p.overhead.overhead_pct):>8s}")

    # Anomaly summary.
    if report.high_cv_count or report.status_mismatch_count:
        lines.append("")
        lines.append("  Anomalies:")
        if report.high_cv_count:
            lines.append(f"    High CV (>10%):        {report.high_cv_count} packages")
        if report.status_mismatch_count:
            lines.append(f"    Status mismatch:       {report.status_mismatch_count} packages")

    return "\n".join(lines)
