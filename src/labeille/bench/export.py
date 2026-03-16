"""Export benchmark results to CSV and Markdown formats.

CSV format: one row per package per condition per iteration
(long format for pandas/R). This is the raw data — every single
measurement.

Markdown format: a summary table suitable for reports, README
files, and GitHub issues.
"""

from __future__ import annotations

import csv
import io

from labeille.bench.anomaly import AnomalyReport
from labeille.bench.compare import compare_conditions
from labeille.bench.results import BenchMeta, BenchPackageResult
from labeille.bench.trends import SeriesTrend


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(
    meta: BenchMeta,
    results: list[BenchPackageResult],
) -> str:
    """Export results as CSV (long format).

    One row per package x condition x iteration. Includes both
    warm-up and measured iterations (marked by the 'warmup' column).

    Columns:
        package, condition, iteration, warmup, wall_time_s,
        user_time_s, sys_time_s, peak_rss_mb, exit_code, status,
        outlier, load_avg_start, load_avg_end, ram_available_start_gb
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header.
    writer.writerow(
        [
            "package",
            "condition",
            "iteration",
            "warmup",
            "wall_time_s",
            "user_time_s",
            "sys_time_s",
            "peak_rss_mb",
            "exit_code",
            "status",
            "outlier",
            "load_avg_start",
            "load_avg_end",
            "ram_available_start_gb",
        ]
    )

    for r in sorted(results, key=lambda r: r.package):
        if r.skipped:
            continue
        for cond_name, cond in sorted(r.conditions.items()):
            for it in cond.iterations:
                writer.writerow(
                    [
                        r.package,
                        cond_name,
                        it.index,
                        it.warmup,
                        f"{it.wall_time_s:.6f}",
                        f"{it.user_time_s:.6f}",
                        f"{it.sys_time_s:.6f}",
                        f"{it.peak_rss_mb:.1f}",
                        it.exit_code,
                        it.status,
                        it.outlier,
                        f"{it.load_avg_start:.2f}",
                        f"{it.load_avg_end:.2f}",
                        f"{it.ram_available_start_gb:.2f}",
                    ]
                )

    return output.getvalue()


def export_csv_summary(
    meta: BenchMeta,
    results: list[BenchPackageResult],
) -> str:
    """Export summary statistics as CSV.

    One row per package x condition. Includes mean, median, stdev,
    min, max, CV for wall time.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "package",
            "condition",
            "n",
            "wall_mean_s",
            "wall_median_s",
            "wall_stdev_s",
            "wall_min_s",
            "wall_max_s",
            "wall_cv",
            "cpu_mean_s",
            "rss_median_mb",
            "outliers",
        ]
    )

    for r in sorted(results, key=lambda r: r.package):
        if r.skipped:
            continue
        for cond_name, cond in sorted(r.conditions.items()):
            ws = cond.wall_time_stats
            us = cond.user_time_stats
            rs = cond.peak_rss_stats
            if not ws:
                continue
            writer.writerow(
                [
                    r.package,
                    cond_name,
                    ws.n,
                    f"{ws.mean:.6f}",
                    f"{ws.median:.6f}",
                    f"{ws.stdev:.6f}",
                    f"{ws.min:.6f}",
                    f"{ws.max:.6f}",
                    f"{ws.cv:.6f}",
                    f"{us.mean:.6f}" if us else "",
                    f"{rs.median:.1f}" if rs else "",
                    cond.n_outliers,
                ]
            )

    return output.getvalue()


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def export_markdown(
    meta: BenchMeta,
    results: list[BenchPackageResult],
    *,
    anomaly_report: AnomalyReport | None = None,
) -> str:
    """Export results as a Markdown report.

    Suitable for GitHub issues, README files, and reports.

    Args:
        meta: Benchmark metadata.
        results: Package results.
        anomaly_report: Optional anomaly report to include.
    """
    lines: list[str] = []

    title = meta.name or meta.bench_id
    lines.append(f"# {title}")
    lines.append("")
    if meta.description:
        lines.append(meta.description)
        lines.append("")

    # System info.
    lines.append("## System")
    lines.append("")
    sys_p = meta.system
    lines.append(f"- **CPU:** {sys_p.cpu_model} ({sys_p.cpu_cores_physical} cores)")
    lines.append(f"- **RAM:** {sys_p.ram_total_gb:.1f} GB")
    lines.append(f"- **OS:** {sys_p.os_distro}")
    lines.append("")

    # Conditions.
    lines.append("## Conditions")
    lines.append("")
    for cond_name, cond_def in meta.conditions.items():
        desc = ""
        if hasattr(cond_def, "description"):
            desc = cond_def.description
        elif isinstance(cond_def, dict):
            desc = cond_def.get("description", "")
        lines.append(f"- **{cond_name}**: {desc}")
    lines.append("")

    # Config.
    cfg = meta.config
    lines.append(
        f"Iterations: {cfg.get('iterations', '?')} measured + {cfg.get('warmup', '?')} warmup"
    )
    lines.append("")

    condition_names = list(meta.conditions.keys())

    if len(condition_names) >= 2:
        _export_markdown_multi_condition(lines, results, condition_names)
    else:
        _export_markdown_single_condition(lines, results, condition_names[0])

    # Anomalies section.
    if anomaly_report and anomaly_report.anomalies:
        lines.append("")
        lines.append("## Anomalies")
        lines.append("")

        from labeille.bench.anomaly import SEVERITY_LABELS as severity_labels
        severity_order = ["error", "warning", "info"]
        by_severity = anomaly_report.by_severity
        for severity in severity_order:
            group = by_severity.get(severity, [])
            for a in group:
                label = severity_labels.get(a.severity, a.severity.upper())
                lines.append(
                    f"- **[{label}]** {a.package}/{a.condition}: "
                    f"{a.description} *{a.recommendation}*"
                )

    lines.append("")
    lines.append(f"*Generated by labeille bench on {meta.start_time or 'unknown'}*")

    return "\n".join(lines)


def _export_markdown_multi_condition(
    lines: list[str],
    results: list[BenchPackageResult],
    condition_names: list[str],
) -> None:
    """Append multi-condition comparison table to lines."""
    baseline = condition_names[0]
    treatment = condition_names[1]

    report = compare_conditions(results, baseline, treatment, ci_seed=42)

    lines.append("## Results")
    lines.append("")

    # Table header.
    lines.append(f"| Package | {baseline} (s) | {treatment} (s) | Overhead | 95% CI | Sig. |")
    lines.append("|---|---:|---:|---:|---|---|")

    for p in sorted(
        report.packages,
        key=lambda p: p.overhead.overhead_pct,
        reverse=True,
    ):
        bs = p.overhead.baseline_stats
        ts = p.overhead.treatment_stats
        oh = p.overhead
        if not bs or not ts:
            continue

        ci_lower_pct = oh.ci.lower / bs.median * 100 if bs.median else 0
        ci_upper_pct = oh.ci.upper / bs.median * 100 if bs.median else 0

        lines.append(
            f"| {p.package} | {bs.median:.2f} | {ts.median:.2f} | "
            f"{oh.overhead_pct:+.1f}% | "
            f"[{ci_lower_pct:+.1f}%, {ci_upper_pct:+.1f}%] | "
            f"{oh.ttest.significance_stars} |"
        )

    lines.append("")

    # Summary.
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Packages compared: {report.total_packages}")
    lines.append(f"- Median overhead: {report.median_overhead_pct:+.1f}%")
    lines.append(f"- Mean overhead: {report.mean_overhead_pct:+.1f}%")
    lines.append(f"- Range: {report.min_overhead_pct:+.1f}% to {report.max_overhead_pct:+.1f}%")
    lines.append(f"- Statistically significant (p<0.05): {report.significant_packages}")


def _export_markdown_single_condition(
    lines: list[str],
    results: list[BenchPackageResult],
    cond_name: str,
) -> None:
    """Append single-condition timing table to lines."""
    lines.append("## Results")
    lines.append("")
    lines.append("| Package | Wall (s) | +/- | CPU (s) | RSS (MB) | CV |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    active = [r for r in results if not r.skipped]

    def _sort_key(r: BenchPackageResult) -> float:
        cond = r.conditions.get(cond_name)
        if cond and cond.wall_time_stats:
            return cond.wall_time_stats.median
        return 0.0

    active.sort(key=_sort_key, reverse=True)

    for r in active:
        cond = r.conditions.get(cond_name)
        if not cond or not cond.wall_time_stats:
            continue
        ws = cond.wall_time_stats
        us = cond.user_time_stats
        rs = cond.peak_rss_stats
        cpu_val = f"{us.mean:.2f}" if us else "0.00"
        rss_val = f"{rs.median:.1f}" if rs else "0.0"
        lines.append(
            f"| {r.package} | {ws.median:.2f} | {ws.stdev:.2f} | "
            f"{cpu_val} | {rss_val} | {ws.cv:.3f} |"
        )


# ---------------------------------------------------------------------------
# Trend export
# ---------------------------------------------------------------------------


def export_trend_markdown(trend: SeriesTrend) -> str:
    """Export trend analysis as a Markdown report.

    Sections: Summary table, regression alerts, per-package trend
    table, methodology notes.
    """
    lines: list[str] = []

    lines.append(f"# Trend Analysis: {trend.series_id}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Runs:** {trend.n_runs}")
    if trend.date_range:
        lines.append(f"- **Period:** {trend.date_range[0][:19]} to {trend.date_range[1][:19]}")
    lines.append(f"- **Condition:** {trend.condition}")
    lines.append(f"- **Baseline:** {trend.baseline_bench_id}")
    lines.append(f"- **Aggregate change:** {trend.aggregate_median_change_pct:+.1f}%")
    lines.append("")

    # Summary.
    lines.append("## Classification")
    lines.append("")
    lines.append("| Category | Count | Packages |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| Regressing | {trend.n_regressing} | {', '.join(trend.regressing_packages) or '-'} |"
    )
    lines.append(
        f"| Improving | {trend.n_improving} | {', '.join(trend.improving_packages) or '-'} |"
    )
    lines.append(
        f"| Volatile | {trend.n_volatile} | {', '.join(trend.volatile_packages) or '-'} |"
    )
    lines.append(f"| Stable | {trend.n_stable} | {', '.join(trend.stable_packages) or '-'} |")
    lines.append("")

    # Alerts.
    if trend.alerts:
        lines.append("## Alerts")
        lines.append("")
        from labeille.bench.anomaly import SEVERITY_LABELS as severity_labels
        for a in trend.alerts:
            label = severity_labels.get(a.severity, a.severity.upper())
            lines.append(f"- **[{label}]** {a.package} ({a.condition}): {a.description}")
        lines.append("")

    # Per-package trend table.
    lines.append("## Per-Package Trends")
    lines.append("")
    lines.append(
        "| Package | Direction | Runs | Baseline (s) | Latest (s) | Change (%) | Slope (%/run) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for pt in sorted(
        trend.package_trends,
        key=lambda p: abs(p.cumulative_change_pct),
        reverse=True,
    ):
        baseline_s = f"{pt.medians[0]:.3f}" if pt.medians else "N/A"
        latest_s = f"{pt.medians[-1]:.3f}" if pt.medians else "N/A"
        lines.append(
            f"| {pt.package} | {pt.trend_direction} | {pt.n_runs} | "
            f"{baseline_s} | {latest_s} | "
            f"{pt.cumulative_change_pct:+.1f} | {pt.trend_pct_per_run:+.2f} |"
        )
    lines.append("")

    # Methodology.
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Trend classification uses linear regression of median wall times "
        "across runs. Sustained regression/improvement requires 3+ consecutive "
        "runs with per-run change exceeding the threshold. Volatility is flagged "
        "when the coefficient of variation (CV) increases across runs."
    )
    lines.append("")
    lines.append("*Generated by labeille bench track trend*")

    return "\n".join(lines)


def export_trend_csv(trend: SeriesTrend) -> str:
    """Export per-package trend data as CSV.

    One row per package with: package, condition, n_runs, direction,
    baseline_median, latest_median, cumulative_change_pct, trend_slope,
    recent_change_pct, sustained_regression.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "package",
            "condition",
            "n_runs",
            "direction",
            "baseline_median",
            "latest_median",
            "cumulative_change_pct",
            "trend_slope",
            "recent_change_pct",
            "sustained_regression",
        ]
    )

    for pt in sorted(trend.package_trends, key=lambda p: p.package):
        baseline_s = f"{pt.medians[0]:.6f}" if pt.medians else ""
        latest_s = f"{pt.medians[-1]:.6f}" if pt.medians else ""
        writer.writerow(
            [
                pt.package,
                pt.condition,
                pt.n_runs,
                pt.trend_direction,
                baseline_s,
                latest_s,
                f"{pt.cumulative_change_pct:.2f}",
                f"{pt.trend_slope:.6f}",
                f"{pt.recent_change_pct:.2f}",
                pt.sustained_regression,
            ]
        )

    return output.getvalue()
