"""Trend analysis and regression detection for benchmark tracking series.

Computes per-package timing trends across runs in a tracking series,
classifies packages by stability (stable, regressing, improving,
volatile), and produces regression alerts when new runs are added.

Trend classification uses configurable thresholds:
- Sustained regression: 3+ consecutive runs with median increasing > threshold
- Improving: mirror of regressing but with decreasing medians
- Volatile: CV increasing across runs (measurement becoming less reliable)
- Stable: everything else

Regression alerts compare the latest run against:
1. The previous run (immediate change)
2. The baseline (pinned or first) run (cumulative drift)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labeille.bench.results import BenchMeta, BenchPackageResult
from labeille.io_utils import dataclass_from_dict
from labeille.bench.tracking import (
    TrackingRunEntry,
    TrackingSeries,
    load_series_run,
)
from labeille.logging import get_logger

log = get_logger("bench.trends")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PackageTrend:
    """Longitudinal trend for one package across runs in a series."""

    package: str
    condition: str
    timestamps: list[str]  # ISO timestamps from each run
    medians: list[float]  # Median wall time per run
    cvs: list[float]  # CV per run (measurement quality)
    n_runs: int = 0  # Number of runs this package appeared in

    # Derived classification
    trend_direction: str = "stable"
    # One of: "stable", "regressing", "improving", "volatile"
    trend_slope: float = 0.0
    # Linear regression slope (seconds per run index)
    trend_pct_per_run: float = 0.0
    # Slope as percentage of baseline median
    recent_change_pct: float = 0.0
    # Last run vs previous run as percentage
    cumulative_change_pct: float = 0.0
    # Last run vs baseline as percentage
    sustained_regression: bool = False
    # 3+ consecutive increases > threshold
    sustained_improvement: bool = False
    # 3+ consecutive decreases > threshold
    volatility_increasing: bool = False
    # CV trend is positive (measurements getting noisier)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "package": self.package,
            "condition": self.condition,
            "n_runs": self.n_runs,
            "trend_direction": self.trend_direction,
            "trend_slope": round(self.trend_slope, 6),
            "trend_pct_per_run": round(self.trend_pct_per_run, 2),
            "recent_change_pct": round(self.recent_change_pct, 2),
            "cumulative_change_pct": round(self.cumulative_change_pct, 2),
            "sustained_regression": self.sustained_regression,
            "sustained_improvement": self.sustained_improvement,
            "volatility_increasing": self.volatility_increasing,
            "medians": [round(m, 6) for m in self.medians],
            "cvs": [round(c, 6) for c in self.cvs],
            "timestamps": self.timestamps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackageTrend:
        """Deserialize from a dict, ignoring unknown fields."""
        return dataclass_from_dict(cls, data)


@dataclass
class RegressionAlert:
    """An alert about a specific package's regression or recovery."""

    package: str
    condition: str
    alert_type: str
    # One of: "new_regression", "sustained_regression", "recovery",
    # "new_instability", "new_improvement"
    severity: str  # "error", "warning", "info"
    description: str
    recent_change_pct: float  # vs previous run
    cumulative_change_pct: float  # vs baseline
    baseline_median_s: float
    current_median_s: float
    previous_median_s: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "package": self.package,
            "condition": self.condition,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "description": self.description,
            "recent_change_pct": round(self.recent_change_pct, 2),
            "cumulative_change_pct": round(self.cumulative_change_pct, 2),
            "baseline_median_s": round(self.baseline_median_s, 6),
            "current_median_s": round(self.current_median_s, 6),
            "previous_median_s": round(self.previous_median_s, 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressionAlert:
        """Deserialize from a dict, ignoring unknown fields."""
        return dataclass_from_dict(cls, data)


@dataclass
class SeriesTrend:
    """Aggregate trend summary across a tracked series."""

    series_id: str
    n_runs: int
    date_range: tuple[str, str] | None
    condition: str  # Which condition was analyzed
    baseline_bench_id: str  # Which run is the baseline

    package_trends: list[PackageTrend] = field(default_factory=list)
    alerts: list[RegressionAlert] = field(default_factory=list)

    # Classification counts
    regressing_packages: list[str] = field(default_factory=list)
    improving_packages: list[str] = field(default_factory=list)
    volatile_packages: list[str] = field(default_factory=list)
    stable_packages: list[str] = field(default_factory=list)

    # Aggregate stats
    aggregate_median_change_pct: float = 0.0
    # Median of all packages' cumulative_change_pct

    @property
    def n_regressing(self) -> int:
        """Number of regressing packages."""
        return len(self.regressing_packages)

    @property
    def n_improving(self) -> int:
        """Number of improving packages."""
        return len(self.improving_packages)

    @property
    def n_volatile(self) -> int:
        """Number of volatile packages."""
        return len(self.volatile_packages)

    @property
    def n_stable(self) -> int:
        """Number of stable packages."""
        return len(self.stable_packages)

    @property
    def total_packages(self) -> int:
        """Total number of packages analyzed."""
        return len(self.package_trends)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "series_id": self.series_id,
            "n_runs": self.n_runs,
            "date_range": self.date_range,
            "condition": self.condition,
            "baseline_bench_id": self.baseline_bench_id,
            "package_trends": [t.to_dict() for t in self.package_trends],
            "alerts": [a.to_dict() for a in self.alerts],
            "regressing_packages": self.regressing_packages,
            "improving_packages": self.improving_packages,
            "volatile_packages": self.volatile_packages,
            "stable_packages": self.stable_packages,
            "aggregate_median_change_pct": round(self.aggregate_median_change_pct, 2),
        }


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------


def _linear_regression_slope(ys: list[float]) -> float:
    """Compute the slope of a simple linear regression of ys vs index.

    Pure Python implementation. xs are implicitly [0, 1, 2, ...].
    Returns 0.0 if fewer than 2 points.
    """
    n = len(ys)
    if n < 2:
        return 0.0

    # xs = [0, 1, ..., n-1]
    # mean_x = (n-1)/2, mean_y = mean(ys)
    mean_x = (n - 1) / 2.0
    mean_y = statistics.mean(ys)

    numerator = 0.0
    denominator = 0.0
    for i, y in enumerate(ys):
        dx = i - mean_x
        numerator += dx * (y - mean_y)
        denominator += dx * dx

    if denominator == 0:
        return 0.0

    return numerator / denominator


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------


def compute_package_trend(
    package: str,
    condition: str,
    medians: list[float],
    cvs: list[float],
    timestamps: list[str],
    *,
    regression_threshold: float = 0.02,
    trend_threshold: float = 0.05,
    sustained_count: int = 3,
) -> PackageTrend:
    """Compute trend classification for a single package.

    Args:
        package: Package name.
        condition: Condition name.
        medians: Median wall times per run, in chronological order.
        cvs: CV per run, in chronological order.
        timestamps: Run timestamps, in chronological order.
        regression_threshold: Minimum per-run change to count as
            regression/improvement (as fraction, 0.02 = 2%).
        trend_threshold: Overall slope threshold for trend classification
            (as fraction of baseline median per run, 0.05 = 5%).
        sustained_count: Number of consecutive increases/decreases
            needed for sustained_regression/improvement.

    Returns:
        PackageTrend with all derived fields computed.
    """
    trend = PackageTrend(
        package=package,
        condition=condition,
        timestamps=list(timestamps),
        medians=list(medians),
        cvs=list(cvs),
        n_runs=len(medians),
    )

    if len(medians) < 2:
        return trend

    baseline_median = medians[0]

    # Slope of median vs run index.
    trend.trend_slope = _linear_regression_slope(medians)

    # Slope as percentage of baseline median.
    if baseline_median > 0:
        trend.trend_pct_per_run = (trend.trend_slope / baseline_median) * 100
    else:
        trend.trend_pct_per_run = 0.0

    # Recent change: last vs previous.
    if medians[-2] > 0:
        trend.recent_change_pct = ((medians[-1] - medians[-2]) / medians[-2]) * 100
    else:
        trend.recent_change_pct = 0.0

    # Cumulative change: last vs baseline (first).
    if baseline_median > 0:
        trend.cumulative_change_pct = ((medians[-1] - baseline_median) / baseline_median) * 100
    else:
        trend.cumulative_change_pct = 0.0

    # Detect sustained regression: last N consecutive increases > threshold.
    if len(medians) >= sustained_count + 1:
        tail = medians[-(sustained_count + 1) :]
        increases = all(
            (tail[i + 1] - tail[i]) / tail[i] > regression_threshold
            for i in range(sustained_count)
            if tail[i] > 0
        )
        trend.sustained_regression = increases

    # Detect sustained improvement: last N consecutive decreases > threshold.
    if len(medians) >= sustained_count + 1:
        tail = medians[-(sustained_count + 1) :]
        decreases = all(
            (tail[i] - tail[i + 1]) / tail[i] > regression_threshold
            for i in range(sustained_count)
            if tail[i] > 0
        )
        trend.sustained_improvement = decreases

    # Detect increasing volatility: positive CV slope + last 2 CVs > 0.10.
    if len(cvs) >= 2:
        cv_slope = _linear_regression_slope(cvs)
        if cv_slope > 0 and cvs[-1] > 0.10 and cvs[-2] > 0.10:
            trend.volatility_increasing = True

    # Classify trend direction.
    # trend_pct_per_run is in percentage, trend_threshold is a fraction.
    threshold_pct = trend_threshold * 100
    if trend.volatility_increasing and not (
        trend.sustained_regression or trend.sustained_improvement
    ):
        trend.trend_direction = "volatile"
    elif trend.sustained_regression or trend.trend_pct_per_run > threshold_pct:
        trend.trend_direction = "regressing"
    elif trend.sustained_improvement or trend.trend_pct_per_run < -threshold_pct:
        trend.trend_direction = "improving"
    else:
        trend.trend_direction = "stable"

    return trend


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------


def _generate_alerts(
    package_trends: list[PackageTrend],
    baseline_medians: dict[str, float],
    previous_medians: dict[str, float],
    current_medians: dict[str, float],
    *,
    regression_threshold: float = 0.02,
) -> list[RegressionAlert]:
    """Generate regression alerts from package trends.

    Alert types:
    - new_regression: Latest run significantly slower than previous
      and previous was not regressing. Severity: "warning".
    - sustained_regression: Package has been regressing for 3+ runs.
      Severity: "error".
    - recovery: Package was regressing but latest run returned to
      baseline levels. Severity: "info".
    - new_instability: Package's CV jumped above 10% for the first
      time in the series. Severity: "warning".
    - new_improvement: Latest run significantly faster than previous.
      Severity: "info".

    Args:
        package_trends: Computed trends.
        baseline_medians: Package -> median in baseline run.
        previous_medians: Package -> median in previous run.
        current_medians: Package -> median in latest run.
        regression_threshold: Change threshold for significance.

    Returns:
        List of RegressionAlert, sorted by severity then abs change.
    """
    alerts: list[RegressionAlert] = []

    for pt in package_trends:
        pkg = pt.package
        baseline_med = baseline_medians.get(pkg, 0.0)
        previous_med = previous_medians.get(pkg, 0.0)
        current_med = current_medians.get(pkg, 0.0)

        if baseline_med <= 0 or previous_med <= 0 or current_med <= 0:
            continue

        recent_pct = ((current_med - previous_med) / previous_med) * 100
        cumulative_pct = ((current_med - baseline_med) / baseline_med) * 100

        # Sustained regression.
        if pt.sustained_regression:
            alerts.append(
                RegressionAlert(
                    package=pkg,
                    condition=pt.condition,
                    alert_type="sustained_regression",
                    severity="error",
                    description=(
                        f"Sustained regression over {pt.n_runs} runs "
                        f"({cumulative_pct:+.1f}% from baseline)."
                    ),
                    recent_change_pct=recent_pct,
                    cumulative_change_pct=cumulative_pct,
                    baseline_median_s=baseline_med,
                    current_median_s=current_med,
                    previous_median_s=previous_med,
                )
            )
        # New regression: recent increase > threshold, not already sustained.
        elif recent_pct > regression_threshold * 100 and not pt.sustained_regression:
            alerts.append(
                RegressionAlert(
                    package=pkg,
                    condition=pt.condition,
                    alert_type="new_regression",
                    severity="warning",
                    description=(f"New regression: {recent_pct:+.1f}% slower than previous run."),
                    recent_change_pct=recent_pct,
                    cumulative_change_pct=cumulative_pct,
                    baseline_median_s=baseline_med,
                    current_median_s=current_med,
                    previous_median_s=previous_med,
                )
            )

        # Recovery: was trending up but latest run returned to baseline level.
        if pt.trend_direction == "stable" and pt.n_runs >= 3:
            # Check if medians were regressing then recovered.
            if len(pt.medians) >= 3:
                prev_increasing = pt.medians[-3] < pt.medians[-2]
                now_back = abs(cumulative_pct) < regression_threshold * 100
                if prev_increasing and now_back:
                    alerts.append(
                        RegressionAlert(
                            package=pkg,
                            condition=pt.condition,
                            alert_type="recovery",
                            severity="info",
                            description="Recovered to baseline levels.",
                            recent_change_pct=recent_pct,
                            cumulative_change_pct=cumulative_pct,
                            baseline_median_s=baseline_med,
                            current_median_s=current_med,
                            previous_median_s=previous_med,
                        )
                    )

        # New instability: CV jumped above 10%.
        if pt.volatility_increasing and len(pt.cvs) >= 2:
            # Check if earlier CVs were below 10%.
            early_cvs_low = all(cv < 0.10 for cv in pt.cvs[:-2])
            if early_cvs_low:
                alerts.append(
                    RegressionAlert(
                        package=pkg,
                        condition=pt.condition,
                        alert_type="new_instability",
                        severity="warning",
                        description=(
                            f"Measurement instability: CV increased to "
                            f"{pt.cvs[-1]:.1%} (was below 10%)."
                        ),
                        recent_change_pct=recent_pct,
                        cumulative_change_pct=cumulative_pct,
                        baseline_median_s=baseline_med,
                        current_median_s=current_med,
                        previous_median_s=previous_med,
                    )
                )

        # New improvement: recent decrease > threshold.
        if recent_pct < -regression_threshold * 100 and not pt.sustained_improvement:
            alerts.append(
                RegressionAlert(
                    package=pkg,
                    condition=pt.condition,
                    alert_type="new_improvement",
                    severity="info",
                    description=(f"Improvement: {recent_pct:+.1f}% faster than previous run."),
                    recent_change_pct=recent_pct,
                    cumulative_change_pct=cumulative_pct,
                    baseline_median_s=baseline_med,
                    current_median_s=current_med,
                    previous_median_s=previous_med,
                )
            )

    # Sort: errors first, then warnings, then info. Within severity, by abs change.
    severity_order = {"error": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: (severity_order.get(a.severity, 9), -abs(a.cumulative_change_pct)))

    return alerts


# ---------------------------------------------------------------------------
# Series-level analysis
# ---------------------------------------------------------------------------


def analyze_series_trends(
    series: TrackingSeries,
    series_dir: Path,
    *,
    condition: str | None = None,
    regression_threshold: float = 0.02,
    trend_threshold: float = 0.05,
    sustained_count: int = 3,
) -> SeriesTrend:
    """Analyze trends across all runs in a tracking series.

    Loads each run, extracts per-package median wall time and CV
    for the specified condition, then computes PackageTrend for
    each package and produces regression alerts.

    Args:
        series: The tracking series.
        series_dir: Path to the series directory.
        condition: Which condition to analyze. If None, uses the
            first condition found in the baseline run.
        regression_threshold: Per-run change threshold for regression.
        trend_threshold: Overall slope threshold for classification.
        sustained_count: Consecutive changes for sustained flags.

    Returns:
        SeriesTrend with per-package trends and alerts.
    """
    if not series.runs:
        baseline_entry = series.baseline_run
        return SeriesTrend(
            series_id=series.series_id,
            n_runs=0,
            date_range=series.date_range,
            condition=condition or "",
            baseline_bench_id=baseline_entry.bench_id if baseline_entry else "",
        )

    # Load all runs in chronological order.
    run_data: list[tuple[TrackingRunEntry, BenchMeta, list[BenchPackageResult]]] = []
    for entry in series.runs:
        try:
            meta, results = load_series_run(series, series_dir, entry)
            run_data.append((entry, meta, results))
        except (FileNotFoundError, ValueError) as exc:
            log.warning("Skipping run '%s': %s", entry.bench_id, exc)

    if not run_data:
        baseline_entry = series.baseline_run
        return SeriesTrend(
            series_id=series.series_id,
            n_runs=0,
            date_range=series.date_range,
            condition=condition or "",
            baseline_bench_id=baseline_entry.bench_id if baseline_entry else "",
        )

    # Determine condition to analyze.
    baseline_entry = series.baseline_run
    baseline_bench_id = baseline_entry.bench_id if baseline_entry else run_data[0][0].bench_id
    if condition is None:
        # Use first condition from the baseline (or first run).
        baseline_meta = None
        for entry, meta, _ in run_data:
            if entry.bench_id == baseline_bench_id:
                baseline_meta = meta
                break
        if baseline_meta is None:
            baseline_meta = run_data[0][1]
        cond_names = list(baseline_meta.conditions.keys())
        condition = cond_names[0] if cond_names else ""

    # Collect per-package medians and CVs across runs.
    # package -> {run_index: (median, cv, timestamp)}
    pkg_data: dict[str, list[tuple[float, float, str]]] = {}
    all_packages: set[str] = set()

    for entry, meta, results in run_data:
        for r in results:
            if r.skipped:
                continue
            cond = r.conditions.get(condition)
            if not cond or not cond.wall_time_stats:
                continue
            all_packages.add(r.package)
            if r.package not in pkg_data:
                pkg_data[r.package] = []
            pkg_data[r.package].append(
                (cond.wall_time_stats.median, cond.wall_time_stats.cv, entry.timestamp)
            )

    # Compute trends.
    package_trends: list[PackageTrend] = []
    for pkg in sorted(all_packages):
        data_points = pkg_data.get(pkg, [])
        if not data_points:
            continue
        medians = [d[0] for d in data_points]
        cvs = [d[1] for d in data_points]
        timestamps = [d[2] for d in data_points]

        pt = compute_package_trend(
            pkg,
            condition,
            medians,
            cvs,
            timestamps,
            regression_threshold=regression_threshold,
            trend_threshold=trend_threshold,
            sustained_count=sustained_count,
        )
        package_trends.append(pt)

    # Classify packages.
    regressing_packages: list[str] = []
    improving_packages: list[str] = []
    volatile_packages: list[str] = []
    stable_packages: list[str] = []

    for pt in package_trends:
        if pt.trend_direction == "regressing":
            regressing_packages.append(pt.package)
        elif pt.trend_direction == "improving":
            improving_packages.append(pt.package)
        elif pt.trend_direction == "volatile":
            volatile_packages.append(pt.package)
        else:
            stable_packages.append(pt.package)

    # Build median dicts for alert generation.
    baseline_medians: dict[str, float] = {}
    previous_medians: dict[str, float] = {}
    current_medians: dict[str, float] = {}

    for pt in package_trends:
        if pt.medians:
            baseline_medians[pt.package] = pt.medians[0]
            current_medians[pt.package] = pt.medians[-1]
            if len(pt.medians) >= 2:
                previous_medians[pt.package] = pt.medians[-2]
            else:
                previous_medians[pt.package] = pt.medians[0]

    # Generate alerts.
    alerts = _generate_alerts(
        package_trends,
        baseline_medians,
        previous_medians,
        current_medians,
        regression_threshold=regression_threshold,
    )

    # Compute aggregate change.
    cumulative_changes = [pt.cumulative_change_pct for pt in package_trends if pt.n_runs >= 2]
    aggregate_change = statistics.median(cumulative_changes) if cumulative_changes else 0.0

    return SeriesTrend(
        series_id=series.series_id,
        n_runs=len(run_data),
        date_range=series.date_range,
        condition=condition,
        baseline_bench_id=baseline_bench_id,
        package_trends=package_trends,
        alerts=alerts,
        regressing_packages=regressing_packages,
        improving_packages=improving_packages,
        volatile_packages=volatile_packages,
        stable_packages=stable_packages,
        aggregate_median_change_pct=aggregate_change,
    )
