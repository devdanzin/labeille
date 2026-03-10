"""Package-level anomaly detection for benchmark results.

Identifies measurement-quality problems in benchmark data:
- High coefficient of variation (non-deterministic timing)
- Bimodal timing distributions (two distinct execution modes)
- Outlier-heavy runs (occasional spikes in otherwise stable data)
- Mixed pass/fail status within a condition
- Monotonic iteration trends (warmup leakage or memory leaks)

Anomalies are classified by type and severity. They are computed
lazily on display, not during benchmark execution, so they add
no overhead to the run itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from labeille.bench.results import BenchConditionResult, BenchPackageResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PackageAnomaly:
    """A single anomaly detected in a package's benchmark data."""

    package: str
    condition: str
    anomaly_type: str
    # One of: "high_cv", "bimodal", "outlier_heavy", "status_mixed", "trend"
    severity: str  # "info", "warning", "error"
    metric_value: float  # The value that triggered the anomaly
    threshold: float  # The threshold it exceeded
    description: str  # Human-readable explanation
    recommendation: str  # What to do about it

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "package": self.package,
            "condition": self.condition,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "metric_value": round(self.metric_value, 4),
            "threshold": round(self.threshold, 4),
            "description": self.description,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackageAnomaly:
        """Deserialize from a dict."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class AnomalyReport:
    """Aggregate anomaly report across all packages and conditions."""

    anomalies: list[PackageAnomaly] = field(default_factory=list)

    @property
    def by_severity(self) -> dict[str, list[PackageAnomaly]]:
        """Group anomalies by severity level."""
        groups: dict[str, list[PackageAnomaly]] = {
            "error": [],
            "warning": [],
            "info": [],
        }
        for a in self.anomalies:
            groups.setdefault(a.severity, []).append(a)
        return groups

    @property
    def by_package(self) -> dict[str, list[PackageAnomaly]]:
        """Group anomalies by package name."""
        groups: dict[str, list[PackageAnomaly]] = {}
        for a in self.anomalies:
            groups.setdefault(a.package, []).append(a)
        return groups

    @property
    def by_type(self) -> dict[str, list[PackageAnomaly]]:
        """Group anomalies by anomaly type."""
        groups: dict[str, list[PackageAnomaly]] = {}
        for a in self.anomalies:
            groups.setdefault(a.anomaly_type, []).append(a)
        return groups

    @property
    def error_count(self) -> int:
        """Number of error-severity anomalies."""
        return len(self.by_severity.get("error", []))

    @property
    def warning_count(self) -> int:
        """Number of warning-severity anomalies."""
        return len(self.by_severity.get("warning", []))

    @property
    def info_count(self) -> int:
        """Number of info-severity anomalies."""
        return len(self.by_severity.get("info", []))

    @property
    def affected_packages(self) -> set[str]:
        """Set of package names with at least one anomaly."""
        return {a.package for a in self.anomalies}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "anomalies": [a.to_dict() for a in self.anomalies],
            "summary": {
                "total": len(self.anomalies),
                "errors": self.error_count,
                "warnings": self.warning_count,
                "info": self.info_count,
                "affected_packages": len(self.affected_packages),
            },
        }


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def is_bimodal(values: list[float], *, gap_factor: float = 1.5) -> bool:
    """Detect bimodality via gap analysis.

    Sorts values, computes gaps between consecutive values. If any gap
    exceeds gap_factor x median_gap and splits the data into two groups
    each with at least 2 values, returns True.

    This is a lightweight heuristic -- not a full statistical test -- but
    catches the common case of tests that sometimes take 5s and sometimes
    take 15s.

    Args:
        values: Timing measurements.
        gap_factor: Multiplier for the median gap to identify a split.

    Returns:
        True if the distribution appears bimodal.
    """
    if len(values) < 4:
        return False

    sorted_vals = sorted(values)
    gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]

    if not gaps:
        return False

    sorted_gaps = sorted(gaps)
    n_gaps = len(sorted_gaps)
    median_gap = (
        sorted_gaps[n_gaps // 2]
        if n_gaps % 2
        else (sorted_gaps[n_gaps // 2 - 1] + sorted_gaps[n_gaps // 2]) / 2
    )

    # When median_gap is 0, fall back to mean_gap so gap_factor still works.
    ref_gap = median_gap if median_gap > 0 else (sum(gaps) / len(gaps) if gaps else 0.0)
    if ref_gap <= 0:
        return False

    threshold = gap_factor * ref_gap

    # Minimum absolute gap: must be at least 10% of the mean value to
    # avoid flagging near-identical data as bimodal.
    mean_val = sum(sorted_vals) / len(sorted_vals) if sorted_vals else 0.0
    min_gap = 0.10 * abs(mean_val) if mean_val != 0 else 0.0

    for i, gap in enumerate(gaps):
        if gap > threshold and gap >= min_gap:
            left_count = i + 1  # values[0..i]
            right_count = len(sorted_vals) - left_count  # values[i+1..]
            if left_count >= 2 and right_count >= 2:
                return True

    return False


def has_monotonic_trend(
    values: list[float],
    *,
    correlation_threshold: float = 0.7,
) -> tuple[bool, float]:
    """Detect monotonic trend in iteration timings via Spearman rank correlation.

    A strong positive correlation (increasing times) suggests a memory
    leak or growing state. A strong negative correlation (decreasing
    times) suggests warmup effects that weren't captured by warmup
    iterations.

    Uses Spearman rank correlation (not Pearson) because we care about
    monotonicity, not linearity.

    Args:
        values: Timing measurements in iteration order.
        correlation_threshold: Minimum |rho| to flag as a trend.

    Returns:
        Tuple of (has_trend, rho). rho is the Spearman correlation
        coefficient in [-1, 1].
    """
    if len(values) < 5:
        return False, 0.0

    n = len(values)

    # Check for zero variance.
    if all(v == values[0] for v in values):
        return False, 0.0

    # Compute ranks for values (handling ties via average rank).
    def _average_ranks(data: list[float]) -> list[float]:
        indexed = sorted(range(len(data)), key=lambda i: data[i])
        ranks = [0.0] * len(data)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and data[indexed[j]] == data[indexed[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                ranks[indexed[k]] = avg_rank
            i = j
        return ranks

    # Index ranks are just [1, 2, ..., n].
    index_ranks = [float(i + 1) for i in range(n)]
    value_ranks = _average_ranks(values)

    # Pearson correlation on ranks.
    mean_x = sum(index_ranks) / n
    mean_y = sum(value_ranks) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(index_ranks, value_ranks))
    var_x = sum((x - mean_x) ** 2 for x in index_ranks)
    var_y = sum((y - mean_y) ** 2 for y in value_ranks)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return False, 0.0

    rho = cov / denom
    return abs(rho) >= correlation_threshold, rho


# ---------------------------------------------------------------------------
# Condition-level detection
# ---------------------------------------------------------------------------


def detect_condition_anomalies(
    package: str,
    condition: str,
    cond_result: BenchConditionResult,
    *,
    cv_warning_threshold: float = 0.10,
    cv_error_threshold: float = 0.20,
    outlier_fraction_threshold: float = 0.20,
    gap_factor: float = 1.5,
    trend_threshold: float = 0.7,
) -> list[PackageAnomaly]:
    """Detect anomalies in a single condition's benchmark data.

    Checks (in order):
    1. status_mixed -- some iterations pass and some fail
    2. high_cv -- coefficient of variation exceeds threshold
    3. bimodal -- timing distribution has two distinct clusters
    4. outlier_heavy -- more than 20% of iterations are outliers
    5. trend -- monotonic increase or decrease across iterations

    Args:
        package: Package name (for reporting).
        condition: Condition name (for reporting).
        cond_result: The condition result to analyze.
        cv_warning_threshold: CV above this -> warning.
        cv_error_threshold: CV above this -> error.
        outlier_fraction_threshold: Outlier fraction above this -> info.
        gap_factor: For bimodal detection.
        trend_threshold: Minimum |rho| for trend detection.

    Returns:
        List of PackageAnomaly instances (may be empty).
    """
    anomalies: list[PackageAnomaly] = []
    measured = cond_result.measured_iterations

    if len(measured) < 3:
        return anomalies

    wall_times = [it.wall_time_s for it in measured]

    # 1. status_mixed
    statuses = [it.status for it in measured]
    unique_statuses = set(statuses)
    if len(unique_statuses) > 1:
        status_counts = {s: statuses.count(s) for s in unique_statuses}
        parts = ", ".join(f"{count} '{s}'" for s, count in sorted(status_counts.items()))
        anomalies.append(
            PackageAnomaly(
                package=package,
                condition=condition,
                anomaly_type="status_mixed",
                severity="error",
                metric_value=float(len(unique_statuses)),
                threshold=1.0,
                description=(f"{len(measured)} iterations have mixed statuses: {parts}."),
                recommendation=(
                    "Test suite has intermittent failures. Fix or exclude "
                    "this package from benchmarking."
                ),
            )
        )

    # 2. high_cv
    if cond_result.wall_time_stats:
        cv = cond_result.wall_time_stats.cv
        if cv > cv_error_threshold:
            anomalies.append(
                PackageAnomaly(
                    package=package,
                    condition=condition,
                    anomaly_type="high_cv",
                    severity="error",
                    metric_value=cv,
                    threshold=cv_error_threshold,
                    description=(
                        f"Wall time CV is {cv * 100:.1f}% "
                        f"(threshold: {cv_error_threshold * 100:.0f}%)."
                    ),
                    recommendation=(
                        "Investigate test suite for non-determinism (network calls, "
                        "random seeds, disk I/O). Consider adding "
                        "'--test-command-suffix \"-p no:randomly\"' or pinning seeds."
                    ),
                )
            )
        elif cv > cv_warning_threshold:
            anomalies.append(
                PackageAnomaly(
                    package=package,
                    condition=condition,
                    anomaly_type="high_cv",
                    severity="warning",
                    metric_value=cv,
                    threshold=cv_warning_threshold,
                    description=(
                        f"Wall time CV is {cv * 100:.1f}% "
                        f"(threshold: {cv_warning_threshold * 100:.0f}%)."
                    ),
                    recommendation=(
                        "Investigate test suite for non-determinism (network calls, "
                        "random seeds, disk I/O). Consider adding "
                        "'--test-command-suffix \"-p no:randomly\"' or pinning seeds."
                    ),
                )
            )

    # 3. bimodal
    if is_bimodal(wall_times, gap_factor=gap_factor):
        anomalies.append(
            PackageAnomaly(
                package=package,
                condition=condition,
                anomaly_type="bimodal",
                severity="warning",
                metric_value=0.0,
                threshold=gap_factor,
                description="Wall times appear bimodal (two distinct clusters).",
                recommendation=(
                    "Test suite may exercise different code paths between runs. "
                    "Check for conditional test skipping or "
                    "environment-dependent behavior."
                ),
            )
        )

    # 4. outlier_heavy
    n_measured = len(measured)
    n_outliers = sum(1 for it in measured if it.outlier)
    outlier_fraction = n_outliers / n_measured if n_measured else 0.0
    if outlier_fraction > outlier_fraction_threshold:
        anomalies.append(
            PackageAnomaly(
                package=package,
                condition=condition,
                anomaly_type="outlier_heavy",
                severity="info",
                metric_value=outlier_fraction,
                threshold=outlier_fraction_threshold,
                description=(
                    f"{outlier_fraction * 100:.0f}% of iterations flagged as outliers "
                    f"(threshold: {outlier_fraction_threshold * 100:.0f}%)."
                ),
                recommendation=(
                    "Consider increasing iterations to dilute outlier impact, "
                    "or investigate system interference."
                ),
            )
        )

    # 5. trend
    has_trend, rho = has_monotonic_trend(wall_times, correlation_threshold=trend_threshold)
    if has_trend:
        if rho > 0:
            anomalies.append(
                PackageAnomaly(
                    package=package,
                    condition=condition,
                    anomaly_type="trend",
                    severity="warning",
                    metric_value=rho,
                    threshold=trend_threshold,
                    description=(
                        f"Wall times show increasing trend "
                        f"(\u03c1={rho:.2f}), suggesting memory leak "
                        f"or growing state."
                    ),
                    recommendation=(
                        "Increase warmup iterations or investigate "
                        "growing resource usage across iterations."
                    ),
                )
            )
        else:
            anomalies.append(
                PackageAnomaly(
                    package=package,
                    condition=condition,
                    anomaly_type="trend",
                    severity="info",
                    metric_value=rho,
                    threshold=trend_threshold,
                    description=(
                        f"Wall times show decreasing trend "
                        f"(\u03c1={rho:.2f}), suggesting warmup effects "
                        f"not captured by warmup iterations."
                    ),
                    recommendation=(
                        "Increase warmup iterations to ensure the "
                        "system reaches steady state before measuring."
                    ),
                )
            )

    return anomalies


# ---------------------------------------------------------------------------
# Top-level detection
# ---------------------------------------------------------------------------


def detect_anomalies(
    results: list[BenchPackageResult],
    **kwargs: Any,
) -> AnomalyReport:
    """Detect anomalies across all packages and conditions.

    Iterates over all non-skipped packages and all conditions,
    running detect_condition_anomalies() on each.

    Args:
        results: List of benchmark results.
        **kwargs: Passed through to detect_condition_anomalies().

    Returns:
        AnomalyReport with all detected anomalies.
    """
    all_anomalies: list[PackageAnomaly] = []

    for pkg_result in results:
        if pkg_result.skipped:
            continue
        for cond_name, cond_result in pkg_result.conditions.items():
            anomalies = detect_condition_anomalies(
                pkg_result.package,
                cond_name,
                cond_result,
                **kwargs,
            )
            all_anomalies.extend(anomalies)

    return AnomalyReport(anomalies=all_anomalies)
