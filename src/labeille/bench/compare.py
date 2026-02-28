"""Benchmark comparison analysis.

Compares benchmark conditions or runs, producing structured results
with statistical tests, overhead calculations, and anomaly flags.
"""

from __future__ import annotations

import logging
import statistics as _stats
from dataclasses import dataclass, field

from labeille.bench.results import (
    BenchMeta,
    BenchPackageResult,
)
from labeille.bench.stats import (
    OverheadResult,
    compute_overhead,
)

log = logging.getLogger("labeille")


# ---------------------------------------------------------------------------
# Per-package comparison result
# ---------------------------------------------------------------------------


@dataclass
class PackageOverhead:
    """Comparison of one package between baseline and treatment."""

    package: str
    overhead: OverheadResult
    baseline_condition: str
    treatment_condition: str
    # Anomaly flags
    high_cv_baseline: bool = False  # CV > 10%
    high_cv_treatment: bool = False
    has_outliers: bool = False
    status_mismatch: bool = False  # different predominant status

    @property
    def reliable(self) -> bool:
        """True if the measurement is trustworthy."""
        return not self.high_cv_baseline and not self.high_cv_treatment

    @property
    def significant_and_reliable(self) -> bool:
        """True if significant, practically meaningful, and reliable."""
        return self.overhead.practically_significant and self.reliable


# ---------------------------------------------------------------------------
# Aggregate comparison
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """Complete comparison between two conditions across all packages."""

    baseline_name: str
    treatment_name: str
    packages: list[PackageOverhead] = field(default_factory=list)

    # Aggregate stats (computed from reliable packages only)
    median_overhead_pct: float = 0.0
    mean_overhead_pct: float = 0.0
    min_overhead_pct: float = 0.0
    max_overhead_pct: float = 0.0

    # Counts
    total_packages: int = 0
    significant_packages: int = 0
    practically_significant: int = 0
    reliable_packages: int = 0
    unreliable_packages: int = 0
    faster_packages: int = 0  # treatment faster than baseline
    slower_packages: int = 0  # treatment slower than baseline

    # Anomaly summary
    high_cv_count: int = 0
    outlier_count: int = 0
    status_mismatch_count: int = 0

    @property
    def most_affected(self) -> list[PackageOverhead]:
        """Top 5 packages with highest overhead (reliable only)."""
        reliable = [p for p in self.packages if p.reliable]
        return sorted(
            reliable,
            key=lambda p: p.overhead.overhead_pct,
            reverse=True,
        )[:5]

    @property
    def most_improved(self) -> list[PackageOverhead]:
        """Top 5 packages that are faster under treatment (reliable only)."""
        faster = [p for p in self.packages if p.reliable and p.overhead.overhead_pct < 0]
        return sorted(
            faster,
            key=lambda p: p.overhead.overhead_pct,
        )[:5]


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_conditions(
    results: list[BenchPackageResult],
    baseline_name: str,
    treatment_name: str,
    *,
    cv_threshold: float = 0.10,
    ci_seed: int = 42,
    ci_bootstrap_n: int = 10000,
) -> ComparisonReport:
    """Compare two conditions across all packages.

    Args:
        results: List of BenchPackageResults (with both conditions).
        baseline_name: Name of the baseline condition.
        treatment_name: Name of the treatment condition.
        cv_threshold: CV above this flags unreliable measurements.
        ci_seed: Random seed for bootstrap CI (reproducibility).
        ci_bootstrap_n: Number of bootstrap resamples.

    Returns:
        ComparisonReport with per-package and aggregate results.
    """
    report = ComparisonReport(
        baseline_name=baseline_name,
        treatment_name=treatment_name,
    )

    package_overheads: list[PackageOverhead] = []

    for r in results:
        if r.skipped:
            continue

        base = r.conditions.get(baseline_name)
        treat = r.conditions.get(treatment_name)

        if not base or not treat:
            continue
        if not base.wall_times or not treat.wall_times:
            continue

        overhead = compute_overhead(
            base.wall_times,
            treat.wall_times,
            ci_seed=ci_seed,
            ci_bootstrap_n=ci_bootstrap_n,
        )

        # Check anomalies.
        high_cv_b = base.wall_time_stats is not None and base.wall_time_stats.cv > cv_threshold
        high_cv_t = treat.wall_time_stats is not None and treat.wall_time_stats.cv > cv_threshold
        has_outliers = base.n_outliers > 0 or treat.n_outliers > 0

        # Check if the predominant exit status differs.
        base_statuses = [it.status for it in base.measured_iterations]
        treat_statuses = [it.status for it in treat.measured_iterations]
        base_mode = max(set(base_statuses), key=base_statuses.count) if base_statuses else ""
        treat_mode = max(set(treat_statuses), key=treat_statuses.count) if treat_statuses else ""
        status_mismatch = base_mode != treat_mode

        pkg_oh = PackageOverhead(
            package=r.package,
            overhead=overhead,
            baseline_condition=baseline_name,
            treatment_condition=treatment_name,
            high_cv_baseline=high_cv_b,
            high_cv_treatment=high_cv_t,
            has_outliers=has_outliers,
            status_mismatch=status_mismatch,
        )
        package_overheads.append(pkg_oh)

    report.packages = package_overheads
    report.total_packages = len(package_overheads)

    # Compute aggregates from reliable packages.
    reliable = [p for p in package_overheads if p.reliable]
    report.reliable_packages = len(reliable)
    report.unreliable_packages = report.total_packages - len(reliable)

    pcts = [p.overhead.overhead_pct for p in reliable]
    if pcts:
        report.median_overhead_pct = _stats.median(pcts)
        report.mean_overhead_pct = _stats.mean(pcts)
        report.min_overhead_pct = min(pcts)
        report.max_overhead_pct = max(pcts)

    report.significant_packages = sum(
        1 for p in package_overheads if p.overhead.ttest.significant_05
    )
    report.practically_significant = sum(
        1 for p in package_overheads if p.overhead.practically_significant
    )
    report.faster_packages = sum(1 for p in package_overheads if p.overhead.overhead_pct < 0)
    report.slower_packages = sum(1 for p in package_overheads if p.overhead.overhead_pct > 0)
    report.high_cv_count = sum(
        1 for p in package_overheads if p.high_cv_baseline or p.high_cv_treatment
    )
    report.outlier_count = sum(1 for p in package_overheads if p.has_outliers)
    report.status_mismatch_count = sum(1 for p in package_overheads if p.status_mismatch)

    return report


def compare_runs(
    run_a: tuple[BenchMeta, list[BenchPackageResult]],
    run_b: tuple[BenchMeta, list[BenchPackageResult]],
    *,
    condition_a: str | None = None,
    condition_b: str | None = None,
    ci_seed: int = 42,
) -> ComparisonReport:
    """Compare two separate benchmark runs.

    Merges results by package name, using the specified condition
    from each run (defaults to the first condition in each).

    Args:
        run_a: Tuple of (BenchMeta, results) for the baseline run.
        run_b: Tuple of (BenchMeta, results) for the treatment run.
        condition_a: Condition name in run A (default: first).
        condition_b: Condition name in run B (default: first).
        ci_seed: Random seed for bootstrap CI.

    Returns:
        ComparisonReport comparing run_a's condition vs run_b's.
    """
    meta_a, results_a = run_a
    meta_b, results_b = run_b

    cond_a = condition_a or next(iter(meta_a.conditions.keys()))
    cond_b = condition_b or next(iter(meta_b.conditions.keys()))

    name_a = meta_a.name or meta_a.bench_id
    name_b = meta_b.name or meta_b.bench_id

    # Build lookup by package name.
    pkgs_a = {r.package: r for r in results_a if not r.skipped}
    pkgs_b = {r.package: r for r in results_b if not r.skipped}

    # Find common packages.
    common = sorted(set(pkgs_a.keys()) & set(pkgs_b.keys()))

    # Create merged BenchPackageResults with condition names
    # matching the run names.
    merged: list[BenchPackageResult] = []
    for pkg_name in common:
        ra = pkgs_a[pkg_name]
        rb = pkgs_b[pkg_name]

        cond_result_a = ra.conditions.get(cond_a)
        cond_result_b = rb.conditions.get(cond_b)
        if not cond_result_a or not cond_result_b:
            continue

        merged_pkg = BenchPackageResult(package=pkg_name)
        merged_pkg.conditions[name_a] = cond_result_a
        merged_pkg.conditions[name_b] = cond_result_b
        merged.append(merged_pkg)

    return compare_conditions(
        merged,
        name_a,
        name_b,
        ci_seed=ci_seed,
    )
