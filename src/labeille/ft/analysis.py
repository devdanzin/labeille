"""Deep analysis of free-threading test results.

Provides:
- Flakiness profiling: which tests fail intermittently and how often.
- Failure pattern detection: same test every time vs random failures.
- GIL comparison: isolate free-threading-specific failures.
- Triage prioritization: rank packages by investigation urgency.
- Duration analysis: detect timing anomalies suggesting contention.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from labeille.ft.results import (
    FailureCategory,
    FTPackageResult,
    FTRunSummary,
    IterationOutcome,
)

log = logging.getLogger("labeille")


# ---------------------------------------------------------------------------
# Flakiness profiling
# ---------------------------------------------------------------------------


@dataclass
class FlakyTest:
    """A single test that fails intermittently."""

    test_id: str
    fail_count: int
    total_seen: int
    fail_rate: float
    statuses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "fail_count": self.fail_count,
            "total_seen": self.total_seen,
            "fail_rate": round(self.fail_rate, 4),
        }


@dataclass
class FlakinessProfile:
    """Detailed analysis of an intermittent package.

    Goes deeper than the basic pass_rate in FTPackageResult to
    identify specific flaky tests, failure patterns, and likely
    root causes.
    """

    package: str
    pass_rate: float
    total_iterations: int

    failure_modes: dict[str, int] = field(default_factory=dict)
    flaky_tests: list[FlakyTest] = field(default_factory=list)
    max_consecutive_passes: int = 0
    max_consecutive_failures: int = 0
    pattern: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "pass_rate": round(self.pass_rate, 4),
            "total_iterations": self.total_iterations,
            "failure_modes": self.failure_modes,
            "flaky_tests": [t.to_dict() for t in self.flaky_tests],
            "max_consecutive_passes": self.max_consecutive_passes,
            "max_consecutive_failures": self.max_consecutive_failures,
            "pattern": self.pattern,
        }


def analyze_flakiness(result: FTPackageResult) -> FlakinessProfile:
    """Produce a detailed flakiness profile for a package.

    Analyzes iteration outcomes to determine failure patterns,
    identify specific flaky tests, and classify the type of
    non-determinism.

    Args:
        result: A package result with multiple iterations.

    Returns:
        FlakinessProfile with detailed analysis.
    """
    profile = FlakinessProfile(
        package=result.package,
        pass_rate=result.pass_rate,
        total_iterations=result.iterations_completed,
    )

    if not result.iterations:
        return profile

    modes: dict[str, int] = {}
    for it in result.iterations:
        if it.status != "pass":
            modes[it.status] = modes.get(it.status, 0) + 1
    profile.failure_modes = modes

    profile.max_consecutive_passes = _max_streak(result.iterations, lambda it: it.is_pass)
    profile.max_consecutive_failures = _max_streak(result.iterations, lambda it: not it.is_pass)

    test_outcomes: dict[str, list[str]] = {}
    for it in result.iterations:
        for test_id, status in it.test_results.items():
            test_outcomes.setdefault(test_id, []).append(status)

    for test_id, statuses in sorted(test_outcomes.items()):
        fail_count = sum(1 for s in statuses if s in ("FAILED", "ERROR"))
        total = len(statuses)
        if 0 < fail_count < total:
            profile.flaky_tests.append(
                FlakyTest(
                    test_id=test_id,
                    fail_count=fail_count,
                    total_seen=total,
                    fail_rate=fail_count / total,
                    statuses=statuses,
                )
            )

    profile.flaky_tests.sort(key=lambda t: -t.fail_rate)

    profile.pattern = _classify_failure_pattern(result, profile)

    return profile


def _max_streak(
    iterations: list[IterationOutcome],
    predicate: Callable[[IterationOutcome], bool],
) -> int:
    """Find the longest consecutive streak matching predicate."""
    max_streak = 0
    current = 0
    for it in iterations:
        if predicate(it):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _classify_failure_pattern(
    result: FTPackageResult,
    profile: FlakinessProfile,
) -> str:
    """Classify the type of failure pattern.

    Heuristics:
    - If the same specific tests fail every time they fail ->
      "consistent_test" (likely a deterministic bug triggered by
      specific code paths).
    - If different tests fail each time -> "random_test" (likely a
      race condition in shared state).
    - If crashes have the same signature -> "consistent_crash".
    - If crashes have different signatures -> "varied_crash".
    """
    if result.crash_count > 0 and result.fail_count == 0:
        if len(result.failure_signatures) == 1:
            return "consistent_crash"
        elif len(result.failure_signatures) > 1:
            return "varied_crash"

    if profile.flaky_tests:
        failing_test_sets: list[frozenset[str]] = []
        for it in result.iterations:
            if not it.is_pass and it.test_results:
                failed_in_this = frozenset(
                    tid for tid, status in it.test_results.items() if status in ("FAILED", "ERROR")
                )
                if failed_in_this:
                    failing_test_sets.append(failed_in_this)

        if len(failing_test_sets) >= 2:
            if len(set(failing_test_sets)) == 1:
                return "consistent_test"
            return "random_test"

    return "unknown"


# ---------------------------------------------------------------------------
# GIL comparison analysis
# ---------------------------------------------------------------------------


@dataclass
class GILComparisonResult:
    """Comparison of one package under GIL-enabled vs GIL-disabled."""

    package: str
    gil_disabled_pass_rate: float
    gil_enabled_pass_rate: float
    classification: str
    ft_specific_failing_tests: list[str] = field(default_factory=list)
    shared_failing_tests: list[str] = field(default_factory=list)

    @property
    def free_threading_specific(self) -> bool:
        """True if failures only appear with GIL disabled."""
        return self.gil_enabled_pass_rate == 1.0 and self.gil_disabled_pass_rate < 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "gil_disabled_pass_rate": round(self.gil_disabled_pass_rate, 4),
            "gil_enabled_pass_rate": round(self.gil_enabled_pass_rate, 4),
            "classification": self.classification,
            "ft_specific_failing_tests": self.ft_specific_failing_tests,
            "shared_failing_tests": self.shared_failing_tests,
        }


def compare_gil_modes(
    result: FTPackageResult,
) -> GILComparisonResult | None:
    """Compare GIL-disabled vs GIL-enabled results for one package.

    Returns None if the package doesn't have GIL comparison data.

    Classification:
    - "ft_compatible": both modes pass 100%
    - "ft_intermittent": GIL-enabled passes 100%, GIL-disabled < 100%
    - "ft_incompatible": GIL-enabled passes 100%, GIL-disabled 0%
    - "pre_existing": both modes have failures
    - "ft_exacerbated": both have failures but GIL-disabled is worse
    - "gil_helps": GIL-disabled passes more than GIL-enabled (rare)
    """
    if result.gil_enabled_iterations is None:
        return None
    if result.gil_enabled_pass_rate is None:
        return None

    ft_rate = result.pass_rate
    gil_rate = result.gil_enabled_pass_rate

    if ft_rate == 1.0 and gil_rate == 1.0:
        classification = "ft_compatible"
    elif gil_rate == 1.0 and ft_rate == 0.0:
        classification = "ft_incompatible"
    elif gil_rate == 1.0 and ft_rate < 1.0:
        classification = "ft_intermittent"
    elif ft_rate > gil_rate:
        classification = "gil_helps"
    elif gil_rate < 1.0 and ft_rate < gil_rate:
        classification = "ft_exacerbated"
    elif gil_rate < 1.0:
        classification = "pre_existing"
    else:
        classification = "ft_compatible"

    ft_failures: Counter[str] = Counter()
    for it in result.iterations:
        for tid, status in it.test_results.items():
            if status in ("FAILED", "ERROR"):
                ft_failures[tid] += 1

    gil_failures: Counter[str] = Counter()
    for it in result.gil_enabled_iterations:
        for tid, status in it.test_results.items():
            if status in ("FAILED", "ERROR"):
                gil_failures[tid] += 1

    ft_only = sorted(set(ft_failures) - set(gil_failures))
    shared = sorted(set(ft_failures) & set(gil_failures))

    return GILComparisonResult(
        package=result.package,
        gil_disabled_pass_rate=ft_rate,
        gil_enabled_pass_rate=gil_rate,
        classification=classification,
        ft_specific_failing_tests=ft_only,
        shared_failing_tests=shared,
    )


# ---------------------------------------------------------------------------
# Triage prioritization
# ---------------------------------------------------------------------------


@dataclass
class TriageEntry:
    """A package ranked by investigation priority."""

    package: str
    priority_score: float
    category: str
    reason: str
    pass_rate: float
    crash_count: int = 0
    deadlock_count: int = 0
    flaky_test_count: int = 0
    is_pure_python: bool = True
    has_tsan_warnings: bool = False
    monthly_downloads: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "priority_score": round(self.priority_score, 2),
            "category": self.category,
            "reason": self.reason,
            "pass_rate": round(self.pass_rate, 4),
        }


def prioritize_triage(
    results: list[FTPackageResult],
    *,
    include_compatible: bool = False,
) -> list[TriageEntry]:
    """Rank packages by investigation urgency.

    Priority is based on:
    - Severity: crashes > deadlocks > intermittent > incompatible
    - Popularity: higher-download packages matter more
    - Fixability: intermittent with identifiable flaky tests is
      more actionable than random failures
    - Extension status: C extension bugs are harder to fix but
      more impactful

    Args:
        results: List of package results.
        include_compatible: If True, include compatible packages
            (usually excluded from triage).

    Returns:
        List of TriageEntry sorted by priority (highest first).
    """
    entries: list[TriageEntry] = []

    for r in results:
        if not include_compatible and r.category in (
            FailureCategory.COMPATIBLE,
            FailureCategory.COMPATIBLE_GIL_FALLBACK,
        ):
            continue

        score = 0.0
        reasons: list[str] = []

        severity_scores = {
            FailureCategory.CRASH: 50,
            FailureCategory.DEADLOCK: 45,
            FailureCategory.INTERMITTENT: 30,
            FailureCategory.INCOMPATIBLE: 25,
            FailureCategory.TSAN_WARNINGS: 20,
            FailureCategory.INSTALL_FAILURE: 10,
            FailureCategory.IMPORT_FAILURE: 10,
            FailureCategory.UNKNOWN: 5,
        }
        sev = severity_scores.get(r.category, 0)
        score += sev
        if sev > 0:
            reasons.append(f"severity:{r.category.value}")

        if r.crash_count > 0:
            score += min(r.crash_count * 5, 20)
            reasons.append(f"{r.crash_count} crashes")
        if r.deadlock_count > 0:
            score += min(r.deadlock_count * 5, 15)
            reasons.append(f"{r.deadlock_count} deadlocks")

        ext = r.extension_compat or {}
        is_pure = ext.get("is_pure_python", True)
        if not is_pure and r.category in (
            FailureCategory.CRASH,
            FailureCategory.INTERMITTENT,
            FailureCategory.TSAN_WARNINGS,
        ):
            score += 15
            reasons.append("C extension")

        if r.tsan_warning_iterations > 0:
            score += 10
            reasons.append(f"TSAN:{','.join(r.tsan_warning_types[:3])}")

        entries.append(
            TriageEntry(
                package=r.package,
                priority_score=score,
                category=r.category.value,
                reason="; ".join(reasons),
                pass_rate=r.pass_rate,
                crash_count=r.crash_count,
                deadlock_count=r.deadlock_count,
                flaky_test_count=len(r.flaky_tests),
                is_pure_python=is_pure,
                has_tsan_warnings=r.tsan_warning_iterations > 0,
            )
        )

    entries.sort(key=lambda e: -e.priority_score)
    return entries


# ---------------------------------------------------------------------------
# Duration anomaly detection
# ---------------------------------------------------------------------------


@dataclass
class DurationAnomaly:
    """A package whose timing suggests thread contention."""

    package: str
    mean_duration_s: float
    stdev_duration_s: float
    cv: float
    slowest_iteration: int
    slowest_duration_s: float
    anomaly_type: str


def detect_duration_anomalies(
    results: list[FTPackageResult],
    *,
    cv_threshold: float = 0.3,
) -> list[DurationAnomaly]:
    """Find packages with abnormal timing patterns.

    High variance in test duration across iterations can indicate
    thread contention -- the test suite runs fine when threads don't
    overlap but slows dramatically when they do.

    Progressive slowdown (each iteration slower than the last)
    suggests a resource leak or accumulating lock contention.

    Args:
        results: List of package results.
        cv_threshold: Coefficient of variation above which timing
            is flagged as anomalous (default 0.3 = 30%).

    Returns:
        List of DurationAnomaly for flagged packages.
    """
    from labeille.bench.stats import describe

    anomalies: list[DurationAnomaly] = []

    for r in results:
        durations = [it.duration_s for it in r.iterations if it.duration_s > 0]
        if len(durations) < 3:
            continue

        stats = describe(durations)
        if stats.cv < cv_threshold:
            continue

        slowest_idx = max(
            range(len(r.iterations)),
            key=lambda i: r.iterations[i].duration_s,
        )

        anomaly_type = "high_variance"
        if len(durations) >= 4:
            mid = len(durations) // 2
            first_half_mean = sum(durations[:mid]) / mid
            second_half_mean = sum(durations[mid:]) / (len(durations) - mid)
            if second_half_mean > first_half_mean * 1.2:
                anomaly_type = "progressive_slowdown"

        anomalies.append(
            DurationAnomaly(
                package=r.package,
                mean_duration_s=stats.mean,
                stdev_duration_s=stats.stdev,
                cv=stats.cv,
                slowest_iteration=slowest_idx + 1,
                slowest_duration_s=r.iterations[slowest_idx].duration_s,
                anomaly_type=anomaly_type,
            )
        )

    anomalies.sort(key=lambda a: -a.cv)
    return anomalies


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------


@dataclass
class FTAnalysisReport:
    """Complete analysis of a free-threading test run.

    Aggregates flakiness profiles, GIL comparisons, triage priority,
    and duration anomalies into one report.
    """

    total_packages: int = 0
    summary: dict[str, Any] = field(default_factory=dict)
    triage: list[TriageEntry] = field(default_factory=list)
    flaky_profiles: list[FlakinessProfile] = field(default_factory=list)
    gil_comparisons: list[GILComparisonResult] = field(default_factory=list)
    duration_anomalies: list[DurationAnomaly] = field(default_factory=list)

    ft_specific_failures: int = 0
    pre_existing_failures: int = 0
    most_common_crash_sigs: list[tuple[str, int]] = field(default_factory=list)
    most_common_tsan_types: list[tuple[str, int]] = field(default_factory=list)


def analyze_ft_run(
    results: list[FTPackageResult],
) -> FTAnalysisReport:
    """Run the full analysis pipeline on a set of results.

    Args:
        results: List of package results from an ft run.

    Returns:
        FTAnalysisReport with all analyses populated.
    """
    report = FTAnalysisReport(total_packages=len(results))

    report.summary = FTRunSummary.compute(results).to_dict()

    report.triage = prioritize_triage(results)

    for r in results:
        if r.category == FailureCategory.INTERMITTENT:
            report.flaky_profiles.append(analyze_flakiness(r))

    for r in results:
        if r.category == FailureCategory.CRASH and r.pass_count > 0:
            report.flaky_profiles.append(analyze_flakiness(r))

    for r in results:
        comp = compare_gil_modes(r)
        if comp is not None:
            report.gil_comparisons.append(comp)
            if comp.free_threading_specific:
                report.ft_specific_failures += 1
            elif comp.classification == "pre_existing":
                report.pre_existing_failures += 1

    report.duration_anomalies = detect_duration_anomalies(results)

    crash_sigs: Counter[str] = Counter()
    for r in results:
        for sig in r.failure_signatures:
            crash_sigs[sig] += 1
    report.most_common_crash_sigs = crash_sigs.most_common(10)

    tsan_types: Counter[str] = Counter()
    for r in results:
        for t in r.tsan_warning_types:
            tsan_types[t] += 1
    report.most_common_tsan_types = tsan_types.most_common(10)

    return report
