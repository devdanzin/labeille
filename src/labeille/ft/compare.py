"""Cross-run comparison for free-threading test results.

Compares two free-threading test runs to track:
- Category transitions (crash -> compatible, compatible -> crash)
- Pass rate changes
- New/resolved crashes and deadlocks
- Packages added or removed between runs

Designed for tracking ecosystem compatibility across CPython
releases (e.g., 3.14a1 -> 3.14b2) or over time on the same build.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("labeille")


@dataclass
class PackageTransition:
    """A change in a package's status between two runs."""

    package: str
    old_category: str
    new_category: str
    old_pass_rate: float
    new_pass_rate: float
    pass_rate_delta: float  # new - old
    transition_type: str  # "improvement", "regression", "unchanged"
    detail: str = ""  # Human-readable description

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "old_category": self.old_category,
            "new_category": self.new_category,
            "old_pass_rate": round(self.old_pass_rate, 4),
            "new_pass_rate": round(self.new_pass_rate, 4),
            "pass_rate_delta": round(self.pass_rate_delta, 4),
            "transition_type": self.transition_type,
            "detail": self.detail,
        }


@dataclass
class FTComparisonResult:
    """Complete comparison of two free-threading test runs."""

    label_a: str = ""
    label_b: str = ""

    # Packages present in both runs.
    common_packages: int = 0
    # Only in run A or B.
    packages_only_in_a: list[str] = field(default_factory=list)
    packages_only_in_b: list[str] = field(default_factory=list)

    # Transitions.
    improvements: list[PackageTransition] = field(default_factory=list)
    regressions: list[PackageTransition] = field(default_factory=list)
    unchanged: int = 0

    # Aggregate deltas.
    compatible_count_a: int = 0
    compatible_count_b: int = 0
    crash_count_a: int = 0
    crash_count_b: int = 0
    intermittent_count_a: int = 0
    intermittent_count_b: int = 0

    # Summary stats from each run.
    summary_a: dict[str, Any] = field(default_factory=dict)
    summary_b: dict[str, Any] = field(default_factory=dict)

    @property
    def net_improvement(self) -> int:
        """Net change in compatible package count."""
        return self.compatible_count_b - self.compatible_count_a

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "common_packages": self.common_packages,
            "packages_only_in_a": self.packages_only_in_a,
            "packages_only_in_b": self.packages_only_in_b,
            "improvements": [t.to_dict() for t in self.improvements],
            "regressions": [t.to_dict() for t in self.regressions],
            "unchanged": self.unchanged,
            "compatible_count_a": self.compatible_count_a,
            "compatible_count_b": self.compatible_count_b,
            "net_improvement": self.net_improvement,
        }


def compare_ft_runs(
    results_a: list[Any],
    results_b: list[Any],
    *,
    label_a: str = "run_a",
    label_b: str = "run_b",
) -> FTComparisonResult:
    """Compare two free-threading test runs.

    Compares per-package categories and pass rates, classifying
    each change as an improvement, regression, or unchanged.

    Args:
        results_a: Package results from the older/baseline run.
        results_b: Package results from the newer run.
        label_a: Human label for run A.
        label_b: Human label for run B.

    Returns:
        FTComparisonResult with all transitions and statistics.
    """
    from labeille.ft.results import FTRunSummary

    comp = FTComparisonResult(label_a=label_a, label_b=label_b)

    # Index results by package name.
    map_a = {r.package: r for r in results_a}
    map_b = {r.package: r for r in results_b}

    comp.packages_only_in_a = sorted(set(map_a) - set(map_b))
    comp.packages_only_in_b = sorted(set(map_b) - set(map_a))
    common = set(map_a) & set(map_b)
    comp.common_packages = len(common)

    # Compare common packages.
    for pkg_name in sorted(common):
        ra = map_a[pkg_name]
        rb = map_b[pkg_name]

        delta = rb.pass_rate - ra.pass_rate
        transition = _classify_transition(ra, rb)

        if transition == "improvement":
            detail = _describe_transition(ra, rb, "improved")
            comp.improvements.append(
                PackageTransition(
                    package=pkg_name,
                    old_category=ra.category.value,
                    new_category=rb.category.value,
                    old_pass_rate=ra.pass_rate,
                    new_pass_rate=rb.pass_rate,
                    pass_rate_delta=delta,
                    transition_type="improvement",
                    detail=detail,
                )
            )
        elif transition == "regression":
            detail = _describe_transition(ra, rb, "regressed")
            comp.regressions.append(
                PackageTransition(
                    package=pkg_name,
                    old_category=ra.category.value,
                    new_category=rb.category.value,
                    old_pass_rate=ra.pass_rate,
                    new_pass_rate=rb.pass_rate,
                    pass_rate_delta=delta,
                    transition_type="regression",
                    detail=detail,
                )
            )
        else:
            comp.unchanged += 1

    # Sort by magnitude of change.
    comp.improvements.sort(key=lambda t: -abs(t.pass_rate_delta))
    comp.regressions.sort(key=lambda t: -abs(t.pass_rate_delta))

    # Aggregate category counts.
    summary_a = FTRunSummary.compute(results_a)
    summary_b = FTRunSummary.compute(results_b)
    comp.summary_a = summary_a.to_dict()
    comp.summary_b = summary_b.to_dict()

    cats_a = summary_a.categories
    cats_b = summary_b.categories

    comp.compatible_count_a = cats_a.get("compatible", 0) + cats_a.get(
        "compatible_gil_fallback", 0
    )
    comp.compatible_count_b = cats_b.get("compatible", 0) + cats_b.get(
        "compatible_gil_fallback", 0
    )
    comp.crash_count_a = cats_a.get("crash", 0)
    comp.crash_count_b = cats_b.get("crash", 0)
    comp.intermittent_count_a = cats_a.get("intermittent", 0)
    comp.intermittent_count_b = cats_b.get("intermittent", 0)

    return comp


def _classify_transition(ra: Any, rb: Any) -> str:
    """Classify a transition as improvement, regression, or unchanged.

    Uses category severity and pass rate to determine direction.
    A change from a worse category to a better one is an improvement.
    A significant pass rate increase within the same category is
    also an improvement.
    """
    sev_a = ra.category.severity
    sev_b = rb.category.severity

    # Category changed.
    if sev_a != sev_b:
        if sev_b < sev_a:
            return "improvement"
        else:
            return "regression"

    # Same category. Check pass rate change.
    delta = rb.pass_rate - ra.pass_rate
    if abs(delta) < 0.05:
        return "unchanged"  # Less than 5% change — noise.
    if delta > 0:
        return "improvement"
    return "regression"


def _describe_transition(ra: Any, rb: Any, direction: str) -> str:
    """Generate a human-readable description of a transition."""
    parts = [f"{ra.category.value} → {rb.category.value}"]

    if ra.pass_rate != rb.pass_rate:
        parts.append(f"pass rate {ra.pass_rate * 100:.0f}% → {rb.pass_rate * 100:.0f}%")

    # Note specific changes.
    if ra.crash_count > 0 and rb.crash_count == 0:
        parts.append("crashes resolved")
    elif rb.crash_count > 0 and ra.crash_count == 0:
        parts.append("new crashes")

    if ra.deadlock_count > 0 and rb.deadlock_count == 0:
        parts.append("deadlocks resolved")
    elif rb.deadlock_count > 0 and ra.deadlock_count == 0:
        parts.append("new deadlocks")

    return "; ".join(parts)
