"""Tests for labeille.ft.compare — cross-run comparison."""

from __future__ import annotations

import unittest

from labeille.ft.compare import (
    PackageTransition,
    _classify_transition,
    compare_ft_runs,
)
from labeille.ft.results import FailureCategory, FTPackageResult


def _make_result(
    package: str = "pkg",
    category: FailureCategory = FailureCategory.COMPATIBLE,
    pass_rate: float = 1.0,
    crash_count: int = 0,
    deadlock_count: int = 0,
    **kwargs: object,
) -> FTPackageResult:
    """Create a minimal FTPackageResult for comparison tests."""
    if "pass_count" not in kwargs:
        kwargs["pass_count"] = int(pass_rate * 10)
    if "iterations_completed" not in kwargs:
        kwargs["iterations_completed"] = 10
    return FTPackageResult(
        package=package,
        category=category,
        pass_rate=pass_rate,
        crash_count=crash_count,
        deadlock_count=deadlock_count,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# PackageTransition tests
# ---------------------------------------------------------------------------


class TestPackageTransition(unittest.TestCase):
    def test_transition_to_dict(self) -> None:
        t = PackageTransition(
            package="numpy",
            old_category="crash",
            new_category="compatible",
            old_pass_rate=0.7,
            new_pass_rate=1.0,
            pass_rate_delta=0.3,
            transition_type="improvement",
            detail="crash → compatible; pass rate 70% → 100%",
        )
        d = t.to_dict()
        self.assertEqual(d["package"], "numpy")
        self.assertEqual(d["old_category"], "crash")
        self.assertEqual(d["new_category"], "compatible")
        self.assertEqual(d["transition_type"], "improvement")
        self.assertAlmostEqual(d["pass_rate_delta"], 0.3, places=4)


# ---------------------------------------------------------------------------
# compare_ft_runs tests
# ---------------------------------------------------------------------------


class TestCompareFtRuns(unittest.TestCase):
    def test_compare_identical_runs(self) -> None:
        results = [
            _make_result("a", FailureCategory.COMPATIBLE, 1.0),
            _make_result("b", FailureCategory.CRASH, 0.5, crash_count=5),
        ]
        comp = compare_ft_runs(results, results)
        self.assertEqual(len(comp.improvements), 0)
        self.assertEqual(len(comp.regressions), 0)
        self.assertEqual(comp.unchanged, 2)

    def test_compare_improvement(self) -> None:
        a = [_make_result("numpy", FailureCategory.CRASH, 0.7, crash_count=3)]
        b = [_make_result("numpy", FailureCategory.COMPATIBLE, 1.0)]
        comp = compare_ft_runs(a, b)
        self.assertEqual(len(comp.improvements), 1)
        self.assertEqual(comp.improvements[0].package, "numpy")
        self.assertEqual(comp.improvements[0].old_category, "crash")
        self.assertEqual(comp.improvements[0].new_category, "compatible")

    def test_compare_regression(self) -> None:
        a = [_make_result("crypto", FailureCategory.COMPATIBLE, 1.0)]
        b = [_make_result("crypto", FailureCategory.CRASH, 0.3, crash_count=7)]
        comp = compare_ft_runs(a, b)
        self.assertEqual(len(comp.regressions), 1)
        self.assertEqual(comp.regressions[0].package, "crypto")

    def test_compare_pass_rate_improvement(self) -> None:
        a = [_make_result("aiohttp", FailureCategory.INTERMITTENT, 0.5)]
        b = [_make_result("aiohttp", FailureCategory.INTERMITTENT, 0.9)]
        comp = compare_ft_runs(a, b)
        self.assertEqual(len(comp.improvements), 1)

    def test_compare_pass_rate_noise(self) -> None:
        a = [_make_result("aiohttp", FailureCategory.INTERMITTENT, 0.82)]
        b = [_make_result("aiohttp", FailureCategory.INTERMITTENT, 0.85)]
        comp = compare_ft_runs(a, b)
        self.assertEqual(comp.unchanged, 1)

    def test_compare_new_package(self) -> None:
        a = [_make_result("existing")]
        b = [_make_result("existing"), _make_result("new-pkg")]
        comp = compare_ft_runs(a, b)
        self.assertIn("new-pkg", comp.packages_only_in_b)

    def test_compare_removed_package(self) -> None:
        a = [_make_result("existing"), _make_result("old-pkg")]
        b = [_make_result("existing")]
        comp = compare_ft_runs(a, b)
        self.assertIn("old-pkg", comp.packages_only_in_a)

    def test_compare_sorted_by_magnitude(self) -> None:
        a = [
            _make_result("small", FailureCategory.CRASH, 0.8, crash_count=2),
            _make_result("big", FailureCategory.CRASH, 0.3, crash_count=7),
        ]
        b = [
            _make_result("small", FailureCategory.COMPATIBLE, 1.0),
            _make_result("big", FailureCategory.COMPATIBLE, 1.0),
        ]
        comp = compare_ft_runs(a, b)
        self.assertEqual(len(comp.improvements), 2)
        # Biggest delta first.
        self.assertEqual(comp.improvements[0].package, "big")

    def test_compare_aggregate_counts(self) -> None:
        a = [
            _make_result("a", FailureCategory.COMPATIBLE, 1.0),
            _make_result("b", FailureCategory.CRASH, 0.5, crash_count=5),
        ]
        b = [
            _make_result("a", FailureCategory.COMPATIBLE, 1.0),
            _make_result("b", FailureCategory.COMPATIBLE, 1.0),
        ]
        comp = compare_ft_runs(a, b)
        self.assertEqual(comp.compatible_count_a, 1)
        self.assertEqual(comp.compatible_count_b, 2)
        self.assertEqual(comp.net_improvement, 1)
        self.assertEqual(comp.crash_count_a, 1)
        self.assertEqual(comp.crash_count_b, 0)

    def test_compare_mixed_transitions(self) -> None:
        a = [
            _make_result("imp1", FailureCategory.CRASH, 0.5, crash_count=5),
            _make_result("imp2", FailureCategory.CRASH, 0.7, crash_count=3),
            _make_result("imp3", FailureCategory.INCOMPATIBLE, 0.0, pass_count=0),
            _make_result("reg1", FailureCategory.COMPATIBLE, 1.0),
            _make_result("reg2", FailureCategory.COMPATIBLE, 1.0),
            _make_result("unch1", FailureCategory.COMPATIBLE, 1.0),
            _make_result("unch2", FailureCategory.COMPATIBLE, 1.0),
            _make_result("unch3", FailureCategory.CRASH, 0.5, crash_count=5),
            _make_result("unch4", FailureCategory.COMPATIBLE, 1.0),
            _make_result("unch5", FailureCategory.COMPATIBLE, 1.0),
        ]
        b = [
            _make_result("imp1", FailureCategory.COMPATIBLE, 1.0),
            _make_result("imp2", FailureCategory.COMPATIBLE, 1.0),
            _make_result("imp3", FailureCategory.COMPATIBLE, 1.0),
            _make_result("reg1", FailureCategory.CRASH, 0.3, crash_count=7),
            _make_result("reg2", FailureCategory.CRASH, 0.5, crash_count=5),
            _make_result("unch1", FailureCategory.COMPATIBLE, 1.0),
            _make_result("unch2", FailureCategory.COMPATIBLE, 1.0),
            _make_result("unch3", FailureCategory.CRASH, 0.5, crash_count=5),
            _make_result("unch4", FailureCategory.COMPATIBLE, 1.0),
            _make_result("unch5", FailureCategory.COMPATIBLE, 1.0),
        ]
        comp = compare_ft_runs(a, b)
        self.assertEqual(len(comp.improvements), 3)
        self.assertEqual(len(comp.regressions), 2)
        self.assertEqual(comp.unchanged, 5)


# ---------------------------------------------------------------------------
# _classify_transition tests
# ---------------------------------------------------------------------------


class TestClassifyTransition(unittest.TestCase):
    def test_classify_category_improvement(self) -> None:
        ra = _make_result("pkg", FailureCategory.CRASH, 0.5, crash_count=5)
        rb = _make_result("pkg", FailureCategory.COMPATIBLE, 1.0)
        self.assertEqual(_classify_transition(ra, rb), "improvement")

    def test_classify_category_regression(self) -> None:
        ra = _make_result("pkg", FailureCategory.COMPATIBLE, 1.0)
        rb = _make_result("pkg", FailureCategory.CRASH, 0.5, crash_count=5)
        self.assertEqual(_classify_transition(ra, rb), "regression")

    def test_classify_same_category_rate_up(self) -> None:
        ra = _make_result("pkg", FailureCategory.INTERMITTENT, 0.5)
        rb = _make_result("pkg", FailureCategory.INTERMITTENT, 0.9)
        self.assertEqual(_classify_transition(ra, rb), "improvement")

    def test_classify_same_category_rate_stable(self) -> None:
        ra = _make_result("pkg", FailureCategory.INTERMITTENT, 0.82)
        rb = _make_result("pkg", FailureCategory.INTERMITTENT, 0.85)
        self.assertEqual(_classify_transition(ra, rb), "unchanged")


if __name__ == "__main__":
    unittest.main()
