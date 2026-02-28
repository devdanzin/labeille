"""Tests for labeille.bench.compare — benchmark comparison analysis."""

from __future__ import annotations

import unittest

from bench_test_helpers import make_condition_result, make_meta, make_package_result

from labeille.bench.compare import (
    PackageOverhead,
    compare_conditions,
    compare_runs,
)
from labeille.bench.results import BenchPackageResult


class TestCompareConditionsBasic(unittest.TestCase):
    """Tests for compare_conditions core behaviour."""

    def test_compare_basic_overhead(self) -> None:
        """Treatment ~20% slower shows positive overhead."""
        results = [
            make_package_result(
                "pkg1",
                {
                    "baseline": [10.0, 10.1, 10.2, 10.0, 10.1],
                    "treatment": [12.0, 12.1, 12.2, 12.0, 12.1],
                },
            ),
            make_package_result(
                "pkg2",
                {
                    "baseline": [5.0, 5.1, 5.0, 5.1, 5.0],
                    "treatment": [6.0, 6.1, 6.0, 6.1, 6.0],
                },
            ),
            make_package_result(
                "pkg3",
                {
                    "baseline": [2.0, 2.0, 2.1, 2.0, 2.1],
                    "treatment": [2.4, 2.4, 2.5, 2.4, 2.5],
                },
            ),
        ]
        report = compare_conditions(results, "baseline", "treatment")
        self.assertEqual(report.total_packages, 3)
        self.assertGreater(report.median_overhead_pct, 0)
        self.assertGreater(report.mean_overhead_pct, 0)
        # All packages should be slower.
        self.assertEqual(report.slower_packages, 3)
        self.assertEqual(report.faster_packages, 0)

    def test_compare_no_difference(self) -> None:
        """Same wall times -> ~0% overhead, not significant."""
        results = [
            make_package_result(
                "pkg1",
                {
                    "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                    "treatment": [10.0, 10.0, 10.0, 10.0, 10.0],
                },
            ),
        ]
        report = compare_conditions(results, "baseline", "treatment")
        self.assertEqual(report.total_packages, 1)
        self.assertAlmostEqual(report.median_overhead_pct, 0.0, places=1)

    def test_compare_treatment_faster(self) -> None:
        """Treatment faster -> negative overhead."""
        results = [
            make_package_result(
                "pkg1",
                {
                    "baseline": [10.0, 10.1, 10.2, 10.0, 10.1],
                    "treatment": [8.0, 8.1, 8.0, 8.1, 8.0],
                },
            ),
        ]
        report = compare_conditions(results, "baseline", "treatment")
        self.assertLess(report.median_overhead_pct, 0)
        self.assertEqual(report.faster_packages, 1)
        self.assertEqual(report.slower_packages, 0)

    def test_compare_empty_results(self) -> None:
        """Empty results list -> report with total_packages=0."""
        report = compare_conditions([], "baseline", "treatment")
        self.assertEqual(report.total_packages, 0)
        self.assertEqual(report.reliable_packages, 0)
        self.assertEqual(len(report.packages), 0)

    def test_compare_skipped_excluded(self) -> None:
        """Skipped packages not counted."""
        results = [
            make_package_result(
                "active",
                {
                    "baseline": [10.0, 10.1, 10.0, 10.1, 10.0],
                    "treatment": [12.0, 12.1, 12.0, 12.1, 12.0],
                },
            ),
            BenchPackageResult(package="skipped", skipped=True, skip_reason="fail"),
        ]
        report = compare_conditions(results, "baseline", "treatment")
        self.assertEqual(report.total_packages, 1)

    def test_compare_missing_condition(self) -> None:
        """Package without one condition is skipped."""
        results = [
            make_package_result(
                "pkg1",
                {
                    "baseline": [10.0, 10.1, 10.0, 10.1, 10.0],
                },
            ),
        ]
        report = compare_conditions(results, "baseline", "treatment")
        self.assertEqual(report.total_packages, 0)


class TestCompareConditionsAnomalies(unittest.TestCase):
    """Tests for anomaly detection in compare_conditions."""

    def test_high_cv_flagged(self) -> None:
        """High variance baseline -> high_cv_baseline=True, reliable=False."""
        # Create a package with high-variance baseline.
        results = [
            make_package_result(
                "highcv",
                {
                    "baseline": [5.0, 10.0, 15.0, 5.0, 10.0],
                    "treatment": [12.0, 12.1, 12.0, 12.1, 12.0],
                },
            ),
        ]
        report = compare_conditions(results, "baseline", "treatment")
        self.assertEqual(report.total_packages, 1)
        pkg = report.packages[0]
        self.assertTrue(pkg.high_cv_baseline)
        self.assertFalse(pkg.reliable)
        self.assertEqual(report.high_cv_count, 1)

    def test_status_mismatch_flagged(self) -> None:
        """Different predominant status -> status_mismatch=True."""
        results = [
            make_package_result(
                "pkg1",
                {
                    "baseline": [10.0, 10.1, 10.0, 10.1, 10.0],
                },
            ),
        ]
        # Override treatment with "fail" status.
        results[0].conditions["treatment"] = make_condition_result(
            "treatment", [12.0, 12.1, 12.0, 12.1, 12.0], status="fail"
        )
        report = compare_conditions(results, "baseline", "treatment")
        self.assertEqual(report.total_packages, 1)
        pkg = report.packages[0]
        self.assertTrue(pkg.status_mismatch)
        self.assertEqual(report.status_mismatch_count, 1)

    def test_reliable_only_in_aggregate(self) -> None:
        """Aggregate stats computed from reliable packages only."""
        results = [
            # 3 reliable packages with ~20% overhead.
            make_package_result(
                "r1",
                {
                    "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                    "treatment": [12.0, 12.0, 12.0, 12.0, 12.0],
                },
            ),
            make_package_result(
                "r2",
                {
                    "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                    "treatment": [12.0, 12.0, 12.0, 12.0, 12.0],
                },
            ),
            make_package_result(
                "r3",
                {
                    "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                    "treatment": [12.0, 12.0, 12.0, 12.0, 12.0],
                },
            ),
            # 2 unreliable packages with very different overhead.
            make_package_result(
                "u1",
                {
                    "baseline": [1.0, 20.0, 1.0, 20.0, 1.0],
                    "treatment": [50.0, 50.0, 50.0, 50.0, 50.0],
                },
            ),
            make_package_result(
                "u2",
                {
                    "baseline": [1.0, 20.0, 1.0, 20.0, 1.0],
                    "treatment": [50.0, 50.0, 50.0, 50.0, 50.0],
                },
            ),
        ]
        report = compare_conditions(results, "baseline", "treatment")
        self.assertEqual(report.total_packages, 5)
        self.assertEqual(report.reliable_packages, 3)
        self.assertEqual(report.unreliable_packages, 2)
        # Median/mean should be based on reliable packages only (~20%).
        self.assertAlmostEqual(report.median_overhead_pct, 20.0, places=0)


class TestCompareConditionsCounts(unittest.TestCase):
    """Tests for aggregate count accuracy."""

    def test_aggregate_counts(self) -> None:
        """Verify total_packages, faster_packages, slower_packages."""
        results = [
            make_package_result(
                "faster",
                {
                    "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                    "treatment": [8.0, 8.0, 8.0, 8.0, 8.0],
                },
            ),
            make_package_result(
                "slower",
                {
                    "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                    "treatment": [15.0, 15.0, 15.0, 15.0, 15.0],
                },
            ),
        ]
        report = compare_conditions(results, "baseline", "treatment")
        self.assertEqual(report.total_packages, 2)
        self.assertEqual(report.faster_packages, 1)
        self.assertEqual(report.slower_packages, 1)

    def test_most_affected(self) -> None:
        """most_affected sorted by overhead descending, limited to 5."""
        results = []
        for i in range(8):
            overhead_factor = 1.0 + (i + 1) * 0.1
            results.append(
                make_package_result(
                    f"pkg{i}",
                    {
                        "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                        "treatment": [10.0 * overhead_factor] * 5,
                    },
                )
            )
        report = compare_conditions(results, "baseline", "treatment")
        most = report.most_affected
        self.assertLessEqual(len(most), 5)
        # Should be sorted descending.
        pcts = [p.overhead.overhead_pct for p in most]
        self.assertEqual(pcts, sorted(pcts, reverse=True))

    def test_most_improved(self) -> None:
        """most_improved: only negative overhead, sorted ascending, limited to 5."""
        results = []
        for i in range(8):
            factor = 1.0 - (i + 1) * 0.05
            results.append(
                make_package_result(
                    f"pkg{i}",
                    {
                        "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                        "treatment": [10.0 * factor] * 5,
                    },
                )
            )
        report = compare_conditions(results, "baseline", "treatment")
        improved = report.most_improved
        self.assertLessEqual(len(improved), 5)
        for p in improved:
            self.assertLess(p.overhead.overhead_pct, 0)
        pcts = [p.overhead.overhead_pct for p in improved]
        self.assertEqual(pcts, sorted(pcts))


class TestCompareRuns(unittest.TestCase):
    """Tests for compare_runs."""

    def test_compare_runs_common_packages(self) -> None:
        """Only common packages are compared."""
        meta_a = make_meta(name="Run A", conditions=["cond_a"])
        results_a = [
            make_package_result("a", {"cond_a": [10.0, 10.0, 10.0, 10.0, 10.0]}),
            make_package_result("b", {"cond_a": [10.0, 10.0, 10.0, 10.0, 10.0]}),
            make_package_result("c", {"cond_a": [10.0, 10.0, 10.0, 10.0, 10.0]}),
        ]

        meta_b = make_meta(name="Run B", conditions=["cond_b"])
        results_b = [
            make_package_result("b", {"cond_b": [12.0, 12.0, 12.0, 12.0, 12.0]}),
            make_package_result("c", {"cond_b": [12.0, 12.0, 12.0, 12.0, 12.0]}),
            make_package_result("d", {"cond_b": [12.0, 12.0, 12.0, 12.0, 12.0]}),
        ]

        report = compare_runs((meta_a, results_a), (meta_b, results_b))
        self.assertEqual(report.total_packages, 2)
        pkg_names = {p.package for p in report.packages}
        self.assertEqual(pkg_names, {"b", "c"})

    def test_compare_runs_default_condition(self) -> None:
        """Uses the first condition by default."""
        meta_a = make_meta(name="Run A", conditions=["baseline"])
        results_a = [
            make_package_result("pkg", {"baseline": [10.0, 10.0, 10.0, 10.0, 10.0]}),
        ]

        meta_b = make_meta(name="Run B", conditions=["treatment"])
        results_b = [
            make_package_result("pkg", {"treatment": [12.0, 12.0, 12.0, 12.0, 12.0]}),
        ]

        report = compare_runs((meta_a, results_a), (meta_b, results_b))
        self.assertEqual(report.total_packages, 1)
        self.assertEqual(report.baseline_name, "Run A")
        self.assertEqual(report.treatment_name, "Run B")

    def test_compare_runs_specified_conditions(self) -> None:
        """Explicit condition names are used."""
        meta_a = make_meta(name="Run A", conditions=["c1", "c2"])
        results_a = [
            make_package_result(
                "pkg",
                {
                    "c1": [10.0, 10.0, 10.0, 10.0, 10.0],
                    "c2": [8.0, 8.0, 8.0, 8.0, 8.0],
                },
            ),
        ]

        meta_b = make_meta(name="Run B", conditions=["c3", "c4"])
        results_b = [
            make_package_result(
                "pkg",
                {
                    "c3": [12.0, 12.0, 12.0, 12.0, 12.0],
                    "c4": [15.0, 15.0, 15.0, 15.0, 15.0],
                },
            ),
        ]

        report = compare_runs(
            (meta_a, results_a),
            (meta_b, results_b),
            condition_a="c2",
            condition_b="c4",
        )
        self.assertEqual(report.total_packages, 1)
        # Treatment (c4=15) is slower than baseline (c2=8).
        self.assertGreater(report.median_overhead_pct, 0)


class TestPackageOverheadProperties(unittest.TestCase):
    """Tests for PackageOverhead property methods."""

    def _make_pkg_overhead(
        self,
        *,
        high_cv_b: bool = False,
        high_cv_t: bool = False,
    ) -> PackageOverhead:
        """Create a PackageOverhead with controllable flags."""
        from labeille.bench.stats import compute_overhead

        overhead = compute_overhead(
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [12.0, 12.0, 12.0, 12.0, 12.0],
            ci_seed=42,
        )
        return PackageOverhead(
            package="test",
            overhead=overhead,
            baseline_condition="baseline",
            treatment_condition="treatment",
            high_cv_baseline=high_cv_b,
            high_cv_treatment=high_cv_t,
        )

    def test_reliable_no_high_cv(self) -> None:
        """No high CV -> reliable=True."""
        po = self._make_pkg_overhead()
        self.assertTrue(po.reliable)

    def test_unreliable_baseline(self) -> None:
        """high_cv_baseline=True -> reliable=False."""
        po = self._make_pkg_overhead(high_cv_b=True)
        self.assertFalse(po.reliable)

    def test_unreliable_treatment(self) -> None:
        """high_cv_treatment=True -> reliable=False."""
        po = self._make_pkg_overhead(high_cv_t=True)
        self.assertFalse(po.reliable)

    def test_significant_and_reliable(self) -> None:
        """Significant, practical, no high CV -> True."""
        po = self._make_pkg_overhead()
        # Zero-variance samples with different means produce significant results.
        if po.overhead.practically_significant:
            self.assertTrue(po.significant_and_reliable)

    def test_significant_but_unreliable(self) -> None:
        """Significant but high CV -> significant_and_reliable=False."""
        po = self._make_pkg_overhead(high_cv_b=True)
        self.assertFalse(po.significant_and_reliable)


if __name__ == "__main__":
    unittest.main()
