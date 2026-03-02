"""Tests for labeille.ft.analysis — deep free-threading analysis."""

from __future__ import annotations

import unittest

from ft_test_helpers import make_iteration, make_package_result

from labeille.ft.analysis import (
    analyze_flakiness,
    analyze_ft_run,
    compare_gil_modes,
    detect_duration_anomalies,
    prioritize_triage,
)
from labeille.ft.results import (
    FTPackageResult,
)


# ---------------------------------------------------------------------------
# Flakiness profile tests
# ---------------------------------------------------------------------------


class TestAnalyzeFlakiness(unittest.TestCase):
    def test_flakiness_all_pass(self) -> None:
        r = make_package_result("pkg", ["pass"] * 5)
        profile = analyze_flakiness(r)
        self.assertEqual(profile.pass_rate, 1.0)
        self.assertEqual(profile.failure_modes, {})

    def test_flakiness_mixed(self) -> None:
        r = make_package_result("pkg", ["pass", "pass", "pass", "fail", "fail"])
        profile = analyze_flakiness(r)
        self.assertEqual(profile.failure_modes, {"fail": 2})
        self.assertAlmostEqual(profile.pass_rate, 0.6)

    def test_flakiness_with_crashes(self) -> None:
        r = make_package_result("pkg", ["pass", "pass", "pass", "crash", "fail"])
        profile = analyze_flakiness(r)
        self.assertEqual(profile.failure_modes.get("crash"), 1)
        self.assertEqual(profile.failure_modes.get("fail"), 1)

    def test_flakiness_consecutive_streaks(self) -> None:
        r = make_package_result("pkg", ["pass", "pass", "pass", "fail", "fail"])
        profile = analyze_flakiness(r)
        self.assertEqual(profile.max_consecutive_passes, 3)
        self.assertEqual(profile.max_consecutive_failures, 2)

    def test_flakiness_per_test_detection(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(
                    0, "pass", test_results={"test_foo": "PASSED", "test_bar": "PASSED"}
                ),
                make_iteration(
                    1, "fail", test_results={"test_foo": "FAILED", "test_bar": "PASSED"}
                ),
                make_iteration(
                    2, "pass", test_results={"test_foo": "PASSED", "test_bar": "PASSED"}
                ),
                make_iteration(
                    3, "fail", test_results={"test_foo": "FAILED", "test_bar": "FAILED"}
                ),
                make_iteration(
                    4, "pass", test_results={"test_foo": "PASSED", "test_bar": "PASSED"}
                ),
            ],
        )
        r.compute_aggregates()
        profile = analyze_flakiness(r)
        self.assertEqual(len(profile.flaky_tests), 2)
        # test_foo has higher fail rate (2/5) than test_bar (1/5).
        self.assertEqual(profile.flaky_tests[0].test_id, "test_foo")
        self.assertEqual(profile.flaky_tests[0].fail_count, 2)

    def test_flakiness_consistent_failure_not_flaky(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(i, "fail", test_results={"test_baz": "FAILED"}) for i in range(5)
            ],
        )
        r.compute_aggregates()
        profile = analyze_flakiness(r)
        # test_baz fails in ALL iterations, so it's not flaky.
        self.assertEqual(len(profile.flaky_tests), 0)

    def test_pattern_consistent_test(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass", test_results={"test_a": "PASSED"}),
                make_iteration(1, "fail", test_results={"test_a": "FAILED"}),
                make_iteration(2, "pass", test_results={"test_a": "PASSED"}),
                make_iteration(3, "fail", test_results={"test_a": "FAILED"}),
                make_iteration(4, "pass", test_results={"test_a": "PASSED"}),
            ],
        )
        r.compute_aggregates()
        profile = analyze_flakiness(r)
        self.assertEqual(profile.pattern, "consistent_test")

    def test_pattern_random_test(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass", test_results={"test_a": "PASSED", "test_b": "PASSED"}),
                make_iteration(1, "fail", test_results={"test_a": "FAILED", "test_b": "PASSED"}),
                make_iteration(2, "pass", test_results={"test_a": "PASSED", "test_b": "PASSED"}),
                make_iteration(3, "fail", test_results={"test_a": "PASSED", "test_b": "FAILED"}),
                make_iteration(4, "pass", test_results={"test_a": "PASSED", "test_b": "PASSED"}),
            ],
        )
        r.compute_aggregates()
        profile = analyze_flakiness(r)
        self.assertEqual(profile.pattern, "random_test")

    def test_pattern_consistent_crash(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass"),
                make_iteration(1, "crash", crash_signature="sig_a"),
                make_iteration(2, "pass"),
                make_iteration(3, "crash", crash_signature="sig_a"),
            ],
        )
        r.compute_aggregates()
        profile = analyze_flakiness(r)
        self.assertEqual(profile.pattern, "consistent_crash")

    def test_pattern_varied_crash(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass"),
                make_iteration(1, "crash", crash_signature="sig_a"),
                make_iteration(2, "crash", crash_signature="sig_b"),
            ],
        )
        r.compute_aggregates()
        profile = analyze_flakiness(r)
        self.assertEqual(profile.pattern, "varied_crash")

    def test_pattern_unknown_no_test_data(self) -> None:
        r = make_package_result("pkg", ["pass", "fail", "pass"])
        profile = analyze_flakiness(r)
        self.assertEqual(profile.pattern, "unknown")

    def test_flakiness_empty_iterations(self) -> None:
        r = FTPackageResult(package="pkg")
        profile = analyze_flakiness(r)
        self.assertEqual(profile.total_iterations, 0)
        self.assertEqual(profile.failure_modes, {})


# ---------------------------------------------------------------------------
# GIL comparison tests
# ---------------------------------------------------------------------------


class TestCompareGilModes(unittest.TestCase):
    def _make_result_with_gil(
        self,
        ft_statuses: list[str],
        gil_statuses: list[str],
        ft_test_results: list[dict[str, str]] | None = None,
        gil_test_results: list[dict[str, str]] | None = None,
    ) -> FTPackageResult:
        ft_iters = [
            make_iteration(
                i,
                s,
                test_results=ft_test_results[i] if ft_test_results else {},
            )
            for i, s in enumerate(ft_statuses)
        ]
        gil_iters = [
            make_iteration(
                i,
                s,
                test_results=gil_test_results[i] if gil_test_results else {},
            )
            for i, s in enumerate(gil_statuses)
        ]
        r = FTPackageResult(
            package="pkg",
            iterations=ft_iters,
            gil_enabled_iterations=gil_iters,
        )
        r.compute_aggregates()
        return r

    def test_compare_ft_compatible(self) -> None:
        r = self._make_result_with_gil(["pass"] * 5, ["pass"] * 5)
        comp = compare_gil_modes(r)
        assert comp is not None
        self.assertEqual(comp.classification, "ft_compatible")

    def test_compare_ft_intermittent(self) -> None:
        r = self._make_result_with_gil(
            ["pass", "pass", "fail", "pass", "pass"],
            ["pass"] * 5,
        )
        comp = compare_gil_modes(r)
        assert comp is not None
        self.assertEqual(comp.classification, "ft_intermittent")
        self.assertTrue(comp.free_threading_specific)

    def test_compare_ft_incompatible(self) -> None:
        r = self._make_result_with_gil(["fail"] * 5, ["pass"] * 5)
        comp = compare_gil_modes(r)
        assert comp is not None
        self.assertEqual(comp.classification, "ft_incompatible")

    def test_compare_pre_existing(self) -> None:
        r = self._make_result_with_gil(
            ["pass", "pass", "fail", "pass", "pass"],
            ["pass", "fail", "pass", "pass", "pass"],
        )
        comp = compare_gil_modes(r)
        assert comp is not None
        self.assertEqual(comp.classification, "pre_existing")

    def test_compare_ft_exacerbated(self) -> None:
        r = self._make_result_with_gil(
            ["pass", "fail", "fail", "fail", "pass"],
            ["pass", "pass", "pass", "fail", "pass"],
        )
        comp = compare_gil_modes(r)
        assert comp is not None
        self.assertEqual(comp.classification, "ft_exacerbated")

    def test_compare_no_data(self) -> None:
        r = make_package_result("pkg", ["pass"] * 5)
        comp = compare_gil_modes(r)
        self.assertIsNone(comp)

    def test_compare_ft_specific_tests(self) -> None:
        ft_results = [{"test_x": "FAILED"}] * 5
        gil_results = [{"test_x": "PASSED"}] * 5
        r = self._make_result_with_gil(
            ["fail"] * 5,
            ["pass"] * 5,
            ft_test_results=ft_results,
            gil_test_results=gil_results,
        )
        comp = compare_gil_modes(r)
        assert comp is not None
        self.assertIn("test_x", comp.ft_specific_failing_tests)

    def test_compare_shared_tests(self) -> None:
        ft_results = [{"test_y": "FAILED"}] * 5
        gil_results = [{"test_y": "FAILED"}] * 5
        r = self._make_result_with_gil(
            ["fail"] * 5,
            ["fail"] * 5,
            ft_test_results=ft_results,
            gil_test_results=gil_results,
        )
        comp = compare_gil_modes(r)
        assert comp is not None
        self.assertIn("test_y", comp.shared_failing_tests)


# ---------------------------------------------------------------------------
# Triage prioritization tests
# ---------------------------------------------------------------------------


class TestPrioritizeTriage(unittest.TestCase):
    def test_triage_crashes_highest(self) -> None:
        results = [
            make_package_result("crash_pkg", ["pass", "pass", "crash"]),
            make_package_result("inter_pkg", ["pass", "fail", "pass"]),
        ]
        triage = prioritize_triage(results)
        self.assertEqual(triage[0].package, "crash_pkg")
        self.assertGreater(triage[0].priority_score, triage[1].priority_score)

    def test_triage_deadlock_high(self) -> None:
        results = [
            make_package_result("dead_pkg", ["pass", "deadlock"]),
            make_package_result("inter_pkg", ["pass", "fail", "pass"]),
        ]
        triage = prioritize_triage(results)
        self.assertEqual(triage[0].package, "dead_pkg")

    def test_triage_compatible_excluded(self) -> None:
        results = [
            make_package_result("ok_pkg", ["pass"] * 5),
            make_package_result("crash_pkg", ["pass", "crash"]),
        ]
        triage = prioritize_triage(results)
        packages = [e.package for e in triage]
        self.assertNotIn("ok_pkg", packages)
        self.assertIn("crash_pkg", packages)

    def test_triage_extension_bonus(self) -> None:
        results = [
            make_package_result(
                "ext_crash",
                ["pass", "crash"],
                extension_compat={"is_pure_python": False},
            ),
            make_package_result("py_crash", ["pass", "crash"]),
        ]
        triage = prioritize_triage(results)
        self.assertEqual(triage[0].package, "ext_crash")
        self.assertGreater(triage[0].priority_score, triage[1].priority_score)

    def test_triage_tsan_bonus(self) -> None:
        r = FTPackageResult(
            package="tsan_pkg",
            iterations=[make_iteration(i, "pass", tsan_warnings=["data race"]) for i in range(5)],
        )
        r.compute_aggregates()
        r.categorize()
        triage = prioritize_triage([r])
        self.assertEqual(len(triage), 1)
        self.assertTrue(triage[0].has_tsan_warnings)

    def test_triage_sorted_by_score(self) -> None:
        results = [
            make_package_result("low", ["pass", "fail"]),
            make_package_result("high", ["pass", "crash", "crash"]),
            make_package_result("mid", ["fail"] * 5),
        ]
        triage = prioritize_triage(results)
        scores = [e.priority_score for e in triage]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_triage_includes_compatible_when_requested(self) -> None:
        results = [make_package_result("ok_pkg", ["pass"] * 5)]
        triage = prioritize_triage(results, include_compatible=True)
        packages = [e.package for e in triage]
        self.assertIn("ok_pkg", packages)


# ---------------------------------------------------------------------------
# Duration anomaly tests
# ---------------------------------------------------------------------------


class TestDetectDurationAnomalies(unittest.TestCase):
    def test_no_anomaly_consistent_durations(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[make_iteration(i, "pass", duration_s=10.0) for i in range(5)],
        )
        r.compute_aggregates()
        anomalies = detect_duration_anomalies([r])
        self.assertEqual(anomalies, [])

    def test_anomaly_high_variance(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass", duration_s=50.0),
                make_iteration(1, "pass", duration_s=10.0),
                make_iteration(2, "pass", duration_s=10.0),
                make_iteration(3, "pass", duration_s=10.0),
                make_iteration(4, "pass", duration_s=10.0),
            ],
        )
        r.compute_aggregates()
        anomalies = detect_duration_anomalies([r])
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].anomaly_type, "high_variance")

    def test_anomaly_progressive_slowdown(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass", duration_s=10.0),
                make_iteration(1, "pass", duration_s=12.0),
                make_iteration(2, "pass", duration_s=15.0),
                make_iteration(3, "pass", duration_s=20.0),
                make_iteration(4, "pass", duration_s=28.0),
            ],
        )
        r.compute_aggregates()
        anomalies = detect_duration_anomalies([r])
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].anomaly_type, "progressive_slowdown")

    def test_anomaly_too_few_iterations(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass", duration_s=10.0),
                make_iteration(1, "pass", duration_s=50.0),
            ],
        )
        r.compute_aggregates()
        anomalies = detect_duration_anomalies([r])
        self.assertEqual(anomalies, [])

    def test_anomaly_custom_threshold(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass", duration_s=10.0),
                make_iteration(1, "pass", duration_s=10.0),
                make_iteration(2, "pass", duration_s=15.0),
                make_iteration(3, "pass", duration_s=10.0),
            ],
        )
        r.compute_aggregates()
        # With default threshold 0.3, might not trigger. With 0.1, should.
        anomalies_strict = detect_duration_anomalies([r], cv_threshold=0.1)
        anomalies_lenient = detect_duration_anomalies([r], cv_threshold=0.5)
        self.assertGreaterEqual(len(anomalies_strict), len(anomalies_lenient))


# ---------------------------------------------------------------------------
# Full analysis report tests
# ---------------------------------------------------------------------------


class TestAnalyzeFtRun(unittest.TestCase):
    def test_analyze_ft_run_basic(self) -> None:
        results = [
            make_package_result("compat", ["pass"] * 5),
            make_package_result("inter", ["pass", "fail", "pass", "fail", "pass"]),
            make_package_result("crash", ["pass", "pass", "crash"]),
        ]
        report = analyze_ft_run(results)
        self.assertEqual(report.total_packages, 3)
        self.assertGreater(len(report.triage), 0)
        # intermittent and crash (with passes) get flaky profiles.
        self.assertGreaterEqual(len(report.flaky_profiles), 1)

    def test_analyze_ft_run_with_gil_comparison(self) -> None:
        r = FTPackageResult(
            package="ft_only",
            iterations=[
                make_iteration(0, "pass"),
                make_iteration(1, "fail"),
                make_iteration(2, "pass"),
            ],
            gil_enabled_iterations=[
                make_iteration(0, "pass"),
                make_iteration(1, "pass"),
                make_iteration(2, "pass"),
            ],
        )
        r.compute_aggregates()
        r.categorize()
        report = analyze_ft_run([r])
        self.assertEqual(report.ft_specific_failures, 1)

    def test_analyze_crash_signatures_aggregated(self) -> None:
        r1 = FTPackageResult(
            package="c1",
            iterations=[make_iteration(0, "crash", crash_signature="sig_x")],
        )
        r1.compute_aggregates()
        r1.categorize()
        r2 = FTPackageResult(
            package="c2",
            iterations=[
                make_iteration(0, "crash", crash_signature="sig_x"),
                make_iteration(1, "crash", crash_signature="sig_y"),
            ],
        )
        r2.compute_aggregates()
        r2.categorize()
        report = analyze_ft_run([r1, r2])
        sigs = dict(report.most_common_crash_sigs)
        self.assertEqual(sigs.get("sig_x"), 2)
        self.assertEqual(sigs.get("sig_y"), 1)

    def test_analyze_tsan_types_aggregated(self) -> None:
        r1 = FTPackageResult(
            package="t1",
            iterations=[make_iteration(0, "pass", tsan_warnings=["data race"])],
        )
        r1.compute_aggregates()
        r1.categorize()
        r2 = FTPackageResult(
            package="t2",
            iterations=[
                make_iteration(0, "pass", tsan_warnings=["data race", "thread leak"]),
            ],
        )
        r2.compute_aggregates()
        r2.categorize()
        report = analyze_ft_run([r1, r2])
        types = dict(report.most_common_tsan_types)
        self.assertEqual(types.get("data race"), 2)
        self.assertEqual(types.get("thread leak"), 1)


if __name__ == "__main__":
    unittest.main()
