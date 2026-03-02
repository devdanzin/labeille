"""Tests for labeille.bench.anomaly — package-level anomaly detection."""

from __future__ import annotations

import unittest

from bench_test_helpers import make_condition_result, make_package_result
from labeille.bench.anomaly import (
    AnomalyReport,
    PackageAnomaly,
    detect_anomalies,
    detect_condition_anomalies,
    has_monotonic_trend,
    is_bimodal,
)
from labeille.bench.display import format_anomaly_report


# ---------------------------------------------------------------------------
# TestIsBimodal
# ---------------------------------------------------------------------------


class TestIsBimodal(unittest.TestCase):
    """Tests for is_bimodal() gap-analysis heuristic."""

    def test_bimodal_clear_split(self) -> None:
        values = [1.0, 1.1, 1.0, 1.1, 5.0, 5.1, 5.0, 5.1]
        self.assertTrue(is_bimodal(values))

    def test_bimodal_uniform(self) -> None:
        values = [1.0, 1.1, 1.2, 1.3, 1.4]
        self.assertFalse(is_bimodal(values))

    def test_bimodal_too_few(self) -> None:
        values = [1.0, 5.0]
        self.assertFalse(is_bimodal(values))

    def test_bimodal_close_values(self) -> None:
        values = [10.0, 10.01, 10.02, 10.03, 10.04]
        self.assertFalse(is_bimodal(values))

    def test_bimodal_custom_gap_factor(self) -> None:
        # With a very high gap_factor, the split is not detected.
        values = [1.0, 1.1, 1.0, 1.1, 5.0, 5.1, 5.0, 5.1]
        self.assertTrue(is_bimodal(values, gap_factor=1.5))
        self.assertFalse(is_bimodal(values, gap_factor=100.0))

    def test_bimodal_single_outlier_not_bimodal(self) -> None:
        # Only 1 value in the second group — not bimodal.
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]
        self.assertFalse(is_bimodal(values))


# ---------------------------------------------------------------------------
# TestHasMonotonicTrend
# ---------------------------------------------------------------------------


class TestHasMonotonicTrend(unittest.TestCase):
    """Tests for has_monotonic_trend() Spearman rank correlation."""

    def test_increasing_trend(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        has_trend, rho = has_monotonic_trend(values)
        self.assertTrue(has_trend)
        self.assertAlmostEqual(rho, 1.0, places=5)

    def test_decreasing_trend(self) -> None:
        values = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        has_trend, rho = has_monotonic_trend(values)
        self.assertTrue(has_trend)
        self.assertAlmostEqual(rho, -1.0, places=5)

    def test_no_trend(self) -> None:
        values = [1.0, 3.0, 2.0, 4.0, 1.5, 3.5]
        has_trend, _rho = has_monotonic_trend(values)
        self.assertFalse(has_trend)

    def test_too_few_values(self) -> None:
        values = [1.0, 2.0, 3.0]
        has_trend, rho = has_monotonic_trend(values)
        self.assertFalse(has_trend)
        self.assertEqual(rho, 0.0)

    def test_constant_values(self) -> None:
        values = [5.0, 5.0, 5.0, 5.0, 5.0]
        has_trend, rho = has_monotonic_trend(values)
        self.assertFalse(has_trend)
        self.assertEqual(rho, 0.0)

    def test_custom_threshold(self) -> None:
        # Weak trend that passes a low threshold but not a high one.
        values = [1.0, 1.5, 1.2, 1.8, 1.6, 2.0]
        has_trend_low, _ = has_monotonic_trend(values, correlation_threshold=0.5)
        has_trend_high, _ = has_monotonic_trend(values, correlation_threshold=0.99)
        self.assertTrue(has_trend_low)
        self.assertFalse(has_trend_high)


# ---------------------------------------------------------------------------
# TestDetectConditionAnomalies
# ---------------------------------------------------------------------------


class TestDetectConditionAnomalies(unittest.TestCase):
    """Tests for detect_condition_anomalies()."""

    def test_no_anomalies_clean_data(self) -> None:
        cond = make_condition_result("baseline", [5.0, 5.1, 4.9, 5.0, 5.1])
        anomalies = detect_condition_anomalies("pkg", "baseline", cond)
        self.assertEqual(anomalies, [])

    def test_high_cv_warning(self) -> None:
        # CV > 10% but < 20% (CV ≈ 14.4%).
        cond = make_condition_result("baseline", [6.0, 8.5, 7.0, 8.0, 6.5])
        anomalies = detect_condition_anomalies("pkg", "baseline", cond)
        cv_anomalies = [a for a in anomalies if a.anomaly_type == "high_cv"]
        self.assertEqual(len(cv_anomalies), 1)
        self.assertEqual(cv_anomalies[0].severity, "warning")

    def test_high_cv_error(self) -> None:
        # CV > 20%.
        cond = make_condition_result("baseline", [1.0, 5.0, 2.0, 8.0, 3.0])
        anomalies = detect_condition_anomalies("pkg", "baseline", cond)
        cv_anomalies = [a for a in anomalies if a.anomaly_type == "high_cv"]
        self.assertEqual(len(cv_anomalies), 1)
        self.assertEqual(cv_anomalies[0].severity, "error")

    def test_status_mixed(self) -> None:
        cond = make_condition_result("baseline", [5.0, 5.0, 5.0, 5.0, 5.0])
        # Manually set mixed statuses on measured iterations.
        cond.measured_iterations[0].status = "fail"
        cond.measured_iterations[1].status = "fail"
        anomalies = detect_condition_anomalies("pkg", "baseline", cond)
        mixed = [a for a in anomalies if a.anomaly_type == "status_mixed"]
        self.assertEqual(len(mixed), 1)
        self.assertEqual(mixed[0].severity, "error")

    def test_bimodal_detected(self) -> None:
        cond = make_condition_result("baseline", [1.0, 1.1, 1.0, 1.1, 5.0, 5.1, 5.0, 5.1])
        anomalies = detect_condition_anomalies("pkg", "baseline", cond)
        bimodal = [a for a in anomalies if a.anomaly_type == "bimodal"]
        self.assertEqual(len(bimodal), 1)
        self.assertEqual(bimodal[0].severity, "warning")

    def test_outlier_heavy(self) -> None:
        # Create data where many values are outliers: several extreme values.
        # IQR-based outliers: values well outside [Q1-1.5*IQR, Q3+1.5*IQR].
        cond = make_condition_result("baseline", [5.0, 5.0, 5.0, 5.0, 50.0, 50.0, 5.0, 5.0, 5.0])
        # Force outlier flags. compute_stats should have flagged the extreme values.
        anomalies = detect_condition_anomalies(
            "pkg", "baseline", cond, outlier_fraction_threshold=0.10
        )
        outlier = [a for a in anomalies if a.anomaly_type == "outlier_heavy"]
        self.assertEqual(len(outlier), 1)
        self.assertEqual(outlier[0].severity, "info")

    def test_trend_increasing(self) -> None:
        cond = make_condition_result("baseline", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        anomalies = detect_condition_anomalies("pkg", "baseline", cond)
        trends = [a for a in anomalies if a.anomaly_type == "trend"]
        self.assertEqual(len(trends), 1)
        self.assertEqual(trends[0].severity, "warning")
        self.assertIn("increasing", trends[0].description)

    def test_trend_decreasing(self) -> None:
        cond = make_condition_result("baseline", [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        anomalies = detect_condition_anomalies("pkg", "baseline", cond)
        trends = [a for a in anomalies if a.anomaly_type == "trend"]
        self.assertEqual(len(trends), 1)
        self.assertEqual(trends[0].severity, "info")
        self.assertIn("decreasing", trends[0].description)

    def test_too_few_iterations(self) -> None:
        cond = make_condition_result("baseline", [5.0, 5.1])
        anomalies = detect_condition_anomalies("pkg", "baseline", cond)
        self.assertEqual(anomalies, [])

    def test_skips_warmup_iterations(self) -> None:
        # Warmup iterations should not affect anomaly detection.
        # First iteration is warmup (extreme value), measured ones are stable.
        cond = make_condition_result(
            "baseline",
            [100.0, 5.0, 5.1, 4.9, 5.0, 5.1],
            warmup_count=1,
        )
        anomalies = detect_condition_anomalies("pkg", "baseline", cond)
        self.assertEqual(anomalies, [])


# ---------------------------------------------------------------------------
# TestDetectAnomalies
# ---------------------------------------------------------------------------


class TestDetectAnomalies(unittest.TestCase):
    """Tests for detect_anomalies() top-level function."""

    def test_empty_results(self) -> None:
        report = detect_anomalies([])
        self.assertEqual(len(report.anomalies), 0)

    def test_skipped_packages_ignored(self) -> None:
        from labeille.bench.results import BenchPackageResult

        pkg = BenchPackageResult(package="skipped_pkg", skipped=True, skip_reason="test")
        report = detect_anomalies([pkg])
        self.assertEqual(len(report.anomalies), 0)

    def test_multiple_packages_multiple_conditions(self) -> None:
        # Package with high CV in one condition.
        pkg1 = make_package_result(
            "unstable",
            {"baseline": [1.0, 5.0, 2.0, 8.0, 3.0]},
        )
        # Package with trend in another condition.
        pkg2 = make_package_result(
            "trending",
            {"jit": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]},
        )
        report = detect_anomalies([pkg1, pkg2])
        self.assertGreater(len(report.anomalies), 0)
        packages = {a.package for a in report.anomalies}
        self.assertIn("unstable", packages)
        self.assertIn("trending", packages)

    def test_anomaly_report_properties(self) -> None:
        pkg = make_package_result(
            "unstable",
            {"baseline": [1.0, 5.0, 2.0, 8.0, 3.0]},
        )
        report = detect_anomalies([pkg])
        self.assertGreater(len(report.anomalies), 0)

        # Test by_severity.
        by_severity = report.by_severity
        self.assertIn("error", by_severity)
        self.assertIn("warning", by_severity)
        self.assertIn("info", by_severity)

        # Test by_package.
        by_package = report.by_package
        self.assertIn("unstable", by_package)

        # Test by_type.
        by_type = report.by_type
        self.assertTrue(len(by_type) > 0)

        # Test affected_packages.
        self.assertIn("unstable", report.affected_packages)


# ---------------------------------------------------------------------------
# TestPackageAnomalySerialize
# ---------------------------------------------------------------------------


class TestPackageAnomalySerialize(unittest.TestCase):
    """Tests for PackageAnomaly and AnomalyReport serialization."""

    def test_to_dict_roundtrip(self) -> None:
        anomaly = PackageAnomaly(
            package="pkg",
            condition="baseline",
            anomaly_type="high_cv",
            severity="warning",
            metric_value=0.1234,
            threshold=0.10,
            description="CV is high.",
            recommendation="Fix it.",
        )
        d = anomaly.to_dict()
        restored = PackageAnomaly.from_dict(d)
        self.assertEqual(restored.package, "pkg")
        self.assertEqual(restored.anomaly_type, "high_cv")
        self.assertEqual(restored.severity, "warning")
        self.assertAlmostEqual(restored.metric_value, 0.1234, places=4)

    def test_anomaly_report_to_dict(self) -> None:
        anomaly = PackageAnomaly(
            package="pkg",
            condition="baseline",
            anomaly_type="bimodal",
            severity="warning",
            metric_value=0.0,
            threshold=1.5,
            description="Bimodal.",
            recommendation="Check it.",
        )
        report = AnomalyReport(anomalies=[anomaly])
        d = report.to_dict()
        self.assertEqual(d["summary"]["total"], 1)
        self.assertEqual(d["summary"]["warnings"], 1)
        self.assertEqual(d["summary"]["errors"], 0)
        self.assertEqual(d["summary"]["affected_packages"], 1)
        self.assertEqual(len(d["anomalies"]), 1)


# ---------------------------------------------------------------------------
# TestFormatAnomalyReport
# ---------------------------------------------------------------------------


class TestFormatAnomalyReport(unittest.TestCase):
    """Tests for format_anomaly_report() display function."""

    def test_format_empty_report(self) -> None:
        report = AnomalyReport(anomalies=[])
        self.assertEqual(format_anomaly_report(report), "")

    def test_format_errors_first(self) -> None:
        anomalies = [
            PackageAnomaly(
                package="a",
                condition="c",
                anomaly_type="bimodal",
                severity="warning",
                metric_value=0.0,
                threshold=1.5,
                description="Warning msg.",
                recommendation="Fix warning.",
            ),
            PackageAnomaly(
                package="b",
                condition="c",
                anomaly_type="status_mixed",
                severity="error",
                metric_value=2.0,
                threshold=1.0,
                description="Error msg.",
                recommendation="Fix error.",
            ),
            PackageAnomaly(
                package="c",
                condition="c",
                anomaly_type="outlier_heavy",
                severity="info",
                metric_value=0.3,
                threshold=0.2,
                description="Info msg.",
                recommendation="Check info.",
            ),
        ]
        report = AnomalyReport(anomalies=anomalies)
        text = format_anomaly_report(report)

        # Errors should appear before warnings, warnings before info.
        error_pos = text.index("[ERROR]")
        warning_pos = text.index("[WARNING]")
        info_pos = text.index("[INFO]")
        self.assertLess(error_pos, warning_pos)
        self.assertLess(warning_pos, info_pos)

    def test_format_includes_recommendations(self) -> None:
        anomaly = PackageAnomaly(
            package="pkg",
            condition="cond",
            anomaly_type="high_cv",
            severity="error",
            metric_value=0.25,
            threshold=0.20,
            description="CV is high.",
            recommendation="Unique recommendation text here.",
        )
        report = AnomalyReport(anomalies=[anomaly])
        text = format_anomaly_report(report)
        self.assertIn("Unique recommendation text here.", text)


if __name__ == "__main__":
    unittest.main()
