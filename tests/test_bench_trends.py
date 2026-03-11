"""Tests for labeille.bench.trends — trend analysis and regression alerts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from labeille.bench.results import (
    BenchMeta,
    ConditionDef,
)
from labeille.bench.system import SystemProfile
from labeille.bench.tracking import (
    TrackingRunEntry,
    TrackingSeries,
    save_series,
)
from labeille.bench.trends import (
    PackageTrend,
    RegressionAlert,
    SeriesTrend,
    _generate_alerts,
    _linear_regression_slope,
    analyze_series_trends,
    compute_package_trend,
)
from tests.bench_test_helpers import make_package_result


def _make_bench_run_dir(
    parent: Path,
    bench_id: str,
    *,
    packages: dict[str, dict[str, list[float]]] | None = None,
    start_time: str = "2026-01-01T00:00:00",
    conditions: list[str] | None = None,
) -> Path:
    """Create a mock benchmark run directory with meta and results."""
    run_dir = parent / bench_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cond_names = conditions or ["baseline"]
    meta = BenchMeta(
        bench_id=bench_id,
        name=bench_id,
        start_time=start_time,
        system=SystemProfile(
            cpu_model="Test CPU",
            cpu_cores_physical=4,
            cpu_cores_logical=8,
            ram_total_gb=16.0,
            ram_available_gb=12.0,
            os_distro="Test Linux",
            hostname="testhost",
        ),
        conditions={n: ConditionDef(name=n) for n in cond_names},
        config={"iterations": 5, "warmup": 1, "timeout": 600},
        packages_total=len(packages) if packages else 0,
        packages_completed=len(packages) if packages else 0,
    )

    meta_path = run_dir / "bench_meta.json"
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n")

    results_path = run_dir / "bench_results.jsonl"
    with open(results_path, "w") as f:
        if packages:
            for pkg_name, pkg_conditions in packages.items():
                pkg = make_package_result(pkg_name, pkg_conditions, warmup_count=1)
                f.write(json.dumps(pkg.to_dict()) + "\n")

    return run_dir


# ---------------------------------------------------------------------------
# TestLinearRegressionSlope
# ---------------------------------------------------------------------------


class TestLinearRegressionSlope(unittest.TestCase):
    """Tests for _linear_regression_slope()."""

    def test_increasing(self) -> None:
        slope = _linear_regression_slope([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(slope, 1.0)

    def test_decreasing(self) -> None:
        slope = _linear_regression_slope([4.0, 3.0, 2.0, 1.0])
        self.assertAlmostEqual(slope, -1.0)

    def test_flat(self) -> None:
        slope = _linear_regression_slope([5.0, 5.0, 5.0, 5.0])
        self.assertAlmostEqual(slope, 0.0)

    def test_single_point(self) -> None:
        slope = _linear_regression_slope([1.0])
        self.assertEqual(slope, 0.0)

    def test_two_points(self) -> None:
        slope = _linear_regression_slope([1.0, 3.0])
        self.assertAlmostEqual(slope, 2.0)


# ---------------------------------------------------------------------------
# TestComputePackageTrend
# ---------------------------------------------------------------------------


class TestComputePackageTrend(unittest.TestCase):
    """Tests for compute_package_trend()."""

    def test_stable_package(self) -> None:
        medians = [10.0, 10.1, 9.9, 10.0, 10.1]
        cvs = [0.02, 0.02, 0.02, 0.02, 0.02]
        timestamps = [f"2026-01-0{i}T00:00:00" for i in range(1, 6)]
        pt = compute_package_trend("pkg-a", "baseline", medians, cvs, timestamps)
        self.assertEqual(pt.trend_direction, "stable")
        self.assertFalse(pt.sustained_regression)
        self.assertFalse(pt.sustained_improvement)

    def test_regressing_package(self) -> None:
        # Each run 5% slower: 10.0, 10.5, 11.025, 11.576, 12.155
        base = 10.0
        medians = [base * (1.05**i) for i in range(5)]
        cvs = [0.02] * 5
        timestamps = [f"2026-01-0{i}T00:00:00" for i in range(1, 6)]
        pt = compute_package_trend("pkg-b", "baseline", medians, cvs, timestamps)
        self.assertEqual(pt.trend_direction, "regressing")

    def test_improving_package(self) -> None:
        # Each run 5% faster: 10.0, 9.5, 9.025, 8.574, 8.145
        base = 10.0
        medians = [base * (0.95**i) for i in range(5)]
        cvs = [0.02] * 5
        timestamps = [f"2026-01-0{i}T00:00:00" for i in range(1, 6)]
        pt = compute_package_trend("pkg-c", "baseline", medians, cvs, timestamps)
        self.assertEqual(pt.trend_direction, "improving")

    def test_volatile_package(self) -> None:
        medians = [10.0, 10.2, 9.8, 10.3, 9.7]
        # CVs increasing and above 10%.
        cvs = [0.05, 0.08, 0.11, 0.15, 0.20]
        timestamps = [f"2026-01-0{i}T00:00:00" for i in range(1, 6)]
        pt = compute_package_trend("pkg-d", "baseline", medians, cvs, timestamps)
        self.assertEqual(pt.trend_direction, "volatile")
        self.assertTrue(pt.volatility_increasing)

    def test_sustained_regression(self) -> None:
        # 3 consecutive increases > 2%.
        medians = [10.0, 10.3, 10.6, 10.95]
        cvs = [0.02] * 4
        timestamps = [f"2026-01-0{i}T00:00:00" for i in range(1, 5)]
        pt = compute_package_trend("pkg-e", "baseline", medians, cvs, timestamps)
        self.assertTrue(pt.sustained_regression)

    def test_sustained_improvement(self) -> None:
        # 3 consecutive decreases > 2%.
        medians = [10.0, 9.7, 9.4, 9.1]
        cvs = [0.02] * 4
        timestamps = [f"2026-01-0{i}T00:00:00" for i in range(1, 5)]
        pt = compute_package_trend("pkg-f", "baseline", medians, cvs, timestamps)
        self.assertTrue(pt.sustained_improvement)

    def test_not_sustained_two_increases(self) -> None:
        # Only 2 consecutive increases, not 3.
        medians = [10.0, 10.3, 10.6]
        cvs = [0.02] * 3
        timestamps = [f"2026-01-0{i}T00:00:00" for i in range(1, 4)]
        pt = compute_package_trend("pkg-g", "baseline", medians, cvs, timestamps)
        self.assertFalse(pt.sustained_regression)

    def test_recent_change_pct(self) -> None:
        medians = [10.0, 12.0]
        cvs = [0.02, 0.02]
        timestamps = ["2026-01-01T00:00:00", "2026-01-02T00:00:00"]
        pt = compute_package_trend("pkg-h", "baseline", medians, cvs, timestamps)
        self.assertAlmostEqual(pt.recent_change_pct, 20.0)

    def test_cumulative_change_pct(self) -> None:
        medians = [10.0, 11.0, 12.0]
        cvs = [0.02] * 3
        timestamps = [f"2026-01-0{i}T00:00:00" for i in range(1, 4)]
        pt = compute_package_trend("pkg-i", "baseline", medians, cvs, timestamps)
        self.assertAlmostEqual(pt.cumulative_change_pct, 20.0)

    def test_single_run(self) -> None:
        medians = [10.0]
        cvs = [0.02]
        timestamps = ["2026-01-01T00:00:00"]
        pt = compute_package_trend("pkg-j", "baseline", medians, cvs, timestamps)
        self.assertEqual(pt.trend_direction, "stable")
        self.assertEqual(pt.trend_slope, 0.0)
        self.assertEqual(pt.n_runs, 1)

    def test_two_runs(self) -> None:
        medians = [10.0, 12.0]
        cvs = [0.02, 0.02]
        timestamps = ["2026-01-01T00:00:00", "2026-01-02T00:00:00"]
        pt = compute_package_trend("pkg-k", "baseline", medians, cvs, timestamps)
        self.assertEqual(pt.n_runs, 2)
        self.assertAlmostEqual(pt.cumulative_change_pct, 20.0)
        self.assertFalse(pt.sustained_regression)

    def test_custom_thresholds(self) -> None:
        # With default threshold (5%) this is regressing.
        # With very high thresholds it should be stable.
        base = 10.0
        medians = [base * (1.05**i) for i in range(5)]
        cvs = [0.02] * 5
        timestamps = [f"2026-01-0{i}T00:00:00" for i in range(1, 6)]

        pt_default = compute_package_trend("pkg", "baseline", medians, cvs, timestamps)
        self.assertEqual(pt_default.trend_direction, "regressing")

        pt_high = compute_package_trend(
            "pkg",
            "baseline",
            medians,
            cvs,
            timestamps,
            trend_threshold=0.50,
            regression_threshold=0.50,
        )
        self.assertEqual(pt_high.trend_direction, "stable")


# ---------------------------------------------------------------------------
# TestGenerateAlerts
# ---------------------------------------------------------------------------


class TestGenerateAlerts(unittest.TestCase):
    """Tests for _generate_alerts()."""

    def _make_trend(
        self,
        package: str = "pkg-a",
        *,
        direction: str = "stable",
        sustained_regression: bool = False,
        sustained_improvement: bool = False,
        volatility_increasing: bool = False,
        medians: list[float] | None = None,
        cvs: list[float] | None = None,
        n_runs: int = 5,
    ) -> PackageTrend:
        medians = medians or [10.0] * n_runs
        cvs = cvs or [0.02] * n_runs
        return PackageTrend(
            package=package,
            condition="baseline",
            timestamps=[f"2026-01-0{i}T00:00:00" for i in range(1, n_runs + 1)],
            medians=medians,
            cvs=cvs,
            n_runs=n_runs,
            trend_direction=direction,
            sustained_regression=sustained_regression,
            sustained_improvement=sustained_improvement,
            volatility_increasing=volatility_increasing,
        )

    def test_new_regression_alert(self) -> None:
        pt = self._make_trend(medians=[10.0, 10.0, 10.0, 10.0, 11.0])
        alerts = _generate_alerts(
            [pt],
            {"pkg-a": 10.0},
            {"pkg-a": 10.0},
            {"pkg-a": 11.0},
        )
        types = [a.alert_type for a in alerts]
        self.assertIn("new_regression", types)

    def test_sustained_regression_alert(self) -> None:
        pt = self._make_trend(
            direction="regressing",
            sustained_regression=True,
            medians=[10.0, 10.5, 11.0, 11.5, 12.0],
        )
        alerts = _generate_alerts(
            [pt],
            {"pkg-a": 10.0},
            {"pkg-a": 11.5},
            {"pkg-a": 12.0},
        )
        types = [a.alert_type for a in alerts]
        self.assertIn("sustained_regression", types)
        sustained = [a for a in alerts if a.alert_type == "sustained_regression"]
        self.assertEqual(sustained[0].severity, "error")

    def test_recovery_alert(self) -> None:
        # medians[-3] < medians[-2] (was increasing), then latest back to baseline.
        pt = self._make_trend(
            direction="stable",
            medians=[10.0, 10.5, 11.0, 11.5, 10.0],
            n_runs=5,
        )
        alerts = _generate_alerts(
            [pt],
            {"pkg-a": 10.0},
            {"pkg-a": 11.5},
            {"pkg-a": 10.0},
        )
        types = [a.alert_type for a in alerts]
        self.assertIn("recovery", types)

    def test_new_improvement_alert(self) -> None:
        pt = self._make_trend(medians=[10.0, 10.0, 10.0, 10.0, 9.0])
        alerts = _generate_alerts(
            [pt],
            {"pkg-a": 10.0},
            {"pkg-a": 10.0},
            {"pkg-a": 9.0},
        )
        types = [a.alert_type for a in alerts]
        self.assertIn("new_improvement", types)

    def test_new_instability_alert(self) -> None:
        pt = self._make_trend(
            direction="volatile",
            volatility_increasing=True,
            cvs=[0.02, 0.03, 0.05, 0.12, 0.15],
        )
        alerts = _generate_alerts(
            [pt],
            {"pkg-a": 10.0},
            {"pkg-a": 10.0},
            {"pkg-a": 10.0},
        )
        types = [a.alert_type for a in alerts]
        self.assertIn("new_instability", types)

    def test_no_alerts_stable(self) -> None:
        pt = self._make_trend()
        alerts = _generate_alerts(
            [pt],
            {"pkg-a": 10.0},
            {"pkg-a": 10.0},
            {"pkg-a": 10.0},
        )
        self.assertEqual(len(alerts), 0)

    def test_alerts_sorted_by_severity(self) -> None:
        pt_error = self._make_trend(
            package="pkg-err",
            direction="regressing",
            sustained_regression=True,
            medians=[10.0, 10.5, 11.0, 11.5, 12.0],
        )
        pt_warning = self._make_trend(
            package="pkg-warn",
            medians=[10.0, 10.0, 10.0, 10.0, 11.0],
        )
        pt_info = self._make_trend(
            package="pkg-info",
            medians=[10.0, 10.0, 10.0, 10.0, 9.0],
        )
        alerts = _generate_alerts(
            [pt_error, pt_warning, pt_info],
            {"pkg-err": 10.0, "pkg-warn": 10.0, "pkg-info": 10.0},
            {"pkg-err": 11.5, "pkg-warn": 10.0, "pkg-info": 10.0},
            {"pkg-err": 12.0, "pkg-warn": 11.0, "pkg-info": 9.0},
        )
        severities = [a.severity for a in alerts]
        # Errors should come before warnings, warnings before info.
        error_indices = [i for i, s in enumerate(severities) if s == "error"]
        warning_indices = [i for i, s in enumerate(severities) if s == "warning"]
        info_indices = [i for i, s in enumerate(severities) if s == "info"]
        if error_indices and warning_indices:
            self.assertLess(max(error_indices), min(warning_indices))
        if warning_indices and info_indices:
            self.assertLess(max(warning_indices), min(info_indices))


# ---------------------------------------------------------------------------
# TestAnalyzeSeriesTrends
# ---------------------------------------------------------------------------


class TestAnalyzeSeriesTrends(unittest.TestCase):
    """Tests for analyze_series_trends() using mock data."""

    def _setup_series(
        self,
        tmpdir: Path,
        n_runs: int = 3,
        packages_per_run: dict[str, dict[str, list[float]]] | None = None,
        pinned_baseline_id: str | None = None,
    ) -> tuple[TrackingSeries, Path]:
        """Create a series with mock runs."""
        tracking_dir = tmpdir / "tracking"
        series_dir = tracking_dir / "test-series"
        series_dir.mkdir(parents=True)

        runs: list[TrackingRunEntry] = []
        for i in range(n_runs):
            bench_id = f"bench_{i:03d}"
            ts = f"2026-01-{i + 1:02d}T00:00:00"

            # Create run dir.
            pkgs = packages_per_run or {
                "pkg-a": {"baseline": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]},
                "pkg-b": {"baseline": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]},
            }
            run_dir = _make_bench_run_dir(
                tmpdir, bench_id, packages=pkgs, start_time=ts, conditions=["baseline"]
            )

            # Create symlink in series dir.
            link = series_dir / bench_id
            link.symlink_to(run_dir)

            runs.append(
                TrackingRunEntry(
                    bench_id=bench_id,
                    timestamp=ts,
                    run_dir=bench_id,
                    packages_completed=len(pkgs),
                    config_fingerprint="abcdef1234567890",
                )
            )

        series = TrackingSeries(
            series_id="test-series",
            description="Test series",
            created="2026-01-01T00:00:00",
            config_fingerprint="abcdef1234567890",
            pinned_baseline_id=pinned_baseline_id,
            runs=runs,
        )
        save_series(series, series_dir)
        return series, series_dir

    def test_basic_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            series, series_dir = self._setup_series(Path(tmpdir))
            trend = analyze_series_trends(series, series_dir)
            self.assertEqual(trend.series_id, "test-series")
            self.assertEqual(trend.n_runs, 3)
            self.assertEqual(trend.condition, "baseline")
            self.assertEqual(len(trend.package_trends), 2)
            # All stable since values are constant.
            for pt in trend.package_trends:
                self.assertEqual(pt.trend_direction, "stable")

    def test_analysis_with_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = Path(tmpdir) / "tracking"
            series_dir = tracking_dir / "test-series"
            series_dir.mkdir(parents=True)

            runs: list[TrackingRunEntry] = []
            for i in range(5):
                bench_id = f"bench_{i:03d}"
                ts = f"2026-01-{i + 1:02d}T00:00:00"
                # pkg-a regresses 10% each run.
                factor_a = 1.10**i
                pkgs = {
                    "pkg-a": {
                        "baseline": [10.0 * factor_a] * 6,
                    },
                    "pkg-b": {
                        "baseline": [5.0] * 6,
                    },
                }
                run_dir = _make_bench_run_dir(Path(tmpdir), bench_id, packages=pkgs, start_time=ts)
                link = series_dir / bench_id
                link.symlink_to(run_dir)
                runs.append(
                    TrackingRunEntry(
                        bench_id=bench_id,
                        timestamp=ts,
                        run_dir=bench_id,
                        packages_completed=2,
                        config_fingerprint="abcdef1234567890",
                    )
                )

            series = TrackingSeries(
                series_id="test-series",
                created="2026-01-01T00:00:00",
                config_fingerprint="abcdef1234567890",
                runs=runs,
            )
            save_series(series, series_dir)

            trend = analyze_series_trends(series, series_dir)
            self.assertIn("pkg-a", trend.regressing_packages)
            self.assertIn("pkg-b", trend.stable_packages)

    def test_analysis_default_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            series, series_dir = self._setup_series(Path(tmpdir))
            trend = analyze_series_trends(series, series_dir)
            self.assertEqual(trend.condition, "baseline")

    def test_analysis_specific_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            series, series_dir = self._setup_series(Path(tmpdir))
            trend = analyze_series_trends(series, series_dir, condition="baseline")
            self.assertEqual(trend.condition, "baseline")

    def test_analysis_missing_packages(self) -> None:
        """Package not in all runs still gets analyzed with shorter history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = Path(tmpdir) / "tracking"
            series_dir = tracking_dir / "test-series"
            series_dir.mkdir(parents=True)

            runs: list[TrackingRunEntry] = []
            for i in range(3):
                bench_id = f"bench_{i:03d}"
                ts = f"2026-01-{i + 1:02d}T00:00:00"
                pkgs: dict[str, dict[str, list[float]]] = {
                    "pkg-a": {"baseline": [10.0] * 6},
                }
                if i >= 1:
                    pkgs["pkg-late"] = {"baseline": [5.0] * 6}
                run_dir = _make_bench_run_dir(Path(tmpdir), bench_id, packages=pkgs, start_time=ts)
                link = series_dir / bench_id
                link.symlink_to(run_dir)
                runs.append(
                    TrackingRunEntry(
                        bench_id=bench_id,
                        timestamp=ts,
                        run_dir=bench_id,
                        packages_completed=len(pkgs),
                        config_fingerprint="abcdef1234567890",
                    )
                )

            series = TrackingSeries(
                series_id="test-series",
                created="2026-01-01T00:00:00",
                config_fingerprint="abcdef1234567890",
                runs=runs,
            )
            save_series(series, series_dir)

            trend = analyze_series_trends(series, series_dir)
            pkg_names = [pt.package for pt in trend.package_trends]
            self.assertIn("pkg-a", pkg_names)
            self.assertIn("pkg-late", pkg_names)
            late_trend = next(pt for pt in trend.package_trends if pt.package == "pkg-late")
            self.assertEqual(late_trend.n_runs, 2)

    def test_analysis_aggregate_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = Path(tmpdir) / "tracking"
            series_dir = tracking_dir / "test-series"
            series_dir.mkdir(parents=True)

            runs: list[TrackingRunEntry] = []
            for i in range(3):
                bench_id = f"bench_{i:03d}"
                ts = f"2026-01-{i + 1:02d}T00:00:00"
                # pkg-a: 10 -> 11 -> 12 (20% cumulative)
                # pkg-b: 10 -> 10 -> 10 (0% cumulative)
                pkgs = {
                    "pkg-a": {"baseline": [10.0 + i] * 6},
                    "pkg-b": {"baseline": [10.0] * 6},
                }
                run_dir = _make_bench_run_dir(Path(tmpdir), bench_id, packages=pkgs, start_time=ts)
                link = series_dir / bench_id
                link.symlink_to(run_dir)
                runs.append(
                    TrackingRunEntry(
                        bench_id=bench_id,
                        timestamp=ts,
                        run_dir=bench_id,
                        packages_completed=2,
                        config_fingerprint="abcdef1234567890",
                    )
                )

            series = TrackingSeries(
                series_id="test-series",
                created="2026-01-01T00:00:00",
                config_fingerprint="abcdef1234567890",
                runs=runs,
            )
            save_series(series, series_dir)

            trend = analyze_series_trends(series, series_dir)
            # Median of [20%, 0%] = 10%.
            self.assertAlmostEqual(trend.aggregate_median_change_pct, 10.0)

    def test_analysis_with_pinned_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            series, series_dir = self._setup_series(Path(tmpdir), pinned_baseline_id="bench_001")
            trend = analyze_series_trends(series, series_dir)
            self.assertEqual(trend.baseline_bench_id, "bench_001")


# ---------------------------------------------------------------------------
# TestSeriesTrendSerialization
# ---------------------------------------------------------------------------


class TestSeriesTrendSerialization(unittest.TestCase):
    """Tests for serialization/deserialization of trend dataclasses."""

    def test_to_dict_structure(self) -> None:
        trend = SeriesTrend(
            series_id="test",
            n_runs=3,
            date_range=("2026-01-01", "2026-01-03"),
            condition="baseline",
            baseline_bench_id="bench_001",
        )
        d = trend.to_dict()
        self.assertIn("series_id", d)
        self.assertIn("n_runs", d)
        self.assertIn("date_range", d)
        self.assertIn("condition", d)
        self.assertIn("baseline_bench_id", d)
        self.assertIn("package_trends", d)
        self.assertIn("alerts", d)
        self.assertIn("aggregate_median_change_pct", d)

    def test_package_trend_roundtrip(self) -> None:
        pt = PackageTrend(
            package="pkg-a",
            condition="baseline",
            timestamps=["2026-01-01T00:00:00", "2026-01-02T00:00:00"],
            medians=[10.0, 11.0],
            cvs=[0.02, 0.03],
            n_runs=2,
            trend_direction="regressing",
            trend_slope=1.0,
            trend_pct_per_run=10.0,
            recent_change_pct=10.0,
            cumulative_change_pct=10.0,
        )
        d = pt.to_dict()
        restored = PackageTrend.from_dict(d)
        self.assertEqual(restored.package, pt.package)
        self.assertEqual(restored.condition, pt.condition)
        self.assertEqual(restored.trend_direction, pt.trend_direction)
        self.assertEqual(restored.n_runs, pt.n_runs)

    def test_regression_alert_roundtrip(self) -> None:
        alert = RegressionAlert(
            package="pkg-a",
            condition="baseline",
            alert_type="new_regression",
            severity="warning",
            description="New regression: +10.0% slower.",
            recent_change_pct=10.0,
            cumulative_change_pct=10.0,
            baseline_median_s=10.0,
            current_median_s=11.0,
            previous_median_s=10.0,
        )
        d = alert.to_dict()
        restored = RegressionAlert.from_dict(d)
        self.assertEqual(restored.package, alert.package)
        self.assertEqual(restored.alert_type, alert.alert_type)
        self.assertEqual(restored.severity, alert.severity)


# ---------------------------------------------------------------------------
# TestFormatSeriesTrend
# ---------------------------------------------------------------------------


class TestFormatSeriesTrend(unittest.TestCase):
    """Tests for format_series_trend()."""

    def test_format_includes_summary(self) -> None:
        from labeille.bench.display import format_series_trend

        trend = SeriesTrend(
            series_id="test",
            n_runs=3,
            date_range=("2026-01-01", "2026-01-03"),
            condition="baseline",
            baseline_bench_id="bench_001",
            regressing_packages=["pkg-a"],
            improving_packages=["pkg-b"],
            stable_packages=["pkg-c"],
        )
        output = format_series_trend(trend)
        self.assertIn("Regressing:", output)
        self.assertIn("Improving:", output)
        self.assertIn("Stable:", output)
        self.assertIn("1", output)  # counts

    def test_format_includes_alerts(self) -> None:
        from labeille.bench.display import format_series_trend

        alert = RegressionAlert(
            package="pkg-a",
            condition="baseline",
            alert_type="new_regression",
            severity="warning",
            description="New regression detected.",
            recent_change_pct=10.0,
            cumulative_change_pct=10.0,
            baseline_median_s=10.0,
            current_median_s=11.0,
            previous_median_s=10.0,
        )
        trend = SeriesTrend(
            series_id="test",
            n_runs=3,
            date_range=("2026-01-01", "2026-01-03"),
            condition="baseline",
            baseline_bench_id="bench_001",
            alerts=[alert],
        )
        output = format_series_trend(trend)
        self.assertIn("Regression Alerts", output)
        self.assertIn("pkg-a", output)

    def test_format_empty_series(self) -> None:
        from labeille.bench.display import format_series_trend

        trend = SeriesTrend(
            series_id="empty",
            n_runs=0,
            date_range=None,
            condition="baseline",
            baseline_bench_id="",
        )
        output = format_series_trend(trend)
        self.assertIn("empty", output)
        self.assertIn("0", output)


# ---------------------------------------------------------------------------
# TestExportTrendMarkdown
# ---------------------------------------------------------------------------


class TestExportTrendMarkdown(unittest.TestCase):
    """Tests for export_trend_markdown()."""

    def _make_trend(self) -> SeriesTrend:
        pt = PackageTrend(
            package="pkg-a",
            condition="baseline",
            timestamps=["2026-01-01T00:00:00", "2026-01-02T00:00:00"],
            medians=[10.0, 11.0],
            cvs=[0.02, 0.03],
            n_runs=2,
            trend_direction="regressing",
            cumulative_change_pct=10.0,
            trend_pct_per_run=10.0,
        )
        alert = RegressionAlert(
            package="pkg-a",
            condition="baseline",
            alert_type="new_regression",
            severity="warning",
            description="Regression detected.",
            recent_change_pct=10.0,
            cumulative_change_pct=10.0,
            baseline_median_s=10.0,
            current_median_s=11.0,
            previous_median_s=10.0,
        )
        return SeriesTrend(
            series_id="test-series",
            n_runs=2,
            date_range=("2026-01-01T00:00:00", "2026-01-02T00:00:00"),
            condition="baseline",
            baseline_bench_id="bench_001",
            package_trends=[pt],
            alerts=[alert],
            regressing_packages=["pkg-a"],
        )

    def test_markdown_has_title(self) -> None:
        from labeille.bench.export import export_trend_markdown

        output = export_trend_markdown(self._make_trend())
        self.assertTrue(output.startswith("# Trend Analysis: test-series"))

    def test_markdown_has_trend_table(self) -> None:
        from labeille.bench.export import export_trend_markdown

        output = export_trend_markdown(self._make_trend())
        self.assertIn("Per-Package Trends", output)
        self.assertIn("pkg-a", output)
        self.assertIn("regressing", output)

    def test_markdown_has_alerts_section(self) -> None:
        from labeille.bench.export import export_trend_markdown

        output = export_trend_markdown(self._make_trend())
        self.assertIn("## Alerts", output)
        self.assertIn("Regression detected.", output)


# ---------------------------------------------------------------------------
# TestExportTrendCsv
# ---------------------------------------------------------------------------


class TestExportTrendCsv(unittest.TestCase):
    """Tests for export_trend_csv()."""

    def _make_trend(self) -> SeriesTrend:
        pt1 = PackageTrend(
            package="pkg-a",
            condition="baseline",
            timestamps=["2026-01-01T00:00:00", "2026-01-02T00:00:00"],
            medians=[10.0, 11.0],
            cvs=[0.02, 0.03],
            n_runs=2,
            trend_direction="regressing",
            trend_slope=1.0,
            recent_change_pct=10.0,
            cumulative_change_pct=10.0,
        )
        pt2 = PackageTrend(
            package="pkg-b",
            condition="baseline",
            timestamps=["2026-01-01T00:00:00", "2026-01-02T00:00:00"],
            medians=[5.0, 5.0],
            cvs=[0.02, 0.02],
            n_runs=2,
            trend_direction="stable",
        )
        return SeriesTrend(
            series_id="test-series",
            n_runs=2,
            date_range=("2026-01-01T00:00:00", "2026-01-02T00:00:00"),
            condition="baseline",
            baseline_bench_id="bench_001",
            package_trends=[pt1, pt2],
        )

    def test_csv_header(self) -> None:
        from labeille.bench.export import export_trend_csv

        output = export_trend_csv(self._make_trend())
        header = output.strip().split("\n")[0]
        self.assertIn("package", header)
        self.assertIn("condition", header)
        self.assertIn("n_runs", header)
        self.assertIn("direction", header)
        self.assertIn("baseline_median", header)
        self.assertIn("latest_median", header)
        self.assertIn("cumulative_change_pct", header)
        self.assertIn("trend_slope", header)
        self.assertIn("recent_change_pct", header)
        self.assertIn("sustained_regression", header)

    def test_csv_row_count(self) -> None:
        from labeille.bench.export import export_trend_csv

        output = export_trend_csv(self._make_trend())
        lines = [line for line in output.strip().split("\n") if line]
        # Header + 2 data rows.
        self.assertEqual(len(lines), 3)

    def test_csv_values(self) -> None:
        from labeille.bench.export import export_trend_csv

        output = export_trend_csv(self._make_trend())
        lines = output.strip().split("\n")
        # pkg-a is first alphabetically.
        self.assertIn("pkg-a", lines[1])
        self.assertIn("regressing", lines[1])
        self.assertIn("pkg-b", lines[2])
        self.assertIn("stable", lines[2])


if __name__ == "__main__":
    unittest.main()
