"""Tests for labeille.bench.display — terminal formatting for benchmarks."""

from __future__ import annotations

import unittest

from labeille.bench.display import (
    _format_ci,
    _format_pct,
    _format_quality_summary,
    _format_time,
    _significance_marker,
    format_bench_show,
    format_comparison_summary,
)
from labeille.bench.results import (
    BenchConditionResult,
    BenchIteration,
    BenchMeta,
    BenchPackageResult,
    ConditionDef,
)
from labeille.bench.stats import DescriptiveStats
from labeille.bench.system import PythonProfile, SystemProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stats(
    median: float = 1.5,
    mean: float = 1.5,
    stdev: float = 0.05,
    cv: float = 0.03,
) -> DescriptiveStats:
    """Create a DescriptiveStats with sensible defaults."""
    return DescriptiveStats(
        n=5,
        mean=mean,
        median=median,
        stdev=stdev,
        min=median - stdev,
        max=median + stdev,
        q1=median - stdev * 0.5,
        q3=median + stdev * 0.5,
        iqr=stdev,
        cv=cv,
    )


def _make_iteration(
    index: int = 1,
    warmup: bool = False,
    wall: float = 1.5,
    status: str = "ok",
) -> BenchIteration:
    return BenchIteration(
        index=index,
        warmup=warmup,
        wall_time_s=wall,
        user_time_s=wall * 0.8,
        sys_time_s=wall * 0.05,
        peak_rss_mb=100.0,
        exit_code=0 if status == "ok" else 1,
        status=status,
        load_avg_start=0.5,
        load_avg_end=0.6,
    )


def _make_condition_result(
    name: str = "baseline",
    median: float = 1.5,
    cv: float = 0.03,
    n_iters: int = 5,
) -> BenchConditionResult:
    """Create a BenchConditionResult with stats pre-populated."""
    cr = BenchConditionResult(condition_name=name)
    cr.iterations = [_make_iteration(i + 1, wall=median) for i in range(n_iters)]
    cr.wall_time_stats = _make_stats(median=median, cv=cv)
    cr.user_time_stats = _make_stats(median=median * 0.8, cv=cv)
    cr.peak_rss_stats = _make_stats(median=100.0, cv=cv)
    return cr


def _make_package_result(
    name: str = "requests",
    conditions: dict[str, BenchConditionResult] | None = None,
) -> BenchPackageResult:
    return BenchPackageResult(
        package=name,
        conditions=conditions or {"baseline": _make_condition_result()},
    )


def _make_meta(
    condition_names: list[str] | None = None,
    packages_completed: int = 5,
) -> BenchMeta:
    cond_names = condition_names or ["baseline"]
    conditions = {n: ConditionDef(name=n) for n in cond_names}
    return BenchMeta(
        bench_id="bench_test",
        name="Test Benchmark",
        description="A test benchmark run.",
        system=SystemProfile(),
        python_profiles={n: PythonProfile() for n in cond_names},
        conditions=conditions,
        config={"iterations": 5, "warmup": 1, "alternate": False},
        start_time="2026-02-28T10:00:00+0000",
        end_time="2026-02-28T11:00:00+0000",
        packages_total=packages_completed,
        packages_completed=packages_completed,
        packages_skipped=0,
    )


# ---------------------------------------------------------------------------
# _format_time tests
# ---------------------------------------------------------------------------


class TestFormatTime(unittest.TestCase):
    def test_format_time_seconds(self) -> None:
        self.assertEqual(_format_time(12.34), "12.34s")

    def test_format_time_minutes(self) -> None:
        result = _format_time(125.5)
        self.assertEqual(result, "2m6s")

    def test_format_time_milliseconds(self) -> None:
        result = _format_time(0.045)
        self.assertEqual(result, "45.00ms")

    def test_format_time_microseconds(self) -> None:
        result = _format_time(0.000500)
        self.assertEqual(result, "500\u00b5s")

    def test_format_time_nan(self) -> None:
        self.assertEqual(_format_time(float("nan")), "N/A")


# ---------------------------------------------------------------------------
# _format_pct tests
# ---------------------------------------------------------------------------


class TestFormatPct(unittest.TestCase):
    def test_format_pct_positive(self) -> None:
        self.assertEqual(_format_pct(20.3), "+20.3%")

    def test_format_pct_negative(self) -> None:
        self.assertEqual(_format_pct(-5.1), "-5.1%")

    def test_format_pct_nan(self) -> None:
        self.assertEqual(_format_pct(float("nan")), "N/A")

    def test_format_pct_zero(self) -> None:
        self.assertEqual(_format_pct(0.0), "+0.0%")


# ---------------------------------------------------------------------------
# _format_ci tests
# ---------------------------------------------------------------------------


class TestFormatCI(unittest.TestCase):
    def test_format_ci(self) -> None:
        result = _format_ci(0.15, 0.25)
        self.assertEqual(result, "[+15.0%, +25.0%]")

    def test_format_ci_negative(self) -> None:
        result = _format_ci(-0.10, 0.05)
        self.assertEqual(result, "[-10.0%, +5.0%]")

    def test_format_ci_nan(self) -> None:
        result = _format_ci(float("nan"), 0.1)
        self.assertEqual(result, "[N/A]")


# ---------------------------------------------------------------------------
# _significance_marker tests
# ---------------------------------------------------------------------------


class TestSignificanceMarker(unittest.TestCase):
    def test_stars(self) -> None:
        self.assertEqual(_significance_marker("***"), "***")
        self.assertEqual(_significance_marker("**"), "**")
        self.assertEqual(_significance_marker("*"), "*")

    def test_not_significant(self) -> None:
        self.assertEqual(_significance_marker("ns"), "ns")


# ---------------------------------------------------------------------------
# format_bench_show tests
# ---------------------------------------------------------------------------


class TestFormatBenchShow(unittest.TestCase):
    def test_single_condition_output(self) -> None:
        """Single condition shows package names and wall times."""
        meta = _make_meta(["baseline"])
        results = [
            _make_package_result("requests"),
            _make_package_result("click"),
        ]
        output = format_bench_show(meta, results)
        self.assertIn("Test Benchmark", output)
        self.assertIn("requests", output)
        self.assertIn("click", output)
        self.assertIn("Wall (s)", output)
        self.assertIn("1.50", output)

    def test_multi_condition_output(self) -> None:
        """Multi condition shows overhead percentages."""
        meta = _make_meta(["baseline", "coverage"])
        results = [
            _make_package_result(
                "requests",
                conditions={
                    "baseline": _make_condition_result("baseline", median=1.0),
                    "coverage": _make_condition_result("coverage", median=1.2),
                },
            ),
        ]
        output = format_bench_show(meta, results)
        self.assertIn("Overhead", output)
        self.assertIn("Sig.", output)

    def test_description_shown(self) -> None:
        meta = _make_meta()
        meta.description = "Custom description"
        output = format_bench_show(meta, [])
        self.assertIn("Custom description", output)

    def test_strategy_interleaved(self) -> None:
        meta = _make_meta()
        meta.config["interleave"] = True
        output = format_bench_show(meta, [])
        self.assertIn("interleaved", output)

    def test_skipped_packages_not_in_table(self) -> None:
        """Skipped packages are excluded from the table."""
        meta = _make_meta()
        results = [
            _make_package_result("active"),
            BenchPackageResult(package="skipped", skipped=True, skip_reason="fail"),
        ]
        output = format_bench_show(meta, results)
        self.assertIn("active", output)
        # Skipped package should not appear in the table rows.


# ---------------------------------------------------------------------------
# Quality summary tests
# ---------------------------------------------------------------------------


class TestQualitySummary(unittest.TestCase):
    def test_quality_excellent(self) -> None:
        """All CVs < 0.03 yields 'Excellent'."""
        results = [
            _make_package_result(
                f"pkg{i}",
                conditions={"A": _make_condition_result("A", cv=0.02)},
            )
            for i in range(5)
        ]
        output = _format_quality_summary(results, ["A"])
        self.assertIn("Excellent", output)

    def test_quality_poor(self) -> None:
        """CVs > 0.10 yields 'Poor'."""
        results = [
            _make_package_result(
                f"pkg{i}",
                conditions={"A": _make_condition_result("A", cv=0.15)},
            )
            for i in range(5)
        ]
        output = _format_quality_summary(results, ["A"])
        self.assertIn("Poor", output)

    def test_quality_good(self) -> None:
        """CVs between 0.03 and 0.05 yields 'Good'."""
        results = [
            _make_package_result(
                f"pkg{i}",
                conditions={"A": _make_condition_result("A", cv=0.04)},
            )
            for i in range(5)
        ]
        output = _format_quality_summary(results, ["A"])
        self.assertIn("Good", output)

    def test_high_cv_count(self) -> None:
        """Reports count of packages with CV > 10%."""
        results = [
            _make_package_result(
                "high",
                conditions={"A": _make_condition_result("A", cv=0.15)},
            ),
            _make_package_result(
                "low",
                conditions={"A": _make_condition_result("A", cv=0.02)},
            ),
        ]
        output = _format_quality_summary(results, ["A"])
        self.assertIn("CV > 10%: 1", output)


# ---------------------------------------------------------------------------
# Comparison summary tests
# ---------------------------------------------------------------------------


class TestComparisonSummary(unittest.TestCase):
    def test_aggregate_summary(self) -> None:
        """Verify median/mean overhead and most affected packages."""
        results = [
            _make_package_result(
                "fast",
                conditions={
                    "baseline": _make_condition_result("baseline", median=1.0),
                    "treatment": _make_condition_result("treatment", median=1.1),
                },
            ),
            _make_package_result(
                "slow",
                conditions={
                    "baseline": _make_condition_result("baseline", median=2.0),
                    "treatment": _make_condition_result("treatment", median=2.5),
                },
            ),
        ]
        output = format_comparison_summary(results, "baseline", "treatment")
        self.assertIn("Packages compared:", output)
        self.assertIn("Median overhead:", output)
        self.assertIn("Mean overhead:", output)
        self.assertIn("Most affected", output)

    def test_no_comparable_packages(self) -> None:
        output = format_comparison_summary([], "A", "B")
        self.assertIn("No comparable packages found", output)

    def test_improved_packages_shown(self) -> None:
        """Packages faster under treatment are listed."""
        results = [
            _make_package_result(
                "improved",
                conditions={
                    "baseline": _make_condition_result("baseline", median=2.0),
                    "treatment": _make_condition_result("treatment", median=1.5),
                },
            ),
        ]
        output = format_comparison_summary(results, "baseline", "treatment")
        self.assertIn("improved", output)


if __name__ == "__main__":
    unittest.main()
