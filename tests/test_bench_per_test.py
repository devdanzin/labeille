"""Tests for per-test timing capture, parsing, and comparison."""

from __future__ import annotations

import unittest

from bench_test_helpers import make_package_result
from labeille.bench.compare import TestOverhead, compare_per_test
from labeille.bench.display import format_per_test_comparison, format_per_test_summary
from labeille.bench.results import BenchIteration, BenchPackageResult
from labeille.bench.timing import (
    PerTestTimings,
    TestTiming,
    parse_pytest_durations,
    prepare_per_test_command,
)


# ---------------------------------------------------------------------------
# Standard pytest durations output for testing
# ---------------------------------------------------------------------------

STANDARD_DURATIONS_OUTPUT = """\
============================= test session starts ==============================
platform linux -- Python 3.15.0a4, pytest-8.3.4
collected 42 items

tests/test_foo.py ....
tests/test_bar.py ....

=============================== slowest durations ===============================
1.23s call     tests/test_foo.py::test_heavy_computation
0.45s call     tests/test_bar.py::test_network
0.12s setup    tests/test_foo.py::test_heavy_computation
0.01s teardown tests/test_foo.py::test_heavy_computation
0.08s call     tests/test_foo.py::test_light
=============================== 42 passed in 3.50s ===============================
"""


# ---------------------------------------------------------------------------
# TestParsePytestDurations
# ---------------------------------------------------------------------------


class TestParsePytestDurations(unittest.TestCase):
    """Tests for parse_pytest_durations() parser."""

    def test_parse_standard_output(self) -> None:
        result = parse_pytest_durations(STANDARD_DURATIONS_OUTPUT)
        self.assertTrue(result.parse_success)
        self.assertEqual(len(result.timings), 5)

        calls = [t for t in result.timings if t.phase == "call"]
        setups = [t for t in result.timings if t.phase == "setup"]
        teardowns = [t for t in result.timings if t.phase == "teardown"]
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(setups), 1)
        self.assertEqual(len(teardowns), 1)

    def test_parse_empty_output(self) -> None:
        result = parse_pytest_durations("")
        self.assertFalse(result.parse_success)
        self.assertEqual(len(result.timings), 0)

    def test_parse_no_durations_section(self) -> None:
        output = "===== test session starts =====\n42 passed in 3.50s\n"
        result = parse_pytest_durations(output)
        self.assertFalse(result.parse_success)
        self.assertEqual(len(result.timings), 0)

    def test_parse_partial_output(self) -> None:
        output = """\
=== slowest durations ===
1.23s call     tests/test_foo.py::test_heavy
this is a bad line
0.45s call     tests/test_bar.py::test_network
=== 42 passed ===
"""
        result = parse_pytest_durations(output)
        self.assertTrue(result.parse_success)
        self.assertEqual(len(result.timings), 2)

    def test_parse_multiple_tests(self) -> None:
        lines = ["=== slowest durations ==="]
        for i in range(7):
            lines.append(f"{0.5 + i * 0.1:.2f}s call     tests/test_mod.py::test_{i}")
        lines.append("=== 7 passed ===")
        output = "\n".join(lines)
        result = parse_pytest_durations(output)
        self.assertTrue(result.parse_success)
        self.assertEqual(len(result.timings), 7)

    def test_parse_stops_at_next_section(self) -> None:
        output = """\
=== slowest durations ===
1.00s call     tests/test_a.py::test_one
=== short test summary info ===
FAILED tests/test_a.py::test_one
=== 1 failed ===
"""
        result = parse_pytest_durations(output)
        self.assertTrue(result.parse_success)
        self.assertEqual(len(result.timings), 1)
        self.assertEqual(result.timings[0].test_id, "tests/test_a.py::test_one")

    def test_parse_case_insensitive_header(self) -> None:
        output = "=== SLOWEST DURATIONS ===\n0.50s call     tests/test.py::test_x\n"
        result = parse_pytest_durations(output)
        self.assertTrue(result.parse_success)
        self.assertEqual(len(result.timings), 1)

    def test_parse_slow_test_durations_header(self) -> None:
        output = "=== slowest test durations ===\n0.50s call     tests/test.py::test_x\n"
        result = parse_pytest_durations(output)
        self.assertTrue(result.parse_success)
        self.assertEqual(len(result.timings), 1)


# ---------------------------------------------------------------------------
# TestPreparePerTestCommand
# ---------------------------------------------------------------------------


class TestPreparePerTestCommand(unittest.TestCase):
    """Tests for prepare_per_test_command()."""

    def test_pytest_gets_durations(self) -> None:
        cmd, enabled = prepare_per_test_command("python -m pytest tests/", "pytest")
        self.assertEqual(cmd, "python -m pytest tests/ --durations=0")
        self.assertTrue(enabled)

    def test_unittest_unchanged(self) -> None:
        cmd, enabled = prepare_per_test_command("python -m unittest discover", "unittest")
        self.assertEqual(cmd, "python -m unittest discover")
        self.assertFalse(enabled)

    def test_existing_durations_unchanged(self) -> None:
        cmd, enabled = prepare_per_test_command("python -m pytest --durations=10", "pytest")
        self.assertEqual(cmd, "python -m pytest --durations=10")
        self.assertFalse(enabled)

    def test_empty_framework(self) -> None:
        cmd, enabled = prepare_per_test_command("python -m pytest tests/", "")
        self.assertEqual(cmd, "python -m pytest tests/")
        self.assertFalse(enabled)


# ---------------------------------------------------------------------------
# TestPerTestTimingsProperties
# ---------------------------------------------------------------------------


class TestPerTestTimingsProperties(unittest.TestCase):
    """Tests for PerTestTimings properties."""

    def _make_timings(self) -> PerTestTimings:
        return PerTestTimings(
            timings=[
                TestTiming(test_id="test_a", phase="setup", duration_s=0.1),
                TestTiming(test_id="test_a", phase="call", duration_s=1.0),
                TestTiming(test_id="test_a", phase="teardown", duration_s=0.05),
                TestTiming(test_id="test_b", phase="call", duration_s=2.0),
                TestTiming(test_id="test_c", phase="call", duration_s=0.5),
            ]
        )

    def test_by_test_grouping(self) -> None:
        timings = self._make_timings()
        by_test = timings.by_test
        self.assertIn("test_a", by_test)
        self.assertEqual(by_test["test_a"]["call"], 1.0)
        self.assertEqual(by_test["test_a"]["setup"], 0.1)

    def test_slowest_tests(self) -> None:
        timings = self._make_timings()
        slowest = timings.slowest_tests
        self.assertEqual(slowest[0], ("test_b", 2.0))
        self.assertEqual(slowest[1], ("test_a", 1.0))
        self.assertEqual(slowest[2], ("test_c", 0.5))

    def test_total_test_time(self) -> None:
        timings = self._make_timings()
        self.assertAlmostEqual(timings.total_test_time_s, 3.5)

    def test_test_count(self) -> None:
        timings = self._make_timings()
        self.assertEqual(timings.test_count, 3)

    def test_serialization_roundtrip(self) -> None:
        original = self._make_timings()
        d = original.to_dict()
        restored = PerTestTimings.from_dict(d)
        self.assertEqual(len(restored.timings), 5)
        self.assertTrue(restored.parse_success)
        self.assertEqual(restored.timings[0].test_id, "test_a")


# ---------------------------------------------------------------------------
# TestTestTiming
# ---------------------------------------------------------------------------


class TestTestTiming(unittest.TestCase):
    """Tests for TestTiming serialization."""

    def test_to_dict_roundtrip(self) -> None:
        t = TestTiming(test_id="tests/test_x.py::test_y", phase="call", duration_s=1.234)
        d = t.to_dict()
        restored = TestTiming.from_dict(d)
        self.assertEqual(restored.test_id, t.test_id)
        self.assertEqual(restored.phase, t.phase)
        self.assertAlmostEqual(restored.duration_s, 1.234, places=3)

    def test_from_dict_ignores_unknown(self) -> None:
        d = {
            "test_id": "test_x",
            "phase": "call",
            "duration_s": 0.5,
            "unknown_field": "should be ignored",
        }
        t = TestTiming.from_dict(d)
        self.assertEqual(t.test_id, "test_x")
        self.assertFalse(hasattr(t, "unknown_field"))


# ---------------------------------------------------------------------------
# TestComparePerTest
# ---------------------------------------------------------------------------


def _make_pkg_with_per_test(
    baseline_timings: list[list[TestTiming]],
    treatment_timings: list[list[TestTiming]],
) -> BenchPackageResult:
    """Create a BenchPackageResult with per-test timing data."""
    pkg = make_package_result(
        "test-pkg",
        {
            "baseline": [5.0] * len(baseline_timings),
            "treatment": [5.0] * len(treatment_timings),
        },
    )
    # Attach per-test timings to measured iterations.
    for i, test_timings in enumerate(baseline_timings):
        pkg.conditions["baseline"].measured_iterations[i].per_test_timings = PerTestTimings(
            timings=test_timings
        )
    for i, test_timings in enumerate(treatment_timings):
        pkg.conditions["treatment"].measured_iterations[i].per_test_timings = PerTestTimings(
            timings=test_timings
        )
    return pkg


class TestComparePerTest(unittest.TestCase):
    """Tests for compare_per_test()."""

    def test_compare_two_conditions(self) -> None:
        base_timings = [
            [TestTiming("test_a", "call", 1.0), TestTiming("test_b", "call", 2.0)],
        ]
        treat_timings = [
            [TestTiming("test_a", "call", 1.5), TestTiming("test_b", "call", 2.2)],
        ]
        pkg = _make_pkg_with_per_test(base_timings, treat_timings)
        overheads = compare_per_test([pkg], "baseline", "treatment", "test-pkg")
        self.assertEqual(len(overheads), 2)
        # Sorted by overhead_pct descending.
        self.assertEqual(overheads[0].test_id, "test_a")  # 50% overhead
        self.assertAlmostEqual(overheads[0].overhead_pct, 50.0, places=0)

    def test_compare_missing_per_test_data(self) -> None:
        pkg = make_package_result(
            "test-pkg",
            {"baseline": [5.0, 5.0], "treatment": [5.0, 5.0]},
        )
        overheads = compare_per_test([pkg], "baseline", "treatment", "test-pkg")
        self.assertEqual(overheads, [])

    def test_compare_empty_results(self) -> None:
        overheads = compare_per_test([], "baseline", "treatment", "test-pkg")
        self.assertEqual(overheads, [])

    def test_compare_test_only_in_one_condition(self) -> None:
        base_timings = [
            [TestTiming("test_a", "call", 1.0), TestTiming("test_unique", "call", 3.0)],
        ]
        treat_timings = [
            [TestTiming("test_a", "call", 1.5)],
        ]
        pkg = _make_pkg_with_per_test(base_timings, treat_timings)
        overheads = compare_per_test([pkg], "baseline", "treatment", "test-pkg")
        # Only test_a is in both conditions.
        self.assertEqual(len(overheads), 1)
        self.assertEqual(overheads[0].test_id, "test_a")

    def test_compare_median_across_iterations(self) -> None:
        # Multiple iterations: median should be computed across them.
        base_timings = [
            [TestTiming("test_a", "call", 1.0)],
            [TestTiming("test_a", "call", 2.0)],
            [TestTiming("test_a", "call", 3.0)],
        ]
        treat_timings = [
            [TestTiming("test_a", "call", 2.0)],
            [TestTiming("test_a", "call", 4.0)],
            [TestTiming("test_a", "call", 6.0)],
        ]
        pkg = _make_pkg_with_per_test(base_timings, treat_timings)
        overheads = compare_per_test([pkg], "baseline", "treatment", "test-pkg")
        self.assertEqual(len(overheads), 1)
        # Median baseline = 2.0, median treatment = 4.0
        self.assertAlmostEqual(overheads[0].baseline_median_s, 2.0)
        self.assertAlmostEqual(overheads[0].treatment_median_s, 4.0)
        self.assertAlmostEqual(overheads[0].overhead_pct, 100.0)


# ---------------------------------------------------------------------------
# TestBenchIterationPerTest
# ---------------------------------------------------------------------------


class TestBenchIterationPerTest(unittest.TestCase):
    """Tests for BenchIteration per_test_timings field."""

    def test_iteration_with_per_test_roundtrip(self) -> None:
        iteration = BenchIteration(
            index=1,
            warmup=False,
            wall_time_s=5.0,
            user_time_s=4.0,
            sys_time_s=0.5,
            peak_rss_mb=256.0,
            exit_code=0,
            status="ok",
            per_test_timings=PerTestTimings(
                timings=[TestTiming("test_x", "call", 1.0)],
            ),
        )
        d = iteration.to_dict()
        self.assertIn("per_test_timings", d)

        restored = BenchIteration.from_dict(d)
        self.assertIsNotNone(restored.per_test_timings)
        assert restored.per_test_timings is not None
        self.assertEqual(len(restored.per_test_timings.timings), 1)
        self.assertEqual(restored.per_test_timings.timings[0].test_id, "test_x")

    def test_iteration_without_per_test(self) -> None:
        iteration = BenchIteration(
            index=1,
            warmup=False,
            wall_time_s=5.0,
            user_time_s=4.0,
            sys_time_s=0.5,
            peak_rss_mb=256.0,
            exit_code=0,
            status="ok",
        )
        d = iteration.to_dict()
        self.assertNotIn("per_test_timings", d)

    def test_iteration_from_old_format(self) -> None:
        d = {
            "index": 1,
            "warmup": False,
            "wall_time_s": 5.0,
            "user_time_s": 4.0,
            "sys_time_s": 0.5,
            "peak_rss_mb": 256.0,
            "exit_code": 0,
            "status": "ok",
        }
        iteration = BenchIteration.from_dict(d)
        self.assertIsNone(iteration.per_test_timings)


# ---------------------------------------------------------------------------
# TestFormatPerTest (display integration)
# ---------------------------------------------------------------------------


class TestFormatPerTest(unittest.TestCase):
    """Tests for per-test display formatting functions."""

    def test_format_per_test_summary_empty(self) -> None:
        timings = PerTestTimings(timings=[])
        text = format_per_test_summary(timings)
        self.assertIn("No per-test timing data", text)

    def test_format_per_test_summary_with_data(self) -> None:
        timings = PerTestTimings(
            timings=[
                TestTiming("test_a", "call", 1.5),
                TestTiming("test_b", "call", 0.5),
            ]
        )
        text = format_per_test_summary(timings)
        self.assertIn("test_a", text)
        self.assertIn("test_b", text)
        self.assertIn("2.00s", text)  # total test time

    def test_format_per_test_comparison_empty(self) -> None:
        text = format_per_test_comparison([])
        self.assertIn("No per-test comparison", text)

    def test_format_per_test_comparison_with_data(self) -> None:
        overheads = [
            TestOverhead(
                test_id="tests/test_foo.py::test_heavy",
                baseline_median_s=1.0,
                treatment_median_s=1.5,
                absolute_diff_s=0.5,
                overhead_pct=50.0,
                n_baseline=5,
                n_treatment=5,
            ),
        ]
        text = format_per_test_comparison(overheads)
        self.assertIn("test_heavy", text)
        self.assertIn("50.0%", text)


if __name__ == "__main__":
    unittest.main()
