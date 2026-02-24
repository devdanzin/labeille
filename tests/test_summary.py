"""Tests for labeille.summary."""

import signal
import unittest
from pathlib import Path

from labeille.analyze import result_detail
from labeille.formatting import format_duration, format_signal_name
from labeille.runner import PackageResult, RunnerConfig, RunSummary
from labeille.summary import (
    _format_aggregate,
    _format_crash_detail,
    _format_package_table,
    _format_run_header,
    format_summary,
)


def _make_config() -> RunnerConfig:
    return RunnerConfig(
        target_python=Path("/usr/bin/python3"),
        registry_dir=Path("registry"),
        results_dir=Path("results"),
        run_id="test-run-2026",
    )


def _make_results() -> list[PackageResult]:
    return [
        PackageResult(
            package="crash-pkg",
            status="crash",
            signal=signal.SIGSEGV,
            crash_signature="Fatal Python error: Segmentation fault",
            duration_seconds=83.0,
            test_command="python -m pytest tests/",
        ),
        PackageResult(
            package="pass-pkg",
            status="pass",
            exit_code=0,
            duration_seconds=8.0,
        ),
        PackageResult(
            package="fail-pkg",
            status="fail",
            exit_code=1,
            duration_seconds=12.0,
        ),
        PackageResult(
            package="timeout-pkg",
            status="timeout",
            timeout_hit=True,
            duration_seconds=900.0,
        ),
        PackageResult(
            package="error-pkg",
            status="install_error",
            error_message="Install failed: build error in some-package",
            duration_seconds=3.0,
        ),
    ]


def _make_summary(results: list[PackageResult] | None = None) -> RunSummary:
    if results is None:
        results = _make_results()
    s = RunSummary(total=6, tested=len(results), skipped=1)
    for r in results:
        if r.status == "pass":
            s.passed += 1
        elif r.status == "fail":
            s.failed += 1
        elif r.status == "crash":
            s.crashed += 1
        elif r.status == "timeout":
            s.timed_out += 1
        elif r.status == "install_error":
            s.install_errors += 1
        elif r.status == "clone_error":
            s.clone_errors += 1
        else:
            s.errors += 1
    return s


class TestFormatDuration(unittest.TestCase):
    def test_seconds_only(self) -> None:
        self.assertEqual(format_duration(5.0), "5s")

    def test_zero(self) -> None:
        self.assertEqual(format_duration(0.0), "0s")

    def test_minutes_and_seconds(self) -> None:
        self.assertEqual(format_duration(125.0), "2m  5s")

    def test_exact_minute(self) -> None:
        self.assertEqual(format_duration(60.0), "1m  0s")

    def test_hours_minutes_seconds(self) -> None:
        self.assertEqual(format_duration(3661.0), "1h  1m  1s")

    def test_large_hours(self) -> None:
        self.assertEqual(format_duration(7384.0), "2h  3m  4s")

    def test_just_under_a_minute(self) -> None:
        self.assertEqual(format_duration(59.9), "59s")

    def test_just_over_a_minute(self) -> None:
        self.assertEqual(format_duration(61.0), "1m  1s")


class TestFormatSignalName(unittest.TestCase):
    def test_none(self) -> None:
        self.assertEqual(format_signal_name(None), "")

    def test_sigsegv(self) -> None:
        self.assertEqual(format_signal_name(signal.SIGSEGV), "SIGSEGV")

    def test_sigabrt(self) -> None:
        self.assertEqual(format_signal_name(signal.SIGABRT), "SIGABRT")


class TestResultDetail(unittest.TestCase):
    def test_crash(self) -> None:
        r = PackageResult(package="p", status="crash", crash_signature="Fatal error in something")
        self.assertEqual(result_detail(r), "Fatal error in something")

    def test_crash_truncates_at_60(self) -> None:
        r = PackageResult(package="p", status="crash", crash_signature="A" * 80)
        self.assertEqual(len(result_detail(r)), 60)

    def test_timeout(self) -> None:
        r = PackageResult(package="p", status="timeout", duration_seconds=900.0)
        self.assertEqual(result_detail(r), "(timeout: 900s)")

    def test_fail(self) -> None:
        r = PackageResult(package="p", status="fail", exit_code=2)
        self.assertEqual(result_detail(r), "exit code 2")

    def test_error(self) -> None:
        r = PackageResult(package="p", status="install_error", error_message="build failed")
        self.assertEqual(result_detail(r), "build failed")

    def test_pass(self) -> None:
        r = PackageResult(package="p", status="pass")
        self.assertEqual(result_detail(r), "")


class TestFormatRunHeader(unittest.TestCase):
    def test_header_content(self) -> None:
        header = _format_run_header("my-run", "3.15.0a5", True, 3661.0)
        self.assertIn("my-run", header)
        self.assertIn("3.15.0a5", header)
        self.assertIn("yes", header)
        self.assertIn("1h  1m  1s", header)

    def test_jit_disabled(self) -> None:
        header = _format_run_header("run1", "3.15.0", False, 60.0)
        self.assertIn("no", header)


class TestFormatPackageTable(unittest.TestCase):
    def test_empty_results(self) -> None:
        self.assertEqual(_format_package_table([]), "")

    def test_show_passing_true(self) -> None:
        results = _make_results()
        table = _format_package_table(results, show_passing=True)
        self.assertIn("pass-pkg", table)
        self.assertIn("crash-pkg", table)
        self.assertIn("Results", table)

    def test_show_passing_false(self) -> None:
        results = _make_results()
        table = _format_package_table(results, show_passing=False)
        self.assertNotIn("pass-pkg", table)
        self.assertIn("crash-pkg", table)
        self.assertIn("fail-pkg", table)

    def test_all_passing_hidden(self) -> None:
        results = [PackageResult(package="ok", status="pass", duration_seconds=1.0)]
        table = _format_package_table(results, show_passing=False)
        self.assertEqual(table, "")

    def test_sorted_by_status_order(self) -> None:
        results = _make_results()
        table = _format_package_table(results, show_passing=True)
        lines = table.splitlines()
        data_lines = [ln for ln in lines if ln.startswith("  ") and "Package" not in ln]
        # First data line should be crash, last should be pass
        self.assertIn("CRASH", data_lines[0])
        self.assertIn("PASS", data_lines[-1])

    def test_truncates_longresult_detail(self) -> None:
        results = [
            PackageResult(
                package="long",
                status="install_error",
                error_message="A" * 100,
                duration_seconds=1.0,
            ),
        ]
        table = _format_package_table(results, show_passing=True)
        # The detail column should be truncated with ellipsis
        self.assertIn("\u2026", table)


class TestFormatAggregate(unittest.TestCase):
    def test_has_counts(self) -> None:
        results = _make_results()
        summary = _make_summary(results)
        text = _format_aggregate(results, summary, 1200.0)
        self.assertIn("Packages tested: 5 / 6", text)
        self.assertIn("Passed:", text)
        self.assertIn("Failed:", text)
        self.assertIn("Crashed:", text)

    def test_has_percentages(self) -> None:
        results = _make_results()
        summary = _make_summary(results)
        text = _format_aggregate(results, summary, 100.0)
        self.assertIn("20.0%", text)  # 1 pass out of 5

    def test_has_timing_stats(self) -> None:
        results = _make_results()
        summary = _make_summary(results)
        text = _format_aggregate(results, summary, 1200.0)
        self.assertIn("Total time:", text)
        self.assertIn("Avg per package:", text)
        self.assertIn("Fastest:", text)
        self.assertIn("Slowest:", text)

    def test_fastest_slowest(self) -> None:
        results = _make_results()
        summary = _make_summary(results)
        text = _format_aggregate(results, summary, 100.0)
        self.assertIn("error-pkg", text)  # fastest at 3s
        self.assertIn("timeout-pkg", text)  # slowest at 900s

    def test_empty_results(self) -> None:
        summary = RunSummary(total=5, tested=0)
        text = _format_aggregate([], summary, 0.0)
        self.assertIn("Packages tested: 0 / 5", text)

    def test_zero_tested_percentage(self) -> None:
        summary = RunSummary(total=5, tested=0)
        text = _format_aggregate([], summary, 0.0)
        self.assertIn("0.0%", text)


class TestFormatCrashDetail(unittest.TestCase):
    def test_no_crashes(self) -> None:
        results = [PackageResult(package="ok", status="pass")]
        self.assertEqual(_format_crash_detail(results), "")

    def test_with_crashes(self) -> None:
        results = _make_results()
        text = _format_crash_detail(results)
        self.assertIn("crash-pkg", text)
        self.assertIn("SIGSEGV", text)
        self.assertIn("Crashes", text)

    def test_crash_has_test_command(self) -> None:
        results = _make_results()
        text = _format_crash_detail(results)
        self.assertIn("Test command:", text)
        self.assertIn("python -m pytest tests/", text)

    def test_crash_with_run_dir(self) -> None:
        results = _make_results()
        run_dir = Path("/tmp/results/my-run")
        text = _format_crash_detail(results, run_dir=run_dir)
        self.assertIn("Stderr:", text)
        self.assertIn("crash-pkg.stderr", text)


class TestFormatSummary(unittest.TestCase):
    def test_verbose_mode(self) -> None:
        results = _make_results()
        summary = _make_summary(results)
        config = _make_config()
        text = format_summary(results, summary, config, "3.15.0a5", True, 1200.0, mode="verbose")
        # Should have header, table with all packages, aggregate, crash detail
        self.assertIn("Run ID:", text)
        self.assertIn("pass-pkg", text)
        self.assertIn("crash-pkg", text)
        self.assertIn("Packages tested:", text)
        self.assertIn("Crashes", text)

    def test_default_mode(self) -> None:
        results = _make_results()
        summary = _make_summary(results)
        config = _make_config()
        text = format_summary(results, summary, config, "3.15.0a5", True, 1200.0, mode="default")
        # Should have header, table without passing, aggregate, crash detail
        self.assertIn("Run ID:", text)
        self.assertNotIn("pass-pkg", text)
        self.assertIn("crash-pkg", text)
        self.assertIn("Packages tested:", text)

    def test_quiet_mode_with_crashes(self) -> None:
        results = _make_results()
        summary = _make_summary(results)
        config = _make_config()
        text = format_summary(results, summary, config, "3.15.0a5", True, 1200.0, mode="quiet")
        # Should have crash detail and one-line count
        self.assertIn("crash-pkg", text)
        self.assertIn("1 crash found", text)
        # Should NOT have full header or aggregate
        self.assertNotIn("Run ID:", text)
        self.assertNotIn("Packages tested:", text)

    def test_quiet_mode_no_crashes(self) -> None:
        results = [PackageResult(package="ok", status="pass", duration_seconds=5.0)]
        summary = _make_summary(results)
        config = _make_config()
        text = format_summary(results, summary, config, "3.15.0a5", True, 10.0, mode="quiet")
        self.assertEqual(text, "")

    def test_quiet_mode_plural_crashes(self) -> None:
        results = [
            PackageResult(
                package="c1",
                status="crash",
                signal=signal.SIGSEGV,
                crash_signature="sig1",
                duration_seconds=1.0,
                test_command="pytest",
            ),
            PackageResult(
                package="c2",
                status="crash",
                signal=signal.SIGABRT,
                crash_signature="sig2",
                duration_seconds=2.0,
                test_command="pytest",
            ),
        ]
        summary = _make_summary(results)
        config = _make_config()
        text = format_summary(results, summary, config, "3.15.0", True, 10.0, mode="quiet")
        self.assertIn("2 crashes found", text)

    def test_no_results(self) -> None:
        summary = RunSummary(total=5, tested=0)
        config = _make_config()
        text = format_summary([], summary, config, "3.15.0", True, 0.0, mode="default")
        self.assertIn("Packages tested: 0 / 5", text)

    def test_run_dir_in_crashresult_detail(self) -> None:
        results = _make_results()
        summary = _make_summary(results)
        config = _make_config()
        run_dir = Path("/tmp/results/my-run")
        text = format_summary(
            results,
            summary,
            config,
            "3.15.0",
            True,
            100.0,
            run_dir=run_dir,
            mode="verbose",
        )
        self.assertIn("crash-pkg.stderr", text)


if __name__ == "__main__":
    unittest.main()
