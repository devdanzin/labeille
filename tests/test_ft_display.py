"""Tests for labeille.ft.display — terminal display formatting."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from labeille.ft.display import (
    format_compatibility_summary,
    format_flakiness_profile,
    format_gil_comparison,
    format_package_table,
    format_progress,
    format_triage_list,
)
from labeille.ft.results import FailureCategory, FTPackageResult


def _make_result(
    package: str = "test-pkg",
    category: FailureCategory = FailureCategory.COMPATIBLE,
    pass_count: int = 10,
    iterations_completed: int = 10,
    crash_count: int = 0,
    pass_rate: float = 1.0,
    **kwargs: object,
) -> FTPackageResult:
    """Create a minimal FTPackageResult for display tests."""
    return FTPackageResult(
        package=package,
        category=category,
        pass_count=pass_count,
        iterations_completed=iterations_completed,
        crash_count=crash_count,
        pass_rate=pass_rate,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# format_compatibility_summary tests
# ---------------------------------------------------------------------------


class TestFormatCompatibilitySummary(unittest.TestCase):
    def test_summary_basic(self) -> None:
        summary = {
            "total_packages": 100,
            "categories": {"compatible": 70, "crash": 10, "incompatible": 20},
        }
        output = format_compatibility_summary(summary)
        self.assertIn("Compatible:", output)
        self.assertIn("70", output)
        self.assertIn("Crash", output)

    def test_summary_percentages(self) -> None:
        summary = {
            "total_packages": 100,
            "categories": {"compatible": 70},
        }
        output = format_compatibility_summary(summary)
        self.assertIn("70.0%", output)

    def test_summary_zero_categories_omitted(self) -> None:
        summary = {
            "total_packages": 100,
            "categories": {"compatible": 100, "crash": 0},
        }
        output = format_compatibility_summary(summary)
        self.assertNotIn("Crash", output)

    def test_summary_with_python_info(self) -> None:
        summary = {"total_packages": 10, "categories": {"compatible": 10}}
        output = format_compatibility_summary(
            summary, python_info="Python: 3.14.0b2 (free-threaded)"
        )
        self.assertIn("Python: 3.14.0b2 (free-threaded)", output)

    def test_summary_with_system_info(self) -> None:
        summary = {"total_packages": 10, "categories": {"compatible": 10}}
        output = format_compatibility_summary(summary, system_info="System: AMD Ryzen 9, 64GB RAM")
        self.assertIn("System: AMD Ryzen 9, 64GB RAM", output)

    def test_summary_extension_breakdown(self) -> None:
        summary = {
            "total_packages": 350,
            "categories": {"compatible": 250},
            "pure_python_count": 200,
            "extension_count": 150,
            "pure_python_compatible_pct": 85.0,
            "extension_compatible_pct": 60.0,
        }
        output = format_compatibility_summary(summary)
        self.assertIn("Pure Python:", output)
        self.assertIn("200", output)
        self.assertIn("85.0%", output)
        self.assertIn("C extensions:", output)
        self.assertIn("150", output)
        self.assertIn("60.0%", output)


# ---------------------------------------------------------------------------
# format_package_table tests
# ---------------------------------------------------------------------------


class TestFormatPackageTable(unittest.TestCase):
    def test_table_sorted_by_category(self) -> None:
        results = [
            _make_result("pkg-a", FailureCategory.CRASH, crash_count=3, pass_rate=0.7),
            _make_result("pkg-b", FailureCategory.COMPATIBLE, pass_rate=1.0),
            _make_result("pkg-c", FailureCategory.INCOMPATIBLE, pass_rate=0.0, pass_count=0),
        ]
        output = format_package_table(results, sort_by="category")
        lines = output.split("\n")
        # Compatible has lowest severity → first, then crash and incompatible.
        data_lines = [ln for ln in lines if ln.startswith("pkg-")]
        self.assertTrue(data_lines[0].startswith("pkg-b"))

    def test_table_sorted_by_name(self) -> None:
        results = [
            _make_result("zlib", FailureCategory.COMPATIBLE),
            _make_result("aiohttp", FailureCategory.CRASH),
        ]
        output = format_package_table(results, sort_by="name")
        lines = [ln for ln in output.split("\n") if ln.startswith(("aiohttp", "zlib"))]
        self.assertTrue(lines[0].startswith("aiohttp"))

    def test_table_max_rows(self) -> None:
        results = [_make_result(f"pkg-{i}") for i in range(10)]
        output = format_package_table(results, max_rows=5)
        self.assertIn("5 more", output)

    def test_table_details_column(self) -> None:
        r = _make_result(
            "numpy",
            FailureCategory.CRASH,
            crash_count=3,
            failure_signatures=["SIGSEGV in _multiarray"],
        )
        output = format_package_table([r])
        self.assertIn("SIGSEGV", output)

    def test_table_details_flaky(self) -> None:
        r = _make_result(
            "aiohttp",
            FailureCategory.INTERMITTENT,
            flaky_tests={"test_a": 2, "test_b": 1, "test_c": 3},
        )
        output = format_package_table([r])
        self.assertIn("3 flaky tests", output)

    def test_table_install_failure(self) -> None:
        r = _make_result(
            "broken-pkg",
            FailureCategory.INSTALL_FAILURE,
            install_ok=False,
            pass_count=0,
            iterations_completed=0,
        )
        output = format_package_table([r])
        self.assertIn("install failed", output)


# ---------------------------------------------------------------------------
# format_triage_list tests
# ---------------------------------------------------------------------------


class TestFormatTriageList(unittest.TestCase):
    def _make_triage(
        self,
        package: str = "pkg",
        score: float = 50.0,
        category: str = "crash",
        reason: str = "3 crashes",
    ) -> MagicMock:
        entry = MagicMock()
        entry.package = package
        entry.priority_score = score
        entry.category = category
        entry.reason = reason
        return entry

    def test_triage_ordered_by_score(self) -> None:
        entries = [
            self._make_triage("numpy", 70, "crash"),
            self._make_triage("gevent", 60, "deadlock"),
        ]
        output = format_triage_list(entries)
        lines = output.split("\n")
        data_lines = [ln for ln in lines if ". " in ln]
        self.assertIn("numpy", data_lines[0])
        self.assertIn("gevent", data_lines[1])

    def test_triage_max_entries(self) -> None:
        entries = [self._make_triage(f"pkg-{i}", 100 - i) for i in range(30)]
        output = format_triage_list(entries, max_entries=10)
        self.assertIn("20 more", output)

    def test_triage_shows_reason(self) -> None:
        entries = [self._make_triage("numpy", 70, "crash", "3 crashes; C extension")]
        output = format_triage_list(entries)
        self.assertIn("3 crashes; C extension", output)


# ---------------------------------------------------------------------------
# format_flakiness_profile tests
# ---------------------------------------------------------------------------


class TestFormatFlakinessProfile(unittest.TestCase):
    def _make_profile(self, **kwargs: object) -> MagicMock:
        profile = MagicMock()
        profile.package = kwargs.get("package", "aiohttp")
        profile.pass_rate = kwargs.get("pass_rate", 0.7)
        profile.total_iterations = kwargs.get("total_iterations", 10)
        profile.pattern = kwargs.get("pattern", "consistent_test")
        profile.failure_modes = kwargs.get("failure_modes", {"fail": 3})
        profile.max_consecutive_passes = kwargs.get("max_consecutive_passes", 4)
        profile.max_consecutive_failures = kwargs.get("max_consecutive_failures", 2)
        profile.flaky_tests = kwargs.get("flaky_tests", [])
        return profile

    def _make_flaky_test(
        self, test_id: str = "test_a", fail_count: int = 3, total_seen: int = 5
    ) -> MagicMock:
        ft = MagicMock()
        ft.test_id = test_id
        ft.fail_count = fail_count
        ft.total_seen = total_seen
        ft.fail_rate = fail_count / total_seen
        return ft

    def test_flakiness_shows_pass_rate(self) -> None:
        profile = self._make_profile(pass_rate=0.7)
        output = format_flakiness_profile(profile)
        self.assertIn("70.0%", output)

    def test_flakiness_shows_pattern(self) -> None:
        profile = self._make_profile(pattern="consistent_test")
        output = format_flakiness_profile(profile)
        self.assertIn("consistent_test", output)

    def test_flakiness_shows_flaky_tests(self) -> None:
        tests = [
            self._make_flaky_test("test_connector::test_close", 3, 5),
            self._make_flaky_test("test_client::test_timeout", 1, 5),
            self._make_flaky_test("test_pool::test_reuse", 2, 5),
        ]
        profile = self._make_profile(flaky_tests=tests)
        output = format_flakiness_profile(profile)
        self.assertIn("test_connector::test_close", output)
        self.assertIn("test_client::test_timeout", output)
        self.assertIn("test_pool::test_reuse", output)

    def test_flakiness_truncates_long_test_ids(self) -> None:
        long_id = "test_very_" + "a" * 50 + "::test_something"
        tests = [self._make_flaky_test(long_id, 2, 5)]
        profile = self._make_profile(flaky_tests=tests)
        output = format_flakiness_profile(profile)
        self.assertIn("...", output)

    def test_flakiness_truncates_many_tests(self) -> None:
        tests = [self._make_flaky_test(f"test_{i}", 1, 5) for i in range(20)]
        profile = self._make_profile(flaky_tests=tests)
        output = format_flakiness_profile(profile)
        self.assertIn("5 more", output)


# ---------------------------------------------------------------------------
# format_gil_comparison tests
# ---------------------------------------------------------------------------


class TestFormatGilComparison(unittest.TestCase):
    def _make_comparison(
        self,
        package: str = "pkg",
        disabled_rate: float = 0.7,
        enabled_rate: float = 1.0,
        classification: str = "ft_intermittent",
        ft_specific: bool = True,
    ) -> MagicMock:
        c = MagicMock()
        c.package = package
        c.gil_disabled_pass_rate = disabled_rate
        c.gil_enabled_pass_rate = enabled_rate
        c.classification = classification
        c.free_threading_specific = ft_specific
        return c

    def test_gil_comparison_basic(self) -> None:
        comparisons = [
            self._make_comparison("requests", 1.0, 1.0, "ft_compatible", False),
            self._make_comparison("numpy", 0.7, 1.0, "ft_intermittent", True),
            self._make_comparison("celery", 0.5, 0.5, "pre_existing", False),
        ]
        output = format_gil_comparison(comparisons)
        self.assertIn("requests", output)
        self.assertIn("numpy", output)
        self.assertIn("celery", output)

    def test_gil_comparison_ft_specific_marker(self) -> None:
        comparisons = [self._make_comparison("numpy", 0.7, 1.0, "ft_intermittent", True)]
        output = format_gil_comparison(comparisons)
        self.assertIn("FT-specific", output)

    def test_gil_comparison_summary_counts(self) -> None:
        comparisons = [
            self._make_comparison("a", 0.7, 1.0, "ft_intermittent", True),
            self._make_comparison("b", 0.5, 1.0, "ft_incompatible", True),
            self._make_comparison("c", 0.5, 0.5, "pre_existing", False),
        ]
        output = format_gil_comparison(comparisons)
        self.assertIn("FT-specific failures: 2", output)
        self.assertIn("Pre-existing failures: 1", output)


# ---------------------------------------------------------------------------
# format_progress tests
# ---------------------------------------------------------------------------


class TestFormatProgress(unittest.TestCase):
    def test_progress_basic(self) -> None:
        output = format_progress(
            package="requests",
            iteration=3,
            total_iterations=10,
            status="pass",
            duration=12.3,
            packages_done=15,
            packages_total=350,
        )
        self.assertEqual(output, "(15/350) requests iter 3/10: pass (12.3s)")


if __name__ == "__main__":
    unittest.main()
