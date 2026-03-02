"""Tests for labeille.ft.export — CSV, JSON, and markdown report export."""

from __future__ import annotations

import csv
import io
import json
import unittest

from labeille.ft.export import export_csv, export_json, generate_report
from labeille.ft.results import (
    FailureCategory,
    FTPackageResult,
    FTRunMeta,
    IterationOutcome,
)


def _make_result(
    package: str = "pkg",
    category: FailureCategory = FailureCategory.COMPATIBLE,
    pass_rate: float = 1.0,
    crash_count: int = 0,
    **kwargs: object,
) -> FTPackageResult:
    """Create a minimal FTPackageResult for export tests."""
    if "pass_count" not in kwargs:
        kwargs["pass_count"] = int(pass_rate * 10)
    if "iterations_completed" not in kwargs:
        kwargs["iterations_completed"] = 10
    return FTPackageResult(
        package=package,
        category=category,
        pass_rate=pass_rate,
        crash_count=crash_count,
        **kwargs,  # type: ignore[arg-type]
    )


def _make_meta(**kwargs: object) -> FTRunMeta:
    return FTRunMeta(
        run_id=kwargs.get("run_id", "test-run"),  # type: ignore[arg-type]
        timestamp=kwargs.get("timestamp", "2026-01-15T12:00:00"),  # type: ignore[arg-type]
        python_profile=kwargs.get(  # type: ignore[arg-type]
            "python_profile",
            {"version": "3.14.0b2", "gil_disabled": True, "implementation": "cpython"},
        ),
        system_profile=kwargs.get(  # type: ignore[arg-type]
            "system_profile",
            {"cpu_model": "AMD Ryzen 9", "ram_total_gb": 64, "os_distro": "Ubuntu 24.04"},
        ),
        config=kwargs.get("config", {"iterations": 10}),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# CSV export tests
# ---------------------------------------------------------------------------


class TestExportCsv(unittest.TestCase):
    def test_export_csv_header(self) -> None:
        results = [_make_result("numpy")]
        output = export_csv(results)
        reader = csv.reader(io.StringIO(output))
        header = next(reader)
        self.assertIn("package", header)
        self.assertIn("category", header)
        self.assertIn("pass_rate", header)
        self.assertIn("crash_count", header)

    def test_export_csv_row_count(self) -> None:
        results = [_make_result(f"pkg-{i}") for i in range(5)]
        output = export_csv(results)
        lines = [ln for ln in output.strip().split("\n") if ln]
        self.assertEqual(len(lines), 6)  # header + 5 data rows

    def test_export_csv_values(self) -> None:
        results = [
            _make_result(
                "numpy",
                FailureCategory.CRASH,
                0.7,
                crash_count=3,
            )
        ]
        output = export_csv(results)
        reader = csv.DictReader(io.StringIO(output))
        row = next(reader)
        self.assertEqual(row["package"], "numpy")
        self.assertEqual(row["category"], "crash")
        self.assertEqual(row["crash_count"], "3")
        self.assertAlmostEqual(float(row["pass_rate"]), 0.7, places=1)

    def test_export_csv_sorted_by_name(self) -> None:
        results = [_make_result("zlib"), _make_result("aiohttp"), _make_result("numpy")]
        output = export_csv(results)
        reader = csv.DictReader(io.StringIO(output))
        names = [row["package"] for row in reader]
        self.assertEqual(names, ["aiohttp", "numpy", "zlib"])

    def test_export_csv_special_characters(self) -> None:
        results = [
            _make_result(
                "pkg",
                failure_signatures=["SIGSEGV in foo, bar, baz"],
            )
        ]
        output = export_csv(results)
        reader = csv.DictReader(io.StringIO(output))
        row = next(reader)
        self.assertIn("SIGSEGV in foo, bar, baz", row["failure_signatures"])

    def test_export_csv_empty_results(self) -> None:
        output = export_csv([])
        lines = [ln for ln in output.strip().split("\n") if ln]
        self.assertEqual(len(lines), 1)  # header only


# ---------------------------------------------------------------------------
# JSON export tests
# ---------------------------------------------------------------------------


class TestExportJson(unittest.TestCase):
    def test_export_json_valid(self) -> None:
        results = [_make_result(f"pkg-{i}") for i in range(3)]
        output = export_json(results)
        data = json.loads(output)
        self.assertEqual(len(data), 3)

    def test_export_json_roundtrip(self) -> None:
        results = [
            _make_result("numpy", FailureCategory.CRASH, 0.7, crash_count=3),
            _make_result("requests", FailureCategory.COMPATIBLE, 1.0),
        ]
        output = export_json(results)
        data = json.loads(output)
        for item in data:
            loaded = FTPackageResult.from_dict(item)
            self.assertIsInstance(loaded, FTPackageResult)


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestGenerateReport(unittest.TestCase):
    def _make_test_results(self) -> list[FTPackageResult]:
        return [
            _make_result("requests", FailureCategory.COMPATIBLE, 1.0),
            _make_result(
                "numpy", FailureCategory.CRASH, 0.7, crash_count=3, failure_signatures=["SIGSEGV"]
            ),
            _make_result(
                "aiohttp",
                FailureCategory.INTERMITTENT,
                0.7,
                flaky_tests={"test_a": 2, "test_b": 1},
                iterations=[
                    IterationOutcome(
                        index=i,
                        status="pass" if i < 7 else "fail",
                        exit_code=0 if i < 7 else 1,
                        duration_s=10.0,
                    )
                    for i in range(10)
                ],
            ),
        ]

    def test_report_markdown_has_header(self) -> None:
        meta = _make_meta()
        results = self._make_test_results()
        output = generate_report(meta, results, format="markdown")
        self.assertTrue(output.startswith("# Free-Threading Compatibility Report"))

    def test_report_markdown_has_summary_table(self) -> None:
        meta = _make_meta()
        results = self._make_test_results()
        output = generate_report(meta, results, format="markdown")
        self.assertIn("| Category |", output)

    def test_report_crashes_section(self) -> None:
        meta = _make_meta()
        results = self._make_test_results()
        output = generate_report(meta, results, format="markdown")
        self.assertIn("## Packages with crashes", output)

    def test_report_no_crashes_section(self) -> None:
        meta = _make_meta()
        results = [_make_result("a", FailureCategory.COMPATIBLE, 1.0)]
        output = generate_report(meta, results, format="markdown")
        self.assertNotIn("## Packages with crashes", output)

    def test_report_deadlocks_section(self) -> None:
        meta = _make_meta()
        results = [
            _make_result(
                "gevent",
                FailureCategory.DEADLOCK,
                0.0,
                pass_count=0,
                deadlock_count=5,
                iterations=[
                    IterationOutcome(
                        index=0,
                        status="deadlock",
                        exit_code=-9,
                        duration_s=60.0,
                        last_output_line="PASSED test_a",
                    )
                ],
            )
        ]
        output = generate_report(meta, results, format="markdown")
        self.assertIn("## Packages with deadlocks", output)

    def test_report_intermittent_section(self) -> None:
        meta = _make_meta()
        results = [
            _make_result(
                "aiohttp",
                FailureCategory.INTERMITTENT,
                0.7,
                flaky_tests={"test_a": 2},
                iterations=[
                    IterationOutcome(
                        index=i,
                        status="pass" if i < 7 else "fail",
                        exit_code=0 if i < 7 else 1,
                        duration_s=10.0,
                    )
                    for i in range(10)
                ],
            )
        ]
        output = generate_report(meta, results, format="markdown")
        self.assertIn("## Packages with intermittent failures", output)

    def test_report_tsan_section(self) -> None:
        meta = _make_meta()
        results = [
            _make_result(
                "numpy",
                FailureCategory.TSAN_WARNINGS,
                1.0,
                tsan_warning_iterations=5,
                tsan_warning_types=["data-race"],
            )
        ]
        output = generate_report(meta, results, format="markdown")
        self.assertIn("## TSAN warnings", output)

    def test_report_extensions_section(self) -> None:
        meta = _make_meta()
        results = [
            _make_result(
                "numpy",
                FailureCategory.CRASH,
                0.7,
                crash_count=3,
                failure_signatures=["SIGSEGV"],
                extension_compat={
                    "package": "numpy",
                    "is_pure_python": False,
                    "gil_fallback_active": True,
                },
            )
        ]
        output = generate_report(meta, results, format="markdown")
        self.assertIn("## C extension compatibility", output)

    def test_report_text_format(self) -> None:
        meta = _make_meta()
        results = self._make_test_results()
        output = generate_report(meta, results, format="text")
        self.assertNotIn("##", output)
        self.assertIn("Compatibility Summary", output)

    def test_report_footer(self) -> None:
        meta = _make_meta()
        results = [_make_result("a")]
        output = generate_report(meta, results, format="markdown")
        self.assertIn("Generated by labeille", output)


if __name__ == "__main__":
    unittest.main()
