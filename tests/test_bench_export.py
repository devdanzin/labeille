"""Tests for labeille.bench.export — CSV and Markdown export."""

from __future__ import annotations

import csv
import io
import unittest

from bench_test_helpers import make_meta, make_package_result

from labeille.bench.export import export_csv, export_csv_summary, export_markdown
from labeille.bench.results import BenchPackageResult


class TestExportCSV(unittest.TestCase):
    """Tests for export_csv (long-format CSV)."""

    def _make_data(self) -> tuple[object, list[BenchPackageResult]]:
        meta = make_meta(conditions=["baseline", "treatment"])
        results = [
            make_package_result(
                "pkg1",
                {
                    "baseline": [10.0, 10.1, 10.2],
                    "treatment": [12.0, 12.1, 12.2],
                },
            ),
            make_package_result(
                "pkg2",
                {
                    "baseline": [5.0, 5.1, 5.0],
                    "treatment": [6.0, 6.1, 6.0],
                },
            ),
        ]
        return meta, results

    def test_export_csv_header(self) -> None:
        """First line contains all expected column names."""
        meta, results = self._make_data()
        output = export_csv(meta, results)  # type: ignore[arg-type]
        first_line = output.splitlines()[0]
        for col in [
            "package",
            "condition",
            "iteration",
            "warmup",
            "wall_time_s",
            "user_time_s",
            "sys_time_s",
            "peak_rss_mb",
            "exit_code",
            "status",
            "outlier",
            "load_avg_start",
            "load_avg_end",
            "ram_available_start_gb",
        ]:
            self.assertIn(col, first_line)

    def test_export_csv_row_count(self) -> None:
        """2 packages x 2 conditions x 3 iterations = 12 data rows + 1 header."""
        meta, results = self._make_data()
        output = export_csv(meta, results)  # type: ignore[arg-type]
        lines = [line for line in output.splitlines() if line.strip()]
        self.assertEqual(len(lines), 13)

    def test_export_csv_warmup_included(self) -> None:
        """Warm-up iterations appear with warmup=True."""
        meta = make_meta(conditions=["baseline"])
        results = [
            make_package_result(
                "pkg1",
                {
                    "baseline": [10.0, 10.1, 10.2],
                },
                warmup_count=1,
            ),
        ]
        output = export_csv(meta, results)  # type: ignore[arg-type]
        self.assertIn("True", output)  # warmup=True for first iteration

    def test_export_csv_values_correct(self) -> None:
        """Verify specific cell values match input data."""
        meta = make_meta(conditions=["baseline"])
        results = [
            make_package_result("testpkg", {"baseline": [1.234567]}),
        ]
        output = export_csv(meta, results)  # type: ignore[arg-type]
        self.assertIn("testpkg", output)
        self.assertIn("1.234567", output)

    def test_export_csv_skipped_excluded(self) -> None:
        """Skipped packages don't appear."""
        meta = make_meta(conditions=["baseline"])
        results = [
            make_package_result("active", {"baseline": [10.0, 10.1, 10.0]}),
            BenchPackageResult(package="skipped", skipped=True, skip_reason="fail"),
        ]
        output = export_csv(meta, results)  # type: ignore[arg-type]
        self.assertIn("active", output)
        self.assertNotIn("skipped", output)

    def test_export_csv_parseable(self) -> None:
        """Parse output with csv.reader — no errors, correct column count."""
        meta, results = self._make_data()
        output = export_csv(meta, results)  # type: ignore[arg-type]
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        self.assertTrue(len(rows) > 1)
        header_len = len(rows[0])
        for row in rows[1:]:
            self.assertEqual(len(row), header_len)


class TestExportCSVSummary(unittest.TestCase):
    """Tests for export_csv_summary."""

    def test_summary_header(self) -> None:
        """Header has expected columns."""
        meta = make_meta(conditions=["baseline"])
        results = [
            make_package_result("pkg1", {"baseline": [10.0, 10.1, 10.0, 10.1, 10.0]}),
        ]
        output = export_csv_summary(meta, results)  # type: ignore[arg-type]
        first_line = output.splitlines()[0]
        for col in [
            "package",
            "condition",
            "n",
            "wall_mean_s",
            "wall_median_s",
            "wall_stdev_s",
            "wall_cv",
            "cpu_mean_s",
            "rss_median_mb",
            "outliers",
        ]:
            self.assertIn(col, first_line)

    def test_summary_row_count(self) -> None:
        """2 packages x 1 condition = 2 data rows."""
        meta = make_meta(conditions=["baseline"])
        results = [
            make_package_result("pkg1", {"baseline": [10.0, 10.1, 10.0]}),
            make_package_result("pkg2", {"baseline": [5.0, 5.1, 5.0]}),
        ]
        output = export_csv_summary(meta, results)  # type: ignore[arg-type]
        lines = [line for line in output.splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)  # 1 header + 2 data


class TestExportMarkdown(unittest.TestCase):
    """Tests for export_markdown."""

    def test_markdown_title(self) -> None:
        """First line is '# {name}'."""
        meta = make_meta(name="My Benchmark")
        output = export_markdown(meta, [])
        self.assertTrue(output.startswith("# My Benchmark"))

    def test_markdown_system_info(self) -> None:
        """Output contains CPU model."""
        meta = make_meta()
        output = export_markdown(meta, [])
        self.assertIn("Test CPU", output)

    def test_markdown_single_condition_table(self) -> None:
        """Single condition has wall time table, no overhead column."""
        meta = make_meta(conditions=["baseline"])
        results = [
            make_package_result("pkg1", {"baseline": [10.0, 10.1, 10.0, 10.1, 10.0]}),
        ]
        output = export_markdown(meta, results)
        self.assertIn("| Package |", output)
        self.assertIn("| Wall (s) |", output)
        self.assertNotIn("Overhead", output)
        self.assertIn("pkg1", output)

    def test_markdown_multi_condition_table(self) -> None:
        """Two conditions produces overhead column."""
        meta = make_meta(conditions=["baseline", "treatment"])
        results = [
            make_package_result(
                "pkg1",
                {
                    "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                    "treatment": [12.0, 12.0, 12.0, 12.0, 12.0],
                },
            ),
        ]
        output = export_markdown(meta, results)
        self.assertIn("Overhead", output)
        self.assertIn("pkg1", output)

    def test_markdown_summary_section(self) -> None:
        """Multi-condition includes '## Summary' with median overhead."""
        meta = make_meta(conditions=["baseline", "treatment"])
        results = [
            make_package_result(
                "pkg1",
                {
                    "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                    "treatment": [12.0, 12.0, 12.0, 12.0, 12.0],
                },
            ),
        ]
        output = export_markdown(meta, results)
        self.assertIn("## Summary", output)
        self.assertIn("Median overhead:", output)

    def test_markdown_generated_by(self) -> None:
        """Output ends with 'Generated by labeille bench'."""
        meta = make_meta()
        output = export_markdown(meta, [])
        self.assertIn("Generated by labeille bench", output)

    def test_markdown_table_header(self) -> None:
        """Multi-condition table has proper header."""
        meta = make_meta(conditions=["baseline", "treatment"])
        results = [
            make_package_result(
                "pkg1",
                {
                    "baseline": [10.0, 10.0, 10.0, 10.0, 10.0],
                    "treatment": [12.0, 12.0, 12.0, 12.0, 12.0],
                },
            ),
        ]
        output = export_markdown(meta, results)
        self.assertIn("| Package |", output)


if __name__ == "__main__":
    unittest.main()
