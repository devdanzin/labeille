"""Tests for labeille.bench_cli — Click CLI for bench subcommands."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from click.testing import CliRunner

from labeille.bench_cli import bench


class TestBenchGroupHelp(unittest.TestCase):
    """Tests for the bench group and subcommand help."""

    def test_bench_group_help(self) -> None:
        result = CliRunner().invoke(bench, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Benchmark", result.output)
        self.assertIn("run", result.output)
        self.assertIn("show", result.output)
        self.assertIn("compare", result.output)
        self.assertIn("system", result.output)
        self.assertIn("export", result.output)

    def test_bench_run_help(self) -> None:
        result = CliRunner().invoke(bench, ["run", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--profile", result.output)
        self.assertIn("--condition", result.output)
        self.assertIn("--registry-dir", result.output)

    def test_bench_show_help(self) -> None:
        result = CliRunner().invoke(bench, ["show", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("RESULT_DIR", result.output)

    def test_bench_compare_help(self) -> None:
        result = CliRunner().invoke(bench, ["compare", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--baseline", result.output)

    def test_bench_system_help(self) -> None:
        result = CliRunner().invoke(bench, ["system", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--target-python", result.output)

    def test_bench_export_help(self) -> None:
        result = CliRunner().invoke(bench, ["export", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--format", result.output)


class TestBenchSystem(unittest.TestCase):
    """Tests for bench system command."""

    def test_bench_system_runs(self) -> None:
        result = CliRunner().invoke(bench, ["system"])
        self.assertEqual(result.exit_code, 0)
        # Should contain system profile sections.
        output = result.output.lower()
        self.assertTrue(
            "cpu" in output or "ram" in output or "system" in output,
            f"Expected system info in output, got: {result.output[:200]}",
        )

    def test_bench_system_json(self) -> None:
        result = CliRunner().invoke(bench, ["system", "--json"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("cpu_model", data)


class TestBenchShowErrors(unittest.TestCase):
    """Tests for bench show error handling."""

    def test_bench_show_missing_dir(self) -> None:
        result = CliRunner().invoke(bench, ["show", "/nonexistent/path"])
        self.assertNotEqual(result.exit_code, 0)


class TestBenchRunNoConditions(unittest.TestCase):
    """Tests for bench run with missing configuration."""

    def test_bench_run_no_conditions(self) -> None:
        """Run with no conditions or profile gives error about conditions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal registry dir so the path exists.
            reg_dir = Path(tmpdir) / "registry"
            reg_dir.mkdir()
            result = CliRunner().invoke(
                bench,
                [
                    "run",
                    "--registry-dir",
                    str(reg_dir),
                    "--target-python",
                    "/usr/bin/python3",
                ],
            )
            # Should fail because no conditions are defined.
            self.assertNotEqual(result.exit_code, 0)
            exc_str = str(result.exception) if result.exception else ""
            self.assertIn("condition", (result.output + exc_str).lower())


class TestBenchShow(unittest.TestCase):
    """Tests for bench show with real data."""

    def test_bench_show_loads_results(self) -> None:
        """bench show displays formatted benchmark results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a minimal bench_meta.json and bench_results.jsonl.
            meta_data = {
                "bench_id": "test_bench",
                "name": "Test",
                "description": "",
                "system": {},
                "python_profiles": {},
                "conditions": {"baseline": {"name": "baseline"}},
                "config": {"iterations": 3, "warmup": 0},
                "cli_args": [],
                "start_time": "2026-02-28T10:00:00",
                "end_time": "2026-02-28T10:30:00",
                "packages_total": 0,
                "packages_completed": 0,
                "packages_skipped": 0,
            }
            (Path(tmpdir) / "bench_meta.json").write_text(json.dumps(meta_data))
            (Path(tmpdir) / "bench_results.jsonl").write_text("")

            result = CliRunner().invoke(bench, ["show", tmpdir])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Test", result.output)


class TestBenchExport(unittest.TestCase):
    """Tests for bench export command."""

    def _make_bench_dir(self, tmpdir: str) -> str:
        """Create a minimal bench directory with meta and results."""
        meta_data = {
            "bench_id": "test_bench",
            "name": "Export Test",
            "description": "",
            "system": {},
            "python_profiles": {},
            "conditions": {"baseline": {"name": "baseline"}},
            "config": {"iterations": 3, "warmup": 0},
            "cli_args": [],
            "start_time": "",
            "end_time": "",
            "packages_total": 1,
            "packages_completed": 1,
            "packages_skipped": 0,
        }
        result_data = {
            "package": "testpkg",
            "conditions": {
                "baseline": {
                    "condition_name": "baseline",
                    "iterations": [
                        {
                            "index": i,
                            "warmup": False,
                            "wall_time_s": 1.5 + i * 0.01,
                            "user_time_s": 1.2,
                            "sys_time_s": 0.1,
                            "peak_rss_mb": 100.0,
                            "exit_code": 0,
                            "status": "ok",
                            "outlier": False,
                            "load_avg_start": 0.5,
                            "load_avg_end": 0.5,
                            "ram_available_start_gb": 8.0,
                        }
                        for i in range(1, 4)
                    ],
                    "install_duration_s": 0.5,
                    "venv_setup_duration_s": 0.3,
                }
            },
            "clone_duration_s": 0.5,
            "skipped": False,
            "skip_reason": "",
        }
        bench_dir = Path(tmpdir) / "bench_001"
        bench_dir.mkdir()
        (bench_dir / "bench_meta.json").write_text(json.dumps(meta_data))
        (bench_dir / "bench_results.jsonl").write_text(json.dumps(result_data) + "\n")
        return str(bench_dir)

    def test_export_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = self._make_bench_dir(tmpdir)
            result = CliRunner().invoke(bench, ["export", bench_dir, "--format", "csv"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("testpkg", result.output)
            self.assertIn("wall_time_s", result.output)

    def test_export_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = self._make_bench_dir(tmpdir)
            result = CliRunner().invoke(bench, ["export", bench_dir, "--format", "markdown"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("testpkg", result.output)
            self.assertIn("|", result.output)

    def test_export_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = self._make_bench_dir(tmpdir)
            output_file = Path(tmpdir) / "output.csv"
            result = CliRunner().invoke(
                bench,
                ["export", bench_dir, "--format", "csv", "-o", str(output_file)],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(output_file.exists())
            self.assertIn("testpkg", output_file.read_text())


if __name__ == "__main__":
    unittest.main()
