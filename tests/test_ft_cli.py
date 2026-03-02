"""Tests for labeille.ft_cli — free-threading CLI commands."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from labeille.cli import main


def _write_mock_run(tmpdir: Path) -> None:
    """Write minimal ft_meta.json and ft_results.jsonl for testing."""
    meta = {
        "run_id": "test-run-001",
        "timestamp": "2026-01-15T12:00:00",
        "python_profile": {
            "version": "3.14.0b2",
            "jit_enabled": True,
            "gil_disabled": True,
        },
        "system_profile": {
            "cpu_model": "AMD Ryzen 9 7950X",
            "ram_total_gb": 64,
            "os_distro": "Ubuntu 24.04",
        },
    }
    (tmpdir / "ft_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    results = [
        {
            "package": "requests",
            "category": "compatible",
            "iterations_completed": 10,
            "pass_count": 10,
            "crash_count": 0,
            "pass_rate": 1.0,
        },
        {
            "package": "numpy",
            "category": "crash",
            "iterations_completed": 10,
            "pass_count": 7,
            "crash_count": 3,
            "pass_rate": 0.7,
            "failure_signatures": ["SIGSEGV in _multiarray"],
            "extension_compat": {
                "package": "numpy",
                "is_pure_python": False,
                "extensions": [
                    {
                        "module_name": "_multiarray_umath",
                        "is_extension": True,
                        "triggered_gil_fallback": True,
                    }
                ],
                "gil_fallback_active": True,
            },
        },
        {
            "package": "aiohttp",
            "category": "intermittent",
            "iterations_completed": 10,
            "pass_count": 7,
            "crash_count": 0,
            "pass_rate": 0.7,
            "flaky_tests": {"test_connector::test_close": 3, "test_client::test_timeout": 1},
        },
    ]
    with (tmpdir / "ft_results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


class TestFtGroupHelp(unittest.TestCase):
    def test_ft_group_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["ft", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Free-threading", result.output)
        self.assertIn("run", result.output)
        self.assertIn("show", result.output)


class TestFtRunHelp(unittest.TestCase):
    def test_ft_run_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["ft", "run", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--target-python", result.output)
        self.assertIn("--iterations", result.output)
        self.assertIn("--compare-with-gil", result.output)


class TestFtShowHelp(unittest.TestCase):
    def test_ft_show_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["ft", "show", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("RESULT_DIR", result.output)


class TestFtRunMissingPython(unittest.TestCase):
    def test_ft_run_missing_python(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["ft", "run"])
        self.assertNotEqual(result.exit_code, 0)


class TestFtShowNonexistentDir(unittest.TestCase):
    def test_ft_show_nonexistent_dir(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["ft", "show", "/nonexistent/path"])
        self.assertNotEqual(result.exit_code, 0)


class TestFtShowWithMockData(unittest.TestCase):
    def test_ft_show_with_mock_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_mock_run(Path(tmpdir))
            runner = CliRunner()
            result = runner.invoke(main, ["ft", "show", tmpdir])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Compatibility Summary", result.output)
            self.assertIn("requests", result.output)
            self.assertIn("numpy", result.output)


class TestFtFlakyWithMockData(unittest.TestCase):
    def test_ft_flaky_with_mock_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_mock_run(Path(tmpdir))
            runner = CliRunner()
            result = runner.invoke(main, ["ft", "flaky", tmpdir])
            self.assertEqual(result.exit_code, 0, result.output)
            # aiohttp is intermittent with pass_count>0 and numpy is crash with pass_count>0
            self.assertIn("Flakiness Profile", result.output)

    def test_ft_flaky_specific_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_mock_run(Path(tmpdir))
            runner = CliRunner()
            result = runner.invoke(main, ["ft", "flaky", "--package", "numpy", tmpdir])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("numpy", result.output)


class TestFtCompatWithMockData(unittest.TestCase):
    def test_ft_compat_with_mock_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_mock_run(Path(tmpdir))
            runner = CliRunner()
            result = runner.invoke(main, ["ft", "compat", tmpdir])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("numpy", result.output)


if __name__ == "__main__":
    unittest.main()
