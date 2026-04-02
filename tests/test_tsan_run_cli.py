"""Tests for labeille.tsan_run_cli — TSan-run CLI."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from labeille.tsan_run_cli import tsan_run


class TestTsanRunCli(unittest.TestCase):
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(tsan_run, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("tsan-run", result.output.lower().replace("_", "-"))
        self.assertIn("--target-python", result.output)
        self.assertIn("--suppressions", result.output)
        self.assertIn("--quick", result.output)
        self.assertIn("--stress", result.output)
        self.assertIn("--test-script", result.output)

    def test_no_args(self) -> None:
        runner = CliRunner()
        result = runner.invoke(tsan_run, [])
        self.assertNotEqual(result.exit_code, 0)

    def test_missing_packages_and_registry(self) -> None:
        runner = CliRunner()
        result = runner.invoke(tsan_run, ["--target-python", "/usr/bin/python3"])
        # Should fail because neither --packages nor --registry-dir is given.
        self.assertNotEqual(result.exit_code, 0)

    @patch("labeille.tsan_run.run_tsan_batch")
    def test_invokes_run_tsan_batch(self, mock_run: MagicMock) -> None:
        mock_meta = MagicMock()
        mock_meta.run_id = "tsan_test"
        mock_meta.python_version = "3.14.0+"
        mock_meta.is_free_threaded = True
        mock_meta.is_tsan = True
        mock_meta.total_packages = 1
        mock_meta.packages_with_races = 0
        mock_meta.total_races = 0
        mock_meta.quick_mode = False
        mock_meta.stress_count = 1
        mock_run.return_value = (mock_meta, [])

        runner = CliRunner()
        result = runner.invoke(
            tsan_run,
            ["--target-python", "/usr/bin/python3", "--packages", "testpkg"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
