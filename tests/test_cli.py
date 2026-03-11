"""Tests for labeille.cli — main CLI entry point commands."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from labeille.cli import main


class TestResolveCommand(unittest.TestCase):
    """Tests for the resolve command."""

    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_no_packages_shows_usage_error(self) -> None:
        result = self.runner.invoke(main, ["resolve"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("At least one of PACKAGES", result.output)

    def test_top_without_from_json_shows_usage_error(self) -> None:
        result = self.runner.invoke(main, ["resolve", "--top", "10", "pkg1"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--top requires --from-json", result.output)

    def test_workers_less_than_one_shows_error(self) -> None:
        result = self.runner.invoke(main, ["resolve", "--workers", "0", "pkg1"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--workers must be at least 1", result.output)


class TestRunCommand(unittest.TestCase):
    """Tests for the run command."""

    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_missing_target_python_shows_error(self) -> None:
        result = self.runner.invoke(main, ["run"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--target-python", result.output)

    @patch("labeille.runner.validate_target_python", return_value="3.15.0a5")
    def test_workers_less_than_one_shows_error(self, mock_validate: MagicMock) -> None:
        result = self.runner.invoke(
            main,
            ["run", "--target-python", __file__, "--workers", "0"],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--workers must be at least 1", result.output)

    @patch("labeille.runner.run_all")
    @patch("labeille.summary.format_summary", return_value="summary")
    @patch("labeille.runner.validate_target_python", return_value="3.15.0a5")
    def test_run_basic_invocation(
        self,
        mock_validate: MagicMock,
        mock_format: MagicMock,
        mock_run_all: MagicMock,
    ) -> None:
        mock_output = MagicMock()
        mock_output.results = []
        mock_output.summary = MagicMock()
        mock_output.python_version = "3.15.0a5"
        mock_output.jit_enabled = True
        mock_output.total_duration = 1.0
        mock_output.run_dir = Path("/tmp/results")
        mock_run_all.return_value = mock_output

        result = self.runner.invoke(
            main,
            ["run", "--target-python", __file__, "--dry-run"],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Target Python: 3.15.0a5", result.output)
        self.assertIn("dry-run", result.output)

    @patch("labeille.runner.validate_target_python", side_effect=RuntimeError("bad python"))
    def test_run_invalid_target_python_shows_error(self, mock_validate: MagicMock) -> None:
        result = self.runner.invoke(
            main,
            ["run", "--target-python", __file__],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("bad python", result.output)

    @patch("labeille.runner.validate_target_python", return_value="3.15.0a5")
    @patch("labeille.runner.parse_repo_overrides", side_effect=ValueError("bad format"))
    def test_run_invalid_repo_override_shows_error(
        self, mock_parse: MagicMock, mock_validate: MagicMock
    ) -> None:
        result = self.runner.invoke(
            main,
            ["run", "--target-python", __file__, "--repo-override", "bad"],
        )
        self.assertNotEqual(result.exit_code, 0)

    @patch("labeille.runner.run_all")
    @patch("labeille.summary.format_summary", return_value="")
    @patch("labeille.runner.validate_target_python", return_value="3.15.0a5")
    def test_run_no_shallow_sets_clone_depth_zero(
        self,
        mock_validate: MagicMock,
        mock_format: MagicMock,
        mock_run_all: MagicMock,
    ) -> None:
        mock_output = MagicMock()
        mock_output.results = []
        mock_output.summary = MagicMock()
        mock_output.python_version = "3.15"
        mock_output.jit_enabled = True
        mock_output.total_duration = 0.0
        mock_output.run_dir = Path("/tmp/r")
        mock_run_all.return_value = mock_output

        self.runner.invoke(
            main,
            ["run", "--target-python", __file__, "--no-shallow", "--dry-run"],
        )
        config = mock_run_all.call_args[0][0]
        self.assertEqual(config.clone_depth_override, 0)

    @patch("labeille.runner.run_all")
    @patch("labeille.summary.format_summary", return_value="")
    @patch("labeille.runner.validate_target_python", return_value="3.15.0a5")
    def test_run_work_dir_sets_repos_and_venvs(
        self,
        mock_validate: MagicMock,
        mock_format: MagicMock,
        mock_run_all: MagicMock,
    ) -> None:
        mock_output = MagicMock()
        mock_output.results = []
        mock_output.summary = MagicMock()
        mock_output.python_version = "3.15"
        mock_output.jit_enabled = True
        mock_output.total_duration = 0.0
        mock_output.run_dir = Path("/tmp/r")
        mock_run_all.return_value = mock_output

        self.runner.invoke(
            main,
            [
                "run",
                "--target-python",
                __file__,
                "--work-dir",
                "/tmp/work",
                "--dry-run",
            ],
        )
        config = mock_run_all.call_args[0][0]
        self.assertEqual(config.repos_dir, Path("/tmp/work/repos"))
        self.assertEqual(config.venvs_dir, Path("/tmp/work/venvs"))

    @patch("labeille.runner.run_all")
    @patch("labeille.summary.format_summary", return_value="")
    @patch("labeille.runner.validate_target_python", return_value="3.15.0a5")
    def test_run_env_pairs_passed_to_config(
        self,
        mock_validate: MagicMock,
        mock_format: MagicMock,
        mock_run_all: MagicMock,
    ) -> None:
        mock_output = MagicMock()
        mock_output.results = []
        mock_output.summary = MagicMock()
        mock_output.python_version = "3.15"
        mock_output.jit_enabled = True
        mock_output.total_duration = 0.0
        mock_output.run_dir = Path("/tmp/r")
        mock_run_all.return_value = mock_output

        self.runner.invoke(
            main,
            [
                "run",
                "--target-python",
                __file__,
                "--env",
                "FOO=bar",
                "--env",
                "BAZ=qux",
                "--dry-run",
            ],
        )
        config = mock_run_all.call_args[0][0]
        self.assertEqual(config.env_overrides, {"FOO": "bar", "BAZ": "qux"})


class TestBisectCommand(unittest.TestCase):
    """Tests for the bisect command."""

    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_missing_required_options(self) -> None:
        result = self.runner.invoke(main, ["bisect", "mypkg"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("labeille.bisect.run_bisect")
    def test_bisect_success(self, mock_bisect: MagicMock) -> None:
        from labeille.bisect import BisectResult, BisectStep

        mock_bisect.return_value = BisectResult(
            package="mypkg",
            first_bad_commit="abc1234567890",
            first_bad_commit_short="abc1234",
            good_rev="v1.0",
            bad_rev="v2.0",
            steps=[
                BisectStep(
                    commit="abc1234567890",
                    commit_short="abc1234",
                    status="bad",
                    detail="crash detected",
                    duration_seconds=5.0,
                ),
            ],
            total_commits=10,
            commits_tested=1,
        )

        result = self.runner.invoke(
            main,
            [
                "bisect",
                "mypkg",
                "--good",
                "v1.0",
                "--bad",
                "v2.0",
                "--target-python",
                __file__,
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("abc1234", result.output)
        self.assertIn("First bad commit", result.output)

    @patch("labeille.bisect.run_bisect", side_effect=ValueError("No repo URL"))
    def test_bisect_no_repo_shows_error(self, mock_bisect: MagicMock) -> None:
        result = self.runner.invoke(
            main,
            [
                "bisect",
                "mypkg",
                "--good",
                "v1.0",
                "--bad",
                "v2.0",
                "--target-python",
                __file__,
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No repo URL", result.output)

    @patch("labeille.bisect.run_bisect")
    def test_bisect_not_found_shows_message(self, mock_bisect: MagicMock) -> None:
        from labeille.bisect import BisectResult

        mock_bisect.return_value = BisectResult(
            package="mypkg",
            first_bad_commit=None,
            first_bad_commit_short=None,
            good_rev="v1.0",
            bad_rev="v2.0",
            steps=[],
            total_commits=10,
            commits_tested=2,
        )

        result = self.runner.invoke(
            main,
            [
                "bisect",
                "mypkg",
                "--good",
                "v1.0",
                "--bad",
                "v2.0",
                "--target-python",
                __file__,
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Could not identify", result.output)


class TestScanDepsCommand(unittest.TestCase):
    """Tests for the scan-deps command."""

    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_nonexistent_path_shows_error(self) -> None:
        result = self.runner.invoke(main, ["scan-deps", "/nonexistent/path"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("labeille.scan_deps.scan_package_deps")
    def test_json_output_format(self, mock_scan: MagicMock) -> None:
        from labeille.scan_deps import ScanResult

        mock_scan.return_value = ScanResult(
            package_name="testpkg",
            scan_dirs=["tests/"],
            total_files_scanned=5,
            total_imports_found=10,
            resolved=[],
            unresolved=[],
            already_installed=[],
            missing=[],
            suggested_install="",
        )

        result = self.runner.invoke(
            main,
            ["scan-deps", ".", "--format", "json"],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn('"package_name"', result.output)
        self.assertIn("testpkg", result.output)

    @patch("labeille.scan_deps.scan_package_deps")
    def test_human_output_format(self, mock_scan: MagicMock) -> None:
        from labeille.scan_deps import ScanResult

        mock_scan.return_value = ScanResult(
            package_name="testpkg",
            scan_dirs=["tests/"],
            total_files_scanned=5,
            total_imports_found=10,
            resolved=[],
            unresolved=[],
            already_installed=[],
            missing=["pytest-cov"],
            suggested_install="pip install pytest-cov",
        )

        result = self.runner.invoke(
            main,
            ["scan-deps", ".", "--format", "human"],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Scanning: testpkg", result.output)
        self.assertIn("Suggested install command", result.output)

    @patch("labeille.scan_deps.scan_package_deps")
    def test_pip_format_shows_missing_and_unresolved(self, mock_scan: MagicMock) -> None:
        from labeille.scan_deps import ResolvedDep, ScanResult

        mock_scan.return_value = ScanResult(
            package_name="testpkg",
            scan_dirs=["tests/"],
            total_files_scanned=5,
            total_imports_found=3,
            resolved=[],
            unresolved=[
                ResolvedDep(
                    import_name="unknownlib",
                    pip_package="",
                    source="unresolved",
                    import_files=["test_foo.py"],
                    is_conditional=False,
                ),
            ],
            already_installed=[],
            missing=["pytest", "coverage"],
            suggested_install="pip install coverage pytest",
        )

        result = self.runner.invoke(main, ["scan-deps", ".", "--format", "pip"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("pip install", result.output)
        self.assertIn("coverage", result.output)
        self.assertIn("pytest", result.output)
        self.assertIn("Unresolved", result.output)
        self.assertIn("unknownlib", result.output)

    @patch("labeille.scan_deps.scan_package_deps")
    def test_human_format_shows_resolved_deps(self, mock_scan: MagicMock) -> None:
        from labeille.scan_deps import ResolvedDep, ScanResult

        mock_scan.return_value = ScanResult(
            package_name="mypkg",
            scan_dirs=["tests/"],
            total_files_scanned=10,
            total_imports_found=20,
            resolved=[
                ResolvedDep(
                    import_name="yaml",
                    pip_package="PyYAML",
                    source="mapping",
                    import_files=["test_a.py", "test_b.py"],
                    is_conditional=False,
                ),
            ],
            unresolved=[],
            already_installed=[],
            missing=[],
            suggested_install="",
        )

        result = self.runner.invoke(main, ["scan-deps", "."])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Scanning: mypkg", result.output)
        self.assertIn("Files scanned: 10", result.output)
        self.assertIn("PyYAML", result.output)

    @patch("labeille.scan_deps.scan_package_deps")
    def test_human_format_with_install_command(self, mock_scan: MagicMock) -> None:
        from labeille.scan_deps import ScanResult

        mock_scan.return_value = ScanResult(
            package_name="mypkg",
            scan_dirs=["tests/"],
            total_files_scanned=5,
            total_imports_found=3,
            resolved=[],
            unresolved=[],
            already_installed=["pytest"],
            missing=[],
            suggested_install="",
        )

        result = self.runner.invoke(
            main,
            ["scan-deps", ".", "--install-command", "pip install -e '.[test]'"],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Comparison with install_command", result.output)
        self.assertIn("Already installed: pytest", result.output)


if __name__ == "__main__":
    unittest.main()
