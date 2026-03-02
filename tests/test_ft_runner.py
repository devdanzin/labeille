"""Tests for labeille.ft.runner — free-threading test execution engine."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from labeille.ft.results import FailureCategory, IterationOutcome
from labeille.ft.runner import (
    FTRunConfig,
    OutputMonitor,
    extract_tsan_warnings,
    parse_pytest_summary,
    parse_pytest_verbose,
    run_package_ft,
    run_single_iteration,
)


# ---------------------------------------------------------------------------
# TSAN extraction tests
# ---------------------------------------------------------------------------


class TestExtractTsanWarnings(unittest.TestCase):
    def test_extract_tsan_data_race(self) -> None:
        stderr = "WARNING: ThreadSanitizer: data race (pid=12345)\n"
        result = extract_tsan_warnings(stderr)
        self.assertEqual(result, ["data race"])

    def test_extract_tsan_multiple_types(self) -> None:
        stderr = (
            "WARNING: ThreadSanitizer: data race (pid=1)\n"
            "WARNING: ThreadSanitizer: thread leak (pid=2)\n"
            "WARNING: ThreadSanitizer: lock-order-inversion (pid=3)\n"
        )
        result = extract_tsan_warnings(stderr)
        self.assertEqual(result, ["data race", "lock-order-inversion", "thread leak"])

    def test_extract_tsan_deduplicates(self) -> None:
        stderr = "".join(f"WARNING: ThreadSanitizer: data race (pid={i})\n" for i in range(10))
        result = extract_tsan_warnings(stderr)
        self.assertEqual(result, ["data race"])

    def test_extract_tsan_no_warnings(self) -> None:
        stderr = "normal output\nno tsan here\n"
        result = extract_tsan_warnings(stderr)
        self.assertEqual(result, [])

    def test_extract_tsan_summary_line(self) -> None:
        stderr = "SUMMARY: ThreadSanitizer: data race in foo\n"
        result = extract_tsan_warnings(stderr)
        self.assertEqual(result, ["data race"])


# ---------------------------------------------------------------------------
# Pytest parsing tests
# ---------------------------------------------------------------------------


class TestParsePytestSummary(unittest.TestCase):
    def test_parse_summary_all_passed(self) -> None:
        output = "================ 5 passed in 1.23s ================\n"
        result = parse_pytest_summary(output)
        self.assertEqual(result, {"passed": 5})

    def test_parse_summary_mixed(self) -> None:
        output = "======= 3 passed, 2 failed, 1 error in 5.67s =======\n"
        result = parse_pytest_summary(output)
        self.assertEqual(result, {"passed": 3, "failed": 2, "error": 1})

    def test_parse_summary_with_skipped(self) -> None:
        output = "===== 10 passed, 3 skipped in 2.00s =====\n"
        result = parse_pytest_summary(output)
        self.assertEqual(result, {"passed": 10, "skipped": 3})

    def test_parse_summary_no_match(self) -> None:
        output = "no summary line here\n"
        result = parse_pytest_summary(output)
        self.assertEqual(result, {})

    def test_parse_summary_with_warnings(self) -> None:
        output = "====== 5 passed, 2 warnings in 1.00s ======\n"
        result = parse_pytest_summary(output)
        self.assertEqual(result, {"passed": 5})


class TestParsePytestVerbose(unittest.TestCase):
    def test_parse_verbose_results(self) -> None:
        output = (
            "tests/test_foo.py::test_bar PASSED\n"
            "tests/test_foo.py::test_baz FAILED\n"
            "tests/test_foo.py::test_qux ERROR\n"
        )
        result = parse_pytest_verbose(output)
        self.assertEqual(result["tests/test_foo.py::test_bar"], "PASSED")
        self.assertEqual(result["tests/test_foo.py::test_baz"], "FAILED")
        self.assertEqual(result["tests/test_foo.py::test_qux"], "ERROR")

    def test_parse_verbose_empty(self) -> None:
        output = "no test results here\n"
        result = parse_pytest_verbose(output)
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# OutputMonitor tests
# ---------------------------------------------------------------------------


class TestOutputMonitor(unittest.TestCase):
    def test_monitor_not_stalled(self) -> None:
        monitor = OutputMonitor(stall_threshold=5)
        monitor.feed("line 1\n")
        self.assertFalse(monitor.stalled)

    def test_monitor_stalled(self) -> None:
        monitor = OutputMonitor(stall_threshold=1)
        monitor.feed("line 1\n")
        time.sleep(1.5)
        self.assertTrue(monitor.stalled)

    def test_monitor_stderr_tail(self) -> None:
        monitor = OutputMonitor(tail_bytes=20)
        monitor.feed("a" * 30 + "\n")
        tail = monitor.stderr_tail
        self.assertLessEqual(len(tail), 20)

    def test_monitor_full_stderr(self) -> None:
        monitor = OutputMonitor()
        monitor.feed("line 1\n")
        monitor.feed("line 2\n")
        monitor.feed("line 3\n")
        self.assertEqual(monitor.full_stderr, "line 1\nline 2\nline 3\n")

    def test_monitor_last_line(self) -> None:
        monitor = OutputMonitor()
        monitor.feed("line 1\n")
        monitor.feed("line 2\n")
        monitor.feed("line 3\n")
        self.assertEqual(monitor.last_line, "line 3")

    def test_monitor_thread_safety(self) -> None:
        monitor = OutputMonitor()
        errors: list[Exception] = []

        def feeder(prefix: str) -> None:
            try:
                for i in range(50):
                    monitor.feed(f"{prefix}-{i}\n")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=feeder, args=("A",))
        t2 = threading.Thread(target=feeder, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [])
        # All 100 lines should be captured.
        full = monitor.full_stderr
        self.assertEqual(full.count("\n"), 100)

    def test_monitor_seconds_since_last_output(self) -> None:
        monitor = OutputMonitor()
        monitor.feed("line\n")
        time.sleep(0.1)
        self.assertGreater(monitor.seconds_since_last_output, 0.05)


# ---------------------------------------------------------------------------
# run_single_iteration tests (mocked)
# ---------------------------------------------------------------------------


class TestRunSingleIteration(unittest.TestCase):
    def _make_mock_proc(
        self,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> MagicMock:
        """Create a mock Popen object."""
        proc = MagicMock()
        proc.pid = 12345
        proc.returncode = returncode

        # Make poll() return None once, then the returncode.
        proc.poll.side_effect = [None, returncode]

        # stdout and stderr as line iterators.
        proc.stdout = iter(stdout.splitlines(True))
        proc.stderr = iter(stderr.splitlines(True))
        return proc

    @patch("labeille.ft.runner.os.setpgrp")
    @patch("labeille.ft.runner.subprocess.Popen")
    @patch("labeille.ft.runner.time.sleep")
    def test_iteration_pass(
        self, mock_sleep: MagicMock, mock_popen: MagicMock, mock_setpgrp: MagicMock
    ) -> None:
        proc = self._make_mock_proc(
            returncode=0,
            stdout="tests/test_foo.py::test_bar PASSED\n======= 1 passed in 0.5s =======\n",
        )
        mock_popen.return_value = proc

        result = run_single_iteration(
            venv_python=Path("/fake/bin/python"),
            test_command="python -m pytest tests/",
            cwd=Path("/fake/repo"),
            env={"PYTHON_GIL": "0"},
            timeout=60,
            stall_threshold=30,
            iteration_index=1,
        )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.exit_code, 0)

    @patch("labeille.ft.runner.os.setpgrp")
    @patch("labeille.ft.runner.subprocess.Popen")
    @patch("labeille.ft.runner.time.sleep")
    def test_iteration_fail(
        self, mock_sleep: MagicMock, mock_popen: MagicMock, mock_setpgrp: MagicMock
    ) -> None:
        proc = self._make_mock_proc(returncode=1, stdout="1 failed\n")
        mock_popen.return_value = proc

        with patch("labeille.ft.runner.detect_crash", return_value=None):
            result = run_single_iteration(
                venv_python=Path("/fake/bin/python"),
                test_command="python -m pytest tests/",
                cwd=Path("/fake/repo"),
                env={},
                timeout=60,
                stall_threshold=30,
                iteration_index=1,
            )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.exit_code, 1)

    @patch("labeille.ft.runner.os.setpgrp")
    @patch("labeille.ft.runner.subprocess.Popen")
    @patch("labeille.ft.runner.time.sleep")
    def test_iteration_crash(
        self, mock_sleep: MagicMock, mock_popen: MagicMock, mock_setpgrp: MagicMock
    ) -> None:
        proc = self._make_mock_proc(returncode=-11, stderr="Segmentation fault\n")
        mock_popen.return_value = proc

        result = run_single_iteration(
            venv_python=Path("/fake/bin/python"),
            test_command="python -m pytest tests/",
            cwd=Path("/fake/repo"),
            env={},
            timeout=60,
            stall_threshold=30,
            iteration_index=1,
        )

        self.assertEqual(result.status, "crash")
        self.assertEqual(result.crash_signal, "SIGSEGV")

    @patch("labeille.ft.runner.os.setpgrp")
    @patch("labeille.ft.runner.subprocess.Popen")
    def test_iteration_tsan_warnings(self, mock_popen: MagicMock, mock_setpgrp: MagicMock) -> None:
        proc = self._make_mock_proc(
            returncode=0,
            stderr="WARNING: ThreadSanitizer: data race (pid=123)\n",
        )
        mock_popen.return_value = proc

        with patch("labeille.ft.runner.time.sleep"):
            result = run_single_iteration(
                venv_python=Path("/fake/bin/python"),
                test_command="python -m pytest tests/",
                cwd=Path("/fake/repo"),
                env={},
                timeout=60,
                stall_threshold=30,
                iteration_index=1,
                tsan_build=True,
            )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.tsan_warnings, ["data race"])

    @patch("labeille.ft.runner.os.setpgrp")
    @patch("labeille.ft.runner.subprocess.Popen")
    def test_iteration_tsan_not_parsed_when_disabled(
        self, mock_popen: MagicMock, mock_setpgrp: MagicMock
    ) -> None:
        proc = self._make_mock_proc(
            returncode=0,
            stderr="WARNING: ThreadSanitizer: data race (pid=123)\n",
        )
        mock_popen.return_value = proc

        with patch("labeille.ft.runner.time.sleep"):
            result = run_single_iteration(
                venv_python=Path("/fake/bin/python"),
                test_command="python -m pytest tests/",
                cwd=Path("/fake/repo"),
                env={},
                timeout=60,
                stall_threshold=30,
                iteration_index=1,
                tsan_build=False,
            )

        self.assertEqual(result.tsan_warnings, [])

    @patch("labeille.ft.runner.os.setpgrp")
    @patch("labeille.ft.runner.subprocess.Popen")
    def test_iteration_pytest_results_parsed(
        self, mock_popen: MagicMock, mock_setpgrp: MagicMock
    ) -> None:
        proc = self._make_mock_proc(
            returncode=0,
            stdout=(
                "tests/test_foo.py::test_a PASSED\n"
                "tests/test_foo.py::test_b FAILED\n"
                "======= 1 passed, 1 failed in 1.0s =======\n"
            ),
        )
        mock_popen.return_value = proc

        with patch("labeille.ft.runner.time.sleep"):
            result = run_single_iteration(
                venv_python=Path("/fake/bin/python"),
                test_command="python -m pytest -v tests/",
                cwd=Path("/fake/repo"),
                env={},
                timeout=60,
                stall_threshold=30,
                iteration_index=1,
            )

        self.assertIn("tests/test_foo.py::test_a", result.test_results)
        self.assertEqual(result.tests_passed, 1)
        self.assertEqual(result.tests_failed, 1)

    @patch("labeille.ft.runner.subprocess.Popen")
    def test_iteration_process_start_failure(self, mock_popen: MagicMock) -> None:
        mock_popen.side_effect = OSError("No such file")
        result = run_single_iteration(
            venv_python=Path("/nonexistent/python"),
            test_command="python -m pytest",
            cwd=Path("/fake/repo"),
            env={},
            timeout=60,
            stall_threshold=30,
            iteration_index=1,
        )
        self.assertEqual(result.status, "fail")
        self.assertIn("Failed to start", result.stderr_tail)


# ---------------------------------------------------------------------------
# run_package_ft tests (heavily mocked)
# ---------------------------------------------------------------------------


class TestRunPackageFt(unittest.TestCase):
    def _make_pkg(self, **overrides: object) -> SimpleNamespace:
        """Create a mock PackageEntry."""
        defaults = {
            "package": "mypkg",
            "repo": "https://github.com/x/mypkg",
            "install_command": "pip install -e .",
            "test_command": "python -m pytest tests/",
            "import_name": "mypkg",
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _make_config(self, **overrides: object) -> FTRunConfig:
        """Create a test config with real tmp paths."""
        import tempfile

        tmpdir = Path(tempfile.mkdtemp())
        defaults: dict[str, object] = {
            "target_python": Path("/fake/python"),
            "iterations": 3,
            "timeout": 60,
            "stall_threshold": 30,
            "repos_dir": tmpdir / "repos",
            "venvs_dir": tmpdir / "venvs",
            "results_dir": tmpdir / "results",
            "detect_extensions": False,
            "compare_with_gil": False,
            "stop_on_first_pass": False,
        }
        defaults.update(overrides)
        return FTRunConfig(**defaults)  # type: ignore[arg-type]

    @patch("labeille.ft.runner.clone_repo")
    def test_run_package_clone_failure(self, mock_clone: MagicMock) -> None:
        mock_clone.side_effect = Exception("clone error")
        pkg = self._make_pkg()
        config = self._make_config()

        result = run_package_ft(pkg, config)
        self.assertFalse(result.install_ok)
        self.assertEqual(result.category, FailureCategory.INSTALL_FAILURE)

    @patch("labeille.ft.runner.install_package")
    @patch("labeille.ft.runner.create_venv")
    @patch("labeille.ft.runner.clone_repo")
    def test_run_package_install_failure(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
    ) -> None:
        config = self._make_config()
        # Make repo_dir exist so we hit install path.
        repo_dir = config.repos_dir / "mypkg"
        repo_dir.mkdir(parents=True)

        mock_install.return_value = MagicMock(returncode=1, stderr="build failed")

        with patch("labeille.ft.runner.pull_repo"):
            result = run_package_ft(self._make_pkg(), config)

        self.assertFalse(result.install_ok)
        self.assertEqual(result.category, FailureCategory.INSTALL_FAILURE)

    @patch("labeille.ft.runner.run_single_iteration")
    @patch("labeille.ft.runner.install_package")
    @patch("labeille.ft.runner.create_venv")
    @patch("labeille.ft.runner.clone_repo")
    def test_run_package_all_pass(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_run_iter: MagicMock,
    ) -> None:
        config = self._make_config()
        repo_dir = config.repos_dir / "mypkg"
        repo_dir.mkdir(parents=True)

        mock_install.return_value = MagicMock(returncode=0)
        mock_run_iter.return_value = IterationOutcome(
            index=1, status="pass", exit_code=0, duration_s=5.0
        )

        with patch("labeille.ft.runner.pull_repo"):
            result = run_package_ft(self._make_pkg(), config)

        self.assertEqual(result.category, FailureCategory.COMPATIBLE)
        self.assertEqual(result.pass_rate, 1.0)
        self.assertEqual(len(result.iterations), 3)

    @patch("labeille.ft.runner.run_single_iteration")
    @patch("labeille.ft.runner.install_package")
    @patch("labeille.ft.runner.create_venv")
    @patch("labeille.ft.runner.clone_repo")
    def test_run_package_some_crashes(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_run_iter: MagicMock,
    ) -> None:
        config = self._make_config(iterations=5)
        repo_dir = config.repos_dir / "mypkg"
        repo_dir.mkdir(parents=True)

        mock_install.return_value = MagicMock(returncode=0)

        outcomes = [
            IterationOutcome(index=1, status="pass", exit_code=0, duration_s=5.0),
            IterationOutcome(index=2, status="pass", exit_code=0, duration_s=5.0),
            IterationOutcome(index=3, status="pass", exit_code=0, duration_s=5.0),
            IterationOutcome(index=4, status="crash", exit_code=-11, duration_s=2.0),
            IterationOutcome(index=5, status="crash", exit_code=-11, duration_s=2.0),
        ]
        mock_run_iter.side_effect = outcomes

        with patch("labeille.ft.runner.pull_repo"):
            result = run_package_ft(self._make_pkg(), config)

        self.assertEqual(result.category, FailureCategory.CRASH)

    @patch("labeille.ft.runner.run_single_iteration")
    @patch("labeille.ft.runner.install_package")
    @patch("labeille.ft.runner.create_venv")
    @patch("labeille.ft.runner.clone_repo")
    def test_run_package_stop_on_first_pass(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_run_iter: MagicMock,
    ) -> None:
        config = self._make_config(iterations=5, stop_on_first_pass=True)
        repo_dir = config.repos_dir / "mypkg"
        repo_dir.mkdir(parents=True)

        mock_install.return_value = MagicMock(returncode=0)
        mock_run_iter.return_value = IterationOutcome(
            index=1, status="pass", exit_code=0, duration_s=5.0
        )

        with patch("labeille.ft.runner.pull_repo"):
            result = run_package_ft(self._make_pkg(), config)

        self.assertEqual(len(result.iterations), 1)

    @patch("labeille.ft.runner.run_single_iteration")
    @patch("labeille.ft.runner.install_package")
    @patch("labeille.ft.runner.create_venv")
    @patch("labeille.ft.runner.clone_repo")
    def test_run_package_stop_on_first_pass_no_pass(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_run_iter: MagicMock,
    ) -> None:
        config = self._make_config(iterations=3, stop_on_first_pass=True)
        repo_dir = config.repos_dir / "mypkg"
        repo_dir.mkdir(parents=True)

        mock_install.return_value = MagicMock(returncode=0)
        mock_run_iter.return_value = IterationOutcome(
            index=1, status="fail", exit_code=1, duration_s=5.0
        )

        with patch("labeille.ft.runner.pull_repo"):
            result = run_package_ft(self._make_pkg(), config)

        self.assertEqual(len(result.iterations), 3)

    @patch("labeille.ft.runner.run_single_iteration")
    @patch("labeille.ft.runner.install_package")
    @patch("labeille.ft.runner.create_venv")
    @patch("labeille.ft.runner.clone_repo")
    def test_run_package_compare_with_gil(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_run_iter: MagicMock,
    ) -> None:
        config = self._make_config(iterations=3, compare_with_gil=True)
        repo_dir = config.repos_dir / "mypkg"
        repo_dir.mkdir(parents=True)

        mock_install.return_value = MagicMock(returncode=0)
        mock_run_iter.return_value = IterationOutcome(
            index=1, status="pass", exit_code=0, duration_s=5.0
        )

        with patch("labeille.ft.runner.pull_repo"):
            result = run_package_ft(self._make_pkg(), config)

        self.assertIsNotNone(result.gil_enabled_iterations)
        assert result.gil_enabled_iterations is not None
        self.assertEqual(len(result.gil_enabled_iterations), 3)

    @patch("labeille.ft.runner.assess_extension_compat")
    @patch("labeille.ft.runner.run_single_iteration")
    @patch("labeille.ft.runner.install_package")
    @patch("labeille.ft.runner.create_venv")
    @patch("labeille.ft.runner.clone_repo")
    def test_run_package_extension_compat_checked(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_run_iter: MagicMock,
        mock_compat: MagicMock,
    ) -> None:
        config = self._make_config(iterations=1, detect_extensions=True)
        repo_dir = config.repos_dir / "mypkg"
        repo_dir.mkdir(parents=True)

        mock_install.return_value = MagicMock(returncode=0)

        from labeille.ft.compat import ExtensionCompat

        mock_compat.return_value = ExtensionCompat(
            package="mypkg", import_ok=True, is_pure_python=True
        )
        mock_run_iter.return_value = IterationOutcome(
            index=1, status="pass", exit_code=0, duration_s=5.0
        )

        with patch("labeille.ft.runner.pull_repo"):
            result = run_package_ft(self._make_pkg(), config)

        mock_compat.assert_called_once()
        self.assertIsNotNone(result.extension_compat)

    @patch("labeille.ft.runner.run_single_iteration")
    @patch("labeille.ft.runner.install_package")
    @patch("labeille.ft.runner.create_venv")
    @patch("labeille.ft.runner.clone_repo")
    def test_run_package_extension_compat_skipped(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_run_iter: MagicMock,
    ) -> None:
        config = self._make_config(iterations=1, detect_extensions=False)
        repo_dir = config.repos_dir / "mypkg"
        repo_dir.mkdir(parents=True)

        mock_install.return_value = MagicMock(returncode=0)
        mock_run_iter.return_value = IterationOutcome(
            index=1, status="pass", exit_code=0, duration_s=5.0
        )

        with patch("labeille.ft.runner.pull_repo"):
            with patch("labeille.ft.runner.assess_extension_compat") as mock_compat:
                result = run_package_ft(self._make_pkg(), config)
                mock_compat.assert_not_called()

        self.assertIsNone(result.extension_compat)


# ---------------------------------------------------------------------------
# FTRunConfig tests
# ---------------------------------------------------------------------------


class TestFTRunConfig(unittest.TestCase):
    def test_config_validate_ok(self) -> None:
        # Use a real existing path for target_python.
        config = FTRunConfig(target_python=Path(sys.executable))
        errors = config.validate()
        self.assertEqual(errors, [])

    def test_config_validate_missing_python(self) -> None:
        config = FTRunConfig(target_python=Path("/nonexistent/python"))
        errors = config.validate()
        self.assertTrue(any("not found" in e for e in errors))

    def test_config_validate_low_iterations(self) -> None:
        config = FTRunConfig(target_python=Path(sys.executable), iterations=0)
        errors = config.validate()
        self.assertTrue(any("Iterations" in e for e in errors))

    def test_config_validate_low_stall_threshold(self) -> None:
        config = FTRunConfig(target_python=Path(sys.executable), stall_threshold=2)
        errors = config.validate()
        self.assertTrue(any("Stall threshold" in e for e in errors))

    def test_config_validate_gil_compare_needs_iterations(self) -> None:
        config = FTRunConfig(
            target_python=Path(sys.executable),
            compare_with_gil=True,
            iterations=1,
        )
        errors = config.validate()
        self.assertTrue(any("comparison" in e.lower() for e in errors))


if __name__ == "__main__":
    unittest.main()
