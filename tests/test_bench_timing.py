"""Tests for labeille.bench.timing — timing capture for benchmark iterations."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import venv
from pathlib import Path

from labeille.bench.timing import (
    TimedResult,
    _parse_gnu_time_rss,
    run_timed,
    run_timed_in_venv,
)


# ---------------------------------------------------------------------------
# TimedResult property tests
# ---------------------------------------------------------------------------


class TestTimedResult(unittest.TestCase):
    """Tests for TimedResult dataclass."""

    def test_cpu_time_property(self) -> None:
        """cpu_time_s should be user + system time."""
        r = TimedResult(
            wall_time_s=5.0,
            user_time_s=1.5,
            sys_time_s=0.5,
            peak_rss_mb=100.0,
            exit_code=0,
            stdout="",
            stderr="",
        )
        self.assertAlmostEqual(r.cpu_time_s, 2.0, places=5)

    def test_cpu_time_zero(self) -> None:
        """cpu_time_s should be 0 when both components are 0."""
        r = TimedResult(
            wall_time_s=1.0,
            user_time_s=0.0,
            sys_time_s=0.0,
            peak_rss_mb=0.0,
            exit_code=0,
            stdout="",
            stderr="",
        )
        self.assertAlmostEqual(r.cpu_time_s, 0.0, places=5)


# ---------------------------------------------------------------------------
# run_timed tests
# ---------------------------------------------------------------------------


class TestRunTimed(unittest.TestCase):
    """Tests for run_timed() function."""

    def test_run_timed_simple_command(self) -> None:
        """Running 'echo hello' should succeed with output."""
        result = run_timed("echo hello")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello", result.stdout)
        self.assertGreater(result.wall_time_s, 0)

    def test_run_timed_captures_wall_time(self) -> None:
        """sleep 0.5 should take approximately 0.5 seconds."""
        result = run_timed("sleep 0.5", use_time_wrapper=False)
        self.assertGreater(result.wall_time_s, 0.4)
        self.assertLess(result.wall_time_s, 2.0)

    def test_run_timed_captures_cpu_time(self) -> None:
        """A CPU-bound Python script should register user CPU time."""
        result = run_timed(
            f"{sys.executable} -c 'sum(range(10**7))'",
            use_time_wrapper=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertGreater(result.user_time_s, 0)

    def test_run_timed_captures_exit_code(self) -> None:
        """Non-zero exit code should be captured."""
        result = run_timed("bash -c 'exit 42'")
        self.assertEqual(result.exit_code, 42)

    def test_run_timed_timeout(self) -> None:
        """A command exceeding the timeout should be killed."""
        result = run_timed("sleep 60", timeout=1, use_time_wrapper=False)
        self.assertTrue(result.timed_out)
        self.assertLess(result.wall_time_s, 5.0)

    def test_run_timed_stderr(self) -> None:
        """stderr output should be captured."""
        result = run_timed("echo err >&2")
        self.assertIn("err", result.stderr)

    def test_run_timed_captures_rss(self) -> None:
        """peak_rss_mb should be greater than zero for any process."""
        result = run_timed(f"{sys.executable} -c 'pass'")
        self.assertGreater(result.peak_rss_mb, 0)

    def test_run_timed_without_time_wrapper(self) -> None:
        """Should still work when time wrapper is disabled."""
        result = run_timed("echo hello", use_time_wrapper=False)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello", result.stdout)
        self.assertGreater(result.wall_time_s, 0)
        self.assertGreater(result.peak_rss_mb, 0)

    def test_run_timed_cwd(self) -> None:
        """Commands should run in the specified working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            testfile = Path(tmpdir) / "testfile.txt"
            testfile.write_text("hello from cwd")
            result = run_timed("cat testfile.txt", cwd=tmpdir)
            self.assertEqual(result.exit_code, 0)
            self.assertIn("hello from cwd", result.stdout)

    def test_run_timed_env(self) -> None:
        """Custom environment variables should be passed to the command."""
        result = run_timed("echo $LABEILLE_TEST_VAR", env={"LABEILLE_TEST_VAR": "bar"})
        self.assertIn("bar", result.stdout)

    def test_run_timed_list_command(self) -> None:
        """Command can be passed as a list."""
        result = run_timed(["echo", "hello", "world"], use_time_wrapper=False)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello world", result.stdout)


# ---------------------------------------------------------------------------
# run_timed_in_venv tests
# ---------------------------------------------------------------------------


class TestRunTimedInVenv(unittest.TestCase):
    """Tests for run_timed_in_venv() function.

    These tests create a real venv and may take a few seconds.
    """

    _venv_dir: str

    @classmethod
    def setUpClass(cls) -> None:
        """Create a temporary venv for all tests in this class."""
        cls._venv_dir = tempfile.mkdtemp(prefix="labeille-test-venv-")
        venv.create(cls._venv_dir, with_pip=False)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up the temporary venv."""
        import shutil

        shutil.rmtree(cls._venv_dir, ignore_errors=True)

    def test_run_timed_in_venv_uses_venv_python(self) -> None:
        """Python should report the venv path as its prefix."""
        venv_path = Path(self._venv_dir)
        result = run_timed_in_venv(
            venv_path,
            "python -c 'import sys; print(sys.prefix)'",
            cwd=Path.cwd(),
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(self._venv_dir, result.stdout)

    def test_run_timed_in_venv_isolation(self) -> None:
        """The Python executable should be inside the venv."""
        venv_path = Path(self._venv_dir)
        result = run_timed_in_venv(
            venv_path,
            "python -c 'import sys; print(sys.executable)'",
            cwd=Path.cwd(),
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(self._venv_dir, result.stdout)

    def test_run_timed_in_venv_with_extra_env(self) -> None:
        """Extra env vars should be passed through."""
        venv_path = Path(self._venv_dir)
        result = run_timed_in_venv(
            venv_path,
            "python -c 'import os; print(os.environ[\"LABEILLE_FLAG\"])'",
            cwd=Path.cwd(),
            env={"LABEILLE_FLAG": "yes"},
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("yes", result.stdout)


# ---------------------------------------------------------------------------
# _parse_gnu_time_rss tests
# ---------------------------------------------------------------------------


class TestParseGnuTimeRss(unittest.TestCase):
    """Tests for _parse_gnu_time_rss() helper."""

    def test_parse_rss_valid(self) -> None:
        """Should parse a well-formed GNU time output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                '\tCommand being timed: "sleep 0"\n'
                "\tUser time (seconds): 0.00\n"
                "\tSystem time (seconds): 0.00\n"
                "\tMaximum resident set size (kbytes): 123456\n"
                "\tMinor (reclaiming a frame) page faults: 100\n"
            )
            tmp_path = f.name
        try:
            rss_mb = _parse_gnu_time_rss(tmp_path)
            self.assertAlmostEqual(rss_mb, 123456 / 1024, places=1)
        finally:
            os.unlink(tmp_path)

    def test_parse_rss_missing_file(self) -> None:
        """Missing file should return 0.0."""
        self.assertAlmostEqual(
            _parse_gnu_time_rss("/nonexistent/path/file.txt"),
            0.0,
            places=5,
        )

    def test_parse_rss_malformed(self) -> None:
        """Malformed content should return 0.0."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("this is garbage\nno rss here\n")
            tmp_path = f.name
        try:
            self.assertAlmostEqual(_parse_gnu_time_rss(tmp_path), 0.0, places=5)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
