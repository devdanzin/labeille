"""Tests for labeille.bench.cache — filesystem cache management."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from labeille.bench.cache import (
    DROP_CACHES_SCRIPT,
    check_cache_drop_available,
    check_not_root,
    drop_caches,
    format_setup_instructions,
    generate_drop_caches_script,
)
from labeille.bench.results import BenchIteration


# ---------------------------------------------------------------------------
# TestCheckCacheDropAvailable
# ---------------------------------------------------------------------------


class TestCheckCacheDropAvailable(unittest.TestCase):
    """Tests for check_cache_drop_available()."""

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.Path.exists", return_value=True)
    @patch("labeille.bench.cache.sys.platform", "linux")
    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    def test_linux_all_configured(
        self, mock_uid: MagicMock, mock_exists: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        status = check_cache_drop_available()
        self.assertTrue(status.available)
        self.assertTrue(status.platform_supported)
        self.assertTrue(status.script_exists)
        self.assertTrue(status.sudo_works)
        self.assertFalse(status.running_as_root)

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.Path.exists", return_value=True)
    @patch("labeille.bench.cache.sys.platform", "darwin")
    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    def test_macos_all_configured(
        self, mock_uid: MagicMock, mock_exists: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        status = check_cache_drop_available()
        self.assertTrue(status.available)
        self.assertTrue(status.platform_supported)

    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    @patch("labeille.bench.cache.sys.platform", "win32")
    def test_unsupported_platform(self, mock_uid: MagicMock) -> None:
        status = check_cache_drop_available()
        self.assertFalse(status.available)
        self.assertFalse(status.platform_supported)
        self.assertIn("only supported on Linux and macOS", status.message)

    @patch("labeille.bench.cache.Path.exists", return_value=False)
    @patch("labeille.bench.cache.sys.platform", "linux")
    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    def test_script_missing(self, mock_uid: MagicMock, mock_exists: MagicMock) -> None:
        status = check_cache_drop_available()
        self.assertFalse(status.available)
        self.assertFalse(status.script_exists)
        self.assertIn("not found", status.message)

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.Path.exists", return_value=True)
    @patch("labeille.bench.cache.sys.platform", "linux")
    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    def test_sudo_fails(
        self, mock_uid: MagicMock, mock_exists: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
        status = check_cache_drop_available()
        self.assertFalse(status.available)
        self.assertFalse(status.sudo_works)
        self.assertIn("Passwordless sudo not configured", status.message)

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.Path.exists", return_value=True)
    @patch("labeille.bench.cache.sys.platform", "linux")
    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    def test_sudo_timeout(
        self, mock_uid: MagicMock, mock_exists: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sudo", timeout=5)
        status = check_cache_drop_available()
        self.assertFalse(status.available)
        self.assertFalse(status.sudo_works)

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.Path.exists", return_value=True)
    @patch("labeille.bench.cache.sys.platform", "linux")
    @patch("labeille.bench.cache.os.getuid", return_value=0)
    def test_running_as_root_detected(
        self, mock_uid: MagicMock, mock_exists: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        status = check_cache_drop_available()
        self.assertTrue(status.running_as_root)
        self.assertTrue(status.available)


# ---------------------------------------------------------------------------
# TestDropCaches
# ---------------------------------------------------------------------------


class TestDropCaches(unittest.TestCase):
    """Tests for drop_caches()."""

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    def test_drop_caches_success(self, mock_uid: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"", stderr=b""
        )
        self.assertTrue(drop_caches())

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    def test_drop_caches_failure(self, mock_uid: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"permission denied"
        )
        self.assertFalse(drop_caches())

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.os.getuid", return_value=0)
    def test_drop_caches_refuses_root(self, mock_uid: MagicMock, mock_run: MagicMock) -> None:
        self.assertFalse(drop_caches(allow_root=False))
        mock_run.assert_not_called()

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.os.getuid", return_value=0)
    def test_drop_caches_allows_root(self, mock_uid: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"", stderr=b""
        )
        self.assertTrue(drop_caches(allow_root=True))
        # Should run without sudo.
        args = mock_run.call_args[0][0]
        self.assertEqual(args, [DROP_CACHES_SCRIPT])

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    def test_drop_caches_timeout(self, mock_uid: MagicMock, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sudo", timeout=10)
        self.assertFalse(drop_caches())

    @patch("labeille.bench.cache.subprocess.run")
    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    def test_drop_caches_uses_sudo(self, mock_uid: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"", stderr=b""
        )
        drop_caches()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[:2], ["sudo", "-n"])


# ---------------------------------------------------------------------------
# TestCheckNotRoot
# ---------------------------------------------------------------------------


class TestCheckNotRoot(unittest.TestCase):
    """Tests for check_not_root()."""

    @patch("labeille.bench.cache.os.getuid", return_value=1000)
    def test_not_root_passes(self, mock_uid: MagicMock) -> None:
        # Should not raise.
        check_not_root()

    @patch("labeille.bench.cache.os.getuid", return_value=0)
    def test_root_without_flag_exits(self, mock_uid: MagicMock) -> None:
        with self.assertRaises(SystemExit):
            check_not_root(allow_root=False)

    @patch("labeille.bench.cache.os.getuid", return_value=0)
    def test_root_with_flag_passes(self, mock_uid: MagicMock) -> None:
        # Should not raise.
        check_not_root(allow_root=True)


# ---------------------------------------------------------------------------
# TestGenerateDropCachesScript
# ---------------------------------------------------------------------------


class TestGenerateDropCachesScript(unittest.TestCase):
    """Tests for generate_drop_caches_script()."""

    def test_script_is_valid_sh(self) -> None:
        script = generate_drop_caches_script()
        self.assertTrue(script.startswith("#!/bin/sh"))
        self.assertIn("sync", script)

    def test_script_has_check_flag(self) -> None:
        script = generate_drop_caches_script()
        self.assertIn("--check", script)

    def test_script_handles_linux(self) -> None:
        script = generate_drop_caches_script()
        self.assertIn("echo 3 > /proc/sys/vm/drop_caches", script)

    def test_script_handles_macos(self) -> None:
        script = generate_drop_caches_script()
        self.assertIn("purge", script)


# ---------------------------------------------------------------------------
# TestFormatSetupInstructions
# ---------------------------------------------------------------------------


class TestFormatSetupInstructions(unittest.TestCase):
    """Tests for format_setup_instructions()."""

    def test_instructions_include_script_path(self) -> None:
        text = format_setup_instructions()
        self.assertIn("/usr/local/bin/labeille-drop-caches", text)

    def test_instructions_include_sudoers(self) -> None:
        text = format_setup_instructions()
        self.assertIn("/etc/sudoers.d/", text)

    def test_instructions_include_chmod(self) -> None:
        text = format_setup_instructions()
        self.assertIn("chmod 440", text)


# ---------------------------------------------------------------------------
# TestBenchIterationCachesDropped
# ---------------------------------------------------------------------------


class TestBenchIterationCachesDropped(unittest.TestCase):
    """Tests for BenchIteration caches_dropped field."""

    def test_iteration_caches_dropped_default_false(self) -> None:
        iteration = BenchIteration(
            index=1,
            warmup=False,
            wall_time_s=5.0,
            user_time_s=4.0,
            sys_time_s=0.5,
            peak_rss_mb=256.0,
            exit_code=0,
            status="ok",
        )
        self.assertFalse(iteration.caches_dropped)

    def test_iteration_caches_dropped_serialization(self) -> None:
        iteration = BenchIteration(
            index=1,
            warmup=False,
            wall_time_s=5.0,
            user_time_s=4.0,
            sys_time_s=0.5,
            peak_rss_mb=256.0,
            exit_code=0,
            status="ok",
            caches_dropped=True,
        )
        d = iteration.to_dict()
        self.assertTrue(d["caches_dropped"])
        restored = BenchIteration.from_dict(d)
        self.assertTrue(restored.caches_dropped)

    def test_iteration_caches_dropped_absent_in_old_data(self) -> None:
        d = {
            "index": 1,
            "warmup": False,
            "wall_time_s": 5.0,
            "user_time_s": 4.0,
            "sys_time_s": 0.5,
            "peak_rss_mb": 256.0,
            "exit_code": 0,
            "status": "ok",
        }
        iteration = BenchIteration.from_dict(d)
        self.assertFalse(iteration.caches_dropped)

    def test_iteration_caches_dropped_false_not_in_dict(self) -> None:
        """When caches_dropped is False, it should not appear in to_dict()."""
        iteration = BenchIteration(
            index=1,
            warmup=False,
            wall_time_s=5.0,
            user_time_s=4.0,
            sys_time_s=0.5,
            peak_rss_mb=256.0,
            exit_code=0,
            status="ok",
            caches_dropped=False,
        )
        d = iteration.to_dict()
        self.assertNotIn("caches_dropped", d)


if __name__ == "__main__":
    unittest.main()
