"""Tests for labeille.bench.system — system profiling for benchmarks."""

from __future__ import annotations

import platform
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from labeille.bench.system import (
    PythonProfile,
    StabilityCheck,
    SystemProfile,
    SystemSnapshot,
    capture_python_profile,
    capture_system_profile,
    check_stability,
    format_python_profile,
    format_system_profile,
)


# ---------------------------------------------------------------------------
# SystemProfile tests
# ---------------------------------------------------------------------------


class TestCaptureSystemProfile(unittest.TestCase):
    """Tests for capture_system_profile."""

    def test_capture_system_profile_returns_profile(self) -> None:
        profile = capture_system_profile()
        self.assertIsInstance(profile, SystemProfile)
        self.assertNotEqual(profile.hostname, "")
        self.assertNotEqual(profile.timestamp, "")
        self.assertNotEqual(profile.os_name, "")
        self.assertNotEqual(profile.cpu_architecture, "")

    def test_profile_cpu_cores_positive(self) -> None:
        profile = capture_system_profile()
        self.assertGreater(profile.cpu_cores_logical, 0)

    def test_profile_ram_positive(self) -> None:
        profile = capture_system_profile()
        self.assertGreater(profile.ram_total_gb, 0)

    def test_profile_load_avg_nonnegative(self) -> None:
        profile = capture_system_profile()
        self.assertGreaterEqual(profile.load_avg_1m, 0)

    def test_profile_disk_info_nonnegative(self) -> None:
        profile = capture_system_profile()
        self.assertGreaterEqual(profile.disk_available_gb, 0)


class TestSystemProfileSerialization(unittest.TestCase):
    """Tests for SystemProfile serialization."""

    def test_profile_serialization_roundtrip(self) -> None:
        profile = capture_system_profile()
        data = profile.to_dict()
        restored = SystemProfile.from_dict(data)
        self.assertEqual(profile.hostname, restored.hostname)
        self.assertEqual(profile.timestamp, restored.timestamp)
        self.assertEqual(profile.cpu_model, restored.cpu_model)
        self.assertEqual(profile.cpu_cores_logical, restored.cpu_cores_logical)
        self.assertEqual(profile.ram_total_gb, restored.ram_total_gb)
        self.assertEqual(profile.os_name, restored.os_name)

    def test_profile_json_roundtrip(self) -> None:
        profile = capture_system_profile()
        json_str = profile.to_json()
        restored = SystemProfile.from_json(json_str)
        self.assertEqual(profile.hostname, restored.hostname)
        self.assertEqual(profile.timestamp, restored.timestamp)
        self.assertEqual(profile.cpu_cores_logical, restored.cpu_cores_logical)

    def test_profile_from_dict_ignores_unknown_keys(self) -> None:
        data = {"hostname": "test", "unknown_field": "value", "extra": 42}
        profile = SystemProfile.from_dict(data)
        self.assertEqual(profile.hostname, "test")
        self.assertFalse(hasattr(profile, "unknown_field"))

    def test_profile_from_dict_missing_keys(self) -> None:
        data = {"hostname": "partial"}
        profile = SystemProfile.from_dict(data)
        self.assertEqual(profile.hostname, "partial")
        self.assertEqual(profile.cpu_model, "unknown")
        self.assertEqual(profile.cpu_cores_logical, 0)
        self.assertEqual(profile.ram_total_gb, 0.0)


class TestCaptureGracefulDegradation(unittest.TestCase):
    """Tests that capture functions handle missing /proc files gracefully."""

    @patch("labeille.bench.system.Path.read_text", side_effect=FileNotFoundError)
    @patch("labeille.bench.system.os.cpu_count", return_value=4)
    def test_capture_cpu_info_no_proc(self, _mock_cpu: MagicMock, _mock_read: MagicMock) -> None:
        from labeille.bench.system import _capture_cpu_info

        profile = SystemProfile()
        _capture_cpu_info(profile)
        # Should get logical cores from os.cpu_count, but no model name.
        self.assertEqual(profile.cpu_cores_logical, 4)
        self.assertEqual(profile.cpu_model, "unknown")

    @patch("labeille.bench.system.Path.read_text", side_effect=FileNotFoundError)
    def test_capture_memory_info_no_proc(self, _mock_read: MagicMock) -> None:
        from labeille.bench.system import _capture_memory_info

        profile = SystemProfile()
        _capture_memory_info(profile)
        self.assertEqual(profile.ram_total_gb, 0.0)
        self.assertEqual(profile.ram_available_gb, 0.0)


# ---------------------------------------------------------------------------
# PythonProfile tests
# ---------------------------------------------------------------------------


class TestCapturePythonProfile(unittest.TestCase):
    """Tests for capture_python_profile."""

    def test_capture_python_profile_self(self) -> None:
        profile = capture_python_profile(Path(sys.executable))
        self.assertIsInstance(profile, PythonProfile)
        self.assertNotEqual(profile.version, "")
        self.assertNotEqual(profile.implementation, "")

    def test_python_profile_version_matches(self) -> None:
        profile = capture_python_profile(Path(sys.executable))
        self.assertEqual(profile.version, platform.python_version())

    def test_python_profile_nonexistent_interpreter(self) -> None:
        profile = capture_python_profile(Path("/nonexistent/python"))
        self.assertIsInstance(profile, PythonProfile)
        self.assertEqual(profile.version, "")
        self.assertEqual(profile.path, "/nonexistent/python")

    def test_python_profile_serialization_roundtrip(self) -> None:
        profile = capture_python_profile(Path(sys.executable))
        data = profile.to_dict()
        restored = PythonProfile.from_dict(data)
        self.assertEqual(profile.version, restored.version)
        self.assertEqual(profile.implementation, restored.implementation)
        self.assertEqual(profile.compiler, restored.compiler)
        self.assertEqual(profile.build_flags, restored.build_flags)
        self.assertEqual(profile.jit_available, restored.jit_available)
        self.assertEqual(profile.debug_build, restored.debug_build)

    def test_python_profile_with_env(self) -> None:
        profile = capture_python_profile(Path(sys.executable), env={"PYTHON_JIT": "1"})
        self.assertIsInstance(profile, PythonProfile)
        self.assertNotEqual(profile.version, "")


# ---------------------------------------------------------------------------
# StabilityCheck tests
# ---------------------------------------------------------------------------


class TestCheckStability(unittest.TestCase):
    """Tests for check_stability."""

    def test_check_stability_passes_normally(self) -> None:
        # Use generous thresholds so this passes on typical dev/CI machines.
        result = check_stability(max_load=50.0, min_available_ram_gb=0.1)
        self.assertIsInstance(result, StabilityCheck)
        self.assertTrue(result.stable)

    @patch("labeille.bench.system.os.getloadavg", return_value=(5.0, 3.0, 2.0))
    def test_check_stability_high_load(self, _mock_load: MagicMock) -> None:
        result = check_stability(max_load=1.0)
        self.assertFalse(result.stable)
        self.assertTrue(any("load" in e.lower() for e in result.errors))

    @patch(
        "labeille.bench.system.Path.read_text",
        return_value="MemAvailable:   512000 kB\n",
    )
    def test_check_stability_low_ram(self, _mock_read: MagicMock) -> None:
        result = check_stability(min_available_ram_gb=2.0)
        self.assertFalse(result.stable)
        self.assertTrue(any("ram" in e.lower() for e in result.errors))

    @patch("labeille.bench.system.os.getloadavg", return_value=(0.5, 0.3, 0.2))
    def test_check_stability_custom_thresholds(self, _mock_load: MagicMock) -> None:
        result = check_stability(max_load=0.1)
        self.assertFalse(result.stable)

    @patch("labeille.bench.system.os.getloadavg", return_value=(0.8, 0.5, 0.3))
    def test_check_stability_warning_near_threshold(self, _mock_load: MagicMock) -> None:
        result = check_stability(max_load=1.0)
        self.assertTrue(any("approaching" in w for w in result.warnings))


# ---------------------------------------------------------------------------
# SystemSnapshot tests
# ---------------------------------------------------------------------------


class TestSystemSnapshot(unittest.TestCase):
    """Tests for SystemSnapshot."""

    def test_snapshot_capture(self) -> None:
        snap = SystemSnapshot.capture()
        self.assertGreater(snap.timestamp, 0)
        self.assertGreaterEqual(snap.load_avg_1m, 0)

    def test_snapshot_timing(self) -> None:
        snap1 = SystemSnapshot.capture()
        time.sleep(0.05)
        snap2 = SystemSnapshot.capture()
        self.assertGreater(snap2.timestamp, snap1.timestamp)


# ---------------------------------------------------------------------------
# Display tests
# ---------------------------------------------------------------------------


class TestFormatSystemProfile(unittest.TestCase):
    """Tests for format_system_profile."""

    def test_format_system_profile_contains_hostname(self) -> None:
        profile = SystemProfile(hostname="test-host")
        output = format_system_profile(profile)
        self.assertIn("test-host", output)

    def test_format_system_profile_contains_cpu(self) -> None:
        profile = SystemProfile(cpu_model="Test CPU")
        output = format_system_profile(profile)
        self.assertIn("Test CPU", output)

    def test_format_system_profile_shows_threads(self) -> None:
        profile = SystemProfile(cpu_cores_physical=4, cpu_cores_logical=8)
        output = format_system_profile(profile)
        self.assertIn("4 cores", output)
        self.assertIn("8 threads", output)

    def test_format_system_profile_no_threads_when_equal(self) -> None:
        profile = SystemProfile(cpu_cores_physical=4, cpu_cores_logical=4)
        output = format_system_profile(profile)
        self.assertIn("4 cores", output)
        self.assertNotIn("threads", output)

    def test_format_system_profile_shows_freq(self) -> None:
        profile = SystemProfile(cpu_freq_mhz=3600.0, cpu_freq_max_mhz=4800.0)
        output = format_system_profile(profile)
        self.assertIn("3600 MHz", output)
        self.assertIn("max 4800 MHz", output)

    def test_format_system_profile_shows_swap(self) -> None:
        profile = SystemProfile(swap_total_gb=8.0)
        output = format_system_profile(profile)
        self.assertIn("Swap:", output)
        self.assertIn("8.0 GB", output)

    def test_format_system_profile_hides_zero_swap(self) -> None:
        profile = SystemProfile(swap_total_gb=0.0)
        output = format_system_profile(profile)
        self.assertNotIn("Swap:", output)


class TestFormatPythonProfile(unittest.TestCase):
    """Tests for format_python_profile."""

    def test_format_python_profile_jit_flag(self) -> None:
        profile = PythonProfile(version="3.15.0", implementation="CPython", jit_enabled=True)
        output = format_python_profile(profile)
        self.assertIn("JIT enabled", output)

    def test_format_python_profile_jit_available_not_enabled(self) -> None:
        profile = PythonProfile(
            version="3.15.0",
            implementation="CPython",
            jit_available=True,
            jit_enabled=False,
        )
        output = format_python_profile(profile)
        self.assertIn("JIT available (disabled)", output)

    def test_format_python_profile_freethreaded(self) -> None:
        profile = PythonProfile(version="3.13.0", implementation="CPython", gil_disabled=True)
        output = format_python_profile(profile)
        self.assertIn("free-threaded", output)

    def test_format_python_profile_debug_build(self) -> None:
        profile = PythonProfile(version="3.15.0", implementation="CPython", debug_build=True)
        output = format_python_profile(profile)
        self.assertIn("debug build", output)

    def test_format_python_profile_build_flags(self) -> None:
        profile = PythonProfile(
            version="3.15.0",
            implementation="CPython",
            build_flags=["--enable-experimental-jit", "--with-lto"],
        )
        output = format_python_profile(profile)
        self.assertIn("--enable-experimental-jit", output)
        self.assertIn("--with-lto", output)

    def test_format_python_profile_no_interesting_flags(self) -> None:
        profile = PythonProfile(
            version="3.15.0",
            implementation="CPython",
            build_flags=["--prefix=/usr"],
        )
        output = format_python_profile(profile)
        self.assertNotIn("Build:", output)

    def test_format_python_profile_basic(self) -> None:
        profile = PythonProfile(
            version="3.15.0a5",
            implementation="CPython",
            compiler="GCC 13.2.0",
        )
        output = format_python_profile(profile)
        self.assertIn("3.15.0a5", output)
        self.assertIn("CPython", output)
        self.assertIn("GCC 13.2.0", output)


if __name__ == "__main__":
    unittest.main()
