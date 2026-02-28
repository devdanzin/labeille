"""Tests for labeille.bench.system — system profiling for benchmarks."""

from __future__ import annotations

import platform
import subprocess
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
    _DarwinMemInfo,
    _parse_vm_stat,
    _parse_vm_stat_value,
    _sysctl,
    _sysctl_int,
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

    @patch("labeille.bench.system.sys.platform", "linux")
    @patch("labeille.bench.system.Path.read_text", side_effect=FileNotFoundError)
    @patch("labeille.bench.system.os.cpu_count", return_value=4)
    def test_capture_cpu_info_no_proc(self, _mock_cpu: MagicMock, _mock_read: MagicMock) -> None:
        from labeille.bench.system import _capture_cpu_info_linux

        profile = SystemProfile()
        _capture_cpu_info_linux(profile)
        # Should get logical cores from os.cpu_count, but no model name.
        self.assertEqual(profile.cpu_cores_logical, 4)
        self.assertEqual(profile.cpu_model, "unknown")

    @patch("labeille.bench.system.sys.platform", "linux")
    @patch("labeille.bench.system.Path.read_text", side_effect=FileNotFoundError)
    def test_capture_memory_info_no_proc(self, _mock_read: MagicMock) -> None:
        from labeille.bench.system import _capture_memory_info_linux

        profile = SystemProfile()
        _capture_memory_info_linux(profile)
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

    @patch("labeille.bench.system._get_available_ram_gb", return_value=0.5)
    def test_check_stability_low_ram(self, _mock_ram: MagicMock) -> None:
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

    @patch("labeille.bench.system._get_available_ram_gb", return_value=None)
    def test_check_stability_no_ram_info(self, _mock_ram: MagicMock) -> None:
        result = check_stability()
        self.assertTrue(any("memory" in w.lower() for w in result.warnings))


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

    def test_snapshot_uses_cross_platform_ram(self) -> None:
        """SystemSnapshot uses _get_available_ram_gb (cross-platform)."""
        with patch("labeille.bench.system._get_available_ram_gb", return_value=4.5):
            snap = SystemSnapshot.capture()
        self.assertEqual(snap.ram_available_gb, 4.5)

    def test_snapshot_handles_ram_none(self) -> None:
        """SystemSnapshot defaults to 0.0 when _get_available_ram_gb returns None."""
        with patch("labeille.bench.system._get_available_ram_gb", return_value=None):
            snap = SystemSnapshot.capture()
        self.assertEqual(snap.ram_available_gb, 0.0)


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

    def test_format_system_profile_linux_shows_kernel(self) -> None:
        profile = SystemProfile(
            os_name="Linux",
            os_distro="Ubuntu 24.04",
            os_kernel_version="6.5.0",
        )
        output = format_system_profile(profile)
        self.assertIn("Ubuntu 24.04 (Linux 6.5.0)", output)

    def test_format_system_profile_darwin_no_kernel(self) -> None:
        profile = SystemProfile(
            os_name="Darwin",
            os_distro="macOS 15.2",
            os_kernel_version="24.1.0",
        )
        output = format_system_profile(profile)
        self.assertIn("macOS 15.2", output)
        # Should NOT show the confusing Darwin kernel version.
        self.assertNotIn("24.1.0", output)


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


# ---------------------------------------------------------------------------
# macOS sysctl helpers (mock-based, run on all platforms)
# ---------------------------------------------------------------------------


class TestSysctlHelpers(unittest.TestCase):
    """Tests for _sysctl and _sysctl_int helpers."""

    @patch("labeille.bench.system.subprocess.run")
    def test_sysctl_returns_value(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Apple M2 Pro\n", stderr=""
        )
        result = _sysctl("machdep.cpu.brand_string")
        self.assertEqual(result, "Apple M2 Pro")

    @patch("labeille.bench.system.subprocess.run")
    def test_sysctl_returns_none_on_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="unknown oid"
        )
        result = _sysctl("nonexistent.key")
        self.assertIsNone(result)

    @patch("labeille.bench.system.subprocess.run", side_effect=FileNotFoundError)
    def test_sysctl_file_not_found(self, _mock: MagicMock) -> None:
        result = _sysctl("some.key")
        self.assertIsNone(result)

    @patch(
        "labeille.bench.system.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="sysctl", timeout=5),
    )
    def test_sysctl_timeout(self, _mock: MagicMock) -> None:
        result = _sysctl("some.key")
        self.assertIsNone(result)

    @patch("labeille.bench.system.subprocess.run")
    def test_sysctl_int_returns_integer(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="10\n", stderr=""
        )
        result = _sysctl_int("hw.physicalcpu")
        self.assertEqual(result, 10)

    @patch("labeille.bench.system.subprocess.run")
    def test_sysctl_int_returns_none_for_non_int(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not-a-number\n", stderr=""
        )
        result = _sysctl_int("some.key")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# macOS vm_stat parsing (mock-based, run on all platforms)
# ---------------------------------------------------------------------------

_VM_STAT_OUTPUT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              123456.
Pages active:                            654321.
Pages inactive:                          234567.
Pages speculative:                        12345.
Pages throttled:                              0.
Pages wired down:                        345678.
Pages purgeable:                          11111.
"""


class TestParseVmStat(unittest.TestCase):
    """Tests for _parse_vm_stat and _parse_vm_stat_value."""

    def test_parse_vm_stat_value(self) -> None:
        self.assertEqual(_parse_vm_stat_value("Pages free:    12345."), 12345)

    def test_parse_vm_stat_value_bad_input(self) -> None:
        self.assertEqual(_parse_vm_stat_value("garbage"), 0)

    @patch("labeille.bench.system.subprocess.run")
    def test_parse_vm_stat(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_VM_STAT_OUTPUT, stderr=""
        )
        info = _parse_vm_stat()
        self.assertIsNotNone(info)
        assert info is not None  # for type checker
        self.assertEqual(info.page_size, 16384)
        self.assertEqual(info.pages_free, 123456)
        self.assertEqual(info.pages_inactive, 234567)
        self.assertEqual(info.pages_active, 654321)
        self.assertEqual(info.pages_speculative, 12345)
        self.assertEqual(info.pages_wired, 345678)

    @patch("labeille.bench.system.subprocess.run")
    def test_parse_vm_stat_available_bytes(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_VM_STAT_OUTPUT, stderr=""
        )
        info = _parse_vm_stat()
        assert info is not None
        expected = (123456 + 234567) * 16384
        self.assertEqual(info.available_bytes, expected)

    @patch("labeille.bench.system.subprocess.run")
    def test_parse_vm_stat_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        )
        info = _parse_vm_stat()
        self.assertIsNone(info)

    @patch(
        "labeille.bench.system.subprocess.run",
        side_effect=FileNotFoundError,
    )
    def test_parse_vm_stat_not_found(self, _mock: MagicMock) -> None:
        info = _parse_vm_stat()
        self.assertIsNone(info)


class TestDarwinMemInfo(unittest.TestCase):
    """Tests for _DarwinMemInfo dataclass."""

    def test_available_bytes(self) -> None:
        info = _DarwinMemInfo(page_size=4096, pages_free=100, pages_inactive=50)
        self.assertEqual(info.available_bytes, 150 * 4096)


# ---------------------------------------------------------------------------
# macOS capture functions (mock-based, run on all platforms)
# ---------------------------------------------------------------------------


class TestCaptureCpuInfoDarwin(unittest.TestCase):
    """Tests for _capture_cpu_info_darwin."""

    @patch("labeille.bench.system._sysctl_int")
    @patch("labeille.bench.system._sysctl")
    @patch("labeille.bench.system.os.cpu_count", return_value=12)
    def test_apple_silicon(
        self, _mock_cpu: MagicMock, mock_sysctl: MagicMock, mock_sysctl_int: MagicMock
    ) -> None:
        from labeille.bench.system import _capture_cpu_info_darwin

        mock_sysctl.return_value = "Apple M2 Pro"
        mock_sysctl_int.side_effect = lambda k: {
            "hw.physicalcpu": 10,
            "hw.cpufrequency": None,
            "hw.cpufrequency_max": None,
        }.get(k)

        profile = SystemProfile()
        _capture_cpu_info_darwin(profile)

        self.assertEqual(profile.cpu_model, "Apple M2 Pro")
        self.assertEqual(profile.cpu_cores_physical, 10)
        self.assertEqual(profile.cpu_cores_logical, 12)
        self.assertIsNone(profile.cpu_freq_mhz)

    @patch("labeille.bench.system._sysctl_int")
    @patch("labeille.bench.system._sysctl")
    @patch("labeille.bench.system.os.cpu_count", return_value=16)
    def test_intel_mac(
        self, _mock_cpu: MagicMock, mock_sysctl: MagicMock, mock_sysctl_int: MagicMock
    ) -> None:
        from labeille.bench.system import _capture_cpu_info_darwin

        mock_sysctl.return_value = "Intel(R) Core(TM) i9-9980HK"
        mock_sysctl_int.side_effect = lambda k: {
            "hw.physicalcpu": 8,
            "hw.cpufrequency": 2400000000,
            "hw.cpufrequency_max": 5000000000,
        }.get(k)

        profile = SystemProfile()
        _capture_cpu_info_darwin(profile)

        self.assertEqual(profile.cpu_model, "Intel(R) Core(TM) i9-9980HK")
        self.assertEqual(profile.cpu_cores_physical, 8)
        self.assertEqual(profile.cpu_freq_mhz, 2400.0)
        self.assertEqual(profile.cpu_freq_max_mhz, 5000.0)


class TestCaptureMemoryInfoDarwin(unittest.TestCase):
    """Tests for _capture_memory_info_darwin."""

    @patch("labeille.bench.system._parse_vm_stat")
    @patch("labeille.bench.system._sysctl_int")
    @patch("labeille.bench.system._sysctl")
    def test_memory_info(
        self,
        mock_sysctl: MagicMock,
        mock_sysctl_int: MagicMock,
        mock_vm_stat: MagicMock,
    ) -> None:
        from labeille.bench.system import _capture_memory_info_darwin

        # 32 GB
        mock_sysctl_int.return_value = 34359738368
        mock_sysctl.return_value = None

        mock_vm_stat.return_value = _DarwinMemInfo(
            page_size=16384,
            pages_free=123456,
            pages_inactive=234567,
        )

        profile = SystemProfile()
        _capture_memory_info_darwin(profile)

        self.assertAlmostEqual(profile.ram_total_gb, 32.0, places=0)
        expected_avail = (123456 + 234567) * 16384 / (1024**3)
        self.assertAlmostEqual(profile.ram_available_gb, expected_avail, places=1)

    @patch("labeille.bench.system._parse_vm_stat", return_value=None)
    @patch("labeille.bench.system._sysctl_int", return_value=None)
    @patch("labeille.bench.system._sysctl")
    def test_memory_info_all_fail(
        self,
        mock_sysctl: MagicMock,
        _mock_int: MagicMock,
        _mock_vm: MagicMock,
    ) -> None:
        from labeille.bench.system import _capture_memory_info_darwin

        mock_sysctl.return_value = None
        profile = SystemProfile()
        _capture_memory_info_darwin(profile)
        self.assertEqual(profile.ram_total_gb, 0.0)

    @patch("labeille.bench.system._parse_vm_stat", return_value=None)
    @patch("labeille.bench.system._sysctl_int", return_value=None)
    @patch("labeille.bench.system._sysctl")
    def test_swap_parsing(
        self,
        mock_sysctl: MagicMock,
        _mock_int: MagicMock,
        _mock_vm: MagicMock,
    ) -> None:
        from labeille.bench.system import _capture_memory_info_darwin

        mock_sysctl.side_effect = lambda k: {
            "vm.swapusage": "total = 2048.00M  used = 512.00M  free = 1536.00M"
        }.get(k)

        profile = SystemProfile()
        _capture_memory_info_darwin(profile)
        self.assertAlmostEqual(profile.swap_total_gb, 2.0, places=1)


class TestCaptureOsInfoDarwin(unittest.TestCase):
    """Tests for _capture_os_info_darwin."""

    @patch("labeille.bench.system.subprocess.run")
    def test_os_info(self, mock_run: MagicMock) -> None:
        from labeille.bench.system import _capture_os_info_darwin

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=("ProductName:\tmacOS\nProductVersion:\t15.2\nBuildVersion:\t24C101\n"),
            stderr="",
        )

        profile = SystemProfile()
        _capture_os_info_darwin(profile)
        self.assertEqual(profile.os_distro, "macOS 15.2")

    @patch("labeille.bench.system.subprocess.run", side_effect=FileNotFoundError)
    def test_os_info_fallback(self, _mock: MagicMock) -> None:
        from labeille.bench.system import _capture_os_info_darwin

        profile = SystemProfile()
        _capture_os_info_darwin(profile)
        # Should fall back to platform.mac_ver, which returns empty on Linux.
        # No crash is the important thing.
        self.assertIsInstance(profile.os_distro, str)


class TestCaptureDiskInfoDarwin(unittest.TestCase):
    """Tests for _capture_disk_info_darwin."""

    @patch("labeille.bench.system.subprocess.run")
    @patch("labeille.bench.system.os.statvfs")
    def test_ssd_detection(self, mock_statvfs: MagicMock, mock_run: MagicMock) -> None:
        from labeille.bench.system import _capture_disk_info_darwin

        mock_statvfs.return_value = MagicMock(f_bavail=100000000, f_frsize=4096)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="   Solid State:              Yes\n",
            stderr="",
        )

        profile = SystemProfile()
        _capture_disk_info_darwin(profile)
        self.assertEqual(profile.disk_type, "ssd")

    @patch("labeille.bench.system.subprocess.run")
    @patch("labeille.bench.system.os.statvfs")
    def test_apfs_fallback(self, mock_statvfs: MagicMock, mock_run: MagicMock) -> None:
        from labeille.bench.system import _capture_disk_info_darwin

        mock_statvfs.return_value = MagicMock(f_bavail=100000000, f_frsize=4096)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="   File System:              APFS\n   Volume Name:  Macintosh HD\n",
            stderr="",
        )

        profile = SystemProfile()
        _capture_disk_info_darwin(profile)
        self.assertEqual(profile.disk_type, "ssd")


class TestCaptureProcessCountDarwin(unittest.TestCase):
    """Tests for _capture_process_count_darwin."""

    @patch("labeille.bench.system.subprocess.run")
    def test_process_count(self, mock_run: MagicMock) -> None:
        from labeille.bench.system import _capture_process_count_darwin

        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="STAT\nR\nS\nS\nR\nS\n",
            stderr="",
        )

        profile = SystemProfile()
        _capture_process_count_darwin(profile)
        self.assertEqual(profile.running_processes, 2)


# ---------------------------------------------------------------------------
# Cross-platform available RAM helper
# ---------------------------------------------------------------------------


class TestGetAvailableRamGb(unittest.TestCase):
    """Tests for _get_available_ram_gb."""

    @patch("labeille.bench.system.sys.platform", "linux")
    @patch(
        "labeille.bench.system.Path.read_text",
        return_value="MemTotal:       16000000 kB\nMemAvailable:   8388608 kB\n",
    )
    def test_linux(self, _mock: MagicMock) -> None:
        from labeille.bench.system import _get_available_ram_gb

        result = _get_available_ram_gb()
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 8.0, places=0)  # type: ignore[arg-type]

    @patch("labeille.bench.system.sys.platform", "darwin")
    @patch("labeille.bench.system._parse_vm_stat")
    def test_darwin(self, mock_vm_stat: MagicMock) -> None:
        from labeille.bench.system import _get_available_ram_gb

        mock_vm_stat.return_value = _DarwinMemInfo(
            page_size=16384, pages_free=100000, pages_inactive=50000
        )
        result = _get_available_ram_gb()
        self.assertIsNotNone(result)
        expected = (100000 + 50000) * 16384 / (1024**3)
        self.assertAlmostEqual(result, expected, places=1)  # type: ignore[arg-type]

    @patch("labeille.bench.system.sys.platform", "win32")
    def test_unsupported_platform(self) -> None:
        from labeille.bench.system import _get_available_ram_gb

        result = _get_available_ram_gb()
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Platform dispatch tests
# ---------------------------------------------------------------------------


class TestPlatformDispatch(unittest.TestCase):
    """Tests that dispatch functions call the right platform variant."""

    @patch("labeille.bench.system.sys.platform", "linux")
    @patch("labeille.bench.system._capture_cpu_info_linux")
    def test_cpu_dispatches_linux(self, mock_fn: MagicMock) -> None:
        _capture_cpu_info = __import__(
            "labeille.bench.system", fromlist=["_capture_cpu_info"]
        )._capture_cpu_info
        _capture_cpu_info(SystemProfile())
        mock_fn.assert_called_once()

    @patch("labeille.bench.system.sys.platform", "darwin")
    @patch("labeille.bench.system._capture_cpu_info_darwin")
    def test_cpu_dispatches_darwin(self, mock_fn: MagicMock) -> None:
        from labeille.bench.system import _capture_cpu_info

        _capture_cpu_info(SystemProfile())
        mock_fn.assert_called_once()

    @patch("labeille.bench.system.sys.platform", "darwin")
    @patch("labeille.bench.system._capture_memory_info_darwin")
    def test_memory_dispatches_darwin(self, mock_fn: MagicMock) -> None:
        from labeille.bench.system import _capture_memory_info

        _capture_memory_info(SystemProfile())
        mock_fn.assert_called_once()

    @patch("labeille.bench.system.sys.platform", "darwin")
    @patch("labeille.bench.system._capture_os_info_darwin")
    def test_os_dispatches_darwin(self, mock_fn: MagicMock) -> None:
        from labeille.bench.system import _capture_os_info

        _capture_os_info(SystemProfile())
        mock_fn.assert_called_once()

    @patch("labeille.bench.system.sys.platform", "darwin")
    @patch("labeille.bench.system._capture_disk_info_darwin")
    def test_disk_dispatches_darwin(self, mock_fn: MagicMock) -> None:
        from labeille.bench.system import _capture_disk_info

        _capture_disk_info(SystemProfile())
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# macOS-only live tests (skipped on Linux)
# ---------------------------------------------------------------------------


@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class TestMacOSLive(unittest.TestCase):
    """Live tests that only run on macOS."""

    def test_sysctl_cpu_brand(self) -> None:
        """sysctl should return a CPU model string on macOS."""
        model = _sysctl("machdep.cpu.brand_string")
        self.assertIsNotNone(model)
        assert model is not None
        self.assertGreater(len(model), 0)

    def test_sysctl_hw_memsize(self) -> None:
        """hw.memsize should return a positive integer."""
        memsize = _sysctl_int("hw.memsize")
        self.assertIsNotNone(memsize)
        assert memsize is not None
        self.assertGreater(memsize, 0)

    def test_vm_stat_parses(self) -> None:
        """vm_stat should return parseable data."""
        info = _parse_vm_stat()
        self.assertIsNotNone(info)
        assert info is not None
        self.assertGreater(info.pages_free, 0)

    def test_capture_full_profile(self) -> None:
        """Full system profile should work on macOS."""
        profile = capture_system_profile()
        self.assertGreater(profile.ram_total_gb, 0)
        self.assertGreater(profile.cpu_cores_logical, 0)
        self.assertIn("macOS", profile.os_distro)

    def test_stability_check(self) -> None:
        """Stability check should work on macOS."""
        result = check_stability(max_load=50.0, min_available_ram_gb=0.1)
        self.assertIsInstance(result.stable, bool)


if __name__ == "__main__":
    unittest.main()
