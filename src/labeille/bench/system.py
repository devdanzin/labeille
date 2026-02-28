"""System characterization for benchmark reproducibility.

Captures hardware, OS, Python, and runtime state information so
benchmark results can be properly contextualized and compared.

Supports Linux and macOS. Each capture function dispatches to a
platform-specific implementation; unsupported platforms get defaults.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("labeille")


# ---------------------------------------------------------------------------
# SystemProfile
# ---------------------------------------------------------------------------


@dataclass
class SystemProfile:
    """Complete characterization of the system running a benchmark."""

    # CPU
    cpu_model: str = "unknown"
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_mhz: float | None = None
    cpu_freq_max_mhz: float | None = None
    cpu_architecture: str = ""

    # Memory
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    swap_total_gb: float = 0.0

    # OS
    os_name: str = ""
    os_kernel_version: str = ""
    os_distro: str = ""

    # Python (of the benchmarking process itself, not the target)
    host_python_version: str = ""
    host_python_implementation: str = ""

    # System state at capture time
    load_avg_1m: float = 0.0
    load_avg_5m: float = 0.0
    load_avg_15m: float = 0.0
    running_processes: int = 0

    # Disk
    disk_type: str = "unknown"  # ssd / hdd / unknown
    disk_available_gb: float = 0.0

    # Environment
    hostname: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemProfile:
        """Deserialize from a dict, ignoring unknown fields."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> SystemProfile:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# PythonProfile
# ---------------------------------------------------------------------------


@dataclass
class PythonProfile:
    """Characterization of a target Python interpreter."""

    path: str = ""
    version: str = ""
    implementation: str = ""  # CPython, PyPy, etc.
    compiler: str = ""  # e.g. "GCC 13.2.0"
    build_flags: list[str] = field(default_factory=list)
    jit_available: bool = False
    jit_enabled: bool = False
    gil_disabled: bool = False
    debug_build: bool = False
    hash_randomization: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PythonProfile:
        """Deserialize from a dict, ignoring unknown fields."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# StabilityCheck
# ---------------------------------------------------------------------------


@dataclass
class StabilityCheck:
    """Result of checking system stability for benchmarking."""

    stable: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SystemSnapshot
# ---------------------------------------------------------------------------


@dataclass
class SystemSnapshot:
    """Lightweight system state captured before/after each iteration."""

    timestamp: float  # time.monotonic()
    load_avg_1m: float = 0.0
    ram_available_gb: float = 0.0

    @classmethod
    def capture(cls) -> SystemSnapshot:
        """Capture current system state."""
        snap = cls(timestamp=time.monotonic())
        try:
            snap.load_avg_1m = round(os.getloadavg()[0], 2)
        except Exception:  # noqa: BLE001
            pass

        avail = _get_available_ram_gb()
        if avail is not None:
            snap.ram_available_gb = round(avail, 2)

        return snap


# ---------------------------------------------------------------------------
# macOS sysctl helpers
# ---------------------------------------------------------------------------


def _sysctl(key: str) -> str | None:
    """Read a sysctl string value. Returns None on failure."""
    try:
        proc = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _sysctl_int(key: str) -> int | None:
    """Read a sysctl integer value. Returns None on failure."""
    val = _sysctl(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# macOS vm_stat helper
# ---------------------------------------------------------------------------


@dataclass
class _DarwinMemInfo:
    """Parsed vm_stat output."""

    page_size: int = 16384
    pages_free: int = 0
    pages_inactive: int = 0
    pages_active: int = 0
    pages_speculative: int = 0
    pages_wired: int = 0

    @property
    def available_bytes(self) -> int:
        """Approximate available memory (free + inactive pages)."""
        return (self.pages_free + self.pages_inactive) * self.page_size


def _parse_vm_stat() -> _DarwinMemInfo | None:
    """Parse macOS vm_stat output into structured data."""
    try:
        proc = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None

        info = _DarwinMemInfo()
        for line in proc.stdout.splitlines():
            if "page size of" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "size":
                        try:
                            info.page_size = int(parts[i + 2].rstrip(")"))
                        except (IndexError, ValueError):
                            pass
            elif line.startswith("Pages free:"):
                info.pages_free = _parse_vm_stat_value(line)
            elif line.startswith("Pages inactive:"):
                info.pages_inactive = _parse_vm_stat_value(line)
            elif line.startswith("Pages active:"):
                info.pages_active = _parse_vm_stat_value(line)
            elif line.startswith("Pages speculative:"):
                info.pages_speculative = _parse_vm_stat_value(line)
            elif line.startswith("Pages wired down:"):
                info.pages_wired = _parse_vm_stat_value(line)
        return info
    except Exception:  # noqa: BLE001
        return None


def _parse_vm_stat_value(line: str) -> int:
    """Parse a vm_stat line like 'Pages free:    12345.' -> 12345."""
    try:
        return int(line.split(":")[1].strip().rstrip("."))
    except (IndexError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Cross-platform available RAM helper
# ---------------------------------------------------------------------------


def _get_available_ram_gb() -> float | None:
    """Get available RAM in GB, platform-aware."""
    if sys.platform == "linux":
        return _get_available_ram_gb_linux()
    elif sys.platform == "darwin":
        return _get_available_ram_gb_darwin()
    return None


def _get_available_ram_gb_linux() -> float | None:
    """Get available RAM from /proc/meminfo on Linux."""
    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    return None


def _get_available_ram_gb_darwin() -> float | None:
    """Get available RAM from vm_stat on macOS."""
    info = _parse_vm_stat()
    if info is not None:
        return info.available_bytes / (1024**3)
    return None


# ---------------------------------------------------------------------------
# Capture functions (platform dispatch)
# ---------------------------------------------------------------------------


def capture_system_profile() -> SystemProfile:
    """Capture a complete system profile.

    Gathers CPU, memory, OS, disk, and runtime state information.
    All operations are best-effort — individual failures produce
    default values rather than exceptions.
    """
    profile = SystemProfile(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        hostname=platform.node(),
        cpu_architecture=platform.machine(),
        os_name=platform.system(),
        os_kernel_version=platform.release(),
        host_python_version=platform.python_version(),
        host_python_implementation=platform.python_implementation(),
    )

    _capture_cpu_info(profile)
    _capture_memory_info(profile)
    _capture_os_info(profile)
    _capture_load(profile)
    _capture_disk_info(profile)

    return profile


def _capture_cpu_info(profile: SystemProfile) -> None:
    """Populate CPU fields (dispatches by platform)."""
    if sys.platform == "linux":
        _capture_cpu_info_linux(profile)
    elif sys.platform == "darwin":
        _capture_cpu_info_darwin(profile)
    else:
        log.debug("CPU info capture not supported on %s", sys.platform)


def _capture_memory_info(profile: SystemProfile) -> None:
    """Populate memory fields (dispatches by platform)."""
    if sys.platform == "linux":
        _capture_memory_info_linux(profile)
    elif sys.platform == "darwin":
        _capture_memory_info_darwin(profile)
    else:
        log.debug("Memory info capture not supported on %s", sys.platform)


def _capture_os_info(profile: SystemProfile) -> None:
    """Populate OS distro info (dispatches by platform)."""
    if sys.platform == "linux":
        _capture_os_info_linux(profile)
    elif sys.platform == "darwin":
        _capture_os_info_darwin(profile)
    else:
        log.debug("OS info capture not supported on %s", sys.platform)


def _capture_load(profile: SystemProfile) -> None:
    """Capture system load average and process count."""
    # Load averages are POSIX — work on both Linux and macOS.
    try:
        load = os.getloadavg()
        profile.load_avg_1m = round(load[0], 2)
        profile.load_avg_5m = round(load[1], 2)
        profile.load_avg_15m = round(load[2], 2)
    except Exception:  # noqa: BLE001
        pass

    # Process count is platform-specific.
    if sys.platform == "linux":
        _capture_process_count_linux(profile)
    elif sys.platform == "darwin":
        _capture_process_count_darwin(profile)


def _capture_disk_info(profile: SystemProfile) -> None:
    """Populate disk info (dispatches by platform)."""
    if sys.platform == "linux":
        _capture_disk_info_linux(profile)
    elif sys.platform == "darwin":
        _capture_disk_info_darwin(profile)
    else:
        log.debug("Disk info capture not supported on %s", sys.platform)


# ---------------------------------------------------------------------------
# Linux capture implementations
# ---------------------------------------------------------------------------


def _capture_cpu_info_linux(profile: SystemProfile) -> None:
    """Populate CPU fields from /proc/cpuinfo and os.cpu_count."""
    try:
        profile.cpu_cores_logical = os.cpu_count() or 0
    except Exception:  # noqa: BLE001
        pass

    # Physical cores from /proc/cpuinfo (count unique core ids).
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        # Model name — take the first one (all cores report the same).
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                profile.cpu_model = line.split(":", 1)[1].strip()
                break

        # Physical cores: count unique (physical id, core id) pairs.
        physical_ids: set[tuple[str, str]] = set()
        current_physical: str | None = None
        for line in cpuinfo.splitlines():
            if line.startswith("physical id"):
                current_physical = line.split(":", 1)[1].strip()
            elif line.startswith("core id") and current_physical is not None:
                core_id = line.split(":", 1)[1].strip()
                physical_ids.add((current_physical, core_id))
                current_physical = None
        if physical_ids:
            profile.cpu_cores_physical = len(physical_ids)
        else:
            # Fallback: assume no hyperthreading.
            profile.cpu_cores_physical = profile.cpu_cores_logical
    except Exception:  # noqa: BLE001
        pass

    # CPU frequency from /sys or /proc/cpuinfo.
    try:
        # Try scaling_cur_freq first (more accurate, in kHz).
        freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if freq_path.exists():
            profile.cpu_freq_mhz = int(freq_path.read_text().strip()) / 1000

        max_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
        if max_path.exists():
            profile.cpu_freq_max_mhz = int(max_path.read_text().strip()) / 1000
    except Exception:  # noqa: BLE001
        pass

    # Fallback: parse "cpu MHz" from /proc/cpuinfo.
    if profile.cpu_freq_mhz is None:
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
            for line in cpuinfo.splitlines():
                if line.startswith("cpu MHz"):
                    profile.cpu_freq_mhz = float(line.split(":", 1)[1].strip())
                    break
        except Exception:  # noqa: BLE001
            pass


def _capture_memory_info_linux(profile: SystemProfile) -> None:
    """Populate memory fields from /proc/meminfo."""
    try:
        meminfo = Path("/proc/meminfo").read_text()
        mem: dict[str, int] = {}
        for line in meminfo.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                # Values are in kB.
                value_kb = int(parts[1])
                mem[key] = value_kb

        profile.ram_total_gb = mem.get("MemTotal", 0) / (1024 * 1024)
        profile.ram_available_gb = mem.get("MemAvailable", 0) / (1024 * 1024)
        profile.swap_total_gb = mem.get("SwapTotal", 0) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass


def _capture_os_info_linux(profile: SystemProfile) -> None:
    """Populate OS distro from /etc/os-release or platform."""
    try:
        os_release = Path("/etc/os-release").read_text()
        for line in os_release.splitlines():
            if line.startswith("PRETTY_NAME="):
                profile.os_distro = line.split("=", 1)[1].strip().strip('"')
                break
    except Exception:  # noqa: BLE001
        # Fallback.
        try:
            profile.os_distro = f"{platform.system()} {platform.release()}"
        except Exception:  # noqa: BLE001
            pass


def _capture_process_count_linux(profile: SystemProfile) -> None:
    """Get running process count from /proc/stat on Linux."""
    try:
        stat = Path("/proc/stat").read_text()
        for line in stat.splitlines():
            if line.startswith("procs_running"):
                profile.running_processes = int(line.split()[1])
                break
    except Exception:  # noqa: BLE001
        pass


def _capture_disk_info_linux(profile: SystemProfile) -> None:
    """Capture disk type and available space on Linux."""
    try:
        stat = os.statvfs("/")
        profile.disk_available_gb = round((stat.f_bavail * stat.f_frsize) / (1024**3), 1)
    except Exception:  # noqa: BLE001
        pass

    # Detect SSD vs HDD from /sys/block.
    try:
        # Find the root device.
        with open("/proc/mounts") as f:
            dev: str | None = None
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "/":
                    dev = parts[0]
                    break

        if dev is None:
            return

        # Extract the base device name (e.g., /dev/sda1 → sda).
        dev_name = Path(dev).name
        # Strip partition number.
        base_dev = re.sub(r"\d+$", "", dev_name)
        # Strip 'p' partition prefix for nvme (nvme0n1p1 → nvme0n1).
        if base_dev.startswith("nvme"):
            base_dev = re.sub(r"p\d*$", "", dev_name)

        rotational = Path(f"/sys/block/{base_dev}/queue/rotational")
        if rotational.exists():
            is_rotational = rotational.read_text().strip() == "1"
            profile.disk_type = "hdd" if is_rotational else "ssd"
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# macOS capture implementations
# ---------------------------------------------------------------------------


def _capture_cpu_info_darwin(profile: SystemProfile) -> None:
    """Populate CPU fields using sysctl on macOS."""
    try:
        profile.cpu_cores_logical = os.cpu_count() or 0
    except Exception:  # noqa: BLE001
        pass

    # CPU model.
    model = _sysctl("machdep.cpu.brand_string")
    if model:
        profile.cpu_model = model

    # Physical cores.
    phys = _sysctl_int("hw.physicalcpu")
    if phys is not None:
        profile.cpu_cores_physical = phys
    else:
        profile.cpu_cores_physical = profile.cpu_cores_logical

    # CPU frequency.
    # On Intel Macs, hw.cpufrequency gives the base frequency in Hz.
    # On Apple Silicon, this sysctl doesn't exist — frequency is
    # dynamic and not exposed. We leave it as None.
    freq_hz = _sysctl_int("hw.cpufrequency")
    if freq_hz is not None:
        profile.cpu_freq_mhz = freq_hz / 1_000_000

    freq_max_hz = _sysctl_int("hw.cpufrequency_max")
    if freq_max_hz is not None:
        profile.cpu_freq_max_mhz = freq_max_hz / 1_000_000


def _capture_memory_info_darwin(profile: SystemProfile) -> None:
    """Populate memory fields using sysctl and vm_stat on macOS."""
    # Total RAM from sysctl (in bytes).
    total_bytes = _sysctl_int("hw.memsize")
    if total_bytes is not None:
        profile.ram_total_gb = total_bytes / (1024**3)

    # Available memory from vm_stat.
    info = _parse_vm_stat()
    if info is not None:
        profile.ram_available_gb = round(info.available_bytes / (1024**3), 2)

    # Swap from sysctl.
    swap_usage = _sysctl("vm.swapusage")
    if swap_usage:
        try:
            match = re.search(r"total\s*=\s*([\d.]+)([MG])", swap_usage)
            if match:
                val = float(match.group(1))
                unit = match.group(2)
                if unit == "G":
                    profile.swap_total_gb = val
                elif unit == "M":
                    profile.swap_total_gb = val / 1024
        except Exception:  # noqa: BLE001
            pass


def _capture_os_info_darwin(profile: SystemProfile) -> None:
    """Populate OS info on macOS using sw_vers."""
    try:
        proc = subprocess.run(
            ["sw_vers"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            name = ""
            version = ""
            for line in proc.stdout.splitlines():
                if line.startswith("ProductName:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("ProductVersion:"):
                    version = line.split(":", 1)[1].strip()
            if name and version:
                profile.os_distro = f"{name} {version}"
    except Exception:  # noqa: BLE001
        # Fallback.
        try:
            profile.os_distro = f"macOS {platform.mac_ver()[0]}"
        except Exception:  # noqa: BLE001
            pass


def _capture_process_count_darwin(profile: SystemProfile) -> None:
    """Get running process count on macOS."""
    try:
        proc = subprocess.run(
            ["ps", "-axo", "state"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            # Count processes in running state (R).
            running = sum(1 for line in proc.stdout.splitlines() if line.strip().startswith("R"))
            profile.running_processes = running
    except Exception:  # noqa: BLE001
        pass


def _capture_disk_info_darwin(profile: SystemProfile) -> None:
    """Capture disk info on macOS using diskutil."""
    # Available space (os.statvfs works on macOS).
    try:
        stat = os.statvfs("/")
        profile.disk_available_gb = round((stat.f_bavail * stat.f_frsize) / (1024**3), 1)
    except Exception:  # noqa: BLE001
        pass

    # SSD vs HDD detection.
    try:
        proc = subprocess.run(
            ["diskutil", "info", "/"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if "Solid State" in line:
                    if "Yes" in line:
                        profile.disk_type = "ssd"
                    elif "No" in line:
                        profile.disk_type = "hdd"
                    break
            else:
                # APFS on Apple Silicon is always SSD.
                if "APFS" in proc.stdout:
                    profile.disk_type = "ssd"
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Target Python profiling
# ---------------------------------------------------------------------------

_PYTHON_PROBE_SCRIPT = """\
import json
import platform
import sys
import sysconfig

result = {
    "version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "compiler": platform.python_compiler(),
    "debug_build": hasattr(sys, "gettotalrefcount"),
    "hash_randomization": bool(
        getattr(sys.flags, "hash_randomization", True)
    ),
}

# Build flags from sysconfig.
config_args = sysconfig.get_config_var("CONFIG_ARGS") or ""
result["build_flags"] = [
    arg.strip("'\\\"") for arg in config_args.split()
    if arg.startswith("--enable-") or arg.startswith("--with-")
]

# JIT availability (CPython 3.15+).
try:
    result["jit_available"] = hasattr(sys.flags, "jit")
    result["jit_enabled"] = bool(getattr(sys.flags, "jit", False))
except Exception:
    result["jit_available"] = False
    result["jit_enabled"] = False

# GIL status (CPython 3.13+ free-threading).
try:
    result["gil_disabled"] = not sys._is_gil_enabled()
except AttributeError:
    result["gil_disabled"] = False

print(json.dumps(result))
"""


def capture_python_profile(
    python_path: Path,
    env: dict[str, str] | None = None,
) -> PythonProfile:
    """Characterize a target Python interpreter.

    Runs a small script in the target Python to extract version,
    build configuration, and runtime flags.

    Args:
        python_path: Path to the Python executable.
        env: Environment variables (e.g., PYTHON_JIT=1) to set
             when probing the interpreter.
    """
    profile = PythonProfile(path=str(python_path))

    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    # Don't let host PYTHONHOME/PYTHONPATH contaminate the probe.
    run_env.pop("PYTHONHOME", None)
    run_env.pop("PYTHONPATH", None)
    run_env["ASAN_OPTIONS"] = "detect_leaks=0"

    try:
        proc = subprocess.run(
            [str(python_path), "-c", _PYTHON_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
            env=run_env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout.strip())
            profile.version = data.get("version", "")
            profile.implementation = data.get("implementation", "")
            profile.compiler = data.get("compiler", "")
            profile.build_flags = data.get("build_flags", [])
            profile.jit_available = data.get("jit_available", False)
            profile.jit_enabled = data.get("jit_enabled", False)
            profile.gil_disabled = data.get("gil_disabled", False)
            profile.debug_build = data.get("debug_build", False)
            profile.hash_randomization = data.get("hash_randomization", True)
        else:
            log.warning(
                "Python profile probe failed (exit %d): %s",
                proc.returncode,
                proc.stderr.strip()[:200],
            )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        log.warning("Could not profile Python at %s: %s", python_path, exc)

    return profile


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------


def check_stability(
    *,
    max_load: float = 1.0,
    min_available_ram_gb: float = 2.0,
) -> StabilityCheck:
    """Check whether the system is stable enough for benchmarking.

    Args:
        max_load: Maximum acceptable 1-minute load average.
        min_available_ram_gb: Minimum available RAM in GB.

    Returns:
        StabilityCheck with pass/fail and diagnostic messages.
    """
    result = StabilityCheck(stable=True)

    # Load average — works on both Linux and macOS (POSIX).
    try:
        load = os.getloadavg()
        if load[0] > max_load:
            result.stable = False
            result.errors.append(
                f"1-minute load average is {load[0]:.1f} "
                f"(threshold: {max_load:.1f}). "
                f"System is under load — benchmark results will "
                f"be unreliable."
            )
        elif load[0] > max_load * 0.7:
            result.warnings.append(
                f"1-minute load average is {load[0]:.1f} "
                f"(approaching threshold of {max_load:.1f})."
            )
    except Exception:  # noqa: BLE001
        result.warnings.append("Could not read load average.")

    # Available RAM — platform-specific.
    avail_gb = _get_available_ram_gb()
    if avail_gb is not None:
        if avail_gb < min_available_ram_gb:
            result.stable = False
            result.errors.append(
                f"Available RAM is {avail_gb:.1f} GB "
                f"(minimum: {min_available_ram_gb:.1f} GB). "
                f"Low memory may cause swapping and "
                f"unreliable timings."
            )
    else:
        result.warnings.append("Could not read memory info.")

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def format_system_profile(profile: SystemProfile) -> str:
    """Format a system profile for terminal display."""
    lines = [
        "System Profile",
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    ]

    # CPU
    cores = f"{profile.cpu_cores_physical} cores"
    if profile.cpu_cores_logical != profile.cpu_cores_physical:
        cores += f" / {profile.cpu_cores_logical} threads"
    freq = ""
    if profile.cpu_freq_mhz:
        freq = f", {profile.cpu_freq_mhz:.0f} MHz"
        if profile.cpu_freq_max_mhz:
            freq += f" (max {profile.cpu_freq_max_mhz:.0f} MHz)"
    lines.append(f"CPU:      {profile.cpu_model} ({cores}{freq})")

    # Memory
    lines.append(
        f"RAM:      {profile.ram_total_gb:.1f} GB total, "
        f"{profile.ram_available_gb:.1f} GB available"
    )
    if profile.swap_total_gb > 0:
        lines.append(f"Swap:     {profile.swap_total_gb:.1f} GB")

    # OS — macOS distro string is self-explanatory; no need for kernel version.
    if profile.os_name == "Darwin":
        lines.append(f"OS:       {profile.os_distro}")
    else:
        lines.append(
            f"OS:       {profile.os_distro} ({profile.os_name} {profile.os_kernel_version})"
        )

    # Disk
    lines.append(
        f"Disk:     {profile.disk_type.upper()}, {profile.disk_available_gb:.0f} GB available"
    )

    # Load
    lines.append(
        f"Load:     {profile.load_avg_1m} / {profile.load_avg_5m} / {profile.load_avg_15m}"
    )

    # Host Python
    lines.append(
        f"Host:     Python {profile.host_python_version} ({profile.host_python_implementation})"
    )

    lines.append(f"Hostname: {profile.hostname}")
    lines.append(f"Time:     {profile.timestamp}")

    return "\n".join(lines)


def format_python_profile(profile: PythonProfile) -> str:
    """Format a Python profile for terminal display."""
    lines = [f"Python:   {profile.version} ({profile.implementation})"]
    lines.append(f"Compiler: {profile.compiler}")

    flags: list[str] = []
    if profile.jit_enabled:
        flags.append("JIT enabled")
    elif profile.jit_available:
        flags.append("JIT available (disabled)")
    if profile.gil_disabled:
        flags.append("GIL disabled (free-threaded)")
    if profile.debug_build:
        flags.append("debug build")
    if flags:
        lines.append(f"Flags:    {', '.join(flags)}")

    if profile.build_flags:
        # Show the most interesting build flags.
        interesting = [
            f
            for f in profile.build_flags
            if any(
                kw in f
                for kw in (
                    "jit",
                    "gil",
                    "debug",
                    "optimiz",
                    "lto",
                    "asan",
                    "msan",
                    "tsan",
                    "ubsan",
                )
            )
        ]
        if interesting:
            lines.append(f"Build:    {' '.join(interesting)}")

    return "\n".join(lines)
