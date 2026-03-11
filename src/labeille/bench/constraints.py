"""Resource constraints for benchmark execution.

Applies per-process resource limits using ulimit (portable across
Linux and macOS) and CPU affinity via taskset (Linux only).

Constraints are specified per-condition in the ConditionDef, allowing
A/B comparison under different resource limits. When a process exceeds
a memory limit, it is killed by the OS — this is the intended behavior
for benchmarking, as it clearly identifies packages that cannot run
under the constraint.

No root privileges are required. ulimit operates on the current process
and its children. taskset requires the target CPUs to be available to
the current user (which is the default).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any

from labeille.logging import get_logger

log = get_logger("bench.constraints")


@dataclass
class ResourceConstraints:
    """Resource limits applied during benchmark iteration execution.

    All limits are optional. When set, they're applied via ulimit
    (memory, CPU time) and taskset (CPU affinity).

    This is part of the condition abstraction: different conditions
    can have different constraints.
    """

    # Memory: virtual memory limit in MB.
    # Maps to ulimit -v (in KB). When exceeded, malloc returns NULL
    # or the process gets SIGSEGV/SIGKILL depending on the OS.
    memory_limit_mb: int | None = None

    # CPU time limit in seconds.
    # Maps to ulimit -t. Sends SIGXCPU when exceeded, then SIGKILL.
    cpu_time_limit_s: int | None = None

    # File size limit in MB.
    # Maps to ulimit -f (in 512-byte blocks). Prevents runaway disk usage.
    file_size_limit_mb: int | None = None

    # CPU affinity: list of CPU core indices to pin execution to.
    # Uses taskset on Linux. Ignored on macOS (with a warning).
    cpu_affinity: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (sparse: omits None values)."""
        d: dict[str, Any] = {}
        if self.memory_limit_mb is not None:
            d["memory_limit_mb"] = self.memory_limit_mb
        if self.cpu_time_limit_s is not None:
            d["cpu_time_limit_s"] = self.cpu_time_limit_s
        if self.file_size_limit_mb is not None:
            d["file_size_limit_mb"] = self.file_size_limit_mb
        if self.cpu_affinity is not None:
            d["cpu_affinity"] = self.cpu_affinity
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceConstraints:
        """Deserialize from a dict, ignoring unknown fields."""
        from labeille.io_utils import dataclass_from_dict

        return dataclass_from_dict(cls, data)

    @property
    def has_any(self) -> bool:
        """True if any constraint is set."""
        return any(
            [
                self.memory_limit_mb is not None,
                self.cpu_time_limit_s is not None,
                self.file_size_limit_mb is not None,
                self.cpu_affinity is not None,
            ]
        )

    @property
    def has_cpu_affinity(self) -> bool:
        """True if CPU affinity is set with at least one core."""
        return self.cpu_affinity is not None and len(self.cpu_affinity) > 0


def build_ulimit_prefix(constraints: ResourceConstraints) -> str:
    """Build a shell ulimit prefix string for the given constraints.

    Generates ulimit flags that set resource limits before executing
    the test command.

    Args:
        constraints: Resource constraints to apply.

    Returns:
        A shell prefix string. Empty string if no ulimit constraints.
        The returned string ends with ``"; exec "`` — the caller
        appends the actual command.
    """
    parts: list[str] = []

    if constraints.memory_limit_mb is not None:
        parts.append(f"ulimit -v {constraints.memory_limit_mb * 1024}")
    if constraints.cpu_time_limit_s is not None:
        parts.append(f"ulimit -t {constraints.cpu_time_limit_s}")
    if constraints.file_size_limit_mb is not None:
        parts.append(f"ulimit -f {constraints.file_size_limit_mb * 2048}")

    if not parts:
        return ""

    return "; ".join(parts) + "; exec "


def build_taskset_prefix(constraints: ResourceConstraints) -> str:
    """Build a taskset prefix for CPU affinity.

    Only works on Linux. Returns empty string on other platforms
    or if taskset is not available.

    Args:
        constraints: Resource constraints with cpu_affinity set.

    Returns:
        A command prefix like ``"taskset -c 0,1 "`` or empty string.
    """
    if not constraints.has_cpu_affinity:
        return ""

    if sys.platform != "linux":
        return ""

    if not shutil.which("taskset"):
        log.warning("taskset not found, CPU affinity ignored.")
        return ""

    assert constraints.cpu_affinity is not None  # for type checker
    cores = ",".join(str(c) for c in constraints.cpu_affinity)
    return f"taskset -c {cores} "


def apply_constraints_to_command(
    command: str,
    constraints: ResourceConstraints | None,
) -> str:
    """Wrap a command with resource constraints.

    Applies taskset prefix (if applicable) and ulimit wrapper
    (if applicable) to the given command string.

    Order: taskset wraps the ulimit-wrapped command, so CPU affinity
    applies to the shell that sets ulimits.

    Args:
        command: The original shell command.
        constraints: Resource constraints to apply, or None.

    Returns:
        The (possibly wrapped) command string.
    """
    if constraints is None or not constraints.has_any:
        return command

    result = command

    ulimit = build_ulimit_prefix(constraints)
    if ulimit:
        result = f"bash -c '{ulimit}{result}'"

    taskset = build_taskset_prefix(constraints)
    if taskset:
        result = f"{taskset}{result}"

    return result


def check_constraints_available(
    constraints: ResourceConstraints,
) -> list[str]:
    """Check which constraints are available on this platform.

    Returns a list of warning messages for constraints that cannot
    be applied (e.g., CPU affinity on macOS). Empty list means all
    constraints are available.
    """
    warnings: list[str] = []

    if constraints.has_cpu_affinity:
        if sys.platform != "linux":
            warnings.append(
                "CPU affinity (taskset) is only available on Linux. Affinity will be ignored."
            )
        elif not shutil.which("taskset"):
            warnings.append(
                "taskset not found. Install util-linux or CPU affinity will be ignored."
            )

        # Validate core indices.
        cpu_count = os.cpu_count() or 1
        assert constraints.cpu_affinity is not None  # for type checker
        invalid = [c for c in constraints.cpu_affinity if c >= cpu_count]
        if invalid:
            warnings.append(
                f"CPU core indices {invalid} are >= cpu_count ({cpu_count}). "
                f"These cores may not exist."
            )

    return warnings


def detect_oom_from_result(
    exit_code: int,
    stderr: str,
    constraints: ResourceConstraints | None,
) -> bool:
    """Detect whether a process was killed due to OOM from resource limits.

    Heuristics:
    - Exit code -9 (SIGKILL) with memory_limit_mb set.
    - Exit code -11 (SIGSEGV) with memory_limit_mb set (malloc returns NULL).
    - "Cannot allocate memory" or "MemoryError" in stderr.

    Args:
        exit_code: Process exit code.
        stderr: Process stderr output.
        constraints: The constraints that were applied.

    Returns:
        True if the process likely died from memory limits.
    """
    if constraints is None:
        return False

    has_memory_limit = constraints.memory_limit_mb is not None

    # Signal-based detection (only with memory limit).
    if has_memory_limit and exit_code in (-9, -11):
        return True

    # Stderr-based detection (works with or without explicit limit).
    if has_memory_limit:
        stderr_lower = stderr.lower()
        if "cannot allocate memory" in stderr_lower:
            return True
        if "memoryerror" in stderr_lower:
            return True

    return False
