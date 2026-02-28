"""Benchmark result data structures and serialization.

Hierarchy::

    BenchMeta (top level — one benchmark execution)
      → conditions: dict[str, ConditionDef]
      → python_profiles: dict[str, PythonProfile]
      → system: SystemProfile

    BenchPackageResult (per package)
      → conditions: dict[str, BenchConditionResult]
        → iterations: list[BenchIteration]
        → wall_time_stats / user_time_stats / … : DescriptiveStats

Files produced::

    bench_meta.json      — BenchMeta (system, config, conditions)
    bench_results.jsonl   — one BenchPackageResult per line
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labeille.bench.stats import DescriptiveStats, describe, detect_outliers
from labeille.bench.system import PythonProfile, SystemProfile

log = logging.getLogger("labeille")


# ---------------------------------------------------------------------------
# Iteration-level result
# ---------------------------------------------------------------------------


@dataclass
class BenchIteration:
    """Result of a single test suite execution."""

    index: int  # 1-based iteration number
    warmup: bool  # True if this is a warm-up iteration
    wall_time_s: float
    user_time_s: float
    sys_time_s: float
    peak_rss_mb: float
    exit_code: int
    status: str  # "ok", "fail", "timeout", "error"
    outlier: bool = False  # Set by post-processing
    load_avg_start: float = 0.0
    load_avg_end: float = 0.0
    ram_available_start_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "index": self.index,
            "warmup": self.warmup,
            "wall_time_s": round(self.wall_time_s, 6),
            "user_time_s": round(self.user_time_s, 6),
            "sys_time_s": round(self.sys_time_s, 6),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "exit_code": self.exit_code,
            "status": self.status,
            "outlier": self.outlier,
            "load_avg_start": self.load_avg_start,
            "load_avg_end": self.load_avg_end,
            "ram_available_start_gb": self.ram_available_start_gb,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchIteration:
        """Deserialize from a dict, ignoring unknown fields."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Condition-level result (per package per condition)
# ---------------------------------------------------------------------------


@dataclass
class BenchConditionResult:
    """Results for one condition applied to one package."""

    condition_name: str
    iterations: list[BenchIteration] = field(default_factory=list)
    # Stats computed from measured (non-warmup) iterations:
    wall_time_stats: DescriptiveStats | None = None
    user_time_stats: DescriptiveStats | None = None
    sys_time_stats: DescriptiveStats | None = None
    peak_rss_stats: DescriptiveStats | None = None
    # Setup timing (not part of the benchmark):
    install_duration_s: float = 0.0
    venv_setup_duration_s: float = 0.0

    def compute_stats(self) -> None:
        """Compute summary statistics from measured iterations.

        Call this after all iterations are complete.  Filters out
        warm-up iterations and marks outliers.
        """
        measured = [it for it in self.iterations if not it.warmup]
        if not measured:
            return

        wall_times = [it.wall_time_s for it in measured]
        user_times = [it.user_time_s for it in measured]
        sys_times = [it.sys_time_s for it in measured]
        rss_values = [it.peak_rss_mb for it in measured]

        # Detect and mark outliers on wall time.
        outlier_flags = detect_outliers(wall_times)
        for it, is_outlier in zip(measured, outlier_flags):
            it.outlier = is_outlier

        self.wall_time_stats = describe(wall_times)
        self.user_time_stats = describe(user_times)
        self.sys_time_stats = describe(sys_times)
        self.peak_rss_stats = describe(rss_values)

    @property
    def measured_iterations(self) -> list[BenchIteration]:
        """Non-warmup iterations."""
        return [it for it in self.iterations if not it.warmup]

    @property
    def wall_times(self) -> list[float]:
        """Wall times from measured iterations."""
        return [it.wall_time_s for it in self.measured_iterations]

    @property
    def n_outliers(self) -> int:
        """Number of measured iterations flagged as outliers."""
        return sum(1 for it in self.measured_iterations if it.outlier)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "condition_name": self.condition_name,
            "iterations": [it.to_dict() for it in self.iterations],
            "install_duration_s": round(self.install_duration_s, 2),
            "venv_setup_duration_s": round(self.venv_setup_duration_s, 2),
        }
        if self.wall_time_stats:
            d["wall_time_stats"] = self.wall_time_stats.to_dict()
        if self.user_time_stats:
            d["user_time_stats"] = self.user_time_stats.to_dict()
        if self.sys_time_stats:
            d["sys_time_stats"] = self.sys_time_stats.to_dict()
        if self.peak_rss_stats:
            d["peak_rss_stats"] = self.peak_rss_stats.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchConditionResult:
        """Deserialize from a dict.  Recomputes stats for consistency."""
        result = cls(condition_name=data["condition_name"])
        result.iterations = [BenchIteration.from_dict(it) for it in data.get("iterations", [])]
        result.install_duration_s = data.get("install_duration_s", 0.0)
        result.venv_setup_duration_s = data.get("venv_setup_duration_s", 0.0)
        # Recompute stats rather than deserializing — ensures consistency.
        result.compute_stats()
        return result


# ---------------------------------------------------------------------------
# Package-level result
# ---------------------------------------------------------------------------


@dataclass
class BenchPackageResult:
    """Benchmark results for one package across all conditions."""

    package: str
    conditions: dict[str, BenchConditionResult] = field(default_factory=dict)
    clone_duration_s: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "package": self.package,
            "conditions": {name: cond.to_dict() for name, cond in self.conditions.items()},
            "clone_duration_s": round(self.clone_duration_s, 2),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchPackageResult:
        """Deserialize from a dict."""
        result = cls(
            package=data["package"],
            clone_duration_s=data.get("clone_duration_s", 0.0),
            skipped=data.get("skipped", False),
            skip_reason=data.get("skip_reason", ""),
        )
        for name, cond_data in data.get("conditions", {}).items():
            result.conditions[name] = BenchConditionResult.from_dict(cond_data)
        return result

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSONL line."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_jsonl_line(cls, line: str) -> BenchPackageResult:
        """Deserialize from a single JSONL line."""
        return cls.from_dict(json.loads(line))


# ---------------------------------------------------------------------------
# Condition definition (for bench_meta.json)
# ---------------------------------------------------------------------------


@dataclass
class ConditionDef:
    """Definition of a benchmark condition (stored in metadata)."""

    name: str
    description: str = ""
    target_python: str = ""
    env: dict[str, str] = field(default_factory=dict)
    extra_deps: list[str] = field(default_factory=list)
    test_command_override: str | None = None
    test_command_prefix: str | None = None
    test_command_suffix: str | None = None
    install_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (sparse: omits defaults)."""
        d: dict[str, Any] = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.target_python:
            d["target_python"] = self.target_python
        if self.env:
            d["env"] = self.env
        if self.extra_deps:
            d["extra_deps"] = self.extra_deps
        if self.test_command_override:
            d["test_command_override"] = self.test_command_override
        if self.test_command_prefix:
            d["test_command_prefix"] = self.test_command_prefix
        if self.test_command_suffix:
            d["test_command_suffix"] = self.test_command_suffix
        if self.install_command:
            d["install_command"] = self.install_command
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionDef:
        """Deserialize from a dict."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            target_python=data.get("target_python", ""),
            env=data.get("env", {}),
            extra_deps=data.get("extra_deps", []),
            test_command_override=data.get("test_command_override"),
            test_command_prefix=data.get("test_command_prefix"),
            test_command_suffix=data.get("test_command_suffix"),
            install_command=data.get("install_command"),
        )


# ---------------------------------------------------------------------------
# Run-level metadata
# ---------------------------------------------------------------------------


@dataclass
class BenchMeta:
    """Metadata for a complete benchmark run."""

    bench_id: str
    name: str = ""
    description: str = ""
    system: SystemProfile = field(default_factory=SystemProfile)
    python_profiles: dict[str, PythonProfile] = field(default_factory=dict)
    conditions: dict[str, ConditionDef] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    cli_args: list[str] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    packages_total: int = 0
    packages_completed: int = 0
    packages_skipped: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "bench_id": self.bench_id,
            "name": self.name,
            "description": self.description,
            "system": self.system.to_dict(),
            "python_profiles": {name: pp.to_dict() for name, pp in self.python_profiles.items()},
            "conditions": {name: cond.to_dict() for name, cond in self.conditions.items()},
            "config": self.config,
            "cli_args": self.cli_args,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "packages_total": self.packages_total,
            "packages_completed": self.packages_completed,
            "packages_skipped": self.packages_skipped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchMeta:
        """Deserialize from a dict."""
        meta = cls(bench_id=data["bench_id"])
        meta.name = data.get("name", "")
        meta.description = data.get("description", "")
        meta.system = SystemProfile.from_dict(data.get("system", {}))
        meta.python_profiles = {
            name: PythonProfile.from_dict(pp)
            for name, pp in data.get("python_profiles", {}).items()
        }
        meta.conditions = {
            name: ConditionDef.from_dict(cond) for name, cond in data.get("conditions", {}).items()
        }
        meta.config = data.get("config", {})
        meta.cli_args = data.get("cli_args", [])
        meta.start_time = data.get("start_time", "")
        meta.end_time = data.get("end_time", "")
        meta.packages_total = data.get("packages_total", 0)
        meta.packages_completed = data.get("packages_completed", 0)
        meta.packages_skipped = data.get("packages_skipped", 0)
        return meta


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------


def save_bench_run(
    output_dir: Path,
    meta: BenchMeta,
    results: list[BenchPackageResult],
) -> None:
    """Save a complete benchmark run to disk.

    Creates ``output_dir/bench_meta.json`` and
    ``output_dir/bench_results.jsonl``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "bench_meta.json"
    meta_path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n")
    log.info("Wrote %s", meta_path)

    results_path = output_dir / "bench_results.jsonl"
    with open(results_path, "w") as f:
        for result in results:
            f.write(result.to_jsonl_line() + "\n")
    log.info("Wrote %d package results to %s", len(results), results_path)


def load_bench_run(
    run_dir: Path,
) -> tuple[BenchMeta, list[BenchPackageResult]]:
    """Load a benchmark run from disk.

    Args:
        run_dir: Directory containing ``bench_meta.json`` and
            ``bench_results.jsonl``.

    Returns:
        Tuple of (BenchMeta, list of BenchPackageResult).

    Raises:
        FileNotFoundError: If ``bench_meta.json`` is missing.
    """
    meta_path = run_dir / "bench_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No bench_meta.json in {run_dir}")

    meta = BenchMeta.from_dict(json.loads(meta_path.read_text()))

    results: list[BenchPackageResult] = []
    results_path = run_dir / "bench_results.jsonl"
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            line = line.strip()
            if line:
                results.append(BenchPackageResult.from_jsonl_line(line))

    return meta, results


def append_package_result(
    results_path: Path,
    result: BenchPackageResult,
) -> None:
    """Append a single package result to the JSONL file.

    Used for incremental writing during long benchmark runs so
    that results are preserved if the process is interrupted.
    """
    with open(results_path, "a") as f:
        f.write(result.to_jsonl_line() + "\n")
