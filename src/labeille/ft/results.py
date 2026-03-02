"""Result dataclasses for free-threading compatibility testing.

Three levels of results:
- IterationOutcome: a single run of a package's test suite.
- FTPackageResult: all iterations for one package, with aggregate
  statistics and a compatibility category.
- FTRunResult: all packages in a complete test run, with metadata
  and summary statistics.

Serialization uses JSONL (one line per package) for incremental
writing during long runs, plus JSON for metadata and summaries.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("labeille")


class FailureCategory(enum.Enum):
    """Classification of a package's free-threading compatibility.

    Ordered roughly by severity, from fully working to untestable.
    """

    COMPATIBLE = "compatible"
    COMPATIBLE_GIL_FALLBACK = "compatible_gil_fallback"
    INTERMITTENT = "intermittent"
    INCOMPATIBLE = "incompatible"
    CRASH = "crash"
    DEADLOCK = "deadlock"
    TSAN_WARNINGS = "tsan_warnings"
    INSTALL_FAILURE = "install_failure"
    IMPORT_FAILURE = "import_failure"
    UNKNOWN = "unknown"

    @property
    def is_usable(self) -> bool:
        """True if the package at least works sometimes."""
        return self in (
            FailureCategory.COMPATIBLE,
            FailureCategory.COMPATIBLE_GIL_FALLBACK,
            FailureCategory.INTERMITTENT,
            FailureCategory.TSAN_WARNINGS,
        )

    @property
    def severity(self) -> int:
        """Numeric severity for sorting (higher = worse)."""
        order = {
            FailureCategory.COMPATIBLE: 0,
            FailureCategory.COMPATIBLE_GIL_FALLBACK: 1,
            FailureCategory.TSAN_WARNINGS: 2,
            FailureCategory.INTERMITTENT: 3,
            FailureCategory.INCOMPATIBLE: 4,
            FailureCategory.CRASH: 5,
            FailureCategory.DEADLOCK: 6,
            FailureCategory.INSTALL_FAILURE: 7,
            FailureCategory.IMPORT_FAILURE: 8,
            FailureCategory.UNKNOWN: 9,
        }
        return order.get(self, 9)

    @property
    def symbol(self) -> str:
        """Single-character symbol for compact display."""
        symbols = {
            FailureCategory.COMPATIBLE: "\u2713",
            FailureCategory.COMPATIBLE_GIL_FALLBACK: "\u26a0",
            FailureCategory.INTERMITTENT: "~",
            FailureCategory.INCOMPATIBLE: "\u2717",
            FailureCategory.CRASH: "\U0001f4a5",
            FailureCategory.DEADLOCK: "\U0001f512",
            FailureCategory.TSAN_WARNINGS: "\u26a1",
            FailureCategory.INSTALL_FAILURE: "\u2699",
            FailureCategory.IMPORT_FAILURE: "\U0001f4e6",
            FailureCategory.UNKNOWN: "?",
        }
        return symbols.get(self, "?")


@dataclass
class IterationOutcome:
    """Result of a single test suite execution.

    Captures the status, timing, crash info, deadlock detection,
    and TSAN warnings for one iteration.
    """

    index: int
    status: str  # pass, fail, crash, timeout, deadlock
    exit_code: int | None = None
    duration_s: float = 0.0
    crash_signal: str | None = None
    crash_signature: str | None = None
    tsan_warnings: list[str] = field(default_factory=list)
    output_stalled: bool = False
    last_output_line: str = ""
    stderr_tail: str = ""
    test_results: dict[str, str] = field(default_factory=dict)
    tests_passed: int | None = None
    tests_failed: int | None = None
    tests_errored: int | None = None
    tests_skipped: int | None = None

    @property
    def is_pass(self) -> bool:
        return self.status == "pass"

    @property
    def is_crash(self) -> bool:
        return self.status == "crash"

    @property
    def is_deadlock(self) -> bool:
        return self.status == "deadlock"

    @property
    def is_timeout(self) -> bool:
        return self.status in ("timeout", "deadlock")

    @property
    def has_tsan_warnings(self) -> bool:
        return len(self.tsan_warnings) > 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["test_results"]:
            del d["test_results"]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IterationOutcome:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class FTPackageResult:
    """Aggregate free-threading test results for one package.

    Contains all iteration outcomes, aggregate statistics, extension
    compatibility data, and the final category classification.
    """

    package: str
    category: FailureCategory = FailureCategory.UNKNOWN
    iterations: list[IterationOutcome] = field(default_factory=list)

    iterations_completed: int = 0
    pass_count: int = 0
    fail_count: int = 0
    crash_count: int = 0
    timeout_count: int = 0
    deadlock_count: int = 0
    tsan_warning_iterations: int = 0

    pass_rate: float = 0.0
    mean_duration_s: float = 0.0

    extension_compat: dict[str, Any] | None = None

    failure_signatures: list[str] = field(default_factory=list)
    tsan_warning_types: list[str] = field(default_factory=list)
    flaky_tests: dict[str, int] = field(default_factory=dict)

    install_ok: bool = True
    install_error: str | None = None
    install_duration_s: float = 0.0
    import_ok: bool = True
    import_error: str | None = None
    commit: str | None = None

    gil_enabled_pass_rate: float | None = None
    gil_enabled_iterations: list[IterationOutcome] | None = None

    def compute_aggregates(self) -> None:
        """Recompute aggregate statistics from iterations.

        Call this after all iterations are added.
        """
        if not self.iterations:
            return

        self.iterations_completed = len(self.iterations)
        self.pass_count = sum(1 for i in self.iterations if i.is_pass)
        self.crash_count = sum(1 for i in self.iterations if i.is_crash)
        self.deadlock_count = sum(1 for i in self.iterations if i.is_deadlock)
        self.timeout_count = sum(1 for i in self.iterations if i.is_timeout)
        self.fail_count = sum(1 for i in self.iterations if i.status == "fail")
        self.tsan_warning_iterations = sum(1 for i in self.iterations if i.has_tsan_warnings)

        if self.iterations_completed > 0:
            self.pass_rate = self.pass_count / self.iterations_completed
        else:
            self.pass_rate = 0.0

        durations = [i.duration_s for i in self.iterations if i.duration_s > 0]
        if durations:
            self.mean_duration_s = sum(durations) / len(durations)

        sigs: set[str] = set()
        for i in self.iterations:
            if i.crash_signature:
                sigs.add(i.crash_signature)
        self.failure_signatures = sorted(sigs)

        tsan_types: set[str] = set()
        for i in self.iterations:
            for w in i.tsan_warnings:
                tsan_types.add(w)
        self.tsan_warning_types = sorted(tsan_types)

        test_failures: dict[str, int] = {}
        test_total: dict[str, int] = {}
        for i in self.iterations:
            for test_name, status in i.test_results.items():
                test_total[test_name] = test_total.get(test_name, 0) + 1
                if status in ("FAILED", "ERROR"):
                    test_failures[test_name] = test_failures.get(test_name, 0) + 1
        self.flaky_tests = {
            name: count
            for name, count in sorted(test_failures.items(), key=lambda x: -x[1])
            if count < test_total.get(name, 0)
        }

        if self.gil_enabled_iterations:
            gil_passes = sum(1 for i in self.gil_enabled_iterations if i.is_pass)
            self.gil_enabled_pass_rate = gil_passes / len(self.gil_enabled_iterations)

    def categorize(self) -> FailureCategory:
        """Determine the failure category based on aggregates.

        Call compute_aggregates() first.

        Returns:
            The FailureCategory, also stored in self.category.
        """
        self.category = categorize_package(self)
        return self.category

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "package": self.package,
            "category": self.category.value,
            "iterations_completed": self.iterations_completed,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "crash_count": self.crash_count,
            "timeout_count": self.timeout_count,
            "deadlock_count": self.deadlock_count,
            "tsan_warning_iterations": self.tsan_warning_iterations,
            "pass_rate": round(self.pass_rate, 4),
            "mean_duration_s": round(self.mean_duration_s, 2),
            "failure_signatures": self.failure_signatures,
            "tsan_warning_types": self.tsan_warning_types,
            "flaky_tests": self.flaky_tests,
            "install_ok": self.install_ok,
            "install_error": self.install_error,
            "install_duration_s": round(self.install_duration_s, 2),
            "import_ok": self.import_ok,
            "import_error": self.import_error,
            "commit": self.commit,
            "iterations": [i.to_dict() for i in self.iterations],
        }
        if self.extension_compat is not None:
            d["extension_compat"] = self.extension_compat
        if self.gil_enabled_pass_rate is not None:
            d["gil_enabled_pass_rate"] = round(self.gil_enabled_pass_rate, 4)
        if self.gil_enabled_iterations is not None:
            d["gil_enabled_iterations"] = [i.to_dict() for i in self.gil_enabled_iterations]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FTPackageResult:
        iterations = [IterationOutcome.from_dict(i) for i in data.get("iterations", [])]
        gil_iters = None
        if "gil_enabled_iterations" in data:
            gil_iters = [IterationOutcome.from_dict(i) for i in data["gil_enabled_iterations"]]

        return cls(
            package=data["package"],
            category=FailureCategory(data.get("category", "unknown")),
            iterations=iterations,
            iterations_completed=data.get("iterations_completed", 0),
            pass_count=data.get("pass_count", 0),
            fail_count=data.get("fail_count", 0),
            crash_count=data.get("crash_count", 0),
            timeout_count=data.get("timeout_count", 0),
            deadlock_count=data.get("deadlock_count", 0),
            tsan_warning_iterations=data.get("tsan_warning_iterations", 0),
            pass_rate=data.get("pass_rate", 0.0),
            mean_duration_s=data.get("mean_duration_s", 0.0),
            extension_compat=data.get("extension_compat"),
            failure_signatures=data.get("failure_signatures", []),
            tsan_warning_types=data.get("tsan_warning_types", []),
            flaky_tests=data.get("flaky_tests", {}),
            install_ok=data.get("install_ok", True),
            install_error=data.get("install_error"),
            install_duration_s=data.get("install_duration_s", 0.0),
            import_ok=data.get("import_ok", True),
            import_error=data.get("import_error"),
            commit=data.get("commit"),
            gil_enabled_pass_rate=data.get("gil_enabled_pass_rate"),
            gil_enabled_iterations=gil_iters,
        )


def categorize_package(result: FTPackageResult) -> FailureCategory:
    """Classify a package's free-threading compatibility.

    The classification considers installation status, pass rates,
    crash/deadlock presence, TSAN warnings, and GIL fallback.

    Classification priority (checked in order):
    1. install_ok=False -> INSTALL_FAILURE
    2. import_ok=False -> IMPORT_FAILURE
    3. No iterations completed -> UNKNOWN
    4. Any deadlocks -> DEADLOCK
    5. Any crashes -> CRASH
    6. All pass + TSAN warnings -> TSAN_WARNINGS
    7. All pass + GIL fallback -> COMPATIBLE_GIL_FALLBACK
    8. All pass -> COMPATIBLE
    9. All fail -> INCOMPATIBLE
    10. Mixed -> INTERMITTENT
    """
    if not result.install_ok:
        return FailureCategory.INSTALL_FAILURE

    if not result.import_ok:
        return FailureCategory.IMPORT_FAILURE

    if result.iterations_completed == 0:
        return FailureCategory.UNKNOWN

    if result.deadlock_count > 0:
        return FailureCategory.DEADLOCK

    if result.crash_count > 0:
        return FailureCategory.CRASH

    if result.pass_rate == 1.0:
        if result.tsan_warning_iterations > 0:
            return FailureCategory.TSAN_WARNINGS

        ext = result.extension_compat
        if ext and ext.get("gil_fallback_active", False):
            return FailureCategory.COMPATIBLE_GIL_FALLBACK

        return FailureCategory.COMPATIBLE

    if result.pass_rate == 0.0:
        return FailureCategory.INCOMPATIBLE

    return FailureCategory.INTERMITTENT


# ---------------------------------------------------------------------------
# Run metadata and summary
# ---------------------------------------------------------------------------


@dataclass
class FTRunMeta:
    """Metadata for a complete free-threading test run."""

    run_id: str
    timestamp: str
    system_profile: dict[str, Any] = field(default_factory=dict)
    python_profile: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    cli_args: list[str] = field(default_factory=list)
    packages_total: int = 0
    packages_completed: int = 0
    total_iterations: int = 0
    total_duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FTRunMeta:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FTRunSummary:
    """Aggregate summary of a free-threading test run."""

    total_packages: int = 0
    categories: dict[str, int] = field(default_factory=dict)
    pass_rate_distribution: dict[str, int] = field(default_factory=dict)
    pure_python_count: int = 0
    extension_count: int = 0
    pure_python_compatible_pct: float = 0.0
    extension_compatible_pct: float = 0.0
    tsan_warning_count: int = 0

    @classmethod
    def compute(cls, results: list[FTPackageResult]) -> FTRunSummary:
        """Compute summary from a list of package results."""
        summary = cls(total_packages=len(results))

        for r in results:
            cat = r.category.value
            summary.categories[cat] = summary.categories.get(cat, 0) + 1

        buckets = {"100%": 0, "90-99%": 0, "50-89%": 0, "1-49%": 0, "0%": 0, "N/A": 0}
        for r in results:
            if r.iterations_completed == 0:
                buckets["N/A"] += 1
            elif r.pass_rate == 1.0:
                buckets["100%"] += 1
            elif r.pass_rate >= 0.9:
                buckets["90-99%"] += 1
            elif r.pass_rate >= 0.5:
                buckets["50-89%"] += 1
            elif r.pass_rate > 0.0:
                buckets["1-49%"] += 1
            else:
                buckets["0%"] += 1
        summary.pass_rate_distribution = buckets

        pure_python: list[FTPackageResult] = []
        extensions: list[FTPackageResult] = []
        for r in results:
            ext = r.extension_compat
            if ext and not ext.get("is_pure_python", True):
                extensions.append(r)
            else:
                pure_python.append(r)

        summary.pure_python_count = len(pure_python)
        summary.extension_count = len(extensions)

        if pure_python:
            pp_compat = sum(
                1
                for r in pure_python
                if r.category
                in (
                    FailureCategory.COMPATIBLE,
                    FailureCategory.TSAN_WARNINGS,
                )
            )
            summary.pure_python_compatible_pct = round(pp_compat / len(pure_python) * 100, 1)

        if extensions:
            ext_compat = sum(
                1
                for r in extensions
                if r.category
                in (
                    FailureCategory.COMPATIBLE,
                    FailureCategory.COMPATIBLE_GIL_FALLBACK,
                    FailureCategory.TSAN_WARNINGS,
                )
            )
            summary.extension_compatible_pct = round(ext_compat / len(extensions) * 100, 1)

        summary.tsan_warning_count = sum(1 for r in results if r.tsan_warning_iterations > 0)

        return summary

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FTRunSummary:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Serialization and I/O
# ---------------------------------------------------------------------------


def save_ft_run(
    results_dir: Path,
    meta: FTRunMeta,
    results: list[FTPackageResult],
) -> None:
    """Save a complete free-threading test run to disk.

    Creates:
      {results_dir}/ft_meta.json       -- run metadata
      {results_dir}/ft_results.jsonl   -- per-package results
      {results_dir}/ft_summary.json    -- aggregate summary

    Args:
        results_dir: Directory to write results into (created if
            it doesn't exist).
        meta: Run metadata.
        results: List of package results.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    meta.packages_completed = len(results)
    meta_path = results_dir / "ft_meta.json"
    meta_path.write_text(
        json.dumps(meta.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    results_path = results_dir / "ft_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")

    summary = FTRunSummary.compute(results)
    summary_path = results_dir / "ft_summary.json"
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    log.info(
        "Saved free-threading results: %d packages \u2192 %s",
        len(results),
        results_dir,
    )


def append_ft_result(
    results_dir: Path,
    result: FTPackageResult,
) -> None:
    """Append a single package result to the JSONL file.

    Used for incremental writing during long runs so results
    survive interruption.
    """
    results_path = results_dir / "ft_results.jsonl"
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


def load_ft_run(
    results_dir: Path,
) -> tuple[FTRunMeta, list[FTPackageResult]]:
    """Load a complete free-threading test run from disk.

    Returns:
        (meta, results) tuple.

    Raises:
        FileNotFoundError: if the results directory doesn't exist
            or is missing required files.
    """
    meta_path = results_dir / "ft_meta.json"
    results_path = results_dir / "ft_results.jsonl"

    if not meta_path.exists():
        raise FileNotFoundError(f"No ft_meta.json in {results_dir}")
    if not results_path.exists():
        raise FileNotFoundError(f"No ft_results.jsonl in {results_dir}")

    meta = FTRunMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))

    results: list[FTPackageResult] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(FTPackageResult.from_dict(json.loads(line)))

    return meta, results


def load_ft_summary(results_dir: Path) -> FTRunSummary:
    """Load the summary file, or recompute from results."""
    summary_path = results_dir / "ft_summary.json"
    if summary_path.exists():
        return FTRunSummary.from_dict(json.loads(summary_path.read_text(encoding="utf-8")))
    _, results = load_ft_run(results_dir)
    return FTRunSummary.compute(results)
