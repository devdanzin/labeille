"""Analysis and data loading for labeille run results.

Provides :class:`RunData` for lazily loading individual run data,
:class:`ResultsStore` for discovering and accessing runs, and pure
analysis functions that operate on loaded data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labeille.registry import PackageEntry


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunMeta:
    """Metadata for a single run, loaded from run_meta.json."""

    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    target_python: str = ""
    python_version: str = ""
    jit_enabled: bool = False
    hostname: str = ""
    platform: str = ""
    packages_tested: int = 0
    packages_skipped: int = 0
    crashes_found: int = 0
    total_duration_seconds: float = 0.0
    cli_args: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMeta:
        """Create a RunMeta from a JSON-loaded dict, tolerating missing keys."""
        return cls(
            run_id=data.get("run_id", ""),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
            target_python=data.get("target_python", ""),
            python_version=data.get("python_version", ""),
            jit_enabled=data.get("jit_enabled", False),
            hostname=data.get("hostname", ""),
            platform=data.get("platform", ""),
            packages_tested=data.get("packages_tested", 0),
            packages_skipped=data.get("packages_skipped", 0),
            crashes_found=data.get("crashes_found", 0),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            cli_args=data.get("cli_args", []),
            env_overrides=data.get("env_overrides", {}),
        )


@dataclass
class PackageResult:
    """Result of testing a single package, loaded from results.jsonl."""

    package: str = ""
    repo: str | None = None
    package_version: str | None = None
    git_revision: str | None = None
    status: str = "error"
    exit_code: int = -1
    signal: int | None = None
    crash_signature: str | None = None
    duration_seconds: float = 0.0
    install_duration_seconds: float = 0.0
    test_command: str = ""
    timeout_hit: bool = False
    stderr_tail: str = ""
    installed_dependencies: dict[str, str] = field(default_factory=dict)
    error_message: str | None = None
    timestamp: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackageResult:
        """Create a PackageResult from a JSON-loaded dict."""
        return cls(
            package=data.get("package", ""),
            repo=data.get("repo"),
            package_version=data.get("package_version"),
            git_revision=data.get("git_revision"),
            status=data.get("status", "error"),
            exit_code=data.get("exit_code", -1),
            signal=data.get("signal"),
            crash_signature=data.get("crash_signature"),
            duration_seconds=data.get("duration_seconds", 0.0),
            install_duration_seconds=data.get("install_duration_seconds", 0.0),
            test_command=data.get("test_command", ""),
            timeout_hit=data.get("timeout_hit", False),
            stderr_tail=data.get("stderr_tail", ""),
            installed_dependencies=data.get("installed_dependencies", {}),
            error_message=data.get("error_message"),
            timestamp=data.get("timestamp", ""),
        )


class RunData:
    """Lazily loaded data for a single run."""

    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self._meta: RunMeta | None = None
        self._results: list[PackageResult] | None = None
        self._results_by_pkg: dict[str, PackageResult] | None = None

    def __repr__(self) -> str:
        return f"RunData(run_id={self.run_id!r})"

    @property
    def meta(self) -> RunMeta:
        """Load run_meta.json on first access."""
        if self._meta is None:
            self._meta = self._load_meta()
        return self._meta

    @property
    def results(self) -> list[PackageResult]:
        """Load results.jsonl on first access."""
        if self._results is None:
            self._results = self._load_results()
        return self._results

    def result_for(self, package: str) -> PackageResult | None:
        """Find result for a specific package, or None. O(1) after first call."""
        if self._results_by_pkg is None:
            self._results_by_pkg = {r.package: r for r in self.results}
        return self._results_by_pkg.get(package)

    def results_by_status(self) -> dict[str, list[PackageResult]]:
        """Group results by status."""
        grouped: dict[str, list[PackageResult]] = {}
        for r in self.results:
            grouped.setdefault(r.status, []).append(r)
        return grouped

    def _load_meta(self) -> RunMeta:
        meta_file = self.run_dir / "run_meta.json"
        if not meta_file.exists():
            return RunMeta(run_id=self.run_id)
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        return RunMeta.from_dict(data)

    def _load_results(self) -> list[PackageResult]:
        results_file = self.run_dir / "results.jsonl"
        if not results_file.exists():
            return []
        results: list[PackageResult] = []
        for line in results_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    results.append(PackageResult.from_dict(data))
                except json.JSONDecodeError:
                    continue
        return results


def extract_minor_version(version_string: str) -> str:
    """Extract major.minor from a full Python version string.

    See also :func:`labeille.runner.extract_python_minor_version` which
    handles the same task in the runner context.
    """
    parts = version_string.strip().split(".")
    if len(parts) >= 2:
        minor = ""
        for ch in parts[1]:
            if ch.isdigit():
                minor += ch
            else:
                break
        if parts[0].isdigit() and minor:
            return f"{parts[0]}.{minor}"
    return version_string


class ResultsStore:
    """Discovers and provides access to run results."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir

    def list_runs(self, *, python_version: str | None = None) -> list[RunData]:
        """List all runs sorted newest first.

        If *python_version* is given, filter to runs with that Python minor
        version (e.g. ``'3.15'``).
        """
        if not self.results_dir.is_dir():
            return []
        runs: list[RunData] = []
        for d in sorted(self.results_dir.iterdir(), reverse=True):
            if d.is_dir() and (d / "run_meta.json").exists():
                runs.append(RunData(run_id=d.name, run_dir=d))

        if python_version is not None:
            filtered: list[RunData] = []
            for run in runs:
                pv = extract_minor_version(run.meta.python_version)
                if pv == python_version:
                    filtered.append(run)
            runs = filtered

        return runs

    def latest(self) -> RunData | None:
        """Return the most recent run, or None."""
        runs = self.list_runs()
        return runs[0] if runs else None

    def get(self, run_id: str) -> RunData | None:
        """Get a specific run. Accepts ``'latest'`` as alias."""
        if run_id == "latest":
            return self.latest()
        run_dir = self.results_dir / run_id
        if run_dir.is_dir() and (run_dir / "run_meta.json").exists():
            return RunData(run_id=run_id, run_dir=run_dir)
        return None

    def runs_for_package(self, package: str) -> list[tuple[RunData, PackageResult]]:
        """Find all runs that tested a specific package.

        Returns ``(run, result)`` pairs, newest first.
        """
        pairs: list[tuple[RunData, PackageResult]] = []
        for run in self.list_runs():
            result = run.result_for(package)
            if result is not None:
                pairs.append((run, result))
        return pairs


# ---------------------------------------------------------------------------
# Registry analysis
# ---------------------------------------------------------------------------


@dataclass
class RegistryStats:
    """Aggregate statistics about the registry."""

    total: int = 0
    active: int = 0
    skipped: int = 0
    by_extension_type: dict[str, tuple[int, int]] = field(default_factory=dict)
    by_skip_category: dict[str, int] = field(default_factory=dict)
    by_test_framework: dict[str, int] = field(default_factory=dict)
    notable: dict[str, int] = field(default_factory=dict)
    quality_warnings: list[tuple[str, str]] = field(default_factory=list)


def categorize_skip_reason(reason: str) -> str:
    """Categorize a skip reason into a human-readable bucket."""
    lower = reason.lower()
    if any(kw in lower for kw in ("pyo3", "rust", "maturin")):
        return "PyO3/Rust (no 3.15)"
    if any(kw in lower for kw in ("monorepo", "mono repo", "workspace")):
        return "Monorepo"
    if any(kw in lower for kw in ("no repo", "no source", "no url", "no repository")):
        return "No repo URL"
    if any(kw in lower for kw in ("c build", "c++", "meson", "cmake", "fortran", "header")):
        return "C/C++ build complexity"
    if any(kw in lower for kw in ("no test", "no tests")):
        return "No test suite"
    if any(kw in lower for kw in ("credential", "api key", "cloud", "auth")):
        return "Cloud/API credentials"
    if any(kw in lower for kw in ("jit crash", "crash during install", "segfault")):
        return "JIT crash during install"
    if any(kw in lower for kw in ("heavy", "large dep", "numpy", "pandas")):
        return "Heavy dependencies"
    return "Other"


def detect_quality_warnings(pkg: PackageEntry) -> list[str]:
    """Detect potential enrichment quality issues for a single package."""
    warnings: list[str] = []
    if not pkg.skip and pkg.enriched:
        if not pkg.install_command:
            warnings.append("active but empty install_command")
        if not pkg.test_command:
            warnings.append("active but empty test_command")
    if not pkg.skip and pkg.skip_reason:
        warnings.append("active but skip_reason is non-null")
    if pkg.skip and pkg.skip_versions:
        warnings.append("skip is true but skip_versions has entries (redundant)")
    return warnings


def analyze_registry(
    packages: list[PackageEntry],
    *,
    target_python_version: str | None = None,
) -> RegistryStats:
    """Analyze registry composition.

    If *target_python_version* is given, ``skip_versions`` entries for that
    version are counted as skipped.
    """
    stats = RegistryStats(total=len(packages))

    for pkg in packages:
        is_skipped = pkg.skip
        if (
            not is_skipped
            and target_python_version
            and pkg.skip_versions
            and target_python_version in pkg.skip_versions
        ):
            is_skipped = True

        if is_skipped:
            stats.skipped += 1
        else:
            stats.active += 1

        # Extension type breakdown.
        ext = pkg.extension_type or "unknown"
        active_count, skipped_count = stats.by_extension_type.get(ext, (0, 0))
        if is_skipped:
            stats.by_extension_type[ext] = (active_count, skipped_count + 1)
        else:
            stats.by_extension_type[ext] = (active_count + 1, skipped_count)

        # Skip reason categorization.
        if is_skipped:
            reason = pkg.skip_reason or ""
            if target_python_version and pkg.skip_versions:
                ver_reason = pkg.skip_versions.get(target_python_version)
                if ver_reason:
                    reason = ver_reason
            if reason:
                cat = categorize_skip_reason(reason)
            else:
                cat = "Other"
            stats.by_skip_category[cat] = stats.by_skip_category.get(cat, 0) + 1

        # Test framework (active only).
        if not is_skipped:
            fw = pkg.test_framework or "unknown"
            stats.by_test_framework[fw] = stats.by_test_framework.get(fw, 0) + 1

        # Notable attributes (active only).
        if not is_skipped:
            if pkg.timeout is not None:
                stats.notable["Custom timeout"] = stats.notable.get("Custom timeout", 0) + 1
            if pkg.clone_depth is not None:
                stats.notable["clone_depth set"] = stats.notable.get("clone_depth set", 0) + 1
            if pkg.uses_xdist:
                stats.notable["uses_xdist"] = stats.notable.get("uses_xdist", 0) + 1
            if pkg.import_name is not None:
                stats.notable["import_name set"] = stats.notable.get("import_name set", 0) + 1

        # Quality warnings.
        for warning in detect_quality_warnings(pkg):
            stats.quality_warnings.append((pkg.package, warning))

    return stats


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------


@dataclass
class StatusChange:
    """A status change for a single package between two runs."""

    package: str
    old_status: str
    new_status: str
    old_detail: str = ""
    new_detail: str = ""


@dataclass
class RunAnalysis:
    """Analysis of a single run."""

    run: RunData
    status_counts: dict[str, int] = field(default_factory=dict)
    total_duration: float = 0.0
    install_duration: float = 0.0
    test_duration: float = 0.0
    avg_duration: float = 0.0
    fastest: PackageResult | None = None
    slowest: PackageResult | None = None
    duration_buckets: list[tuple[str, int]] = field(default_factory=list)
    crashes: list[PackageResult] = field(default_factory=list)
    status_changes: list[StatusChange] | None = None


def result_detail(r: PackageResult) -> str:
    """Build a brief detail string for a result."""
    if r.status == "crash":
        return (r.crash_signature or "")[:60]
    if r.status == "timeout":
        return f"(timeout: {int(r.duration_seconds)}s)"
    if r.status == "fail":
        return f"exit code {r.exit_code}"
    if r.status in ("install_error", "clone_error", "error"):
        return (r.error_message or "")[:60]
    return ""


def compute_duration_buckets(
    results: list[PackageResult],
) -> list[tuple[str, int]]:
    """Bucket durations into ranges for histogram display."""
    buckets = [
        ("0-10s", 0),
        ("10-30s", 0),
        ("30s-1m", 0),
        ("1-5m", 0),
        ("5-15m", 0),
        ("15m+", 0),
    ]
    thresholds = [10, 30, 60, 300, 900]
    for r in results:
        d = r.duration_seconds
        placed = False
        for i, t in enumerate(thresholds):
            if d < t:
                label, count = buckets[i]
                buckets[i] = (label, count + 1)
                placed = True
                break
        if not placed:
            label, count = buckets[-1]
            buckets[-1] = (label, count + 1)
    return buckets


def analyze_run(
    run: RunData,
    *,
    previous_run: RunData | None = None,
) -> RunAnalysis:
    """Full analysis of a single run."""
    results = run.results
    analysis = RunAnalysis(run=run)

    # Status counts.
    for r in results:
        analysis.status_counts[r.status] = analysis.status_counts.get(r.status, 0) + 1

    # Timing.
    if results:
        analysis.total_duration = sum(r.duration_seconds for r in results)
        analysis.install_duration = sum(r.install_duration_seconds for r in results)
        analysis.test_duration = analysis.total_duration - analysis.install_duration
        analysis.avg_duration = analysis.total_duration / len(results)
        analysis.fastest = min(results, key=lambda r: r.duration_seconds)
        analysis.slowest = max(results, key=lambda r: r.duration_seconds)

    # Duration histogram.
    analysis.duration_buckets = compute_duration_buckets(results)

    # Crashes.
    analysis.crashes = [r for r in results if r.status == "crash"]

    # Status changes vs previous run.
    if previous_run is not None:
        analysis.status_changes = _compute_status_changes(previous_run, run)

    return analysis


def _compute_status_changes(old_run: RunData, new_run: RunData) -> list[StatusChange]:
    """Compute status changes between two runs."""
    changes: list[StatusChange] = []

    for r in new_run.results:
        old_r = old_run.result_for(r.package)
        if old_r is None:
            continue
        if old_r.status != r.status:
            changes.append(
                StatusChange(
                    package=r.package,
                    old_status=old_r.status,
                    new_status=r.status,
                    old_detail=result_detail(old_r),
                    new_detail=result_detail(r),
                )
            )

    return changes


def build_reproduce_command(
    result: PackageResult,
    registry_entry: PackageEntry,
    python_path: str,
) -> str:
    """Build a shell script to reproduce a crash from scratch."""
    lines: list[str] = []
    repo = result.repo or registry_entry.repo or ""
    pkg_name = result.package

    clone_cmd = f"git clone --depth=1 {repo} /tmp/{pkg_name}-repro"
    if registry_entry.clone_depth is not None and registry_entry.clone_depth > 1:
        clone_cmd = f"git clone --depth={registry_entry.clone_depth} {repo} /tmp/{pkg_name}-repro"

    lines.append(clone_cmd)
    lines.append(f"cd /tmp/{pkg_name}-repro")
    lines.append(f"{python_path} -m venv .venv")

    # Activate venv via PATH so pip, python, pytest, and all entry
    # points resolve to the .venv automatically.
    lines.append('export PATH="$PWD/.venv/bin:$PATH"')

    install_cmd = registry_entry.install_command or "pip install -e ."
    lines.append(install_cmd)

    test_cmd = result.test_command or registry_entry.test_command or "python -m pytest"
    lines.append(f"PYTHON_JIT=1 PYTHONFAULTHANDLER=1 {test_cmd}")

    return "\n".join(lines)


def categorize_install_errors(
    results: list[PackageResult],
) -> dict[str, list[str]]:
    """Group install errors by error pattern."""
    categories: dict[str, list[str]] = {}
    for r in results:
        if r.status != "install_error":
            continue
        msg = (r.error_message or "").lower()
        if any(kw in msg for kw in ("header", "include", ".h")):
            cat = "Missing headers"
        elif any(kw in msg for kw in ("resolution", "conflict", "incompatible")):
            cat = "Pip resolution"
        elif any(kw in msg for kw in ("build", "compile", "setup.py", "wheel")):
            cat = "Build error"
        elif any(kw in msg for kw in ("import", "modulenotfound")):
            cat = "Import failure"
        else:
            cat = "Other"
        categories.setdefault(cat, []).append(r.package)
    return categories


# ---------------------------------------------------------------------------
# Comparison analysis
# ---------------------------------------------------------------------------


@dataclass
class TimingChange:
    """A significant timing change for a single package between two runs."""

    package: str
    old_seconds: float
    new_seconds: float
    change_pct: float


@dataclass
class ComparisonResult:
    """Comparison between two runs."""

    run_a: RunData
    run_b: RunData
    packages_in_common: int = 0
    packages_only_in_a: list[str] = field(default_factory=list)
    packages_only_in_b: list[str] = field(default_factory=list)
    status_changes: list[StatusChange] = field(default_factory=list)
    signature_changes: list[tuple[str, str | None, str | None]] = field(default_factory=list)
    timing_changes: list[TimingChange] = field(default_factory=list)
    unchanged_counts: dict[str, int] = field(default_factory=dict)


def compare_runs(
    run_a: RunData,
    run_b: RunData,
    *,
    timing_threshold_pct: float = 20.0,
    timing_threshold_abs: float = 30.0,
) -> ComparisonResult:
    """Compare two runs.

    Timing changes are only reported if they exceed BOTH the percentage
    threshold AND the absolute threshold.
    """
    result = ComparisonResult(run_a=run_a, run_b=run_b)

    names_a = {r.package for r in run_a.results}
    names_b = {r.package for r in run_b.results}
    common = names_a & names_b

    result.packages_in_common = len(common)
    result.packages_only_in_a = sorted(names_a - names_b)
    result.packages_only_in_b = sorted(names_b - names_a)

    for pkg in sorted(common):
        ra = run_a.result_for(pkg)
        rb = run_b.result_for(pkg)
        if ra is None or rb is None:
            continue  # shouldn't happen, but satisfies type checker

        if ra.status != rb.status:
            result.status_changes.append(
                StatusChange(
                    package=pkg,
                    old_status=ra.status,
                    new_status=rb.status,
                    old_detail=result_detail(ra),
                    new_detail=result_detail(rb),
                )
            )
        else:
            result.unchanged_counts[ra.status] = result.unchanged_counts.get(ra.status, 0) + 1

        # Crash signature comparison.
        if ra.status == "crash" or rb.status == "crash":
            if ra.crash_signature != rb.crash_signature:
                result.signature_changes.append((pkg, ra.crash_signature, rb.crash_signature))

        # Timing changes.
        if ra.duration_seconds > 0 and rb.duration_seconds > 0:
            abs_change = abs(rb.duration_seconds - ra.duration_seconds)
            pct_change = (rb.duration_seconds - ra.duration_seconds) / ra.duration_seconds * 100

            if abs(pct_change) >= timing_threshold_pct and abs_change >= timing_threshold_abs:
                result.timing_changes.append(
                    TimingChange(
                        package=pkg,
                        old_seconds=ra.duration_seconds,
                        new_seconds=rb.duration_seconds,
                        change_pct=pct_change,
                    )
                )

    return result


# ---------------------------------------------------------------------------
# History analysis
# ---------------------------------------------------------------------------


@dataclass
class HistoryAnalysis:
    """Analysis across multiple runs."""

    runs: list[RunData] = field(default_factory=list)
    crash_trend: list[int] = field(default_factory=list)
    pass_rate_trend: list[float] = field(default_factory=list)
    total_unique_crashes: int = 0
    currently_reproducing: int = 0
    likely_fixed: int = 0
    flaky_packages: list[tuple[str, list[str]]] = field(default_factory=list)


def analyze_history(
    runs: list[RunData],
    *,
    flaky_min_oscillations: int = 2,
) -> HistoryAnalysis:
    """Analyze trends across multiple runs."""
    analysis = HistoryAnalysis(runs=runs)

    if not runs:
        return analysis

    # Crash and pass rate trends (oldest to newest for sparklines).
    all_crash_sigs: set[str] = set()
    latest_crash_sigs: set[str] = set()

    for run in reversed(runs):  # oldest first
        results = run.results
        crash_count = sum(1 for r in results if r.status == "crash")
        analysis.crash_trend.append(crash_count)

        total = len(results)
        passed = sum(1 for r in results if r.status == "pass")
        rate = (passed / total * 100) if total > 0 else 0.0
        analysis.pass_rate_trend.append(rate)

        for r in results:
            if r.status == "crash" and r.crash_signature:
                all_crash_sigs.add(r.crash_signature)

    # Latest run stats.
    if runs:
        latest = runs[0]
        for r in latest.results:
            if r.status == "crash" and r.crash_signature:
                latest_crash_sigs.add(r.crash_signature)

    analysis.total_unique_crashes = len(all_crash_sigs)
    analysis.currently_reproducing = len(latest_crash_sigs)
    analysis.likely_fixed = len(all_crash_sigs - latest_crash_sigs)

    # Flaky packages.
    analysis.flaky_packages = detect_flaky_packages(runs, min_oscillations=flaky_min_oscillations)

    return analysis


def detect_flaky_packages(
    runs: list[RunData],
    *,
    min_oscillations: int = 2,
) -> list[tuple[str, list[str]]]:
    """Find packages with inconsistent results across runs.

    Only considers runs with the same Python minor version.
    Ignores crash status (crashes are not flaky tests).
    """
    if not runs:
        return []

    # Group runs by Python minor version.
    by_version: dict[str, list[RunData]] = {}
    for run in runs:
        pv = extract_minor_version(run.meta.python_version)
        by_version.setdefault(pv, []).append(run)

    flaky: list[tuple[str, list[str]]] = []

    for version_runs in by_version.values():
        if len(version_runs) < 2:
            continue

        # Collect all packages across these runs.
        all_packages: set[str] = set()
        for run in version_runs:
            for r in run.results:
                all_packages.add(r.package)

        for pkg in sorted(all_packages):
            statuses: list[str] = []
            for run in reversed(version_runs):  # oldest first
                pkg_result = run.result_for(pkg)
                if pkg_result is not None:
                    statuses.append(pkg_result.status)

            if len(statuses) < 2:
                continue

            # Count oscillations (ignoring crashes).
            filtered = [s for s in statuses if s != "crash"]
            if len(filtered) < 2:
                continue

            oscillations = 0
            for i in range(1, len(filtered)):
                if filtered[i] != filtered[i - 1]:
                    oscillations += 1

            if oscillations >= min_oscillations:
                flaky.append((pkg, statuses))

    return flaky


# ---------------------------------------------------------------------------
# Package history
# ---------------------------------------------------------------------------


@dataclass
class PackageHistory:
    """Complete history for a single package across all runs."""

    package: str = ""
    registry_entry: PackageEntry | None = None
    run_results: list[tuple[RunData, PackageResult]] = field(default_factory=list)
    crash_signatures: dict[str, int] = field(default_factory=dict)
    latest_crash_date: str | None = None
    likely_fixed: bool = False
    dependency_changes: list[tuple[str, dict[str, str], dict[str, str]]] = field(
        default_factory=list
    )


def analyze_package(
    package: str,
    store: ResultsStore,
    registry_entry: PackageEntry | None = None,
) -> PackageHistory:
    """Build complete history for a single package across all runs."""
    history = PackageHistory(
        package=package,
        registry_entry=registry_entry,
    )

    pairs = store.runs_for_package(package)
    history.run_results = pairs

    if not pairs:
        return history

    # Crash signatures.
    for run, result in pairs:
        if result.status == "crash" and result.crash_signature:
            sig = result.crash_signature
            history.crash_signatures[sig] = history.crash_signatures.get(sig, 0) + 1
            if history.latest_crash_date is None:
                history.latest_crash_date = run.run_id

    # Likely fixed: had crashes in the past but not in the latest run.
    latest_result = pairs[0][1] if pairs else None
    if latest_result and latest_result.status != "crash" and history.crash_signatures:
        history.likely_fixed = True

    # Dependency changes at status transition boundaries.
    for i in range(len(pairs) - 1):
        run_new, result_new = pairs[i]
        run_old, result_old = pairs[i + 1]
        if result_new.status != result_old.status:
            old_deps = result_old.installed_dependencies
            new_deps = result_new.installed_dependencies
            if old_deps and new_deps and old_deps != new_deps:
                history.dependency_changes.append((run_new.run_id, old_deps, new_deps))

    return history
