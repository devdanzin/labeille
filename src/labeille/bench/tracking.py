"""Benchmark tracking series for longitudinal comparison.

A tracking series is a named collection of benchmark runs intended
for regression detection over time. Runs in a series should share
the same benchmark configuration (conditions, iteration count,
warmup count) so that results are comparable.

The series tracks a "config fingerprint" — a hash of the
configuration aspects that affect comparability. Package list
changes do NOT break the fingerprint, since the registry naturally
grows over time and packages missing from earlier runs simply have
shorter histories.

Series data is stored as JSON files alongside symlinks to the
original benchmark run directories.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labeille.bench.results import BenchMeta, BenchPackageResult, load_bench_run

log = logging.getLogger("labeille")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrackingRunEntry:
    """Index entry for one run within a tracking series."""

    bench_id: str
    timestamp: str  # ISO 8601 from bench_meta.json start_time
    run_dir: str  # Relative path within tracking dir (or absolute symlink target)
    packages_completed: int
    config_fingerprint: str  # Fingerprint of this run's config
    commit_info: dict[str, str] = field(default_factory=dict)
    # e.g. {"cpython": "abc1234", "labeille": "def5678"}
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (sparse)."""
        d: dict[str, Any] = {
            "bench_id": self.bench_id,
            "timestamp": self.timestamp,
            "run_dir": self.run_dir,
            "packages_completed": self.packages_completed,
            "config_fingerprint": self.config_fingerprint,
        }
        if self.commit_info:
            d["commit_info"] = self.commit_info
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackingRunEntry:
        """Deserialize from a dict, ignoring unknown fields."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TrackingSeries:
    """A named series of benchmark runs for longitudinal tracking."""

    series_id: str  # e.g. "jit-overhead"
    description: str = ""
    created: str = ""  # ISO 8601 timestamp
    config_fingerprint: str = ""  # Fingerprint from the first run
    pinned_baseline_id: str | None = None  # bench_id of the pinned baseline
    runs: list[TrackingRunEntry] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        """Number of runs in the series."""
        return len(self.runs)

    @property
    def latest_run(self) -> TrackingRunEntry | None:
        """Most recent run by timestamp."""
        if not self.runs:
            return None
        return max(self.runs, key=lambda r: r.timestamp)

    @property
    def baseline_run(self) -> TrackingRunEntry | None:
        """The pinned baseline run, or the first run if not pinned."""
        if self.pinned_baseline_id:
            for r in self.runs:
                if r.bench_id == self.pinned_baseline_id:
                    return r
        return self.runs[0] if self.runs else None

    @property
    def date_range(self) -> tuple[str, str] | None:
        """Earliest and latest timestamps, or None if no runs."""
        if not self.runs:
            return None
        timestamps = [r.timestamp for r in self.runs]
        return (min(timestamps), max(timestamps))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "series_id": self.series_id,
            "description": self.description,
            "created": self.created,
            "config_fingerprint": self.config_fingerprint,
            "runs": [r.to_dict() for r in self.runs],
        }
        if self.pinned_baseline_id is not None:
            d["pinned_baseline_id"] = self.pinned_baseline_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackingSeries:
        """Deserialize from a dict."""
        runs = [TrackingRunEntry.from_dict(r) for r in data.get("runs", [])]
        return cls(
            series_id=data["series_id"],
            description=data.get("description", ""),
            created=data.get("created", ""),
            config_fingerprint=data.get("config_fingerprint", ""),
            pinned_baseline_id=data.get("pinned_baseline_id"),
            runs=runs,
        )


# ---------------------------------------------------------------------------
# Config fingerprinting
# ---------------------------------------------------------------------------


def compute_config_fingerprint(meta: BenchMeta) -> str:
    """Compute a fingerprint for a benchmark's configuration.

    The fingerprint covers aspects that affect comparability:
    - Condition names and their definitions (env, deps, python, constraints)
    - Iteration count and warmup count
    - Timeout

    It does NOT include:
    - Package list (registry grows over time)
    - System profile (hardware may change, but results should still be trackable)
    - Start/end times, bench_id
    - CLI args (different invocations of the same profile are fine)

    Returns:
        A hex digest string (SHA-256, first 16 chars).
    """
    # Build a canonical dict of comparability-affecting config.
    conditions_canonical: dict[str, Any] = {}
    for name in sorted(meta.conditions):
        cond = meta.conditions[name]
        cond_dict = cond.to_dict()
        # Remove fields that don't affect comparability.
        cond_dict.pop("name", None)
        cond_dict.pop("description", None)
        conditions_canonical[name] = cond_dict

    fingerprint_data = {
        "conditions": conditions_canonical,
        "iterations": meta.config.get("iterations", 5),
        "warmup": meta.config.get("warmup", 1),
        "timeout": meta.config.get("timeout", 600),
    }

    canonical = json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------


def load_series(series_dir: Path) -> TrackingSeries:
    """Load a tracking series from its directory.

    Args:
        series_dir: Path to the series directory containing tracking.json.

    Returns:
        The loaded TrackingSeries.

    Raises:
        FileNotFoundError: If tracking.json doesn't exist.
        ValueError: If tracking.json is malformed.
    """
    tracking_file = series_dir / "tracking.json"
    if not tracking_file.exists():
        raise FileNotFoundError(f"No tracking.json found in {series_dir}")

    text = tracking_file.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed tracking.json in {series_dir}: {exc}") from exc

    if not isinstance(data, dict) or "series_id" not in data:
        raise ValueError(f"Invalid tracking.json in {series_dir}: missing series_id")

    return TrackingSeries.from_dict(data)


def save_series(series: TrackingSeries, series_dir: Path) -> None:
    """Save a tracking series to its directory.

    Creates the directory if needed. Writes tracking.json atomically
    (write to temp file, then rename).

    Args:
        series: The series to save.
        series_dir: Path to the series directory.
    """
    series_dir.mkdir(parents=True, exist_ok=True)
    tracking_file = series_dir / "tracking.json"
    tmp_file = series_dir / "tracking.json.tmp"

    text = json.dumps(series.to_dict(), indent=2) + "\n"
    tmp_file.write_text(text)
    os.replace(str(tmp_file), str(tracking_file))


def list_series(tracking_dir: Path) -> list[TrackingSeries]:
    """List all tracking series in a tracking directory.

    Args:
        tracking_dir: Parent directory containing series subdirectories.

    Returns:
        List of TrackingSeries, sorted by series_id.
    """
    if not tracking_dir.exists():
        return []

    result: list[TrackingSeries] = []
    for entry in sorted(tracking_dir.iterdir()):
        if not entry.is_dir():
            continue
        tracking_file = entry / "tracking.json"
        if not tracking_file.exists():
            continue
        try:
            series = load_series(entry)
            result.append(series)
        except (ValueError, FileNotFoundError):
            log.warning("Skipping malformed series in %s", entry)

    return result


# ---------------------------------------------------------------------------
# Series management
# ---------------------------------------------------------------------------


def init_series(
    tracking_dir: Path,
    series_id: str,
    *,
    description: str = "",
) -> TrackingSeries:
    """Create a new tracking series.

    Creates the series directory and initial tracking.json.
    The config fingerprint is set when the first run is added.

    Args:
        tracking_dir: Parent directory for all series.
        series_id: Unique name (used as directory name).
        description: Human-readable description.

    Returns:
        The newly created TrackingSeries.

    Raises:
        ValueError: If series_id already exists.
    """
    series_dir = tracking_dir / series_id
    if series_dir.exists():
        raise ValueError(f"Series '{series_id}' already exists at {series_dir}")

    series = TrackingSeries(
        series_id=series_id,
        description=description,
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )

    save_series(series, series_dir)
    return series


def add_run_to_series(
    series_dir: Path,
    bench_run_dir: Path,
    *,
    notes: str = "",
    commit_info: dict[str, str] | None = None,
) -> TrackingRunEntry:
    """Add a benchmark run to a tracking series.

    Validates the run's config fingerprint against the series.
    On the first run, sets the series fingerprint. On subsequent runs,
    warns if the fingerprint differs (but still adds the run — the user
    may have intentionally changed iterations, etc.).

    Creates a symlink from the series directory to the bench run.

    Args:
        series_dir: Path to the series directory.
        bench_run_dir: Path to the benchmark output directory.
        notes: Free-form annotation for this run.
        commit_info: Optional commit hashes for tracking.

    Returns:
        The TrackingRunEntry that was added.

    Raises:
        FileNotFoundError: If bench_run_dir doesn't contain valid bench data.
    """
    series = load_series(series_dir)
    meta, results = load_bench_run(bench_run_dir)

    fingerprint = compute_config_fingerprint(meta)

    # Set or check fingerprint.
    if not series.config_fingerprint:
        series.config_fingerprint = fingerprint
    elif series.config_fingerprint != fingerprint:
        log.warning(
            "Config fingerprint for run '%s' (%s) differs from series '%s' (%s). "
            "Results may not be directly comparable.",
            meta.bench_id,
            fingerprint,
            series.series_id,
            series.config_fingerprint,
        )

    # Check for duplicate.
    for existing in series.runs:
        if existing.bench_id == meta.bench_id:
            log.info("Run '%s' already in series, skipping.", meta.bench_id)
            return existing

    # Create symlink.
    link_path = series_dir / bench_run_dir.name
    if not link_path.exists():
        link_path.symlink_to(bench_run_dir.resolve())

    entry = TrackingRunEntry(
        bench_id=meta.bench_id,
        timestamp=meta.start_time,
        run_dir=bench_run_dir.name,
        packages_completed=meta.packages_completed,
        config_fingerprint=fingerprint,
        commit_info=commit_info or {},
        notes=notes,
    )

    series.runs.append(entry)
    series.runs.sort(key=lambda r: r.timestamp)
    save_series(series, series_dir)
    return entry


def pin_baseline(
    series_dir: Path,
    bench_id: str,
) -> None:
    """Pin a specific run as the baseline for comparison.

    Args:
        series_dir: Path to the series directory.
        bench_id: The bench_id to pin.

    Raises:
        ValueError: If bench_id is not found in the series.
    """
    series = load_series(series_dir)
    if not any(r.bench_id == bench_id for r in series.runs):
        raise ValueError(
            f"Bench ID '{bench_id}' not found in series '{series.series_id}'. "
            f"Available: {', '.join(r.bench_id for r in series.runs)}"
        )
    series.pinned_baseline_id = bench_id
    save_series(series, series_dir)


def unpin_baseline(series_dir: Path) -> None:
    """Remove the pinned baseline, reverting to using the first run."""
    series = load_series(series_dir)
    series.pinned_baseline_id = None
    save_series(series, series_dir)


def load_series_run(
    series: TrackingSeries,
    series_dir: Path,
    entry: TrackingRunEntry,
) -> tuple[BenchMeta, list[BenchPackageResult]]:
    """Load the full benchmark data for a run in a series.

    Resolves the symlink and loads via load_bench_run().

    Args:
        series: The tracking series.
        series_dir: Path to the series directory.
        entry: The run entry to load.

    Returns:
        Tuple of (BenchMeta, list of BenchPackageResult).
    """
    run_path = series_dir / entry.run_dir
    # Resolve symlinks to get the actual directory.
    if run_path.is_symlink():
        run_path = run_path.resolve()
    return load_bench_run(run_path)
