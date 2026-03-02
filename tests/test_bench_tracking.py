"""Tests for labeille.bench.tracking — benchmark tracking series."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from labeille.bench.results import BenchMeta, ConditionDef
from labeille.bench.tracking import (
    TrackingRunEntry,
    TrackingSeries,
    add_run_to_series,
    compute_config_fingerprint,
    init_series,
    list_series,
    load_series,
    pin_baseline,
    save_series,
    unpin_baseline,
)


def _make_bench_run_dir(
    parent: Path,
    *,
    bench_id: str = "bench_20260301_100000",
    start_time: str = "2026-03-01T10:00:00+0000",
    conditions: dict[str, dict[str, object]] | None = None,
    config: dict[str, object] | None = None,
    packages_completed: int = 10,
) -> Path:
    """Create a mock bench run directory with valid files."""
    run_dir = parent / bench_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if conditions is None:
        conditions = {"baseline": {"name": "baseline"}}
    if config is None:
        config = {"iterations": 5, "warmup": 1, "timeout": 600}

    meta = {
        "bench_id": bench_id,
        "name": bench_id,
        "start_time": start_time,
        "end_time": start_time,
        "conditions": conditions,
        "config": config,
        "system": {},
        "python_profiles": {},
        "packages_total": packages_completed,
        "packages_completed": packages_completed,
        "packages_skipped": 0,
    }
    (run_dir / "bench_meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "bench_results.jsonl").write_text("")
    return run_dir


# ---------------------------------------------------------------------------
# TestComputeConfigFingerprint
# ---------------------------------------------------------------------------


class TestComputeConfigFingerprint(unittest.TestCase):
    """Tests for compute_config_fingerprint()."""

    def _make_meta(
        self,
        *,
        conditions: dict[str, ConditionDef] | None = None,
        config: dict[str, object] | None = None,
        bench_id: str = "test_001",
        packages_completed: int = 10,
    ) -> BenchMeta:
        meta = BenchMeta(bench_id=bench_id)
        meta.conditions = conditions or {"baseline": ConditionDef(name="baseline")}
        meta.config = config or {"iterations": 5, "warmup": 1, "timeout": 600}
        meta.packages_completed = packages_completed
        return meta

    def test_same_config_same_fingerprint(self) -> None:
        m1 = self._make_meta(bench_id="run1")
        m2 = self._make_meta(bench_id="run2")
        self.assertEqual(compute_config_fingerprint(m1), compute_config_fingerprint(m2))

    def test_different_conditions_different_fingerprint(self) -> None:
        m1 = self._make_meta()
        m2 = self._make_meta(
            conditions={
                "baseline": ConditionDef(name="baseline"),
                "jit": ConditionDef(name="jit", env={"PYTHON_JIT": "1"}),
            }
        )
        self.assertNotEqual(compute_config_fingerprint(m1), compute_config_fingerprint(m2))

    def test_different_iterations_different_fingerprint(self) -> None:
        m1 = self._make_meta()
        m2 = self._make_meta(config={"iterations": 10, "warmup": 1, "timeout": 600})
        self.assertNotEqual(compute_config_fingerprint(m1), compute_config_fingerprint(m2))

    def test_different_packages_same_fingerprint(self) -> None:
        m1 = self._make_meta(packages_completed=10)
        m2 = self._make_meta(packages_completed=50)
        self.assertEqual(compute_config_fingerprint(m1), compute_config_fingerprint(m2))

    def test_different_bench_id_same_fingerprint(self) -> None:
        m1 = self._make_meta(bench_id="run_001")
        m2 = self._make_meta(bench_id="run_002")
        self.assertEqual(compute_config_fingerprint(m1), compute_config_fingerprint(m2))

    def test_different_system_same_fingerprint(self) -> None:
        m1 = self._make_meta()
        m2 = self._make_meta()
        m2.system.cpu_model = "AMD EPYC 7763"
        self.assertEqual(compute_config_fingerprint(m1), compute_config_fingerprint(m2))

    def test_fingerprint_is_deterministic(self) -> None:
        m = self._make_meta()
        fp1 = compute_config_fingerprint(m)
        fp2 = compute_config_fingerprint(m)
        self.assertEqual(fp1, fp2)


# ---------------------------------------------------------------------------
# TestTrackingSeries
# ---------------------------------------------------------------------------


class TestTrackingSeries(unittest.TestCase):
    """Tests for TrackingSeries dataclass."""

    def _make_entries(self) -> list[TrackingRunEntry]:
        return [
            TrackingRunEntry(
                bench_id="run_001",
                timestamp="2026-03-01T10:00:00+0000",
                run_dir="run_001",
                packages_completed=10,
                config_fingerprint="abc123",
            ),
            TrackingRunEntry(
                bench_id="run_002",
                timestamp="2026-03-08T10:00:00+0000",
                run_dir="run_002",
                packages_completed=15,
                config_fingerprint="abc123",
            ),
        ]

    def test_series_roundtrip(self) -> None:
        entries = self._make_entries()
        series = TrackingSeries(
            series_id="test",
            description="Test series",
            created="2026-03-01T10:00:00",
            config_fingerprint="abc123",
            pinned_baseline_id="run_001",
            runs=entries,
        )
        d = series.to_dict()
        restored = TrackingSeries.from_dict(d)
        self.assertEqual(restored.series_id, "test")
        self.assertEqual(restored.description, "Test series")
        self.assertEqual(restored.pinned_baseline_id, "run_001")
        self.assertEqual(len(restored.runs), 2)

    def test_latest_run(self) -> None:
        series = TrackingSeries(series_id="test", runs=self._make_entries())
        latest = series.latest_run
        assert latest is not None
        self.assertEqual(latest.bench_id, "run_002")

    def test_latest_run_empty(self) -> None:
        series = TrackingSeries(series_id="test")
        self.assertIsNone(series.latest_run)

    def test_baseline_run_pinned(self) -> None:
        series = TrackingSeries(
            series_id="test",
            pinned_baseline_id="run_002",
            runs=self._make_entries(),
        )
        baseline = series.baseline_run
        assert baseline is not None
        self.assertEqual(baseline.bench_id, "run_002")

    def test_baseline_run_unpinned(self) -> None:
        series = TrackingSeries(series_id="test", runs=self._make_entries())
        baseline = series.baseline_run
        assert baseline is not None
        self.assertEqual(baseline.bench_id, "run_001")

    def test_baseline_run_pinned_not_found(self) -> None:
        series = TrackingSeries(
            series_id="test",
            pinned_baseline_id="nonexistent",
            runs=self._make_entries(),
        )
        baseline = series.baseline_run
        assert baseline is not None
        self.assertEqual(baseline.bench_id, "run_001")  # Falls back to first

    def test_date_range(self) -> None:
        series = TrackingSeries(series_id="test", runs=self._make_entries())
        dr = series.date_range
        assert dr is not None
        self.assertEqual(dr[0], "2026-03-01T10:00:00+0000")
        self.assertEqual(dr[1], "2026-03-08T10:00:00+0000")

    def test_date_range_empty(self) -> None:
        series = TrackingSeries(series_id="test")
        self.assertIsNone(series.date_range)


# ---------------------------------------------------------------------------
# TestTrackingRunEntry
# ---------------------------------------------------------------------------


class TestTrackingRunEntry(unittest.TestCase):
    """Tests for TrackingRunEntry dataclass."""

    def test_roundtrip(self) -> None:
        entry = TrackingRunEntry(
            bench_id="run_001",
            timestamp="2026-03-01T10:00:00+0000",
            run_dir="run_001",
            packages_completed=10,
            config_fingerprint="abc123",
            commit_info={"cpython": "abc1234"},
            notes="Test run",
        )
        d = entry.to_dict()
        restored = TrackingRunEntry.from_dict(d)
        self.assertEqual(restored.bench_id, "run_001")
        self.assertEqual(restored.commit_info, {"cpython": "abc1234"})
        self.assertEqual(restored.notes, "Test run")

    def test_sparse_serialization(self) -> None:
        entry = TrackingRunEntry(
            bench_id="run_001",
            timestamp="2026-03-01T10:00:00+0000",
            run_dir="run_001",
            packages_completed=10,
            config_fingerprint="abc123",
        )
        d = entry.to_dict()
        self.assertNotIn("commit_info", d)
        self.assertNotIn("notes", d)


# ---------------------------------------------------------------------------
# TestInitSeries
# ---------------------------------------------------------------------------


class TestInitSeries(unittest.TestCase):
    """Tests for init_series()."""

    def test_init_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            series = init_series(Path(tmpdir), "test-series")
            self.assertTrue((Path(tmpdir) / "test-series").is_dir())
            self.assertEqual(series.series_id, "test-series")

    def test_init_creates_tracking_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            init_series(Path(tmpdir), "test-series")
            tracking_file = Path(tmpdir) / "test-series" / "tracking.json"
            self.assertTrue(tracking_file.exists())
            data = json.loads(tracking_file.read_text())
            self.assertEqual(data["series_id"], "test-series")

    def test_init_duplicate_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            init_series(Path(tmpdir), "test-series")
            with self.assertRaises(ValueError):
                init_series(Path(tmpdir), "test-series")

    def test_init_empty_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            series = init_series(Path(tmpdir), "test-series")
            self.assertEqual(series.config_fingerprint, "")


# ---------------------------------------------------------------------------
# TestAddRunToSeries
# ---------------------------------------------------------------------------


class TestAddRunToSeries(unittest.TestCase):
    """Tests for add_run_to_series()."""

    def test_add_first_run_sets_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base / "tracking", "test-series")
            run_dir = _make_bench_run_dir(base / "runs")

            add_run_to_series(base / "tracking" / "test-series", run_dir)
            series = load_series(base / "tracking" / "test-series")
            self.assertNotEqual(series.config_fingerprint, "")
            self.assertEqual(series.n_runs, 1)

    def test_add_second_run_same_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base / "tracking", "test-series")
            run1 = _make_bench_run_dir(
                base / "runs", bench_id="run_001", start_time="2026-03-01T10:00:00"
            )
            run2 = _make_bench_run_dir(
                base / "runs", bench_id="run_002", start_time="2026-03-08T10:00:00"
            )

            add_run_to_series(base / "tracking" / "test-series", run1)
            add_run_to_series(base / "tracking" / "test-series", run2)
            series = load_series(base / "tracking" / "test-series")
            self.assertEqual(series.n_runs, 2)

    def test_add_run_different_fingerprint_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base / "tracking", "test-series")
            run1 = _make_bench_run_dir(base / "runs", bench_id="run_001")
            run2 = _make_bench_run_dir(
                base / "runs",
                bench_id="run_002",
                config={"iterations": 10, "warmup": 2, "timeout": 600},
            )

            add_run_to_series(base / "tracking" / "test-series", run1)
            # Should warn but still add.
            add_run_to_series(base / "tracking" / "test-series", run2)
            series = load_series(base / "tracking" / "test-series")
            self.assertEqual(series.n_runs, 2)

    def test_add_run_creates_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base / "tracking", "test-series")
            run_dir = _make_bench_run_dir(base / "runs")

            add_run_to_series(base / "tracking" / "test-series", run_dir)
            link = base / "tracking" / "test-series" / run_dir.name
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), run_dir.resolve())

    def test_add_duplicate_bench_id_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base / "tracking", "test-series")
            run_dir = _make_bench_run_dir(base / "runs")

            add_run_to_series(base / "tracking" / "test-series", run_dir)
            add_run_to_series(base / "tracking" / "test-series", run_dir)
            series = load_series(base / "tracking" / "test-series")
            self.assertEqual(series.n_runs, 1)

    def test_add_run_sorted_by_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base / "tracking", "test-series")
            run2 = _make_bench_run_dir(
                base / "runs", bench_id="run_002", start_time="2026-03-08T10:00:00"
            )
            run1 = _make_bench_run_dir(
                base / "runs", bench_id="run_001", start_time="2026-03-01T10:00:00"
            )

            # Add in reverse chronological order.
            add_run_to_series(base / "tracking" / "test-series", run2)
            add_run_to_series(base / "tracking" / "test-series", run1)
            series = load_series(base / "tracking" / "test-series")
            self.assertEqual(series.runs[0].bench_id, "run_001")
            self.assertEqual(series.runs[1].bench_id, "run_002")

    def test_add_run_with_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base / "tracking", "test-series")
            run_dir = _make_bench_run_dir(base / "runs")

            entry = add_run_to_series(
                base / "tracking" / "test-series", run_dir, notes="First run"
            )
            self.assertEqual(entry.notes, "First run")

    def test_add_run_with_commit_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base / "tracking", "test-series")
            run_dir = _make_bench_run_dir(base / "runs")

            entry = add_run_to_series(
                base / "tracking" / "test-series",
                run_dir,
                commit_info={"cpython": "abc1234"},
            )
            self.assertEqual(entry.commit_info, {"cpython": "abc1234"})

    def test_add_invalid_run_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base / "tracking", "test-series")
            fake_dir = base / "not_a_run"
            fake_dir.mkdir()

            with self.assertRaises(FileNotFoundError):
                add_run_to_series(base / "tracking" / "test-series", fake_dir)


# ---------------------------------------------------------------------------
# TestPinUnpinBaseline
# ---------------------------------------------------------------------------


class TestPinUnpinBaseline(unittest.TestCase):
    """Tests for pin_baseline() and unpin_baseline()."""

    def _setup_series_with_run(self) -> tuple[Path, Path]:
        self._tmpdir = tempfile.mkdtemp()
        base = Path(self._tmpdir)
        init_series(base / "tracking", "test-series")
        run_dir = _make_bench_run_dir(base / "runs")
        add_run_to_series(base / "tracking" / "test-series", run_dir)
        return base / "tracking" / "test-series", run_dir

    def tearDown(self) -> None:
        import shutil

        if hasattr(self, "_tmpdir"):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_pin_baseline(self) -> None:
        series_dir, _ = self._setup_series_with_run()
        series = load_series(series_dir)
        bench_id = series.runs[0].bench_id
        pin_baseline(series_dir, bench_id)
        reloaded = load_series(series_dir)
        self.assertEqual(reloaded.pinned_baseline_id, bench_id)

    def test_pin_invalid_bench_id(self) -> None:
        series_dir, _ = self._setup_series_with_run()
        with self.assertRaises(ValueError):
            pin_baseline(series_dir, "nonexistent")

    def test_unpin_baseline(self) -> None:
        series_dir, _ = self._setup_series_with_run()
        series = load_series(series_dir)
        pin_baseline(series_dir, series.runs[0].bench_id)
        unpin_baseline(series_dir)
        reloaded = load_series(series_dir)
        self.assertIsNone(reloaded.pinned_baseline_id)

    def test_unpin_when_not_pinned(self) -> None:
        series_dir, _ = self._setup_series_with_run()
        # Should not raise.
        unpin_baseline(series_dir)


# ---------------------------------------------------------------------------
# TestLoadSaveSeries
# ---------------------------------------------------------------------------


class TestLoadSaveSeries(unittest.TestCase):
    """Tests for load_series() and save_series()."""

    def test_save_then_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            series_dir = Path(tmpdir) / "test-series"
            series = TrackingSeries(
                series_id="test-series",
                description="Test",
                created="2026-03-01T10:00:00",
                config_fingerprint="abc123",
            )
            save_series(series, series_dir)
            loaded = load_series(series_dir)
            self.assertEqual(loaded.series_id, "test-series")
            self.assertEqual(loaded.config_fingerprint, "abc123")

    def test_save_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            series_dir = Path(tmpdir) / "test-series"
            series = TrackingSeries(series_id="test-series", created="2026-03-01T10:00:00")
            save_series(series, series_dir)
            # Verify no temp file left behind.
            self.assertFalse((series_dir / "tracking.json.tmp").exists())

    def test_load_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_series(Path(tmpdir) / "nonexistent")

    def test_load_malformed_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            series_dir = Path(tmpdir) / "bad"
            series_dir.mkdir()
            (series_dir / "tracking.json").write_text("not valid json")
            with self.assertRaises(ValueError):
                load_series(series_dir)


# ---------------------------------------------------------------------------
# TestListSeries
# ---------------------------------------------------------------------------


class TestListSeries(unittest.TestCase):
    """Tests for list_series()."""

    def test_list_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_series(Path(tmpdir))
            self.assertEqual(result, [])

    def test_list_multiple_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base, "beta-series")
            init_series(base, "alpha-series")
            result = list_series(base)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].series_id, "alpha-series")
            self.assertEqual(result[1].series_id, "beta-series")

    def test_list_ignores_non_series_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            init_series(base, "real-series")
            # Create a dir without tracking.json.
            (base / "not-a-series").mkdir()
            result = list_series(base)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].series_id, "real-series")


if __name__ == "__main__":
    unittest.main()
