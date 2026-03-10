"""Tests for labeille.bench_cli — Click CLI for bench subcommands."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from click.testing import CliRunner

from labeille.bench_cli import bench


class TestBenchGroupHelp(unittest.TestCase):
    """Tests for the bench group and subcommand help."""

    def test_bench_group_help(self) -> None:
        result = CliRunner().invoke(bench, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Benchmark", result.output)
        self.assertIn("run", result.output)
        self.assertIn("show", result.output)
        self.assertIn("compare", result.output)
        self.assertIn("system", result.output)
        self.assertIn("export", result.output)

    def test_bench_run_help(self) -> None:
        result = CliRunner().invoke(bench, ["run", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--profile", result.output)
        self.assertIn("--condition", result.output)
        self.assertIn("--registry-dir", result.output)

    def test_bench_show_help(self) -> None:
        result = CliRunner().invoke(bench, ["show", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("RESULT_DIR", result.output)

    def test_bench_compare_help(self) -> None:
        result = CliRunner().invoke(bench, ["compare", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--baseline", result.output)

    def test_bench_system_help(self) -> None:
        result = CliRunner().invoke(bench, ["system", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--target-python", result.output)

    def test_bench_export_help(self) -> None:
        result = CliRunner().invoke(bench, ["export", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--format", result.output)


class TestBenchSystem(unittest.TestCase):
    """Tests for bench system command."""

    def test_bench_system_runs(self) -> None:
        result = CliRunner().invoke(bench, ["system"])
        self.assertEqual(result.exit_code, 0)
        # Should contain system profile sections.
        output = result.output.lower()
        self.assertTrue(
            "cpu" in output or "ram" in output or "system" in output,
            f"Expected system info in output, got: {result.output[:200]}",
        )

    def test_bench_system_json(self) -> None:
        result = CliRunner().invoke(bench, ["system", "--json"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("cpu_model", data)


class TestBenchShowErrors(unittest.TestCase):
    """Tests for bench show error handling."""

    def test_bench_show_missing_dir(self) -> None:
        result = CliRunner().invoke(bench, ["show", "/nonexistent/path"])
        self.assertNotEqual(result.exit_code, 0)


class TestBenchRunNoConditions(unittest.TestCase):
    """Tests for bench run with missing configuration."""

    def test_bench_run_no_conditions(self) -> None:
        """Run with no conditions or profile gives error about conditions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal registry dir so the path exists.
            reg_dir = Path(tmpdir) / "registry"
            reg_dir.mkdir()
            result = CliRunner().invoke(
                bench,
                [
                    "run",
                    "--registry-dir",
                    str(reg_dir),
                    "--target-python",
                    "/usr/bin/python3",
                ],
            )
            # Should fail because no conditions are defined.
            self.assertNotEqual(result.exit_code, 0)
            exc_str = str(result.exception) if result.exception else ""
            self.assertIn("condition", (result.output + exc_str).lower())


class TestBenchShow(unittest.TestCase):
    """Tests for bench show with real data."""

    def test_bench_show_loads_results(self) -> None:
        """bench show displays formatted benchmark results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a minimal bench_meta.json and bench_results.jsonl.
            meta_data = {
                "bench_id": "test_bench",
                "name": "Test",
                "description": "",
                "system": {},
                "python_profiles": {},
                "conditions": {"baseline": {"name": "baseline"}},
                "config": {"iterations": 3, "warmup": 0},
                "cli_args": [],
                "start_time": "2026-02-28T10:00:00",
                "end_time": "2026-02-28T10:30:00",
                "packages_total": 0,
                "packages_completed": 0,
                "packages_skipped": 0,
            }
            (Path(tmpdir) / "bench_meta.json").write_text(json.dumps(meta_data))
            (Path(tmpdir) / "bench_results.jsonl").write_text("")

            result = CliRunner().invoke(bench, ["show", tmpdir])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Test", result.output)


class TestBenchExport(unittest.TestCase):
    """Tests for bench export command."""

    def _make_bench_dir(self, tmpdir: str) -> str:
        """Create a minimal bench directory with meta and results."""
        meta_data = {
            "bench_id": "test_bench",
            "name": "Export Test",
            "description": "",
            "system": {},
            "python_profiles": {},
            "conditions": {"baseline": {"name": "baseline"}},
            "config": {"iterations": 3, "warmup": 0},
            "cli_args": [],
            "start_time": "",
            "end_time": "",
            "packages_total": 1,
            "packages_completed": 1,
            "packages_skipped": 0,
        }
        result_data = {
            "package": "testpkg",
            "conditions": {
                "baseline": {
                    "condition_name": "baseline",
                    "iterations": [
                        {
                            "index": i,
                            "warmup": False,
                            "wall_time_s": 1.5 + i * 0.01,
                            "user_time_s": 1.2,
                            "sys_time_s": 0.1,
                            "peak_rss_mb": 100.0,
                            "exit_code": 0,
                            "status": "ok",
                            "outlier": False,
                            "load_avg_start": 0.5,
                            "load_avg_end": 0.5,
                            "ram_available_start_gb": 8.0,
                        }
                        for i in range(1, 4)
                    ],
                    "install_duration_s": 0.5,
                    "venv_setup_duration_s": 0.3,
                }
            },
            "clone_duration_s": 0.5,
            "skipped": False,
            "skip_reason": "",
        }
        bench_dir = Path(tmpdir) / "bench_001"
        bench_dir.mkdir()
        (bench_dir / "bench_meta.json").write_text(json.dumps(meta_data))
        (bench_dir / "bench_results.jsonl").write_text(json.dumps(result_data) + "\n")
        return str(bench_dir)

    def test_export_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = self._make_bench_dir(tmpdir)
            result = CliRunner().invoke(bench, ["export", bench_dir, "--format", "csv"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("testpkg", result.output)
            self.assertIn("wall_time_s", result.output)

    def test_export_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = self._make_bench_dir(tmpdir)
            result = CliRunner().invoke(bench, ["export", bench_dir, "--format", "markdown"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("testpkg", result.output)
            self.assertIn("|", result.output)

    def test_export_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = self._make_bench_dir(tmpdir)
            output_file = Path(tmpdir) / "output.csv"
            result = CliRunner().invoke(
                bench,
                ["export", bench_dir, "--format", "csv", "-o", str(output_file)],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(output_file.exists())
            self.assertIn("testpkg", output_file.read_text())


class _TrackTestBase(unittest.TestCase):
    """Shared helpers for bench track subcommand tests."""

    def _make_bench_run_dir(
        self,
        parent: Path,
        bench_id: str,
        *,
        start_time: str = "2026-03-01T10:00:00",
        packages_completed: int = 1,
        iterations: int = 3,
        warmup: int = 0,
        timeout: int = 600,
        wall_time_base: float = 1.5,
    ) -> Path:
        """Create a minimal bench run directory with meta and results."""
        run_dir = parent / bench_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_data = {
            "bench_id": bench_id,
            "name": "Test",
            "description": "",
            "system": {},
            "python_profiles": {},
            "conditions": {"baseline": {"name": "baseline"}},
            "config": {
                "iterations": iterations,
                "warmup": warmup,
                "timeout": timeout,
            },
            "cli_args": [],
            "start_time": start_time,
            "end_time": start_time,
            "packages_total": packages_completed,
            "packages_completed": packages_completed,
            "packages_skipped": 0,
        }
        result_data = {
            "package": "testpkg",
            "conditions": {
                "baseline": {
                    "condition_name": "baseline",
                    "iterations": [
                        {
                            "index": i,
                            "warmup": False,
                            "wall_time_s": wall_time_base + i * 0.01,
                            "user_time_s": 1.2,
                            "sys_time_s": 0.1,
                            "peak_rss_mb": 100.0,
                            "exit_code": 0,
                            "status": "ok",
                            "outlier": False,
                            "load_avg_start": 0.5,
                            "load_avg_end": 0.5,
                            "ram_available_start_gb": 8.0,
                        }
                        for i in range(1, iterations + 1)
                    ],
                    "install_duration_s": 0.5,
                    "venv_setup_duration_s": 0.3,
                }
            },
            "clone_duration_s": 0.5,
            "skipped": False,
            "skip_reason": "",
        }
        (run_dir / "bench_meta.json").write_text(json.dumps(meta_data))
        (run_dir / "bench_results.jsonl").write_text(json.dumps(result_data) + "\n")
        return run_dir


class TestTrackHelp(unittest.TestCase):
    """Tests for bench track subgroup help."""

    def test_track_group_help(self) -> None:
        result = CliRunner().invoke(bench, ["track", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("init", result.output)
        self.assertIn("add", result.output)
        self.assertIn("show", result.output)
        self.assertIn("pin", result.output)
        self.assertIn("unpin", result.output)
        self.assertIn("list", result.output)
        self.assertIn("trend", result.output)
        self.assertIn("alert", result.output)

    def test_track_init_help(self) -> None:
        result = CliRunner().invoke(bench, ["track", "init", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("SERIES_NAME", result.output)

    def test_track_add_help(self) -> None:
        result = CliRunner().invoke(bench, ["track", "add", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("SERIES_NAME", result.output)
        self.assertIn("BENCH_RUN_DIR", result.output)

    def test_track_trend_help(self) -> None:
        result = CliRunner().invoke(bench, ["track", "trend", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--condition", result.output)
        self.assertIn("--format", result.output)


class TestTrackInit(_TrackTestBase):
    """Tests for bench track init."""

    def test_init_creates_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            result = CliRunner().invoke(
                bench, ["track", "init", "my-series", "--tracking-dir", tracking_dir]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Created tracking series", result.output)
            self.assertIn("my-series", result.output)
            tracking_json = Path(tracking_dir) / "my-series" / "tracking.json"
            self.assertTrue(tracking_json.exists())

    def test_init_with_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            result = CliRunner().invoke(
                bench,
                [
                    "track",
                    "init",
                    "jit-overhead",
                    "-d",
                    "JIT overhead tracking",
                    "--tracking-dir",
                    tracking_dir,
                ],
            )
            self.assertEqual(result.exit_code, 0)
            data = json.loads((Path(tracking_dir) / "jit-overhead" / "tracking.json").read_text())
            self.assertEqual(data["description"], "JIT overhead tracking")

    def test_init_duplicate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            CliRunner().invoke(bench, ["track", "init", "dup", "--tracking-dir", tracking_dir])
            result = CliRunner().invoke(
                bench, ["track", "init", "dup", "--tracking-dir", tracking_dir]
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("already exists", result.output)


class TestTrackAdd(_TrackTestBase):
    """Tests for bench track add."""

    def test_add_run_to_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            run_dir = self._make_bench_run_dir(tmppath, "bench_001")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            result = CliRunner().invoke(
                bench,
                ["track", "add", "s1", str(run_dir), "--tracking-dir", tracking_dir],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Added run", result.output)
            self.assertIn("bench_001", result.output)

    def test_add_with_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            run_dir = self._make_bench_run_dir(tmppath, "bench_002")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            result = CliRunner().invoke(
                bench,
                [
                    "track",
                    "add",
                    "s1",
                    str(run_dir),
                    "-n",
                    "first run",
                    "--tracking-dir",
                    tracking_dir,
                ],
            )
            self.assertEqual(result.exit_code, 0)
            data = json.loads((Path(tracking_dir) / "s1" / "tracking.json").read_text())
            self.assertEqual(data["runs"][0]["notes"], "first run")

    def test_add_with_commit_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            run_dir = self._make_bench_run_dir(tmppath, "bench_003")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            result = CliRunner().invoke(
                bench,
                [
                    "track",
                    "add",
                    "s1",
                    str(run_dir),
                    "--commit",
                    "cpython=abc123",
                    "--tracking-dir",
                    tracking_dir,
                ],
            )
            self.assertEqual(result.exit_code, 0)
            data = json.loads((Path(tracking_dir) / "s1" / "tracking.json").read_text())
            self.assertEqual(data["runs"][0]["commit_info"]["cpython"], "abc123")

    def test_add_nonexistent_dir_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            result = CliRunner().invoke(
                bench,
                [
                    "track",
                    "add",
                    "s1",
                    "/nonexistent/bench_dir",
                    "--tracking-dir",
                    tracking_dir,
                ],
            )
            self.assertNotEqual(result.exit_code, 0)


class TestTrackShow(_TrackTestBase):
    """Tests for bench track show."""

    def test_show_empty_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            result = CliRunner().invoke(
                bench, ["track", "show", "s1", "--tracking-dir", tracking_dir]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No runs yet", result.output)

    def test_show_with_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            run_dir = self._make_bench_run_dir(tmppath, "bench_001")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            CliRunner().invoke(
                bench,
                ["track", "add", "s1", str(run_dir), "--tracking-dir", tracking_dir],
            )
            result = CliRunner().invoke(
                bench, ["track", "show", "s1", "--tracking-dir", tracking_dir]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("bench_001", result.output)
            self.assertIn("Series: s1", result.output)

    def test_show_nonexistent_series_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            result = CliRunner().invoke(
                bench, ["track", "show", "nope", "--tracking-dir", tracking_dir]
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("not found", result.output)

    def test_show_last_n(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            for i in range(1, 4):
                run_dir = self._make_bench_run_dir(
                    tmppath,
                    f"bench_{i:03d}",
                    start_time=f"2026-03-0{i}T10:00:00",
                )
                CliRunner().invoke(
                    bench,
                    [
                        "track",
                        "add",
                        "s1",
                        str(run_dir),
                        "--tracking-dir",
                        tracking_dir,
                    ],
                )
            result = CliRunner().invoke(
                bench,
                ["track", "show", "s1", "--last", "1", "--tracking-dir", tracking_dir],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("bench_003", result.output)
            self.assertNotIn("bench_001", result.output)


class TestTrackPinUnpin(_TrackTestBase):
    """Tests for bench track pin and unpin."""

    def test_pin_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            run_dir = self._make_bench_run_dir(tmppath, "bench_001")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            CliRunner().invoke(
                bench,
                ["track", "add", "s1", str(run_dir), "--tracking-dir", tracking_dir],
            )
            result = CliRunner().invoke(
                bench,
                [
                    "track",
                    "pin",
                    "s1",
                    "bench_001",
                    "--tracking-dir",
                    tracking_dir,
                ],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Pinned", result.output)
            data = json.loads((Path(tracking_dir) / "s1" / "tracking.json").read_text())
            self.assertEqual(data["pinned_baseline_id"], "bench_001")

    def test_pin_nonexistent_bench_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            run_dir = self._make_bench_run_dir(tmppath, "bench_001")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            CliRunner().invoke(
                bench,
                ["track", "add", "s1", str(run_dir), "--tracking-dir", tracking_dir],
            )
            result = CliRunner().invoke(
                bench,
                [
                    "track",
                    "pin",
                    "s1",
                    "nonexistent_id",
                    "--tracking-dir",
                    tracking_dir,
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("not found", result.output)

    def test_unpin_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            run_dir = self._make_bench_run_dir(tmppath, "bench_001")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            CliRunner().invoke(
                bench,
                ["track", "add", "s1", str(run_dir), "--tracking-dir", tracking_dir],
            )
            CliRunner().invoke(
                bench,
                ["track", "pin", "s1", "bench_001", "--tracking-dir", tracking_dir],
            )
            result = CliRunner().invoke(
                bench, ["track", "unpin", "s1", "--tracking-dir", tracking_dir]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Unpinned", result.output)
            data = json.loads((Path(tracking_dir) / "s1" / "tracking.json").read_text())
            self.assertNotIn("pinned_baseline_id", data)


class TestTrackList(_TrackTestBase):
    """Tests for bench track list."""

    def test_list_no_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            result = CliRunner().invoke(bench, ["track", "list", "--tracking-dir", tracking_dir])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No tracking series found", result.output)

    def test_list_shows_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            CliRunner().invoke(
                bench,
                [
                    "track",
                    "init",
                    "alpha",
                    "-d",
                    "Alpha series",
                    "--tracking-dir",
                    tracking_dir,
                ],
            )
            CliRunner().invoke(
                bench,
                [
                    "track",
                    "init",
                    "beta",
                    "-d",
                    "Beta series",
                    "--tracking-dir",
                    tracking_dir,
                ],
            )
            result = CliRunner().invoke(bench, ["track", "list", "--tracking-dir", tracking_dir])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("alpha", result.output)
            self.assertIn("beta", result.output)
            self.assertIn("Alpha series", result.output)


class TestTrackTrend(_TrackTestBase):
    """Tests for bench track trend."""

    def test_trend_nonexistent_series_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            result = CliRunner().invoke(
                bench,
                ["track", "trend", "nope", "--tracking-dir", tracking_dir],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("not found", result.output)

    def test_trend_empty_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            result = CliRunner().invoke(
                bench,
                ["track", "trend", "s1", "--tracking-dir", tracking_dir],
            )
            self.assertEqual(result.exit_code, 0)

    def test_trend_with_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            for i in range(1, 4):
                run_dir = self._make_bench_run_dir(
                    tmppath,
                    f"bench_{i:03d}",
                    start_time=f"2026-03-0{i}T10:00:00",
                    wall_time_base=1.5 + i * 0.1,
                )
                CliRunner().invoke(
                    bench,
                    [
                        "track",
                        "add",
                        "s1",
                        str(run_dir),
                        "--tracking-dir",
                        tracking_dir,
                    ],
                )
            result = CliRunner().invoke(
                bench,
                ["track", "trend", "s1", "--tracking-dir", tracking_dir],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("testpkg", result.output)

    def test_trend_csv_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            for i in range(1, 3):
                run_dir = self._make_bench_run_dir(
                    tmppath,
                    f"bench_{i:03d}",
                    start_time=f"2026-03-0{i}T10:00:00",
                )
                CliRunner().invoke(
                    bench,
                    [
                        "track",
                        "add",
                        "s1",
                        str(run_dir),
                        "--tracking-dir",
                        tracking_dir,
                    ],
                )
            result = CliRunner().invoke(
                bench,
                [
                    "track",
                    "trend",
                    "s1",
                    "--format",
                    "csv",
                    "--tracking-dir",
                    tracking_dir,
                ],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("testpkg", result.output)


class TestTrackAlert(_TrackTestBase):
    """Tests for bench track alert."""

    def test_alert_nonexistent_series_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_dir = str(Path(tmpdir) / "tracking")
            result = CliRunner().invoke(
                bench,
                ["track", "alert", "nope", "--tracking-dir", tracking_dir],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("not found", result.output)

    def test_alert_no_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tracking_dir = str(tmppath / "tracking")
            CliRunner().invoke(bench, ["track", "init", "s1", "--tracking-dir", tracking_dir])
            run_dir = self._make_bench_run_dir(tmppath, "bench_001")
            CliRunner().invoke(
                bench,
                ["track", "add", "s1", str(run_dir), "--tracking-dir", tracking_dir],
            )
            result = CliRunner().invoke(
                bench,
                ["track", "alert", "s1", "--tracking-dir", tracking_dir],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No regression alerts", result.output)


if __name__ == "__main__":
    unittest.main()
