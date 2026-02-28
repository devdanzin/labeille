"""Tests for labeille.bench.results — benchmark result structures and I/O."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from labeille.bench.results import (
    BenchConditionResult,
    BenchIteration,
    BenchMeta,
    BenchPackageResult,
    ConditionDef,
    append_package_result,
    load_bench_run,
    save_bench_run,
)
from labeille.bench.system import PythonProfile, SystemProfile


# ---------------------------------------------------------------------------
# BenchIteration tests
# ---------------------------------------------------------------------------


class TestBenchIteration(unittest.TestCase):
    """Tests for BenchIteration dataclass."""

    def _make_iteration(self, **kwargs: object) -> BenchIteration:
        """Create a BenchIteration with sensible defaults."""
        defaults: dict[str, object] = {
            "index": 1,
            "warmup": False,
            "wall_time_s": 1.5,
            "user_time_s": 1.2,
            "sys_time_s": 0.1,
            "peak_rss_mb": 120.5,
            "exit_code": 0,
            "status": "ok",
            "outlier": False,
            "load_avg_start": 0.5,
            "load_avg_end": 0.6,
            "ram_available_start_gb": 8.0,
        }
        defaults.update(kwargs)
        return BenchIteration(**defaults)  # type: ignore[arg-type]

    def test_iteration_to_dict_roundtrip(self) -> None:
        """to_dict → from_dict should preserve all fields."""
        it = self._make_iteration()
        d = it.to_dict()
        restored = BenchIteration.from_dict(d)
        self.assertEqual(restored.index, it.index)
        self.assertEqual(restored.warmup, it.warmup)
        self.assertAlmostEqual(restored.wall_time_s, it.wall_time_s, places=5)
        self.assertAlmostEqual(restored.user_time_s, it.user_time_s, places=5)
        self.assertAlmostEqual(restored.sys_time_s, it.sys_time_s, places=5)
        self.assertAlmostEqual(restored.peak_rss_mb, it.peak_rss_mb, places=0)
        self.assertEqual(restored.exit_code, it.exit_code)
        self.assertEqual(restored.status, it.status)
        self.assertEqual(restored.outlier, it.outlier)

    def test_iteration_roundtrip_preserves_types(self) -> None:
        """Types should survive a roundtrip through dict."""
        it = self._make_iteration()
        d = it.to_dict()
        restored = BenchIteration.from_dict(d)
        self.assertIsInstance(restored.index, int)
        self.assertIsInstance(restored.warmup, bool)
        self.assertIsInstance(restored.wall_time_s, float)
        self.assertIsInstance(restored.exit_code, int)
        self.assertIsInstance(restored.status, str)

    def test_iteration_from_dict_ignores_unknown_fields(self) -> None:
        """Unknown keys should be silently dropped."""
        d = {
            "index": 1,
            "warmup": False,
            "wall_time_s": 1.0,
            "user_time_s": 0.5,
            "sys_time_s": 0.1,
            "peak_rss_mb": 50.0,
            "exit_code": 0,
            "status": "ok",
            "future_field": "ignored",
        }
        it = BenchIteration.from_dict(d)
        self.assertEqual(it.index, 1)
        self.assertFalse(hasattr(it, "future_field"))


# ---------------------------------------------------------------------------
# BenchConditionResult tests
# ---------------------------------------------------------------------------


class TestBenchConditionResult(unittest.TestCase):
    """Tests for BenchConditionResult dataclass."""

    def _make_iteration(self, index: int, wall: float, warmup: bool = False) -> BenchIteration:
        return BenchIteration(
            index=index,
            warmup=warmup,
            wall_time_s=wall,
            user_time_s=wall * 0.8,
            sys_time_s=wall * 0.1,
            peak_rss_mb=100.0,
            exit_code=0,
            status="ok",
        )

    def test_condition_compute_stats(self) -> None:
        """Stats should be computed from non-warmup iterations."""
        cond = BenchConditionResult(condition_name="baseline")
        for i in range(1, 6):
            cond.iterations.append(self._make_iteration(i, float(i)))
        cond.compute_stats()
        self.assertIsNotNone(cond.wall_time_stats)
        assert cond.wall_time_stats is not None
        self.assertEqual(cond.wall_time_stats.n, 5)
        self.assertAlmostEqual(cond.wall_time_stats.mean, 3.0, places=5)
        self.assertAlmostEqual(cond.wall_time_stats.median, 3.0, places=5)

    def test_condition_warmup_excluded(self) -> None:
        """Warmup iterations should not affect stats."""
        cond = BenchConditionResult(condition_name="baseline")
        # 2 warmup + 3 measured
        cond.iterations.append(self._make_iteration(1, 100.0, warmup=True))
        cond.iterations.append(self._make_iteration(2, 200.0, warmup=True))
        cond.iterations.append(self._make_iteration(3, 1.0))
        cond.iterations.append(self._make_iteration(4, 2.0))
        cond.iterations.append(self._make_iteration(5, 3.0))
        cond.compute_stats()
        assert cond.wall_time_stats is not None
        self.assertEqual(cond.wall_time_stats.n, 3)
        self.assertAlmostEqual(cond.wall_time_stats.mean, 2.0, places=5)

    def test_condition_outlier_detection(self) -> None:
        """An extreme wall time should be flagged as an outlier."""
        cond = BenchConditionResult(condition_name="baseline")
        for i in range(1, 6):
            cond.iterations.append(self._make_iteration(i, 1.0))
        # Add an extreme value
        cond.iterations.append(self._make_iteration(6, 100.0))
        cond.compute_stats()
        measured = cond.measured_iterations
        outlier_flags = [it.outlier for it in measured]
        self.assertTrue(outlier_flags[-1])  # 100.0 is an outlier

    def test_condition_measured_iterations_property(self) -> None:
        """measured_iterations should filter out warmup."""
        cond = BenchConditionResult(condition_name="test")
        cond.iterations.append(self._make_iteration(1, 1.0, warmup=True))
        cond.iterations.append(self._make_iteration(2, 2.0))
        cond.iterations.append(self._make_iteration(3, 3.0))
        self.assertEqual(len(cond.measured_iterations), 2)

    def test_condition_wall_times_property(self) -> None:
        """wall_times should return a list of floats."""
        cond = BenchConditionResult(condition_name="test")
        cond.iterations.append(self._make_iteration(1, 1.5))
        cond.iterations.append(self._make_iteration(2, 2.5))
        self.assertEqual(cond.wall_times, [1.5, 2.5])

    def test_condition_n_outliers_property(self) -> None:
        """n_outliers should count correctly after compute_stats."""
        cond = BenchConditionResult(condition_name="test")
        for i in range(1, 6):
            cond.iterations.append(self._make_iteration(i, 1.0))
        cond.iterations.append(self._make_iteration(6, 100.0))
        cond.compute_stats()
        self.assertGreater(cond.n_outliers, 0)

    def test_condition_serialization_roundtrip(self) -> None:
        """to_dict → from_dict should preserve structure and recompute stats."""
        cond = BenchConditionResult(condition_name="jit_on")
        cond.install_duration_s = 5.5
        cond.venv_setup_duration_s = 2.3
        for i in range(1, 4):
            cond.iterations.append(self._make_iteration(i, float(i)))
        cond.compute_stats()

        d = cond.to_dict()
        restored = BenchConditionResult.from_dict(d)
        self.assertEqual(restored.condition_name, "jit_on")
        self.assertEqual(len(restored.iterations), 3)
        self.assertAlmostEqual(restored.install_duration_s, 5.5, places=1)
        self.assertAlmostEqual(restored.venv_setup_duration_s, 2.3, places=1)
        # Stats should be recomputed.
        self.assertIsNotNone(restored.wall_time_stats)

    def test_condition_empty_iterations(self) -> None:
        """compute_stats with no iterations should not crash."""
        cond = BenchConditionResult(condition_name="empty")
        cond.compute_stats()
        self.assertIsNone(cond.wall_time_stats)
        self.assertIsNone(cond.user_time_stats)

    def test_condition_only_warmup_iterations(self) -> None:
        """compute_stats with only warmup iterations leaves stats as None."""
        cond = BenchConditionResult(condition_name="warmup_only")
        cond.iterations.append(self._make_iteration(1, 1.0, warmup=True))
        cond.iterations.append(self._make_iteration(2, 2.0, warmup=True))
        cond.compute_stats()
        self.assertIsNone(cond.wall_time_stats)


# ---------------------------------------------------------------------------
# BenchPackageResult tests
# ---------------------------------------------------------------------------


class TestBenchPackageResult(unittest.TestCase):
    """Tests for BenchPackageResult dataclass."""

    def _make_condition(self, name: str, n_iters: int = 3) -> BenchConditionResult:
        cond = BenchConditionResult(condition_name=name)
        for i in range(1, n_iters + 1):
            cond.iterations.append(
                BenchIteration(
                    index=i,
                    warmup=False,
                    wall_time_s=float(i),
                    user_time_s=float(i) * 0.8,
                    sys_time_s=float(i) * 0.1,
                    peak_rss_mb=100.0,
                    exit_code=0,
                    status="ok",
                )
            )
        cond.compute_stats()
        return cond

    def test_package_result_to_dict_roundtrip(self) -> None:
        """Two conditions should survive a roundtrip."""
        pkg = BenchPackageResult(package="requests", clone_duration_s=1.5)
        pkg.conditions["baseline"] = self._make_condition("baseline")
        pkg.conditions["jit"] = self._make_condition("jit")

        d = pkg.to_dict()
        restored = BenchPackageResult.from_dict(d)
        self.assertEqual(restored.package, "requests")
        self.assertAlmostEqual(restored.clone_duration_s, 1.5, places=1)
        self.assertIn("baseline", restored.conditions)
        self.assertIn("jit", restored.conditions)
        self.assertEqual(len(restored.conditions["baseline"].iterations), 3)

    def test_package_result_jsonl_roundtrip(self) -> None:
        """JSONL serialization should preserve all fields."""
        pkg = BenchPackageResult(package="click", clone_duration_s=2.0)
        pkg.conditions["baseline"] = self._make_condition("baseline", n_iters=2)

        line = pkg.to_jsonl_line()
        restored = BenchPackageResult.from_jsonl_line(line)
        self.assertEqual(restored.package, "click")
        self.assertAlmostEqual(restored.clone_duration_s, 2.0, places=1)
        self.assertEqual(len(restored.conditions["baseline"].iterations), 2)

    def test_package_result_skipped(self) -> None:
        """Skipped packages should roundtrip correctly."""
        pkg = BenchPackageResult(
            package="numpy",
            skipped=True,
            skip_reason="no repo",
        )
        d = pkg.to_dict()
        restored = BenchPackageResult.from_dict(d)
        self.assertTrue(restored.skipped)
        self.assertEqual(restored.skip_reason, "no repo")


# ---------------------------------------------------------------------------
# ConditionDef tests
# ---------------------------------------------------------------------------


class TestConditionDef(unittest.TestCase):
    """Tests for ConditionDef dataclass."""

    def test_condition_def_to_dict_minimal(self) -> None:
        """Minimal condition should only have 'name' key."""
        cond = ConditionDef(name="baseline")
        d = cond.to_dict()
        self.assertEqual(d, {"name": "baseline"})

    def test_condition_def_to_dict_full(self) -> None:
        """All fields set should all be present in dict."""
        cond = ConditionDef(
            name="jit_on",
            description="With JIT enabled",
            target_python="/usr/bin/python3.15",
            env={"PYTHON_JIT": "1"},
            extra_deps=["coverage"],
            test_command_override="coverage run -m pytest",
            test_command_prefix="nice -n 19",
            test_command_suffix="-v --tb=short",
            install_command="pip install -e .",
        )
        d = cond.to_dict()
        self.assertEqual(d["name"], "jit_on")
        self.assertEqual(d["description"], "With JIT enabled")
        self.assertEqual(d["target_python"], "/usr/bin/python3.15")
        self.assertEqual(d["env"], {"PYTHON_JIT": "1"})
        self.assertEqual(d["extra_deps"], ["coverage"])
        self.assertEqual(d["test_command_override"], "coverage run -m pytest")
        self.assertEqual(d["test_command_prefix"], "nice -n 19")
        self.assertEqual(d["test_command_suffix"], "-v --tb=short")
        self.assertEqual(d["install_command"], "pip install -e .")

    def test_condition_def_roundtrip(self) -> None:
        """from_dict(to_dict(x)) should produce equivalent object."""
        cond = ConditionDef(
            name="test",
            description="A test condition",
            env={"FOO": "bar"},
            extra_deps=["dep1", "dep2"],
        )
        d = cond.to_dict()
        restored = ConditionDef.from_dict(d)
        self.assertEqual(restored.name, cond.name)
        self.assertEqual(restored.description, cond.description)
        self.assertEqual(restored.env, cond.env)
        self.assertEqual(restored.extra_deps, cond.extra_deps)
        self.assertIsNone(restored.test_command_override)


# ---------------------------------------------------------------------------
# BenchMeta tests
# ---------------------------------------------------------------------------


class TestBenchMeta(unittest.TestCase):
    """Tests for BenchMeta dataclass."""

    def test_meta_roundtrip(self) -> None:
        """Full BenchMeta should survive a to_dict/from_dict roundtrip."""
        meta = BenchMeta(
            bench_id="bench-20260227-001",
            name="JIT overhead",
            description="Measure JIT overhead on top-50 packages",
            config={"iterations": 5, "warmup": 1},
            cli_args=["bench", "--iterations", "5"],
            start_time="2026-02-27T10:00:00",
            end_time="2026-02-27T12:00:00",
            packages_total=50,
            packages_completed=48,
            packages_skipped=2,
        )
        meta.conditions["baseline"] = ConditionDef(
            name="baseline",
            description="Without JIT",
            env={"PYTHON_JIT": "0"},
        )
        meta.conditions["jit"] = ConditionDef(
            name="jit",
            description="With JIT",
            env={"PYTHON_JIT": "1"},
        )

        d = meta.to_dict()
        restored = BenchMeta.from_dict(d)
        self.assertEqual(restored.bench_id, "bench-20260227-001")
        self.assertEqual(restored.name, "JIT overhead")
        self.assertEqual(restored.packages_total, 50)
        self.assertEqual(restored.packages_completed, 48)
        self.assertIn("baseline", restored.conditions)
        self.assertIn("jit", restored.conditions)

    def test_meta_with_python_profiles(self) -> None:
        """PythonProfile entries should survive roundtrip."""
        meta = BenchMeta(bench_id="test-001")
        meta.python_profiles["baseline"] = PythonProfile(
            path="/usr/bin/python3.15",
            version="3.15.0a4",
            implementation="CPython",
            jit_enabled=False,
        )
        meta.python_profiles["jit"] = PythonProfile(
            path="/usr/bin/python3.15",
            version="3.15.0a4",
            implementation="CPython",
            jit_enabled=True,
        )

        d = meta.to_dict()
        restored = BenchMeta.from_dict(d)
        self.assertIn("baseline", restored.python_profiles)
        self.assertIn("jit", restored.python_profiles)
        self.assertFalse(restored.python_profiles["baseline"].jit_enabled)
        self.assertTrue(restored.python_profiles["jit"].jit_enabled)

    def test_meta_with_system_profile(self) -> None:
        """SystemProfile should survive roundtrip."""
        meta = BenchMeta(bench_id="test-002")
        meta.system = SystemProfile(
            cpu_model="Intel Core i9",
            cpu_cores_physical=8,
            ram_total_gb=32.0,
        )

        d = meta.to_dict()
        restored = BenchMeta.from_dict(d)
        self.assertEqual(restored.system.cpu_model, "Intel Core i9")
        self.assertEqual(restored.system.cpu_cores_physical, 8)
        self.assertAlmostEqual(restored.system.ram_total_gb, 32.0, places=1)

    def test_meta_defaults(self) -> None:
        """Default values should be sane."""
        meta = BenchMeta(bench_id="minimal")
        d = meta.to_dict()
        self.assertEqual(d["bench_id"], "minimal")
        self.assertEqual(d["name"], "")
        self.assertEqual(d["packages_total"], 0)
        self.assertEqual(d["conditions"], {})
        self.assertEqual(d["python_profiles"], {})


# ---------------------------------------------------------------------------
# I/O tests
# ---------------------------------------------------------------------------


class TestIO(unittest.TestCase):
    """Tests for save_bench_run, load_bench_run, append_package_result."""

    def _make_package_result(self, name: str) -> BenchPackageResult:
        """Create a BenchPackageResult with one condition and iterations."""
        cond = BenchConditionResult(condition_name="baseline")
        for i in range(1, 4):
            cond.iterations.append(
                BenchIteration(
                    index=i,
                    warmup=(i == 1),
                    wall_time_s=float(i),
                    user_time_s=float(i) * 0.8,
                    sys_time_s=float(i) * 0.1,
                    peak_rss_mb=100.0,
                    exit_code=0,
                    status="ok",
                )
            )
        cond.compute_stats()
        pkg = BenchPackageResult(package=name, clone_duration_s=1.0)
        pkg.conditions["baseline"] = cond
        return pkg

    def test_save_and_load_bench_run(self) -> None:
        """Save and load should produce equivalent data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "bench-001"
            meta = BenchMeta(
                bench_id="bench-001",
                name="Test run",
                packages_total=2,
                packages_completed=2,
            )
            results = [
                self._make_package_result("requests"),
                self._make_package_result("click"),
            ]

            save_bench_run(out_dir, meta, results)

            # Verify files exist.
            self.assertTrue((out_dir / "bench_meta.json").exists())
            self.assertTrue((out_dir / "bench_results.jsonl").exists())

            loaded_meta, loaded_results = load_bench_run(out_dir)
            self.assertEqual(loaded_meta.bench_id, "bench-001")
            self.assertEqual(loaded_meta.name, "Test run")
            self.assertEqual(len(loaded_results), 2)
            self.assertEqual(loaded_results[0].package, "requests")
            self.assertEqual(loaded_results[1].package, "click")

    def test_load_missing_dir(self) -> None:
        """Loading from a nonexistent directory should raise."""
        with self.assertRaises(FileNotFoundError):
            load_bench_run(Path("/nonexistent/bench-dir"))

    def test_append_package_result(self) -> None:
        """Appending results should produce valid JSONL lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "bench_results.jsonl"
            append_package_result(results_path, self._make_package_result("flask"))
            append_package_result(results_path, self._make_package_result("django"))

            lines = results_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)

            pkg1 = BenchPackageResult.from_jsonl_line(lines[0])
            pkg2 = BenchPackageResult.from_jsonl_line(lines[1])
            self.assertEqual(pkg1.package, "flask")
            self.assertEqual(pkg2.package, "django")

    def test_incremental_write_readable(self) -> None:
        """save_bench_run then append_package_result should all load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "bench-inc"
            meta = BenchMeta(bench_id="bench-inc", packages_total=3)
            initial_results = [self._make_package_result("requests")]

            save_bench_run(out_dir, meta, initial_results)

            # Append a second result incrementally.
            results_path = out_dir / "bench_results.jsonl"
            append_package_result(results_path, self._make_package_result("click"))

            loaded_meta, loaded_results = load_bench_run(out_dir)
            self.assertEqual(loaded_meta.bench_id, "bench-inc")
            self.assertEqual(len(loaded_results), 2)
            self.assertEqual(loaded_results[0].package, "requests")
            self.assertEqual(loaded_results[1].package, "click")

    def test_save_creates_directories(self) -> None:
        """save_bench_run should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_dir = Path(tmpdir) / "a" / "b" / "c"
            meta = BenchMeta(bench_id="deep")
            save_bench_run(deep_dir, meta, [])
            self.assertTrue((deep_dir / "bench_meta.json").exists())

    def test_load_empty_results(self) -> None:
        """Loading with no JSONL file should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "bench-empty"
            meta = BenchMeta(bench_id="empty")
            save_bench_run(out_dir, meta, [])

            loaded_meta, loaded_results = load_bench_run(out_dir)
            self.assertEqual(loaded_meta.bench_id, "empty")
            self.assertEqual(len(loaded_results), 0)

    def test_meta_json_is_valid(self) -> None:
        """bench_meta.json should be valid, pretty-printed JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "bench-json"
            meta = BenchMeta(bench_id="json-test", name="Pretty")
            save_bench_run(out_dir, meta, [])

            content = (out_dir / "bench_meta.json").read_text()
            parsed = json.loads(content)
            self.assertEqual(parsed["bench_id"], "json-test")
            # Should be indented (pretty-printed).
            self.assertIn("\n", content)


if __name__ == "__main__":
    unittest.main()
