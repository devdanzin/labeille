"""Tests for labeille.analyze — data loading and analysis functions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from labeille.analyze import (
    PackageResult,
    ResultsStore,
    RunData,
    analyze_history,
    analyze_package,
    analyze_registry,
    analyze_run,
    build_reproduce_command,
    categorize_install_errors,
    categorize_skip_reason,
    compare_runs,
    compute_duration_buckets,
    detect_flaky_packages,
    detect_quality_warnings,
)
from labeille.registry import PackageEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_run(
    results_dir: Path,
    run_id: str,
    *,
    python_version: str = "3.15.0a5+ (heads/main:abc1234)",
    jit_enabled: bool = True,
    results: list[dict[str, object]] | None = None,
) -> Path:
    """Create a mock run directory with run_meta.json and results.jsonl."""
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "crashes").mkdir(exist_ok=True)

    meta = {
        "run_id": run_id,
        "started_at": f"{run_id}T00:00:00Z",
        "finished_at": f"{run_id}T01:00:00Z",
        "target_python": "/usr/bin/python3",
        "python_version": python_version,
        "jit_enabled": jit_enabled,
        "hostname": "test",
        "platform": "Linux",
        "packages_tested": len(results or []),
        "packages_skipped": 0,
        "crashes_found": 0,
        "total_duration_seconds": 0.0,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    if results:
        lines = [json.dumps(r) for r in results]
        (run_dir / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return run_dir


def _make_result(
    package: str = "testpkg",
    status: str = "pass",
    duration: float = 10.0,
    **kwargs: object,
) -> dict[str, object]:
    """Create a result dict for testing."""
    data: dict[str, object] = {
        "package": package,
        "status": status,
        "duration_seconds": duration,
        "install_duration_seconds": kwargs.get("install_duration", 2.0),
        "exit_code": kwargs.get("exit_code", 0 if status == "pass" else 1),
        "signal": kwargs.get("signal"),
        "crash_signature": kwargs.get("crash_signature"),
        "test_command": kwargs.get("test_command", "python -m pytest"),
        "timeout_hit": status == "timeout",
        "stderr_tail": kwargs.get("stderr_tail", ""),
        "installed_dependencies": kwargs.get("installed_dependencies", {}),
        "error_message": kwargs.get("error_message"),
        "repo": kwargs.get("repo", f"https://github.com/user/{package}"),
        "git_revision": kwargs.get("git_revision"),
        "timestamp": "2026-02-23T00:00:00Z",
    }
    return data


def _make_pkg(
    name: str = "testpkg",
    skip: bool = False,
    skip_reason: str | None = None,
    enriched: bool = True,
    extension_type: str = "pure",
    test_framework: str = "pytest",
    **kwargs: object,
) -> PackageEntry:
    """Create a PackageEntry for testing."""
    return PackageEntry(
        package=name,
        repo=f"https://github.com/user/{name}",
        extension_type=extension_type,
        enriched=enriched,
        skip=skip,
        skip_reason=skip_reason,
        test_framework=test_framework,
        install_command=str(kwargs.get("install_command", "pip install -e '.[dev]'")),
        test_command=str(kwargs.get("test_command", "python -m pytest")),
        timeout=kwargs.get("timeout"),  # type: ignore[arg-type]
        clone_depth=kwargs.get("clone_depth"),  # type: ignore[arg-type]
        import_name=kwargs.get("import_name"),  # type: ignore[arg-type]
        uses_xdist=bool(kwargs.get("uses_xdist", False)),
        skip_versions=kwargs.get("skip_versions") or {},  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------


class TestRunData(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_lazy_loading(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("alpha"), _make_result("beta")],
        )
        run = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        # Properties should not be loaded yet.
        self.assertIsNone(run._meta)
        self.assertIsNone(run._results)
        # Accessing meta loads it.
        meta = run.meta
        self.assertIsNotNone(run._meta)
        self.assertEqual(meta.run_id, "run1")
        # Accessing results loads them.
        results = run.results
        self.assertIsNotNone(run._results)
        self.assertEqual(len(results), 2)

    def test_result_for(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("alpha"), _make_result("beta")],
        )
        run = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        r = run.result_for("alpha")
        self.assertIsNotNone(r)
        self.assertEqual(r.package, "alpha")  # type: ignore[union-attr]

    def test_result_for_missing(self) -> None:
        _write_run(self.results_dir, "run1", results=[_make_result("alpha")])
        run = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        self.assertIsNone(run.result_for("nonexistent"))

    def test_results_by_status(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[
                _make_result("a", status="pass"),
                _make_result("b", status="fail"),
                _make_result("c", status="pass"),
            ],
        )
        run = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        by_status = run.results_by_status()
        self.assertEqual(len(by_status["pass"]), 2)
        self.assertEqual(len(by_status["fail"]), 1)


class TestResultsStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_list_runs(self) -> None:
        _write_run(self.results_dir, "2026-02-20", results=[_make_result()])
        _write_run(self.results_dir, "2026-02-22", results=[_make_result()])
        store = ResultsStore(self.results_dir)
        runs = store.list_runs()
        self.assertEqual(len(runs), 2)
        # Newest first.
        self.assertEqual(runs[0].run_id, "2026-02-22")

    def test_latest(self) -> None:
        _write_run(self.results_dir, "2026-02-20", results=[_make_result()])
        _write_run(self.results_dir, "2026-02-22", results=[_make_result()])
        store = ResultsStore(self.results_dir)
        latest = store.latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.run_id, "2026-02-22")  # type: ignore[union-attr]

    def test_get_by_id(self) -> None:
        _write_run(self.results_dir, "run1", results=[_make_result()])
        store = ResultsStore(self.results_dir)
        run = store.get("run1")
        self.assertIsNotNone(run)
        self.assertEqual(run.run_id, "run1")  # type: ignore[union-attr]

    def test_get_latest_alias(self) -> None:
        _write_run(self.results_dir, "run1", results=[_make_result()])
        store = ResultsStore(self.results_dir)
        run = store.get("latest")
        self.assertIsNotNone(run)

    def test_empty(self) -> None:
        store = ResultsStore(self.results_dir)
        self.assertEqual(store.list_runs(), [])
        self.assertIsNone(store.latest())

    def test_filter_python_version(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            python_version="3.15.0a5+",
            results=[_make_result()],
        )
        _write_run(
            self.results_dir,
            "run2",
            python_version="3.14.2",
            results=[_make_result()],
        )
        store = ResultsStore(self.results_dir)
        runs = store.list_runs(python_version="3.15")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].run_id, "run1")

    def test_runs_for_package(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("alpha"), _make_result("beta")],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[_make_result("alpha")],
        )
        store = ResultsStore(self.results_dir)
        pairs = store.runs_for_package("alpha")
        self.assertEqual(len(pairs), 2)


# ---------------------------------------------------------------------------
# Registry analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzeRegistry(unittest.TestCase):
    def test_counts(self) -> None:
        packages = [
            _make_pkg("a", skip=False),
            _make_pkg("b", skip=True, skip_reason="PyO3"),
            _make_pkg("c", skip=False),
        ]
        stats = analyze_registry(packages)
        self.assertEqual(stats.total, 3)
        self.assertEqual(stats.active, 2)
        self.assertEqual(stats.skipped, 1)

    def test_extension_types(self) -> None:
        packages = [
            _make_pkg("a", extension_type="pure"),
            _make_pkg("b", extension_type="extensions", skip=True, skip_reason="reason"),
            _make_pkg("c", extension_type="pure"),
        ]
        stats = analyze_registry(packages)
        self.assertEqual(stats.by_extension_type["pure"], (2, 0))
        self.assertEqual(stats.by_extension_type["extensions"], (0, 1))

    def test_with_skip_versions(self) -> None:
        packages = [
            _make_pkg("a", skip_versions={"3.15": "PyO3 not supported"}),
            _make_pkg("b"),
        ]
        stats = analyze_registry(packages, target_python_version="3.15")
        self.assertEqual(stats.active, 1)
        self.assertEqual(stats.skipped, 1)

    def test_without_target_version(self) -> None:
        packages = [
            _make_pkg("a", skip_versions={"3.15": "PyO3 not supported"}),
        ]
        stats = analyze_registry(packages)
        # Without target_python_version, skip_versions doesn't count as skipped.
        self.assertEqual(stats.active, 1)
        self.assertEqual(stats.skipped, 0)


class TestCategorizeSkipReason(unittest.TestCase):
    def test_pyo3(self) -> None:
        self.assertEqual(categorize_skip_reason("PyO3 not supported"), "PyO3/Rust (no 3.15)")

    def test_monorepo(self) -> None:
        self.assertEqual(categorize_skip_reason("Part of monorepo"), "Monorepo")

    def test_no_repo(self) -> None:
        self.assertEqual(categorize_skip_reason("No repo URL available"), "No repo URL")

    def test_other(self) -> None:
        self.assertEqual(categorize_skip_reason("something else entirely"), "Other")

    def test_rust(self) -> None:
        self.assertEqual(categorize_skip_reason("Built with Rust"), "PyO3/Rust (no 3.15)")

    def test_cloud(self) -> None:
        self.assertEqual(
            categorize_skip_reason("Needs cloud credentials"),
            "Cloud/API credentials",
        )


class TestDetectQualityWarnings(unittest.TestCase):
    def test_empty_test_command(self) -> None:
        pkg = _make_pkg("a", test_command="")
        warnings = detect_quality_warnings(pkg)
        self.assertTrue(any("test_command" in w for w in warnings))

    def test_clean(self) -> None:
        pkg = _make_pkg("a")
        warnings = detect_quality_warnings(pkg)
        self.assertEqual(warnings, [])

    def test_skip_reason_on_active(self) -> None:
        pkg = _make_pkg("a", skip_reason="some reason")
        warnings = detect_quality_warnings(pkg)
        self.assertTrue(any("skip_reason" in w for w in warnings))


# ---------------------------------------------------------------------------
# Run analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzeRun(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_status_counts(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[
                _make_result("a", status="pass"),
                _make_result("b", status="fail"),
                _make_result("c", status="crash", signal=11, crash_signature="SIGSEGV"),
            ],
        )
        run = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        analysis = analyze_run(run)
        self.assertEqual(analysis.status_counts.get("pass"), 1)
        self.assertEqual(analysis.status_counts.get("fail"), 1)
        self.assertEqual(analysis.status_counts.get("crash"), 1)

    def test_timing(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[
                _make_result("a", duration=5.0),
                _make_result("b", duration=15.0),
                _make_result("c", duration=10.0),
            ],
        )
        run = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        analysis = analyze_run(run)
        self.assertAlmostEqual(analysis.total_duration, 30.0)
        self.assertAlmostEqual(analysis.avg_duration, 10.0)
        self.assertIsNotNone(analysis.fastest)
        self.assertEqual(analysis.fastest.package, "a")  # type: ignore[union-attr]
        self.assertIsNotNone(analysis.slowest)
        self.assertEqual(analysis.slowest.package, "b")  # type: ignore[union-attr]

    def test_duration_buckets(self) -> None:
        results = [
            PackageResult(package="a", duration_seconds=5.0),
            PackageResult(package="b", duration_seconds=25.0),
            PackageResult(package="c", duration_seconds=120.0),
        ]
        buckets = compute_duration_buckets(results)
        self.assertEqual(buckets[0][0], "0-10s")
        self.assertEqual(buckets[0][1], 1)  # 5s in 0-10s
        self.assertEqual(buckets[1][0], "10-30s")
        self.assertEqual(buckets[1][1], 1)  # 25s in 10-30s
        self.assertEqual(buckets[3][0], "1-5m")
        self.assertEqual(buckets[3][1], 1)  # 120s in 1-5m

    def test_with_previous(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[
                _make_result("a", status="crash", signal=11, crash_signature="SIGSEGV"),
                _make_result("b", status="pass"),
            ],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[
                _make_result("a", status="pass"),
                _make_result("b", status="fail"),
            ],
        )
        old = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        new = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        analysis = analyze_run(new, previous_run=old)
        self.assertIsNotNone(analysis.status_changes)
        self.assertEqual(len(analysis.status_changes), 2)  # type: ignore[arg-type]

    def test_no_previous(self) -> None:
        _write_run(self.results_dir, "run1", results=[_make_result()])
        run = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        analysis = analyze_run(run)
        self.assertIsNone(analysis.status_changes)


class TestBuildReproduceCommand(unittest.TestCase):
    def test_basic(self) -> None:
        result = PackageResult(
            package="urllib3",
            repo="https://github.com/urllib3/urllib3",
            test_command="python -m pytest tests/",
        )
        entry = _make_pkg("urllib3")
        cmd = build_reproduce_command(result, entry, "/path/to/python3.15")
        self.assertIn("git clone", cmd)
        self.assertIn("urllib3", cmd)
        self.assertIn("venv", cmd)
        self.assertIn("PYTHON_JIT=1", cmd)

    def test_with_clone_depth(self) -> None:
        result = PackageResult(package="pkg", repo="https://example.com/repo")
        entry = _make_pkg("pkg", clone_depth=50)
        cmd = build_reproduce_command(result, entry, "/path/to/python")
        self.assertIn("--depth=50", cmd)


class TestCategorizeInstallErrors(unittest.TestCase):
    def test_basic(self) -> None:
        results = [
            PackageResult(
                package="a",
                status="install_error",
                error_message="build error: failed to compile",
            ),
            PackageResult(
                package="b",
                status="install_error",
                error_message="ModuleNotFoundError: no module named 'foo'",
            ),
            PackageResult(package="c", status="pass"),
        ]
        cats = categorize_install_errors(results)
        self.assertIn("Build error", cats)
        self.assertIn("a", cats["Build error"])
        self.assertIn("Import failure", cats)
        self.assertIn("b", cats["Import failure"])
        # "c" is pass, should not appear.
        for pkgs in cats.values():
            self.assertNotIn("c", pkgs)


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------


class TestCompareRuns(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_status_changes(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[
                _make_result("a", status="crash", signal=11, crash_signature="SIGSEGV"),
                _make_result("b", status="pass"),
            ],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[
                _make_result("a", status="pass"),
                _make_result("b", status="fail"),
            ],
        )
        ra = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        rb = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        comp = compare_runs(ra, rb)
        self.assertEqual(len(comp.status_changes), 2)

    def test_packages_only_in_a(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a"), _make_result("b")],
        )
        _write_run(self.results_dir, "run2", results=[_make_result("a")])
        ra = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        rb = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        comp = compare_runs(ra, rb)
        self.assertIn("b", comp.packages_only_in_a)

    def test_packages_only_in_b(self) -> None:
        _write_run(self.results_dir, "run1", results=[_make_result("a")])
        _write_run(
            self.results_dir,
            "run2",
            results=[_make_result("a"), _make_result("c")],
        )
        ra = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        rb = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        comp = compare_runs(ra, rb)
        self.assertIn("c", comp.packages_only_in_b)

    def test_timing_changes(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a", duration=100.0)],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[_make_result("a", duration=200.0)],
        )
        ra = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        rb = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        comp = compare_runs(ra, rb, timing_threshold_pct=20.0, timing_threshold_abs=30.0)
        self.assertEqual(len(comp.timing_changes), 1)
        self.assertAlmostEqual(comp.timing_changes[0].change_pct, 100.0)

    def test_timing_below_threshold(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a", duration=10.0)],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[_make_result("a", duration=15.0)],
        )
        ra = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        rb = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        # 50% change but only 5s absolute — below 30s threshold.
        comp = compare_runs(ra, rb, timing_threshold_pct=20.0, timing_threshold_abs=30.0)
        self.assertEqual(len(comp.timing_changes), 0)

    def test_crash_signature_different(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a", status="crash", crash_signature="SIGSEGV")],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[_make_result("a", status="crash", crash_signature="SIGABRT")],
        )
        ra = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        rb = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        comp = compare_runs(ra, rb)
        self.assertEqual(len(comp.signature_changes), 1)

    def test_identical(self) -> None:
        results = [_make_result("a"), _make_result("b")]
        _write_run(self.results_dir, "run1", results=results)
        _write_run(self.results_dir, "run2", results=results)
        ra = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        rb = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        comp = compare_runs(ra, rb)
        self.assertEqual(len(comp.status_changes), 0)
        self.assertEqual(comp.packages_in_common, 2)


# ---------------------------------------------------------------------------
# History analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzeHistory(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_crash_trend(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[
                _make_result("a", status="crash", crash_signature="sig1"),
                _make_result("b", status="crash", crash_signature="sig2"),
            ],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[
                _make_result("a", status="crash", crash_signature="sig1"),
                _make_result("b", status="pass"),
            ],
        )
        store = ResultsStore(self.results_dir)
        runs = store.list_runs()
        history = analyze_history(runs)
        # Trend is oldest to newest.
        self.assertEqual(history.crash_trend, [2, 1])

    def test_pass_rate(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a", status="pass"), _make_result("b", status="fail")],
        )
        store = ResultsStore(self.results_dir)
        runs = store.list_runs()
        history = analyze_history(runs)
        self.assertAlmostEqual(history.pass_rate_trend[0], 50.0)

    def test_unique_crashes(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[
                _make_result("a", status="crash", crash_signature="sig1"),
                _make_result("b", status="crash", crash_signature="sig2"),
            ],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[
                _make_result("a", status="crash", crash_signature="sig1"),
            ],
        )
        store = ResultsStore(self.results_dir)
        runs = store.list_runs()
        history = analyze_history(runs)
        self.assertEqual(history.total_unique_crashes, 2)

    def test_likely_fixed(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[
                _make_result("a", status="crash", crash_signature="sig1"),
            ],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[
                _make_result("a", status="pass"),
            ],
        )
        store = ResultsStore(self.results_dir)
        runs = store.list_runs()
        history = analyze_history(runs)
        self.assertEqual(history.likely_fixed, 1)
        self.assertEqual(history.currently_reproducing, 0)

    def test_empty(self) -> None:
        history = analyze_history([])
        self.assertEqual(history.crash_trend, [])
        self.assertEqual(history.total_unique_crashes, 0)


class TestDetectFlakyPackages(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_oscillating(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a", status="pass")],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[_make_result("a", status="fail")],
        )
        _write_run(
            self.results_dir,
            "run3",
            results=[_make_result("a", status="pass")],
        )
        store = ResultsStore(self.results_dir)
        runs = store.list_runs()
        flaky = detect_flaky_packages(runs, min_oscillations=2)
        self.assertEqual(len(flaky), 1)
        self.assertEqual(flaky[0][0], "a")

    def test_ignores_crashes(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a", status="pass")],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[_make_result("a", status="crash")],
        )
        _write_run(
            self.results_dir,
            "run3",
            results=[_make_result("a", status="pass")],
        )
        store = ResultsStore(self.results_dir)
        runs = store.list_runs()
        flaky = detect_flaky_packages(runs, min_oscillations=2)
        # Crash is ignored, so pass→pass with crash in middle = 0 oscillations.
        self.assertEqual(len(flaky), 0)

    def test_same_python_only(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            python_version="3.15.0a5+",
            results=[_make_result("a", status="pass")],
        )
        _write_run(
            self.results_dir,
            "run2",
            python_version="3.14.2",
            results=[_make_result("a", status="fail")],
        )
        store = ResultsStore(self.results_dir)
        runs = store.list_runs()
        flaky = detect_flaky_packages(runs, min_oscillations=1)
        # Different Python versions, so they're not compared.
        self.assertEqual(len(flaky), 0)


# ---------------------------------------------------------------------------
# Package history tests
# ---------------------------------------------------------------------------


class TestAnalyzePackage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_basic(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("alpha", status="pass")],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[_make_result("alpha", status="fail")],
        )
        store = ResultsStore(self.results_dir)
        history = analyze_package("alpha", store)
        self.assertEqual(len(history.run_results), 2)

    def test_crash_signatures(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a", status="crash", crash_signature="SIGSEGV")],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[_make_result("a", status="crash", crash_signature="SIGSEGV")],
        )
        store = ResultsStore(self.results_dir)
        history = analyze_package("a", store)
        self.assertEqual(history.crash_signatures.get("SIGSEGV"), 2)

    def test_likely_fixed(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a", status="crash", crash_signature="SIGSEGV")],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[_make_result("a", status="pass")],
        )
        store = ResultsStore(self.results_dir)
        history = analyze_package("a", store)
        self.assertTrue(history.likely_fixed)

    def test_dependency_changes(self) -> None:
        _write_run(
            self.results_dir,
            "run1",
            results=[
                _make_result(
                    "a",
                    status="crash",
                    installed_dependencies={"dep1": "1.0"},
                ),
            ],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[
                _make_result(
                    "a",
                    status="pass",
                    installed_dependencies={"dep1": "2.0"},
                ),
            ],
        )
        store = ResultsStore(self.results_dir)
        history = analyze_package("a", store)
        self.assertEqual(len(history.dependency_changes), 1)

    def test_not_found(self) -> None:
        store = ResultsStore(self.results_dir)
        history = analyze_package("nonexistent", store)
        self.assertEqual(len(history.run_results), 0)
        self.assertFalse(history.likely_fixed)


if __name__ == "__main__":
    unittest.main()
