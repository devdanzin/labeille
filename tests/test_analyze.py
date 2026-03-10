"""Tests for labeille.analyze — data loading and analysis functions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from labeille.analyze import (
    PackageComparison,
    PackageResult,
    ResultsStore,
    RunData,
    StatusChange,
    _classify_compat_blocker,
    _classify_install_complexity,
    _classify_repo_host,
    analyze_history,
    analyze_package,
    analyze_run,
    build_reproduce_command,
    categorize_install_errors,
    categorize_skip_reason,
    compare_runs,
    compute_duration_buckets,
    detect_flaky_packages,
    detect_quality_warnings,
    generate_registry_report,
)
from labeille.registry import Index, IndexEntry, PackageEntry


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

    def test_result_for_cache_built_once(self) -> None:
        """The _results_by_pkg dict is built lazily on first call."""
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("alpha"), _make_result("beta")],
        )
        run = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        self.assertIsNone(run._results_by_pkg)
        run.result_for("alpha")
        self.assertIsNotNone(run._results_by_pkg)
        # Second call reuses the same dict (no rebuild).
        cache = run._results_by_pkg
        run.result_for("beta")
        self.assertIs(run._results_by_pkg, cache)

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

    def test_uses_path_export(self) -> None:
        """Reproduce script uses PATH export instead of string replacement."""
        result = PackageResult(
            package="mypkg",
            repo="https://github.com/user/mypkg",
            test_command="python -m pytest tests/",
        )
        entry = _make_pkg("mypkg", install_command="pip install -e .")
        cmd = build_reproduce_command(result, entry, "/opt/python")
        self.assertIn('export PATH="$PWD/.venv/bin:$PATH"', cmd)
        # Install and test commands appear unmodified.
        self.assertIn("pip install -e .", cmd)
        self.assertIn("python -m pytest tests/", cmd)
        # No .venv/bin/pip or .venv/bin/python string replacements.
        self.assertNotIn(".venv/bin/pip", cmd)
        self.assertNotIn(".venv/bin/python", cmd)

    def test_sdist_mode(self) -> None:
        result = PackageResult(
            package="urllib3",
            repo="https://github.com/urllib3/urllib3",
            test_command="python -m pytest tests/",
            install_from="sdist",
        )
        entry = _make_pkg("urllib3", install_command="pip install -e . && pip install pytest")
        cmd = build_reproduce_command(result, entry, "/opt/python")
        self.assertIn("pip install --no-binary urllib3 urllib3", cmd)
        self.assertIn("pip install pytest", cmd)
        self.assertNotIn("pip install -e .", cmd)

    def test_sdist_mode_with_extras(self) -> None:
        result = PackageResult(
            package="click",
            repo="https://github.com/pallets/click",
            test_command="python -m pytest tests/",
            install_from="sdist",
        )
        entry = _make_pkg("click", install_command='pip install -e ".[test]"')
        cmd = build_reproduce_command(result, entry, "/opt/python")
        self.assertIn("pip install --no-binary click 'click[test]'", cmd)

    def test_source_mode_unchanged(self) -> None:
        result = PackageResult(
            package="urllib3",
            repo="https://github.com/urllib3/urllib3",
            test_command="python -m pytest tests/",
            install_from="source",
        )
        entry = _make_pkg("urllib3", install_command="pip install -e .")
        cmd = build_reproduce_command(result, entry, "/opt/python")
        self.assertIn("pip install -e .", cmd)


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


# ---------------------------------------------------------------------------
# Commit-aware comparison tests
# ---------------------------------------------------------------------------


class TestStatusChangeCommits(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_status_change_includes_commits(self) -> None:
        """Status changes include old_commit and new_commit from git_revision."""
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a", status="pass", git_revision="abc1234")],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[
                _make_result(
                    "a",
                    status="crash",
                    signal=11,
                    crash_signature="SIGSEGV",
                    git_revision="def5678",
                ),
            ],
        )
        old = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        new = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        analysis = analyze_run(new, previous_run=old)
        self.assertIsNotNone(analysis.status_changes)
        changes = analysis.status_changes
        assert changes is not None
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].old_commit, "abc1234")
        self.assertEqual(changes[0].new_commit, "def5678")

    def test_status_change_missing_commit(self) -> None:
        """Results without git_revision produce None commit fields."""
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
        old = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        new = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        analysis = analyze_run(new, previous_run=old)
        self.assertIsNotNone(analysis.status_changes)
        changes = analysis.status_changes
        assert changes is not None
        self.assertEqual(len(changes), 1)
        self.assertIsNone(changes[0].old_commit)
        self.assertIsNone(changes[0].new_commit)

    def test_status_change_has_commit_fields(self) -> None:
        """StatusChange dataclass has old_commit and new_commit fields."""
        sc = StatusChange(
            package="pkg",
            old_status="pass",
            new_status="crash",
            old_commit="aaa1111",
            new_commit="bbb2222",
        )
        self.assertEqual(sc.old_commit, "aaa1111")
        self.assertEqual(sc.new_commit, "bbb2222")


class TestPackageComparison(unittest.TestCase):
    def test_commit_changed(self) -> None:
        pc = PackageComparison(
            package="pkg",
            status_a="pass",
            status_b="crash",
            duration_a=10.0,
            duration_b=12.0,
            commit_a="abc1234",
            commit_b="def5678",
        )
        self.assertTrue(pc.commit_changed)
        self.assertFalse(pc.commit_unchanged)

    def test_commit_unchanged(self) -> None:
        pc = PackageComparison(
            package="pkg",
            status_a="pass",
            status_b="crash",
            duration_a=10.0,
            duration_b=12.0,
            commit_a="abc1234",
            commit_b="abc1234",
        )
        self.assertFalse(pc.commit_changed)
        self.assertTrue(pc.commit_unchanged)

    def test_commit_unknown(self) -> None:
        pc = PackageComparison(
            package="pkg",
            status_a="pass",
            status_b="crash",
            duration_a=10.0,
            duration_b=12.0,
            commit_a=None,
            commit_b="abc1234",
        )
        self.assertFalse(pc.commit_changed)
        self.assertFalse(pc.commit_unchanged)

    def test_both_none(self) -> None:
        pc = PackageComparison(
            package="pkg",
            status_a="pass",
            status_b="pass",
            duration_a=10.0,
            duration_b=12.0,
            commit_a=None,
            commit_b=None,
        )
        self.assertFalse(pc.commit_changed)
        self.assertFalse(pc.commit_unchanged)


class TestCompareRunsCommitInfo(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_compare_runs_populates_commit_info(self) -> None:
        """compare_runs populates package_details with commit info."""
        _write_run(
            self.results_dir,
            "run1",
            results=[
                _make_result("a", status="pass", git_revision="aaa1111"),
                _make_result("b", status="pass", git_revision="bbb1111"),
            ],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[
                _make_result(
                    "a",
                    status="crash",
                    signal=11,
                    crash_signature="SIGSEGV",
                    git_revision="aaa1111",
                ),
                _make_result(
                    "b",
                    status="crash",
                    signal=6,
                    crash_signature="SIGABRT",
                    git_revision="bbb2222",
                ),
            ],
        )
        ra = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        rb = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        comp = compare_runs(ra, rb)

        self.assertEqual(len(comp.package_details), 2)
        detail_a = next(d for d in comp.package_details if d.package == "a")
        detail_b = next(d for d in comp.package_details if d.package == "b")

        self.assertEqual(detail_a.commit_a, "aaa1111")
        self.assertEqual(detail_a.commit_b, "aaa1111")
        self.assertTrue(detail_a.commit_unchanged)

        self.assertEqual(detail_b.commit_a, "bbb1111")
        self.assertEqual(detail_b.commit_b, "bbb2222")
        self.assertTrue(detail_b.commit_changed)

    def test_compare_runs_status_changes_have_commits(self) -> None:
        """Status changes from compare_runs include commit fields."""
        _write_run(
            self.results_dir,
            "run1",
            results=[_make_result("a", status="pass", git_revision="abc1234")],
        )
        _write_run(
            self.results_dir,
            "run2",
            results=[
                _make_result("a", status="fail", git_revision="def5678"),
            ],
        )
        ra = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        rb = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        comp = compare_runs(ra, rb)
        self.assertEqual(len(comp.status_changes), 1)
        self.assertEqual(comp.status_changes[0].old_commit, "abc1234")
        self.assertEqual(comp.status_changes[0].new_commit, "def5678")

    def test_compare_runs_missing_commits(self) -> None:
        """Missing git_revision results in None commit fields."""
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
        ra = RunData(run_id="run1", run_dir=self.results_dir / "run1")
        rb = RunData(run_id="run2", run_dir=self.results_dir / "run2")
        comp = compare_runs(ra, rb)
        self.assertEqual(len(comp.package_details), 1)
        self.assertIsNone(comp.package_details[0].commit_a)
        self.assertIsNone(comp.package_details[0].commit_b)


# ---------------------------------------------------------------------------
# Registry report tests
# ---------------------------------------------------------------------------


class TestGenerateRegistryReport(unittest.TestCase):
    def test_report_basic_counts(self) -> None:
        packages = [
            _make_pkg("a"),
            _make_pkg("b"),
            _make_pkg("c"),
            _make_pkg("d", skip=True, skip_reason="PyO3"),
            _make_pkg("e", skip=True, skip_reason="no repo"),
        ]
        report = generate_registry_report(packages)
        self.assertEqual(report.total, 5)
        self.assertEqual(report.active, 3)
        self.assertEqual(report.skipped, 2)

    def test_report_enrichment_progress(self) -> None:
        packages = [
            _make_pkg("a", enriched=True),
            _make_pkg("b", enriched=True),
            _make_pkg("c", enriched=True, skip=True, skip_reason="x"),
            _make_pkg("d", enriched=True),
            _make_pkg("e", enriched=False),
        ]
        report = generate_registry_report(packages)
        self.assertEqual(report.enrichment.enriched, 4)
        self.assertEqual(report.enrichment.enriched_active, 3)
        self.assertEqual(report.enrichment.enriched_skipped, 1)
        self.assertEqual(report.enrichment.not_enriched, 1)

    def test_report_extension_types(self) -> None:
        packages = [
            _make_pkg("a", extension_type="pure"),
            _make_pkg("b", extension_type="pure", skip=True, skip_reason="x"),
            _make_pkg("c", extension_type="c_extension"),
        ]
        report = generate_registry_report(packages)
        self.assertEqual(report.by_extension_type["pure"], (1, 1))
        self.assertEqual(report.by_extension_type["c_extension"], (1, 0))

    def test_report_per_version_auto_detect(self) -> None:
        packages = [
            _make_pkg("a", skip_versions={"3.15": "PyO3", "3.14": "broken"}),
            _make_pkg("b"),
        ]
        report = generate_registry_report(packages)
        versions = [va.version for va in report.per_version]
        self.assertIn("3.15", versions)
        self.assertIn("3.14", versions)

    def test_report_per_version_explicit(self) -> None:
        packages = [
            _make_pkg("a", skip_versions={"3.15": "PyO3", "3.14": "broken"}),
        ]
        report = generate_registry_report(packages, target_python_versions=["3.15"])
        self.assertEqual(len(report.per_version), 1)
        self.assertEqual(report.per_version[0].version, "3.15")

    def test_report_per_version_counts(self) -> None:
        packages = [
            _make_pkg("a", skip_versions={"3.15": "PyO3"}),
            _make_pkg("b", skip=True, skip_reason="no repo"),
            _make_pkg("c"),
        ]
        report = generate_registry_report(packages, target_python_versions=["3.15"])
        va = report.per_version[0]
        self.assertEqual(va.total_active, 1)
        self.assertEqual(va.skipped, 2)

    def test_report_repo_hosts(self) -> None:
        packages = [
            _make_pkg("a"),  # default: github
            _make_pkg("b"),  # default: github
        ]
        packages[0].repo = "https://github.com/user/a"
        packages[1].repo = "https://gitlab.com/user/b"
        pkg_none = _make_pkg("c")
        pkg_none.repo = None
        packages.append(pkg_none)
        report = generate_registry_report(packages)
        self.assertEqual(report.repo_hosts.github, 1)
        self.assertEqual(report.repo_hosts.gitlab, 1)
        self.assertEqual(report.repo_hosts.no_repo, 1)

    def test_report_install_complexity(self) -> None:
        packages = [
            _make_pkg("a", install_command="pip install -e ."),
            _make_pkg("b", install_command="pip install -e '.[test]'"),
            _make_pkg("c", install_command="pip install -e . && pip install pytest"),
            _make_pkg("d", install_command="make install"),
        ]
        report = generate_registry_report(packages)
        self.assertEqual(report.install_complexity.simple_editable, 1)
        self.assertEqual(report.install_complexity.editable_with_extras, 1)
        self.assertEqual(report.install_complexity.multi_step, 1)
        self.assertEqual(report.install_complexity.custom, 1)

    def test_report_install_git_fetch_tags(self) -> None:
        packages = [
            _make_pkg(
                "a",
                install_command="git fetch --tags --depth 1 && pip install -e .",
            ),
        ]
        report = generate_registry_report(packages)
        self.assertEqual(report.install_complexity.has_git_fetch_tags, 1)
        self.assertEqual(report.install_complexity.multi_step, 1)

    def test_report_compat_blockers(self) -> None:
        packages = [
            _make_pkg("a", skip=True, skip_reason="PyO3 not supported"),
            _make_pkg("b", skip=True, skip_reason="Cython build failure"),
            _make_pkg("c", skip=True, skip_reason="meson build error"),
        ]
        report = generate_registry_report(packages)
        self.assertEqual(report.compat_blockers.pyo3_rust, 1)
        self.assertEqual(report.compat_blockers.cython, 1)
        self.assertEqual(report.compat_blockers.meson, 1)

    def test_report_compat_blockers_dedup(self) -> None:
        """Package with same blocker in skip_reason and skip_versions counted once."""
        packages = [
            _make_pkg(
                "a",
                skip=True,
                skip_reason="PyO3 not supported",
                skip_versions={"3.15": "PyO3 build fails"},
            ),
        ]
        report = generate_registry_report(packages)
        self.assertEqual(report.compat_blockers.pyo3_rust, 1)

    def test_report_compat_blockers_package_lists(self) -> None:
        packages = [
            _make_pkg("cryptography", skip=True, skip_reason="PyO3"),
            _make_pkg("orjson", skip=True, skip_reason="maturin/Rust"),
        ]
        report = generate_registry_report(packages, collect_package_lists=True)
        self.assertIn("pyo3_rust", report.compat_blockers.packages_by_blocker)
        names = report.compat_blockers.packages_by_blocker["pyo3_rust"]
        self.assertIn("cryptography", names)
        self.assertIn("orjson", names)

    def test_report_download_tiers(self) -> None:
        packages = [_make_pkg(f"pkg{i}") for i in range(150)]
        entries = [IndexEntry(name=f"pkg{i}", download_count=10000 - i) for i in range(150)]
        index = Index(packages=entries)
        report = generate_registry_report(packages, index=index)
        self.assertEqual(report.download_tiers.top_100[1], 100)
        self.assertEqual(report.download_tiers.top_100[0], 100)

    def test_report_download_tiers_no_index(self) -> None:
        packages = [_make_pkg("a")]
        report = generate_registry_report(packages, index=None)
        self.assertEqual(report.download_tiers.all_packages, (0, 0))

    def test_report_test_framework(self) -> None:
        packages = [
            _make_pkg("a", test_framework="pytest"),
            _make_pkg("b", test_framework="pytest"),
            _make_pkg("c", test_framework="unittest"),
        ]
        report = generate_registry_report(packages)
        self.assertEqual(report.by_test_framework["pytest"], 2)
        self.assertEqual(report.by_test_framework["unittest"], 1)

    def test_report_notable_attributes(self) -> None:
        packages = [_make_pkg("a", timeout=300, uses_xdist=True)]
        report = generate_registry_report(packages)
        self.assertEqual(report.notable["Custom timeout"], 1)
        self.assertEqual(report.notable["uses_xdist"], 1)

    def test_report_quality_warnings(self) -> None:
        packages = [_make_pkg("a", enriched=True, test_command="")]
        report = generate_registry_report(packages)
        self.assertTrue(len(report.quality_warnings) > 0)

    def test_report_generated_at(self) -> None:
        packages = [_make_pkg("a")]
        report = generate_registry_report(packages)
        self.assertTrue(len(report.generated_at) > 0)

    def test_report_empty_packages(self) -> None:
        report = generate_registry_report([])
        self.assertEqual(report.total, 0)
        self.assertEqual(report.active, 0)
        self.assertEqual(report.skipped, 0)


class TestClassifyRepoHost(unittest.TestCase):
    def test_github(self) -> None:
        self.assertEqual(_classify_repo_host("https://github.com/user/repo"), "github")

    def test_gitlab(self) -> None:
        self.assertEqual(_classify_repo_host("https://gitlab.com/user/repo"), "gitlab")

    def test_self_hosted_gitlab(self) -> None:
        self.assertEqual(_classify_repo_host("https://gitlab.example.com/repo"), "gitlab")

    def test_bitbucket(self) -> None:
        self.assertEqual(_classify_repo_host("https://bitbucket.org/user/repo"), "bitbucket")

    def test_codeberg(self) -> None:
        self.assertEqual(_classify_repo_host("https://codeberg.org/user/repo"), "codeberg")

    def test_no_repo(self) -> None:
        self.assertEqual(_classify_repo_host(None), "no_repo")

    def test_other(self) -> None:
        self.assertEqual(_classify_repo_host("https://example.com/repo"), "other")


class TestClassifyInstallComplexity(unittest.TestCase):
    def test_simple_editable(self) -> None:
        self.assertEqual(_classify_install_complexity("pip install -e ."), "simple_editable")

    def test_editable_with_extras(self) -> None:
        self.assertEqual(
            _classify_install_complexity("pip install -e '.[test]'"),
            "editable_with_extras",
        )

    def test_multi_step(self) -> None:
        self.assertEqual(
            _classify_install_complexity("pip install -e . && pip install pytest"),
            "multi_step",
        )

    def test_custom(self) -> None:
        self.assertEqual(_classify_install_complexity("make install"), "custom")

    def test_empty(self) -> None:
        self.assertEqual(_classify_install_complexity(""), "custom")


class TestClassifyCompatBlocker(unittest.TestCase):
    def test_pyo3(self) -> None:
        self.assertEqual(_classify_compat_blocker("PyO3 not supported"), "pyo3_rust")

    def test_rust(self) -> None:
        self.assertEqual(_classify_compat_blocker("Built with Rust via maturin"), "pyo3_rust")

    def test_cython(self) -> None:
        self.assertEqual(_classify_compat_blocker("Cython build failure"), "cython")

    def test_meson(self) -> None:
        self.assertEqual(_classify_compat_blocker("Meson build error"), "meson")

    def test_cmake(self) -> None:
        self.assertEqual(_classify_compat_blocker("CMake error"), "cmake")

    def test_fortran(self) -> None:
        self.assertEqual(_classify_compat_blocker("Uses f2py"), "fortran")

    def test_c_api(self) -> None:
        self.assertEqual(_classify_compat_blocker("Removed C API: tp_print"), "c_api_removed")

    def test_no_support(self) -> None:
        self.assertEqual(_classify_compat_blocker("No 3.15 support"), "no_python_support")

    def test_other_build(self) -> None:
        self.assertEqual(_classify_compat_blocker("c++ compilation error"), "other_build")

    def test_unrelated(self) -> None:
        self.assertIsNone(_classify_compat_blocker("Part of monorepo"))


if __name__ == "__main__":
    unittest.main()
