"""Integration tests for labeille.analyze_cli — CLI commands."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from click.testing import CliRunner

from labeille.analyze_cli import analyze


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
) -> None:
    """Create a mock run directory."""
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


def _make_result(
    package: str = "testpkg",
    status: str = "pass",
    duration: float = 10.0,
    **kwargs: object,
) -> dict[str, object]:
    """Create a result dict for testing."""
    return {
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


def _write_package(
    registry_dir: Path,
    name: str,
    *,
    skip: bool = False,
    skip_reason: str | None = None,
    extension_type: str = "pure",
) -> None:
    """Create a package YAML file in the registry."""
    data = {
        "package": name,
        "repo": f"https://github.com/user/{name}",
        "pypi_url": f"https://pypi.org/project/{name}/",
        "extension_type": extension_type,
        "python_versions": [],
        "install_method": "pip",
        "install_command": "pip install -e '.[dev]'",
        "test_command": "python -m pytest tests/",
        "test_framework": "pytest",
        "uses_xdist": False,
        "timeout": None,
        "skip": skip,
        "skip_reason": skip_reason,
        "skip_versions": {},
        "notes": "",
        "enriched": True,
        "clone_depth": None,
        "import_name": None,
    }
    pkg_dir = registry_dir / "packages"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / f"{name}.yaml").write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FixtureMixin:
    """Mixin that sets up a temp directory with mock runs and registry."""

    def _setup_fixtures(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.results_dir = self.base / "results"
        self.registry_dir = self.base / "registry"
        self.results_dir.mkdir()
        self.registry_dir.mkdir()

        # Create packages.
        for name in ["alpha", "beta", "gamma", "delta", "epsilon"]:
            _write_package(self.registry_dir, name)
        _write_package(self.registry_dir, "skipped", skip=True, skip_reason="PyO3 not supported")

        # Create runs.
        _write_run(
            self.results_dir,
            "2026-02-20T10-00-00",
            results=[
                _make_result("alpha", status="crash", signal=11, crash_signature="SIGSEGV"),
                _make_result("beta", status="pass", duration=20.0),
                _make_result("gamma", status="fail", duration=15.0),
                _make_result("delta", status="install_error", error_message="build error"),
                _make_result("epsilon", status="timeout", duration=600.0),
            ],
        )
        _write_run(
            self.results_dir,
            "2026-02-22T10-00-00",
            results=[
                _make_result("alpha", status="pass", duration=12.0),
                _make_result("beta", status="pass", duration=18.0),
                _make_result("gamma", status="pass", duration=10.0),
                _make_result("delta", status="fail", duration=25.0),
                _make_result("epsilon", status="pass", duration=30.0),
            ],
        )

        self.runner = CliRunner()

    def _cleanup_fixtures(self) -> None:
        self.tmpdir.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegistryCommand(_FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._setup_fixtures()

    def tearDown(self) -> None:
        self._cleanup_fixtures()

    def test_counts_output(self) -> None:
        result = self.runner.invoke(
            analyze,
            ["registry", "--registry-dir", str(self.registry_dir)],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Registry:", result.output)
        self.assertIn("active", result.output)
        self.assertIn("skipped", result.output)

    def test_table_output(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "registry",
                "--format",
                "table",
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("alpha", result.output)
        self.assertIn("Package", result.output)

    def test_with_where_filter(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "registry",
                "--format",
                "table",
                "--where",
                "skip:false",
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("skipped", result.output)


class TestRunCommand(_FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._setup_fixtures()

    def tearDown(self) -> None:
        self._cleanup_fixtures()

    def test_default_latest(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "run",
                "--results-dir",
                str(self.results_dir),
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Run ID:", result.output)

    def test_summary_format(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "run",
                "2026-02-20T10-00-00",
                "--results-dir",
                str(self.results_dir),
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Packages tested:", result.output)
        self.assertIn("CRASH", result.output)

    def test_table_format(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "run",
                "--format",
                "table",
                "--results-dir",
                str(self.results_dir),
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("alpha", result.output)

    def test_full_format(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "run",
                "2026-02-20T10-00-00",
                "--format",
                "full",
                "--results-dir",
                str(self.results_dir),
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)

    def test_quiet_mode(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "run",
                "2026-02-20T10-00-00",
                "-q",
                "--results-dir",
                str(self.results_dir),
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        # Should have crash info.
        self.assertIn("crash", result.output.lower())

    def test_quiet_no_crashes(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "run",
                "2026-02-22T10-00-00",
                "-q",
                "--results-dir",
                str(self.results_dir),
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        # No crashes → no output.
        self.assertEqual(result.output.strip(), "")

    def test_nonexistent_id(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "run",
                "nonexistent",
                "--results-dir",
                str(self.results_dir),
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("not found", result.output.lower())


class TestCompareCommand(_FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._setup_fixtures()

    def tearDown(self) -> None:
        self._cleanup_fixtures()

    def test_basic(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "compare",
                "2026-02-20T10-00-00",
                "2026-02-22T10-00-00",
                "--results-dir",
                str(self.results_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Comparing:", result.output)
        self.assertIn("Status changes", result.output)

    def test_only_changes(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "compare",
                "2026-02-20T10-00-00",
                "2026-02-22T10-00-00",
                "--only-changes",
                "--results-dir",
                str(self.results_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("Unchanged:", result.output)

    def test_bad_run_id(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "compare",
                "nonexistent",
                "2026-02-22T10-00-00",
                "--results-dir",
                str(self.results_dir),
            ],
        )
        self.assertNotEqual(result.exit_code, 0)


class TestHistoryCommand(_FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._setup_fixtures()

    def tearDown(self) -> None:
        self._cleanup_fixtures()

    def test_table(self) -> None:
        result = self.runner.invoke(
            analyze,
            ["history", "--results-dir", str(self.results_dir)],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Run history", result.output)
        self.assertIn("Crash summary:", result.output)

    def test_timeline(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "history",
                "--format",
                "timeline",
                "--results-dir",
                str(self.results_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("trend", result.output.lower())

    def test_last_n(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "history",
                "--last",
                "1",
                "--results-dir",
                str(self.results_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("last 1", result.output)

    def test_empty(self) -> None:
        empty_dir = Path(self.tmpdir.name) / "empty_results"
        empty_dir.mkdir()
        result = self.runner.invoke(
            analyze,
            ["history", "--results-dir", str(empty_dir)],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("No runs found", result.output)


class TestPackageCommand(_FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._setup_fixtures()

    def tearDown(self) -> None:
        self._cleanup_fixtures()

    def test_basic(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "package",
                "alpha",
                "--results-dir",
                str(self.results_dir),
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Package: alpha", result.output)
        self.assertIn("Run history", result.output)

    def test_not_found_in_registry(self) -> None:
        result = self.runner.invoke(
            analyze,
            [
                "package",
                "nonexistent",
                "--results-dir",
                str(self.results_dir),
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Package: nonexistent", result.output)
        self.assertIn("No run history", result.output)

    def test_no_runs(self) -> None:
        # Package exists in registry but no runs contain it.
        _write_package(self.registry_dir, "lonely")
        result = self.runner.invoke(
            analyze,
            [
                "package",
                "lonely",
                "--results-dir",
                str(self.results_dir),
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("No run history", result.output)


if __name__ == "__main__":
    unittest.main()
