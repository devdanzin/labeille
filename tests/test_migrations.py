"""Tests for labeille.migrations — migration framework and built-in migrations."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from click.testing import CliRunner

from labeille.migrations import (
    MigrationDryRun,
    MigrationExecution,
    MigrationLogEntry,
    MigrationResult,
    MigrationSpec,
    _is_version_specific_skip,
    append_migration_log,
    execute_migration,
    get_migration,
    has_been_applied,
    list_migrations,
    migrate_skip_to_skip_versions,
    read_migration_log,
)
from labeille.registry_cli import registry


def _write_package(
    registry_dir: Path,
    name: str,
    *,
    skip: bool = False,
    skip_reason: str | None = None,
    skip_versions: dict[str, str] | None = None,
    enriched: bool = True,
) -> Path:
    """Create a package YAML file for testing."""
    data: dict[str, object] = {
        "package": name,
        "repo": f"https://github.com/user/{name}",
        "pypi_url": f"https://pypi.org/project/{name}/",
        "extension_type": "pure",
        "python_versions": [],
        "install_method": "pip",
        "install_command": "pip install -e .",
        "test_command": "python -m pytest tests/",
        "test_framework": "pytest",
        "uses_xdist": False,
        "timeout": None,
        "skip": skip,
        "skip_reason": skip_reason,
        "skip_versions": skip_versions or {},
        "notes": "",
        "enriched": enriched,
        "clone_depth": None,
        "import_name": None,
    }
    pkg_dir = registry_dir / "packages"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    p = pkg_dir / f"{name}.yaml"
    p.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return p


# ===========================================================================
# Framework tests
# ===========================================================================


class TestMigrationRegistry(unittest.TestCase):
    def test_register_migration(self) -> None:
        # The skip-to-skip-versions migration is already registered.
        spec = get_migration("skip-to-skip-versions")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "skip-to-skip-versions")

    def test_get_migration_exists(self) -> None:
        spec = get_migration("skip-to-skip-versions")
        self.assertIsNotNone(spec)
        self.assertIsInstance(spec, MigrationSpec)
        self.assertTrue(callable(spec.func))

    def test_get_migration_not_found(self) -> None:
        self.assertIsNone(get_migration("nonexistent-migration"))

    def test_list_migrations(self) -> None:
        migrations = list_migrations()
        self.assertGreaterEqual(len(migrations), 1)
        names = [m.name for m in migrations]
        self.assertIn("skip-to-skip-versions", names)


class TestMigrationLog(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.registry_dir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_migration_log_empty(self) -> None:
        entries = read_migration_log(self.registry_dir)
        self.assertEqual(entries, [])

    def test_migration_log_write_read(self) -> None:
        entry = MigrationLogEntry(
            migration="test-migration",
            applied_at="2026-02-23T14:00:00Z",
            files_modified=5,
            files_skipped=10,
        )
        append_migration_log(self.registry_dir, entry)
        entries = read_migration_log(self.registry_dir)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].migration, "test-migration")
        self.assertEqual(entries[0].files_modified, 5)
        self.assertEqual(entries[0].files_skipped, 10)

    def test_has_been_applied_false(self) -> None:
        self.assertFalse(has_been_applied(self.registry_dir, "nonexistent"))

    def test_has_been_applied_true(self) -> None:
        entry = MigrationLogEntry(
            migration="applied-one",
            applied_at="2026-02-23T14:00:00Z",
            files_modified=1,
            files_skipped=0,
        )
        append_migration_log(self.registry_dir, entry)
        self.assertTrue(has_been_applied(self.registry_dir, "applied-one"))
        self.assertFalse(has_been_applied(self.registry_dir, "other"))


class TestExecuteMigration(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.registry_dir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_execute_migration_dry_run(self) -> None:
        _write_package(
            self.registry_dir,
            "rpds-py",
            skip=True,
            skip_reason="Requires PyO3/maturin. Not buildable until PyO3 supports 3.15.",
        )
        _write_package(self.registry_dir, "click", skip=False)

        spec = get_migration("skip-to-skip-versions")
        assert spec is not None
        result = execute_migration(spec, self.registry_dir, dry_run=True)

        self.assertIsInstance(result, MigrationDryRun)
        assert isinstance(result, MigrationDryRun)
        self.assertEqual(result.affected_count, 1)
        self.assertEqual(result.skipped_count, 1)

        # File should NOT be modified in dry run.
        raw = yaml.safe_load((self.registry_dir / "packages" / "rpds-py.yaml").read_text())
        self.assertTrue(raw["skip"])

    def test_execute_migration_apply(self) -> None:
        _write_package(
            self.registry_dir,
            "rpds-py",
            skip=True,
            skip_reason="Requires PyO3/maturin. Not buildable until PyO3 supports 3.15.",
        )
        _write_package(self.registry_dir, "click", skip=False)

        spec = get_migration("skip-to-skip-versions")
        assert spec is not None
        result = execute_migration(spec, self.registry_dir, dry_run=False)

        self.assertIsInstance(result, MigrationExecution)
        assert isinstance(result, MigrationExecution)
        self.assertEqual(result.modified_count, 1)
        self.assertEqual(result.skipped_count, 1)

        # File should be modified.
        raw = yaml.safe_load((self.registry_dir / "packages" / "rpds-py.yaml").read_text())
        self.assertFalse(raw["skip"])
        self.assertIsNone(raw["skip_reason"])
        self.assertIn("3.15", raw["skip_versions"])

        # Log entry should exist.
        self.assertTrue(has_been_applied(self.registry_dir, "skip-to-skip-versions"))

    def test_execute_migration_empty_registry(self) -> None:
        spec = get_migration("skip-to-skip-versions")
        assert spec is not None
        result = execute_migration(spec, self.registry_dir, dry_run=True)
        self.assertIsInstance(result, MigrationDryRun)
        assert isinstance(result, MigrationDryRun)
        self.assertEqual(result.affected_count, 0)


# ===========================================================================
# skip-to-skip-versions migration tests
# ===========================================================================


class TestSkipToSkipVersionsMigration(unittest.TestCase):
    def _run_migration(
        self,
        *,
        skip: bool = True,
        skip_reason: str | None = None,
        skip_versions: dict[str, str] | None = None,
    ) -> tuple[MigrationResult, dict[str, object]]:
        """Helper to run the migration on a single data dict."""
        data: dict[str, object] = {
            "package": "testpkg",
            "skip": skip,
            "skip_reason": skip_reason,
            "skip_versions": skip_versions or {},
        }
        result = migrate_skip_to_skip_versions(Path("testpkg.yaml"), data)
        return result, data

    def test_converts_pyo3_skip(self) -> None:
        result, data = self._run_migration(
            skip_reason="Requires PyO3/maturin for source builds.",
        )
        self.assertTrue(result.modified)
        self.assertFalse(data["skip"])
        self.assertIsNone(data["skip_reason"])
        sv = data["skip_versions"]
        assert isinstance(sv, dict)
        self.assertIn("3.15", sv)
        self.assertIn("PyO3", sv["3.15"])

    def test_converts_pydantic_core_dep(self) -> None:
        result, data = self._run_migration(
            skip_reason="Depends on pydantic-core (PyO3, no 3.15 support).",
        )
        self.assertTrue(result.modified)
        self.assertFalse(data["skip"])

    def test_converts_rpds_dep(self) -> None:
        result, data = self._run_migration(
            skip_reason="Depends on rpds-py (PyO3).",
        )
        self.assertTrue(result.modified)

    def test_converts_maturin(self) -> None:
        result, data = self._run_migration(
            skip_reason="Requires PyO3/maturin for source builds. Not buildable until PyO3 supports 3.15.",
        )
        self.assertTrue(result.modified)

    def test_converts_jit_crash(self) -> None:
        result, data = self._run_migration(
            skip_reason="JIT crash (_PyOptimizer_Optimize abort) during pip install.",
        )
        self.assertTrue(result.modified)

    def test_converts_cython_build(self) -> None:
        result, data = self._run_migration(
            skip_reason="C extension source files not generated (Cython build step fails on editable install for 3.15).",
        )
        self.assertTrue(result.modified)

    def test_preserves_no_test_suite(self) -> None:
        result, data = self._run_migration(
            skip_reason="Type stub package with no meaningful test suite.",
        )
        self.assertFalse(result.modified)
        self.assertTrue(data["skip"])

    def test_preserves_cloud_credentials(self) -> None:
        result, data = self._run_migration(
            skip_reason="Tests require cloud credentials.",
        )
        self.assertFalse(result.modified)
        self.assertTrue(data["skip"])

    def test_preserves_monorepo(self) -> None:
        result, data = self._run_migration(
            skip_reason="Monorepo package in azure-sdk-for-python. Complex to test standalone.",
        )
        self.assertFalse(result.modified)

    def test_preserves_no_repo_url(self) -> None:
        result, data = self._run_migration(
            skip_reason="No repository URL found.",
        )
        self.assertFalse(result.modified)

    def test_preserves_rust_binary_no_tests(self) -> None:
        result, data = self._run_migration(
            skip_reason="Rust binary with no Python test suite.",
        )
        self.assertFalse(result.modified)

    def test_preserves_non_skipped(self) -> None:
        result, data = self._run_migration(skip=False, skip_reason=None)
        self.assertFalse(result.modified)

    def test_preserves_existing_skip_versions(self) -> None:
        result, data = self._run_migration(
            skip_reason="Requires PyO3/maturin.",
            skip_versions={"3.14": "Some other reason"},
        )
        self.assertTrue(result.modified)
        sv = data["skip_versions"]
        assert isinstance(sv, dict)
        self.assertEqual(sv["3.14"], "Some other reason")
        self.assertIn("3.15", sv)

    def test_skip_reason_set_to_null(self) -> None:
        result, data = self._run_migration(
            skip_reason="Requires PyO3/maturin.",
        )
        self.assertTrue(result.modified)
        self.assertIsNone(data["skip_reason"])

    def test_skip_set_to_false(self) -> None:
        result, data = self._run_migration(
            skip_reason="Requires PyO3/maturin.",
        )
        self.assertTrue(result.modified)
        self.assertFalse(data["skip"])

    def test_yaml_key_is_string(self) -> None:
        """After conversion, the '3.15' key must be a string, not a float."""
        result, data = self._run_migration(
            skip_reason="Requires PyO3/maturin.",
        )
        self.assertTrue(result.modified)
        sv = data["skip_versions"]
        assert isinstance(sv, dict)
        keys = list(sv.keys())
        self.assertEqual(keys, ["3.15"])
        self.assertIsInstance(keys[0], str)


# ===========================================================================
# _is_version_specific_skip tests
# ===========================================================================


class TestIsVersionSpecificSkip(unittest.TestCase):
    def test_pyo3_keyword(self) -> None:
        self.assertTrue(_is_version_specific_skip("Requires PyO3/maturin"))

    def test_maturin_keyword(self) -> None:
        self.assertTrue(_is_version_specific_skip("maturin for source builds"))

    def test_no_315_support(self) -> None:
        self.assertTrue(_is_version_specific_skip("no 3.15 support"))

    def test_doesnt_support_315(self) -> None:
        self.assertTrue(_is_version_specific_skip("doesn't support Python 3.15"))

    def test_pydantic_core_dep(self) -> None:
        self.assertTrue(_is_version_specific_skip("pydantic-core (PyO3)"))

    def test_rpds_dep(self) -> None:
        self.assertTrue(_is_version_specific_skip("rpds-py (PyO3)"))

    def test_jit_crash(self) -> None:
        self.assertTrue(_is_version_specific_skip("JIT crash (_PyOptimizer_Optimize abort)"))

    def test_cython_fails(self) -> None:
        self.assertTrue(_is_version_specific_skip("Cython build step fails"))

    def test_rust_binary_no_tests(self) -> None:
        self.assertFalse(_is_version_specific_skip("Rust binary with no Python test suite"))

    def test_no_test_suite(self) -> None:
        self.assertFalse(
            _is_version_specific_skip("Type stub package with no meaningful test suite")
        )

    def test_cloud_credentials(self) -> None:
        self.assertFalse(_is_version_specific_skip("Tests require cloud credentials"))

    def test_monorepo(self) -> None:
        self.assertFalse(_is_version_specific_skip("Monorepo package"))

    def test_complex_c_build(self) -> None:
        self.assertFalse(_is_version_specific_skip("Complex C extension build"))

    def test_empty_reason(self) -> None:
        self.assertFalse(_is_version_specific_skip(""))

    def test_no_repo_url(self) -> None:
        self.assertFalse(_is_version_specific_skip("No repository URL found."))


# ===========================================================================
# CLI tests
# ===========================================================================


class TestMigrateCLI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.registry_dir = Path(self._tmpdir.name)
        self.runner = CliRunner()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_migrate_list(self) -> None:
        result = self.runner.invoke(
            registry, ["migrate", "--list", "--registry-dir", str(self.registry_dir)]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("skip-to-skip-versions", result.output)
        self.assertIn("not applied", result.output)

    def test_migrate_dry_run(self) -> None:
        _write_package(
            self.registry_dir,
            "rpds-py",
            skip=True,
            skip_reason="Requires PyO3/maturin. Not buildable until PyO3 supports 3.15.",
        )
        _write_package(self.registry_dir, "click", skip=False)

        result = self.runner.invoke(
            registry,
            ["migrate", "skip-to-skip-versions", "--registry-dir", str(self.registry_dir)],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("DRY RUN", result.output)
        self.assertIn("Would modify 1 files", result.output)
        self.assertIn("rpds-py", result.output)

        # File should NOT be modified.
        raw = yaml.safe_load((self.registry_dir / "packages" / "rpds-py.yaml").read_text())
        self.assertTrue(raw["skip"])

    def test_migrate_apply(self) -> None:
        _write_package(
            self.registry_dir,
            "rpds-py",
            skip=True,
            skip_reason="Requires PyO3/maturin. Not buildable until PyO3 supports 3.15.",
        )

        # Create index for rebuild.
        index_data = {
            "last_updated": "2026-02-23T00:00:00",
            "packages": [
                {"name": "rpds-py", "extension_type": "pure", "enriched": True, "skip": True}
            ],
        }
        (self.registry_dir / "index.yaml").write_text(
            yaml.dump(index_data, default_flow_style=False)
        )

        result = self.runner.invoke(
            registry,
            [
                "migrate",
                "skip-to-skip-versions",
                "--apply",
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Applied migration", result.output)
        self.assertIn("modified 1 files", result.output)

        # File should be modified.
        raw = yaml.safe_load((self.registry_dir / "packages" / "rpds-py.yaml").read_text())
        self.assertFalse(raw["skip"])
        self.assertIn("3.15", raw["skip_versions"])

    def test_migrate_already_applied(self) -> None:
        entry = {
            "migration": "skip-to-skip-versions",
            "applied_at": "2026-02-23T14:00:00Z",
            "files_modified": 5,
            "files_skipped": 10,
        }
        log_path = self.registry_dir / "migrations.log"
        log_path.write_text(json.dumps(entry) + "\n")

        result = self.runner.invoke(
            registry,
            [
                "migrate",
                "skip-to-skip-versions",
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("already applied", result.output)

    def test_migrate_unknown_name(self) -> None:
        result = self.runner.invoke(
            registry,
            ["migrate", "nonexistent-migration", "--registry-dir", str(self.registry_dir)],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("unknown migration", result.output)

    def test_migrate_no_name_no_list(self) -> None:
        result = self.runner.invoke(
            registry,
            ["migrate", "--registry-dir", str(self.registry_dir)],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--list", result.output)

    def test_migrate_list_shows_applied_status(self) -> None:
        entry = {
            "migration": "skip-to-skip-versions",
            "applied_at": "2026-02-23T14:00:00Z",
            "files_modified": 5,
            "files_skipped": 10,
        }
        log_path = self.registry_dir / "migrations.log"
        log_path.write_text(json.dumps(entry) + "\n")

        result = self.runner.invoke(
            registry, ["migrate", "--list", "--registry-dir", str(self.registry_dir)]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("applied on 2026-02-23T14:00:00Z", result.output)


if __name__ == "__main__":
    unittest.main()
