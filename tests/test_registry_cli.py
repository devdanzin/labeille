"""Integration tests for labeille.registry_cli — CLI commands."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from click.testing import CliRunner

from labeille.registry_cli import registry


def _write_package(
    registry_dir: Path,
    name: str,
    *,
    extra_fields: dict[str, object] | None = None,
) -> Path:
    """Create a minimal package YAML file in the registry."""
    data: dict[str, object] = {
        "package": name,
        "repo": f"https://github.com/user/{name}",
        "pypi_url": f"https://pypi.org/project/{name}/",
        "extension_type": "pure",
        "python_versions": [],
        "install_method": "pip",
        "install_command": "pip install -e '.[dev]'",
        "test_command": "python -m pytest tests/",
        "test_framework": "pytest",
        "uses_xdist": False,
        "timeout": None,
        "skip": False,
        "skip_reason": None,
        "skip_versions": {},
        "notes": "",
        "enriched": True,
        "clone_depth": None,
        "import_name": None,
    }
    if extra_fields:
        data.update(extra_fields)
    pkg_dir = registry_dir / "packages"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    p = pkg_dir / f"{name}.yaml"
    p.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return p


def _write_index(registry_dir: Path, names: list[str]) -> None:
    """Create a minimal index.yaml."""
    registry_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "last_updated": "2026-02-23T00:00:00",
        "packages": [
            {"name": n, "extension_type": "pure", "enriched": True, "skip": False} for n in names
        ],
    }
    (registry_dir / "index.yaml").write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


class TestAddFieldCli(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        for name in ["alpha", "beta", "gamma"]:
            _write_package(self.registry, name)
        _write_index(self.registry, ["alpha", "beta", "gamma"])
        self.runner = CliRunner()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_dry_run_output(self) -> None:
        result = self.runner.invoke(
            registry,
            [
                "add-field",
                "priority",
                "--type",
                "int",
                "--default",
                "5",
                "--after",
                "skip_reason",
                "--registry-dir",
                str(self.registry),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("DRY RUN", result.output)
        self.assertIn("Would modify 3 files", result.output)
        self.assertIn("Re-run with --apply", result.output)

    def test_apply(self) -> None:
        result = self.runner.invoke(
            registry,
            [
                "add-field",
                "priority",
                "--type",
                "int",
                "--default",
                "5",
                "--after",
                "skip_reason",
                "--apply",
                "--no-update-index",
                "--registry-dir",
                str(self.registry),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Modified 3 files", result.output)
        self.assertIn("Remember to add 'priority' to PackageEntry", result.output)
        for name in ["alpha", "beta", "gamma"]:
            p = self.registry / "packages" / f"{name}.yaml"
            data = yaml.safe_load(p.read_text())
            self.assertEqual(data["priority"], 5)

    def test_exit_code_strict(self) -> None:
        """Exit code 1 when field already exists in strict mode."""
        result = self.runner.invoke(
            registry,
            ["add-field", "skip", "--type", "bool", "--registry-dir", str(self.registry)],
        )
        self.assertEqual(result.exit_code, 1)

    def test_exit_code_lenient(self) -> None:
        """Exit code 0 when some files skipped with --lenient."""
        # Add field to just alpha first
        self.runner.invoke(
            registry,
            [
                "add-field",
                "priority",
                "--type",
                "int",
                "--default",
                "5",
                "--after",
                "skip_reason",
                "--apply",
                "--no-update-index",
                "--packages",
                "alpha",
                "--registry-dir",
                str(self.registry),
            ],
        )
        # Now try adding to all with --lenient
        result = self.runner.invoke(
            registry,
            [
                "add-field",
                "priority",
                "--type",
                "int",
                "--default",
                "5",
                "--after",
                "skip_reason",
                "--apply",
                "--lenient",
                "--no-update-index",
                "--registry-dir",
                str(self.registry),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)

    def test_exit_code_lenient_all_skipped(self) -> None:
        """Exit code 1 when ALL files skipped (nothing to do)."""
        result = self.runner.invoke(
            registry,
            [
                "add-field",
                "skip",
                "--type",
                "bool",
                "--apply",
                "--lenient",
                "--no-update-index",
                "--registry-dir",
                str(self.registry),
            ],
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Nothing to do", result.output)


class TestRemoveFieldCli(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        for name in ["alpha", "beta"]:
            _write_package(self.registry, name)
        _write_index(self.registry, ["alpha", "beta"])
        self.runner = CliRunner()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_apply(self) -> None:
        result = self.runner.invoke(
            registry,
            [
                "remove-field",
                "notes",
                "--apply",
                "--no-update-index",
                "--registry-dir",
                str(self.registry),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Modified 2 files", result.output)
        for name in ["alpha", "beta"]:
            data = yaml.safe_load((self.registry / "packages" / f"{name}.yaml").read_text())
            self.assertNotIn("notes", data)

    def test_protected(self) -> None:
        result = self.runner.invoke(
            registry,
            ["remove-field", "package", "--apply", "--registry-dir", str(self.registry)],
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("protected", result.output)


class TestSetFieldCli(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        _write_package(self.registry, "pure", extra_fields={"extension_type": "pure"})
        _write_package(self.registry, "ext", extra_fields={"extension_type": "extensions"})
        _write_index(self.registry, ["pure", "ext"])
        self.runner = CliRunner()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_no_filter_no_all(self) -> None:
        result = self.runner.invoke(
            registry,
            ["set-field", "timeout", "600", "--apply", "--registry-dir", str(self.registry)],
        )
        self.assertNotEqual(result.exit_code, 0)


class TestValidateCli(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        self.runner = CliRunner()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_clean(self) -> None:
        for name in ["alpha", "beta"]:
            _write_package(self.registry, name)
        result = self.runner.invoke(
            registry,
            ["validate", "--registry-dir", str(self.registry)],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("0 errors", result.output)

    def test_errors(self) -> None:
        pkg_dir = self.registry / "packages"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        p = pkg_dir / "broken.yaml"
        p.write_text(yaml.dump({"repo": "https://example.com"}), encoding="utf-8")
        result = self.runner.invoke(
            registry,
            ["validate", "--registry-dir", str(self.registry)],
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("ERROR", result.output)


class TestUpdateIndexFlag(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        for name in ["alpha", "beta"]:
            _write_package(self.registry, name)
        _write_index(self.registry, ["alpha", "beta"])
        self.runner = CliRunner()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_update_index_flag(self) -> None:
        result = self.runner.invoke(
            registry,
            [
                "remove-field",
                "notes",
                "--apply",
                "--update-index",
                "--registry-dir",
                str(self.registry),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Index updated", result.output)

    def test_no_update_index_flag(self) -> None:
        result = self.runner.invoke(
            registry,
            [
                "remove-field",
                "notes",
                "--apply",
                "--no-update-index",
                "--registry-dir",
                str(self.registry),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("Index updated", result.output)


if __name__ == "__main__":
    unittest.main()
