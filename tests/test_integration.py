"""End-to-end integration tests exercising the CLI through click's CliRunner."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from labeille import __version__
from labeille.cli import main
from labeille.registry import (
    Index,
    IndexEntry,
    PackageEntry,
    save_index,
    save_package,
)


class TestVersionCommand(unittest.TestCase):
    def test_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(__version__, result.output)


def _fake_pypi_metadata(name: str) -> dict:  # type: ignore[type-arg]
    """Return minimal PyPI-like metadata for mocked fetch calls."""
    return {
        "info": {
            "name": name,
            "project_urls": {"Homepage": f"https://example.com/{name}"},
        },
        "urls": [],
    }


class TestResolveIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.registry_dir = self.base / "registry"
        self.registry_dir.mkdir()
        (self.registry_dir / "packages").mkdir()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_resolve_no_source_prints_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "resolve",
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("At least one of", result.output)

    @patch(
        "labeille.resolve.fetch_pypi_metadata", side_effect=lambda n, **kw: _fake_pypi_metadata(n)
    )
    def test_resolve_dry_run_from_json(self, _mock_fetch: object) -> None:
        # Create a small test JSON fixture.
        json_file = self.base / "packages.json"
        data = {
            "rows": [
                {"project": "fakepkg-alpha", "download_count": 100},
                {"project": "fakepkg-beta", "download_count": 50},
            ]
        }
        json_file.write_text(json.dumps(data), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "resolve",
                "--dry-run",
                "--from-json",
                str(json_file),
                "--registry-dir",
                str(self.registry_dir),
                "--log-file",
                str(self.base / "resolve.log"),
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Resolving 2 package(s)", result.output)
        self.assertIn("dry-run", result.output)
        # Dry-run should NOT create package YAML files.
        packages_dir = self.registry_dir / "packages"
        yaml_files = list(packages_dir.glob("*.yaml"))
        self.assertEqual(yaml_files, [])

    @patch(
        "labeille.resolve.fetch_pypi_metadata", side_effect=lambda n, **kw: _fake_pypi_metadata(n)
    )
    def test_resolve_dry_run_from_json_top(self, _mock_fetch: object) -> None:
        json_file = self.base / "packages.json"
        data = {
            "rows": [
                {"project": "a", "download_count": 300},
                {"project": "b", "download_count": 200},
                {"project": "c", "download_count": 100},
            ]
        }
        json_file.write_text(json.dumps(data), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "resolve",
                "--dry-run",
                "--from-json",
                str(json_file),
                "--top",
                "2",
                "--registry-dir",
                str(self.registry_dir),
                "--log-file",
                str(self.base / "resolve.log"),
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Resolving 2 package(s)", result.output)

    def test_resolve_top_without_from_json_errors(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "resolve",
                "--top",
                "5",
                "somepkg",
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--top requires --from-json", result.output)


class TestRunIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.registry_dir = self.base / "registry"
        self.registry_dir.mkdir()
        (self.registry_dir / "packages").mkdir()
        self.results_dir = self.base / "results"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_run_missing_target_python_errors(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--target-python", result.output)

    def test_run_dry_run_with_registry(self) -> None:
        # Populate a small registry.
        pkg = PackageEntry(
            package="fakepkg",
            repo="https://github.com/user/fakepkg",
            extension_type="pure",
            test_command="python -m pytest",
            install_command="pip install -e .",
        )
        save_package(pkg, self.registry_dir)
        index = Index(packages=[IndexEntry(name="fakepkg", download_count=1000)])
        save_index(index, self.registry_dir)

        import sys

        target = sys.executable

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--dry-run",
                "--target-python",
                target,
                "--registry-dir",
                str(self.registry_dir),
                "--results-dir",
                str(self.results_dir),
                "--run-id",
                "test-dry",
                "--log-file",
                str(self.base / "run.log"),
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("dry-run", result.output)
        self.assertIn("Skipped:", result.output)

    def test_run_nonexistent_target_python_errors(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--target-python",
                "/nonexistent/python3",
                "--registry-dir",
                str(self.registry_dir),
            ],
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_run_empty_registry(self) -> None:
        # Empty index — nothing to test.
        save_index(Index(), self.registry_dir)

        import sys

        target = sys.executable

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "run",
                "--dry-run",
                "--target-python",
                target,
                "--registry-dir",
                str(self.registry_dir),
                "--results-dir",
                str(self.results_dir),
                "--run-id",
                "test-empty",
                "--log-file",
                str(self.base / "run.log"),
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Packages tested: 0", result.output)


if __name__ == "__main__":
    unittest.main()
