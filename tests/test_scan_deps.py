"""Tests for labeille.scan_deps and import_map."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from labeille.cli import main
from labeille.import_map import IMPORT_TO_PIP
from labeille.registry import PackageEntry
from labeille.scan_deps import (
    ImportInfo,
    ResolvedDep,
    _normalize_pip_command,
    _parse_install_packages,
    build_scan_result,
    compare_with_install_command,
    extract_imports,
    extract_imports_from_directory,
    filter_imports,
    get_local_modules,
    get_stdlib_modules,
    resolve_imports,
    scan_package_deps,
)


def _create_test_repo(tmp_dir: Path) -> Path:
    """Create a minimal repo with known imports for testing.

    Structure:
    repo/
      src/
        mypackage/
          __init__.py
          core.py          # imports: os, json, click
      tests/
        __init__.py
        conftest.py        # imports: pytest, _fixtures (relative)
        test_core.py       # imports: pytest, hypothesis, mypackage
        test_compat.py     # imports: pytest, yaml (in try/except)
        _fixtures/
          __init__.py      # local test helper
    """
    repo = tmp_dir / "mypackage"
    repo.mkdir()

    # Source package.
    src = repo / "src" / "mypackage"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "core.py").write_text("import os\nimport json\nimport click\n\ndef main(): pass\n")

    # Tests.
    tests = repo / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "conftest.py").write_text(
        "import pytest\nfrom . import _fixtures\n\n@pytest.fixture\ndef fix(): pass\n"
    )
    (tests / "test_core.py").write_text(
        "import pytest\nimport hypothesis\nimport mypackage\n\ndef test_one(): pass\n"
    )
    (tests / "test_compat.py").write_text(
        "import pytest\n\n"
        "try:\n"
        "    import yaml\n"
        "except ImportError:\n"
        "    yaml = None\n"
        "\ndef test_compat(): pass\n"
    )

    # Local test helper package.
    fixtures = tests / "_fixtures"
    fixtures.mkdir()
    (fixtures / "__init__.py").write_text("DATA = 42\n")

    return repo


# ===========================================================================
# Import extraction tests
# ===========================================================================


class TestExtractImports(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write(self, name: str, content: str) -> Path:
        p = self.tmpdir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def test_extract_import_simple(self) -> None:
        p = self._write("test.py", "import pytest\n")
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "pytest")
        self.assertEqual(imports[0].full_name, "pytest")
        self.assertFalse(imports[0].is_conditional)

    def test_extract_import_dotted(self) -> None:
        p = self._write("test.py", "import os.path\n")
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "os")
        self.assertEqual(imports[0].full_name, "os.path")

    def test_extract_from_import(self) -> None:
        p = self._write("test.py", "from yaml import safe_load\n")
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "yaml")
        self.assertEqual(imports[0].full_name, "yaml")

    def test_extract_from_import_dotted(self) -> None:
        p = self._write("test.py", "from google.cloud import storage\n")
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "google")
        self.assertEqual(imports[0].full_name, "google.cloud")

    def test_extract_relative_import_skipped(self) -> None:
        p = self._write("test.py", "from . import utils\n")
        imports = extract_imports(p)
        self.assertEqual(len(imports), 0)

    def test_extract_from_relative_skipped(self) -> None:
        p = self._write("test.py", "from .helpers import foo\n")
        imports = extract_imports(p)
        self.assertEqual(len(imports), 0)

    def test_extract_conditional_import(self) -> None:
        code = "try:\n    import optional_lib\nexcept ImportError:\n    optional_lib = None\n"
        p = self._write("test.py", code)
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "optional_lib")
        self.assertTrue(imports[0].is_conditional)

    def test_extract_conditional_module_not_found(self) -> None:
        code = (
            "try:\n    import optional_lib\nexcept ModuleNotFoundError:\n    optional_lib = None\n"
        )
        p = self._write("test.py", code)
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertTrue(imports[0].is_conditional)

    def test_extract_non_conditional(self) -> None:
        code = "try:\n    import some_lib\nexcept ValueError:\n    pass\n"
        p = self._write("test.py", code)
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertFalse(imports[0].is_conditional)

    def test_extract_nested_conditional(self) -> None:
        code = (
            "def setup():\n"
            "    try:\n"
            "        import optional_lib\n"
            "    except ImportError:\n"
            "        optional_lib = None\n"
        )
        p = self._write("test.py", code)
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertTrue(imports[0].is_conditional)

    def test_extract_except_exception_not_conditional(self) -> None:
        """import inside try/except Exception is NOT conditional."""
        code = "try:\n    import some_lib\nexcept Exception:\n    some_lib = None\n"
        p = self._write("test.py", code)
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertFalse(imports[0].is_conditional)

    def test_extract_bare_except_still_conditional(self) -> None:
        """import inside bare except IS conditional."""
        code = "try:\n    import some_lib\nexcept:\n    some_lib = None\n"
        p = self._write("test.py", code)
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertTrue(imports[0].is_conditional)

    def test_extract_tuple_with_exception_and_importerror(self) -> None:
        """except (Exception, ImportError) IS conditional (ImportError present)."""
        code = "try:\n    import some_lib\nexcept (Exception, ImportError):\n    some_lib = None\n"
        p = self._write("test.py", code)
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertTrue(imports[0].is_conditional)

    def test_extract_tuple_with_only_exception(self) -> None:
        """except (Exception, ValueError) is NOT conditional."""
        code = "try:\n    import some_lib\nexcept (Exception, ValueError):\n    some_lib = None\n"
        p = self._write("test.py", code)
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertFalse(imports[0].is_conditional)

    def test_extract_syntax_error(self) -> None:
        p = self._write("test.py", "def foo(\n  # broken syntax\n")
        imports = extract_imports(p)
        self.assertEqual(imports, [])

    def test_extract_multiple_imports(self) -> None:
        p = self._write("test.py", "import os, sys, json\n")
        imports = extract_imports(p)
        self.assertEqual(len(imports), 3)
        modules = {imp.module for imp in imports}
        self.assertEqual(modules, {"os", "sys", "json"})

    def test_extract_star_import(self) -> None:
        p = self._write("test.py", "from typing import *\n")
        imports = extract_imports(p)
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "typing")

    def test_extract_from_directory(self) -> None:
        d = self.tmpdir / "pkg"
        d.mkdir()
        (d / "a.py").write_text("import pytest\n")
        (d / "b.py").write_text("import hypothesis\n")
        imports = extract_imports_from_directory(d)
        modules = {imp.module for imp in imports}
        self.assertEqual(modules, {"pytest", "hypothesis"})

    def test_extract_respects_exclude_patterns(self) -> None:
        d = self.tmpdir / "pkg"
        d.mkdir()
        (d / "a.py").write_text("import pytest\n")
        vendor = d / "vendor"
        vendor.mkdir()
        (vendor / "b.py").write_text("import vendored_lib\n")
        imports = extract_imports_from_directory(d)
        # vendor is in default excludes, so vendored_lib should not appear.
        modules = {imp.module for imp in imports}
        self.assertIn("pytest", modules)
        self.assertNotIn("vendored_lib", modules)

    def test_extract_relative_path_with_repo_root(self) -> None:
        d = self.tmpdir / "repo" / "tests"
        d.mkdir(parents=True)
        (d / "test_x.py").write_text("import pytest\n")
        imports = extract_imports(d / "test_x.py", repo_root=self.tmpdir / "repo")
        self.assertEqual(imports[0].source_file, "tests/test_x.py")


# ===========================================================================
# Import filtering tests
# ===========================================================================


class TestFilterImports(unittest.TestCase):
    def test_filter_removes_stdlib(self) -> None:
        stdlib = get_stdlib_modules()
        imports = [
            ImportInfo("os", "os", "test.py", 1, False),
            ImportInfo("sys", "sys", "test.py", 2, False),
            ImportInfo("json", "json", "test.py", 3, False),
            ImportInfo("pytest", "pytest", "test.py", 4, False),
        ]
        result = filter_imports(imports, stdlib=stdlib, local_modules=set())
        modules = {imp.module for imp in result}
        self.assertNotIn("os", modules)
        self.assertNotIn("sys", modules)
        self.assertNotIn("json", modules)
        self.assertIn("pytest", modules)

    def test_filter_removes_local_modules(self) -> None:
        imports = [
            ImportInfo("mypackage", "mypackage", "test.py", 1, False),
            ImportInfo("pytest", "pytest", "test.py", 2, False),
        ]
        result = filter_imports(imports, stdlib=set(), local_modules={"mypackage"})
        modules = {imp.module for imp in result}
        self.assertNotIn("mypackage", modules)
        self.assertIn("pytest", modules)

    def test_filter_removes_local_test_modules(self) -> None:
        imports = [
            ImportInfo("conftest", "conftest", "test.py", 1, False),
            ImportInfo("fixtures", "fixtures", "test.py", 2, False),
            ImportInfo("helpers", "helpers", "test.py", 3, False),
            ImportInfo("pytest", "pytest", "test.py", 4, False),
        ]
        # conftest, fixtures, helpers are in _TEST_INFRA_NAMES which
        # get_local_modules includes.
        local = {"conftest", "fixtures", "helpers"}
        result = filter_imports(imports, stdlib=set(), local_modules=local)
        modules = {imp.module for imp in result}
        self.assertEqual(modules, {"pytest"})

    def test_filter_keeps_external(self) -> None:
        imports = [
            ImportInfo("pytest", "pytest", "test.py", 1, False),
            ImportInfo("hypothesis", "hypothesis", "test.py", 2, False),
            ImportInfo("trustme", "trustme", "test.py", 3, False),
        ]
        result = filter_imports(imports, stdlib=set(), local_modules=set())
        self.assertEqual(len(result), 3)

    def test_filter_conditional_flag(self) -> None:
        imports = [
            ImportInfo("pytest", "pytest", "test.py", 1, False),
            ImportInfo("yaml", "yaml", "test.py", 2, True),
        ]
        result = filter_imports(
            imports, stdlib=set(), local_modules=set(), include_conditional=False
        )
        modules = {imp.module for imp in result}
        self.assertIn("pytest", modules)
        self.assertNotIn("yaml", modules)

    def test_get_stdlib_modules_contains_basics(self) -> None:
        stdlib = get_stdlib_modules()
        for name in ("os", "sys", "json", "pathlib", "typing", "importlib"):
            self.assertIn(name, stdlib)

    def test_get_local_modules_src_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "src" / "mylib").mkdir(parents=True)
            (repo / "src" / "mylib" / "__init__.py").write_text("")
            local = get_local_modules(repo, "mylib")
            self.assertIn("mylib", local)

    def test_get_local_modules_flat_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "mylib").mkdir()
            (repo / "mylib" / "__init__.py").write_text("")
            local = get_local_modules(repo, "mylib")
            self.assertIn("mylib", local)

    def test_get_local_modules_test_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "tests" / "_helpers").mkdir(parents=True)
            (repo / "tests" / "_helpers" / "__init__.py").write_text("")
            local = get_local_modules(repo, "mylib")
            self.assertIn("_helpers", local)
            self.assertIn("tests", local)


# ===========================================================================
# Resolution tests
# ===========================================================================


class TestResolveImports(unittest.TestCase):
    def test_resolve_identity(self) -> None:
        imports = [ImportInfo("pytest", "pytest", "test.py", 1, False)]
        # Use empty map so pytest falls through to identity.
        resolved, unresolved = resolve_imports(imports, import_map={})
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].pip_package, "pytest")
        self.assertEqual(resolved[0].source, "identity")

    def test_resolve_mapping(self) -> None:
        imports = [ImportInfo("yaml", "yaml", "test.py", 1, False)]
        resolved, unresolved = resolve_imports(imports)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].pip_package, "PyYAML")
        self.assertEqual(resolved[0].source, "mapping")

    def test_resolve_registry(self) -> None:
        imports = [ImportInfo("websocket", "websocket", "test.py", 1, False)]
        entry = PackageEntry(package="websocket-client", import_name="websocket")
        # Use empty import_map so it doesn't match from the static table.
        resolved, unresolved = resolve_imports(imports, import_map={}, registry_entries=[entry])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].pip_package, "websocket-client")
        self.assertEqual(resolved[0].source, "registry")

    def test_resolve_unresolved_underscore(self) -> None:
        imports = [ImportInfo("_internal", "_internal", "test.py", 1, False)]
        resolved, unresolved = resolve_imports(imports, import_map={})
        self.assertEqual(len(resolved), 0)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].import_name, "_internal")

    def test_resolve_unresolved_single_char(self) -> None:
        imports = [ImportInfo("T", "T", "test.py", 1, False)]
        resolved, unresolved = resolve_imports(imports, import_map={})
        self.assertEqual(len(resolved), 0)
        self.assertEqual(len(unresolved), 1)

    def test_resolve_deduplication(self) -> None:
        imports = [
            ImportInfo("pytest", "pytest", "test_a.py", 1, False),
            ImportInfo("pytest", "pytest", "test_b.py", 2, False),
        ]
        resolved, _ = resolve_imports(imports, import_map={})
        self.assertEqual(len(resolved), 1)
        self.assertEqual(sorted(resolved[0].import_files), ["test_a.py", "test_b.py"])

    def test_resolve_conditional_all(self) -> None:
        imports = [
            ImportInfo("yaml", "yaml", "test_a.py", 1, True),
            ImportInfo("yaml", "yaml", "test_b.py", 2, True),
        ]
        resolved, _ = resolve_imports(imports)
        self.assertEqual(len(resolved), 1)
        self.assertTrue(resolved[0].is_conditional)

    def test_resolve_conditional_mixed(self) -> None:
        imports = [
            ImportInfo("yaml", "yaml", "test_a.py", 1, True),
            ImportInfo("yaml", "yaml", "test_b.py", 2, False),
        ]
        resolved, _ = resolve_imports(imports)
        self.assertEqual(len(resolved), 1)
        self.assertFalse(resolved[0].is_conditional)

    def test_resolve_hyphens_normalized(self) -> None:
        imports = [ImportInfo("pytest_asyncio", "pytest_asyncio", "test.py", 1, False)]
        resolved, _ = resolve_imports(imports)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].pip_package, "pytest-asyncio")
        self.assertEqual(resolved[0].source, "mapping")


# ===========================================================================
# Install command comparison tests
# ===========================================================================


class TestCompareInstallCommand(unittest.TestCase):
    def test_compare_simple_pip_install(self) -> None:
        resolved = [
            _make_dep("pytest", "pytest"),
            _make_dep("hypothesis", "hypothesis"),
        ]
        already, missing = compare_with_install_command(resolved, "pip install pytest hypothesis")
        self.assertEqual(sorted(already), ["hypothesis", "pytest"])
        self.assertEqual(missing, [])

    def test_compare_editable_plus_deps(self) -> None:
        resolved = [
            _make_dep("pytest", "pytest"),
            _make_dep("hypothesis", "hypothesis"),
        ]
        already, missing = compare_with_install_command(
            resolved, "pip install -e . && pip install pytest"
        )
        self.assertEqual(already, ["pytest"])
        self.assertEqual(missing, ["hypothesis"])

    def test_compare_with_extras(self) -> None:
        resolved = [_make_dep("pytest", "pytest")]
        already, missing = compare_with_install_command(resolved, "pip install -e '.[test]'")
        # pytest not explicitly listed, so it's in missing.
        self.assertEqual(missing, ["pytest"])

    def test_compare_identifies_missing(self) -> None:
        resolved = [
            _make_dep("pytest", "pytest"),
            _make_dep("trustme", "trustme"),
            _make_dep("hypothesis", "hypothesis"),
        ]
        already, missing = compare_with_install_command(resolved, "pip install pytest")
        self.assertEqual(already, ["pytest"])
        self.assertEqual(sorted(missing), ["hypothesis", "trustme"])

    def test_compare_identifies_present(self) -> None:
        resolved = [
            _make_dep("pytest", "pytest"),
            _make_dep("yaml", "PyYAML"),
        ]
        already, missing = compare_with_install_command(
            resolved, "pip install -e . && pip install pytest PyYAML"
        )
        self.assertEqual(sorted(already), ["PyYAML", "pytest"])
        self.assertEqual(missing, [])

    def test_compare_empty_command(self) -> None:
        resolved = [_make_dep("pytest", "pytest")]
        already, missing = compare_with_install_command(resolved, "")
        self.assertEqual(already, [])
        self.assertEqual(missing, ["pytest"])

    def test_compare_case_insensitive(self) -> None:
        resolved = [_make_dep("yaml", "PyYAML")]
        already, missing = compare_with_install_command(resolved, "pip install pyyaml")
        self.assertEqual(already, ["PyYAML"])
        self.assertEqual(missing, [])


# ===========================================================================
# Full scan integration tests
# ===========================================================================


class TestScanPackageDeps(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.repo = _create_test_repo(self.tmpdir)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_scan_package_deps_autodetect_test_dir(self) -> None:
        result = scan_package_deps(self.repo, "mypackage")
        self.assertIn("tests", result.scan_dirs)
        self.assertGreater(result.total_files_scanned, 0)

    def test_scan_package_deps_explicit_test_dir(self) -> None:
        result = scan_package_deps(self.repo, "mypackage", test_dirs=["tests"])
        self.assertEqual(result.scan_dirs, ["tests"])

    def test_scan_package_deps_with_install_command(self) -> None:
        result = scan_package_deps(
            self.repo,
            "mypackage",
            install_command="pip install -e . && pip install pytest",
        )
        self.assertIn("pytest", result.already_installed)

    def test_scan_package_deps_no_test_dir(self) -> None:
        # Create repo with no standard test directory.
        repo = self.tmpdir / "norepo"
        repo.mkdir()
        (repo / "main.py").write_text("import click\n")
        result = scan_package_deps(repo, "norepo")
        self.assertEqual(result.scan_dirs, ["."])

    def test_scan_package_deps_suggested_install(self) -> None:
        result = scan_package_deps(self.repo, "mypackage")
        # Should suggest installing external deps.
        if result.missing:
            self.assertTrue(result.suggested_install.startswith("pip install"))

    def test_scan_filters_out_stdlib(self) -> None:
        result = scan_package_deps(self.repo, "mypackage", scan_source=True)
        # os, json are stdlib and should not appear in resolved deps.
        resolved_names = {dep.import_name for dep in result.resolved}
        self.assertNotIn("os", resolved_names)
        self.assertNotIn("json", resolved_names)

    def test_scan_filters_out_local(self) -> None:
        result = scan_package_deps(self.repo, "mypackage")
        resolved_names = {dep.import_name for dep in result.resolved}
        # mypackage is local and should be filtered.
        self.assertNotIn("mypackage", resolved_names)

    def test_scan_finds_external_deps(self) -> None:
        result = scan_package_deps(self.repo, "mypackage")
        resolved_names = {dep.import_name for dep in result.resolved}
        self.assertIn("pytest", resolved_names)
        self.assertIn("hypothesis", resolved_names)

    def test_scan_detects_conditional(self) -> None:
        result = scan_package_deps(self.repo, "mypackage")
        yaml_deps = [d for d in result.resolved if d.import_name == "yaml"]
        self.assertEqual(len(yaml_deps), 1)
        self.assertTrue(yaml_deps[0].is_conditional)

    def test_scan_with_scan_source(self) -> None:
        result = scan_package_deps(self.repo, "mypackage", scan_source=True)
        resolved_names = {dep.import_name for dep in result.resolved}
        # click is only in source, should appear with scan_source=True.
        self.assertIn("click", resolved_names)

    def test_scan_without_scan_source(self) -> None:
        result = scan_package_deps(self.repo, "mypackage", scan_source=False)
        resolved_names = {dep.import_name for dep in result.resolved}
        # click is only in source, should NOT appear without scan_source.
        self.assertNotIn("click", resolved_names)


# ===========================================================================
# Build scan result tests
# ===========================================================================


class TestBuildScanResult(unittest.TestCase):
    def test_build_with_install_command(self) -> None:
        imports = [ImportInfo("pytest", "pytest", "test.py", 1, False)]
        resolved = [_make_dep("pytest", "pytest")]
        result = build_scan_result(
            "mypkg",
            ["tests"],
            5,
            imports,
            resolved,
            [],
            install_command="pip install pytest",
        )
        self.assertEqual(result.already_installed, ["pytest"])
        self.assertEqual(result.missing, [])
        self.assertEqual(result.suggested_install, "")

    def test_build_without_install_command(self) -> None:
        imports = [ImportInfo("pytest", "pytest", "test.py", 1, False)]
        resolved = [_make_dep("pytest", "pytest")]
        result = build_scan_result("mypkg", ["tests"], 5, imports, resolved, [])
        self.assertEqual(result.already_installed, [])
        self.assertEqual(result.missing, ["pytest"])
        self.assertIn("pytest", result.suggested_install)


# ===========================================================================
# Import map tests
# ===========================================================================


class TestImportMap(unittest.TestCase):
    def test_map_has_known_entries(self) -> None:
        self.assertEqual(IMPORT_TO_PIP["PIL"], "Pillow")
        self.assertEqual(IMPORT_TO_PIP["yaml"], "PyYAML")
        self.assertEqual(IMPORT_TO_PIP["sklearn"], "scikit-learn")
        self.assertEqual(IMPORT_TO_PIP["bs4"], "beautifulsoup4")

    def test_map_has_testing_entries(self) -> None:
        self.assertEqual(IMPORT_TO_PIP["pytest_asyncio"], "pytest-asyncio")
        self.assertEqual(IMPORT_TO_PIP["pytest_mock"], "pytest-mock")
        self.assertEqual(IMPORT_TO_PIP["pytest_cov"], "pytest-cov")

    def test_map_size(self) -> None:
        self.assertGreaterEqual(len(IMPORT_TO_PIP), 100)


# ===========================================================================
# CLI tests
# ===========================================================================


class TestScanDepsCLI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.repo = _create_test_repo(self.tmpdir)
        self.runner = CliRunner()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_cli_human_format(self) -> None:
        result = self.runner.invoke(main, ["scan-deps", str(self.repo), "--format", "human"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Scanning:", result.output)
        self.assertIn("Files scanned:", result.output)
        self.assertIn("External dependencies:", result.output)

    def test_cli_json_format(self) -> None:
        result = self.runner.invoke(main, ["scan-deps", str(self.repo), "--format", "json"])
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["package_name"], "mypackage")
        self.assertIn("resolved", data)
        self.assertIn("unresolved", data)

    def test_cli_pip_format(self) -> None:
        result = self.runner.invoke(main, ["scan-deps", str(self.repo), "--format", "pip"])
        self.assertEqual(result.exit_code, 0, result.output)
        if result.output.strip():
            self.assertTrue(
                result.output.strip().startswith("pip install")
                or result.output.strip().startswith("#")
            )

    def test_cli_pip_format_nothing_missing(self) -> None:
        # Create repo with no external imports.
        repo = self.tmpdir / "noimports"
        repo.mkdir()
        tests = repo / "tests"
        tests.mkdir()
        (tests / "test_empty.py").write_text("import os\n\ndef test_x(): pass\n")
        result = self.runner.invoke(main, ["scan-deps", str(repo), "--format", "pip"])
        self.assertEqual(result.exit_code, 0, result.output)
        # No pip install line should be printed.
        self.assertNotIn("pip install", result.output)

    def test_cli_nonexistent_path(self) -> None:
        result = self.runner.invoke(main, ["scan-deps", "/nonexistent/path/123"])
        self.assertNotEqual(result.exit_code, 0)

    def test_cli_auto_detect_package_name(self) -> None:
        result = self.runner.invoke(main, ["scan-deps", str(self.repo), "--format", "json"])
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["package_name"], "mypackage")

    def test_cli_explicit_package_name(self) -> None:
        result = self.runner.invoke(
            main,
            ["scan-deps", str(self.repo), "--package-name", "custom-name", "--format", "json"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["package_name"], "custom-name")

    def test_cli_with_install_command(self) -> None:
        result = self.runner.invoke(
            main,
            [
                "scan-deps",
                str(self.repo),
                "--install-command",
                "pip install -e . && pip install pytest",
                "--format",
                "human",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Comparison with install_command:", result.output)

    def test_cli_with_test_dirs(self) -> None:
        result = self.runner.invoke(
            main,
            ["scan-deps", str(self.repo), "--test-dirs", "tests", "--format", "json"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        self.assertEqual(data["scan_dirs"], ["tests"])

    def test_cli_no_conditional(self) -> None:
        result = self.runner.invoke(
            main,
            ["scan-deps", str(self.repo), "--no-conditional", "--format", "json"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        data = json.loads(result.output)
        # yaml is conditional, should be excluded.
        resolved_names = {d["import_name"] for d in data["resolved"]}
        self.assertNotIn("yaml", resolved_names)

    def test_cli_with_registry_dir(self) -> None:
        # Create a minimal registry.
        reg_dir = self.tmpdir / "registry"
        pkg_dir = reg_dir / "packages"
        pkg_dir.mkdir(parents=True)

        import yaml

        data = {"package": "websocket-client", "import_name": "websocket"}
        (pkg_dir / "websocket-client.yaml").write_text(yaml.dump(data, default_flow_style=False))

        result = self.runner.invoke(
            main,
            [
                "scan-deps",
                str(self.repo),
                "--registry-dir",
                str(reg_dir),
                "--format",
                "json",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)


# ===========================================================================
# Helpers
# ===========================================================================


class TestNamespacePackages(unittest.TestCase):
    def test_resolve_namespace_package_note(self) -> None:
        imports = [ImportInfo("google", "google.cloud", "test.py", 1, False)]
        resolved, _ = resolve_imports(imports)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].pip_package, "google-cloud-core")
        self.assertIn("namespace package", resolved[0].note)
        self.assertIn("google.cloud", resolved[0].note)

    def test_resolve_namespace_full_path(self) -> None:
        imports = [ImportInfo("google", "google.cloud.storage", "test.py", 1, False)]
        resolved, _ = resolve_imports(imports)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].pip_package, "google-cloud-storage")

    def test_resolve_non_namespace_no_note(self) -> None:
        imports = [ImportInfo("pytest", "pytest", "test.py", 1, False)]
        resolved, _ = resolve_imports(imports, import_map={"pytest": "pytest"})
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].note, "")

    def test_resolved_dep_note_field(self) -> None:
        dep = ResolvedDep(
            import_name="google",
            pip_package="google-cloud-core",
            source="mapping",
            import_files=["test.py"],
            is_conditional=False,
            note="namespace package warning",
        )
        self.assertEqual(dep.note, "namespace package warning")

    def test_namespace_identity_gets_note(self) -> None:
        # A namespace package not in import map falls to identity mapping.
        imports = [ImportInfo("jaraco", "jaraco.functools", "test.py", 1, False)]
        resolved, _ = resolve_imports(imports, import_map={})
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].source, "identity")
        self.assertIn("namespace package", resolved[0].note)

    def test_namespace_full_path_no_match_falls_to_toplevel(self) -> None:
        # google.cloud.unknown isn't in the map, falls to top-level google.
        imports = [ImportInfo("google", "google.cloud.unknown", "test.py", 1, False)]
        resolved, _ = resolve_imports(imports)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].pip_package, "google-cloud-core")
        self.assertIn("namespace package", resolved[0].note)


def _make_dep(
    import_name: str,
    pip_package: str,
    source: str = "identity",
    files: list[str] | None = None,
    conditional: bool = False,
) -> ResolvedDep:
    return ResolvedDep(
        import_name=import_name,
        pip_package=pip_package,
        source=source,
        import_files=files or ["test.py"],
        is_conditional=conditional,
    )


class TestNormalizePipCommand(unittest.TestCase):
    def test_pip_install(self) -> None:
        result = _normalize_pip_command("pip install foo bar")
        self.assertEqual(result, " foo bar")

    def test_pip3_install(self) -> None:
        result = _normalize_pip_command("pip3 install foo")
        self.assertEqual(result, " foo")

    def test_python_m_pip_install(self) -> None:
        result = _normalize_pip_command("python -m pip install foo bar")
        self.assertEqual(result, " foo bar")

    def test_python3_m_pip_install(self) -> None:
        result = _normalize_pip_command("python3 -m pip install baz")
        self.assertEqual(result, " baz")

    def test_venv_pip_install(self) -> None:
        result = _normalize_pip_command("/tmp/venv/bin/pip install foo")
        self.assertEqual(result, " foo")

    def test_non_pip_command(self) -> None:
        result = _normalize_pip_command("make install")
        self.assertIsNone(result)

    def test_git_fetch_not_matched(self) -> None:
        result = _normalize_pip_command("git fetch --tags")
        self.assertIsNone(result)


class TestParseInstallPackages(unittest.TestCase):
    def test_basic_pip_install(self) -> None:
        pkgs, extras = _parse_install_packages("pip install foo bar")
        self.assertEqual(pkgs, ["foo", "bar"])
        self.assertEqual(extras, [])

    def test_python_m_pip(self) -> None:
        pkgs, extras = _parse_install_packages("python -m pip install foo bar")
        self.assertEqual(pkgs, ["foo", "bar"])

    def test_python3_m_pip(self) -> None:
        pkgs, extras = _parse_install_packages("python3 -m pip install baz")
        self.assertEqual(pkgs, ["baz"])

    def test_venv_pip(self) -> None:
        pkgs, extras = _parse_install_packages("/tmp/venv/bin/pip install foo")
        self.assertEqual(pkgs, ["foo"])

    def test_pip3_install(self) -> None:
        pkgs, extras = _parse_install_packages("pip3 install foo")
        self.assertEqual(pkgs, ["foo"])

    def test_chained_with_python_m(self) -> None:
        cmd = "python -m pip install -e . && python -m pip install pytest mock"
        pkgs, extras = _parse_install_packages(cmd)
        self.assertIn("pytest", pkgs)
        self.assertIn("mock", pkgs)

    def test_non_pip_command_ignored(self) -> None:
        pkgs, extras = _parse_install_packages("make install && pip install foo")
        self.assertEqual(pkgs, ["foo"])

    def test_editable_install_with_extras(self) -> None:
        pkgs, extras = _parse_install_packages("pip install -e '.[test]'")
        self.assertEqual(pkgs, [])
        self.assertTrue(len(extras) > 0)

    def test_version_specifiers(self) -> None:
        pkgs, extras = _parse_install_packages("pip install foo>=1.0 bar==2.0")
        self.assertEqual(pkgs, ["foo", "bar"])

    def test_tilde_equals(self) -> None:
        pkgs, extras = _parse_install_packages("pip install click~=8.0")
        self.assertEqual(pkgs, ["click"])

    def test_not_equals(self) -> None:
        pkgs, extras = _parse_install_packages("pip install foo!=1.0")
        self.assertEqual(pkgs, ["foo"])

    def test_env_marker(self) -> None:
        pkgs, extras = _parse_install_packages("pip install bar>=1.0;python_version>='3.8'")
        self.assertEqual(pkgs, ["bar"])

    def test_complex_specifier(self) -> None:
        pkgs, extras = _parse_install_packages("pip install baz[extra]>=2.0,<3.0")
        self.assertEqual(pkgs, ["baz"])


if __name__ == "__main__":
    unittest.main()
