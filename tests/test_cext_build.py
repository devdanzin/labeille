"""Tests for labeille.cext_build — compile database generation."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from labeille.cext_build import (
    CextBuildConfig,
    CextBuildMeta,
    CextBuildResult,
    build_package_cext,
    detect_bear,
    detect_build_system,
    extract_build_requires,
    find_compile_db,
    postprocess_compile_db,
)


# ---------------------------------------------------------------------------
# detect_bear
# ---------------------------------------------------------------------------


class TestDetectBear(unittest.TestCase):
    @patch("labeille.cext_build.subprocess.run")
    @patch("labeille.cext_build.shutil.which", return_value="/usr/bin/bear")
    def test_bear_found(self, _mock_which: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="bear 3.1.0\n", stderr="")
        path, version = detect_bear()
        self.assertEqual(path, "/usr/bin/bear")
        self.assertEqual(version, "bear 3.1.0")

    @patch("labeille.cext_build.shutil.which", return_value=None)
    def test_bear_not_found(self, _mock_which: MagicMock) -> None:
        path, version = detect_bear()
        self.assertIsNone(path)
        self.assertEqual(version, "")

    @patch("labeille.cext_build.subprocess.run", side_effect=OSError("fail"))
    @patch("labeille.cext_build.shutil.which", return_value="/usr/bin/bear")
    def test_bear_version_fails(self, _w: MagicMock, _r: MagicMock) -> None:
        path, version = detect_bear()
        self.assertEqual(path, "/usr/bin/bear")
        self.assertEqual(version, "unknown")


# ---------------------------------------------------------------------------
# detect_build_system
# ---------------------------------------------------------------------------


class TestDetectBuildSystem(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_meson_build_file(self) -> None:
        (self.repo / "meson.build").write_text("project('x')\n")
        self.assertEqual(detect_build_system(self.repo), "meson")

    def test_cmakelists(self) -> None:
        (self.repo / "CMakeLists.txt").write_text("cmake_minimum_required()\n")
        self.assertEqual(detect_build_system(self.repo), "cmake")

    def test_pyproject_mesonpy(self) -> None:
        (self.repo / "pyproject.toml").write_text('[build-system]\nbuild-backend = "mesonpy"\n')
        self.assertEqual(detect_build_system(self.repo), "meson")

    def test_pyproject_skbuild(self) -> None:
        (self.repo / "pyproject.toml").write_text(
            '[build-system]\nbuild-backend = "scikit_build_core.build"\n'
        )
        self.assertEqual(detect_build_system(self.repo), "cmake")

    def test_pyproject_setuptools(self) -> None:
        (self.repo / "pyproject.toml").write_text(
            '[build-system]\nbuild-backend = "setuptools.build_meta"\n'
        )
        self.assertEqual(detect_build_system(self.repo), "setuptools")

    def test_pyproject_flit(self) -> None:
        (self.repo / "pyproject.toml").write_text(
            '[build-system]\nbuild-backend = "flit_core.buildapi"\n'
        )
        self.assertEqual(detect_build_system(self.repo), "flit")

    def test_pyproject_hatch(self) -> None:
        (self.repo / "pyproject.toml").write_text(
            '[build-system]\nbuild-backend = "hatchling.build"\n'
        )
        self.assertEqual(detect_build_system(self.repo), "hatch")

    def test_setup_py_only(self) -> None:
        (self.repo / "setup.py").write_text("from setuptools import setup\n")
        self.assertEqual(detect_build_system(self.repo), "setuptools")

    def test_empty_dir(self) -> None:
        self.assertEqual(detect_build_system(self.repo), "unknown")

    def test_meson_build_overrides_pyproject(self) -> None:
        (self.repo / "meson.build").write_text("project('x')\n")
        (self.repo / "pyproject.toml").write_text(
            '[build-system]\nbuild-backend = "setuptools.build_meta"\n'
        )
        self.assertEqual(detect_build_system(self.repo), "meson")


# ---------------------------------------------------------------------------
# extract_build_requires
# ---------------------------------------------------------------------------


class TestExtractBuildRequires(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_standard_pyproject(self) -> None:
        (self.repo / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["setuptools>=68", "cython"]\n'
        )
        self.assertEqual(extract_build_requires(self.repo), ["setuptools>=68", "cython"])

    def test_no_pyproject(self) -> None:
        self.assertEqual(extract_build_requires(self.repo), ["setuptools", "wheel"])

    def test_no_build_system_key(self) -> None:
        (self.repo / "pyproject.toml").write_text("[project]\nname = 'foo'\n")
        self.assertEqual(extract_build_requires(self.repo), ["setuptools", "wheel"])

    def test_empty_requires(self) -> None:
        (self.repo / "pyproject.toml").write_text("[build-system]\nrequires = []\n")
        self.assertEqual(extract_build_requires(self.repo), ["setuptools", "wheel"])


# ---------------------------------------------------------------------------
# find_compile_db
# ---------------------------------------------------------------------------


class TestFindCompileDb(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_in_repo_root(self) -> None:
        db = self.repo / "compile_commands.json"
        db.write_text("[]")
        self.assertEqual(find_compile_db(self.repo), db)

    def test_in_builddir(self) -> None:
        (self.repo / "builddir").mkdir()
        db = self.repo / "builddir" / "compile_commands.json"
        db.write_text("[]")
        self.assertEqual(find_compile_db(self.repo), db)

    def test_in_build(self) -> None:
        (self.repo / "build").mkdir()
        db = self.repo / "build" / "compile_commands.json"
        db.write_text("[]")
        self.assertEqual(find_compile_db(self.repo), db)

    def test_in_random_subdir(self) -> None:
        (self.repo / "my_build").mkdir()
        db = self.repo / "my_build" / "compile_commands.json"
        db.write_text("[]")
        result = find_compile_db(self.repo)
        self.assertIsNotNone(result)
        self.assertEqual(result, db)

    def test_not_found(self) -> None:
        self.assertIsNone(find_compile_db(self.repo))


# ---------------------------------------------------------------------------
# postprocess_compile_db
# ---------------------------------------------------------------------------


class TestPostprocessCompileDb(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_valid_paths_unchanged(self) -> None:
        src = self.repo / "src" / "foo.c"
        src.parent.mkdir(parents=True)
        src.write_text("int main() {}")
        db_path = self.repo / "compile_commands.json"
        entries = [{"directory": str(self.repo), "file": str(src), "command": "cc foo.c"}]
        db_path.write_text(json.dumps(entries))
        count = postprocess_compile_db(db_path, self.repo)
        self.assertEqual(count, 1)

    def test_rewrites_broken_paths(self) -> None:
        src = self.repo / "src" / "foo.c"
        src.parent.mkdir(parents=True)
        src.write_text("int main() {}")
        db_path = self.repo / "compile_commands.json"
        entries = [
            {"directory": "/tmp/nonexistent", "file": "/tmp/nonexistent/foo.c", "command": "cc"}
        ]
        db_path.write_text(json.dumps(entries))
        count = postprocess_compile_db(db_path, self.repo)
        self.assertEqual(count, 1)
        updated = json.loads(db_path.read_text())
        self.assertEqual(updated[0]["file"], str(src))

    def test_ambiguous_filename_not_rewritten(self) -> None:
        for d in ("src", "lib"):
            p = self.repo / d / "bar.c"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("int x;")
        db_path = self.repo / "compile_commands.json"
        entries = [{"directory": "/tmp/x", "file": "/tmp/x/bar.c", "command": "cc"}]
        db_path.write_text(json.dumps(entries))
        count = postprocess_compile_db(db_path, self.repo)
        self.assertEqual(count, 0)

    def test_empty_compile_db(self) -> None:
        db_path = self.repo / "compile_commands.json"
        db_path.write_text("[]")
        self.assertEqual(postprocess_compile_db(db_path, self.repo), 0)


# ---------------------------------------------------------------------------
# CextBuildResult serialization
# ---------------------------------------------------------------------------


class TestCextBuildResult(unittest.TestCase):
    def test_to_dict_sparse(self) -> None:
        r = CextBuildResult(package="numpy", status="ok", compile_db_entries=14)
        d = r.to_dict()
        self.assertEqual(d["package"], "numpy")
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["compile_db_entries"], 14)
        self.assertNotIn("error_summary", d)

    def test_to_dict_roundtrip(self) -> None:
        r = CextBuildResult(package="lxml", status="build_fail", exit_code=1, error_summary="err")
        d = r.to_dict()
        r2 = CextBuildResult.from_dict(d)
        self.assertEqual(r2.package, "lxml")
        self.assertEqual(r2.status, "build_fail")
        self.assertEqual(r2.exit_code, 1)

    def test_from_dict_unknown_fields(self) -> None:
        d: dict[str, Any] = {"package": "foo", "status": "ok", "unknown_field": 42}
        r = CextBuildResult.from_dict(d)
        self.assertEqual(r.package, "foo")


# ---------------------------------------------------------------------------
# CextBuildMeta serialization
# ---------------------------------------------------------------------------


class TestCextBuildMeta(unittest.TestCase):
    def test_to_dict_roundtrip(self) -> None:
        m = CextBuildMeta(
            build_id="cext_test",
            target_python="/usr/bin/python3",
            python_version="3.15.0",
            started_at="2026-01-01T00:00:00",
            bear_available=True,
            bear_version="3.1.0",
        )
        d = m.to_dict()
        m2 = CextBuildMeta.from_dict(d)
        self.assertEqual(m2.build_id, "cext_test")
        self.assertTrue(m2.bear_available)


# ---------------------------------------------------------------------------
# build_package_cext
# ---------------------------------------------------------------------------


class TestBuildPackageCext(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.output_dir = self.tmpdir / "output"
        self.output_dir.mkdir()
        self.config = CextBuildConfig(
            target_python=Path("/usr/bin/python3"),
            output_dir=self.tmpdir / "builds",
            repos_dir=self.tmpdir / "repos",
            venvs_dir=self.tmpdir / "venvs",
            skip_if_exists=False,
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_pkg(self, **kwargs: Any) -> MagicMock:
        pkg = MagicMock()
        pkg.package = kwargs.get("package", "mypkg")
        pkg.repo = kwargs.get("repo", "https://github.com/user/mypkg")
        pkg.extension_type = kwargs.get("extension_type", "extensions")
        pkg.install_command = kwargs.get("install_command", "pip install -e .")
        return pkg

    def test_pure_python_skipped(self) -> None:
        pkg = self._make_pkg(extension_type="pure")
        result = build_package_cext(pkg, self.config, self.output_dir)
        self.assertEqual(result.status, "pure_python")

    def test_no_repo_skipped(self) -> None:
        pkg = self._make_pkg(repo=None)
        result = build_package_cext(pkg, self.config, self.output_dir)
        self.assertEqual(result.status, "no_repo")

    @patch("labeille.cext_build.install_package")
    @patch("labeille.cext_build.create_venv")
    @patch("labeille.cext_build.clone_repo")
    def test_build_success_with_bear(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
    ) -> None:
        mock_clone.return_value = "abc123"

        # Set up repo dir and source file.
        repo_dir = self.config.repos_dir / "mypkg"  # type: ignore[operator]
        repo_dir.mkdir(parents=True)
        src_file = repo_dir / "src" / "mod.c"
        src_file.parent.mkdir(parents=True)
        src_file.write_text("int x;")

        # install_package side_effect: simulate bear creating compile_commands.json.
        def _create_db(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            db = repo_dir / "compile_commands.json"
            db.write_text(
                json.dumps(
                    [{"directory": str(repo_dir), "file": str(src_file), "command": "cc"}] * 5
                )
            )
            return subprocess.CompletedProcess(args="", returncode=0, stdout="", stderr="")

        mock_install.side_effect = _create_db

        pkg = self._make_pkg()
        result = build_package_cext(pkg, self.config, self.output_dir, bear_path="/usr/bin/bear")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.compile_db_method, "bear")
        self.assertEqual(result.compile_db_entries, 5)

    @patch("labeille.cext_build.install_package")
    @patch("labeille.cext_build.create_venv")
    @patch("labeille.cext_build.clone_repo")
    def test_build_fail(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
    ) -> None:
        mock_clone.return_value = "abc123"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=1, stdout="", stderr="error: compile failed"
        )
        repo_dir = self.config.repos_dir / "mypkg"  # type: ignore[operator]
        repo_dir.mkdir(parents=True)

        pkg = self._make_pkg()
        result = build_package_cext(pkg, self.config, self.output_dir)
        self.assertEqual(result.status, "build_fail")
        self.assertEqual(result.exit_code, 1)

    @patch("labeille.cext_build.install_package")
    @patch("labeille.cext_build.create_venv")
    @patch("labeille.cext_build.clone_repo")
    def test_build_timeout(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
    ) -> None:
        mock_clone.return_value = "abc123"
        mock_install.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=600)
        repo_dir = self.config.repos_dir / "mypkg"  # type: ignore[operator]
        repo_dir.mkdir(parents=True)

        pkg = self._make_pkg()
        result = build_package_cext(pkg, self.config, self.output_dir)
        self.assertEqual(result.status, "timeout")

    @patch("labeille.cext_build.install_package")
    @patch("labeille.cext_build.create_venv")
    @patch("labeille.cext_build.clone_repo")
    def test_no_compile_db_generated(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
    ) -> None:
        mock_clone.return_value = "abc123"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        repo_dir = self.config.repos_dir / "mypkg"  # type: ignore[operator]
        repo_dir.mkdir(parents=True)

        pkg = self._make_pkg()
        result = build_package_cext(pkg, self.config, self.output_dir)
        self.assertEqual(result.status, "no_compile_db")

    def test_skip_if_exists(self) -> None:
        self.config.skip_if_exists = True
        (self.output_dir / "compile_commands.json").write_text("[]")
        pkg = self._make_pkg()
        result = build_package_cext(pkg, self.config, self.output_dir)
        self.assertEqual(result.status, "skipped")

    @patch("labeille.cext_build.clone_repo", side_effect=OSError("network error"))
    def test_clone_error(self, _mock: MagicMock) -> None:
        pkg = self._make_pkg()
        result = build_package_cext(pkg, self.config, self.output_dir)
        self.assertEqual(result.status, "clone_error")
        self.assertIn("network error", result.error_summary)
