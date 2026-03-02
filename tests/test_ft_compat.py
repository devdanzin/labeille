"""Tests for labeille.ft.compat — extension GIL compatibility detection."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from labeille.ft.compat import (
    ExtensionCompat,
    ExtensionInfo,
    ModGilDeclaration,
    SourceScanResult,
    assess_extension_compat,
    format_extension_compat,
    guess_import_name,
    probe_gil_fallback,
    scan_source_for_mod_gil,
)


# ---------------------------------------------------------------------------
# ExtensionCompat dataclass tests
# ---------------------------------------------------------------------------


class TestExtensionCompat(unittest.TestCase):
    def test_extension_compat_pure_python(self) -> None:
        compat = ExtensionCompat(
            package="mypackage",
            is_pure_python=True,
            import_ok=True,
            gil_fallback_active=False,
        )
        self.assertTrue(compat.fully_compatible)

    def test_extension_compat_with_fallback(self) -> None:
        compat = ExtensionCompat(
            package="mypackage",
            gil_fallback_active=True,
        )
        self.assertFalse(compat.fully_compatible)

    def test_extension_compat_import_failure(self) -> None:
        compat = ExtensionCompat(
            package="mypackage",
            import_ok=False,
        )
        self.assertFalse(compat.fully_compatible)

    def test_extension_compat_incompatible_extensions(self) -> None:
        compat = ExtensionCompat(
            package="mypackage",
            all_extensions_compatible=False,
        )
        self.assertFalse(compat.fully_compatible)

    def test_extension_compat_serialization_roundtrip(self) -> None:
        compat = ExtensionCompat(
            package="mypackage",
            is_pure_python=False,
            extensions=[
                ExtensionInfo(
                    module_name="mypackage._accel",
                    is_extension=True,
                    import_ok=True,
                ),
            ],
            gil_fallback_active=True,
            all_extensions_compatible=False,
            import_ok=True,
            source_scan=SourceScanResult(
                files_scanned=10,
                files_with_mod_gil=1,
                declarations=[
                    ModGilDeclaration(
                        file="src/_accel.c",
                        line_number=42,
                        line_text="{Py_mod_gil, Py_MOD_GIL_NOT_USED}",
                        is_not_used=True,
                    ),
                ],
            ),
            probe_error=None,
        )
        d = compat.to_dict()
        restored = ExtensionCompat.from_dict(d)
        self.assertEqual(restored.package, "mypackage")
        self.assertFalse(restored.is_pure_python)
        self.assertTrue(restored.gil_fallback_active)
        self.assertFalse(restored.all_extensions_compatible)
        self.assertTrue(restored.import_ok)
        self.assertEqual(len(restored.extensions), 1)
        self.assertEqual(restored.extensions[0].module_name, "mypackage._accel")
        self.assertIsNotNone(restored.source_scan)
        assert restored.source_scan is not None
        self.assertEqual(restored.source_scan.files_scanned, 10)
        self.assertEqual(len(restored.source_scan.declarations), 1)
        self.assertTrue(restored.source_scan.declarations[0].is_not_used)

    def test_extension_compat_from_dict_missing_fields(self) -> None:
        d = {"package": "minimal"}
        restored = ExtensionCompat.from_dict(d)
        self.assertEqual(restored.package, "minimal")
        self.assertTrue(restored.is_pure_python)
        self.assertTrue(restored.import_ok)
        self.assertFalse(restored.gil_fallback_active)
        self.assertIsNone(restored.source_scan)
        self.assertIsNone(restored.probe_error)


# ---------------------------------------------------------------------------
# SourceScanResult tests
# ---------------------------------------------------------------------------


class TestSourceScanResult(unittest.TestCase):
    def test_source_scan_all_not_used(self) -> None:
        result = SourceScanResult(
            declarations=[
                ModGilDeclaration("a.c", 1, "...", is_not_used=True),
                ModGilDeclaration("b.c", 2, "...", is_not_used=True),
            ],
        )
        self.assertTrue(result.all_not_used)
        self.assertFalse(result.has_required)
        self.assertTrue(result.has_any_declaration)

    def test_source_scan_mixed(self) -> None:
        result = SourceScanResult(
            declarations=[
                ModGilDeclaration("a.c", 1, "...", is_not_used=True),
                ModGilDeclaration("b.c", 2, "...", is_not_used=False),
            ],
        )
        self.assertFalse(result.all_not_used)
        self.assertTrue(result.has_required)

    def test_source_scan_empty(self) -> None:
        result = SourceScanResult()
        self.assertFalse(result.has_any_declaration)
        self.assertFalse(result.all_not_used)
        self.assertFalse(result.has_required)

    def test_source_scan_serialization_roundtrip(self) -> None:
        result = SourceScanResult(
            files_scanned=5,
            files_with_mod_gil=2,
            declarations=[
                ModGilDeclaration("x.c", 10, "line text", is_not_used=True),
            ],
        )
        d = result.to_dict()
        restored = SourceScanResult.from_dict(d)
        self.assertEqual(restored.files_scanned, 5)
        self.assertEqual(restored.files_with_mod_gil, 2)
        self.assertEqual(len(restored.declarations), 1)
        self.assertTrue(restored.declarations[0].is_not_used)


# ---------------------------------------------------------------------------
# Import name guessing tests
# ---------------------------------------------------------------------------


class TestGuessImportName(unittest.TestCase):
    def test_guess_import_name_override(self) -> None:
        self.assertEqual(guess_import_name("pillow"), "PIL")

    def test_guess_import_name_override_case_insensitive(self) -> None:
        self.assertEqual(guess_import_name("PyYAML"), "yaml")

    def test_guess_import_name_hyphen(self) -> None:
        self.assertEqual(guess_import_name("my-package"), "my_package")

    def test_guess_import_name_simple(self) -> None:
        self.assertEqual(guess_import_name("requests"), "requests")

    def test_guess_import_name_scikit_learn(self) -> None:
        self.assertEqual(guess_import_name("scikit-learn"), "sklearn")

    def test_guess_import_name_attrs(self) -> None:
        self.assertEqual(guess_import_name("attrs"), "attr")

    def test_guess_import_name_beautifulsoup4(self) -> None:
        self.assertEqual(guess_import_name("beautifulsoup4"), "bs4")


# ---------------------------------------------------------------------------
# Source scanning tests
# ---------------------------------------------------------------------------


class TestScanSourceForModGil(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = Path(self.tmpdir)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_file(self, relpath: str, content: str) -> Path:
        p = self.repo / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def test_scan_finds_not_used(self) -> None:
        self._write_file("src/mod.c", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        result = scan_source_for_mod_gil(self.repo)
        self.assertEqual(len(result.declarations), 1)
        self.assertTrue(result.declarations[0].is_not_used)
        self.assertEqual(result.files_with_mod_gil, 1)

    def test_scan_finds_used(self) -> None:
        self._write_file("src/mod.c", "{Py_mod_gil, Py_MOD_GIL_USED}")
        result = scan_source_for_mod_gil(self.repo)
        self.assertEqual(len(result.declarations), 1)
        self.assertFalse(result.declarations[0].is_not_used)

    def test_scan_multiple_files(self) -> None:
        self._write_file("a.c", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        self._write_file("b.c", "{Py_mod_gil, Py_MOD_GIL_USED}")
        result = scan_source_for_mod_gil(self.repo)
        self.assertEqual(result.files_with_mod_gil, 2)
        self.assertEqual(len(result.declarations), 2)

    def test_scan_skips_git_dir(self) -> None:
        self._write_file(".git/hooks/mod.c", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        result = scan_source_for_mod_gil(self.repo)
        self.assertEqual(len(result.declarations), 0)

    def test_scan_skips_build_dir(self) -> None:
        self._write_file("build/mod.c", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        result = scan_source_for_mod_gil(self.repo)
        self.assertEqual(len(result.declarations), 0)

    def test_scan_respects_max_files(self) -> None:
        for i in range(5):
            self._write_file(f"mod{i}.c", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        result = scan_source_for_mod_gil(self.repo, max_files=1)
        self.assertLessEqual(result.files_scanned, 1)

    def test_scan_ignores_non_source(self) -> None:
        self._write_file("mod.py", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        self._write_file("mod.txt", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        result = scan_source_for_mod_gil(self.repo)
        self.assertEqual(result.files_scanned, 0)
        self.assertEqual(len(result.declarations), 0)

    def test_scan_empty_repo(self) -> None:
        result = scan_source_for_mod_gil(self.repo)
        self.assertEqual(result.files_scanned, 0)
        self.assertEqual(len(result.declarations), 0)

    def test_scan_handles_binary_files(self) -> None:
        p = self.repo / "mod.c"
        p.write_bytes(b"\x80\x81\x82 {Py_mod_gil, Py_MOD_GIL_NOT_USED} \xff\xfe")
        result = scan_source_for_mod_gil(self.repo)
        # Should not crash; file is read with errors="replace".
        self.assertEqual(result.files_scanned, 1)
        self.assertEqual(len(result.declarations), 1)

    def test_scan_pyx_files(self) -> None:
        self._write_file("ext.pyx", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        result = scan_source_for_mod_gil(self.repo)
        self.assertEqual(len(result.declarations), 1)

    def test_scan_realistic_c_source(self) -> None:
        source = """\
#include <Python.h>

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, module_exec},
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
    {0, NULL},
};
"""
        self._write_file("src/_accel.c", source)
        result = scan_source_for_mod_gil(self.repo)
        self.assertEqual(len(result.declarations), 1)
        decl = result.declarations[0]
        self.assertTrue(decl.is_not_used)
        self.assertEqual(decl.line_number, 5)
        self.assertIn("Py_MOD_GIL_NOT_USED", decl.line_text)

    def test_scan_multiline_declaration(self) -> None:
        # One-line variant: should be found.
        self._write_file("oneline.c", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        # Two-line variant: line-based regex won't match — this is acceptable.
        self._write_file("twoline.c", "{Py_mod_gil,\n    Py_MOD_GIL_NOT_USED}")
        result = scan_source_for_mod_gil(self.repo)
        # Only the one-line version should be found.
        self.assertEqual(len(result.declarations), 1)
        self.assertEqual(result.declarations[0].file, "oneline.c")

    def test_scan_commented_out(self) -> None:
        # Commented-out declarations are still found — intentionally.
        # They indicate developer awareness of free-threading.
        self._write_file("mod.c", "// {Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        result = scan_source_for_mod_gil(self.repo)
        self.assertEqual(len(result.declarations), 1)

    def test_scan_large_file_skipped(self) -> None:
        # Create a file larger than max_file_size.
        self._write_file("big.c", "x" * 100 + "\n{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        result = scan_source_for_mod_gil(self.repo, max_file_size=50)
        self.assertEqual(len(result.declarations), 0)


# ---------------------------------------------------------------------------
# Runtime probe tests (mocked)
# ---------------------------------------------------------------------------


class TestProbeGilFallback(unittest.TestCase):
    def _make_probe_result(self, **overrides: object) -> str:
        data: dict[str, object] = {
            "package": "mypkg",
            "import_ok": True,
            "import_error": None,
            "gil_enabled_before": False,
            "gil_enabled_after": False,
            "gil_fallback": False,
            "is_pure_python": True,
            "extensions_found": [],
        }
        data.update(overrides)
        return json.dumps(data)

    @patch("labeille.ft.compat.subprocess.run")
    def test_probe_successful_pure_python(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._make_probe_result(),
            stderr="",
        )
        result = probe_gil_fallback("mypkg", Path("/fake/bin/python"))
        self.assertTrue(result.import_ok)
        self.assertTrue(result.is_pure_python)
        self.assertTrue(result.fully_compatible)

    @patch("labeille.ft.compat.subprocess.run")
    def test_probe_successful_with_fallback(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._make_probe_result(
                gil_fallback=True,
                is_pure_python=False,
                extensions_found=[
                    {"module_name": "mypkg._accel", "is_extension": True, "file": "/x.so"},
                ],
            ),
            stderr="",
        )
        result = probe_gil_fallback("mypkg", Path("/fake/bin/python"))
        self.assertTrue(result.gil_fallback_active)
        self.assertFalse(result.is_pure_python)
        self.assertFalse(result.fully_compatible)
        self.assertEqual(len(result.extensions), 1)

    @patch("labeille.ft.compat.subprocess.run")
    def test_probe_import_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._make_probe_result(
                import_ok=False,
                import_error="No module named 'foo'",
            ),
            stderr="",
        )
        result = probe_gil_fallback("foo", Path("/fake/bin/python"))
        self.assertFalse(result.import_ok)
        self.assertEqual(result.import_error, "No module named 'foo'")

    @patch("labeille.ft.compat.subprocess.run")
    def test_probe_subprocess_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=60)
        result = probe_gil_fallback("mypkg", Path("/fake/bin/python"))
        self.assertIn("timed out", result.probe_error or "")
        self.assertFalse(result.import_ok)

    @patch("labeille.ft.compat.subprocess.run")
    def test_probe_subprocess_crash(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=-11,
            stdout="",
            stderr="Segmentation fault",
        )
        result = probe_gil_fallback("mypkg", Path("/fake/bin/python"))
        self.assertFalse(result.import_ok)
        self.assertIn("-11", result.probe_error or "")

    @patch("labeille.ft.compat.subprocess.run")
    def test_probe_invalid_json(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not json at all",
            stderr="",
        )
        result = probe_gil_fallback("mypkg", Path("/fake/bin/python"))
        self.assertIn("JSON", result.probe_error or "")

    @patch("labeille.ft.compat.subprocess.run")
    def test_probe_nonexistent_python(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("No such file")
        result = probe_gil_fallback("mypkg", Path("/nonexistent/python"))
        self.assertIn("Could not run probe", result.probe_error or "")

    @patch("labeille.ft.compat.subprocess.run")
    def test_probe_sets_python_gil_0(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._make_probe_result(),
            stderr="",
        )
        probe_gil_fallback("mypkg", Path("/fake/bin/python"))
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["env"]["PYTHON_GIL"], "0")

    @patch("labeille.ft.compat.subprocess.run")
    def test_probe_strips_pythonhome(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._make_probe_result(),
            stderr="",
        )
        with patch.dict(os.environ, {"PYTHONHOME": "/bad"}):
            probe_gil_fallback("mypkg", Path("/fake/bin/python"))
        call_kwargs = mock_run.call_args[1]
        self.assertNotIn("PYTHONHOME", call_kwargs["env"])

    @patch("labeille.ft.compat.subprocess.run")
    def test_probe_no_output(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )
        result = probe_gil_fallback("mypkg", Path("/fake/bin/python"))
        self.assertIn("no output", result.probe_error or "")


# ---------------------------------------------------------------------------
# Combined assessment tests (mocked)
# ---------------------------------------------------------------------------


class TestAssessExtensionCompat(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.repo = Path(self.tmpdir)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_file(self, relpath: str, content: str) -> None:
        p = self.repo / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    @patch("labeille.ft.compat.probe_gil_fallback")
    def test_assess_both_probe_and_scan(self, mock_probe: MagicMock) -> None:
        mock_probe.return_value = ExtensionCompat(
            package="mypkg",
            import_ok=True,
            is_pure_python=False,
            gil_fallback_active=False,
        )
        self._write_file("src/mod.c", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")

        result = assess_extension_compat(
            "mypkg",
            venv_python=Path("/fake/bin/python"),
            repo_dir=self.repo,
        )
        self.assertTrue(result.import_ok)
        self.assertIsNotNone(result.source_scan)
        assert result.source_scan is not None
        self.assertEqual(len(result.source_scan.declarations), 1)

    def test_assess_scan_only(self) -> None:
        self._write_file("src/mod.c", "{Py_mod_gil, Py_MOD_GIL_NOT_USED}")
        result = assess_extension_compat("mypkg", repo_dir=self.repo)
        self.assertIsNotNone(result.source_scan)
        # Without runtime probe, infer from source.
        self.assertFalse(result.is_pure_python)
        self.assertTrue(result.all_extensions_compatible)

    @patch("labeille.ft.compat.probe_gil_fallback")
    def test_assess_probe_only(self, mock_probe: MagicMock) -> None:
        mock_probe.return_value = ExtensionCompat(
            package="mypkg",
            import_ok=True,
            is_pure_python=True,
        )
        result = assess_extension_compat(
            "mypkg",
            venv_python=Path("/fake/bin/python"),
        )
        self.assertIsNone(result.source_scan)
        self.assertTrue(result.is_pure_python)

    @patch("labeille.ft.compat.probe_gil_fallback")
    def test_assess_uses_custom_import_name(self, mock_probe: MagicMock) -> None:
        mock_probe.return_value = ExtensionCompat(package="PIL")
        assess_extension_compat(
            "pillow",
            venv_python=Path("/fake/bin/python"),
            import_name="PIL",
        )
        mock_probe.assert_called_once()
        self.assertEqual(mock_probe.call_args[0][0], "PIL")

    def test_assess_scan_with_used_declaration(self) -> None:
        self._write_file("src/mod.c", "{Py_mod_gil, Py_MOD_GIL_USED}")
        result = assess_extension_compat("mypkg", repo_dir=self.repo)
        self.assertFalse(result.is_pure_python)
        self.assertFalse(result.all_extensions_compatible)

    def test_assess_no_source_no_probe(self) -> None:
        result = assess_extension_compat("mypkg")
        self.assertEqual(result.package, "mypkg")
        self.assertTrue(result.is_pure_python)
        self.assertIsNone(result.source_scan)


# ---------------------------------------------------------------------------
# Display tests
# ---------------------------------------------------------------------------


class TestFormatExtensionCompat(unittest.TestCase):
    def test_format_pure_python(self) -> None:
        compat = ExtensionCompat(package="mypkg", is_pure_python=True, import_ok=True)
        output = format_extension_compat(compat)
        self.assertIn("Pure Python", output)

    def test_format_with_fallback(self) -> None:
        compat = ExtensionCompat(
            package="mypkg",
            is_pure_python=False,
            import_ok=True,
            gil_fallback_active=True,
            extensions=[
                ExtensionInfo(module_name="mypkg._ext", is_extension=True),
            ],
        )
        output = format_extension_compat(compat)
        self.assertIn("ACTIVE", output)
        self.assertIn("mypkg._ext", output)

    def test_format_with_source_scan(self) -> None:
        compat = ExtensionCompat(
            package="mypkg",
            import_ok=True,
            is_pure_python=False,
            source_scan=SourceScanResult(
                files_scanned=10,
                files_with_mod_gil=1,
                declarations=[
                    ModGilDeclaration("mod.c", 42, "...", is_not_used=True),
                ],
            ),
        )
        output = format_extension_compat(compat)
        self.assertIn("Py_mod_gil declarations", output)
        self.assertIn("mod.c:42", output)
        self.assertIn("free-threading support", output)

    def test_format_import_failure(self) -> None:
        compat = ExtensionCompat(
            package="mypkg",
            import_ok=False,
            import_error="No module named 'mypkg'",
        )
        output = format_extension_compat(compat)
        self.assertIn("Import failed", output)
        self.assertIn("No module named", output)

    def test_format_source_scan_with_used(self) -> None:
        compat = ExtensionCompat(
            package="mypkg",
            import_ok=True,
            is_pure_python=False,
            source_scan=SourceScanResult(
                files_scanned=5,
                declarations=[
                    ModGilDeclaration("mod.c", 10, "...", is_not_used=False),
                ],
            ),
        )
        output = format_extension_compat(compat)
        self.assertIn("GIL requirement", output)

    def test_format_no_declarations_extension(self) -> None:
        compat = ExtensionCompat(
            package="mypkg",
            import_ok=True,
            is_pure_python=False,
            source_scan=SourceScanResult(files_scanned=5),
        )
        output = format_extension_compat(compat)
        self.assertIn("No Py_mod_gil declarations found", output)


if __name__ == "__main__":
    unittest.main()
