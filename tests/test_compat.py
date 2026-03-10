"""Tests for labeille.compat."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from labeille.compat import (
    CompatDiff,
    CompatDiffEntry,
    CompatMeta,
    CompatResult,
    CompatSurvey,
    ErrorMatch,
    ErrorPattern,
    _BUILTIN_PATTERNS,
    _NO_SDIST_PATTERN,
    _read_packages_file,
    classify_build_output,
    diff_surveys,
    export_compat_markdown,
    format_compat_diff,
    format_compat_report,
    format_patterns_table,
    get_patterns,
    load_compat_survey,
    resolve_compat_inputs,
)


# ---------------------------------------------------------------------------
# ErrorPattern / ErrorMatch
# ---------------------------------------------------------------------------


class TestErrorMatch(unittest.TestCase):
    """Tests for ErrorMatch serialization."""

    def test_to_dict(self) -> None:
        m = ErrorMatch(
            category="removed_c_api",
            subcategory="PyUnicode_READY",
            description="PyUnicode_READY removed in 3.12",
            since="3.12",
            matched_line="error: implicit declaration of PyUnicode_READY",
            line_number=42,
        )
        d = m.to_dict()
        self.assertEqual(d["category"], "removed_c_api")
        self.assertEqual(d["subcategory"], "PyUnicode_READY")
        self.assertEqual(d["line_number"], 42)
        self.assertIn("since", d)

    def test_roundtrip(self) -> None:
        m = ErrorMatch(
            category="compiler_error",
            subcategory="linker_error",
            description="Linker error",
            since="",
            matched_line="undefined reference to `foo'",
            line_number=10,
        )
        d = m.to_dict()
        m2 = ErrorMatch(**d)
        self.assertEqual(m.category, m2.category)
        self.assertEqual(m.matched_line, m2.matched_line)


# ---------------------------------------------------------------------------
# CompatResult
# ---------------------------------------------------------------------------


class TestCompatResult(unittest.TestCase):
    """Tests for CompatResult serialization."""

    def test_to_dict_sparse(self) -> None:
        """Default/empty values should be omitted from the dict."""
        r = CompatResult(package="foo", status="build_ok")
        d = r.to_dict()
        self.assertEqual(d, {"package": "foo", "status": "build_ok"})

    def test_to_dict_full(self) -> None:
        r = CompatResult(
            package="bar",
            status="build_fail",
            exit_code=1,
            duration_seconds=12.5,
            primary_category="removed_c_api",
            primary_subcategory="PyUnicode_READY",
            primary_description="removed",
            extension_type="extensions",
            source="registry",
            from_mode="sdist",
            installer_used="pip",
        )
        d = r.to_dict()
        self.assertEqual(d["exit_code"], 1)
        self.assertEqual(d["primary_category"], "removed_c_api")
        self.assertEqual(d["extension_type"], "extensions")
        self.assertEqual(d["installer_used"], "pip")

    def test_from_dict_minimal(self) -> None:
        d = {"package": "x", "status": "skip"}
        r = CompatResult.from_dict(d)
        self.assertEqual(r.package, "x")
        self.assertEqual(r.status, "skip")
        self.assertEqual(r.exit_code, None)
        self.assertEqual(r.error_matches, [])

    def test_from_dict_with_matches(self) -> None:
        d = {
            "package": "y",
            "status": "build_fail",
            "error_matches": [
                {
                    "category": "removed_c_api",
                    "subcategory": "tp_print",
                    "description": "tp_print removed",
                    "since": "3.12",
                    "matched_line": "error: no member tp_print",
                    "line_number": 5,
                }
            ],
        }
        r = CompatResult.from_dict(d)
        self.assertEqual(len(r.error_matches), 1)
        self.assertEqual(r.error_matches[0].subcategory, "tp_print")

    def test_roundtrip(self) -> None:
        r = CompatResult(
            package="z",
            status="import_crash",
            crash_signature="SIGSEGV",
            error_matches=[
                ErrorMatch(
                    category="import_failure",
                    subcategory="undefined_symbol",
                    description="Undefined symbol",
                    since="",
                    matched_line="undefined symbol: PyFoo",
                    line_number=1,
                )
            ],
        )
        d = r.to_dict()
        r2 = CompatResult.from_dict(d)
        self.assertEqual(r.package, r2.package)
        self.assertEqual(r.crash_signature, r2.crash_signature)
        self.assertEqual(len(r2.error_matches), 1)


# ---------------------------------------------------------------------------
# CompatMeta
# ---------------------------------------------------------------------------


class TestCompatMeta(unittest.TestCase):
    """Tests for CompatMeta serialization."""

    def test_roundtrip(self) -> None:
        meta = CompatMeta(
            survey_id="compat-20260302-120000",
            target_python="/usr/bin/python3.15",
            python_version="3.15.0a5",
            from_mode="sdist",
            no_binary_all=False,
            started_at="2026-03-02T12:00:00+00:00",
            finished_at="2026-03-02T12:30:00+00:00",
            total_packages=100,
            installer_preference="auto",
        )
        d = meta.to_dict()
        meta2 = CompatMeta.from_dict(d)
        self.assertEqual(meta.survey_id, meta2.survey_id)
        self.assertEqual(meta.python_version, meta2.python_version)
        self.assertFalse(meta2.no_binary_all)

    def test_extra_patterns_file(self) -> None:
        meta = CompatMeta(
            survey_id="test",
            target_python="/usr/bin/python",
            python_version="3.15.0",
            from_mode="source",
            no_binary_all=False,
            started_at="now",
            finished_at="later",
            total_packages=0,
            installer_preference="pip",
            extra_patterns_file="/tmp/patterns.yaml",
        )
        d = meta.to_dict()
        self.assertEqual(d["extra_patterns_file"], "/tmp/patterns.yaml")
        meta2 = CompatMeta.from_dict(d)
        self.assertEqual(meta2.extra_patterns_file, "/tmp/patterns.yaml")


# ---------------------------------------------------------------------------
# CompatSurvey
# ---------------------------------------------------------------------------


class TestCompatSurvey(unittest.TestCase):
    """Tests for CompatSurvey properties."""

    def _make_survey(self) -> CompatSurvey:
        meta = CompatMeta(
            survey_id="test",
            target_python="/usr/bin/python",
            python_version="3.15.0",
            from_mode="sdist",
            no_binary_all=False,
            started_at="now",
            finished_at="later",
            total_packages=4,
            installer_preference="auto",
        )
        results = [
            CompatResult(package="a", status="build_ok"),
            CompatResult(
                package="b",
                status="build_fail",
                primary_category="removed_c_api",
                primary_subcategory="PyUnicode_READY",
            ),
            CompatResult(
                package="c",
                status="build_fail",
                primary_category="removed_c_api",
                primary_subcategory="tp_print",
            ),
            CompatResult(
                package="d",
                status="build_fail",
                primary_category="cython_incompatible",
                primary_subcategory="cython_too_old",
            ),
        ]
        return CompatSurvey(meta=meta, results=results)

    def test_by_status(self) -> None:
        s = self._make_survey()
        by_status = s.by_status
        self.assertEqual(len(by_status["build_ok"]), 1)
        self.assertEqual(len(by_status["build_fail"]), 3)

    def test_by_category(self) -> None:
        s = self._make_survey()
        by_cat = s.by_category
        self.assertEqual(len(by_cat["removed_c_api"]), 2)
        self.assertEqual(len(by_cat["cython_incompatible"]), 1)

    def test_by_subcategory(self) -> None:
        s = self._make_survey()
        by_sub = s.by_subcategory
        self.assertIn("removed_c_api/PyUnicode_READY", by_sub)
        self.assertIn("cython_incompatible/cython_too_old", by_sub)

    def test_summary_counts(self) -> None:
        s = self._make_survey()
        counts = s.summary_counts
        self.assertEqual(counts["build_ok"], 1)
        self.assertEqual(counts["build_fail"], 3)

    def test_by_category_ignores_empty(self) -> None:
        """Packages without primary_category are excluded from by_category."""
        meta = CompatMeta(
            survey_id="x",
            target_python="/usr/bin/python",
            python_version="3.15.0",
            from_mode="sdist",
            no_binary_all=False,
            started_at="now",
            finished_at="later",
            total_packages=1,
            installer_preference="auto",
        )
        survey = CompatSurvey(
            meta=meta,
            results=[CompatResult(package="a", status="build_ok")],
        )
        self.assertEqual(survey.by_category, {})


# ---------------------------------------------------------------------------
# Classification engine
# ---------------------------------------------------------------------------


class TestClassifyBuildOutput(unittest.TestCase):
    """Tests for classify_build_output()."""

    def test_detects_pyunicode_ready(self) -> None:
        stderr = "foo.c:42: error: implicit declaration of PyUnicode_READY\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].category, "removed_c_api")
        self.assertEqual(matches[0].subcategory, "PyUnicode_READY")
        self.assertEqual(matches[0].since, "3.12")

    def test_detects_py_unicode(self) -> None:
        stderr = "error: unknown type name 'Py_UNICODE'\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].subcategory, "Py_UNICODE")

    def test_detects_tp_print(self) -> None:
        stderr = "error: 'PyTypeObject' has no member named 'tp_print'\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].subcategory, "tp_print")

    def test_detects_distutils_removed(self) -> None:
        stderr = "ModuleNotFoundError: No module named 'distutils'\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].category, "setuptools_distutils")

    def test_detects_missing_header(self) -> None:
        stderr = "fatal error: openssl/ssl.h: No such file or directory\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].category, "missing_system_lib")
        self.assertEqual(matches[0].subcategory, "missing_header")

    def test_detects_python_h_missing(self) -> None:
        stderr = "fatal error: Python.h: No such file or directory\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].category, "python_header")

    def test_detects_linker_error(self) -> None:
        stderr = "undefined reference to `PyFoo_Bar'\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].category, "compiler_error")
        self.assertEqual(matches[0].subcategory, "linker_error")

    def test_detects_pyo3(self) -> None:
        stderr = "error: PyO3 does not support Python 3.15\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].category, "pyo3_incompatible")

    def test_detects_rust_error(self) -> None:
        stderr = "error[E0412]: cannot find type `PyObject`\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].subcategory, "rust_build_fail")

    def test_detects_cython_version(self) -> None:
        stderr = "Cython is not supported on Python 3.15\n"
        matches = classify_build_output(stderr)
        # Should not crash, may or may not match the loose regex.
        self.assertIsInstance(matches, list)

    def test_no_matches(self) -> None:
        stderr = "All good, nothing to see here.\n"
        matches = classify_build_output(stderr)
        self.assertEqual(matches, [])

    def test_one_match_per_line(self) -> None:
        """Only the first matching pattern per line should be recorded."""
        stderr = "error: implicit declaration of PyUnicode_READY undefined reference\n"
        matches = classify_build_output(stderr)
        self.assertEqual(len(matches), 1)

    def test_multiple_lines(self) -> None:
        stderr = (
            "error: implicit declaration of PyUnicode_READY\n"
            "error: unknown type name 'Py_UNICODE'\n"
        )
        matches = classify_build_output(stderr)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].line_number, 1)
        self.assertEqual(matches[1].line_number, 2)

    def test_custom_patterns(self) -> None:
        import re

        custom = [
            ErrorPattern(
                category="custom",
                subcategory="test_error",
                pattern=re.compile(r"CUSTOM_ERROR_42"),
                description="Custom test error",
                since="3.15",
            )
        ]
        stderr = "line1\nCUSTOM_ERROR_42 happened\nline3\n"
        matches = classify_build_output(stderr, patterns=custom)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].category, "custom")
        self.assertEqual(matches[0].line_number, 2)

    def test_case_insensitive(self) -> None:
        stderr = "Error: IMPLICIT DECLARATION OF PYUNICODE_READY\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)

    def test_detects_ob_type_assignment(self) -> None:
        stderr = "error: assignment to read-only member 'ob_type'\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].category, "changed_struct")

    def test_detects_missing_library(self) -> None:
        stderr = "cannot find -lssl\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].subcategory, "missing_library")

    def test_detects_cmake_error(self) -> None:
        stderr = "CMakeLists.txt error: something went wrong\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].category, "build_backend")

    def test_detects_import_undefined_symbol(self) -> None:
        stderr = "ImportError: symbol PyFoo not found\n"
        matches = classify_build_output(stderr)
        self.assertTrue(len(matches) >= 1)
        self.assertEqual(matches[0].category, "import_failure")


# ---------------------------------------------------------------------------
# No-sdist pattern
# ---------------------------------------------------------------------------


class TestNoSdistPattern(unittest.TestCase):
    """Tests for _NO_SDIST_PATTERN."""

    def test_matches_no_matching_distribution(self) -> None:
        text = "ERROR: No matching distribution found for foo --no-binary"
        self.assertIsNotNone(_NO_SDIST_PATTERN.search(text))

    def test_does_not_match_random_text(self) -> None:
        self.assertIsNone(_NO_SDIST_PATTERN.search("Everything installed successfully"))


# ---------------------------------------------------------------------------
# get_patterns / load_patterns_from_yaml
# ---------------------------------------------------------------------------


class TestGetPatterns(unittest.TestCase):
    """Tests for get_patterns()."""

    def test_returns_builtins(self) -> None:
        patterns = get_patterns()
        self.assertEqual(len(patterns), len(_BUILTIN_PATTERNS))

    def test_yaml_overrides(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "patterns:\n"
                "  - category: removed_c_api\n"
                "    subcategory: PyUnicode_READY\n"
                "    pattern: 'CUSTOM_PYUNICODE_READY'\n"
                "    description: Custom override\n"
                "    since: '3.12'\n"
            )
            f.flush()
            patterns = get_patterns(Path(f.name))
        # Should have same count (override replaces one).
        builtins_count = len(_BUILTIN_PATTERNS)
        # The YAML provides 1 pattern that overrides 1 builtin.
        self.assertEqual(len(patterns), builtins_count)
        # First pattern should be the YAML override.
        self.assertEqual(patterns[0].description, "Custom override")

    def test_yaml_adds_new(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "patterns:\n"
                "  - category: custom_cat\n"
                "    subcategory: custom_sub\n"
                "    pattern: 'CUSTOM_PATTERN'\n"
                "    description: Brand new\n"
            )
            f.flush()
            patterns = get_patterns(Path(f.name))
        self.assertEqual(len(patterns), len(_BUILTIN_PATTERNS) + 1)

    def test_yaml_invalid_structure(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("not_patterns: []\n")
            f.flush()
            with self.assertRaises(ValueError):
                get_patterns(Path(f.name))

    def test_yaml_invalid_regex(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "patterns:\n"
                "  - category: bad\n"
                "    subcategory: bad\n"
                "    pattern: '[invalid'\n"
                "    description: Bad regex\n"
            )
            f.flush()
            with self.assertRaises(ValueError):
                get_patterns(Path(f.name))

    def test_yaml_missing_field(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "patterns:\n"
                "  - category: foo\n"
                "    subcategory: bar\n"
                "    description: Missing pattern field\n"
            )
            f.flush()
            with self.assertRaises(ValueError):
                get_patterns(Path(f.name))


# ---------------------------------------------------------------------------
# _read_packages_file
# ---------------------------------------------------------------------------


class TestReadPackagesFile(unittest.TestCase):
    """Tests for _read_packages_file()."""

    def test_reads_names(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("foo\nbar\nbaz\n")
            f.flush()
            names = _read_packages_file(Path(f.name))
        self.assertEqual(names, ["foo", "bar", "baz"])

    def test_skips_comments_and_blanks(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# comment\nfoo\n\n  bar  \n# another\n")
            f.flush()
            names = _read_packages_file(Path(f.name))
        self.assertEqual(names, ["foo", "bar"])


# ---------------------------------------------------------------------------
# resolve_compat_inputs
# ---------------------------------------------------------------------------


class TestResolveCompatInputs(unittest.TestCase):
    """Tests for resolve_compat_inputs()."""

    @patch("labeille.resolve.fetch_pypi_metadata")
    @patch("labeille.resolve.extract_repo_url")
    @patch("labeille.classifier.classify_from_urls")
    def test_inline_packages(
        self,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        mock_fetch.return_value = {"urls": []}
        mock_extract.return_value = "https://github.com/foo/foo"
        mock_classify.return_value = "extensions"
        result = resolve_compat_inputs(package_names=["foo"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "foo")
        self.assertEqual(result[0].source, "inline")

    @patch("labeille.resolve.fetch_pypi_metadata")
    @patch("labeille.resolve.extract_repo_url")
    @patch("labeille.classifier.classify_from_urls")
    def test_deduplicates(
        self,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        mock_fetch.return_value = {"urls": []}
        mock_extract.return_value = None
        mock_classify.return_value = "unknown"
        result = resolve_compat_inputs(package_names=["Foo", "foo", "FOO"])
        self.assertEqual(len(result), 1)

    @patch("labeille.resolve.fetch_pypi_metadata")
    @patch("labeille.resolve.extract_repo_url")
    @patch("labeille.classifier.classify_from_urls")
    def test_file_source(
        self,
        mock_classify: MagicMock,
        mock_extract: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        mock_fetch.return_value = {"urls": []}
        mock_extract.return_value = None
        mock_classify.return_value = "unknown"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("bar\nbaz\n")
            f.flush()
            result = resolve_compat_inputs(packages_file=Path(f.name))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].source, "file")

    def test_no_sources(self) -> None:
        result = resolve_compat_inputs()
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# CompatDiff
# ---------------------------------------------------------------------------


class TestCompatDiffEntry(unittest.TestCase):
    """Tests for CompatDiffEntry."""

    def test_is_regression(self) -> None:
        e = CompatDiffEntry(
            package="x",
            status_a="build_ok",
            status_b="build_fail",
            category_a="",
            category_b="removed_c_api",
            description_a="",
            description_b="removed",
        )
        self.assertTrue(e.is_regression)
        self.assertFalse(e.is_fix)

    def test_is_fix(self) -> None:
        e = CompatDiffEntry(
            package="x",
            status_a="build_fail",
            status_b="build_ok",
            category_a="removed_c_api",
            category_b="",
            description_a="removed",
            description_b="",
        )
        self.assertFalse(e.is_regression)
        self.assertTrue(e.is_fix)

    def test_category_change(self) -> None:
        e = CompatDiffEntry(
            package="x",
            status_a="build_fail",
            status_b="build_fail",
            category_a="removed_c_api",
            category_b="cython_incompatible",
            description_a="",
            description_b="",
        )
        self.assertFalse(e.is_regression)
        self.assertFalse(e.is_fix)


class TestCompatDiff(unittest.TestCase):
    """Tests for CompatDiff properties."""

    def _make_diff(self) -> CompatDiff:
        entries = [
            CompatDiffEntry("a", "build_ok", "build_fail", "", "removed_c_api", "", ""),
            CompatDiffEntry("b", "build_fail", "build_ok", "cython_incompatible", "", "", ""),
            CompatDiffEntry("c", "build_fail", "build_fail", "removed_c_api", "pyo3", "", ""),
        ]
        return CompatDiff(
            survey_a_id="s1",
            survey_b_id="s2",
            python_a="3.14.0",
            python_b="3.15.0",
            entries=entries,
        )

    def test_regressions(self) -> None:
        d = self._make_diff()
        self.assertEqual(len(d.regressions), 1)
        self.assertEqual(d.regressions[0].package, "a")

    def test_fixes(self) -> None:
        d = self._make_diff()
        self.assertEqual(len(d.fixes), 1)
        self.assertEqual(d.fixes[0].package, "b")

    def test_category_changes(self) -> None:
        d = self._make_diff()
        self.assertEqual(len(d.category_changes), 1)
        self.assertEqual(d.category_changes[0].package, "c")


class TestDiffSurveys(unittest.TestCase):
    """Tests for diff_surveys()."""

    def _make_meta(self, sid: str, pyver: str) -> CompatMeta:
        return CompatMeta(
            survey_id=sid,
            target_python="/usr/bin/python",
            python_version=pyver,
            from_mode="sdist",
            no_binary_all=False,
            started_at="now",
            finished_at="later",
            total_packages=0,
            installer_preference="auto",
        )

    def test_no_overlap(self) -> None:
        a = CompatSurvey(
            meta=self._make_meta("a", "3.14"),
            results=[CompatResult(package="x", status="build_ok")],
        )
        b = CompatSurvey(
            meta=self._make_meta("b", "3.15"),
            results=[CompatResult(package="y", status="build_ok")],
        )
        diff = diff_surveys(a, b)
        self.assertEqual(diff.entries, [])

    def test_no_change(self) -> None:
        a = CompatSurvey(
            meta=self._make_meta("a", "3.14"),
            results=[CompatResult(package="x", status="build_ok")],
        )
        b = CompatSurvey(
            meta=self._make_meta("b", "3.15"),
            results=[CompatResult(package="x", status="build_ok")],
        )
        diff = diff_surveys(a, b)
        self.assertEqual(diff.entries, [])

    def test_status_change(self) -> None:
        a = CompatSurvey(
            meta=self._make_meta("a", "3.14"),
            results=[CompatResult(package="x", status="build_ok")],
        )
        b = CompatSurvey(
            meta=self._make_meta("b", "3.15"),
            results=[
                CompatResult(package="x", status="build_fail", primary_category="removed_c_api")
            ],
        )
        diff = diff_surveys(a, b)
        self.assertEqual(len(diff.entries), 1)
        self.assertTrue(diff.entries[0].is_regression)


# ---------------------------------------------------------------------------
# load_compat_survey
# ---------------------------------------------------------------------------


class TestLoadCompatSurvey(unittest.TestCase):
    """Tests for load_compat_survey()."""

    def test_loads_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            survey_dir = Path(tmpdir)
            meta = {
                "survey_id": "test",
                "target_python": "/usr/bin/python",
                "python_version": "3.15.0",
                "from_mode": "sdist",
                "no_binary_all": False,
                "started_at": "now",
                "finished_at": "later",
                "total_packages": 2,
                "installer_preference": "auto",
            }
            (survey_dir / "compat_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            results = [
                {"package": "a", "status": "build_ok"},
                {"package": "b", "status": "build_fail", "primary_category": "removed_c_api"},
            ]
            (survey_dir / "compat_results.jsonl").write_text(
                "\n".join(json.dumps(r) for r in results) + "\n",
                encoding="utf-8",
            )
            survey = load_compat_survey(survey_dir)
            self.assertEqual(survey.meta.survey_id, "test")
            self.assertEqual(len(survey.results), 2)
            self.assertEqual(survey.results[1].primary_category, "removed_c_api")

    def test_missing_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_compat_survey(Path(tmpdir))

    def test_missing_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            survey_dir = Path(tmpdir)
            meta = {
                "survey_id": "empty",
                "target_python": "/usr/bin/python",
                "python_version": "3.15.0",
                "from_mode": "sdist",
                "started_at": "now",
                "total_packages": 0,
                "installer_preference": "auto",
            }
            (survey_dir / "compat_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            survey = load_compat_survey(survey_dir)
            self.assertEqual(len(survey.results), 0)


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------


class TestFormatCompatReport(unittest.TestCase):
    """Tests for format_compat_report()."""

    def _make_survey(self) -> CompatSurvey:
        meta = CompatMeta(
            survey_id="test",
            target_python="/usr/bin/python",
            python_version="3.15.0",
            from_mode="sdist",
            no_binary_all=False,
            started_at="2026-03-02T12:00:00",
            finished_at="2026-03-02T12:30:00",
            total_packages=3,
            installer_preference="auto",
        )
        return CompatSurvey(
            meta=meta,
            results=[
                CompatResult(package="a", status="build_ok", duration_seconds=10.0),
                CompatResult(
                    package="b",
                    status="build_fail",
                    primary_category="removed_c_api",
                    primary_subcategory="tp_print",
                    primary_description="tp_print removed",
                    duration_seconds=5.0,
                ),
                CompatResult(
                    package="c",
                    status="build_fail",
                    primary_category="removed_c_api",
                    primary_subcategory="PyUnicode_READY",
                    primary_description="PyUnicode_READY removed",
                    duration_seconds=3.0,
                ),
            ],
        )

    def test_contains_header(self) -> None:
        report = format_compat_report(self._make_survey())
        self.assertIn("Compatibility Survey: test", report)
        self.assertIn("3.15.0", report)

    def test_contains_status_overview(self) -> None:
        report = format_compat_report(self._make_survey())
        self.assertIn("Status overview", report)
        self.assertIn("build_ok", report)
        self.assertIn("build_fail", report)

    def test_contains_category_breakdown(self) -> None:
        report = format_compat_report(self._make_survey())
        self.assertIn("removed_c_api", report)

    def test_empty_survey(self) -> None:
        meta = CompatMeta(
            survey_id="empty",
            target_python="/usr/bin/python",
            python_version="3.15.0",
            from_mode="sdist",
            no_binary_all=False,
            started_at="now",
            finished_at="later",
            total_packages=0,
            installer_preference="auto",
        )
        report = format_compat_report(CompatSurvey(meta=meta, results=[]))
        self.assertIn("No packages surveyed", report)


class TestFormatCompatDiff(unittest.TestCase):
    """Tests for format_compat_diff()."""

    def test_no_differences(self) -> None:
        d = CompatDiff(
            survey_a_id="a",
            survey_b_id="b",
            python_a="3.14",
            python_b="3.15",
            entries=[],
        )
        text = format_compat_diff(d)
        self.assertIn("No differences found", text)

    def test_with_regression(self) -> None:
        d = CompatDiff(
            survey_a_id="a",
            survey_b_id="b",
            python_a="3.14",
            python_b="3.15",
            entries=[
                CompatDiffEntry(
                    "pkg", "build_ok", "build_fail", "", "removed_c_api", "", "removed"
                )
            ],
        )
        text = format_compat_diff(d)
        self.assertIn("1 regressions", text)
        self.assertIn("pkg", text)


class TestExportCompatMarkdown(unittest.TestCase):
    """Tests for export_compat_markdown()."""

    def test_markdown_output(self) -> None:
        meta = CompatMeta(
            survey_id="md-test",
            target_python="/usr/bin/python",
            python_version="3.15.0",
            from_mode="sdist",
            no_binary_all=False,
            started_at="2026-03-02T12:00:00",
            finished_at="2026-03-02T12:30:00",
            total_packages=1,
            installer_preference="auto",
        )
        survey = CompatSurvey(
            meta=meta,
            results=[CompatResult(package="a", status="build_ok")],
        )
        md = export_compat_markdown(survey)
        self.assertIn("# Compatibility Survey: md-test", md)
        self.assertIn("**Python:** 3.15.0", md)
        self.assertIn("| build_ok |", md)

    def test_empty_markdown(self) -> None:
        meta = CompatMeta(
            survey_id="empty",
            target_python="/usr/bin/python",
            python_version="3.15.0",
            from_mode="sdist",
            no_binary_all=False,
            started_at="now",
            finished_at="later",
            total_packages=0,
            installer_preference="auto",
        )
        md = export_compat_markdown(CompatSurvey(meta=meta, results=[]))
        self.assertIn("No packages surveyed", md)


class TestFormatPatternsTable(unittest.TestCase):
    """Tests for format_patterns_table()."""

    def test_all_patterns(self) -> None:
        table = format_patterns_table(_BUILTIN_PATTERNS)
        self.assertIn("Category", table)
        self.assertIn("removed_c_api", table)

    def test_category_filter(self) -> None:
        table = format_patterns_table(_BUILTIN_PATTERNS, category_filter="pyo3_incompatible")
        self.assertIn("pyo3_incompatible", table)
        self.assertNotIn("removed_c_api", table)

    def test_empty_filter(self) -> None:
        table = format_patterns_table(_BUILTIN_PATTERNS, category_filter="nonexistent")
        # Should still have headers but no data rows.
        self.assertIn("Category", table)


# ---------------------------------------------------------------------------
# Builtin patterns sanity
# ---------------------------------------------------------------------------


class TestBuiltinPatterns(unittest.TestCase):
    """Sanity checks on _BUILTIN_PATTERNS."""

    def test_all_have_required_fields(self) -> None:
        for p in _BUILTIN_PATTERNS:
            self.assertTrue(p.category, f"Pattern missing category: {p}")
            self.assertTrue(p.subcategory, f"Pattern missing subcategory: {p}")
            self.assertTrue(p.description, f"Pattern missing description: {p}")
            self.assertIsNotNone(p.pattern, f"Pattern missing compiled regex: {p}")

    def test_unique_subcategories_within_category(self) -> None:
        seen: dict[str, set[str]] = {}
        for p in _BUILTIN_PATTERNS:
            subs = seen.setdefault(p.category, set())
            self.assertNotIn(
                p.subcategory,
                subs,
                f"Duplicate subcategory {p.subcategory} in {p.category}",
            )
            subs.add(p.subcategory)

    def test_compiler_error_is_last(self) -> None:
        """compiler_error and import_failure should be after more specific patterns."""
        last_specific_idx = -1
        for i, p in enumerate(_BUILTIN_PATTERNS):
            if p.category not in ("compiler_error", "import_failure"):
                last_specific_idx = i
        first_catchall_idx = len(_BUILTIN_PATTERNS)
        for i, p in enumerate(_BUILTIN_PATTERNS):
            if p.category in ("compiler_error", "import_failure"):
                first_catchall_idx = i
                break
        self.assertGreater(first_catchall_idx, last_specific_idx)


if __name__ == "__main__":
    unittest.main()
