"""Tests for labeille.compat_cli."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from labeille.compat_cli import compat


class TestCompatCLIGroup(unittest.TestCase):
    """Tests for the compat CLI group."""

    def test_group_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(compat, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("survey", result.output)
        self.assertIn("show", result.output)
        self.assertIn("diff", result.output)
        self.assertIn("patterns", result.output)


# ---------------------------------------------------------------------------
# compat survey
# ---------------------------------------------------------------------------


class TestSurveyCommand(unittest.TestCase):
    """Tests for compat survey command."""

    def test_requires_target_python(self) -> None:
        runner = CliRunner()
        result = runner.invoke(compat, ["survey"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--target-python", result.output)

    def test_requires_package_source(self) -> None:
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".py") as f:
            result = runner.invoke(compat, ["survey", "--target-python", f.name])
        self.assertNotEqual(result.exit_code, 0)

    def test_workers_must_be_positive(self) -> None:
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".py") as f:
            result = runner.invoke(
                compat,
                [
                    "survey",
                    "--target-python",
                    f.name,
                    "--packages",
                    "foo",
                    "--workers",
                    "0",
                ],
            )
        self.assertNotEqual(result.exit_code, 0)

    @patch("labeille.runner.validate_target_python", return_value="3.15.0a5")
    @patch("labeille.runner.extract_python_minor_version", return_value="3.15")
    @patch("labeille.compat.resolve_compat_inputs", return_value=[])
    def test_no_packages_found(
        self,
        mock_resolve: MagicMock,
        mock_minor: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".py") as f:
            result = runner.invoke(
                compat,
                ["survey", "--target-python", f.name, "--packages", "foo"],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No packages to survey", result.output)

    @patch("labeille.compat.run_compat_survey")
    @patch("labeille.compat.resolve_compat_inputs")
    @patch("labeille.runner.extract_python_minor_version", return_value="3.15")
    @patch("labeille.runner.validate_target_python", return_value="3.15.0a5")
    def test_runs_survey(
        self,
        mock_validate: MagicMock,
        mock_minor: MagicMock,
        mock_resolve: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        from labeille.compat import CompatMeta, CompatPackageInput, CompatResult, CompatSurvey

        mock_resolve.return_value = [CompatPackageInput(name="foo", source="inline")]
        mock_run.return_value = CompatSurvey(
            meta=CompatMeta(
                survey_id="test",
                target_python="/usr/bin/python",
                python_version="3.15.0a5",
                from_mode="sdist",
                no_binary_all=False,
                started_at="now",
                finished_at="later",
                total_packages=1,
                installer_preference="auto",
            ),
            results=[CompatResult(package="foo", status="build_ok")],
        )

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(suffix=".py") as f:
                result = runner.invoke(
                    compat,
                    [
                        "survey",
                        "--target-python",
                        f.name,
                        "--packages",
                        "foo",
                        "--output-dir",
                        tmpdir,
                    ],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Surveying 1 package(s)", result.output)
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# compat show
# ---------------------------------------------------------------------------


class TestShowCommand(unittest.TestCase):
    """Tests for compat show command."""

    def test_missing_survey_dir(self) -> None:
        runner = CliRunner()
        result = runner.invoke(compat, ["show", "/nonexistent/path"])
        self.assertNotEqual(result.exit_code, 0)

    def test_shows_survey(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            survey_dir = Path(tmpdir)
            meta = {
                "survey_id": "show-test",
                "target_python": "/usr/bin/python",
                "python_version": "3.15.0",
                "from_mode": "sdist",
                "no_binary_all": False,
                "started_at": "2026-03-02T12:00:00",
                "finished_at": "2026-03-02T12:30:00",
                "total_packages": 1,
                "installer_preference": "auto",
            }
            (survey_dir / "compat_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (survey_dir / "compat_results.jsonl").write_text(
                json.dumps({"package": "a", "status": "build_ok"}) + "\n",
                encoding="utf-8",
            )
            runner = CliRunner()
            result = runner.invoke(compat, ["show", tmpdir])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("show-test", result.output)
        self.assertIn("build_ok", result.output)

    def test_show_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            survey_dir = Path(tmpdir)
            meta = {
                "survey_id": "md-test",
                "target_python": "/usr/bin/python",
                "python_version": "3.15.0",
                "from_mode": "sdist",
                "no_binary_all": False,
                "started_at": "2026-03-02T12:00:00",
                "finished_at": "2026-03-02T12:30:00",
                "total_packages": 1,
                "installer_preference": "auto",
            }
            (survey_dir / "compat_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (survey_dir / "compat_results.jsonl").write_text(
                json.dumps({"package": "a", "status": "build_ok"}) + "\n",
                encoding="utf-8",
            )
            runner = CliRunner()
            result = runner.invoke(compat, ["show", tmpdir, "--format", "markdown"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("# Compatibility Survey", result.output)

    def test_show_with_status_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            survey_dir = Path(tmpdir)
            meta = {
                "survey_id": "filter-test",
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
            runner = CliRunner()
            result = runner.invoke(compat, ["show", tmpdir, "--status", "build_fail"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Filtered to 1 result(s)", result.output)


# ---------------------------------------------------------------------------
# compat diff
# ---------------------------------------------------------------------------


class TestDiffCommand(unittest.TestCase):
    """Tests for compat diff command."""

    def _create_survey_dir(self, tmpdir: str, sid: str, results: list[dict[str, Any]]) -> str:
        survey_dir = Path(tmpdir) / sid
        survey_dir.mkdir()
        meta = {
            "survey_id": sid,
            "target_python": "/usr/bin/python",
            "python_version": "3.15.0",
            "from_mode": "sdist",
            "no_binary_all": False,
            "started_at": "now",
            "finished_at": "later",
            "total_packages": len(results),
            "installer_preference": "auto",
        }
        (survey_dir / "compat_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (survey_dir / "compat_results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in results) + "\n",
            encoding="utf-8",
        )
        return str(survey_dir)

    def test_diff_no_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            a = self._create_survey_dir(tmpdir, "a", [{"package": "x", "status": "build_ok"}])
            b = self._create_survey_dir(tmpdir, "b", [{"package": "x", "status": "build_ok"}])
            runner = CliRunner()
            result = runner.invoke(compat, ["diff", a, b])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("No differences found", result.output)

    def test_diff_with_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            a = self._create_survey_dir(tmpdir, "a", [{"package": "x", "status": "build_ok"}])
            b = self._create_survey_dir(
                tmpdir,
                "b",
                [{"package": "x", "status": "build_fail", "primary_category": "removed_c_api"}],
            )
            runner = CliRunner()
            result = runner.invoke(compat, ["diff", a, b])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("1 regressions", result.output)


# ---------------------------------------------------------------------------
# compat patterns
# ---------------------------------------------------------------------------


class TestPatternsCommand(unittest.TestCase):
    """Tests for compat patterns command."""

    def test_lists_patterns(self) -> None:
        runner = CliRunner()
        result = runner.invoke(compat, ["patterns"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Category", result.output)
        self.assertIn("removed_c_api", result.output)

    def test_category_filter(self) -> None:
        runner = CliRunner()
        result = runner.invoke(compat, ["patterns", "--category", "pyo3_incompatible"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("pyo3_incompatible", result.output)
        self.assertNotIn("removed_c_api", result.output)

    def test_extra_patterns(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "patterns:\n"
                "  - category: custom\n"
                "    subcategory: my_pattern\n"
                "    pattern: 'MY_CUSTOM'\n"
                "    description: Custom test\n"
            )
            f.flush()
            runner = CliRunner()
            result = runner.invoke(compat, ["patterns", "--extra-patterns", f.name])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("custom", result.output)

    def test_invalid_extra_patterns(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("not_patterns: []\n")
            f.flush()
            runner = CliRunner()
            result = runner.invoke(compat, ["patterns", "--extra-patterns", f.name])
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
