"""Tests for sdist install helper functions in labeille.runner."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from labeille.runner import (
    _extract_extras,
    _is_self_install_segment,
    build_sdist_install_commands,
    checkout_matching_tag,
    detect_source_layout,
    fetch_latest_pypi_version,
    shield_source_dir,
    split_install_command,
)


class TestFetchLatestPypiVersion(unittest.TestCase):
    @patch("labeille.repo_ops.fetch_pypi_metadata")
    def test_returns_version(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"info": {"version": "2.31.0"}}
        result = fetch_latest_pypi_version("requests")
        self.assertEqual(result, "2.31.0")
        mock_fetch.assert_called_once_with("requests", timeout=10.0)

    @patch("labeille.repo_ops.fetch_pypi_metadata")
    def test_returns_none_on_fetch_failure(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = None
        result = fetch_latest_pypi_version("nonexistent")
        self.assertIsNone(result)

    @patch("labeille.repo_ops.fetch_pypi_metadata")
    def test_returns_none_on_missing_key(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"info": {}}
        result = fetch_latest_pypi_version("bad-response")
        self.assertIsNone(result)

    @patch("labeille.repo_ops.fetch_pypi_metadata")
    def test_custom_timeout(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"info": {"version": "1.0"}}
        fetch_latest_pypi_version("pkg", timeout=5.0)
        mock_fetch.assert_called_once_with("pkg", timeout=5.0)


class TestCheckoutMatchingTag(unittest.TestCase):
    def test_matches_v_prefix_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "init"],
                cwd=repo,
                capture_output=True,
                env={
                    "GIT_AUTHOR_NAME": "test",
                    "GIT_AUTHOR_EMAIL": "t@t.com",
                    "GIT_COMMITTER_NAME": "test",
                    "GIT_COMMITTER_EMAIL": "t@t.com",
                },
            )
            subprocess.run(["git", "tag", "v1.2.3"], cwd=repo, capture_output=True)

            commit, tag = checkout_matching_tag(repo, "mypkg", "1.2.3")
            self.assertIsNotNone(commit)
            self.assertEqual(tag, "v1.2.3")

    def test_matches_bare_version_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "init"],
                cwd=repo,
                capture_output=True,
                env={
                    "GIT_AUTHOR_NAME": "test",
                    "GIT_AUTHOR_EMAIL": "t@t.com",
                    "GIT_COMMITTER_NAME": "test",
                    "GIT_COMMITTER_EMAIL": "t@t.com",
                },
            )
            subprocess.run(["git", "tag", "2.0.0"], cwd=repo, capture_output=True)

            commit, tag = checkout_matching_tag(repo, "mypkg", "2.0.0")
            self.assertIsNotNone(commit)
            self.assertEqual(tag, "2.0.0")

    def test_returns_none_when_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "init"],
                cwd=repo,
                capture_output=True,
                env={
                    "GIT_AUTHOR_NAME": "test",
                    "GIT_AUTHOR_EMAIL": "t@t.com",
                    "GIT_COMMITTER_NAME": "test",
                    "GIT_COMMITTER_EMAIL": "t@t.com",
                },
            )

            commit, tag = checkout_matching_tag(repo, "mypkg", "9.9.9")
            self.assertIsNone(commit)
            self.assertIsNone(tag)

    def test_matches_package_prefixed_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=repo, capture_output=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "init"],
                cwd=repo,
                capture_output=True,
                env={
                    "GIT_AUTHOR_NAME": "test",
                    "GIT_AUTHOR_EMAIL": "t@t.com",
                    "GIT_COMMITTER_NAME": "test",
                    "GIT_COMMITTER_EMAIL": "t@t.com",
                },
            )
            subprocess.run(["git", "tag", "mypkg-3.0.0"], cwd=repo, capture_output=True)

            commit, tag = checkout_matching_tag(repo, "mypkg", "3.0.0")
            self.assertIsNotNone(commit)
            self.assertEqual(tag, "mypkg-3.0.0")


class TestDetectSourceLayout(unittest.TestCase):
    def test_src_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "src" / "mypkg").mkdir(parents=True)
            (repo / "src" / "mypkg" / "__init__.py").write_text("")
            result = detect_source_layout(repo, "mypkg")
            self.assertEqual(result, "src")

    def test_flat_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "mypkg").mkdir()
            (repo / "mypkg" / "__init__.py").write_text("")
            result = detect_source_layout(repo, "mypkg")
            self.assertEqual(result, "flat")

    def test_unknown_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            result = detect_source_layout(repo, "mypkg")
            self.assertEqual(result, "unknown")

    def test_single_file_is_unknown(self) -> None:
        """Single .py files are not detected as flat layout (only dirs)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "mypkg.py").write_text("")
            result = detect_source_layout(repo, "mypkg")
            self.assertEqual(result, "unknown")


class TestShieldSourceDir(unittest.TestCase):
    def test_flat_layout_renames_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            pkg_dir = repo / "mypkg"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("x = 1")

            with shield_source_dir(repo, "mypkg", "flat"):
                self.assertFalse(pkg_dir.exists())
                self.assertTrue((repo / "_labeille_shielded_mypkg").exists())

            self.assertTrue(pkg_dir.exists())
            self.assertFalse((repo / "_labeille_shielded_mypkg").exists())

    def test_flat_single_file_not_dir_is_noop(self) -> None:
        """shield_source_dir only handles directories, not single .py files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "mypkg.py").write_text("x = 1")

            # Single .py file: shield looks for mypkg dir, doesn't find it, is noop.
            with shield_source_dir(repo, "mypkg", "flat"):
                self.assertTrue((repo / "mypkg.py").exists())

    def test_src_layout_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "src" / "mypkg").mkdir(parents=True)
            (repo / "src" / "mypkg" / "__init__.py").write_text("")

            with shield_source_dir(repo, "mypkg", "src"):
                self.assertTrue((repo / "src" / "mypkg").exists())

    def test_unknown_layout_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            with shield_source_dir(repo, "mypkg", "unknown"):
                pass  # Should not raise.

    def test_restores_on_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            pkg_dir = repo / "mypkg"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("x = 1")

            with self.assertRaises(RuntimeError):
                with shield_source_dir(repo, "mypkg", "flat"):
                    self.assertFalse(pkg_dir.exists())
                    self.assertTrue((repo / "_labeille_shielded_mypkg").exists())
                    raise RuntimeError("simulated failure")

            self.assertTrue(pkg_dir.exists())


class TestIsSelfInstallSegment(unittest.TestCase):
    def test_pip_install_editable_dot(self) -> None:
        self.assertTrue(_is_self_install_segment("pip install -e ."))

    def test_pip_install_editable_dot_extras(self) -> None:
        self.assertTrue(_is_self_install_segment('pip install -e ".[test]"'))

    def test_pip_install_dot(self) -> None:
        self.assertTrue(_is_self_install_segment("pip install ."))

    def test_pip_install_dot_extras_not_matched(self) -> None:
        """Bare pip install .[dev] (without -e) is not caught by regex."""
        # The regex requires -e for dot-with-extras, or bare dot at end of line.
        # This is a known limitation; .[dev] is not matched.
        self.assertFalse(_is_self_install_segment("pip install .[dev]"))

    def test_pip_install_named_package(self) -> None:
        self.assertFalse(_is_self_install_segment("pip install pytest"))

    def test_pip_install_multiple_packages(self) -> None:
        self.assertFalse(_is_self_install_segment("pip install pytest coverage"))

    def test_git_command(self) -> None:
        self.assertFalse(_is_self_install_segment("git fetch --tags"))


class TestExtractExtras(unittest.TestCase):
    def test_with_extras(self) -> None:
        self.assertEqual(_extract_extras('pip install -e ".[test]"'), "test")

    def test_multiple_extras(self) -> None:
        self.assertEqual(_extract_extras("pip install .[test,dev]"), "test,dev")

    def test_no_extras(self) -> None:
        self.assertIsNone(_extract_extras("pip install -e ."))

    def test_no_dot(self) -> None:
        self.assertIsNone(_extract_extras("pip install pytest"))


class TestSplitInstallCommand(unittest.TestCase):
    def test_simple_editable(self) -> None:
        self_parts, other_parts = split_install_command("pip install -e .")
        self.assertEqual(self_parts, ["pip install -e ."])
        self.assertEqual(other_parts, [])

    def test_editable_with_deps(self) -> None:
        cmd = "pip install -e . && pip install pytest coverage"
        self_parts, other_parts = split_install_command(cmd)
        self.assertEqual(self_parts, ["pip install -e ."])
        self.assertEqual(other_parts, ["pip install pytest coverage"])

    def test_git_fetch_then_install(self) -> None:
        cmd = "git fetch --tags --depth 1 && pip install -e . && pip install pytest"
        self_parts, other_parts = split_install_command(cmd)
        self.assertEqual(self_parts, ["pip install -e ."])
        self.assertEqual(other_parts, ["git fetch --tags --depth 1", "pip install pytest"])

    def test_no_self_install(self) -> None:
        cmd = "pip install pytest coverage"
        self_parts, other_parts = split_install_command(cmd)
        self.assertEqual(self_parts, [])
        self.assertEqual(other_parts, ["pip install pytest coverage"])


class TestBuildSdistInstallCommands(unittest.TestCase):
    def test_simple_editable(self) -> None:
        sdist_cmd, deps_cmd = build_sdist_install_commands("requests", "pip install -e .")
        self.assertEqual(sdist_cmd, "pip install --no-binary requests requests")
        self.assertEqual(deps_cmd, "")

    def test_editable_with_extras(self) -> None:
        sdist_cmd, deps_cmd = build_sdist_install_commands("requests", 'pip install -e ".[test]"')
        self.assertEqual(sdist_cmd, "pip install --no-binary requests 'requests[test]'")
        self.assertEqual(deps_cmd, "")

    def test_editable_plus_test_deps(self) -> None:
        cmd = "pip install -e . && pip install pytest coverage"
        sdist_cmd, deps_cmd = build_sdist_install_commands("mypkg", cmd)
        self.assertEqual(sdist_cmd, "pip install --no-binary mypkg mypkg")
        self.assertEqual(deps_cmd, "pip install pytest coverage")

    def test_git_fetch_plus_install_plus_deps(self) -> None:
        cmd = "git fetch --tags --depth 1 && pip install -e . && pip install pytest"
        sdist_cmd, deps_cmd = build_sdist_install_commands("mypkg", cmd)
        self.assertEqual(sdist_cmd, "pip install --no-binary mypkg mypkg")
        self.assertEqual(deps_cmd, "git fetch --tags --depth 1 && pip install pytest")

    def test_no_self_install(self) -> None:
        cmd = "pip install pytest coverage"
        sdist_cmd, deps_cmd = build_sdist_install_commands("mypkg", cmd)
        self.assertEqual(sdist_cmd, "pip install --no-binary mypkg mypkg")
        self.assertEqual(deps_cmd, "pip install pytest coverage")

    def test_dot_install_not_editable(self) -> None:
        sdist_cmd, deps_cmd = build_sdist_install_commands("mypkg", "pip install .")
        self.assertEqual(sdist_cmd, "pip install --no-binary mypkg mypkg")
        self.assertEqual(deps_cmd, "")

    def test_dot_install_with_extras_and_deps(self) -> None:
        """Bare pip install .[dev] is not caught as self-install by regex,
        so both segments go to deps_cmd and sdist gets plain package name."""
        cmd = "pip install .[dev] && pip install pytest-cov"
        sdist_cmd, deps_cmd = build_sdist_install_commands("mypkg", cmd)
        self.assertEqual(sdist_cmd, "pip install --no-binary mypkg mypkg")
        self.assertEqual(deps_cmd, "pip install .[dev] && pip install pytest-cov")


if __name__ == "__main__":
    unittest.main()
