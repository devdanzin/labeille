"""Tests for git operations and package spec parsing in labeille.repo_ops."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from labeille.repo_ops import (
    checkout_revision,
    clone_repo,
    parse_package_specs,
    parse_repo_overrides,
    pull_repo,
)


class TestCloneRepo(unittest.TestCase):
    """Tests for clone_repo()."""

    @patch("labeille.repo_ops.subprocess.run")
    def test_full_clone_default(self, mock_run: MagicMock) -> None:
        """Default clone (clone_depth=None) produces a full clone with no --depth flag."""
        rev_proc = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        clone_proc = MagicMock(returncode=0, stderr="")
        mock_run.side_effect = [clone_proc, rev_proc]

        result = clone_repo("https://github.com/user/repo", Path("/tmp/dest"))

        self.assertEqual(result, "abc123")
        clone_call = mock_run.call_args_list[0]
        cmd = clone_call[0][0]
        self.assertNotIn("--depth=1", cmd)
        for arg in cmd:
            self.assertFalse(arg.startswith("--depth="), f"Unexpected depth flag: {arg}")

    @patch("labeille.repo_ops.subprocess.run")
    def test_deep_clone_fetches_tags(self, mock_run: MagicMock) -> None:
        """When clone_depth > 1, tags are fetched after clone."""
        clone_proc = MagicMock(returncode=0, stderr="")
        fetch_proc = MagicMock(returncode=0, stderr="")
        rev_proc = MagicMock(returncode=0, stdout="def456\n", stderr="")
        mock_run.side_effect = [clone_proc, fetch_proc, rev_proc]

        result = clone_repo("https://github.com/user/repo", Path("/tmp/dest"), clone_depth=100)

        self.assertEqual(result, "def456")
        self.assertEqual(mock_run.call_count, 3)
        # Second call should be git fetch --tags
        fetch_call = mock_run.call_args_list[1]
        self.assertEqual(fetch_call[0][0], ["git", "fetch", "--tags"])

    @patch("labeille.repo_ops.subprocess.run")
    def test_clone_depth_1_no_tag_fetch(self, mock_run: MagicMock) -> None:
        """clone_depth=1 (explicit) does not fetch tags."""
        clone_proc = MagicMock(returncode=0, stderr="")
        rev_proc = MagicMock(returncode=0, stdout="aaa\n", stderr="")
        mock_run.side_effect = [clone_proc, rev_proc]

        clone_repo("https://github.com/user/repo", Path("/tmp/dest"), clone_depth=1)

        # Only clone + rev-parse, no fetch --tags
        self.assertEqual(mock_run.call_count, 2)

    @patch("labeille.repo_ops.subprocess.run")
    def test_rev_parse_failure_returns_none(self, mock_run: MagicMock) -> None:
        """Returns None when rev-parse fails."""
        clone_proc = MagicMock(returncode=0, stderr="")
        rev_proc = MagicMock(returncode=128, stdout="", stderr="fatal")
        mock_run.side_effect = [clone_proc, rev_proc]

        result = clone_repo("https://github.com/user/repo", Path("/tmp/dest"))

        self.assertIsNone(result)

    @patch("labeille.repo_ops.subprocess.run")
    def test_clone_failure_raises(self, mock_run: MagicMock) -> None:
        """CalledProcessError propagates from failed clone."""
        mock_run.side_effect = subprocess.CalledProcessError(128, "git clone")

        with self.assertRaises(subprocess.CalledProcessError):
            clone_repo("https://bad-url", Path("/tmp/dest"))

    @patch("labeille.repo_ops.subprocess.run")
    def test_tag_fetch_failure_non_fatal(self, mock_run: MagicMock) -> None:
        """Failed tag fetch is non-fatal; clone still returns revision."""
        clone_proc = MagicMock(returncode=0, stderr="")
        fetch_proc = MagicMock(returncode=1, stderr="fetch error")
        rev_proc = MagicMock(returncode=0, stdout="bbb\n", stderr="")
        mock_run.side_effect = [clone_proc, fetch_proc, rev_proc]

        result = clone_repo("https://github.com/user/repo", Path("/tmp/dest"), clone_depth=50)

        self.assertEqual(result, "bbb")


class TestPullRepo(unittest.TestCase):
    """Tests for pull_repo()."""

    @patch("labeille.repo_ops.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        """Successful pull returns HEAD hash."""
        fetch_proc = MagicMock(returncode=0, stderr="")
        reset_proc = MagicMock(returncode=0, stderr="")
        clean_proc = MagicMock(returncode=0, stderr="")
        rev_proc = MagicMock(returncode=0, stdout="abc123\n", stderr="")
        mock_run.side_effect = [fetch_proc, reset_proc, clean_proc, rev_proc]

        result = pull_repo(Path("/tmp/repo"))

        self.assertEqual(result, "abc123")
        self.assertEqual(mock_run.call_count, 4)
        # Verify command sequence
        cmds = [c[0][0] for c in mock_run.call_args_list]
        self.assertEqual(cmds[0], ["git", "fetch", "origin"])
        self.assertEqual(cmds[1], ["git", "reset", "--hard", "FETCH_HEAD"])
        self.assertEqual(cmds[2], ["git", "clean", "-fdx"])
        self.assertEqual(cmds[3], ["git", "rev-parse", "HEAD"])

    @patch("labeille.repo_ops.subprocess.run")
    def test_fetch_failure_raises(self, mock_run: MagicMock) -> None:
        """Fetch failure (check=True) propagates."""
        mock_run.side_effect = subprocess.CalledProcessError(128, "git fetch")

        with self.assertRaises(subprocess.CalledProcessError):
            pull_repo(Path("/tmp/repo"))

    @patch("labeille.repo_ops.subprocess.run")
    def test_reset_failure_non_fatal(self, mock_run: MagicMock) -> None:
        """Reset failure is non-fatal; still cleans and returns revision."""
        fetch_proc = MagicMock(returncode=0, stderr="")
        reset_proc = MagicMock(returncode=1, stderr="reset error")
        clean_proc = MagicMock(returncode=0, stderr="")
        rev_proc = MagicMock(returncode=0, stdout="def456\n", stderr="")
        mock_run.side_effect = [fetch_proc, reset_proc, clean_proc, rev_proc]

        result = pull_repo(Path("/tmp/repo"))

        self.assertEqual(result, "def456")

    @patch("labeille.repo_ops.subprocess.run")
    def test_clean_failure_non_fatal(self, mock_run: MagicMock) -> None:
        """Clean failure is non-fatal; still returns revision."""
        fetch_proc = MagicMock(returncode=0, stderr="")
        reset_proc = MagicMock(returncode=0, stderr="")
        clean_proc = MagicMock(returncode=1, stderr="clean error")
        rev_proc = MagicMock(returncode=0, stdout="ghi789\n", stderr="")
        mock_run.side_effect = [fetch_proc, reset_proc, clean_proc, rev_proc]

        result = pull_repo(Path("/tmp/repo"))

        self.assertEqual(result, "ghi789")

    @patch("labeille.repo_ops.subprocess.run")
    def test_rev_parse_failure_returns_none(self, mock_run: MagicMock) -> None:
        """Returns None when rev-parse fails."""
        fetch_proc = MagicMock(returncode=0, stderr="")
        reset_proc = MagicMock(returncode=0, stderr="")
        clean_proc = MagicMock(returncode=0, stderr="")
        rev_proc = MagicMock(returncode=128, stdout="", stderr="")
        mock_run.side_effect = [fetch_proc, reset_proc, clean_proc, rev_proc]

        result = pull_repo(Path("/tmp/repo"))

        self.assertIsNone(result)


class TestCheckoutRevision(unittest.TestCase):
    """Tests for checkout_revision()."""

    @patch("labeille.repo_ops.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        """Successful checkout returns resolved hash."""
        checkout_proc = MagicMock(returncode=0, stderr="")
        rev_proc = MagicMock(returncode=0, stdout="abc123full\n", stderr="")
        mock_run.side_effect = [checkout_proc, rev_proc]

        result = checkout_revision(Path("/tmp/repo"), "v1.0")

        self.assertEqual(result, "abc123full")
        self.assertEqual(mock_run.call_args_list[0][0][0], ["git", "checkout", "v1.0"])

    @patch("labeille.repo_ops.subprocess.run")
    def test_checkout_failure_returns_none(self, mock_run: MagicMock) -> None:
        """Returns None when checkout fails."""
        checkout_proc = MagicMock(returncode=1, stderr="error: pathspec")
        mock_run.return_value = checkout_proc

        result = checkout_revision(Path("/tmp/repo"), "nonexistent")

        self.assertIsNone(result)
        # Should only call checkout, not rev-parse
        mock_run.assert_called_once()

    @patch("labeille.repo_ops.subprocess.run")
    def test_rev_parse_failure_returns_none(self, mock_run: MagicMock) -> None:
        """Returns None when rev-parse fails after successful checkout."""
        checkout_proc = MagicMock(returncode=0, stderr="")
        rev_proc = MagicMock(returncode=128, stdout="", stderr="")
        mock_run.side_effect = [checkout_proc, rev_proc]

        result = checkout_revision(Path("/tmp/repo"), "HEAD~5")

        self.assertIsNone(result)

    @patch("labeille.repo_ops.subprocess.run")
    def test_uses_correct_cwd(self, mock_run: MagicMock) -> None:
        """Passes repo_dir as cwd to subprocess."""
        checkout_proc = MagicMock(returncode=0, stderr="")
        rev_proc = MagicMock(returncode=0, stdout="aaa\n", stderr="")
        mock_run.side_effect = [checkout_proc, rev_proc]

        checkout_revision(Path("/my/repo"), "abc123")

        for c in mock_run.call_args_list:
            self.assertEqual(c[1]["cwd"], "/my/repo")


class TestParsePackageSpecs(unittest.TestCase):
    """Tests for parse_package_specs()."""

    def test_simple_names(self) -> None:
        names, revs = parse_package_specs("requests,click,flask")
        self.assertEqual(names, ["requests", "click", "flask"])
        self.assertEqual(revs, {})

    def test_with_revisions(self) -> None:
        names, revs = parse_package_specs("requests@abc123,click")
        self.assertEqual(names, ["requests", "click"])
        self.assertEqual(revs, {"requests": "abc123"})

    def test_multiple_revisions(self) -> None:
        names, revs = parse_package_specs("a@rev1,b@rev2,c")
        self.assertEqual(names, ["a", "b", "c"])
        self.assertEqual(revs, {"a": "rev1", "b": "rev2"})

    def test_head_tilde_revision(self) -> None:
        names, revs = parse_package_specs("numpy@HEAD~5")
        self.assertEqual(names, ["numpy"])
        self.assertEqual(revs, {"numpy": "HEAD~5"})

    def test_empty_string(self) -> None:
        names, revs = parse_package_specs("")
        self.assertEqual(names, [])
        self.assertEqual(revs, {})

    def test_whitespace_handling(self) -> None:
        names, revs = parse_package_specs(" requests , click @ abc ")
        self.assertEqual(names, ["requests", "click"])
        self.assertEqual(revs, {"click": "abc"})

    def test_trailing_comma(self) -> None:
        names, revs = parse_package_specs("requests,")
        self.assertEqual(names, ["requests"])
        self.assertEqual(revs, {})

    def test_at_with_empty_revision(self) -> None:
        """name@ with no revision treats it as plain name."""
        names, revs = parse_package_specs("requests@")
        self.assertEqual(names, ["requests"])
        self.assertEqual(revs, {})

    def test_single_package(self) -> None:
        names, revs = parse_package_specs("requests")
        self.assertEqual(names, ["requests"])
        self.assertEqual(revs, {})


class TestParseRepoOverrides(unittest.TestCase):
    """Tests for parse_repo_overrides()."""

    def test_single_override(self) -> None:
        result = parse_repo_overrides(("requests=https://github.com/fork/requests",))
        self.assertEqual(result, {"requests": "https://github.com/fork/requests"})

    def test_multiple_overrides(self) -> None:
        result = parse_repo_overrides(
            (
                "requests=https://github.com/fork/requests",
                "click=https://github.com/fork/click",
            )
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result["requests"], "https://github.com/fork/requests")
        self.assertEqual(result["click"], "https://github.com/fork/click")

    def test_empty_tuple(self) -> None:
        result = parse_repo_overrides(())
        self.assertEqual(result, {})

    def test_no_equals_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_repo_overrides(("invalid-format",))
        self.assertIn("Invalid --repo-override format", str(ctx.exception))

    def test_empty_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_repo_overrides(("=https://example.com",))

    def test_empty_url_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_repo_overrides(("requests=",))

    def test_url_with_equals(self) -> None:
        """URL containing = should not break parsing."""
        result = parse_repo_overrides(("pkg=https://example.com/repo?ref=main",))
        self.assertEqual(result, {"pkg": "https://example.com/repo?ref=main"})


if __name__ == "__main__":
    unittest.main()
