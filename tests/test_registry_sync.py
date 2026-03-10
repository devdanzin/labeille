"""Tests for the ``labeille registry sync`` command."""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from click.testing import CliRunner

from labeille.registry_cli import registry


class TestSyncCmd(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.target = Path(self.tmpdir.name) / "registry"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @unittest.mock.patch("subprocess.run")
    def test_fresh_clone(self, mock_run: unittest.mock.MagicMock) -> None:
        mock_run.return_value = unittest.mock.Mock(returncode=0)
        result = self.runner.invoke(registry, ["sync", "--registry-dir", str(self.target)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Cloning", result.output)
        # Assert git clone was called with the laruche URL.
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "git")
        self.assertEqual(args[1], "clone")
        self.assertIn("laruche", args[2])
        self.assertEqual(args[3], str(self.target))

    @unittest.mock.patch("subprocess.run")
    def test_existing_repo_pulls(self, mock_run: unittest.mock.MagicMock) -> None:
        # Create a fake git repo.
        self.target.mkdir(parents=True)
        (self.target / ".git").mkdir()
        mock_run.return_value = unittest.mock.Mock(returncode=0)
        result = self.runner.invoke(registry, ["sync", "--registry-dir", str(self.target)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Updating", result.output)
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "git")
        self.assertIn("pull", args)

    def test_non_git_directory_errors(self) -> None:
        self.target.mkdir(parents=True)
        (self.target / "somefile.txt").write_text("data")
        result = self.runner.invoke(registry, ["sync", "--registry-dir", str(self.target)])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("not a git repository", result.output)

    @unittest.mock.patch("subprocess.run")
    def test_clone_failure(self, mock_run: unittest.mock.MagicMock) -> None:
        mock_run.return_value = unittest.mock.Mock(returncode=128, stderr="fatal: error")
        result = self.runner.invoke(registry, ["sync", "--registry-dir", str(self.target)])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Clone failed", result.output)

    @unittest.mock.patch("subprocess.run")
    def test_custom_repo_url(self, mock_run: unittest.mock.MagicMock) -> None:
        mock_run.return_value = unittest.mock.Mock(returncode=0)
        result = self.runner.invoke(
            registry,
            [
                "sync",
                "--registry-dir",
                str(self.target),
                "--repo-url",
                "https://example.com/custom-registry.git",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        args = mock_run.call_args[0][0]
        self.assertEqual(args[2], "https://example.com/custom-registry.git")

    @unittest.mock.patch("subprocess.run")
    def test_pull_failure(self, mock_run: unittest.mock.MagicMock) -> None:
        self.target.mkdir(parents=True)
        (self.target / ".git").mkdir()
        mock_run.return_value = unittest.mock.Mock(returncode=1, stderr="error: cannot pull")
        result = self.runner.invoke(registry, ["sync", "--registry-dir", str(self.target)])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Pull failed", result.output)


if __name__ == "__main__":
    unittest.main()
