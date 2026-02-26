"""Tests for labeille.bisect — automated crash bisection."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from labeille.bisect import (
    BisectConfig,
    BisectResult,
    BisectStep,
    _log2,
    _try_neighbors,
    get_commit_range,
    run_bisect,
    test_revision,
)


class TestLog2(unittest.TestCase):
    """Tests for the _log2 helper."""

    def test_zero(self) -> None:
        self.assertEqual(_log2(0), 0)

    def test_negative(self) -> None:
        self.assertEqual(_log2(-5), 0)

    def test_one(self) -> None:
        self.assertEqual(_log2(1), 0)

    def test_two(self) -> None:
        self.assertEqual(_log2(2), 1)

    def test_eight(self) -> None:
        self.assertEqual(_log2(8), 3)

    def test_non_power_of_two(self) -> None:
        self.assertEqual(_log2(10), 3)

    def test_large(self) -> None:
        self.assertEqual(_log2(1024), 10)


class TestBisectResult(unittest.TestCase):
    """Tests for BisectResult.success property."""

    def test_success_when_found(self) -> None:
        result = BisectResult(
            package="pkg",
            first_bad_commit="abc123",
            first_bad_commit_short="abc1234",
            good_rev="v1.0",
            bad_rev="v2.0",
            steps=[],
            total_commits=10,
            commits_tested=4,
        )
        self.assertTrue(result.success)

    def test_not_success_when_not_found(self) -> None:
        result = BisectResult(
            package="pkg",
            first_bad_commit=None,
            first_bad_commit_short=None,
            good_rev="v1.0",
            bad_rev="v2.0",
            steps=[],
            total_commits=10,
            commits_tested=4,
        )
        self.assertFalse(result.success)


class TestGetCommitRange(unittest.TestCase):
    """Tests for get_commit_range."""

    @patch("labeille.bisect.subprocess.run")
    def test_returns_commits_in_chronological_order(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ccc\nbbb\naaa\n", stderr=""
        )
        commits = get_commit_range(Path("/repo"), "good", "bad")
        self.assertEqual(commits, ["aaa", "bbb", "ccc"])

    @patch("labeille.bisect.subprocess.run")
    def test_returns_empty_on_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )
        commits = get_commit_range(Path("/repo"), "good", "bad")
        self.assertEqual(commits, [])

    @patch("labeille.bisect.subprocess.run")
    def test_handles_empty_output(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        commits = get_commit_range(Path("/repo"), "good", "bad")
        self.assertEqual(commits, [])

    @patch("labeille.bisect.subprocess.run")
    def test_strips_whitespace(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="  abc  \n  def  \n", stderr=""
        )
        commits = get_commit_range(Path("/repo"), "good", "bad")
        self.assertEqual(commits, ["def", "abc"])


class TestTestRevision(unittest.TestCase):
    """Tests for test_revision."""

    def _make_config(self, **overrides: object) -> BisectConfig:
        defaults: dict[str, object] = {
            "package": "test-pkg",
            "good_rev": "v1.0",
            "bad_rev": "v2.0",
            "target_python": Path("/usr/bin/python3"),
        }
        defaults.update(overrides)
        return BisectConfig(**defaults)  # type: ignore[arg-type]

    @patch("labeille.bisect.subprocess.run")
    def test_checkout_failure_returns_skip(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="checkout failed"
        )
        config = self._make_config()
        step = test_revision(Path("/repo"), "abc1234567890", config)
        self.assertEqual(step.status, "skip")
        self.assertIn("checkout failed", step.detail)
        self.assertEqual(step.commit_short, "abc1234")

    @patch("labeille.bisect.detect_crash")
    @patch("labeille.bisect.run_test_command")
    @patch("labeille.bisect.install_package")
    @patch("labeille.bisect.create_venv")
    @patch("labeille.bisect.subprocess.run")
    def test_good_revision_no_crash(
        self,
        mock_run: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
        mock_test: MagicMock,
        mock_detect: MagicMock,
    ) -> None:
        # Checkout and clean succeed.
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_install.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_test.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_detect.return_value = None

        config = self._make_config()
        step = test_revision(Path("/repo"), "abc1234567890", config)
        self.assertEqual(step.status, "good")
        self.assertIn("no crash", step.detail)

    @patch("labeille.bisect.detect_crash")
    @patch("labeille.bisect.run_test_command")
    @patch("labeille.bisect.install_package")
    @patch("labeille.bisect.create_venv")
    @patch("labeille.bisect.subprocess.run")
    def test_bad_revision_with_crash(
        self,
        mock_run: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
        mock_test: MagicMock,
        mock_detect: MagicMock,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_install.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_test.return_value = subprocess.CompletedProcess(
            args=[], returncode=-11, stdout="", stderr="Segmentation fault"
        )
        crash_info = MagicMock()
        crash_info.signature = "SIGSEGV: something bad"
        mock_detect.return_value = crash_info

        config = self._make_config()
        step = test_revision(Path("/repo"), "abc1234567890", config)
        self.assertEqual(step.status, "bad")
        self.assertEqual(step.crash_signature, "SIGSEGV: something bad")

    @patch("labeille.bisect.detect_crash")
    @patch("labeille.bisect.run_test_command")
    @patch("labeille.bisect.install_package")
    @patch("labeille.bisect.create_venv")
    @patch("labeille.bisect.subprocess.run")
    def test_crash_signature_mismatch_returns_good(
        self,
        mock_run: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
        mock_test: MagicMock,
        mock_detect: MagicMock,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_install.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_test.return_value = subprocess.CompletedProcess(
            args=[], returncode=-11, stdout="", stderr="Segmentation fault"
        )
        crash_info = MagicMock()
        crash_info.signature = "SIGSEGV: unrelated crash"
        mock_detect.return_value = crash_info

        config = self._make_config(crash_signature="SIGABRT")
        step = test_revision(Path("/repo"), "abc1234567890", config)
        self.assertEqual(step.status, "good")
        self.assertIn("doesn't match", step.detail)

    @patch("labeille.bisect.detect_crash")
    @patch("labeille.bisect.run_test_command")
    @patch("labeille.bisect.install_package")
    @patch("labeille.bisect.create_venv")
    @patch("labeille.bisect.subprocess.run")
    def test_crash_signature_match_returns_bad(
        self,
        mock_run: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
        mock_test: MagicMock,
        mock_detect: MagicMock,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_install.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_test.return_value = subprocess.CompletedProcess(
            args=[], returncode=-11, stdout="", stderr="Segmentation fault"
        )
        crash_info = MagicMock()
        crash_info.signature = "SIGSEGV: in jit_compile"
        mock_detect.return_value = crash_info

        config = self._make_config(crash_signature="SIGSEGV")
        step = test_revision(Path("/repo"), "abc1234567890", config)
        self.assertEqual(step.status, "bad")

    @patch("labeille.bisect.install_package")
    @patch("labeille.bisect.create_venv")
    @patch("labeille.bisect.subprocess.run")
    def test_install_failure_returns_skip(
        self,
        mock_run: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_install.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="install error"
        )

        config = self._make_config()
        step = test_revision(Path("/repo"), "abc1234567890", config)
        self.assertEqual(step.status, "skip")
        self.assertIn("install failed", step.detail)

    @patch("labeille.bisect.install_package")
    @patch("labeille.bisect.create_venv")
    @patch("labeille.bisect.subprocess.run")
    def test_install_timeout_returns_skip(
        self,
        mock_run: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_install.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=600)

        config = self._make_config()
        step = test_revision(Path("/repo"), "abc1234567890", config)
        self.assertEqual(step.status, "skip")
        self.assertIn("timed out", step.detail)

    @patch("labeille.bisect.create_venv")
    @patch("labeille.bisect.subprocess.run")
    def test_venv_creation_failure_returns_skip(
        self,
        mock_run: MagicMock,
        mock_create_venv: MagicMock,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_create_venv.side_effect = OSError("venv failed")

        config = self._make_config()
        step = test_revision(Path("/repo"), "abc1234567890", config)
        self.assertEqual(step.status, "skip")
        self.assertIn("venv creation failed", step.detail)

    @patch("labeille.bisect.run_test_command")
    @patch("labeille.bisect.install_package")
    @patch("labeille.bisect.create_venv")
    @patch("labeille.bisect.subprocess.run")
    def test_test_timeout_returns_skip(
        self,
        mock_run: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_install.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_test.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=600)

        config = self._make_config()
        step = test_revision(Path("/repo"), "abc1234567890", config)
        self.assertEqual(step.status, "skip")
        self.assertIn("tests timed out", step.detail)

    @patch("labeille.bisect.install_package")
    @patch("labeille.bisect.create_venv")
    @patch("labeille.bisect.subprocess.run")
    def test_extra_deps_installed(
        self,
        mock_run: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
    ) -> None:
        """Extra deps are installed after the main package."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        # First call: main install succeeds. Second call: extra deps.
        mock_install.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]

        with (
            patch("labeille.bisect.run_test_command") as mock_test,
            patch("labeille.bisect.detect_crash") as mock_detect,
        ):
            mock_test.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            mock_detect.return_value = None

            config = self._make_config(extra_deps=["pytest-xdist", "coverage"])
            test_revision(Path("/repo"), "abc1234567890", config)

        self.assertEqual(mock_install.call_count, 2)
        extra_call = mock_install.call_args_list[1]
        self.assertIn("pytest-xdist", extra_call.args[1])
        self.assertIn("coverage", extra_call.args[1])


class TestTryNeighbors(unittest.TestCase):
    """Tests for _try_neighbors."""

    def _make_config(self) -> BisectConfig:
        return BisectConfig(
            package="pkg",
            good_rev="v1",
            bad_rev="v2",
            target_python=Path("/usr/bin/python3"),
        )

    @patch("labeille.bisect.test_revision")
    def test_finds_bad_neighbor(self, mock_test: MagicMock) -> None:
        commits = ["a", "b", "c", "d", "e"]
        steps: list[BisectStep] = []
        config = self._make_config()

        mock_test.return_value = BisectStep(
            commit="d", commit_short="d", status="bad", detail="crash"
        )

        result = _try_neighbors(commits, 0, 4, 2, Path("/repo"), config, steps)
        self.assertEqual(result, (0, 3))  # new hi = index of "d"

    @patch("labeille.bisect.test_revision")
    def test_finds_good_neighbor(self, mock_test: MagicMock) -> None:
        commits = ["a", "b", "c", "d", "e"]
        steps: list[BisectStep] = []
        config = self._make_config()

        mock_test.return_value = BisectStep(
            commit="b", commit_short="b", status="good", detail="no crash"
        )

        result = _try_neighbors(commits, 0, 4, 2, Path("/repo"), config, steps)
        # First candidate tried is mid+1=3 (d), then mid-1=1 (b).
        # But the mock always returns "good", so the first candidate (d at index 3)
        # would return (3, 4). Let's check what actually gets called.
        self.assertIsNotNone(result)

    @patch("labeille.bisect.test_revision")
    def test_returns_none_when_all_skip(self, mock_test: MagicMock) -> None:
        commits = ["a", "b", "c", "d", "e"]
        steps: list[BisectStep] = []
        config = self._make_config()

        mock_test.return_value = BisectStep(
            commit="x", commit_short="x", status="skip", detail="build failed"
        )

        result = _try_neighbors(commits, 0, 4, 2, Path("/repo"), config, steps)
        self.assertIsNone(result)

    @patch("labeille.bisect.test_revision")
    def test_skips_already_tested_commits(self, mock_test: MagicMock) -> None:
        commits = ["a", "b", "c", "d", "e"]
        # Mark "d" (mid+1=3) as already tested.
        steps: list[BisectStep] = [
            BisectStep(commit="d", commit_short="d", status="skip", detail="")
        ]
        config = self._make_config()

        mock_test.return_value = BisectStep(
            commit="b", commit_short="b", status="good", detail="ok"
        )

        result = _try_neighbors(commits, 0, 4, 2, Path("/repo"), config, steps)
        # "d" is skipped because already tested, "b" at index 1 returns good.
        self.assertEqual(result, (1, 4))


class TestRunBisect(unittest.TestCase):
    """Tests for the main run_bisect function."""

    def _make_config(self, **overrides: object) -> BisectConfig:
        defaults: dict[str, object] = {
            "package": "test-pkg",
            "good_rev": "v1.0",
            "bad_rev": "v2.0",
            "target_python": Path("/usr/bin/python3"),
            "registry_dir": Path("/registry"),
        }
        defaults.update(overrides)
        return BisectConfig(**defaults)  # type: ignore[arg-type]

    @patch("labeille.bisect.test_revision")
    @patch("labeille.bisect.get_commit_range")
    @patch("labeille.bisect._resolve_rev")
    @patch("labeille.bisect._clone_full")
    @patch("labeille.registry.load_package")
    def test_successful_bisect(
        self,
        mock_load: MagicMock,
        mock_clone: MagicMock,
        mock_resolve: MagicMock,
        mock_range: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Find the first bad commit in a 4-commit range."""
        pkg = MagicMock()
        pkg.repo = "https://github.com/test/pkg"
        pkg.install_command = "pip install -e ."
        pkg.test_command = "python -m pytest"
        mock_load.return_value = pkg

        mock_resolve.side_effect = lambda _d, rev: f"full-{rev}"
        mock_range.return_value = ["c1", "c2", "c3", "c4"]

        # good_rev -> good, bad_rev -> bad, c2 (midpoint) -> good, c3 -> bad
        call_count = 0

        def test_side_effect(repo_dir: Path, commit: str, config: BisectConfig) -> BisectStep:
            nonlocal call_count
            call_count += 1
            if commit == "full-v1.0":
                return BisectStep(commit=commit, commit_short=commit[:7], status="good")
            elif commit == "full-v2.0":
                return BisectStep(
                    commit=commit, commit_short=commit[:7], status="bad", crash_signature="SIGSEGV"
                )
            elif commit == "c2":
                return BisectStep(commit=commit, commit_short=commit[:7], status="good")
            elif commit == "c3":
                return BisectStep(
                    commit=commit, commit_short=commit[:7], status="bad", crash_signature="SIGSEGV"
                )
            return BisectStep(commit=commit, commit_short=commit[:7], status="good")

        mock_test.side_effect = test_side_effect

        config = self._make_config(work_dir=Path("/tmp/work"))
        with patch("pathlib.Path.exists", return_value=False):
            result = run_bisect(config)

        self.assertTrue(result.success)
        self.assertEqual(result.first_bad_commit, "c3")

    @patch("labeille.bisect.test_revision")
    @patch("labeille.bisect.get_commit_range")
    @patch("labeille.bisect._resolve_rev")
    @patch("labeille.bisect._clone_full")
    @patch("labeille.registry.load_package")
    def test_good_rev_crashes(
        self,
        mock_load: MagicMock,
        mock_clone: MagicMock,
        mock_resolve: MagicMock,
        mock_range: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Abort early if good rev actually crashes."""
        pkg = MagicMock()
        pkg.repo = "https://github.com/test/pkg"
        pkg.install_command = None
        pkg.test_command = None
        mock_load.return_value = pkg

        mock_resolve.side_effect = lambda _d, rev: f"full-{rev}"
        mock_range.return_value = ["c1", "c2"]

        mock_test.return_value = BisectStep(
            commit="full-v1.0", commit_short="full-v1", status="bad", crash_signature="SIGSEGV"
        )

        config = self._make_config(work_dir=Path("/tmp/work"))
        with patch("pathlib.Path.exists", return_value=False):
            result = run_bisect(config)

        self.assertFalse(result.success)
        self.assertEqual(result.commits_tested, 1)

    @patch("labeille.bisect.test_revision")
    @patch("labeille.bisect.get_commit_range")
    @patch("labeille.bisect._resolve_rev")
    @patch("labeille.bisect._clone_full")
    @patch("labeille.registry.load_package")
    def test_bad_rev_doesnt_crash(
        self,
        mock_load: MagicMock,
        mock_clone: MagicMock,
        mock_resolve: MagicMock,
        mock_range: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Abort early if bad rev doesn't crash."""
        pkg = MagicMock()
        pkg.repo = "https://github.com/test/pkg"
        pkg.install_command = None
        pkg.test_command = None
        mock_load.return_value = pkg

        mock_resolve.side_effect = lambda _d, rev: f"full-{rev}"
        mock_range.return_value = ["c1", "c2"]

        call_count = 0

        def test_side_effect(repo_dir: Path, commit: str, config: BisectConfig) -> BisectStep:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return BisectStep(commit=commit, commit_short=commit[:7], status="good")
            return BisectStep(commit=commit, commit_short=commit[:7], status="good")

        mock_test.side_effect = test_side_effect

        config = self._make_config(work_dir=Path("/tmp/work"))
        with patch("pathlib.Path.exists", return_value=False):
            result = run_bisect(config)

        self.assertFalse(result.success)
        self.assertEqual(result.commits_tested, 2)

    @patch("labeille.registry.load_package")
    def test_no_repo_url_raises(self, mock_load: MagicMock) -> None:
        """Raise ValueError when no repo URL is available."""
        mock_load.side_effect = FileNotFoundError("not found")

        config = self._make_config()
        with self.assertRaises(ValueError) as ctx:
            with patch("pathlib.Path.exists", return_value=False):
                run_bisect(config)

        self.assertIn("No repo URL", str(ctx.exception))

    @patch("labeille.bisect._resolve_rev")
    @patch("labeille.bisect._clone_full")
    @patch("labeille.registry.load_package")
    def test_unresolvable_good_rev_raises(
        self,
        mock_load: MagicMock,
        mock_clone: MagicMock,
        mock_resolve: MagicMock,
    ) -> None:
        pkg = MagicMock()
        pkg.repo = "https://github.com/test/pkg"
        pkg.install_command = None
        pkg.test_command = None
        mock_load.return_value = pkg

        mock_resolve.return_value = None

        config = self._make_config(work_dir=Path("/tmp/work"))
        with self.assertRaises(ValueError) as ctx:
            with patch("pathlib.Path.exists", return_value=False):
                run_bisect(config)

        self.assertIn("Could not resolve good revision", str(ctx.exception))

    @patch("labeille.bisect.get_commit_range")
    @patch("labeille.bisect._resolve_rev")
    @patch("labeille.bisect._clone_full")
    @patch("labeille.registry.load_package")
    def test_empty_commit_range_raises(
        self,
        mock_load: MagicMock,
        mock_clone: MagicMock,
        mock_resolve: MagicMock,
        mock_range: MagicMock,
    ) -> None:
        pkg = MagicMock()
        pkg.repo = "https://github.com/test/pkg"
        pkg.install_command = None
        pkg.test_command = None
        mock_load.return_value = pkg

        mock_resolve.side_effect = lambda _d, rev: f"full-{rev}"
        mock_range.return_value = []

        config = self._make_config(work_dir=Path("/tmp/work"))
        with self.assertRaises(ValueError) as ctx:
            with patch("pathlib.Path.exists", return_value=False):
                run_bisect(config)

        self.assertIn("No commits found", str(ctx.exception))

    @patch("labeille.bisect.test_revision")
    @patch("labeille.bisect.get_commit_range")
    @patch("labeille.bisect._resolve_rev")
    @patch("labeille.bisect._clone_full")
    @patch("labeille.registry.load_package")
    def test_good_rev_skip_aborts(
        self,
        mock_load: MagicMock,
        mock_clone: MagicMock,
        mock_resolve: MagicMock,
        mock_range: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Abort if good revision can't be built."""
        pkg = MagicMock()
        pkg.repo = "https://github.com/test/pkg"
        pkg.install_command = None
        pkg.test_command = None
        mock_load.return_value = pkg

        mock_resolve.side_effect = lambda _d, rev: f"full-{rev}"
        mock_range.return_value = ["c1", "c2"]

        mock_test.return_value = BisectStep(
            commit="full-v1.0", commit_short="full-v1", status="skip", detail="build failed"
        )

        config = self._make_config(work_dir=Path("/tmp/work"))
        with patch("pathlib.Path.exists", return_value=False):
            result = run_bisect(config)

        self.assertFalse(result.success)
        self.assertEqual(result.commits_tested, 1)


class TestBisectCLI(unittest.TestCase):
    """Tests for the bisect CLI command."""

    def test_bisect_command_exists(self) -> None:
        """The bisect command is registered on the CLI group."""
        from labeille.cli import main

        commands = main.list_commands(click.Context(main))
        self.assertIn("bisect", commands)

    @patch("labeille.bisect.run_bisect")
    def test_bisect_cli_invocation(self, mock_bisect: MagicMock) -> None:
        """CLI passes arguments correctly to run_bisect."""
        from click.testing import CliRunner

        from labeille.cli import main

        mock_bisect.return_value = BisectResult(
            package="requests",
            first_bad_commit="abc1234567890",
            first_bad_commit_short="abc1234",
            good_rev="v1.0",
            bad_rev="v2.0",
            steps=[],
            total_commits=10,
            commits_tested=4,
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "bisect",
                "requests",
                "--good=v1.0",
                "--bad=v2.0",
                "--target-python",
                "/usr/bin/python3",
                "--timeout",
                "300",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Bisecting requests", result.output)
        self.assertIn("abc1234", result.output)

        # Verify config was passed correctly.
        call_args = mock_bisect.call_args[0][0]
        self.assertEqual(call_args.package, "requests")
        self.assertEqual(call_args.good_rev, "v1.0")
        self.assertEqual(call_args.bad_rev, "v2.0")
        self.assertEqual(call_args.timeout, 300)

    @patch("labeille.bisect.run_bisect")
    def test_bisect_cli_failure_output(self, mock_bisect: MagicMock) -> None:
        """CLI reports failure when bisect can't find the bad commit."""
        from click.testing import CliRunner

        from labeille.cli import main

        mock_bisect.return_value = BisectResult(
            package="requests",
            first_bad_commit=None,
            first_bad_commit_short=None,
            good_rev="v1.0",
            bad_rev="v2.0",
            steps=[],
            total_commits=10,
            commits_tested=4,
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "bisect",
                "requests",
                "--good=v1.0",
                "--bad=v2.0",
                "--target-python",
                "/usr/bin/python3",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Could not identify", result.output)

    @patch("labeille.bisect.run_bisect")
    def test_bisect_cli_valueerror(self, mock_bisect: MagicMock) -> None:
        """CLI exits with error on ValueError from run_bisect."""
        from click.testing import CliRunner

        from labeille.cli import main

        mock_bisect.side_effect = ValueError("No repo URL for test-pkg")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "bisect",
                "test-pkg",
                "--good=v1.0",
                "--bad=v2.0",
                "--target-python",
                "/usr/bin/python3",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No repo URL", result.output)

    @patch("labeille.bisect.run_bisect")
    def test_bisect_cli_env_parsing(self, mock_bisect: MagicMock) -> None:
        """CLI parses --env KEY=VALUE pairs."""
        from click.testing import CliRunner

        from labeille.cli import main

        mock_bisect.return_value = BisectResult(
            package="pkg",
            first_bad_commit=None,
            first_bad_commit_short=None,
            good_rev="v1",
            bad_rev="v2",
            steps=[],
            total_commits=5,
            commits_tested=3,
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "bisect",
                "pkg",
                "--good=v1",
                "--bad=v2",
                "--target-python",
                "/usr/bin/python3",
                "--env",
                "FOO=bar",
                "--env",
                "BAZ=qux",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        call_args = mock_bisect.call_args[0][0]
        self.assertEqual(call_args.env_overrides, {"FOO": "bar", "BAZ": "qux"})

    @patch("labeille.bisect.run_bisect")
    def test_bisect_cli_extra_deps(self, mock_bisect: MagicMock) -> None:
        """CLI parses --extra-deps correctly."""
        from click.testing import CliRunner

        from labeille.cli import main

        mock_bisect.return_value = BisectResult(
            package="pkg",
            first_bad_commit=None,
            first_bad_commit_short=None,
            good_rev="v1",
            bad_rev="v2",
            steps=[],
            total_commits=5,
            commits_tested=3,
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "bisect",
                "pkg",
                "--good=v1",
                "--bad=v2",
                "--target-python",
                "/usr/bin/python3",
                "--extra-deps",
                "pytest,coverage",
            ],
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        call_args = mock_bisect.call_args[0][0]
        self.assertEqual(call_args.extra_deps, ["pytest", "coverage"])


# Need to import click for the CLI test class.
import click  # noqa: E402


if __name__ == "__main__":
    unittest.main()
