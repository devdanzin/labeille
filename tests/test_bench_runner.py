"""Tests for labeille.bench.runner — benchmark execution engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from labeille.bench.config import BenchConfig
from labeille.bench.results import (
    BenchIteration,
    BenchMeta,
    ConditionDef,
)
from labeille.bench.runner import BenchProgress, BenchRunner, quick_config
from labeille.bench.system import PythonProfile, StabilityCheck, SystemProfile, SystemSnapshot
from labeille.bench.timing import TimedResult
from labeille.registry import Index, IndexEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs: object) -> BenchConfig:
    """Create a BenchConfig with sensible test defaults."""
    defaults: dict[str, object] = {
        "iterations": 3,
        "warmup": 1,
        "timeout": 60,
        "default_target_python": "/usr/bin/python3",
    }
    defaults.update(kwargs)
    return BenchConfig(**defaults)  # type: ignore[arg-type]


def _make_condition(name: str = "baseline", **kwargs: object) -> ConditionDef:
    """Create a ConditionDef with defaults."""
    return ConditionDef(name=name, **kwargs)  # type: ignore[arg-type]


@dataclass
class FakePackage:
    """Stand-in for PackageEntry to avoid importing registry internals."""

    package: str
    repo: str | None = "https://github.com/test/repo"
    install_command: str = "pip install -e ."
    test_command: str = "python -m pytest"
    test_framework: str = "pytest"
    enriched: bool = True


def _make_timed_result(**kwargs: object) -> TimedResult:
    """Create a TimedResult with sensible defaults."""
    defaults: dict[str, object] = {
        "wall_time_s": 1.5,
        "user_time_s": 1.2,
        "sys_time_s": 0.1,
        "peak_rss_mb": 120.5,
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
    }
    defaults.update(kwargs)
    return TimedResult(**defaults)  # type: ignore[arg-type]


def _make_snapshot(load_avg: float = 0.5, ram_gb: float = 8.0) -> SystemSnapshot:
    """Create a fake SystemSnapshot."""
    return SystemSnapshot(timestamp=1000.0, load_avg_1m=load_avg, ram_available_gb=ram_gb)


def _stable_check() -> StabilityCheck:
    return StabilityCheck(stable=True, warnings=[], errors=[])


def _unstable_check() -> StabilityCheck:
    return StabilityCheck(
        stable=False,
        warnings=[],
        errors=["Load too high"],
    )


def _make_index(*names: str) -> Index:
    """Create a fake Index with the given package names."""
    return Index(packages=[IndexEntry(name=n) for n in names])


# ---------------------------------------------------------------------------
# BenchProgress tests
# ---------------------------------------------------------------------------


class TestBenchProgress(unittest.TestCase):
    """Tests for BenchProgress dataclass."""

    def test_progress_fields(self) -> None:
        p = BenchProgress(
            phase="measure",
            package="requests",
            condition="baseline",
            iteration=3,
            total_iterations=5,
            packages_done=2,
            packages_total=10,
            wall_time_s=1.23,
            status="ok",
            detail="extra info",
        )
        self.assertEqual(p.phase, "measure")
        self.assertEqual(p.package, "requests")
        self.assertEqual(p.condition, "baseline")
        self.assertEqual(p.iteration, 3)
        self.assertEqual(p.total_iterations, 5)
        self.assertEqual(p.packages_done, 2)
        self.assertEqual(p.packages_total, 10)
        self.assertAlmostEqual(p.wall_time_s, 1.23)
        self.assertEqual(p.status, "ok")
        self.assertEqual(p.detail, "extra info")


# ---------------------------------------------------------------------------
# BenchRunner init tests
# ---------------------------------------------------------------------------


class TestBenchRunnerInit(unittest.TestCase):
    """Tests for BenchRunner initialization."""

    def test_default_progress_callback(self) -> None:
        config = _make_config()
        runner = BenchRunner(config)
        self.assertEqual(runner.config, config)
        # Default progress callback is _default_progress.
        self.assertEqual(runner.progress, BenchRunner._default_progress)

    def test_custom_progress_callback(self) -> None:
        config = _make_config()
        cb = MagicMock()
        runner = BenchRunner(config, progress_callback=cb)
        self.assertEqual(runner.progress, cb)


# ---------------------------------------------------------------------------
# Validation integration tests
# ---------------------------------------------------------------------------


class TestValidation(unittest.TestCase):
    """Tests for config validation during run()."""

    def test_run_raises_on_invalid_config(self) -> None:
        """run() raises ValueError if config has no conditions."""
        config = _make_config(conditions={})
        runner = BenchRunner(config)
        with self.assertRaises(ValueError) as cm:
            runner.run()
        self.assertIn("Invalid benchmark configuration", str(cm.exception))

    def test_run_raises_on_missing_target_python(self) -> None:
        """run() raises ValueError if a condition has no target Python."""
        config = _make_config(
            default_target_python="",
            conditions={"baseline": _make_condition(target_python="")},
        )
        runner = BenchRunner(config)
        with self.assertRaises(ValueError) as cm:
            runner.run()
        self.assertIn("target python", str(cm.exception).lower())


# ---------------------------------------------------------------------------
# _setup_package tests
# ---------------------------------------------------------------------------


class TestSetupPackage(unittest.TestCase):
    """Tests for _setup_package."""

    def _make_runner(self, **config_kwargs: object) -> BenchRunner:
        """Create a BenchRunner with a single condition and temp dirs."""
        conditions = config_kwargs.pop("conditions", None) or {  # type: ignore[union-attr]
            "baseline": _make_condition(),
        }
        config = _make_config(conditions=conditions, **config_kwargs)
        return BenchRunner(config)

    @patch("labeille.bench.runner.time.monotonic", side_effect=[0.0, 0.5, 0.5, 1.0, 1.0, 1.2])
    @patch("labeille.bench.runner.resolve_target_python", return_value=Path("/usr/bin/python3"))
    @patch("labeille.bench.runner.resolve_env", return_value={})
    @patch("labeille.bench.runner.resolve_extra_deps", return_value=[])
    def test_setup_success(
        self,
        mock_extra: MagicMock,
        mock_env: MagicMock,
        mock_python: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """Successful setup returns dict with repo_dir, venvs, durations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self._make_runner(
                repos_dir=Path(tmpdir) / "repos",
                venvs_dir=Path(tmpdir) / "venvs",
            )
            pkg = FakePackage(package="testpkg")

            with (
                patch("labeille.runner.clone_repo") as mock_clone,
                patch("labeille.runner.create_venv") as mock_create,
                patch("labeille.runner.install_package") as mock_install,
            ):
                mock_install.return_value = SimpleNamespace(returncode=0)
                result = runner._setup_package(pkg)

            self.assertIsNotNone(result)
            self.assertIn("repo_dir", result)
            self.assertIn("venvs", result)
            self.assertIn("clone_duration", result)
            self.assertIn("baseline", result["venvs"])
            mock_clone.assert_called_once()
            mock_create.assert_called_once()
            mock_install.assert_called_once()

    def test_setup_no_repo(self) -> None:
        """Returns None when package has no repo URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self._make_runner(repos_dir=Path(tmpdir) / "repos")
            pkg = FakePackage(package="norepo", repo=None)
            result = runner._setup_package(pkg)
            self.assertIsNone(result)

    @patch("labeille.bench.runner.time.monotonic", side_effect=[0.0, 0.5])
    def test_setup_clone_failure(self, mock_time: MagicMock) -> None:
        """Returns None when clone fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self._make_runner(repos_dir=Path(tmpdir) / "repos")
            pkg = FakePackage(package="clonefail")

            with patch("labeille.runner.clone_repo", side_effect=RuntimeError("clone failed")):
                result = runner._setup_package(pkg)

            self.assertIsNone(result)

    @patch("labeille.bench.runner.time.monotonic", side_effect=[0.0, 0.5, 0.5, 1.0, 1.0, 1.5])
    @patch("labeille.bench.runner.resolve_target_python", return_value=Path("/usr/bin/python3"))
    @patch("labeille.bench.runner.resolve_env", return_value={})
    @patch("labeille.bench.runner.resolve_extra_deps", return_value=[])
    def test_setup_install_failure(
        self,
        mock_extra: MagicMock,
        mock_env: MagicMock,
        mock_python: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """Returns None when install fails with non-zero exit code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self._make_runner(
                repos_dir=Path(tmpdir) / "repos",
                venvs_dir=Path(tmpdir) / "venvs",
            )
            pkg = FakePackage(package="installfail")

            with (
                patch("labeille.runner.clone_repo"),
                patch("labeille.runner.create_venv"),
                patch("labeille.runner.install_package") as mock_install,
            ):
                mock_install.return_value = SimpleNamespace(returncode=1)
                result = runner._setup_package(pkg)

            self.assertIsNone(result)

    @patch("labeille.bench.runner.time.monotonic", side_effect=[0.0, 0.5, 0.5, 1.0, 1.0, 1.5])
    @patch("labeille.bench.runner.resolve_target_python", return_value=Path("/usr/bin/python3"))
    @patch("labeille.bench.runner.resolve_env", return_value={})
    @patch("labeille.bench.runner.resolve_extra_deps", return_value=[])
    def test_setup_install_exception(
        self,
        mock_extra: MagicMock,
        mock_env: MagicMock,
        mock_python: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """Returns None when install raises an exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self._make_runner(
                repos_dir=Path(tmpdir) / "repos",
                venvs_dir=Path(tmpdir) / "venvs",
            )
            pkg = FakePackage(package="instexc")

            with (
                patch("labeille.runner.clone_repo"),
                patch("labeille.runner.create_venv"),
                patch("labeille.runner.install_package", side_effect=RuntimeError("install boom")),
            ):
                result = runner._setup_package(pkg)

            self.assertIsNone(result)

    @patch(
        "labeille.bench.runner.time.monotonic",
        side_effect=[
            0.0,
            0.5,  # clone
            0.5,
            1.0,  # venv A
            1.0,
            1.5,  # install A
            1.5,
            2.0,  # venv B
            2.0,
            2.5,  # install B
        ],
    )
    @patch("labeille.bench.runner.resolve_target_python", return_value=Path("/usr/bin/python3"))
    @patch("labeille.bench.runner.resolve_env", return_value={})
    @patch("labeille.bench.runner.resolve_extra_deps", return_value=[])
    def test_setup_venv_per_condition(
        self,
        mock_extra: MagicMock,
        mock_env: MagicMock,
        mock_python: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """Creates separate venvs for each condition."""
        conditions = {
            "baseline": _make_condition("baseline"),
            "coverage": _make_condition("coverage"),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self._make_runner(
                conditions=conditions,
                repos_dir=Path(tmpdir) / "repos",
                venvs_dir=Path(tmpdir) / "venvs",
            )
            pkg = FakePackage(package="multipkg")

            with (
                patch("labeille.runner.clone_repo"),
                patch("labeille.runner.create_venv") as mock_create,
                patch("labeille.runner.install_package") as mock_install,
            ):
                mock_install.return_value = SimpleNamespace(returncode=0)
                result = runner._setup_package(pkg)

            self.assertIsNotNone(result)
            self.assertEqual(mock_create.call_count, 2)
            self.assertEqual(mock_install.call_count, 2)
            self.assertIn("baseline", result["venvs"])
            self.assertIn("coverage", result["venvs"])

    @patch(
        "labeille.bench.runner.time.monotonic",
        side_effect=[0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0],
    )
    @patch("labeille.bench.runner.resolve_target_python", return_value=Path("/usr/bin/python3"))
    @patch("labeille.bench.runner.resolve_env", return_value={})
    @patch("labeille.bench.runner.resolve_extra_deps", return_value=["pytest-cov", "coverage"])
    def test_setup_extra_deps_installed(
        self,
        mock_extra: MagicMock,
        mock_env: MagicMock,
        mock_python: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """Extra deps are installed after the main package."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self._make_runner(
                repos_dir=Path(tmpdir) / "repos",
                venvs_dir=Path(tmpdir) / "venvs",
            )
            pkg = FakePackage(package="extradeps")

            with (
                patch("labeille.runner.clone_repo"),
                patch("labeille.runner.create_venv"),
                patch("labeille.runner.install_package") as mock_install,
            ):
                mock_install.return_value = SimpleNamespace(returncode=0)
                result = runner._setup_package(pkg)

            self.assertIsNotNone(result)
            # 1st call: main install, 2nd call: extra deps.
            self.assertEqual(mock_install.call_count, 2)
            second_call = mock_install.call_args_list[1]
            self.assertIn("pytest-cov", str(second_call))

    @patch(
        "labeille.bench.runner.time.monotonic",
        side_effect=[0.0, 0.5, 0.5, 1.0, 1.0, 1.5],
    )
    @patch("labeille.bench.runner.resolve_target_python", return_value=Path("/usr/bin/python3"))
    @patch("labeille.bench.runner.resolve_env", return_value={})
    @patch("labeille.bench.runner.resolve_extra_deps", return_value=["badpkg"])
    def test_setup_extra_deps_failure_nonfatal(
        self,
        mock_extra: MagicMock,
        mock_env: MagicMock,
        mock_python: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        """Extra deps install failure is non-fatal (logged as warning)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self._make_runner(
                repos_dir=Path(tmpdir) / "repos",
                venvs_dir=Path(tmpdir) / "venvs",
            )
            pkg = FakePackage(package="extrafail")

            with (
                patch("labeille.runner.clone_repo"),
                patch("labeille.runner.create_venv"),
                patch("labeille.runner.install_package") as mock_install,
            ):
                # Main install succeeds, extra deps install raises.
                mock_install.side_effect = [
                    SimpleNamespace(returncode=0),
                    RuntimeError("extra deps failed"),
                ]
                result = runner._setup_package(pkg)

            # Setup still succeeds despite extra deps failure.
            self.assertIsNotNone(result)
            self.assertIn("venvs", result)


# ---------------------------------------------------------------------------
# _run_iteration tests
# ---------------------------------------------------------------------------


class TestRunIteration(unittest.TestCase):
    """Tests for _run_iteration."""

    def _make_runner_and_setup(
        self,
        cond_name: str = "baseline",
        **config_kwargs: object,
    ) -> tuple[BenchRunner, dict[str, object], ConditionDef]:
        """Create runner with a basic setup dict."""
        cond = _make_condition(cond_name)
        config = _make_config(
            conditions={cond_name: cond},
            **config_kwargs,
        )
        runner = BenchRunner(config)
        setup: dict[str, object] = {
            "repo_dir": Path("/tmp/fake/repo"),
            "venvs": {cond_name: Path("/tmp/fake/venv")},
        }
        return runner, setup, cond

    @patch("labeille.bench.runner.SystemSnapshot.capture")
    @patch("labeille.bench.runner.run_timed_in_venv")
    def test_iteration_ok(
        self,
        mock_timed: MagicMock,
        mock_snap: MagicMock,
    ) -> None:
        """Exit code 0 yields status 'ok'."""
        mock_timed.return_value = _make_timed_result(exit_code=0)
        mock_snap.return_value = _make_snapshot()
        runner, setup, cond = self._make_runner_and_setup()

        it = runner._run_iteration(
            pkg=FakePackage(package="mypkg"),
            cond=cond,
            setup=setup,
            iter_index=1,
            is_warmup=False,
        )
        self.assertEqual(it.status, "ok")
        self.assertEqual(it.exit_code, 0)
        self.assertFalse(it.warmup)
        self.assertEqual(it.index, 1)
        self.assertAlmostEqual(it.wall_time_s, 1.5)

    @patch("labeille.bench.runner.SystemSnapshot.capture")
    @patch("labeille.bench.runner.run_timed_in_venv")
    def test_iteration_fail(
        self,
        mock_timed: MagicMock,
        mock_snap: MagicMock,
    ) -> None:
        """Positive non-zero exit code yields status 'fail'."""
        mock_timed.return_value = _make_timed_result(exit_code=1)
        mock_snap.return_value = _make_snapshot()
        runner, setup, cond = self._make_runner_and_setup()

        it = runner._run_iteration(
            pkg=FakePackage(package="mypkg"),
            cond=cond,
            setup=setup,
            iter_index=2,
            is_warmup=False,
        )
        self.assertEqual(it.status, "fail")
        self.assertEqual(it.exit_code, 1)

    @patch("labeille.bench.runner.SystemSnapshot.capture")
    @patch("labeille.bench.runner.run_timed_in_venv")
    def test_iteration_timeout(
        self,
        mock_timed: MagicMock,
        mock_snap: MagicMock,
    ) -> None:
        """Timed out yields status 'timeout'."""
        mock_timed.return_value = _make_timed_result(timed_out=True, exit_code=-9)
        mock_snap.return_value = _make_snapshot()
        runner, setup, cond = self._make_runner_and_setup()

        it = runner._run_iteration(
            pkg=FakePackage(package="mypkg"),
            cond=cond,
            setup=setup,
            iter_index=1,
            is_warmup=False,
        )
        self.assertEqual(it.status, "timeout")

    @patch("labeille.bench.runner.SystemSnapshot.capture")
    @patch("labeille.bench.runner.run_timed_in_venv")
    def test_iteration_signal(
        self,
        mock_timed: MagicMock,
        mock_snap: MagicMock,
    ) -> None:
        """Negative exit code (signal) yields status 'error'."""
        mock_timed.return_value = _make_timed_result(exit_code=-11)
        mock_snap.return_value = _make_snapshot()
        runner, setup, cond = self._make_runner_and_setup()

        it = runner._run_iteration(
            pkg=FakePackage(package="mypkg"),
            cond=cond,
            setup=setup,
            iter_index=1,
            is_warmup=False,
        )
        self.assertEqual(it.status, "error")
        self.assertEqual(it.exit_code, -11)

    @patch("labeille.bench.runner.SystemSnapshot.capture")
    @patch("labeille.bench.runner.run_timed_in_venv")
    def test_iteration_captures_system_state(
        self,
        mock_timed: MagicMock,
        mock_snap: MagicMock,
    ) -> None:
        """System snapshots before/after populate load_avg and ram fields."""
        snap_before = _make_snapshot(load_avg=0.3, ram_gb=12.0)
        snap_after = _make_snapshot(load_avg=0.8, ram_gb=10.0)
        mock_snap.side_effect = [snap_before, snap_after]
        mock_timed.return_value = _make_timed_result()
        runner, setup, cond = self._make_runner_and_setup()

        it = runner._run_iteration(
            pkg=FakePackage(package="mypkg"),
            cond=cond,
            setup=setup,
            iter_index=1,
            is_warmup=False,
        )
        self.assertAlmostEqual(it.load_avg_start, 0.3)
        self.assertAlmostEqual(it.load_avg_end, 0.8)
        self.assertAlmostEqual(it.ram_available_start_gb, 12.0)

    @patch("labeille.bench.runner.SystemSnapshot.capture")
    @patch("labeille.bench.runner.run_timed_in_venv")
    def test_iteration_test_command_resolution(
        self,
        mock_timed: MagicMock,
        mock_snap: MagicMock,
    ) -> None:
        """Test command is resolved from registry and condition."""
        mock_timed.return_value = _make_timed_result()
        mock_snap.return_value = _make_snapshot()

        cond = _make_condition(test_command_suffix="--tb=short")
        config = _make_config(conditions={"baseline": cond})
        runner = BenchRunner(config)
        setup: dict[str, object] = {
            "repo_dir": Path("/tmp/fake/repo"),
            "venvs": {"baseline": Path("/tmp/fake/venv")},
        }

        runner._run_iteration(
            pkg=FakePackage(package="mypkg", test_command="python -m pytest"),
            cond=cond,
            setup=setup,
            iter_index=1,
            is_warmup=False,
        )

        # Verify the test command was passed to run_timed_in_venv.
        mock_timed.assert_called_once()
        called_cmd = mock_timed.call_args[1]["test_command"]
        self.assertIn("pytest", called_cmd)
        self.assertIn("--tb=short", called_cmd)

    @patch("labeille.bench.runner.SystemSnapshot.capture")
    @patch("labeille.bench.runner.run_timed_in_venv")
    def test_iteration_env_passing(
        self,
        mock_timed: MagicMock,
        mock_snap: MagicMock,
    ) -> None:
        """Env vars from condition and defaults are passed to run_timed_in_venv."""
        mock_timed.return_value = _make_timed_result()
        mock_snap.return_value = _make_snapshot()

        cond = _make_condition("cov", env={"COVERAGE_PROCESS_START": "1"})
        config = _make_config(
            conditions={"cov": cond},
            default_env={"PYTHONFAULTHANDLER": "1"},
        )
        runner = BenchRunner(config)
        setup: dict[str, object] = {
            "repo_dir": Path("/tmp/fake/repo"),
            "venvs": {"cov": Path("/tmp/fake/venv")},
        }

        runner._run_iteration(
            pkg=FakePackage(package="mypkg"),
            cond=cond,
            setup=setup,
            iter_index=1,
            is_warmup=False,
        )

        called_env = mock_timed.call_args[1]["env"]
        self.assertEqual(called_env["COVERAGE_PROCESS_START"], "1")
        self.assertIn("PYTHONFAULTHANDLER", called_env)
        self.assertIn("PYTHONDONTWRITEBYTECODE", called_env)


# ---------------------------------------------------------------------------
# Execution strategy tests
# ---------------------------------------------------------------------------


class TestExecutionStrategies(unittest.TestCase):
    """Tests for block, alternating, and interleaved execution."""

    def _make_tracking_runner(
        self,
        conditions: dict[str, ConditionDef],
        *,
        warmup: int = 0,
        iterations: int = 3,
        alternate: bool | None = None,
        interleave: bool = False,
    ) -> tuple[BenchRunner, list[tuple[str, str, int, bool]]]:
        """Create a runner that records iteration order.

        Returns (runner, call_log) where call_log is populated as
        (package, condition, iter_index, is_warmup) tuples.
        """
        config = _make_config(
            conditions=conditions,
            warmup=warmup,
            iterations=iterations,
            alternate=alternate,
            interleave=interleave,
            default_target_python="/usr/bin/python3",
        )
        runner = BenchRunner(config, progress_callback=lambda p: None)

        call_log: list[tuple[str, str, int, bool]] = []

        def tracking_run_iteration(
            *,
            pkg: object,
            cond: object,
            setup: object,
            iter_index: int,
            is_warmup: bool,
        ) -> BenchIteration:
            call_log.append(
                (
                    pkg.package,  # type: ignore[union-attr]
                    cond.name,  # type: ignore[union-attr]
                    iter_index,
                    is_warmup,
                )
            )
            return BenchIteration(
                index=iter_index,
                warmup=is_warmup,
                wall_time_s=1.0,
                user_time_s=0.8,
                sys_time_s=0.1,
                peak_rss_mb=100.0,
                exit_code=0,
                status="ok",
            )

        runner._run_iteration = tracking_run_iteration  # type: ignore[assignment]
        return runner, call_log

    def _fake_setup(
        self, pkg: FakePackage, conditions: dict[str, ConditionDef]
    ) -> dict[str, object]:
        """Create a fake setup dict for a package."""
        return {
            "repo_dir": Path("/tmp/fake/repo"),
            "venvs": {name: Path(f"/tmp/fake/venv_{name}") for name in conditions},
            "clone_duration": 0.5,
        }

    def test_block_order(self) -> None:
        """Block mode: all iterations of condition A, then all of B."""
        conds = {
            "A": _make_condition("A"),
            "B": _make_condition("B"),
        }
        runner, call_log = self._make_tracking_runner(
            conds,
            iterations=3,
            alternate=False,
        )
        pkg = FakePackage(package="mypkg")
        setup = self._fake_setup(pkg, conds)

        with patch.object(runner, "_setup_package", return_value=setup):
            runner._benchmark_package(pkg, 0, 1)

        # Block: A1, A2, A3, B1, B2, B3.
        conditions_order = [c[1] for c in call_log]
        self.assertEqual(conditions_order, ["A", "A", "A", "B", "B", "B"])

    def test_alternating_order(self) -> None:
        """Alternating mode: A1, B1, A2, B2, A3, B3."""
        conds = {
            "A": _make_condition("A"),
            "B": _make_condition("B"),
        }
        runner, call_log = self._make_tracking_runner(
            conds,
            iterations=3,
            alternate=True,
        )
        pkg = FakePackage(package="mypkg")
        setup = self._fake_setup(pkg, conds)

        with patch.object(runner, "_setup_package", return_value=setup):
            runner._benchmark_package(pkg, 0, 1)

        conditions_order = [c[1] for c in call_log]
        self.assertEqual(conditions_order, ["A", "B", "A", "B", "A", "B"])

    def test_interleaved_order(self) -> None:
        """Interleaved: iteration 1 of all pkgs, then iteration 2, etc."""
        conds = {"A": _make_condition("A")}
        runner, call_log = self._make_tracking_runner(
            conds,
            iterations=2,
            interleave=True,
        )
        pkgs = [FakePackage(package="pkg1"), FakePackage(package="pkg2")]

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(runner, "_setup_package") as mock_setup,
        ):
            results_path = Path(tmpdir) / "results.jsonl"
            mock_setup.side_effect = [self._fake_setup(p, conds) for p in pkgs]
            runner._run_interleaved(pkgs, results_path)

        # Interleaved: pkg1-A-iter1, pkg2-A-iter1, pkg1-A-iter2, pkg2-A-iter2.
        order = [(c[0], c[2]) for c in call_log]
        self.assertEqual(
            order,
            [
                ("pkg1", 1),
                ("pkg2", 1),
                ("pkg1", 2),
                ("pkg2", 2),
            ],
        )

    def test_warmup_marking(self) -> None:
        """Warmup iterations are marked correctly."""
        conds = {"A": _make_condition("A")}
        runner, call_log = self._make_tracking_runner(
            conds,
            warmup=1,
            iterations=3,
        )
        pkg = FakePackage(package="mypkg")
        setup = self._fake_setup(pkg, conds)

        with patch.object(runner, "_setup_package", return_value=setup):
            runner._benchmark_package(pkg, 0, 1)

        # Total iterations = warmup(1) + measured(3) = 4.
        warmup_flags = [c[3] for c in call_log]
        self.assertEqual(warmup_flags, [True, False, False, False])

    def test_progress_callback_called(self) -> None:
        """Progress callback is called for each iteration."""
        conds = {"A": _make_condition("A")}
        config = _make_config(
            conditions=conds,
            warmup=0,
            iterations=2,
            default_target_python="/usr/bin/python3",
        )
        progress_log: list[BenchProgress] = []
        runner = BenchRunner(config, progress_callback=progress_log.append)
        pkg = FakePackage(package="mypkg")
        setup: dict[str, object] = {
            "repo_dir": Path("/tmp/fake/repo"),
            "venvs": {"A": Path("/tmp/fake/venv_A")},
            "clone_duration": 0.5,
        }

        with (
            patch.object(runner, "_setup_package", return_value=setup),
            patch("labeille.bench.runner.SystemSnapshot.capture", return_value=_make_snapshot()),
            patch("labeille.bench.runner.run_timed_in_venv", return_value=_make_timed_result()),
        ):
            runner._benchmark_package(pkg, 0, 5)

        self.assertEqual(len(progress_log), 2)
        self.assertEqual(progress_log[0].package, "mypkg")
        self.assertEqual(progress_log[0].condition, "A")
        self.assertEqual(progress_log[0].iteration, 1)
        self.assertEqual(progress_log[1].iteration, 2)
        self.assertEqual(progress_log[0].packages_total, 5)


# ---------------------------------------------------------------------------
# Result writing tests
# ---------------------------------------------------------------------------


class TestResultWriting(unittest.TestCase):
    """Tests for incremental result writing."""

    def test_incremental_write(self) -> None:
        """Results are written to JSONL incrementally during sequential runs."""
        conds = {"A": _make_condition("A")}
        config = _make_config(
            conditions=conds,
            warmup=0,
            iterations=3,
            default_target_python="/usr/bin/python3",
        )
        runner = BenchRunner(config, progress_callback=lambda p: None)

        pkgs = [FakePackage(package="pkg1"), FakePackage(package="pkg2")]

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.jsonl"

            def fake_setup(pkg: object) -> dict[str, object]:
                return {
                    "repo_dir": Path("/tmp/fake/repo"),
                    "venvs": {"A": Path("/tmp/fake/venv_A")},
                    "clone_duration": 0.5,
                }

            with (
                patch.object(runner, "_setup_package", side_effect=fake_setup),
                patch(
                    "labeille.bench.runner.SystemSnapshot.capture", return_value=_make_snapshot()
                ),
                patch(
                    "labeille.bench.runner.run_timed_in_venv", return_value=_make_timed_result()
                ),
            ):
                results = runner._run_sequential(pkgs, results_path)

            # Two packages written.
            self.assertEqual(len(results), 2)
            lines = results_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)

            # Each line is valid JSON with a package field.
            for i, line in enumerate(lines):
                data = json.loads(line)
                self.assertIn("package", data)
                self.assertEqual(data["package"], pkgs[i].package)

    def test_output_dir_created(self) -> None:
        """Output directory is created during run()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bench_output" / "run1"
            config = _make_config(
                conditions={"baseline": _make_condition()},
                default_target_python="/usr/bin/python3",
                results_dir=Path(tmpdir) / "bench_output",
                registry_dir=Path(tmpdir) / "registry",
            )
            config.bench_id = "run1"
            runner = BenchRunner(config)

            # Mock all the phases.
            with (
                patch("labeille.bench.runner.validate_config", return_value=[]),
                patch(
                    "labeille.bench.runner.capture_system_profile", return_value=SystemProfile()
                ),
                patch(
                    "labeille.bench.runner.capture_python_profile", return_value=PythonProfile()
                ),
                patch("labeille.bench.runner.format_system_profile", return_value=""),
                patch("labeille.bench.runner.format_python_profile", return_value=""),
                patch(
                    "labeille.bench.runner.resolve_target_python",
                    return_value=Path("/usr/bin/python3"),
                ),
                patch("labeille.bench.runner.resolve_env", return_value={}),
                patch.object(runner, "_load_packages", return_value=[]),
                patch("labeille.bench.runner.save_bench_run"),
            ):
                meta, results = runner.run()

            self.assertTrue(output_dir.exists())

    def test_meta_written(self) -> None:
        """Metadata file is written at start and final save is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(
                conditions={"baseline": _make_condition()},
                default_target_python="/usr/bin/python3",
                results_dir=Path(tmpdir),
                registry_dir=Path(tmpdir) / "registry",
            )
            runner = BenchRunner(config)

            with (
                patch("labeille.bench.runner.validate_config", return_value=[]),
                patch(
                    "labeille.bench.runner.capture_system_profile", return_value=SystemProfile()
                ),
                patch(
                    "labeille.bench.runner.capture_python_profile", return_value=PythonProfile()
                ),
                patch("labeille.bench.runner.format_system_profile", return_value=""),
                patch("labeille.bench.runner.format_python_profile", return_value=""),
                patch(
                    "labeille.bench.runner.resolve_target_python",
                    return_value=Path("/usr/bin/python3"),
                ),
                patch("labeille.bench.runner.resolve_env", return_value={}),
                patch.object(runner, "_load_packages", return_value=[]),
                patch("labeille.bench.runner.save_bench_run") as mock_save,
            ):
                meta, results = runner.run()

            # save_bench_run should have been called (final save).
            mock_save.assert_called_once()
            # Initial meta file should also exist.
            meta_path = config.output_dir / "bench_meta.json"
            self.assertTrue(meta_path.exists())


# ---------------------------------------------------------------------------
# _wait_for_stability tests
# ---------------------------------------------------------------------------


class TestWaitForStability(unittest.TestCase):
    """Tests for _wait_for_stability."""

    @patch("labeille.bench.runner.check_stability")
    @patch("labeille.bench.runner.time.sleep")
    def test_already_stable(
        self,
        mock_sleep: MagicMock,
        mock_stability: MagicMock,
    ) -> None:
        """Returns immediately if system is already stable."""
        mock_stability.return_value = _stable_check()
        runner = BenchRunner(_make_config())
        runner._wait_for_stability()
        mock_sleep.assert_not_called()

    @patch("labeille.bench.runner.check_stability")
    @patch("labeille.bench.runner.time.sleep")
    def test_becomes_stable(
        self,
        mock_sleep: MagicMock,
        mock_stability: MagicMock,
    ) -> None:
        """Waits and returns when system becomes stable."""
        mock_stability.side_effect = [
            _unstable_check(),
            _unstable_check(),
            _stable_check(),
        ]
        runner = BenchRunner(_make_config())
        runner._wait_for_stability()
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("labeille.bench.runner.check_stability")
    @patch("labeille.bench.runner.time.sleep")
    def test_stability_timeout(
        self,
        mock_sleep: MagicMock,
        mock_stability: MagicMock,
    ) -> None:
        """Proceeds after max wait even if never stable."""
        mock_stability.return_value = _unstable_check()
        runner = BenchRunner(_make_config())
        runner._wait_for_stability()
        # max_wait=300, interval=10 → 30 iterations.
        self.assertEqual(mock_sleep.call_count, 30)


# ---------------------------------------------------------------------------
# quick_config tests
# ---------------------------------------------------------------------------


class TestQuickConfig(unittest.TestCase):
    """Tests for quick_config helper."""

    def test_quick_reduces_iterations(self) -> None:
        config = BenchConfig(iterations=10, warmup=3)
        result = quick_config(config)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.warmup, 0)

    def test_quick_limits_top_n(self) -> None:
        config = BenchConfig(top_n=100)
        result = quick_config(config)
        self.assertEqual(result.top_n, 20)

    def test_quick_preserves_low_top_n(self) -> None:
        config = BenchConfig(top_n=5)
        result = quick_config(config)
        self.assertEqual(result.top_n, 5)

    def test_quick_sets_name(self) -> None:
        config = BenchConfig(name="")
        result = quick_config(config)
        self.assertEqual(result.name, "Quick benchmark")

    def test_quick_appends_to_name(self) -> None:
        config = BenchConfig(name="My bench")
        result = quick_config(config)
        self.assertEqual(result.name, "My bench (quick)")


# ---------------------------------------------------------------------------
# Full integration test (heavily mocked)
# ---------------------------------------------------------------------------


class TestFullIntegration(unittest.TestCase):
    """Full run() integration test with all externals mocked."""

    def test_full_run_single_package(self) -> None:
        """A complete run with one package and one condition produces results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cond = _make_condition()
            config = _make_config(
                conditions={"baseline": cond},
                iterations=3,
                warmup=0,
                results_dir=Path(tmpdir),
                registry_dir=Path(tmpdir) / "registry",
                default_target_python="/usr/bin/python3",
            )
            progress_log: list[BenchProgress] = []
            runner = BenchRunner(config, progress_callback=progress_log.append)
            pkg = FakePackage(package="testpkg")

            with (
                # Phase 1: validation.
                patch("labeille.bench.runner.validate_config", return_value=[]),
                # Phase 2: system profiling.
                patch(
                    "labeille.bench.runner.capture_system_profile", return_value=SystemProfile()
                ),
                patch(
                    "labeille.bench.runner.capture_python_profile", return_value=PythonProfile()
                ),
                patch("labeille.bench.runner.format_system_profile", return_value=""),
                patch("labeille.bench.runner.format_python_profile", return_value=""),
                patch(
                    "labeille.bench.runner.resolve_target_python",
                    return_value=Path("/usr/bin/python3"),
                ),
                patch("labeille.bench.runner.resolve_env", return_value={}),
                # Phase 5: load packages.
                patch.object(runner, "_load_packages", return_value=[pkg]),
                # Setup.
                patch.object(
                    runner,
                    "_setup_package",
                    return_value={
                        "repo_dir": Path("/tmp/fake/repo"),
                        "venvs": {"baseline": Path("/tmp/fake/venv")},
                        "clone_duration": 0.5,
                        "install_baseline": 1.0,
                        "venv_baseline": 0.5,
                    },
                ),
                # Timing.
                patch(
                    "labeille.bench.runner.SystemSnapshot.capture", return_value=_make_snapshot()
                ),
                patch(
                    "labeille.bench.runner.run_timed_in_venv", return_value=_make_timed_result()
                ),
                # Final save.
                patch("labeille.bench.runner.save_bench_run") as mock_save,
            ):
                meta, results = runner.run()

            # Verify results.
            self.assertIsInstance(meta, BenchMeta)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].package, "testpkg")
            self.assertIn("baseline", results[0].conditions)

            # 3 measured iterations.
            cond_result = results[0].conditions["baseline"]
            self.assertEqual(len(cond_result.iterations), 3)
            for it in cond_result.iterations:
                self.assertEqual(it.status, "ok")
                self.assertFalse(it.warmup)

            # Progress callback was called 3 times.
            self.assertEqual(len(progress_log), 3)

            # Meta has expected fields.
            self.assertTrue(meta.start_time)
            self.assertTrue(meta.end_time)
            self.assertEqual(meta.packages_total, 1)
            self.assertEqual(meta.packages_completed, 1)
            self.assertEqual(meta.packages_skipped, 0)

            # Final save was called.
            mock_save.assert_called_once()

    def test_full_run_skipped_package(self) -> None:
        """A package that fails setup is marked skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cond = _make_condition()
            config = _make_config(
                conditions={"baseline": cond},
                iterations=3,
                warmup=0,
                results_dir=Path(tmpdir),
                registry_dir=Path(tmpdir) / "registry",
                default_target_python="/usr/bin/python3",
            )
            runner = BenchRunner(config, progress_callback=lambda p: None)
            pkg = FakePackage(package="badpkg")

            with (
                patch("labeille.bench.runner.validate_config", return_value=[]),
                patch(
                    "labeille.bench.runner.capture_system_profile", return_value=SystemProfile()
                ),
                patch(
                    "labeille.bench.runner.capture_python_profile", return_value=PythonProfile()
                ),
                patch("labeille.bench.runner.format_system_profile", return_value=""),
                patch("labeille.bench.runner.format_python_profile", return_value=""),
                patch(
                    "labeille.bench.runner.resolve_target_python",
                    return_value=Path("/usr/bin/python3"),
                ),
                patch("labeille.bench.runner.resolve_env", return_value={}),
                patch.object(runner, "_load_packages", return_value=[pkg]),
                # Setup fails.
                patch.object(runner, "_setup_package", return_value=None),
                patch("labeille.bench.runner.save_bench_run"),
            ):
                meta, results = runner.run()

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].skipped)
            self.assertEqual(results[0].skip_reason, "setup failed")
            self.assertEqual(meta.packages_skipped, 1)
            self.assertEqual(meta.packages_completed, 0)


# ---------------------------------------------------------------------------
# _load_packages tests
# ---------------------------------------------------------------------------


class TestLoadPackages(unittest.TestCase):
    """Tests for _load_packages."""

    def test_load_packages_requires_registry_dir(self) -> None:
        """Raises ValueError if registry_dir is not set."""
        config = _make_config(registry_dir=None)
        runner = BenchRunner(config)
        with self.assertRaises(ValueError) as cm:
            runner._load_packages()
        self.assertIn("Registry directory", str(cm.exception))

    def test_load_packages_with_filter(self) -> None:
        """Only loads packages in the filter list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(
                registry_dir=Path(tmpdir),
                packages_filter=["pkg1", "pkg3"],
            )
            runner = BenchRunner(config)

            with (
                patch(
                    "labeille.registry.load_index",
                    return_value=_make_index("pkg1", "pkg2", "pkg3"),
                ),
                patch(
                    "labeille.registry.load_package",
                    side_effect=lambda name, d: FakePackage(package=name),
                ),
            ):
                result = runner._load_packages()

            names = [p.package for p in result]
            self.assertEqual(names, ["pkg1", "pkg3"])

    def test_load_packages_with_top_n(self) -> None:
        """Top N limits the number of packages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(
                registry_dir=Path(tmpdir),
                top_n=2,
            )
            runner = BenchRunner(config)

            with (
                patch(
                    "labeille.registry.load_index",
                    return_value=_make_index("a", "b", "c", "d"),
                ),
                patch(
                    "labeille.registry.load_package",
                    side_effect=lambda name, d: FakePackage(package=name),
                ),
            ):
                result = runner._load_packages()

            self.assertEqual(len(result), 2)

    def test_load_packages_with_skip(self) -> None:
        """Skip filter excludes packages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_config(
                registry_dir=Path(tmpdir),
                skip_packages=["pkg2"],
            )
            runner = BenchRunner(config)

            with (
                patch(
                    "labeille.registry.load_index",
                    return_value=_make_index("pkg1", "pkg2", "pkg3"),
                ),
                patch(
                    "labeille.registry.load_package",
                    side_effect=lambda name, d: FakePackage(package=name),
                ),
            ):
                result = runner._load_packages()

            names = [p.package for p in result]
            self.assertNotIn("pkg2", names)
            self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Default progress callback tests
# ---------------------------------------------------------------------------


class TestDefaultProgress(unittest.TestCase):
    """Tests for _default_progress static method."""

    @patch("labeille.bench.runner.log")
    def test_warmup_marker(self, mock_log: MagicMock) -> None:
        p = BenchProgress(
            phase="warmup",
            package="pkg",
            condition="A",
            iteration=1,
            total_iterations=5,
            packages_done=0,
            packages_total=1,
        )
        BenchRunner._default_progress(p)
        logged = mock_log.info.call_args[0][0]
        self.assertIn("W", logged)

    @patch("labeille.bench.runner.log")
    def test_measure_marker(self, mock_log: MagicMock) -> None:
        p = BenchProgress(
            phase="measure",
            package="pkg",
            condition="A",
            iteration=2,
            total_iterations=5,
            packages_done=0,
            packages_total=1,
            wall_time_s=1.23,
            status="ok",
        )
        BenchRunner._default_progress(p)
        logged = mock_log.info.call_args[0][0]
        self.assertIn("M", logged)
        self.assertIn("1.23", logged)
        self.assertIn("ok", logged)


if __name__ == "__main__":
    unittest.main()
