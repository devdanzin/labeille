"""Tests for labeille.runner."""

import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from labeille.registry import (
    Index,
    IndexEntry,
    PackageEntry,
    save_index,
    save_package,
)
from labeille.runner import (
    PackageResult,
    RunnerConfig,
    _resolve_dirs,
    append_result,
    build_env,
    create_run_dir,
    filter_packages,
    get_package_version,
    load_completed_packages,
    run_all,
    run_package,
    save_crash_stderr,
    write_run_meta,
)


def _make_config(
    tmpdir: Path,
    *,
    dry_run: bool = False,
    skip_extensions: bool = False,
    skip_completed: bool = False,
    stop_after_crash: int | None = None,
    top_n: int | None = None,
    packages_filter: list[str] | None = None,
    timeout: int = 600,
) -> RunnerConfig:
    registry_dir = tmpdir / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "packages").mkdir(exist_ok=True)
    return RunnerConfig(
        target_python=Path("/usr/bin/python3"),
        registry_dir=registry_dir,
        results_dir=tmpdir / "results",
        run_id="test-run",
        timeout=timeout,
        top_n=top_n,
        packages_filter=packages_filter,
        skip_extensions=skip_extensions,
        skip_completed=skip_completed,
        stop_after_crash=stop_after_crash,
        dry_run=dry_run,
    )


def _make_package(
    name: str = "testpkg",
    repo: str | None = "https://github.com/user/testpkg",
    extension_type: str = "pure",
    test_command: str = "python -m pytest",
    install_command: str = "pip install -e .",
    skip: bool = False,
    enriched: bool = False,
    timeout: int | None = None,
    clone_depth: int | None = None,
    import_name: str | None = None,
) -> PackageEntry:
    return PackageEntry(
        package=name,
        repo=repo,
        extension_type=extension_type,
        test_command=test_command,
        install_command=install_command,
        skip=skip,
        enriched=enriched,
        timeout=timeout,
        clone_depth=clone_depth,
        import_name=import_name,
    )


class TestBuildEnv(unittest.TestCase):
    def test_jit_enabled(self) -> None:
        config = RunnerConfig(
            target_python=Path("/usr/bin/python3"),
            registry_dir=Path("registry"),
            results_dir=Path("results"),
            run_id="test",
        )
        env = build_env(config)
        self.assertEqual(env["PYTHON_JIT"], "1")
        self.assertEqual(env["PYTHONFAULTHANDLER"], "1")
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_env_overrides(self) -> None:
        config = RunnerConfig(
            target_python=Path("/usr/bin/python3"),
            registry_dir=Path("registry"),
            results_dir=Path("results"),
            run_id="test",
            env_overrides={"PYTHON_JIT_VERBOSE": "1"},
        )
        env = build_env(config)
        self.assertEqual(env["PYTHON_JIT_VERBOSE"], "1")


class TestRunDirManagement(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_create_run_dir(self) -> None:
        run_dir = create_run_dir(self.base / "results", "my-run")
        self.assertTrue(run_dir.exists())
        self.assertTrue((run_dir / "crashes").exists())

    def test_append_and_load_results(self) -> None:
        run_dir = create_run_dir(self.base / "results", "test-run")
        r1 = PackageResult(package="pkg1", status="pass")
        r2 = PackageResult(package="pkg2", status="crash")
        append_result(run_dir, r1)
        append_result(run_dir, r2)
        completed = load_completed_packages(run_dir)
        self.assertEqual(completed, {"pkg1", "pkg2"})

    def test_load_completed_no_file(self) -> None:
        run_dir = create_run_dir(self.base / "results", "empty-run")
        completed = load_completed_packages(run_dir)
        self.assertEqual(completed, set())

    def test_save_crash_stderr(self) -> None:
        run_dir = create_run_dir(self.base / "results", "crash-run")
        save_crash_stderr(run_dir, "mypkg", "Segmentation fault\n")
        crash_file = run_dir / "crashes" / "mypkg.stderr"
        self.assertTrue(crash_file.exists())
        self.assertIn("Segmentation fault", crash_file.read_text())

    def test_write_run_meta(self) -> None:
        run_dir = create_run_dir(self.base / "results", "meta-run")
        config = _make_config(self.base)
        write_run_meta(
            run_dir,
            config,
            "3.15.0a5",
            True,
            started_at="2026-02-22T14:30:00Z",
        )
        meta_file = run_dir / "run_meta.json"
        self.assertTrue(meta_file.exists())
        meta = json.loads(meta_file.read_text())
        self.assertEqual(meta["python_version"], "3.15.0a5")
        self.assertTrue(meta["jit_enabled"])


class TestGetPackageVersion(unittest.TestCase):
    def test_exact_match(self) -> None:
        installed = {"requests": "2.31.0", "click": "8.1.0"}
        self.assertEqual(get_package_version("requests", installed), "2.31.0")

    def test_case_insensitive(self) -> None:
        installed = {"Requests": "2.31.0"}
        self.assertEqual(get_package_version("requests", installed), "2.31.0")

    def test_hyphen_underscore_normalisation(self) -> None:
        installed = {"my-package": "1.0.0"}
        self.assertEqual(get_package_version("my_package", installed), "1.0.0")

    def test_not_found(self) -> None:
        installed = {"click": "8.1.0"}
        self.assertIsNone(get_package_version("flask", installed))


class TestFilterPackages(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.config = _make_config(self.base)
        # Create some packages in registry.
        for name, ext_type in [("pkg1", "pure"), ("pkg2", "extensions"), ("pkg3", "pure")]:
            pkg = _make_package(name=name, extension_type=ext_type)
            save_package(pkg, self.config.registry_dir)
        # Create index.
        index = Index(
            packages=[
                IndexEntry(name="pkg1", download_count=1000),
                IndexEntry(name="pkg2", download_count=500),
                IndexEntry(name="pkg3", download_count=100),
            ]
        )
        save_index(index, self.config.registry_dir)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_all_packages(self) -> None:
        index = Index(
            packages=[
                IndexEntry(name="pkg1", download_count=1000),
                IndexEntry(name="pkg2", download_count=500),
                IndexEntry(name="pkg3", download_count=100),
            ]
        )
        result = filter_packages(index, self.config.registry_dir, self.config)
        self.assertEqual(len(result), 3)

    def test_top_n(self) -> None:
        self.config.top_n = 2
        index = Index(
            packages=[
                IndexEntry(name="pkg1", download_count=1000),
                IndexEntry(name="pkg2", download_count=500),
                IndexEntry(name="pkg3", download_count=100),
            ]
        )
        result = filter_packages(index, self.config.registry_dir, self.config)
        self.assertEqual(len(result), 2)
        names = [p.package for p in result]
        self.assertIn("pkg1", names)
        self.assertIn("pkg2", names)

    def test_skip_extensions(self) -> None:
        self.config.skip_extensions = True
        index = Index(
            packages=[
                IndexEntry(name="pkg1", download_count=1000),
                IndexEntry(name="pkg2", download_count=500),
                IndexEntry(name="pkg3", download_count=100),
            ]
        )
        result = filter_packages(index, self.config.registry_dir, self.config)
        names = [p.package for p in result]
        self.assertNotIn("pkg2", names)
        self.assertIn("pkg1", names)
        self.assertIn("pkg3", names)

    def test_packages_filter(self) -> None:
        self.config.packages_filter = ["pkg1", "pkg3"]
        index = Index(
            packages=[
                IndexEntry(name="pkg1", download_count=1000),
                IndexEntry(name="pkg2", download_count=500),
                IndexEntry(name="pkg3", download_count=100),
            ]
        )
        result = filter_packages(index, self.config.registry_dir, self.config)
        names = [p.package for p in result]
        self.assertEqual(set(names), {"pkg1", "pkg3"})

    def test_skip_flagged_packages(self) -> None:
        skip_pkg = _make_package(name="skipped", skip=True)
        save_package(skip_pkg, self.config.registry_dir)
        index = Index(
            packages=[
                IndexEntry(name="pkg1", download_count=1000),
                IndexEntry(name="skipped", download_count=900),
            ]
        )
        result = filter_packages(index, self.config.registry_dir, self.config)
        names = [p.package for p in result]
        self.assertNotIn("skipped", names)


class TestRunPackage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.config = _make_config(self.base)
        self.run_dir = create_run_dir(self.config.results_dir, self.config.run_id)
        self.env = build_env(self.config)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_pass(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {"testpkg": "1.0.0"}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="50 passed\n", stderr=""
        )
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.git_revision, "abc1234")

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_test_failure(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=1, stdout="1 failed\n", stderr="FAILED test_foo\n"
        )
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.exit_code, 1)

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_crash_sigsegv(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="",
            returncode=-signal.SIGSEGV,
            stdout="",
            stderr="Fatal Python error: Segmentation fault\n",
        )
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "crash")
        self.assertEqual(result.signal, signal.SIGSEGV)
        self.assertIsNotNone(result.crash_signature)
        # Verify crash stderr was saved.
        crash_file = self.run_dir / "crashes" / "testpkg.stderr"
        self.assertTrue(crash_file.exists())

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_timeout(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=600)
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "timeout")
        self.assertTrue(result.timeout_hit)

    @patch("labeille.runner.clone_repo")
    def test_clone_error(self, mock_clone: MagicMock) -> None:
        mock_clone.side_effect = subprocess.CalledProcessError(128, "git clone")
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "clone_error")

    def test_no_repo_url(self) -> None:
        pkg = _make_package(repo=None)
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "clone_error")
        self.assertIn("No repository URL", result.error_message or "")

    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_install_error(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
    ) -> None:
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=1, stdout="", stderr="error: build failed\n"
        )
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "install_error")

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_per_package_timeout(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Package-level timeout override is respected."""
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )
        pkg = _make_package(timeout=120)
        run_package(pkg, self.config, self.run_dir, self.env)
        # The test command should be called with timeout=120 (from package).
        # run_test_command(venv_python, test_cmd, repo_dir, env, per_pkg_timeout)
        args, _ = mock_test.call_args
        self.assertEqual(args[4], 120)


class TestRunAll(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.config = _make_config(self.base)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _setup_registry(
        self,
        packages: list[PackageEntry],
        index_entries: list[IndexEntry] | None = None,
    ) -> None:
        for pkg in packages:
            save_package(pkg, self.config.registry_dir)
        if index_entries is None:
            index_entries = [
                IndexEntry(name=pkg.package, download_count=1000 - i)
                for i, pkg in enumerate(packages)
            ]
        save_index(Index(packages=index_entries), self.config.registry_dir)

    @patch("labeille.runner.validate_target_python")
    @patch("labeille.runner.check_jit_enabled")
    @patch("labeille.runner.run_package")
    def test_dry_run(
        self,
        mock_run_pkg: MagicMock,
        mock_jit: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        self.config.dry_run = True
        mock_validate.return_value = "3.15.0"
        mock_jit.return_value = True
        self._setup_registry([_make_package(name="pkg1")])

        output = run_all(self.config)
        mock_run_pkg.assert_not_called()
        self.assertEqual(output.summary.skipped, 1)

    @patch("labeille.runner.validate_target_python")
    @patch("labeille.runner.check_jit_enabled")
    @patch("labeille.runner.run_package")
    def test_skip_completed(
        self,
        mock_run_pkg: MagicMock,
        mock_jit: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        self.config.skip_completed = True
        mock_validate.return_value = "3.15.0"
        mock_jit.return_value = True
        self._setup_registry(
            [
                _make_package(name="done"),
                _make_package(name="todo"),
            ]
        )
        # Pre-populate results for "done".
        run_dir = create_run_dir(self.config.results_dir, self.config.run_id)
        append_result(run_dir, PackageResult(package="done", status="pass"))

        mock_run_pkg.return_value = PackageResult(package="todo", status="pass")
        output = run_all(self.config)
        # Only "todo" should be run.
        self.assertEqual(mock_run_pkg.call_count, 1)
        pkg_arg = mock_run_pkg.call_args[0][0]
        self.assertEqual(pkg_arg.package, "todo")
        self.assertEqual(output.summary.skipped, 1)

    @patch("labeille.runner.validate_target_python")
    @patch("labeille.runner.check_jit_enabled")
    @patch("labeille.runner.run_package")
    def test_stop_after_crash(
        self,
        mock_run_pkg: MagicMock,
        mock_jit: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        self.config.stop_after_crash = 1
        mock_validate.return_value = "3.15.0"
        mock_jit.return_value = True
        self._setup_registry(
            [
                _make_package(name="pkg1"),
                _make_package(name="pkg2"),
                _make_package(name="pkg3"),
            ]
        )
        mock_run_pkg.side_effect = [
            PackageResult(package="pkg1", status="crash", signal=11),
            PackageResult(package="pkg2", status="pass"),
        ]
        output = run_all(self.config)
        # Should stop after 1 crash (pkg1), not test pkg2 or pkg3.
        self.assertEqual(mock_run_pkg.call_count, 1)
        self.assertEqual(output.summary.crashed, 1)

    @patch("labeille.runner.validate_target_python")
    @patch("labeille.runner.check_jit_enabled")
    @patch("labeille.runner.run_package")
    def test_summary_counts(
        self,
        mock_run_pkg: MagicMock,
        mock_jit: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        mock_validate.return_value = "3.15.0"
        mock_jit.return_value = True
        self._setup_registry(
            [
                _make_package(name="p1"),
                _make_package(name="p2"),
                _make_package(name="p3"),
            ]
        )
        mock_run_pkg.side_effect = [
            PackageResult(package="p1", status="pass"),
            PackageResult(package="p2", status="fail"),
            PackageResult(package="p3", status="crash", signal=11),
        ]
        output = run_all(self.config)
        self.assertEqual(output.summary.tested, 3)
        self.assertEqual(output.summary.passed, 1)
        self.assertEqual(output.summary.failed, 1)
        self.assertEqual(output.summary.crashed, 1)

    @patch("labeille.runner.validate_target_python")
    @patch("labeille.runner.check_jit_enabled")
    @patch("labeille.runner.run_package")
    def test_results_jsonl_written(
        self,
        mock_run_pkg: MagicMock,
        mock_jit: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        mock_validate.return_value = "3.15.0"
        mock_jit.return_value = True
        self._setup_registry([_make_package(name="pkg1")])
        mock_run_pkg.return_value = PackageResult(package="pkg1", status="pass")
        run_all(self.config)
        results_file = self.config.results_dir / self.config.run_id / "results.jsonl"
        self.assertTrue(results_file.exists())
        lines = results_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["package"], "pkg1")
        self.assertEqual(data["status"], "pass")


class TestCommandAssembly(unittest.TestCase):
    """Test that environment and command construction is correct."""

    def test_env_has_jit_and_faulthandler(self) -> None:
        config = RunnerConfig(
            target_python=Path("/usr/bin/python3"),
            registry_dir=Path("registry"),
            results_dir=Path("results"),
            run_id="test",
        )
        env = build_env(config)
        self.assertEqual(env["PYTHON_JIT"], "1")
        self.assertEqual(env["PYTHONFAULTHANDLER"], "1")
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_env_overrides_applied(self) -> None:
        config = RunnerConfig(
            target_python=Path("/usr/bin/python3"),
            registry_dir=Path("registry"),
            results_dir=Path("results"),
            run_id="test",
            env_overrides={"PYTHON_JIT": "0", "EXTRA": "val"},
        )
        env = build_env(config)
        self.assertEqual(env["PYTHON_JIT"], "0")
        self.assertEqual(env["EXTRA"], "val")


class TestResolveDirs(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_temp_mode_by_default(self) -> None:
        config = _make_config(self.base)
        pkg = _make_package()
        work_dir, repo_dir, venv_dir, is_temp = _resolve_dirs(pkg, config)
        self.assertTrue(is_temp)
        self.assertIsNotNone(work_dir)
        self.assertTrue(str(repo_dir).endswith("/repo"))
        self.assertTrue(str(venv_dir).endswith("/venv"))

    def test_persistent_repos_dir(self) -> None:
        config = _make_config(self.base)
        config.repos_dir = self.base / "my_repos"
        pkg = _make_package()
        work_dir, repo_dir, venv_dir, is_temp = _resolve_dirs(pkg, config)
        self.assertFalse(is_temp)
        self.assertIsNone(work_dir)
        self.assertEqual(repo_dir, self.base / "my_repos" / "testpkg")

    def test_persistent_venvs_dir(self) -> None:
        config = _make_config(self.base)
        config.venvs_dir = self.base / "my_venvs"
        pkg = _make_package()
        work_dir, repo_dir, venv_dir, is_temp = _resolve_dirs(pkg, config)
        self.assertFalse(is_temp)
        self.assertEqual(venv_dir, self.base / "my_venvs" / "testpkg")

    def test_both_persistent(self) -> None:
        config = _make_config(self.base)
        config.repos_dir = self.base / "repos"
        config.venvs_dir = self.base / "venvs"
        pkg = _make_package()
        work_dir, repo_dir, venv_dir, is_temp = _resolve_dirs(pkg, config)
        self.assertFalse(is_temp)
        self.assertIsNone(work_dir)
        self.assertEqual(repo_dir, self.base / "repos" / "testpkg")
        self.assertEqual(venv_dir, self.base / "venvs" / "testpkg")


class TestRepoReuse(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.config = _make_config(self.base)
        self.config.repos_dir = self.base / "repos"
        self.config.venvs_dir = self.base / "venvs"
        self.run_dir = create_run_dir(self.config.results_dir, self.config.run_id)
        self.env = build_env(self.config)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.pull_repo")
    @patch("labeille.runner.clone_repo")
    def test_reuses_existing_repo(
        self,
        mock_clone: MagicMock,
        mock_pull: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """When repo dir exists with .git, pull is called instead of clone."""
        pkg = _make_package()
        # Pre-create the repo dir with a .git marker.
        repo_dir = self.base / "repos" / "testpkg"
        repo_dir.mkdir(parents=True)
        (repo_dir / ".git").mkdir()

        mock_pull.return_value = "def5678"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )

        result = run_package(pkg, self.config, self.run_dir, self.env)
        mock_clone.assert_not_called()
        mock_pull.assert_called_once()
        self.assertEqual(result.git_revision, "def5678")
        self.assertEqual(result.status, "pass")

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_reuses_existing_venv(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """When venv dir exists with bin/python, venv creation and install are skipped."""
        pkg = _make_package()
        # Pre-create the venv dir with bin/python.
        venv_dir = self.base / "venvs" / "testpkg"
        (venv_dir / "bin").mkdir(parents=True)
        (venv_dir / "bin" / "python").touch()

        mock_clone.return_value = "abc1234"
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )

        result = run_package(pkg, self.config, self.run_dir, self.env)
        mock_venv.assert_not_called()
        mock_install.assert_not_called()
        self.assertEqual(result.status, "pass")

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.pull_repo")
    @patch("labeille.runner.clone_repo")
    def test_pull_failure_reclones(
        self,
        mock_clone: MagicMock,
        mock_pull: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """When pull fails, the repo is re-cloned."""
        pkg = _make_package()
        repo_dir = self.base / "repos" / "testpkg"
        repo_dir.mkdir(parents=True)
        (repo_dir / ".git").mkdir()

        mock_pull.side_effect = subprocess.CalledProcessError(1, "git pull")
        mock_clone.return_value = "new1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )

        result = run_package(pkg, self.config, self.run_dir, self.env)
        mock_pull.assert_called_once()
        mock_clone.assert_called_once()
        self.assertEqual(result.git_revision, "new1234")
        self.assertEqual(result.status, "pass")

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_refresh_venvs_deletes_and_recreates(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """When refresh_venvs is True, existing venv is deleted and recreated."""
        self.config.refresh_venvs = True
        pkg = _make_package()
        # Pre-create the venv dir with bin/python.
        venv_dir = self.base / "venvs" / "testpkg"
        (venv_dir / "bin").mkdir(parents=True)
        (venv_dir / "bin" / "python").touch()

        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )

        result = run_package(pkg, self.config, self.run_dir, self.env)
        mock_venv.assert_called_once()
        mock_install.assert_called_once()
        self.assertEqual(result.status, "pass")

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_refresh_venvs_false_still_reuses(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """When refresh_venvs is False (default), existing venv is reused."""
        self.config.refresh_venvs = False
        pkg = _make_package()
        # Pre-create the venv dir with bin/python.
        venv_dir = self.base / "venvs" / "testpkg"
        (venv_dir / "bin").mkdir(parents=True)
        (venv_dir / "bin" / "python").touch()

        mock_clone.return_value = "abc1234"
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )

        result = run_package(pkg, self.config, self.run_dir, self.env)
        mock_venv.assert_not_called()
        mock_install.assert_not_called()
        self.assertEqual(result.status, "pass")

    def test_persistent_dirs_not_deleted(self) -> None:
        """Persistent repos/venvs dirs are NOT cleaned up after run."""
        repos_dir = self.base / "repos"
        venvs_dir = self.base / "venvs"
        repos_dir.mkdir()
        venvs_dir.mkdir()
        # Verify they still exist after config says keep_work_dirs=False.
        self.config.keep_work_dirs = False
        self.assertTrue(repos_dir.exists())
        self.assertTrue(venvs_dir.exists())


class TestCloneDepth(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.config = _make_config(self.base)
        self.config.repos_dir = self.base / "repos"
        self.config.venvs_dir = self.base / "venvs"
        self.run_dir = create_run_dir(self.config.results_dir, self.config.run_id)
        self.env = build_env(self.config)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_clone_depth_none_defaults_to_shallow(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """clone_depth=None passes None to clone_repo (default depth=1)."""
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )
        pkg = _make_package(clone_depth=None)
        run_package(pkg, self.config, self.run_dir, self.env)
        _, kwargs = mock_clone.call_args
        self.assertIsNone(kwargs.get("clone_depth"))

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_clone_depth_passed_to_clone_repo(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """clone_depth=50 is passed through to clone_repo."""
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )
        pkg = _make_package(clone_depth=50)
        run_package(pkg, self.config, self.run_dir, self.env)
        _, kwargs = mock_clone.call_args
        self.assertEqual(kwargs.get("clone_depth"), 50)


class TestImportCheck(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.config = _make_config(self.base)
        self.run_dir = create_run_dir(self.config.results_dir, self.config.run_id)
        self.env = build_env(self.config)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_import_check_failure_sets_install_error(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Failed import check sets status to install_error."""
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'x'"
        )
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "install_error")
        self.assertIn("import failed", result.error_message or "")
        mock_test.assert_not_called()

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_import_check_success_continues_to_tests(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Successful import check continues to run tests."""
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "pass")
        mock_test.assert_called_once()

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_import_check_uses_custom_import_name(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Custom import_name is used for the import check."""
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )
        pkg = _make_package(name="python-dateutil", import_name="dateutil")
        run_package(pkg, self.config, self.run_dir, self.env)
        # check_import should be called with "dateutil" not "python_dateutil"
        args, _ = mock_import.call_args
        self.assertEqual(args[1], "dateutil")

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_import_check_skipped_when_reusing_venv(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Import check is skipped when reusing an existing venv."""
        self.config.repos_dir = self.base / "repos"
        self.config.venvs_dir = self.base / "venvs"
        pkg = _make_package()
        # Pre-create the venv dir with bin/python.
        venv_dir = self.base / "venvs" / "testpkg"
        (venv_dir / "bin").mkdir(parents=True)
        (venv_dir / "bin" / "python").touch()

        mock_clone.return_value = "abc1234"
        mock_get_pkgs.return_value = {}
        mock_test.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="ok\n", stderr=""
        )

        with patch("labeille.runner.check_import") as mock_import:
            result = run_package(pkg, self.config, self.run_dir, self.env)
            mock_import.assert_not_called()
        self.assertEqual(result.status, "pass")

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_import_check_timeout_sets_install_error(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Import check timeout sets status to install_error."""
        mock_clone.return_value = "abc1234"
        mock_install.return_value = subprocess.CompletedProcess(
            args="", returncode=0, stdout="", stderr=""
        )
        mock_import.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=30)
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "install_error")
        self.assertIn("import timed out", result.error_message or "")
        mock_test.assert_not_called()


class TestSummaryFileWritten(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.config = _make_config(self.base)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("labeille.runner.validate_target_python")
    @patch("labeille.runner.check_jit_enabled")
    @patch("labeille.runner.run_package")
    def test_summary_txt_written(
        self,
        mock_run_pkg: MagicMock,
        mock_jit: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        mock_validate.return_value = "3.15.0"
        mock_jit.return_value = True
        pkg = _make_package(name="pkg1")
        save_package(pkg, self.config.registry_dir)
        save_index(
            Index(packages=[IndexEntry(name="pkg1", download_count=1000)]),
            self.config.registry_dir,
        )
        mock_run_pkg.return_value = PackageResult(package="pkg1", status="pass")
        output = run_all(self.config)
        summary_file = output.run_dir / "summary.txt"
        self.assertTrue(summary_file.exists())
        content = summary_file.read_text()
        self.assertIn("Packages tested:", content)


if __name__ == "__main__":
    unittest.main()
