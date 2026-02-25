"""Tests for labeille.runner."""

import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
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
    _clean_env,
    _resolve_dirs,
    _run_in_process_group,
    append_result,
    build_env,
    check_jit_enabled,
    checkout_revision,
    create_run_dir,
    extract_python_minor_version,
    filter_packages,
    get_package_version,
    load_completed_packages,
    parse_package_specs,
    pull_repo,
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
    force_run: bool = False,
    target_python_version: str = "",
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
        force_run=force_run,
        target_python_version=target_python_version,
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
    skip_versions: dict[str, str] | None = None,
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
        skip_versions=skip_versions or {},
    )


class TestCleanEnv(unittest.TestCase):
    @patch.dict("os.environ", {"PYTHONHOME": "/bad", "PYTHONPATH": "/also/bad", "HOME": "/ok"})
    def test_strips_python_vars(self) -> None:
        env = _clean_env()
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["HOME"], "/ok")

    @patch.dict("os.environ", {"HOME": "/ok"}, clear=True)
    def test_overrides_applied(self) -> None:
        env = _clean_env(PYTHON_JIT="1", ASAN_OPTIONS="detect_leaks=0")
        self.assertEqual(env["PYTHON_JIT"], "1")
        self.assertEqual(env["ASAN_OPTIONS"], "detect_leaks=0")

    @patch.dict("os.environ", {"PYTHONHOME": "/bad"}, clear=True)
    def test_overrides_after_strip(self) -> None:
        """Overrides don't re-add stripped vars unless explicitly passed."""
        env = _clean_env()
        self.assertNotIn("PYTHONHOME", env)

    @patch.dict("os.environ", {}, clear=True)
    def test_no_error_when_vars_absent(self) -> None:
        """Works fine when PYTHONHOME/PYTHONPATH aren't set at all."""
        env = _clean_env(FOO="bar")
        self.assertEqual(env["FOO"], "bar")


class TestBuildEnvStripsVars(unittest.TestCase):
    @patch.dict("os.environ", {"PYTHONHOME": "/bad", "PYTHONPATH": "/bad"})
    def test_build_env_strips_python_vars(self) -> None:
        config = RunnerConfig(
            target_python=Path("/usr/bin/python3"),
            registry_dir=Path("registry"),
            results_dir=Path("results"),
            run_id="test",
        )
        env = build_env(config)
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["PYTHON_JIT"], "1")


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


class TestCheckJitEnabled(unittest.TestCase):
    @patch("labeille.runner.subprocess.run")
    def test_returns_true_when_jit_available(self, mock_run: Any) -> None:
        mock_run.return_value = MagicMock(stdout="True\n")
        self.assertTrue(check_jit_enabled(Path("/usr/bin/python3")))

    @patch("labeille.runner.subprocess.run")
    def test_returns_false_when_no_jit(self, mock_run: Any) -> None:
        mock_run.return_value = MagicMock(stdout="False\n")
        self.assertFalse(check_jit_enabled(Path("/usr/bin/python3")))

    @patch("labeille.runner.subprocess.run")
    def test_returns_false_on_timeout(self, mock_run: Any) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired("python", 30)
        self.assertFalse(check_jit_enabled(Path("/usr/bin/python3")))

    @patch("labeille.runner.subprocess.run")
    def test_returns_false_on_file_not_found(self, mock_run: Any) -> None:
        mock_run.side_effect = FileNotFoundError()
        self.assertFalse(check_jit_enabled(Path("/nonexistent/python")))

    @patch("labeille.runner.subprocess.run")
    def test_exact_match_not_substring(self, mock_run: Any) -> None:
        """'TrueColor' in stdout should not trigger a True result."""
        mock_run.return_value = MagicMock(stdout="TrueColor\n")
        self.assertFalse(check_jit_enabled(Path("/usr/bin/python3")))

    @patch("labeille.runner.subprocess.run")
    def test_returns_false_on_empty_stdout(self, mock_run: Any) -> None:
        mock_run.return_value = MagicMock(stdout="")
        self.assertFalse(check_jit_enabled(Path("/usr/bin/python3")))


class TestRunInProcessGroup(unittest.TestCase):
    @patch("labeille.runner.subprocess.Popen")
    def test_success_returns_completed_process(self, mock_popen: Any) -> None:
        proc = MagicMock()
        proc.communicate.return_value = ("stdout output", "stderr output")
        proc.returncode = 0
        proc.args = "echo hello"
        mock_popen.return_value = proc

        result = _run_in_process_group("echo hello", cwd="/tmp", env={}, timeout=30)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "stdout output")
        self.assertEqual(result.stderr, "stderr output")
        mock_popen.assert_called_once()
        # Verify start_new_session=True is passed.
        _, kwargs = mock_popen.call_args
        self.assertTrue(kwargs["start_new_session"])

    @patch("labeille.runner.os.killpg")
    @patch("labeille.runner.os.getpgid")
    @patch("labeille.runner.subprocess.Popen")
    def test_timeout_kills_process_group(
        self, mock_popen: Any, mock_getpgid: Any, mock_killpg: Any
    ) -> None:
        proc = MagicMock()
        proc.pid = 12345
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired("cmd", 30),
            ("partial out", "partial err"),
        ]
        mock_popen.return_value = proc
        mock_getpgid.return_value = 12345

        with self.assertRaises(subprocess.TimeoutExpired) as cm:
            _run_in_process_group("sleep 999", cwd="/tmp", env={}, timeout=30)

        mock_getpgid.assert_called_once_with(12345)
        mock_killpg.assert_called_once_with(12345, signal.SIGKILL)
        exc = cm.exception
        self.assertEqual(exc.output, "partial out")
        self.assertEqual(exc.stderr, "partial err")

    @patch("labeille.runner.os.killpg")
    @patch("labeille.runner.os.getpgid")
    @patch("labeille.runner.subprocess.Popen")
    def test_timeout_process_already_exited(
        self, mock_popen: Any, mock_getpgid: Any, mock_killpg: Any
    ) -> None:
        """ProcessLookupError from killpg is handled gracefully."""
        proc = MagicMock()
        proc.pid = 99999
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired("cmd", 30),
            ("", ""),
        ]
        mock_popen.return_value = proc
        mock_getpgid.return_value = 99999
        mock_killpg.side_effect = ProcessLookupError()

        with self.assertRaises(subprocess.TimeoutExpired):
            _run_in_process_group("cmd", cwd="/tmp", env={}, timeout=30)

    @patch("labeille.runner.os.killpg")
    @patch("labeille.runner.os.getpgid")
    @patch("labeille.runner.subprocess.Popen")
    def test_timeout_partial_output_in_exception(
        self, mock_popen: Any, mock_getpgid: Any, mock_killpg: Any
    ) -> None:
        """TimeoutExpired exception carries partial stdout/stderr."""
        proc = MagicMock()
        proc.pid = 111
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired("cmd", 10),
            ("partial stdout", "partial stderr"),
        ]
        mock_popen.return_value = proc
        mock_getpgid.return_value = 111

        with self.assertRaises(subprocess.TimeoutExpired) as cm:
            _run_in_process_group("cmd", cwd="/tmp", env={}, timeout=10)

        self.assertEqual(cm.exception.output, "partial stdout")
        self.assertEqual(cm.exception.stderr, "partial stderr")

    @patch("labeille.runner.os.killpg")
    @patch("labeille.runner.os.getpgid")
    @patch("labeille.runner.subprocess.Popen")
    def test_timeout_second_communicate_also_times_out(
        self, mock_popen: Any, mock_getpgid: Any, mock_killpg: Any
    ) -> None:
        """If second communicate also times out, falls back to proc.kill()."""
        proc = MagicMock()
        proc.pid = 222
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired("cmd", 30),
            subprocess.TimeoutExpired("cmd", 5),
            ("", "fallback stderr"),
        ]
        mock_popen.return_value = proc
        mock_getpgid.return_value = 222

        with self.assertRaises(subprocess.TimeoutExpired):
            _run_in_process_group("cmd", cwd="/tmp", env={}, timeout=30)

        proc.kill.assert_called_once()


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
        result, ver_skipped = filter_packages(index, self.config.registry_dir, self.config)
        self.assertEqual(len(result), 3)
        self.assertEqual(ver_skipped, 0)

    def test_top_n(self) -> None:
        self.config.top_n = 2
        index = Index(
            packages=[
                IndexEntry(name="pkg1", download_count=1000),
                IndexEntry(name="pkg2", download_count=500),
                IndexEntry(name="pkg3", download_count=100),
            ]
        )
        result, _ = filter_packages(index, self.config.registry_dir, self.config)
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
        result, _ = filter_packages(index, self.config.registry_dir, self.config)
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
        result, _ = filter_packages(index, self.config.registry_dir, self.config)
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
        result, _ = filter_packages(index, self.config.registry_dir, self.config)
        names = [p.package for p in result]
        self.assertNotIn("skipped", names)

    def test_skip_versions_filters_matching_version(self) -> None:
        """Packages with skip_versions matching target are filtered out."""
        self.config.target_python_version = "3.15"
        ver_pkg = _make_package(name="verpkg", skip_versions={"3.15": "PyO3 not supported"})
        save_package(ver_pkg, self.config.registry_dir)
        index = Index(
            packages=[
                IndexEntry(name="pkg1", download_count=1000),
                IndexEntry(name="verpkg", download_count=900),
            ]
        )
        result, ver_skipped = filter_packages(index, self.config.registry_dir, self.config)
        names = [p.package for p in result]
        self.assertNotIn("verpkg", names)
        self.assertIn("pkg1", names)
        self.assertEqual(ver_skipped, 1)

    def test_skip_versions_allows_non_matching_version(self) -> None:
        """Packages with skip_versions for other versions are NOT filtered."""
        self.config.target_python_version = "3.14"
        ver_pkg = _make_package(name="verpkg", skip_versions={"3.15": "PyO3 not supported"})
        save_package(ver_pkg, self.config.registry_dir)
        index = Index(
            packages=[
                IndexEntry(name="verpkg", download_count=900),
            ]
        )
        result, ver_skipped = filter_packages(index, self.config.registry_dir, self.config)
        names = [p.package for p in result]
        self.assertIn("verpkg", names)
        self.assertEqual(ver_skipped, 0)

    def test_force_run_overrides_skip(self) -> None:
        """--force-run overrides both skip and skip_versions."""
        self.config.force_run = True
        self.config.target_python_version = "3.15"
        skip_pkg = _make_package(name="skipped", skip=True)
        ver_pkg = _make_package(name="verpkg", skip_versions={"3.15": "broken"})
        save_package(skip_pkg, self.config.registry_dir)
        save_package(ver_pkg, self.config.registry_dir)
        index = Index(
            packages=[
                IndexEntry(name="pkg1", download_count=1000),
                IndexEntry(name="skipped", download_count=900),
                IndexEntry(name="verpkg", download_count=800),
            ]
        )
        result, ver_skipped = filter_packages(index, self.config.registry_dir, self.config)
        names = [p.package for p in result]
        self.assertIn("skipped", names)
        self.assertIn("verpkg", names)
        self.assertIn("pkg1", names)
        self.assertEqual(ver_skipped, 0)

    def test_skip_versions_no_target_version_set(self) -> None:
        """When target_python_version is empty, skip_versions is not checked."""
        self.config.target_python_version = ""
        ver_pkg = _make_package(name="verpkg", skip_versions={"3.15": "broken"})
        save_package(ver_pkg, self.config.registry_dir)
        index = Index(
            packages=[
                IndexEntry(name="verpkg", download_count=900),
            ]
        )
        result, ver_skipped = filter_packages(index, self.config.registry_dir, self.config)
        names = [p.package for p in result]
        self.assertIn("verpkg", names)
        self.assertEqual(ver_skipped, 0)

    @patch("labeille.runner.load_package")
    def test_filter_uses_index_skip_versions_keys(self, mock_load: Any) -> None:
        """Index skip_versions_keys allows skipping without loading YAML."""
        self.config.target_python_version = "3.15"
        # pkg1 has no version skip → should be loaded.
        # indexpkg has skip_versions_keys=["3.15"] → should be skipped via index.
        pkg1 = _make_package(name="pkg1")
        mock_load.return_value = pkg1
        index = Index(
            packages=[
                IndexEntry(name="pkg1", download_count=1000),
                IndexEntry(
                    name="indexpkg",
                    download_count=900,
                    skip_versions_keys=["3.15"],
                ),
            ]
        )
        result, ver_skipped = filter_packages(index, self.config.registry_dir, self.config)
        # indexpkg should be skipped at the index level.
        names = [p.package for p in result]
        self.assertNotIn("indexpkg", names)
        self.assertEqual(ver_skipped, 1)
        # load_package should NOT be called for indexpkg.
        loaded_names = [call.args[0] for call in mock_load.call_args_list]
        self.assertNotIn("indexpkg", loaded_names)

    def test_force_run_overrides_index_skip_versions(self) -> None:
        """--force-run bypasses index-level skip_versions_keys filtering."""
        self.config.force_run = True
        self.config.target_python_version = "3.15"
        ver_pkg = _make_package(name="verpkg", skip_versions={"3.15": "broken"})
        save_package(ver_pkg, self.config.registry_dir)
        index = Index(
            packages=[
                IndexEntry(
                    name="verpkg",
                    download_count=900,
                    skip_versions_keys=["3.15"],
                ),
            ]
        )
        result, ver_skipped = filter_packages(index, self.config.registry_dir, self.config)
        names = [p.package for p in result]
        self.assertIn("verpkg", names)
        self.assertEqual(ver_skipped, 0)


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


class TestParallelExecution(unittest.TestCase):
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
    def test_parallel_collects_all_results(
        self,
        mock_run_pkg: MagicMock,
        mock_jit: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """Parallel mode collects results from all packages."""
        self.config.workers = 2
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
            PackageResult(package="p2", status="fail", exit_code=1),
            PackageResult(package="p3", status="pass"),
        ]
        output = run_all(self.config)
        self.assertEqual(output.summary.tested, 3)
        self.assertEqual(output.summary.passed, 2)
        self.assertEqual(output.summary.failed, 1)

    @patch("labeille.runner.validate_target_python")
    @patch("labeille.runner.check_jit_enabled")
    @patch("labeille.runner.run_package")
    def test_parallel_stop_after_crash(
        self,
        mock_run_pkg: MagicMock,
        mock_jit: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """In parallel mode, stop-after-crash cancels remaining packages."""
        self.config.workers = 2
        self.config.stop_after_crash = 1
        mock_validate.return_value = "3.15.0"
        mock_jit.return_value = True
        self._setup_registry(
            [
                _make_package(name="p1"),
                _make_package(name="p2"),
                _make_package(name="p3"),
            ]
        )

        def _side_effect(pkg: PackageEntry, *args: object, **kwargs: object) -> PackageResult:
            if pkg.package == "p1":
                return PackageResult(package="p1", status="crash", signal=11)
            # p2 and p3 should either run normally or be cancelled.
            return PackageResult(package=pkg.package, status="pass")

        mock_run_pkg.side_effect = _side_effect
        output = run_all(self.config)
        self.assertGreaterEqual(output.summary.crashed, 1)
        # All packages have some result (either tested or cancelled).
        self.assertEqual(len(output.results), 3)

    @patch("labeille.runner.validate_target_python")
    @patch("labeille.runner.check_jit_enabled")
    @patch("labeille.runner.run_package")
    def test_workers_1_uses_sequential(
        self,
        mock_run_pkg: MagicMock,
        mock_jit: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """workers=1 produces identical results to default sequential mode."""
        self.config.workers = 1
        mock_validate.return_value = "3.15.0"
        mock_jit.return_value = True
        self._setup_registry(
            [
                _make_package(name="p1"),
                _make_package(name="p2"),
            ]
        )
        mock_run_pkg.side_effect = [
            PackageResult(package="p1", status="pass"),
            PackageResult(package="p2", status="fail", exit_code=1),
        ]
        output = run_all(self.config)
        self.assertEqual(output.summary.tested, 2)
        self.assertEqual(output.summary.passed, 1)
        self.assertEqual(output.summary.failed, 1)
        # Verify sequential: results should be in order.
        self.assertEqual(output.results[0].package, "p1")
        self.assertEqual(output.results[1].package, "p2")

    @patch("labeille.runner.validate_target_python")
    @patch("labeille.runner.check_jit_enabled")
    @patch("labeille.runner.run_package")
    def test_parallel_jsonl_written(
        self,
        mock_run_pkg: MagicMock,
        mock_jit: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """Parallel mode writes all results to JSONL."""
        self.config.workers = 2
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
            PackageResult(package="p2", status="fail", exit_code=1),
            PackageResult(package="p3", status="pass"),
        ]
        output = run_all(self.config)
        results_file = output.run_dir / "results.jsonl"
        self.assertTrue(results_file.exists())
        lines = results_file.read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)
        packages_in_jsonl = set()
        for line in lines:
            data = json.loads(line)
            packages_in_jsonl.add(data["package"])
        self.assertEqual(packages_in_jsonl, {"p1", "p2", "p3"})


class TestCancelEvent(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.config = _make_config(self.base)
        self.run_dir = create_run_dir(self.config.results_dir, self.config.run_id)
        self.env = build_env(self.config)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("labeille.runner.clone_repo")
    def test_cancel_event_stops_before_clone(self, mock_clone: MagicMock) -> None:
        """A pre-set cancel event causes run_package to return early."""
        import threading

        cancel = threading.Event()
        cancel.set()
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env, cancel_event=cancel)
        self.assertEqual(result.status, "error")
        self.assertIn("crash limit reached", result.error_message or "")
        mock_clone.assert_not_called()

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_cancel_event_none_runs_normally(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """cancel_event=None (default) runs the full pipeline."""
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
        result = run_package(pkg, self.config, self.run_dir, self.env, cancel_event=None)
        self.assertEqual(result.status, "pass")


class TestExtractPythonMinorVersion(unittest.TestCase):
    def test_full_version_string(self) -> None:
        self.assertEqual(extract_python_minor_version("3.15.0a5+"), "3.15")

    def test_release_version(self) -> None:
        self.assertEqual(extract_python_minor_version("3.14.2"), "3.14")

    def test_version_with_build_info(self) -> None:
        self.assertEqual(extract_python_minor_version("3.15.0a5+ (heads/main:abc1234)"), "3.15")

    def test_short_version(self) -> None:
        self.assertEqual(extract_python_minor_version("3.15"), "3.15")

    def test_empty_string(self) -> None:
        self.assertEqual(extract_python_minor_version(""), "")

    def test_malformed_version(self) -> None:
        self.assertEqual(extract_python_minor_version("not-a-version"), "not-a-version")


class TestPullRepo(unittest.TestCase):
    @patch("labeille.runner.subprocess.run")
    def test_success(self, mock_run: Any) -> None:
        """Fetch + reset + clean + rev-parse all succeed → returns revision."""

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            cmd = args[0]
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[0:2] == ["git", "fetch"]:
                pass  # check=True, no exception needed
            elif cmd[0:3] == ["git", "reset", "--hard"]:
                pass
            elif cmd[0:2] == ["git", "clean"]:
                pass
            elif cmd[0:3] == ["git", "rev-parse", "HEAD"]:
                result.stdout = "abc123\n"
            return result

        mock_run.side_effect = side_effect
        rev = pull_repo(Path("/tmp/repo"))
        self.assertEqual(rev, "abc123")
        # Should have called fetch, reset, clean, rev-parse (4 calls).
        self.assertEqual(mock_run.call_count, 4)

    @patch("labeille.runner.subprocess.run")
    def test_fetch_fails_propagates(self, mock_run: Any) -> None:
        """git fetch failing raises CalledProcessError."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git fetch")
        with self.assertRaises(subprocess.CalledProcessError):
            pull_repo(Path("/tmp/repo"))

    @patch("labeille.runner.subprocess.run")
    def test_reset_fails_nonfatal(self, mock_run: Any) -> None:
        """git reset --hard failing is non-fatal; revision still returned."""

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            cmd = args[0]
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[0:3] == ["git", "reset", "--hard"]:
                result.returncode = 1
                result.stderr = "reset failed"
            elif cmd[0:3] == ["git", "rev-parse", "HEAD"]:
                result.stdout = "def456\n"
            return result

        mock_run.side_effect = side_effect
        rev = pull_repo(Path("/tmp/repo"))
        self.assertEqual(rev, "def456")

    @patch("labeille.runner.subprocess.run")
    def test_clean_fails_nonfatal(self, mock_run: Any) -> None:
        """git clean failing is non-fatal; revision still returned."""

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            cmd = args[0]
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[0:2] == ["git", "clean"]:
                result.returncode = 1
                result.stderr = "clean failed"
            elif cmd[0:3] == ["git", "rev-parse", "HEAD"]:
                result.stdout = "ghi789\n"
            return result

        mock_run.side_effect = side_effect
        rev = pull_repo(Path("/tmp/repo"))
        self.assertEqual(rev, "ghi789")


class TestParsePackageSpecs(unittest.TestCase):
    def test_simple(self) -> None:
        names, revs = parse_package_specs("requests,click")
        self.assertEqual(names, ["requests", "click"])
        self.assertEqual(revs, {})

    def test_with_revision(self) -> None:
        names, revs = parse_package_specs("requests@abc123,click")
        self.assertEqual(names, ["requests", "click"])
        self.assertEqual(revs, {"requests": "abc123"})

    def test_head_tilde(self) -> None:
        names, revs = parse_package_specs("numpy@HEAD~5")
        self.assertEqual(names, ["numpy"])
        self.assertEqual(revs, {"numpy": "HEAD~5"})

    def test_all_with_revisions(self) -> None:
        names, revs = parse_package_specs("a@rev1,b@rev2")
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(revs, {"a": "rev1", "b": "rev2"})

    def test_empty_revision_ignored(self) -> None:
        """Empty revision after @ is ignored — name is still included."""
        names, revs = parse_package_specs("a@,b")
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(revs, {})

    def test_empty_string(self) -> None:
        names, revs = parse_package_specs("")
        self.assertEqual(names, [])
        self.assertEqual(revs, {})

    def test_whitespace_handling(self) -> None:
        names, revs = parse_package_specs(" a @ rev1 , b ")
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(revs, {"a": "rev1"})


class TestCheckoutRevision(unittest.TestCase):
    @patch("labeille.runner.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        """Successful checkout returns the resolved commit hash."""

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            cmd = args[0]
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            if cmd[0:2] == ["git", "checkout"]:
                pass
            elif cmd[0:3] == ["git", "rev-parse", "HEAD"]:
                result.stdout = "abc123full\n"
            return result

        mock_run.side_effect = side_effect
        commit = checkout_revision(Path("/tmp/repo"), "abc123")
        self.assertEqual(commit, "abc123full")

    @patch("labeille.runner.subprocess.run")
    def test_failure(self, mock_run: MagicMock) -> None:
        """Failed checkout returns None."""
        result = MagicMock()
        result.returncode = 128
        result.stderr = "error: pathspec 'badref' did not match"
        mock_run.return_value = result
        commit = checkout_revision(Path("/tmp/repo"), "badref")
        self.assertIsNone(commit)


class TestCommitRecording(unittest.TestCase):
    """Verify the git_revision field flows end-to-end through the pipeline."""

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
    def test_result_includes_commit_hash(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Commit hash from clone_repo flows into JSONL result."""
        mock_clone.return_value = "abc1234deadbeef"
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
        self.assertEqual(result.git_revision, "abc1234deadbeef")

        # Verify it's written to JSONL.
        append_result(self.run_dir, result)
        jsonl_file = self.run_dir / "results.jsonl"
        data = json.loads(jsonl_file.read_text().strip())
        self.assertEqual(data["git_revision"], "abc1234deadbeef")


class TestCloneDepthOverride(unittest.TestCase):
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
    def test_clone_depth_override_takes_precedence(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """CLI clone_depth_override takes precedence over package clone_depth."""
        self.config.clone_depth_override = 5
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
        pkg = _make_package(clone_depth=1)
        run_package(pkg, self.config, self.run_dir, self.env)
        _, kwargs = mock_clone.call_args
        self.assertEqual(kwargs.get("clone_depth"), 5)

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_clone_depth_none_uses_package_value(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """clone_depth_override=None falls back to package clone_depth."""
        self.config.clone_depth_override = None
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
        pkg = _make_package(clone_depth=3)
        run_package(pkg, self.config, self.run_dir, self.env)
        _, kwargs = mock_clone.call_args
        self.assertEqual(kwargs.get("clone_depth"), 3)

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_clone_depth_zero_means_full_clone(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """clone_depth_override=0 means full clone (no --depth flag)."""
        self.config.clone_depth_override = 0
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
        pkg = _make_package(clone_depth=10)
        run_package(pkg, self.config, self.run_dir, self.env)
        _, kwargs = mock_clone.call_args
        # depth=0 is converted to None (full clone).
        self.assertIsNone(kwargs.get("clone_depth"))


class TestRevisionOverride(unittest.TestCase):
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

    @patch("labeille.runner.checkout_revision")
    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_run_single_package_with_revision(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
        mock_checkout: MagicMock,
    ) -> None:
        """Revision override triggers checkout and records requested_revision."""
        self.config.revision_overrides = {"testpkg": "abc123"}
        mock_clone.return_value = "original_head"
        mock_checkout.return_value = "abc123fullhash"
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
        self.assertEqual(result.git_revision, "abc123fullhash")
        self.assertEqual(result.requested_revision, "abc123")
        mock_checkout.assert_called_once()

    @patch("labeille.runner.checkout_revision")
    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_run_single_package_checkout_fails(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
        mock_checkout: MagicMock,
    ) -> None:
        """Failed checkout sets error status."""
        self.config.revision_overrides = {"testpkg": "badref"}
        mock_clone.return_value = "original_head"
        mock_checkout.return_value = None
        pkg = _make_package()
        result = run_package(pkg, self.config, self.run_dir, self.env)
        self.assertEqual(result.status, "error")
        self.assertIn("Failed to checkout revision badref", result.error_message or "")
        mock_test.assert_not_called()

    @patch("labeille.runner.run_test_command")
    @patch("labeille.runner.get_installed_packages")
    @patch("labeille.runner.check_import")
    @patch("labeille.runner.install_package")
    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.clone_repo")
    def test_no_revision_override_skips_checkout(
        self,
        mock_clone: MagicMock,
        mock_venv: MagicMock,
        mock_install: MagicMock,
        mock_import: MagicMock,
        mock_get_pkgs: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        """Without revision override, no checkout happens and requested_revision is None."""
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
        self.assertIsNone(result.requested_revision)


if __name__ == "__main__":
    unittest.main()
