"""Tests for installer backend integration (uv/pip)."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from labeille.runner import (
    InstallerBackend,
    PackageResult,
    RunnerConfig,
    _rewrite_install_command,
    create_venv,
    detect_uv,
    get_installed_packages,
    install_package,
    install_with_fallback,
    resolve_installer,
)


class TestDetectUv(unittest.TestCase):
    """Tests for detect_uv()."""

    @patch("labeille.runner.shutil.which")
    def test_found(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/local/bin/uv"
        self.assertEqual(detect_uv(), "/usr/local/bin/uv")
        mock_which.assert_called_once_with("uv")

    @patch("labeille.runner.shutil.which")
    def test_not_found(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        self.assertIsNone(detect_uv())


class TestResolveInstaller(unittest.TestCase):
    """Tests for resolve_installer()."""

    @patch("labeille.runner.detect_uv", return_value="/usr/bin/uv")
    def test_auto_with_uv(self, _mock: MagicMock) -> None:
        self.assertEqual(resolve_installer("auto"), InstallerBackend.UV)

    @patch("labeille.runner.detect_uv", return_value=None)
    def test_auto_without_uv(self, _mock: MagicMock) -> None:
        self.assertEqual(resolve_installer("auto"), InstallerBackend.PIP)

    def test_pip_explicit(self) -> None:
        self.assertEqual(resolve_installer("pip"), InstallerBackend.PIP)

    @patch("labeille.runner.detect_uv", return_value="/usr/bin/uv")
    def test_uv_explicit(self, _mock: MagicMock) -> None:
        self.assertEqual(resolve_installer("uv"), InstallerBackend.UV)

    @patch("labeille.runner.detect_uv", return_value=None)
    def test_uv_explicit_not_found(self, _mock: MagicMock) -> None:
        with self.assertRaises(RuntimeError):
            resolve_installer("uv")

    @patch("labeille.runner.detect_uv", return_value="/usr/bin/uv")
    def test_case_insensitive(self, _mock: MagicMock) -> None:
        self.assertEqual(resolve_installer("UV"), InstallerBackend.UV)
        self.assertEqual(resolve_installer("Pip"), InstallerBackend.PIP)
        self.assertEqual(resolve_installer("AUTO"), InstallerBackend.UV)


class TestRewriteInstallCommand(unittest.TestCase):
    """Tests for _rewrite_install_command()."""

    def test_pip_basic(self) -> None:
        venv_python = Path("/venv/bin/python")
        result = _rewrite_install_command("pip install -e .", venv_python, InstallerBackend.PIP)
        self.assertIn("/venv/bin/pip", result)
        self.assertIn("install -e .", result)

    def test_pip_python_prefix(self) -> None:
        venv_python = Path("/venv/bin/python")
        result = _rewrite_install_command(
            "python -m pip install foo", venv_python, InstallerBackend.PIP
        )
        self.assertIn("/venv/bin/python", result)

    def test_pip_standalone(self) -> None:
        venv_python = Path("/venv/bin/python")
        result = _rewrite_install_command("pip list", venv_python, InstallerBackend.PIP)
        self.assertIn("/venv/bin/pip", result)

    @patch("labeille.runner.detect_uv", return_value="/usr/bin/uv")
    def test_uv_basic(self, _mock: MagicMock) -> None:
        venv_python = Path("/venv/bin/python")
        result = _rewrite_install_command("pip install -e .", venv_python, InstallerBackend.UV)
        self.assertIn("/usr/bin/uv", result)
        self.assertIn("pip install", result)
        self.assertIn("--python", result)
        self.assertIn("/venv/bin/python", result)

    @patch("labeille.runner.detect_uv", return_value="/usr/bin/uv")
    def test_uv_standalone_pip(self, _mock: MagicMock) -> None:
        venv_python = Path("/venv/bin/python")
        result = _rewrite_install_command("pip list", venv_python, InstallerBackend.UV)
        self.assertIn("/usr/bin/uv", result)
        self.assertIn("--python", result)

    @patch("labeille.runner.detect_uv", return_value=None)
    def test_uv_fallback_to_pip_when_not_found(self, _mock: MagicMock) -> None:
        """When UV is selected but detect_uv returns None, falls back to pip-style rewrite."""
        venv_python = Path("/venv/bin/python")
        result = _rewrite_install_command("pip install -e .", venv_python, InstallerBackend.UV)
        self.assertIn("/venv/bin/pip", result)

    def test_compound_command(self) -> None:
        venv_python = Path("/venv/bin/python")
        cmd = "pip install -e . && pip install pytest"
        result = _rewrite_install_command(cmd, venv_python, InstallerBackend.PIP)
        self.assertIn("/venv/bin/pip install -e .", result)
        self.assertIn("/venv/bin/pip install pytest", result)


class TestCreateVenvInstaller(unittest.TestCase):
    """Tests for create_venv() with installer parameter."""

    @patch("labeille.runner.subprocess.run")
    def test_pip_backend(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        create_venv(Path("/usr/bin/python3"), Path("/tmp/venv"), InstallerBackend.PIP)
        # Should call python -m venv and ensurepip.
        calls = mock_run.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIn("-m", calls[0][0][0])
        self.assertIn("venv", calls[0][0][0])

    @patch("labeille.runner.detect_uv", return_value="/usr/bin/uv")
    @patch("labeille.runner.subprocess.run")
    def test_uv_backend(self, mock_run: MagicMock, _mock_uv: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        create_venv(Path("/usr/bin/python3"), Path("/tmp/venv"), InstallerBackend.UV)
        # Should call uv venv only (no ensurepip).
        calls = mock_run.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0][0], "/usr/bin/uv")
        self.assertEqual(calls[0][0][0][1], "venv")

    @patch("labeille.runner.detect_uv", return_value=None)
    @patch("labeille.runner.subprocess.run")
    def test_uv_backend_falls_back_when_not_found(
        self, mock_run: MagicMock, _mock_uv: MagicMock
    ) -> None:
        """When UV is selected but uv binary not found, falls back to python -m venv."""
        mock_run.return_value = MagicMock(returncode=0)
        create_venv(Path("/usr/bin/python3"), Path("/tmp/venv"), InstallerBackend.UV)
        calls = mock_run.call_args_list
        self.assertEqual(len(calls), 2)  # venv + ensurepip

    @patch("labeille.runner.subprocess.run")
    def test_default_is_pip(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        create_venv(Path("/usr/bin/python3"), Path("/tmp/venv"))
        calls = mock_run.call_args_list
        self.assertEqual(len(calls), 2)  # venv + ensurepip


class TestInstallPackageInstaller(unittest.TestCase):
    """Tests for install_package() with installer parameter."""

    @patch("labeille.runner._run_in_process_group")
    def test_pip_backend(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        install_package(
            Path("/venv/bin/python"),
            "pip install -e .",
            Path("/repo"),
            {},
            600,
            InstallerBackend.PIP,
        )
        cmd = mock_run.call_args[0][0]
        self.assertIn("/venv/bin/pip", cmd)

    @patch("labeille.runner.detect_uv", return_value="/usr/bin/uv")
    @patch("labeille.runner._run_in_process_group")
    def test_uv_backend(self, mock_run: MagicMock, _mock_uv: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        install_package(
            Path("/venv/bin/python"),
            "pip install -e .",
            Path("/repo"),
            {},
            600,
            InstallerBackend.UV,
        )
        cmd = mock_run.call_args[0][0]
        self.assertIn("/usr/bin/uv", cmd)
        self.assertIn("--python", cmd)

    @patch("labeille.runner._run_in_process_group")
    def test_default_is_pip(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        install_package(
            Path("/venv/bin/python"),
            "pip install -e .",
            Path("/repo"),
            {},
            600,
        )
        cmd = mock_run.call_args[0][0]
        self.assertIn("/venv/bin/pip", cmd)


class TestGetInstalledPackagesInstaller(unittest.TestCase):
    """Tests for get_installed_packages() with installer parameter."""

    @patch("labeille.runner.subprocess.run")
    def test_pip_backend(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='[{"name": "requests", "version": "2.31.0"}]',
        )
        result = get_installed_packages(Path("/venv/bin/python"), {}, InstallerBackend.PIP)
        self.assertEqual(result, {"requests": "2.31.0"})
        cmd = mock_run.call_args[0][0]
        self.assertIn("-m", cmd)
        self.assertIn("pip", cmd)

    @patch("labeille.runner.detect_uv", return_value="/usr/bin/uv")
    @patch("labeille.runner.subprocess.run")
    def test_uv_backend(self, mock_run: MagicMock, _mock_uv: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='[{"name": "requests", "version": "2.31.0"}]',
        )
        result = get_installed_packages(Path("/venv/bin/python"), {}, InstallerBackend.UV)
        self.assertEqual(result, {"requests": "2.31.0"})
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "/usr/bin/uv")

    @patch("labeille.runner.subprocess.run")
    def test_default_is_pip(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='[{"name": "foo", "version": "1.0"}]',
        )
        result = get_installed_packages(Path("/venv/bin/python"), {})
        self.assertEqual(result, {"foo": "1.0"})


class TestInstallWithFallback(unittest.TestCase):
    """Tests for install_with_fallback()."""

    @patch("labeille.runner.install_package")
    def test_pip_success(self, mock_install: MagicMock) -> None:
        mock_install.return_value = MagicMock(returncode=0)
        proc, backend = install_with_fallback(
            Path("/usr/bin/python3"),
            Path("/tmp/venv"),
            "pip install -e .",
            Path("/repo"),
            {},
            600,
            InstallerBackend.PIP,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(backend, InstallerBackend.PIP)

    @patch("labeille.runner.install_package")
    def test_pip_failure_no_fallback(self, mock_install: MagicMock) -> None:
        mock_install.return_value = MagicMock(returncode=1)
        proc, backend = install_with_fallback(
            Path("/usr/bin/python3"),
            Path("/tmp/venv"),
            "pip install -e .",
            Path("/repo"),
            {},
            600,
            InstallerBackend.PIP,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(backend, InstallerBackend.PIP)

    @patch("labeille.runner.install_package")
    def test_uv_success(self, mock_install: MagicMock) -> None:
        mock_install.return_value = MagicMock(returncode=0)
        proc, backend = install_with_fallback(
            Path("/usr/bin/python3"),
            Path("/tmp/venv"),
            "pip install -e .",
            Path("/repo"),
            {},
            600,
            InstallerBackend.UV,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(backend, InstallerBackend.UV)

    @patch("labeille.runner.create_venv")
    @patch("labeille.runner.shutil.rmtree")
    @patch("labeille.runner.install_package")
    def test_uv_failure_falls_back_to_pip(
        self,
        mock_install: MagicMock,
        mock_rmtree: MagicMock,
        mock_create_venv: MagicMock,
    ) -> None:
        # First call (uv) fails, second call (pip) succeeds.
        mock_install.side_effect = [
            MagicMock(returncode=1, stderr="uv error"),
            MagicMock(returncode=0),
        ]
        proc, backend = install_with_fallback(
            Path("/usr/bin/python3"),
            Path("/tmp/venv"),
            "pip install -e .",
            Path("/repo"),
            {},
            600,
            InstallerBackend.UV,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(backend, InstallerBackend.PIP)
        mock_rmtree.assert_called_once()
        mock_create_venv.assert_called_once_with(
            Path("/usr/bin/python3"),
            Path("/tmp/venv"),
            InstallerBackend.PIP,
        )


class TestPackageResultInstaller(unittest.TestCase):
    """Tests for installer_backend field on PackageResult."""

    def test_default_empty(self) -> None:
        r = PackageResult(package="test")
        self.assertEqual(r.installer_backend, "")

    def test_set_value(self) -> None:
        r = PackageResult(package="test", installer_backend="uv")
        self.assertEqual(r.installer_backend, "uv")


class TestRunnerConfigInstaller(unittest.TestCase):
    """Tests for installer field on RunnerConfig."""

    def test_default_auto(self) -> None:
        config = RunnerConfig(
            target_python=Path("/usr/bin/python3"),
            registry_dir=Path("/tmp/registry"),
            results_dir=Path("/tmp/results"),
            run_id="test",
        )
        self.assertEqual(config.installer, "auto")

    def test_set_value(self) -> None:
        config = RunnerConfig(
            target_python=Path("/usr/bin/python3"),
            registry_dir=Path("/tmp/registry"),
            results_dir=Path("/tmp/results"),
            run_id="test",
            installer="uv",
        )
        self.assertEqual(config.installer, "uv")


class TestInstallerBackendEnum(unittest.TestCase):
    """Tests for InstallerBackend enum."""

    def test_values(self) -> None:
        self.assertEqual(InstallerBackend.PIP.value, "pip")
        self.assertEqual(InstallerBackend.UV.value, "uv")

    def test_members(self) -> None:
        self.assertIn("PIP", InstallerBackend.__members__)
        self.assertIn("UV", InstallerBackend.__members__)


if __name__ == "__main__":
    unittest.main()
