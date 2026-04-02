"""Tests for labeille.tsan_run — TSan-enabled test runner."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from labeille.tsan_run import (
    PythonTsanInfo,
    TsanRunConfig,
    TsanRunMeta,
    TsanRunResult,
    build_tsan_options,
    find_extension_sos,
    parse_race_count,
    parse_race_types,
    validate_tsan_python,
)


# ---------------------------------------------------------------------------
# parse_race_count / parse_race_types
# ---------------------------------------------------------------------------


SAMPLE_REPORT = """\
==================
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 8 at 0x7f8a1c003420 by thread T1:
    #0 update_cache /path/to/myext.c:42 (myext.cpython-314t-x86_64-linux-gnu.so+0x1234)

  Previous read of size 8 at 0x7f8a1c003420 by thread T2:
    #0 get_cache /path/to/myext.c:55 (myext.cpython-314t-x86_64-linux-gnu.so+0x9abc)

==================
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 4 at 0x7f8a1c003500 by thread T1:
    #0 set_flag /path/to/myext.c:100 (myext.cpython-314t-x86_64-linux-gnu.so+0x5678)

  Previous write of size 4 at 0x7f8a1c003500 by thread T3:
    #0 set_flag /path/to/myext.c:100 (myext.cpython-314t-x86_64-linux-gnu.so+0x5678)

==================
WARNING: ThreadSanitizer: thread leak (pid=12345)
  Thread T4 (tid=99999, finished) created by main thread at:
    #0 pthread_create ...

==================
ThreadSanitizer: reported 3 warnings
"""


class TestParseRaceCount(unittest.TestCase):
    def test_sample_report(self) -> None:
        self.assertEqual(parse_race_count(SAMPLE_REPORT), 3)

    def test_empty_report(self) -> None:
        self.assertEqual(parse_race_count(""), 0)

    def test_no_races(self) -> None:
        self.assertEqual(parse_race_count("All tests passed.\n"), 0)

    def test_single_race(self) -> None:
        report = "WARNING: ThreadSanitizer: data race (pid=1)\n  ...\n"
        self.assertEqual(parse_race_count(report), 1)

    def test_no_pid(self) -> None:
        # Some TSan versions omit pid.
        report = "WARNING: ThreadSanitizer: data race\n  ...\n"
        self.assertEqual(parse_race_count(report), 1)


class TestParseRaceTypes(unittest.TestCase):
    def test_sample_report(self) -> None:
        types = parse_race_types(SAMPLE_REPORT)
        self.assertEqual(types["data race"], 2)
        self.assertEqual(types["thread leak"], 1)
        self.assertEqual(len(types), 2)

    def test_empty_report(self) -> None:
        self.assertEqual(parse_race_types(""), {})


# ---------------------------------------------------------------------------
# validate_tsan_python
# ---------------------------------------------------------------------------


class TestValidateTsanPython(unittest.TestCase):
    @patch("labeille.tsan_run.subprocess.run")
    def test_valid_tsan_ft_python(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="3.14.0+ (free-threading)|True|True|True|cpython-314t-x86_64-linux-gnu",
            stderr="",
        )
        info = validate_tsan_python(Path("/usr/bin/python3.14t"))
        self.assertTrue(info.is_free_threaded)
        self.assertTrue(info.is_tsan)
        self.assertTrue(info.is_debug)
        self.assertIn("3.14.0+", info.version)

    @patch("labeille.tsan_run.subprocess.run")
    def test_not_free_threaded(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="3.14.0+|False|True|False|cpython-314-x86_64-linux-gnu",
            stderr="",
        )
        with self.assertRaises(ValueError) as ctx:
            validate_tsan_python(Path("/usr/bin/python3.14"))
        self.assertIn("not free-threaded", str(ctx.exception))

    @patch("labeille.tsan_run.subprocess.run")
    def test_no_tsan_warns(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="3.14.0+ (free-threading)|True|False|False|cpython-314t-x86_64-linux-gnu",
            stderr="",
        )
        with self.assertLogs("labeille.tsan_run", level="WARNING") as cm:
            info = validate_tsan_python(Path("/usr/bin/python3.14t"))
        self.assertFalse(info.is_tsan)
        self.assertTrue(any("does not appear to have TSan" in m for m in cm.output))

    @patch("labeille.tsan_run.subprocess.run", side_effect=OSError("not found"))
    def test_python_not_found(self, _mock: MagicMock) -> None:
        with self.assertRaises(RuntimeError):
            validate_tsan_python(Path("/nonexistent/python"))

    @patch("labeille.tsan_run.subprocess.run")
    def test_python_crashes(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Segfault")
        with self.assertRaises(RuntimeError):
            validate_tsan_python(Path("/usr/bin/python3.14t"))


# ---------------------------------------------------------------------------
# find_extension_sos
# ---------------------------------------------------------------------------


class TestFindExtensionSos(unittest.TestCase):
    def test_finds_matching_sos(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            venv = Path(td)
            sp = venv / "lib" / "python3.14t" / "site-packages" / "myext"
            sp.mkdir(parents=True)
            so1 = sp / "_myext.cpython-314t-x86_64-linux-gnu.so"
            so1.touch()
            so2 = sp / "_helper.cpython-314t-x86_64-linux-gnu.so"
            so2.touch()
            # Unrelated package.
            other = venv / "lib" / "python3.14t" / "site-packages" / "other"
            other.mkdir(parents=True)
            (other / "_other.cpython-314t-x86_64-linux-gnu.so").touch()

            sos = find_extension_sos(venv, "myext")
            self.assertEqual(len(sos), 2)
            self.assertTrue(all("myext" in s for s in sos))

    def test_no_lib_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(find_extension_sos(Path(td), "pkg"), [])

    def test_hyphen_underscore_normalisation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            venv = Path(td)
            sp = venv / "lib" / "python3.14t" / "site-packages" / "my_ext"
            sp.mkdir(parents=True)
            (sp / "_core.cpython-314t-x86_64-linux-gnu.so").touch()

            sos = find_extension_sos(venv, "my-ext")
            self.assertEqual(len(sos), 1)


# ---------------------------------------------------------------------------
# build_tsan_options
# ---------------------------------------------------------------------------


class TestBuildTsanOptions(unittest.TestCase):
    def test_defaults(self) -> None:
        opts = build_tsan_options()
        self.assertIn("history_size=7", opts)
        self.assertIn("halt_on_error=0", opts)
        self.assertIn("exitcode=0", opts)
        self.assertIn("second_deadlock_stack=1", opts)
        self.assertIn("report_signal_unsafe=0", opts)
        self.assertNotIn("suppressions=", opts)

    def test_with_suppressions(self) -> None:
        opts = build_tsan_options(suppressions_path=Path("/tmp/supp.txt"))
        self.assertIn("suppressions=/tmp/supp.txt", opts)

    def test_quick_mode(self) -> None:
        opts = build_tsan_options(halt_on_error=True, exitcode=66)
        self.assertIn("halt_on_error=1", opts)
        self.assertIn("exitcode=66", opts)

    def test_log_path(self) -> None:
        opts = build_tsan_options(log_path="/tmp/tsan_log")
        self.assertIn("log_path=/tmp/tsan_log", opts)

    def test_custom_history_size(self) -> None:
        opts = build_tsan_options(history_size=5)
        self.assertIn("history_size=5", opts)


# ---------------------------------------------------------------------------
# TsanRunResult serialization
# ---------------------------------------------------------------------------


class TestTsanRunResult(unittest.TestCase):
    def test_to_dict_sparse(self) -> None:
        r = TsanRunResult(package="myext", status="ok", race_count=5)
        d = r.to_dict()
        self.assertEqual(d["package"], "myext")
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["race_count"], 5)
        # Empty fields should be omitted.
        self.assertNotIn("extension_so_paths", d)
        self.assertNotIn("error_summary", d)

    def test_to_dict_full(self) -> None:
        r = TsanRunResult(
            package="myext",
            status="ok",
            race_count=3,
            race_types={"data race": 2, "thread leak": 1},
            report_path="/tmp/report.txt",
            metadata_path="/tmp/meta.json",
            test_exit_code=0,
            test_duration_s=45.123,
            install_duration_s=12.5,
            extension_so_paths=["/path/to/ext.so"],
        )
        d = r.to_dict()
        self.assertEqual(d["race_types"], {"data race": 2, "thread leak": 1})
        self.assertEqual(d["test_duration_s"], 45.12)
        self.assertEqual(d["install_duration_s"], 12.5)
        self.assertEqual(d["extension_so_paths"], ["/path/to/ext.so"])

    def test_from_dict(self) -> None:
        data: dict[str, Any] = {
            "package": "myext",
            "status": "ok",
            "race_count": 5,
            "unknown_field": "ignored",
        }
        r = TsanRunResult.from_dict(data)
        self.assertEqual(r.package, "myext")
        self.assertEqual(r.race_count, 5)


# ---------------------------------------------------------------------------
# TsanRunMeta serialization
# ---------------------------------------------------------------------------


class TestTsanRunMeta(unittest.TestCase):
    def test_to_dict(self) -> None:
        m = TsanRunMeta(
            run_id="tsan_20260401",
            target_python="/usr/bin/python3.14t",
            python_version="3.14.0+",
            is_free_threaded=True,
            is_tsan=True,
            total_packages=10,
            packages_with_races=3,
            total_races=15,
        )
        d = m.to_dict()
        self.assertEqual(d["run_id"], "tsan_20260401")
        self.assertTrue(d["is_free_threaded"])
        self.assertEqual(d["total_races"], 15)

    def test_from_dict(self) -> None:
        data: dict[str, Any] = {
            "run_id": "tsan_20260401",
            "target_python": "/usr/bin/python3.14t",
            "python_version": "3.14.0+",
            "extra": "ignored",
        }
        m = TsanRunMeta.from_dict(data)
        self.assertEqual(m.run_id, "tsan_20260401")


# ---------------------------------------------------------------------------
# TsanRunConfig defaults
# ---------------------------------------------------------------------------


class TestTsanRunConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        c = TsanRunConfig(target_python=Path("/usr/bin/python3.14t"))
        self.assertEqual(c.timeout, 600)
        self.assertEqual(c.stress, 1)
        self.assertFalse(c.quick)
        self.assertTrue(c.skip_if_exists)
        self.assertEqual(c.workers, 1)
        self.assertEqual(c.extra_deps, [])


# ---------------------------------------------------------------------------
# run_tsan_tests (mocked)
# ---------------------------------------------------------------------------


class TestRunTsanTests(unittest.TestCase):
    def _make_pkg(self, **kwargs: Any) -> MagicMock:
        pkg = MagicMock()
        pkg.package = kwargs.get("package", "testpkg")
        pkg.repo = kwargs.get("repo", "https://github.com/test/testpkg")
        pkg.install_command = kwargs.get("install_command", "pip install -e .")
        pkg.test_command = kwargs.get("test_command", "python -m pytest tests/")
        pkg.dependencies = kwargs.get("dependencies", [])
        pkg.timeout = kwargs.get("timeout", None)
        pkg.version = kwargs.get("version", "1.0.0")
        return pkg

    def test_skip_if_exists(self) -> None:
        from labeille.tsan_run import run_tsan_tests

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "testpkg"
            out.mkdir()
            (out / "tsan_report.txt").write_text("existing report")
            config = TsanRunConfig(
                target_python=Path("/usr/bin/python"),
                skip_if_exists=True,
            )
            result = run_tsan_tests(self._make_pkg(), config, out)
            self.assertEqual(result.status, "skipped")

    def test_no_repo(self) -> None:
        from labeille.tsan_run import run_tsan_tests

        with tempfile.TemporaryDirectory() as td:
            config = TsanRunConfig(target_python=Path("/usr/bin/python"))
            pkg = self._make_pkg(repo=None)
            result = run_tsan_tests(pkg, config, Path(td))
            self.assertEqual(result.status, "no_repo")

    @patch("labeille.tsan_run.run_test_command")
    @patch("labeille.tsan_run.install_package")
    @patch("labeille.tsan_run.create_venv")
    @patch("labeille.tsan_run.pull_repo")
    @patch("labeille.tsan_run.clone_repo")
    def test_full_run_with_races(
        self,
        mock_clone: MagicMock,
        mock_pull: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        from labeille.tsan_run import run_tsan_tests

        mock_install.return_value = MagicMock(returncode=0, stderr="")
        mock_test.return_value = MagicMock(
            returncode=0,
            stderr=(
                "WARNING: ThreadSanitizer: data race (pid=1)\n"
                "  Write of size 8...\n"
                "==================\n"
                "WARNING: ThreadSanitizer: data race (pid=1)\n"
                "  Read of size 4...\n"
            ),
        )

        with tempfile.TemporaryDirectory() as td:
            config = TsanRunConfig(
                target_python=Path("/usr/bin/python"),
                output_dir=Path(td),
                skip_if_exists=False,
            )
            out = Path(td) / "testpkg"
            out.mkdir()
            result = run_tsan_tests(self._make_pkg(), config, out, tsan_options="exitcode=0")
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.race_count, 2)
            self.assertEqual(result.race_types, {"data race": 2})
            self.assertTrue(result.report_path)
            self.assertTrue(result.metadata_path)

    @patch("labeille.tsan_run.run_test_command")
    @patch("labeille.tsan_run.install_package")
    @patch("labeille.tsan_run.create_venv")
    @patch("labeille.tsan_run.clone_repo")
    def test_full_run_no_races(
        self,
        mock_clone: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        from labeille.tsan_run import run_tsan_tests

        mock_install.return_value = MagicMock(returncode=0, stderr="")
        mock_test.return_value = MagicMock(returncode=0, stderr="All tests passed.\n")

        with tempfile.TemporaryDirectory() as td:
            config = TsanRunConfig(
                target_python=Path("/usr/bin/python"),
                output_dir=Path(td),
                skip_if_exists=False,
            )
            out = Path(td) / "testpkg"
            out.mkdir()
            result = run_tsan_tests(self._make_pkg(), config, out)
            self.assertEqual(result.status, "no_races")
            self.assertEqual(result.race_count, 0)

    @patch("labeille.tsan_run.install_package")
    @patch("labeille.tsan_run.create_venv")
    @patch("labeille.tsan_run.clone_repo")
    def test_install_failure(
        self,
        mock_clone: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
    ) -> None:
        from labeille.tsan_run import run_tsan_tests

        mock_install.return_value = MagicMock(returncode=1, stderr="error: build failed")

        with tempfile.TemporaryDirectory() as td:
            config = TsanRunConfig(
                target_python=Path("/usr/bin/python"),
                output_dir=Path(td),
                skip_if_exists=False,
            )
            out = Path(td) / "testpkg"
            out.mkdir()
            result = run_tsan_tests(self._make_pkg(), config, out)
            self.assertEqual(result.status, "install_error")

    @patch("labeille.tsan_run.clone_repo", side_effect=subprocess.CalledProcessError(1, "git"))
    def test_clone_failure(self, mock_clone: MagicMock) -> None:
        from labeille.tsan_run import run_tsan_tests

        with tempfile.TemporaryDirectory() as td:
            config = TsanRunConfig(
                target_python=Path("/usr/bin/python"),
                output_dir=Path(td),
                skip_if_exists=False,
            )
            result = run_tsan_tests(self._make_pkg(), config, Path(td) / "out")
            self.assertEqual(result.status, "clone_error")

    @patch("labeille.tsan_run.run_test_command", side_effect=subprocess.TimeoutExpired("cmd", 600))
    @patch("labeille.tsan_run.install_package")
    @patch("labeille.tsan_run.create_venv")
    @patch("labeille.tsan_run.clone_repo")
    def test_test_timeout(
        self,
        mock_clone: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        from labeille.tsan_run import run_tsan_tests

        mock_install.return_value = MagicMock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as td:
            config = TsanRunConfig(
                target_python=Path("/usr/bin/python"),
                output_dir=Path(td),
                skip_if_exists=False,
            )
            out = Path(td) / "testpkg"
            out.mkdir()
            result = run_tsan_tests(self._make_pkg(), config, out)
            self.assertEqual(result.status, "timeout")

    @patch("labeille.tsan_run.run_test_command")
    @patch("labeille.tsan_run.install_package")
    @patch("labeille.tsan_run.create_venv")
    @patch("labeille.tsan_run.clone_repo")
    def test_test_script_overrides_test_command(
        self,
        mock_clone: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        from labeille.tsan_run import run_tsan_tests

        mock_install.return_value = MagicMock(returncode=0, stderr="")
        mock_test.return_value = MagicMock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "stress_test.py"
            script.write_text("import threading\nprint('stress')\n")
            config = TsanRunConfig(
                target_python=Path("/usr/bin/python"),
                output_dir=Path(td),
                skip_if_exists=False,
                test_script=script,
            )
            out = Path(td) / "testpkg"
            out.mkdir()
            result = run_tsan_tests(self._make_pkg(), config, out)
            self.assertEqual(result.status, "no_races")
            # Verify the custom script was passed to run_test_command.
            call_args = mock_test.call_args
            test_cmd_arg = call_args[1].get("test_command") or call_args[0][1]
            self.assertIn("stress_test.py", test_cmd_arg)
            # The package's test_command should NOT appear.
            self.assertNotIn("pytest", test_cmd_arg)

    @patch("labeille.tsan_run.run_test_command")
    @patch("labeille.tsan_run.install_package")
    @patch("labeille.tsan_run.create_venv")
    @patch("labeille.tsan_run.clone_repo")
    def test_stress_mode(
        self,
        mock_clone: MagicMock,
        mock_create_venv: MagicMock,
        mock_install: MagicMock,
        mock_test: MagicMock,
    ) -> None:
        from labeille.tsan_run import run_tsan_tests

        mock_install.return_value = MagicMock(returncode=0, stderr="")
        # Each iteration finds one race.
        mock_test.return_value = MagicMock(
            returncode=0,
            stderr="WARNING: ThreadSanitizer: data race (pid=1)\n  ...\n",
        )

        with tempfile.TemporaryDirectory() as td:
            config = TsanRunConfig(
                target_python=Path("/usr/bin/python"),
                output_dir=Path(td),
                skip_if_exists=False,
                stress=3,
            )
            out = Path(td) / "testpkg"
            out.mkdir()
            result = run_tsan_tests(self._make_pkg(), config, out)
            self.assertEqual(result.status, "ok")
            # 3 iterations, 1 race each = 3 total.
            self.assertEqual(result.race_count, 3)
            self.assertEqual(mock_test.call_count, 3)


# ---------------------------------------------------------------------------
# PythonTsanInfo
# ---------------------------------------------------------------------------


class TestPythonTsanInfo(unittest.TestCase):
    def test_fields(self) -> None:
        info = PythonTsanInfo(
            version="3.14.0+",
            is_free_threaded=True,
            is_tsan=True,
            is_debug=False,
        )
        self.assertEqual(info.version, "3.14.0+")
        self.assertTrue(info.is_free_threaded)


if __name__ == "__main__":
    unittest.main()
