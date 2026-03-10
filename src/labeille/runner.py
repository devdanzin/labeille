"""Test runner for executing package test suites against JIT-enabled CPython.

This module handles cloning repositories, installing packages into a
JIT-enabled Python environment, running their test suites, and capturing
the results including any crashes (segfaults, aborts, assertion failures).

.. warning::

    This module installs PyPI packages and runs their test suites,
    which means executing arbitrary third-party code.  Run in an
    isolated environment (container, VM) whenever possible.  See the
    README for detailed security guidance.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from labeille.crash import detect_crash
from labeille.io_utils import append_jsonl, utc_now_iso, write_meta_json
from labeille.logging import get_logger
from labeille.registry import Index, PackageEntry, load_index, load_package, package_exists

# Re-export data models from runner_models (preserves all existing imports).
from labeille.runner_models import InstallerBackend as InstallerBackend
from labeille.runner_models import PackageResult as PackageResult
from labeille.runner_models import RunnerConfig as RunnerConfig
from labeille.runner_models import RunOutput as RunOutput
from labeille.runner_models import RunSummary as RunSummary

# Re-export repo operations from repo_ops (preserves all existing imports).
from labeille.repo_ops import _EXTRAS_RE as _EXTRAS_RE
from labeille.repo_ops import _SELF_INSTALL_RE as _SELF_INSTALL_RE
from labeille.repo_ops import _TAG_PATTERNS as _TAG_PATTERNS
from labeille.repo_ops import _extract_extras as _extract_extras
from labeille.repo_ops import _is_self_install_segment as _is_self_install_segment
from labeille.repo_ops import build_sdist_install_commands as build_sdist_install_commands
from labeille.repo_ops import checkout_matching_tag as checkout_matching_tag
from labeille.repo_ops import checkout_revision as checkout_revision
from labeille.repo_ops import clone_repo as clone_repo
from labeille.repo_ops import detect_source_layout as detect_source_layout
from labeille.repo_ops import fetch_latest_pypi_version as fetch_latest_pypi_version
from labeille.repo_ops import parse_package_specs as parse_package_specs
from labeille.repo_ops import parse_repo_overrides as parse_repo_overrides
from labeille.repo_ops import pull_repo as pull_repo
from labeille.repo_ops import shield_source_dir as shield_source_dir
from labeille.repo_ops import split_install_command as split_install_command

log = get_logger("runner")


# ---------------------------------------------------------------------------
# Installer backend helpers
# ---------------------------------------------------------------------------


def detect_uv() -> str | None:
    """Return the path to uv if available on PATH, else None."""
    return shutil.which("uv")


def resolve_installer(preference: str = "auto") -> InstallerBackend:
    """Resolve the installer backend from a preference string.

    Args:
        preference: One of ``"auto"``, ``"uv"``, or ``"pip"``.

    Returns:
        The resolved backend.

    Raises:
        RuntimeError: If ``"uv"`` is requested but not found.
    """
    pref = preference.lower().strip()
    if pref == "pip":
        return InstallerBackend.PIP
    if pref == "uv":
        if detect_uv() is None:
            raise RuntimeError(
                "uv requested as installer but not found on PATH. "
                "Install it: https://docs.astral.sh/uv/getting-started/installation/"
            )
        return InstallerBackend.UV
    # auto: use uv if available, else pip.
    if detect_uv() is not None:
        return InstallerBackend.UV
    return InstallerBackend.PIP


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------


def create_run_dir(results_dir: Path, run_id: str) -> Path:
    """Create and return the directory for a run."""
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "crashes").mkdir(exist_ok=True)
    return run_dir


def write_run_meta(
    run_dir: Path,
    config: RunnerConfig,
    python_version: str,
    jit_enabled: bool,
    summary: RunSummary | None = None,
    started_at: str = "",
    finished_at: str = "",
) -> None:
    """Write run_meta.json to the run directory."""
    meta: dict[str, Any] = {
        "run_id": config.run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "target_python": str(config.target_python),
        "python_version": python_version,
        "jit_enabled": jit_enabled,
        "cli_args": config.cli_args,
        "env_overrides": config.env_overrides,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "installer": config.installer,
        "install_from": config.install_from,
        "uv_available": detect_uv() is not None,
    }
    if summary is not None:
        meta["packages_tested"] = summary.tested
        meta["packages_skipped"] = summary.skipped
        meta["crashes_found"] = summary.crashed
        meta["total_duration_seconds"] = 0.0  # filled by caller
    write_meta_json(run_dir / "run_meta.json", meta)


def append_result(run_dir: Path, result: PackageResult) -> None:
    """Append a single result as a JSON line to results.jsonl."""
    append_jsonl(run_dir / "results.jsonl", result.to_dict())


def load_completed_packages(run_dir: Path) -> set[str]:
    """Load the set of package names already recorded in results.jsonl."""
    from labeille.io_utils import iter_jsonl

    results_file = run_dir / "results.jsonl"
    if not results_file.exists():
        return set()
    return {r["package"] for r in iter_jsonl(results_file, lambda d: d)}


def save_crash_stderr(run_dir: Path, package_name: str, stderr: str) -> None:
    """Save stderr output for a crashed package."""
    crash_file = run_dir / "crashes" / f"{package_name}.stderr"
    try:
        crash_file.parent.mkdir(parents=True, exist_ok=True)
        crash_file.write_text(stderr, encoding="utf-8")
    except OSError as exc:
        log.error("Could not save crash stderr for %s: %s", package_name, exc)


# ---------------------------------------------------------------------------
# Target Python helpers
# ---------------------------------------------------------------------------


def validate_target_python(python_path: Path) -> str:
    """Validate the target Python and return its version string.

    Raises:
        RuntimeError: If the Python interpreter cannot be executed.
    """
    try:
        proc = subprocess.run(
            [str(python_path), "-c", "import sys; print(sys.version)"],
            capture_output=True,
            text=True,
            timeout=30,
            env=clean_env(ASAN_OPTIONS="detect_leaks=0"),
        )
    except FileNotFoundError:
        raise RuntimeError(f"Python interpreter not found: {python_path}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Python interpreter timed out: {python_path}") from None
    if proc.returncode != 0:
        raise RuntimeError(
            f"Python interpreter failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def check_jit_enabled(python_path: Path) -> bool:
    """Check if the target Python build has JIT support.

    Checks for ``sys.flags.jit`` (CPython 3.15+ with ``--enable-experimental-jit``).
    Sets ``PYTHON_JIT=1`` so the JIT is active during the check.
    """
    script = (
        "import sys\n"
        "jit_available = False\n"
        "try:\n"
        "    jit_available = bool(getattr(sys.flags, 'jit', False))\n"
        "except (AttributeError, TypeError):\n"
        "    pass\n"
        "print(jit_available)\n"
    )
    try:
        proc = subprocess.run(
            [str(python_path), "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env=clean_env(PYTHON_JIT="1", ASAN_OPTIONS="detect_leaks=0"),
        )
        return proc.stdout.strip() == "True"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def clean_env(**overrides: str) -> dict[str, str]:
    """Build a clean environment dict, stripping Python-specific pollution.

    Removes ``PYTHONHOME`` and ``PYTHONPATH`` which would corrupt the target
    Python's module resolution if inherited from a conda or custom environment.
    """
    env = {**os.environ}
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.update(overrides)
    return env


def build_env(config: RunnerConfig) -> dict[str, str]:
    """Build the environment dict for test subprocesses."""
    env = clean_env()
    env["PYTHON_JIT"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASAN_OPTIONS"] = "detect_leaks=0"
    env.update(config.env_overrides)
    log.debug(
        "Env: PYTHON_JIT=%s PYTHONFAULTHANDLER=%s ASAN_OPTIONS=%s",
        env["PYTHON_JIT"],
        env["PYTHONFAULTHANDLER"],
        env["ASAN_OPTIONS"],
    )
    if config.env_overrides:
        log.debug("Env overrides: %s", config.env_overrides)
    return env


def create_venv(
    python_path: Path,
    venv_dir: Path,
    installer: InstallerBackend = InstallerBackend.PIP,
) -> None:
    """Create a virtual environment using the target Python.

    When *installer* is :attr:`InstallerBackend.UV`, uv creates the venv
    directly (no ensurepip step).  Otherwise falls back to ``python -m venv``
    with ensurepip.

    Raises:
        subprocess.CalledProcessError: If venv creation fails.
    """
    if installer is InstallerBackend.UV:
        uv_path = detect_uv()
        if uv_path:
            log.debug("Running: %s venv --python %s %s", uv_path, python_path, venv_dir)
            subprocess.run(
                [uv_path, "venv", "--python", str(python_path), str(venv_dir)],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
                env=clean_env(ASAN_OPTIONS="detect_leaks=0"),
            )
            return

    log.debug("Running: %s -m venv %s", python_path, venv_dir)
    subprocess.run(
        [str(python_path), "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env=clean_env(ASAN_OPTIONS="detect_leaks=0"),
    )
    # Ensure pip is available.
    venv_python = venv_dir / "bin" / "python"
    log.debug("Running: %s -m ensurepip --upgrade", venv_python)
    ensurepip_proc = subprocess.run(
        [str(venv_python), "-m", "ensurepip", "--upgrade"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # ensurepip may fail on some builds; pip might already exist
        env=clean_env(ASAN_OPTIONS="detect_leaks=0"),
    )
    if ensurepip_proc.returncode != 0:
        log.debug("ensurepip exited %d (non-fatal)", ensurepip_proc.returncode)


def _run_in_process_group(
    cmd: str,
    *,
    cwd: str,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run a command in its own process group.

    On timeout, kills the entire process group (not just the immediate
    child) to prevent orphaned grandchild processes from accumulating.

    Args:
        cmd: The shell command to run.
        cwd: Working directory.
        env: Environment variables.
        timeout: Timeout in seconds.

    Returns:
        A :class:`~subprocess.CompletedProcess` with stdout, stderr, and returncode.

    Raises:
        subprocess.TimeoutExpired: If the command exceeds the timeout.
            The exception's ``stdout`` and ``stderr`` attributes contain any
            partial output captured before the timeout.
    """
    import signal as signal_module

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        # Kill the entire process group.
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal_module.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # Process already exited or we can't kill it.
            pass
        # Wait for the process to actually terminate and collect output.
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)


def _rewrite_install_command(
    install_command: str,
    venv_python: Path,
    installer: InstallerBackend = InstallerBackend.PIP,
) -> str:
    """Rewrite an install command for the given backend.

    For pip: replaces ``pip`` and ``python`` with venv-local paths.
    For uv: rewrites ``pip install`` to ``uv pip install --python <path>``.
    """
    venv_bin = venv_python.parent
    cmd = install_command

    if installer is InstallerBackend.UV:
        uv_path = detect_uv()
        if uv_path:
            # Replace "python " with venv python FIRST (before introducing --python).
            cmd = cmd.replace("python ", f"{venv_python} ")
            # Replace "pip install" with "uv pip install --python <venv_python>"
            cmd = cmd.replace("pip install", f"{uv_path} pip install --python {venv_python}")
            # Replace standalone "pip " at the start
            if cmd.startswith("pip "):
                cmd = f"{uv_path} pip --python {venv_python} {cmd[4:]}"
            return cmd

    # pip path.
    venv_pip = venv_bin / "pip"
    cmd = cmd.replace("pip install", f"{venv_pip} install")
    cmd = cmd.replace("python ", f"{venv_python} ")

    # If the command still starts with "pip", prefix with venv pip path.
    if cmd.startswith("pip "):
        cmd = f"{venv_pip} {cmd[4:]}"

    return cmd


def install_package(
    venv_python: Path,
    install_command: str,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    installer: InstallerBackend = InstallerBackend.PIP,
) -> subprocess.CompletedProcess[str]:
    """Install a package in the venv.

    The *install_command* is interpreted as a shell command with ``pip`` and
    ``python`` replaced by the venv paths (or ``uv pip`` when using uv).
    Runs in its own process group so the entire process tree is killed on
    timeout.

    Returns:
        The completed process.
    """
    cmd = _rewrite_install_command(install_command, venv_python, installer)
    log.debug("Install command (resolved): %s", cmd)
    return _run_in_process_group(cmd, cwd=str(cwd), env=env, timeout=timeout)


def run_test_command(
    venv_python: Path,
    test_command: str,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run a test command in the venv.

    Runs in its own process group so the entire process tree is killed
    on timeout.

    Returns:
        The completed process.
    """
    venv_bin = venv_python.parent

    cmd = test_command
    cmd = cmd.replace("python ", f"{venv_python} ")
    # Handle "python -m pytest" style commands.
    if cmd.startswith("python -m"):
        cmd = f"{venv_python} -m{cmd[len('python -m') :]}"

    # Ensure pytest/etc. from the venv is used.
    run_env = {**env, "PATH": f"{venv_bin}:{env.get('PATH', '')}"}

    log.debug("Test command (resolved): %s", cmd)
    return _run_in_process_group(cmd, cwd=str(cwd), env=run_env, timeout=timeout)


def get_installed_packages(
    venv_python: Path,
    env: dict[str, str],
    installer: InstallerBackend = InstallerBackend.PIP,
) -> dict[str, str]:
    """Get installed packages and versions from the venv."""
    try:
        if installer is InstallerBackend.UV:
            uv_path = detect_uv()
            if uv_path:
                cmd = [uv_path, "pip", "list", "--python", str(venv_python), "--format=json"]
            else:
                cmd = [str(venv_python), "-m", "pip", "list", "--format=json"]
        else:
            cmd = [str(venv_python), "-m", "pip", "list", "--format=json"]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env={**env, "ASAN_OPTIONS": "detect_leaks=0"},
        )
        if proc.returncode == 0:
            packages = json.loads(proc.stdout)
            return {p["name"]: p["version"] for p in packages}
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, OSError) as exc:
        log.info("Could not list installed packages: %s", exc)
    return {}


def get_package_version(package_name: str, installed: dict[str, str]) -> str | None:
    """Look up a package version from the installed packages dict.

    Performs case-insensitive matching and normalises hyphens/underscores.
    """
    normalised = package_name.lower().replace("-", "_")
    for name, version in installed.items():
        if name.lower().replace("-", "_") == normalised:
            return version
    return None


def install_with_fallback(
    python_path: Path,
    venv_dir: Path,
    install_command: str,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    installer: InstallerBackend,
) -> tuple[subprocess.CompletedProcess[str], InstallerBackend]:
    """Install a package with automatic fallback from uv to pip.

    If the initial install with the chosen backend fails and the backend
    is uv, deletes the venv, recreates it with pip, and retries.

    Returns:
        A tuple of (completed_process, actual_backend_used).
    """
    venv_python = venv_dir / "bin" / "python"
    proc = install_package(venv_python, install_command, cwd, env, timeout, installer)

    if proc.returncode == 0 or installer is InstallerBackend.PIP:
        return proc, installer

    # uv failed — fall back to pip.
    log.warning(
        "uv install failed (exit %d), falling back to pip for %s",
        proc.returncode,
        cwd.name,
    )
    shutil.rmtree(venv_dir, ignore_errors=True)
    create_venv(python_path, venv_dir, InstallerBackend.PIP)
    venv_python = venv_dir / "bin" / "python"
    pip_proc = install_package(
        venv_python, install_command, cwd, env, timeout, InstallerBackend.PIP
    )
    return pip_proc, InstallerBackend.PIP


def check_import(
    venv_python: Path,
    import_name: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Check that a package can be imported in the venv.

    Uses ``PYTHON_JIT=0`` to avoid JIT bugs during import validation.
    """
    import_env = {**env, "PYTHON_JIT": "0"}
    return subprocess.run(
        [str(venv_python), "-c", f"import {import_name}"],
        capture_output=True,
        text=True,
        timeout=30,
        env=import_env,
    )


# ---------------------------------------------------------------------------
# Per-package runner
# ---------------------------------------------------------------------------


def _resolve_dirs(pkg: PackageEntry, config: RunnerConfig) -> tuple[Path | None, Path, Path, bool]:
    """Determine repo/venv directories and whether a temp work_dir is used.

    Returns:
        (work_dir_or_None, repo_dir, venv_dir, is_temp)
        *work_dir_or_None* is a temp directory path only when ``is_temp`` is True.
    """
    if config.repos_dir is not None or config.venvs_dir is not None:
        # Persistent mode: use named subdirectories.
        repos_base = config.repos_dir or Path(tempfile.mkdtemp(prefix="labeille_repos_"))
        venvs_base = config.venvs_dir or Path(tempfile.mkdtemp(prefix="labeille_venvs_"))
        repos_base.mkdir(parents=True, exist_ok=True)
        venvs_base.mkdir(parents=True, exist_ok=True)
        return None, repos_base / pkg.package, venvs_base / pkg.package, False

    # Legacy temp-dir mode.
    work_dir = Path(tempfile.mkdtemp(prefix=f"labeille_{pkg.package}_"))
    return work_dir, work_dir / "repo", work_dir / "venv", True


def run_package(
    pkg: PackageEntry,
    config: RunnerConfig,
    run_dir: Path,
    env: dict[str, str],
    cancel_event: threading.Event | None = None,
) -> PackageResult:
    """Run a single package's test suite end-to-end.

    Handles: clone -> venv -> install -> test -> analyse result -> record.

    Args:
        pkg: The package configuration.
        config: The runner configuration.
        run_dir: The run output directory.
        env: The environment dict for subprocesses.
        cancel_event: If set, the worker should stop early.

    Returns:
        A :class:`PackageResult` with the outcome.
    """
    result = PackageResult(
        package=pkg.package,
        repo=pkg.repo,
        test_command=pkg.test_command,
        timestamp=utc_now_iso(),
    )

    per_pkg_timeout = pkg.timeout if pkg.timeout is not None else config.timeout

    work_dir, repo_dir, venv_dir, is_temp = _resolve_dirs(pkg, config)

    start = time.monotonic()
    try:
        result = _run_package_inner(
            pkg,
            config,
            run_dir,
            env,
            result,
            per_pkg_timeout,
            work_dir,
            repo_dir,
            venv_dir,
            cancel_event=cancel_event,
        )
    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        log.error("Unexpected error testing %s: %s", pkg.package, exc, exc_info=True)
    finally:
        result.duration_seconds = round(time.monotonic() - start, 2)
        if is_temp and not config.keep_work_dirs and work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# _run_package_inner phase helpers
# ---------------------------------------------------------------------------


def _ensure_repo(
    pkg: PackageEntry,
    config: RunnerConfig,
    result: PackageResult,
    repo_dir: Path,
) -> bool:
    """Resolve repo URL, clone or pull, and checkout revision. Return True on success."""
    # Apply repo URL override if specified.
    repo_url = pkg.repo
    if config.repo_overrides and pkg.package in config.repo_overrides:
        repo_url = config.repo_overrides[pkg.package]
        log.info("Using overridden repo for %s: %s", pkg.package, repo_url)
        result.repo = repo_url

    if not repo_url:
        result.status = "clone_error"
        result.error_message = "No repository URL"
        log.warning("Skipping %s: no repo URL", pkg.package)
        return False

    repo_existed = repo_dir.exists() and (repo_dir / ".git").exists()
    clone_start = time.monotonic()
    if repo_existed:
        log.info("Reusing repo for %s at %s (pulling)", pkg.package, repo_dir)
        try:
            result.git_revision = pull_repo(repo_dir)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            log.warning("Pull failed for %s, re-cloning: %s", pkg.package, exc)
            shutil.rmtree(repo_dir, ignore_errors=True)
            repo_existed = False

    if not repo_existed:
        depth = config.clone_depth_override
        if depth is None:
            depth = pkg.clone_depth
        if depth == 0:
            depth = None
        log.info("Cloning %s from %s to %s", pkg.package, repo_url, repo_dir)
        try:
            result.git_revision = clone_repo(repo_url, repo_dir, clone_depth=depth)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            result.status = "clone_error"
            result.error_message = f"Clone failed: {exc}"
            log.error("Clone failed for %s: %s", pkg.package, exc)
            return False

    clone_dur = round(time.monotonic() - clone_start, 2)
    log.debug("Git revision for %s: %s (%.2fs)", pkg.package, result.git_revision, clone_dur)

    # Checkout specific revision if requested.
    revision = config.revision_overrides.get(pkg.package)
    if revision:
        log.info("Checking out revision %s for %s", revision, pkg.package)
        commit = checkout_revision(repo_dir, revision)
        if commit is None:
            result.status = "error"
            result.error_message = f"Failed to checkout revision {revision}"
            return False
        result.git_revision = commit
        result.requested_revision = revision

    return True


def _run_install(
    pkg: PackageEntry,
    config: RunnerConfig,
    result: PackageResult,
    venv_dir: Path,
    repo_dir: Path,
    env: dict[str, str],
    per_pkg_timeout: int,
    installer: InstallerBackend,
    install_cmd: str,
    *,
    label: str = "",
) -> tuple[Path, InstallerBackend] | None:
    """Run install_with_fallback and handle errors. Return (venv_python, installer) or None."""
    install_start = time.monotonic()
    try:
        install_proc, actual_backend = install_with_fallback(
            config.target_python,
            venv_dir,
            install_cmd,
            repo_dir,
            env,
            per_pkg_timeout,
            installer,
        )
        venv_python = venv_dir / "bin" / "python"
        if actual_backend is not installer:
            result.installer_backend = actual_backend.value
            log.info(
                "Installer fell back from %s to %s for %s",
                installer.value,
                actual_backend.value,
                pkg.package,
            )
        installer = actual_backend
    except subprocess.TimeoutExpired:
        result.status = "install_error"
        result.error_message = f"Install timed out{label}"
        result.install_duration_seconds = round(time.monotonic() - install_start, 2)
        log.error("Install timed out for %s%s", pkg.package, label)
        return None
    except OSError as exc:
        result.status = "install_error"
        result.error_message = f"Install failed{label}: {exc}"
        result.install_duration_seconds = round(time.monotonic() - install_start, 2)
        log.error("Install failed for %s%s: %s", pkg.package, label, exc)
        return None

    result.install_duration_seconds = round(time.monotonic() - install_start, 2)

    if install_proc.returncode != 0:
        result.status = "install_error"
        result.exit_code = install_proc.returncode
        result.error_message = (
            install_proc.stderr[-500:] if install_proc.stderr else f"non-zero{label}"
        )
        log.error("Install failed for %s%s (exit %d)", pkg.package, label, install_proc.returncode)
        return None

    if install_proc.stdout:
        log.debug("Install stdout:\n%s", install_proc.stdout[-3000:])
    if install_proc.stderr:
        log.debug("Install stderr:\n%s", install_proc.stderr[-3000:])

    return venv_python, installer


def _analyze_test_result(
    test_proc: subprocess.CompletedProcess[str],
    result: PackageResult,
    run_dir: Path,
    pkg_name: str,
    test_dur: float,
) -> None:
    """Populate *result* with crash detection, pass/fail status."""
    crash = detect_crash(test_proc.returncode, test_proc.stderr)
    if crash is not None:
        result.status = "crash"
        result.signal = crash.signal_number
        result.crash_signature = crash.signature
        save_crash_stderr(run_dir, pkg_name, test_proc.stderr)
        log.warning(
            "CRASH in %s: %s (signal %d)",
            pkg_name,
            crash.signature,
            crash.signal_number,
        )
    elif test_proc.returncode == 0:
        result.status = "pass"
        log.info("PASS: %s (%.2fs)", pkg_name, test_dur)
    else:
        result.status = "fail"
        log.info("FAIL: %s (exit %d, %.2fs)", pkg_name, test_proc.returncode, test_dur)


def _run_package_inner(
    pkg: PackageEntry,
    config: RunnerConfig,
    run_dir: Path,
    env: dict[str, str],
    result: PackageResult,
    per_pkg_timeout: int,
    work_dir: Path | None,
    repo_dir: Path,
    venv_dir: Path,
    cancel_event: threading.Event | None = None,
) -> PackageResult:
    """Inner implementation of per-package testing (no cleanup responsibility)."""

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    # --- Clone or pull ---
    if _cancelled():
        result.status = "error"
        result.error_message = "Run stopped (crash limit reached)"
        return result

    if not _ensure_repo(pkg, config, result, repo_dir):
        return result

    # --- Sdist version alignment ---
    sdist_version: str | None = None
    sdist_tag_matched: bool | None = None
    source_layout: str = "unknown"
    import_name = pkg.import_name or pkg.package.replace("-", "_")

    if config.install_from == "sdist":
        result.install_from = "sdist"
        sdist_version = fetch_latest_pypi_version(pkg.package)
        if sdist_version:
            log.info("PyPI latest version for %s: %s", pkg.package, sdist_version)
            commit_hash, matched_tag = checkout_matching_tag(repo_dir, pkg.package, sdist_version)
            if commit_hash:
                result.git_revision = commit_hash
                sdist_tag_matched = True
                log.info(
                    "Checked out tag %s for %s (commit %s)",
                    matched_tag,
                    pkg.package,
                    commit_hash[:12],
                )
            else:
                sdist_tag_matched = False
                log.warning(
                    "No matching tag for %s version %s, staying on HEAD",
                    pkg.package,
                    sdist_version,
                )
        else:
            log.warning("Could not fetch PyPI version for %s", pkg.package)

        result.sdist_version = sdist_version
        result.sdist_tag_matched = sdist_tag_matched
        source_layout = detect_source_layout(repo_dir, import_name)
        log.debug("Source layout for %s: %s", pkg.package, source_layout)
    else:
        result.install_from = "source"

    # --- Create or reuse venv ---
    if _cancelled():
        result.status = "error"
        result.error_message = "Run stopped (crash limit reached)"
        return result

    venv_python = venv_dir / "bin" / "python"
    venv_existed = venv_dir.exists() and venv_python.exists()

    if venv_existed and config.refresh_venvs:
        log.info("Refreshing venv for %s at %s", pkg.package, venv_dir)
        shutil.rmtree(venv_dir)
        venv_existed = False

    # Resolve installer backend.
    installer = resolve_installer(config.installer)
    result.installer_backend = installer.value
    log.debug("Installer backend for %s: %s", pkg.package, installer.value)

    if venv_existed:
        log.info("Reusing venv for %s at %s", pkg.package, venv_dir)
    else:
        log.info("Creating venv for %s at %s", pkg.package, venv_dir)
        venv_start = time.monotonic()
        try:
            create_venv(config.target_python, venv_dir, installer)
        except (subprocess.CalledProcessError, OSError) as exc:
            result.status = "error"
            result.error_message = f"Venv creation failed: {exc}"
            log.error("Venv creation failed for %s: %s", pkg.package, exc)
            return result
        log.debug("Venv created for %s in %.2fs", pkg.package, time.monotonic() - venv_start)

    # --- Install (skip if reusing venv) ---
    if _cancelled():
        result.status = "error"
        result.error_message = "Run stopped (crash limit reached)"
        return result

    if venv_existed:
        log.info("Skipping install for %s (reusing venv)", pkg.package)
    elif config.install_from == "sdist":
        # --- Sdist install mode ---
        raw_install_cmd = pkg.install_command or "pip install -e ."
        sdist_install_cmd, deps_install_cmd = build_sdist_install_commands(
            pkg.package, raw_install_cmd
        )
        log.info("Installing %s from sdist: %s", pkg.package, sdist_install_cmd)

        install_result = _run_install(
            pkg,
            config,
            result,
            venv_dir,
            repo_dir,
            env,
            per_pkg_timeout,
            installer,
            sdist_install_cmd,
            label=" (sdist)",
        )
        if install_result is None:
            return result
        venv_python, installer = install_result
        log.debug(
            "Sdist install for %s completed in %.2fs",
            pkg.package,
            result.install_duration_seconds,
        )

        # Step 2: Install test dependencies from the repo.
        if deps_install_cmd:
            log.info("Installing test deps for %s: %s", pkg.package, deps_install_cmd)
            try:
                deps_proc = install_package(
                    venv_python, deps_install_cmd, repo_dir, env, per_pkg_timeout, installer
                )
                if deps_proc.returncode != 0:
                    log.warning(
                        "Test deps install had non-zero exit for %s (exit %d, continuing)",
                        pkg.package,
                        deps_proc.returncode,
                    )
            except (subprocess.TimeoutExpired, OSError) as exc:
                log.warning("Test deps install failed for %s: %s (continuing)", pkg.package, exc)
    else:
        # --- Source install mode (original behaviour) ---
        install_cmd = pkg.install_command or "pip install -e ."
        log.info("Installing %s: %s", pkg.package, install_cmd)

        install_result = _run_install(
            pkg,
            config,
            result,
            venv_dir,
            repo_dir,
            env,
            per_pkg_timeout,
            installer,
            install_cmd,
        )
        if install_result is None:
            return result
        venv_python, installer = install_result

    # --- Import check (skip if reusing venv) ---
    # Use shield_source_dir in sdist mode to prevent importing from local source.
    _shield = (
        shield_source_dir(repo_dir, import_name, source_layout)
        if config.install_from == "sdist"
        else nullcontext()
    )
    with _shield:
        if not venv_existed:
            log.info("Checking import for %s: import %s", pkg.package, import_name)
            try:
                import_proc = check_import(venv_python, import_name, env)
                if import_proc.returncode != 0:
                    result.status = "install_error"
                    stderr_msg = import_proc.stderr.strip()[-200:] if import_proc.stderr else ""
                    result.error_message = f"Package installed but import failed: {stderr_msg}"
                    log.error(
                        "Import check failed for %s: %s",
                        pkg.package,
                        result.error_message,
                    )
                    return result
            except subprocess.TimeoutExpired:
                result.status = "install_error"
                result.error_message = "Package installed but import timed out"
                log.error("Import check timed out for %s", pkg.package)
                return result
            except OSError as exc:
                log.warning("Import check failed for %s: %s (continuing)", pkg.package, exc)

    # --- Install extra dependencies if specified ---
    if config.extra_deps and not venv_existed:
        extra_cmd = f"pip install {' '.join(config.extra_deps)}"
        log.info("Installing extra deps for %s: %s", pkg.package, extra_cmd)
        try:
            extra_proc = install_package(
                venv_python, extra_cmd, repo_dir, env, per_pkg_timeout, installer
            )
            if extra_proc.returncode != 0:
                log.warning(
                    "Extra deps install failed for %s (exit %d, non-fatal)",
                    pkg.package,
                    extra_proc.returncode,
                )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("Failed to install extra deps for %s: %s", pkg.package, exc)

    # --- Collect installed packages ---
    result.installed_dependencies = get_installed_packages(venv_python, env, installer)
    result.package_version = get_package_version(pkg.package, result.installed_dependencies)
    if result.installed_dependencies:
        dep_count = len(result.installed_dependencies)
        log.debug(
            "Installed %d packages for %s (version: %s)",
            dep_count,
            pkg.package,
            result.package_version or "unknown",
        )
        dep_lines = [f"  {n}=={v}" for n, v in sorted(result.installed_dependencies.items())]
        log.debug("Dependency list:\n%s", "\n".join(dep_lines))

    # --- Run tests ---
    if _cancelled():
        result.status = "error"
        result.error_message = "Run stopped (crash limit reached)"
        return result

    test_cmd = pkg.test_command or "python -m pytest"
    if config.test_command_override:
        test_cmd = config.test_command_override
    elif config.test_command_suffix:
        test_cmd = f"{test_cmd} {config.test_command_suffix}"
    result.test_command = test_cmd
    log.info("Running tests for %s: %s", pkg.package, test_cmd)
    log.debug("Test timeout: %ds", per_pkg_timeout)

    # Shield source directory in sdist mode to prevent local imports.
    _test_shield = (
        shield_source_dir(repo_dir, import_name, source_layout)
        if config.install_from == "sdist"
        else nullcontext()
    )
    with _test_shield:
        test_start = time.monotonic()
        try:
            test_proc = run_test_command(venv_python, test_cmd, repo_dir, env, per_pkg_timeout)
        except subprocess.TimeoutExpired as exc:
            result.status = "timeout"
            result.timeout_hit = True
            result.stderr_tail = (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else ""
            log.warning("Tests timed out for %s after %ds", pkg.package, per_pkg_timeout)
            return result

    test_dur = round(time.monotonic() - test_start, 2)
    result.exit_code = test_proc.returncode
    result.stderr_tail = test_proc.stderr[-2000:] if test_proc.stderr else ""

    log.debug("Tests for %s exited %d in %.2fs", pkg.package, test_proc.returncode, test_dur)
    if test_proc.stdout:
        log.debug("Test stdout:\n%s", test_proc.stdout[-5000:])
    if test_proc.stderr:
        log.debug("Test stderr:\n%s", test_proc.stderr[-5000:])

    # --- Analyse result ---
    _analyze_test_result(test_proc, result, run_dir, pkg.package, test_dur)
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def extract_python_minor_version(version_string: str) -> str:
    """Extract the ``major.minor`` version from a full Python version string.

    For example, ``"3.15.0a5+ (heads/main:abc1234)"`` returns ``"3.15"``.
    """
    parts = version_string.strip().split(".")
    if len(parts) >= 2:
        # The minor component may contain alpha/rc suffixes — strip non-digits.
        minor = ""
        for ch in parts[1]:
            if ch.isdigit():
                minor += ch
            else:
                break
        if parts[0].isdigit() and minor:
            return f"{parts[0]}.{minor}"
    return version_string


def filter_packages(
    index: Index,
    registry_dir: Path,
    config: RunnerConfig,
) -> tuple[list[PackageEntry], int]:
    """Select and load packages to test based on config filters.

    Returns a tuple of (loaded :class:`PackageEntry` objects in download-count
    order, count of packages skipped due to version-specific skip_versions).
    """
    entries = list(index.packages)

    # --top N
    if config.top_n is not None:
        entries = entries[: config.top_n]

    # --packages filter
    if config.packages_filter:
        allowed = {n.lower() for n in config.packages_filter}
        entries = [e for e in entries if e.name.lower() in allowed]

    packages: list[PackageEntry] = []
    version_skipped = 0
    py_ver = config.target_python_version

    for entry in entries:
        # Fast skip checks from index (no YAML load needed).
        if not config.force_run and entry.skip:
            log.debug("Skipping %s (marked skip in index)", entry.name)
            continue
        if not config.force_run and py_ver and py_ver in entry.skip_versions_keys:
            log.debug("Skipping %s (skip_versions[%s] in index)", entry.name, py_ver)
            version_skipped += 1
            continue

        # Now load the full YAML (only for non-skipped packages).
        if not package_exists(entry.name, registry_dir):
            log.debug("Skipping %s (no package YAML)", entry.name)
            continue
        pkg = load_package(entry.name, registry_dir)

        # Double-check against the full package data (in case index is stale).
        if not config.force_run and pkg.skip:
            log.debug("Skipping %s (marked skip in package)", pkg.package)
            continue
        if not config.force_run and py_ver and pkg.skip_versions:
            reason = pkg.skip_versions.get(py_ver)
            if reason:
                log.debug("Skipping %s (skip_versions[%s]: %s)", pkg.package, py_ver, reason)
                version_skipped += 1
                continue

        if config.skip_extensions and pkg.extension_type == "extensions":
            log.debug("Skipping %s (extension package)", pkg.package)
            continue
        packages.append(pkg)

    return packages, version_skipped


def _update_summary(summary: RunSummary, result: PackageResult) -> None:
    """Update summary counters from a single result."""
    summary.tested += 1
    if result.status == "pass":
        summary.passed += 1
    elif result.status == "fail":
        summary.failed += 1
    elif result.status == "crash":
        summary.crashed += 1
    elif result.status == "timeout":
        summary.timed_out += 1
    elif result.status == "install_error":
        summary.install_errors += 1
    elif result.status == "clone_error":
        summary.clone_errors += 1
    else:
        summary.errors += 1


def _log_parallel_progress(
    result: PackageResult,
    completed_count: int,
    total: int,
    crashes: int,
    config: RunnerConfig,
) -> None:
    """Log a one-line progress update for parallel mode."""
    duration = f"{result.duration_seconds:.0f}s"
    if result.status == "crash":
        sig_name = ""
        if result.signal is not None:
            sig_name = f" SIG{result.signal}"
        line = f"[{completed_count}/{total}] CRASH {result.package} ({duration}){sig_name}"
        if result.crash_signature:
            line += f": {result.crash_signature}"
        log.warning(line)
    elif config.quiet:
        return  # quiet mode: only print crashes
    elif result.status == "pass":
        log.info("[%d/%d] pass %s (%s)", completed_count, total, result.package, duration)
    elif result.status == "fail":
        log.info(
            "[%d/%d] fail %s (exit %d, %s)",
            completed_count,
            total,
            result.package,
            result.exit_code,
            duration,
        )
    else:
        log.info(
            "[%d/%d] %s %s (%s)",
            completed_count,
            total,
            result.status,
            result.package,
            duration,
        )


def _run_all_sequential(
    packages: list[PackageEntry],
    config: RunnerConfig,
    run_dir: Path,
    env: dict[str, str],
    summary: RunSummary,
    completed: set[str],
) -> list[PackageResult]:
    """Sequential execution path (workers=1)."""
    results: list[PackageResult] = []
    crashes_found = 0

    for pkg in packages:
        if pkg.package in completed:
            log.info("Skipping %s: already completed in this run", pkg.package)
            summary.skipped += 1
            continue

        if config.dry_run:
            log.info("[dry-run] Would test: %s", pkg.package)
            summary.skipped += 1
            continue

        if config.stop_after_crash is not None and crashes_found >= config.stop_after_crash:
            log.info("Stopping: reached %d crash(es)", crashes_found)
            break

        log.info("--- Testing %s (%d/%d) ---", pkg.package, summary.tested + 1, summary.total)
        result = run_package(pkg, config, run_dir, env)
        results.append(result)
        append_result(run_dir, result)
        log.debug(
            "Result for %s: %s (%.2fs total)",
            pkg.package,
            result.status,
            result.duration_seconds,
        )

        _update_summary(summary, result)
        if result.status == "crash":
            crashes_found += 1

    return results


def _run_all_parallel(
    packages: list[PackageEntry],
    config: RunnerConfig,
    run_dir: Path,
    env: dict[str, str],
    summary: RunSummary,
    completed: set[str],
) -> list[PackageResult]:
    """Parallel execution path (workers>=2)."""
    results: list[PackageResult] = []
    results_lock = threading.Lock()
    cancel_event = threading.Event()
    crashes_found = 0
    completed_count = 0

    # Filter out already-completed and dry-run packages before submitting.
    to_run: list[PackageEntry] = []
    for pkg in packages:
        if pkg.package in completed:
            log.info("Skipping %s: already completed in this run", pkg.package)
            summary.skipped += 1
            continue
        if config.dry_run:
            log.info("[dry-run] Would test: %s", pkg.package)
            summary.skipped += 1
            continue
        to_run.append(pkg)

    def _worker(pkg: PackageEntry) -> PackageResult:
        nonlocal crashes_found, completed_count

        if cancel_event.is_set():
            return PackageResult(
                package=pkg.package,
                repo=pkg.repo,
                status="error",
                error_message="Run stopped (crash limit reached)",
                timestamp=utc_now_iso(),
            )

        result = run_package(pkg, config, run_dir, env, cancel_event=cancel_event)

        with results_lock:
            append_result(run_dir, result)
            completed_count += 1
            if result.status == "crash":
                crashes_found += 1
                if (
                    config.stop_after_crash is not None
                    and crashes_found >= config.stop_after_crash
                ):
                    cancel_event.set()
            _log_parallel_progress(result, completed_count, len(to_run), crashes_found, config)

        return result

    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {pool.submit(_worker, pkg): pkg for pkg in to_run}
        for future in as_completed(futures):
            pkg = futures[future]
            try:
                result = future.result()
                results.append(result)
                _update_summary(summary, result)
            except Exception as exc:
                log.error("Worker exception for %s: %s", pkg.package, exc, exc_info=True)
                error_result = PackageResult(
                    package=pkg.package,
                    repo=pkg.repo,
                    status="error",
                    error_message=f"Worker exception: {exc}",
                    timestamp=utc_now_iso(),
                )
                results.append(error_result)
                _update_summary(summary, error_result)

    return results


def run_all(config: RunnerConfig) -> RunOutput:
    """Run test suites for all selected packages.

    When ``config.workers`` is 1 (default), packages are tested sequentially.
    When ``config.workers`` >= 2, packages are tested in parallel using a
    thread pool.  All CPU-intensive work happens in subprocesses, so the GIL
    is not a bottleneck.

    Args:
        config: The runner configuration.

    Returns:
        A :class:`RunOutput` with results, summary, and run metadata.
    """
    run_start = time.monotonic()
    summary = RunSummary()

    # Load index and filter.
    index = load_index(config.registry_dir)
    packages, version_skipped = filter_packages(index, config.registry_dir, config)
    summary.total = len(packages)
    summary.version_skipped = version_skipped

    log.debug(
        "Selected %d packages from registry (%d in index)", len(packages), len(index.packages)
    )
    if packages:
        log.debug("Packages: %s", ", ".join(p.package for p in packages))

    # Set up run directory.
    run_dir = create_run_dir(config.results_dir, config.run_id)
    log.debug("Run directory: %s", run_dir)
    if config.repos_dir:
        log.debug("Repos directory: %s", config.repos_dir)
    if config.venvs_dir:
        log.debug("Venvs directory: %s", config.venvs_dir)

    # Validate target Python.
    python_version = validate_target_python(config.target_python)
    jit_enabled = check_jit_enabled(config.target_python)
    log.info("Target Python: %s", python_version)
    log.info("JIT enabled: %s", jit_enabled)
    log.debug("Target binary: %s", config.target_python)
    if config.workers > 1:
        log.info("Parallel mode: %d workers", config.workers)

    started_at = utc_now_iso()
    write_run_meta(run_dir, config, python_version, jit_enabled, started_at=started_at)

    # Resumability: load already-completed packages.
    completed: set[str] = set()
    if config.skip_completed:
        completed = load_completed_packages(run_dir)
        if completed:
            log.info("Resuming: %d packages already completed", len(completed))

    env = build_env(config)

    workers = max(1, config.workers)
    if workers == 1:
        results = _run_all_sequential(packages, config, run_dir, env, summary, completed)
    else:
        results = _run_all_parallel(packages, config, run_dir, env, summary, completed)

    # Finalise run metadata.
    total_duration = round(time.monotonic() - run_start, 2)
    finished_at = utc_now_iso()
    write_run_meta(
        run_dir,
        config,
        python_version,
        jit_enabled,
        summary=summary,
        started_at=started_at,
        finished_at=finished_at,
    )

    # Write verbose summary to file and log.
    from labeille.summary import format_summary

    summary_text = format_summary(
        results,
        summary,
        config,
        python_version,
        jit_enabled,
        total_duration,
        run_dir=run_dir,
        mode="verbose",
    )
    (run_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    log.info("Run summary:\n%s", summary_text)

    return RunOutput(
        results=results,
        summary=summary,
        total_duration=total_duration,
        python_version=python_version,
        jit_enabled=jit_enabled,
        run_dir=run_dir,
    )
