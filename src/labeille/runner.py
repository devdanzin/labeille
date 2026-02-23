"""Test runner for executing package test suites against JIT-enabled CPython.

This module handles cloning repositories, installing packages into a
JIT-enabled Python environment, running their test suites, and capturing
the results including any crashes (segfaults, aborts, assertion failures).
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from labeille.crash import detect_crash
from labeille.logging import get_logger
from labeille.registry import Index, PackageEntry, load_index, load_package, package_exists

log = get_logger("runner")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Top-level configuration for a test run."""

    target_python: Path
    registry_dir: Path
    results_dir: Path
    run_id: str
    timeout: int = 600
    top_n: int | None = None
    packages_filter: list[str] | None = None
    skip_extensions: bool = False
    skip_completed: bool = False
    stop_after_crash: int | None = None
    env_overrides: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    verbose: bool = False
    quiet: bool = False
    keep_work_dirs: bool = False
    repos_dir: Path | None = None
    venvs_dir: Path | None = None
    refresh_venvs: bool = False
    force_run: bool = False
    target_python_version: str = ""
    workers: int = 1
    cli_args: list[str] = field(default_factory=list)


@dataclass
class PackageResult:
    """Result of testing a single package."""

    package: str
    repo: str | None = None
    package_version: str | None = None
    git_revision: str | None = None
    status: str = "error"  # pass | fail | crash | timeout | install_error | clone_error | error
    exit_code: int = -1
    signal: int | None = None
    crash_signature: str | None = None
    duration_seconds: float = 0.0
    install_duration_seconds: float = 0.0
    test_command: str = ""
    timeout_hit: bool = False
    stderr_tail: str = ""
    installed_dependencies: dict[str, str] = field(default_factory=dict)
    error_message: str | None = None
    timestamp: str = ""


@dataclass
class RunSummary:
    """Aggregate summary of a test run."""

    total: int = 0
    tested: int = 0
    passed: int = 0
    failed: int = 0
    crashed: int = 0
    timed_out: int = 0
    install_errors: int = 0
    clone_errors: int = 0
    errors: int = 0
    skipped: int = 0
    version_skipped: int = 0


@dataclass
class RunOutput:
    """Complete output from a test run."""

    results: list[PackageResult]
    summary: RunSummary
    total_duration: float
    python_version: str
    jit_enabled: bool
    run_dir: Path = field(default_factory=lambda: Path("."))


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
    }
    if summary is not None:
        meta["packages_tested"] = summary.tested
        meta["packages_skipped"] = summary.skipped
        meta["crashes_found"] = summary.crashed
        meta["total_duration_seconds"] = 0.0  # filled by caller
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def append_result(run_dir: Path, result: PackageResult) -> None:
    """Append a single result as a JSON line to results.jsonl."""
    data: dict[str, Any] = {
        "package": result.package,
        "repo": result.repo,
        "package_version": result.package_version,
        "git_revision": result.git_revision,
        "status": result.status,
        "exit_code": result.exit_code,
        "signal": result.signal,
        "crash_signature": result.crash_signature,
        "duration_seconds": result.duration_seconds,
        "install_duration_seconds": result.install_duration_seconds,
        "test_command": result.test_command,
        "timeout_hit": result.timeout_hit,
        "stderr_tail": result.stderr_tail,
        "installed_dependencies": result.installed_dependencies,
        "error_message": result.error_message,
        "timestamp": result.timestamp,
    }
    with open(run_dir / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def load_completed_packages(run_dir: Path) -> set[str]:
    """Load the set of package names already recorded in results.jsonl."""
    results_file = run_dir / "results.jsonl"
    if not results_file.exists():
        return set()
    completed: set[str] = set()
    for line in results_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                completed.add(data["package"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def save_crash_stderr(run_dir: Path, package_name: str, stderr: str) -> None:
    """Save stderr output for a crashed package."""
    crash_file = run_dir / "crashes" / f"{package_name}.stderr"
    crash_file.write_text(stderr, encoding="utf-8")


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
            env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"},
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
    """Check if the JIT is enabled in the target Python build."""
    try:
        proc = subprocess.run(
            [
                str(python_path),
                "-c",
                "import sys; print(hasattr(sys, '_jit') or hasattr(sys, 'flags') "
                "and getattr(getattr(sys, 'flags', None), 'jit', False))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHON_JIT": "1", "ASAN_OPTIONS": "detect_leaks=0"},
        )
        return "True" in proc.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def build_env(config: RunnerConfig) -> dict[str, str]:
    """Build the environment dict for test subprocesses."""
    env = {**os.environ}
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


def clone_repo(repo_url: str, dest: Path, clone_depth: int | None = None) -> str | None:
    """Clone a git repository and return the HEAD revision.

    Args:
        repo_url: The URL of the git repository.
        dest: The destination directory for the clone.
        clone_depth: Clone depth. ``None`` means shallow (depth=1).
            A positive integer uses that depth.

    Returns:
        The HEAD commit hash, or ``None`` on failure.

    Raises:
        subprocess.CalledProcessError: If the clone fails.
    """
    depth = clone_depth if clone_depth is not None else 1
    log.debug("Running: git clone --depth=%d %s %s", depth, repo_url, dest)
    clone_proc = subprocess.run(
        ["git", "clone", f"--depth={depth}", repo_url, str(dest)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    if clone_proc.stderr:
        log.debug("git clone stderr: %s", clone_proc.stderr.strip())

    # Fetch tags when using deeper clones (needed for setuptools-scm etc.).
    if clone_depth is not None and clone_depth > 1:
        log.debug("Fetching tags for %s (clone_depth=%d)", dest, clone_depth)
        fetch_proc = subprocess.run(
            ["git", "fetch", "--tags"],
            capture_output=True,
            text=True,
            cwd=str(dest),
            timeout=120,
            check=False,
        )
        if fetch_proc.returncode != 0:
            log.debug("git fetch --tags failed (non-fatal): %s", fetch_proc.stderr.strip())

    # Get the revision.
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(dest),
        timeout=10,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return None


def pull_repo(dest: Path) -> str | None:
    """Pull latest changes in an existing repo clone and return the HEAD revision.

    Args:
        dest: The directory containing the existing clone.

    Returns:
        The HEAD commit hash, or ``None`` on failure.

    Raises:
        subprocess.CalledProcessError: If the pull fails.
    """
    log.debug("Running: git pull --ff-only (in %s)", dest)
    pull_proc = subprocess.run(
        ["git", "pull", "--ff-only"],
        capture_output=True,
        text=True,
        cwd=str(dest),
        timeout=120,
        check=True,
    )
    if pull_proc.stdout.strip():
        log.debug("git pull: %s", pull_proc.stdout.strip())
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(dest),
        timeout=10,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return None


def create_venv(python_path: Path, venv_dir: Path) -> None:
    """Create a virtual environment using the target Python.

    Raises:
        subprocess.CalledProcessError: If venv creation fails.
    """
    log.debug("Running: %s -m venv %s", python_path, venv_dir)
    subprocess.run(
        [str(python_path), "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"},
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
        env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"},
    )
    if ensurepip_proc.returncode != 0:
        log.debug("ensurepip exited %d (non-fatal)", ensurepip_proc.returncode)


def install_package(
    venv_python: Path,
    install_command: str,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Install a package in the venv.

    The *install_command* is interpreted as a shell command with ``pip`` and
    ``python`` replaced by the venv paths.

    Returns:
        The completed process.
    """
    venv_bin = venv_python.parent
    venv_pip = venv_bin / "pip"

    cmd = install_command
    cmd = cmd.replace("pip install", f"{venv_pip} install")
    cmd = cmd.replace("python ", f"{venv_python} ")

    # If the command still starts with "pip", prefix with venv pip path.
    if cmd.startswith("pip "):
        cmd = f"{venv_pip} {cmd[4:]}"

    log.debug("Install command (resolved): %s", cmd)
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=timeout,
        env=env,
    )


def run_test_command(
    venv_python: Path,
    test_command: str,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run a test command in the venv.

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
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=timeout,
        env=run_env,
    )


def get_installed_packages(venv_python: Path, env: dict[str, str]) -> dict[str, str]:
    """Get installed packages and versions from the venv."""
    try:
        proc = subprocess.run(
            [str(venv_python), "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**env, "ASAN_OPTIONS": "detect_leaks=0"},
        )
        if proc.returncode == 0:
            packages = json.loads(proc.stdout)
            return {p["name"]: p["version"] for p in packages}
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, OSError):
        pass
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
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        log.error("Unexpected error testing %s: %s", pkg.package, exc)
    finally:
        result.duration_seconds = round(time.monotonic() - start, 2)
        if is_temp and not config.keep_work_dirs and work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

    return result


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

    if not pkg.repo:
        result.status = "clone_error"
        result.error_message = "No repository URL"
        log.warning("Skipping %s: no repo URL", pkg.package)
        return result

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
        log.info("Cloning %s from %s to %s", pkg.package, pkg.repo, repo_dir)
        try:
            result.git_revision = clone_repo(pkg.repo, repo_dir, clone_depth=pkg.clone_depth)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            result.status = "clone_error"
            result.error_message = f"Clone failed: {exc}"
            log.error("Clone failed for %s: %s", pkg.package, exc)
            return result

    clone_dur = round(time.monotonic() - clone_start, 2)
    log.debug("Git revision for %s: %s (%.2fs)", pkg.package, result.git_revision, clone_dur)

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

    if venv_existed:
        log.info("Reusing venv for %s at %s", pkg.package, venv_dir)
    else:
        log.info("Creating venv for %s at %s", pkg.package, venv_dir)
        venv_start = time.monotonic()
        try:
            create_venv(config.target_python, venv_dir)
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
    else:
        install_cmd = pkg.install_command or "pip install -e ."
        log.info("Installing %s: %s", pkg.package, install_cmd)
        install_start = time.monotonic()
        try:
            install_proc = install_package(
                venv_python, install_cmd, repo_dir, env, per_pkg_timeout
            )
        except subprocess.TimeoutExpired:
            result.status = "install_error"
            result.error_message = "Install timed out"
            result.install_duration_seconds = round(time.monotonic() - install_start, 2)
            log.error("Install timed out for %s", pkg.package)
            return result
        except OSError as exc:
            result.status = "install_error"
            result.error_message = f"Install failed: {exc}"
            result.install_duration_seconds = round(time.monotonic() - install_start, 2)
            log.error("Install failed for %s: %s", pkg.package, exc)
            return result

        result.install_duration_seconds = round(time.monotonic() - install_start, 2)
        log.debug(
            "Install for %s exited %d in %.2fs",
            pkg.package,
            install_proc.returncode,
            result.install_duration_seconds,
        )
        if install_proc.stdout:
            log.debug("Install stdout:\n%s", install_proc.stdout[-3000:])
        if install_proc.stderr:
            log.debug("Install stderr:\n%s", install_proc.stderr[-3000:])

        if install_proc.returncode != 0:
            result.status = "install_error"
            result.exit_code = install_proc.returncode
            result.error_message = (
                install_proc.stderr[-500:] if install_proc.stderr else "non-zero"
            )
            log.error("Install failed for %s (exit %d)", pkg.package, install_proc.returncode)
            return result

    # --- Import check (skip if reusing venv) ---
    if not venv_existed:
        import_name = pkg.import_name or pkg.package.replace("-", "_")
        log.info("Checking import for %s: import %s", pkg.package, import_name)
        try:
            import_proc = check_import(venv_python, import_name, env)
            if import_proc.returncode != 0:
                result.status = "install_error"
                stderr_msg = import_proc.stderr.strip()[-200:] if import_proc.stderr else ""
                result.error_message = f"Package installed but import failed: {stderr_msg}"
                log.error("Import check failed for %s: %s", pkg.package, result.error_message)
                return result
        except subprocess.TimeoutExpired:
            result.status = "install_error"
            result.error_message = "Package installed but import timed out"
            log.error("Import check timed out for %s", pkg.package)
            return result
        except OSError as exc:
            log.warning("Import check failed for %s: %s (continuing)", pkg.package, exc)

    # --- Collect installed packages ---
    result.installed_dependencies = get_installed_packages(venv_python, env)
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
    result.test_command = test_cmd
    log.info("Running tests for %s: %s", pkg.package, test_cmd)
    log.debug("Test timeout: %ds", per_pkg_timeout)
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
    crash = detect_crash(test_proc.returncode, test_proc.stderr)
    if crash is not None:
        result.status = "crash"
        result.signal = crash.signal_number
        result.crash_signature = crash.signature
        save_crash_stderr(run_dir, pkg.package, test_proc.stderr)
        log.warning(
            "CRASH in %s: %s (signal %d)",
            pkg.package,
            crash.signature,
            crash.signal_number,
        )
    elif test_proc.returncode == 0:
        result.status = "pass"
        log.info("PASS: %s (%.2fs)", pkg.package, test_dur)
    else:
        result.status = "fail"
        log.info("FAIL: %s (exit %d, %.2fs)", pkg.package, test_proc.returncode, test_dur)

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
        if not config.force_run and entry.skip:
            log.debug("Skipping %s (marked skip in index)", entry.name)
            continue
        if not package_exists(entry.name, registry_dir):
            log.debug("Skipping %s (no package YAML)", entry.name)
            continue
        pkg = load_package(entry.name, registry_dir)
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
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
                log.error("Worker exception for %s: %s", pkg.package, exc)
                error_result = PackageResult(
                    package=pkg.package,
                    repo=pkg.repo,
                    status="error",
                    error_message=f"Worker exception: {exc}",
                    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
