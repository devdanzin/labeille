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
import time
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
    return env


def clone_repo(repo_url: str, dest: Path) -> str | None:
    """Clone a git repository (shallow, depth=1) and return the HEAD revision.

    Args:
        repo_url: The URL of the git repository.
        dest: The destination directory for the clone.

    Returns:
        The HEAD commit hash, or ``None`` on failure.

    Raises:
        subprocess.CalledProcessError: If the clone fails.
    """
    subprocess.run(
        ["git", "clone", "--depth=1", repo_url, str(dest)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
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


def create_venv(python_path: Path, venv_dir: Path) -> None:
    """Create a virtual environment using the target Python.

    Raises:
        subprocess.CalledProcessError: If venv creation fails.
    """
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
    subprocess.run(
        [str(venv_python), "-m", "ensurepip", "--upgrade"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # ensurepip may fail on some builds; pip might already exist
        env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"},
    )


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


# ---------------------------------------------------------------------------
# Per-package runner
# ---------------------------------------------------------------------------


def run_package(
    pkg: PackageEntry,
    config: RunnerConfig,
    run_dir: Path,
    env: dict[str, str],
) -> PackageResult:
    """Run a single package's test suite end-to-end.

    Handles: clone -> venv -> install -> test -> analyse result -> record.

    Args:
        pkg: The package configuration.
        config: The runner configuration.
        run_dir: The run output directory.
        env: The environment dict for subprocesses.

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

    work_dir = Path(tempfile.mkdtemp(prefix=f"labeille_{pkg.package}_"))
    repo_dir = work_dir / "repo"
    venv_dir = work_dir / "venv"

    start = time.monotonic()
    try:
        result = _run_package_inner(
            pkg, config, run_dir, env, result, per_pkg_timeout, work_dir, repo_dir, venv_dir
        )
    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        log.error("Unexpected error testing %s: %s", pkg.package, exc)
    finally:
        result.duration_seconds = round(time.monotonic() - start, 2)
        if not config.keep_work_dirs:
            shutil.rmtree(work_dir, ignore_errors=True)

    return result


def _run_package_inner(
    pkg: PackageEntry,
    config: RunnerConfig,
    run_dir: Path,
    env: dict[str, str],
    result: PackageResult,
    per_pkg_timeout: int,
    work_dir: Path,
    repo_dir: Path,
    venv_dir: Path,
) -> PackageResult:
    """Inner implementation of per-package testing (no cleanup responsibility)."""
    # --- Clone ---
    if not pkg.repo:
        result.status = "clone_error"
        result.error_message = "No repository URL"
        log.warning("Skipping %s: no repo URL", pkg.package)
        return result

    log.info("Cloning %s from %s", pkg.package, pkg.repo)
    try:
        result.git_revision = clone_repo(pkg.repo, repo_dir)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        result.status = "clone_error"
        result.error_message = f"Clone failed: {exc}"
        log.error("Clone failed for %s: %s", pkg.package, exc)
        return result

    # --- Create venv ---
    log.info("Creating venv for %s", pkg.package)
    try:
        create_venv(config.target_python, venv_dir)
    except (subprocess.CalledProcessError, OSError) as exc:
        result.status = "error"
        result.error_message = f"Venv creation failed: {exc}"
        log.error("Venv creation failed for %s: %s", pkg.package, exc)
        return result

    venv_python = venv_dir / "bin" / "python"

    # --- Install ---
    install_cmd = pkg.install_command or "pip install -e ."
    log.info("Installing %s: %s", pkg.package, install_cmd)
    install_start = time.monotonic()
    try:
        install_proc = install_package(venv_python, install_cmd, repo_dir, env, per_pkg_timeout)
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

    if install_proc.returncode != 0:
        result.status = "install_error"
        result.exit_code = install_proc.returncode
        result.error_message = install_proc.stderr[-500:] if install_proc.stderr else "non-zero"
        log.error("Install failed for %s (exit %d)", pkg.package, install_proc.returncode)
        return result

    # --- Collect installed packages ---
    result.installed_dependencies = get_installed_packages(venv_python, env)
    result.package_version = get_package_version(pkg.package, result.installed_dependencies)

    # --- Run tests ---
    test_cmd = pkg.test_command or "python -m pytest"
    result.test_command = test_cmd
    log.info("Running tests for %s: %s", pkg.package, test_cmd)
    try:
        test_proc = run_test_command(venv_python, test_cmd, repo_dir, env, per_pkg_timeout)
    except subprocess.TimeoutExpired as exc:
        result.status = "timeout"
        result.timeout_hit = True
        result.stderr_tail = (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else ""
        log.warning("Tests timed out for %s after %ds", pkg.package, per_pkg_timeout)
        return result

    result.exit_code = test_proc.returncode
    result.stderr_tail = test_proc.stderr[-2000:] if test_proc.stderr else ""

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
        log.info("PASS: %s", pkg.package)
    else:
        result.status = "fail"
        log.info("FAIL: %s (exit %d)", pkg.package, test_proc.returncode)

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def filter_packages(
    index: Index,
    registry_dir: Path,
    config: RunnerConfig,
) -> list[PackageEntry]:
    """Select and load packages to test based on config filters.

    Returns loaded :class:`PackageEntry` objects in download-count order.
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
    for entry in entries:
        if entry.skip:
            log.debug("Skipping %s (marked skip in index)", entry.name)
            continue
        if not package_exists(entry.name, registry_dir):
            log.debug("Skipping %s (no package YAML)", entry.name)
            continue
        pkg = load_package(entry.name, registry_dir)
        if pkg.skip:
            log.debug("Skipping %s (marked skip in package)", pkg.package)
            continue
        if config.skip_extensions and pkg.extension_type == "extensions":
            log.debug("Skipping %s (extension package)", pkg.package)
            continue
        packages.append(pkg)

    return packages


def run_all(config: RunnerConfig) -> tuple[list[PackageResult], RunSummary]:
    """Run test suites for all selected packages.

    Args:
        config: The runner configuration.

    Returns:
        A tuple of (results list, summary).
    """
    summary = RunSummary()

    # Load index and filter.
    index = load_index(config.registry_dir)
    packages = filter_packages(index, config.registry_dir, config)
    summary.total = len(packages)

    # Set up run directory.
    run_dir = create_run_dir(config.results_dir, config.run_id)

    # Validate target Python.
    python_version = validate_target_python(config.target_python)
    jit_enabled = check_jit_enabled(config.target_python)
    log.info("Target Python: %s", python_version)
    log.info("JIT enabled: %s", jit_enabled)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_run_meta(run_dir, config, python_version, jit_enabled, started_at=started_at)

    # Resumability: load already-completed packages.
    completed: set[str] = set()
    if config.skip_completed:
        completed = load_completed_packages(run_dir)
        if completed:
            log.info("Resuming: %d packages already completed", len(completed))

    env = build_env(config)
    results: list[PackageResult] = []
    crashes_found = 0

    for pkg in packages:
        # Skip completed.
        if pkg.package in completed:
            log.info("Skipping %s: already completed in this run", pkg.package)
            summary.skipped += 1
            continue

        if config.dry_run:
            log.info("[dry-run] Would test: %s", pkg.package)
            summary.skipped += 1
            continue

        # Stop after N crashes.
        if config.stop_after_crash is not None and crashes_found >= config.stop_after_crash:
            log.info("Stopping: reached %d crash(es)", crashes_found)
            break

        log.info("--- Testing %s ---", pkg.package)
        result = run_package(pkg, config, run_dir, env)
        results.append(result)
        append_result(run_dir, result)

        # Update summary.
        summary.tested += 1
        if result.status == "pass":
            summary.passed += 1
        elif result.status == "fail":
            summary.failed += 1
        elif result.status == "crash":
            summary.crashed += 1
            crashes_found += 1
        elif result.status == "timeout":
            summary.timed_out += 1
        elif result.status == "install_error":
            summary.install_errors += 1
        elif result.status == "clone_error":
            summary.clone_errors += 1
        else:
            summary.errors += 1

    # Finalise run metadata.
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

    return results, summary
