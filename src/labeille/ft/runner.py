"""Free-threading test execution engine.

Runs each package's test suite N times under a free-threaded CPython
build, detecting crashes, deadlocks, TSAN warnings, and race
conditions. Optionally compares GIL-enabled vs GIL-disabled behavior
to isolate free-threading-specific failures.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labeille.classifier import has_ft_wheel
from labeille.crash import detect_crash
from labeille.ft.compat import assess_extension_compat
from labeille.ft.results import (
    FailureCategory,
    FTPackageResult,
    FTRunMeta,
    IterationOutcome,
    append_ft_result,
    save_ft_run,
)
from labeille.logging import get_logger
from labeille.resolve import fetch_pypi_metadata
from labeille.runner import (
    clean_env,
    build_sdist_install_commands,
    checkout_matching_tag,
    clone_repo,
    create_venv,
    detect_source_layout,
    fetch_latest_pypi_version,
    install_package,
    pull_repo,
    shield_source_dir,
)

log = get_logger("ft.runner")

_PYTHON_VERSION_RE = re.compile(r"(\d+)\.(\d+)")


def _parse_python_version(version_str: str) -> tuple[int, int] | None:
    """Extract (major, minor) from a Python version string.

    Handles formats like ``"3.15.0a1"``, ``"3.14.0"``, ``"3.15"``.

    Returns:
        Tuple of ``(major, minor)``, or None if parsing fails.
    """
    m = _PYTHON_VERSION_RE.match(version_str)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)))


@dataclass
class FTRunConfig:
    """Configuration for a free-threading test run."""

    target_python: Path
    iterations: int = 10
    timeout: int = 600
    stall_threshold: int = 60
    packages_filter: list[str] | None = None
    top_n: int | None = None
    registry_dir: Path = Path("registry")
    repos_dir: Path = Path("repos")
    venvs_dir: Path = Path("venvs")
    results_dir: Path = Path("results")
    env_overrides: dict[str, str] = field(default_factory=dict)
    extra_deps: list[str] = field(default_factory=list)
    test_command_suffix: str | None = None
    test_command_override: str | None = None
    detect_extensions: bool = True
    stop_on_first_pass: bool = False
    tsan_build: bool = False
    compare_with_gil: bool = False
    check_stability: bool = False
    verbose: bool = False
    stderr_tail_bytes: int = 4096
    install_from: str = "source"
    trust_ft_wheels: bool = False
    trust_ft_wheels_any_version: bool = False
    _target_python_version: tuple[int, int] | None = None

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of error messages."""
        errors: list[str] = []
        if not self.target_python.exists():
            errors.append(f"Target Python not found: {self.target_python}")
        if self.iterations < 1:
            errors.append(f"Iterations must be >= 1, got {self.iterations}")
        if self.timeout < 1:
            errors.append(f"Timeout must be >= 1, got {self.timeout}")
        if self.stall_threshold < 5:
            errors.append(f"Stall threshold must be >= 5, got {self.stall_threshold}")
        if self.compare_with_gil and self.iterations < 3:
            errors.append("GIL comparison mode needs at least 3 iterations for meaningful results")
        if self.trust_ft_wheels_any_version and not self.trust_ft_wheels:
            self.trust_ft_wheels = True
        return errors


# ---------------------------------------------------------------------------
# TSAN warning extraction
# ---------------------------------------------------------------------------

_TSAN_WARNING_PATTERN = re.compile(r"WARNING:\s*ThreadSanitizer:\s*(.+?)\s*\(pid=\d+\)")

_TSAN_SUMMARY_PATTERN = re.compile(r"SUMMARY:\s*ThreadSanitizer:\s*(.+)")


def extract_tsan_warnings(stderr: str) -> list[str]:
    """Extract TSAN warning types from stderr output.

    Returns a list of warning type strings, e.g.:
    ["data race", "thread leak", "lock-order-inversion"]

    Deduplicates within a single iteration -- if the same race is
    reported 50 times, it appears once.
    """
    warnings: set[str] = set()
    for match in _TSAN_WARNING_PATTERN.finditer(stderr):
        warnings.add(match.group(1).strip())
    for match in _TSAN_SUMMARY_PATTERN.finditer(stderr):
        text = match.group(1).strip()
        warning_type = text.split(" in ")[0].strip()
        if warning_type:
            warnings.add(warning_type)
    return sorted(warnings)


# ---------------------------------------------------------------------------
# Pytest output parsing
# ---------------------------------------------------------------------------

_PYTEST_SUMMARY_PATTERN = re.compile(r"=+\s*(.*?)\s+in\s+[\d.]+s\s*=+")

_PYTEST_COUNT_PATTERN = re.compile(
    r"(\d+)\s+(passed|failed|error|errors|skipped|warnings?|deselected|xfailed|xpassed)"
)

_PYTEST_VERBOSE_PATTERN = re.compile(
    r"^(.*?)\s+(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s*$",
    re.MULTILINE,
)


def parse_pytest_summary(output: str) -> dict[str, int]:
    """Parse pytest summary line for test counts.

    Returns dict like {"passed": 5, "failed": 2, "error": 1}.
    """
    counts: dict[str, int] = {}
    match = _PYTEST_SUMMARY_PATTERN.search(output)
    if match:
        summary_text = match.group(1)
        for count_match in _PYTEST_COUNT_PATTERN.finditer(summary_text):
            count = int(count_match.group(1))
            kind = count_match.group(2).rstrip("s")
            if kind == "warning":
                continue
            counts[kind] = count
    return counts


def parse_pytest_verbose(output: str) -> dict[str, str]:
    """Parse per-test results from pytest -v output.

    Returns dict of {test_id: status} like:
    {"tests/test_foo.py::test_bar": "PASSED"}
    """
    results: dict[str, str] = {}
    for match in _PYTEST_VERBOSE_PATTERN.finditer(output):
        test_id = match.group(1).strip()
        status = match.group(2)
        results[test_id] = status
    return results


# ---------------------------------------------------------------------------
# Deadlock detection via output monitoring
# ---------------------------------------------------------------------------


class OutputMonitor:
    """Monitor a subprocess's stderr for deadlock detection.

    Tracks the last time output was received. If no output arrives
    for longer than ``stall_threshold`` seconds, the process is
    likely deadlocked.

    Also captures the last N bytes of stderr and TSAN warnings.
    """

    def __init__(
        self,
        stall_threshold: int = 60,
        tail_bytes: int = 4096,
    ) -> None:
        self.stall_threshold = stall_threshold
        self.tail_bytes = tail_bytes

        self._last_output_time = time.monotonic()
        self._last_line = ""
        self._stderr_chunks: list[str] = []
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._finished = threading.Event()

    def feed(self, line: str) -> None:
        """Called for each line of stderr output."""
        with self._lock:
            self._last_output_time = time.monotonic()
            self._last_line = line.rstrip()
            self._stderr_chunks.append(line)
            self._total_bytes += len(line)

    def mark_finished(self) -> None:
        """Mark the monitor as finished (pipe closed)."""
        self._finished.set()

    @property
    def stalled(self) -> bool:
        """True if no output for longer than stall_threshold."""
        with self._lock:
            elapsed = time.monotonic() - self._last_output_time
            return elapsed > self.stall_threshold

    @property
    def seconds_since_last_output(self) -> float:
        """Seconds since the last output was received."""
        with self._lock:
            return time.monotonic() - self._last_output_time

    @property
    def last_line(self) -> str:
        """The last line of stderr received."""
        with self._lock:
            return self._last_line

    @property
    def stderr_tail(self) -> str:
        """Return the last tail_bytes of captured stderr."""
        with self._lock:
            full = "".join(self._stderr_chunks)
            if len(full) <= self.tail_bytes:
                return full
            return full[-self.tail_bytes :]

    @property
    def full_stderr(self) -> str:
        """Return all captured stderr."""
        with self._lock:
            return "".join(self._stderr_chunks)


def _start_output_monitor(
    proc: subprocess.Popen[Any],
    monitor: OutputMonitor,
) -> threading.Thread:
    """Start a daemon thread that reads proc.stderr into the monitor.

    The thread reads lines from the process's stderr pipe and feeds
    them to the OutputMonitor. It terminates when the pipe closes
    (process exit) or when the process is killed.
    """

    def _reader() -> None:
        try:
            assert proc.stderr is not None
            for line in proc.stderr:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                monitor.feed(line)
        except (ValueError, OSError):
            pass
        finally:
            monitor.mark_finished()

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Single iteration execution
# ---------------------------------------------------------------------------


def run_single_iteration(
    *,
    venv_python: Path,
    test_command: str,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    stall_threshold: int,
    iteration_index: int,
    tsan_build: bool = False,
    stderr_tail_bytes: int = 4096,
) -> IterationOutcome:
    """Run a single test iteration with deadlock and crash detection.

    Executes the test command as a subprocess, monitors stderr for
    output stalls (deadlock detection), captures crash info and
    TSAN warnings.

    Args:
        venv_python: Python executable in the package's venv.
        test_command: Full test command to execute.
        cwd: Working directory (the package's repo).
        env: Environment variables.
        timeout: Maximum execution time in seconds.
        stall_threshold: Seconds without output -> deadlock.
        iteration_index: 1-based iteration number.
        tsan_build: Whether to parse TSAN warnings from stderr.
        stderr_tail_bytes: How many bytes of stderr tail to capture.

    Returns:
        IterationOutcome with status, crash info, timing, etc.
    """
    monitor = OutputMonitor(
        stall_threshold=stall_threshold,
        tail_bytes=stderr_tail_bytes,
    )

    start_time = time.monotonic()

    cmd = test_command
    for prefix in ("python -m ", "python3 -m "):
        if cmd.startswith(prefix):
            cmd = f"{venv_python} -m {cmd[len(prefix) :]}"
            break

    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        return IterationOutcome(
            index=iteration_index,
            status="fail",
            exit_code=None,
            duration_s=time.monotonic() - start_time,
            stderr_tail=f"Failed to start process: {exc}",
        )

    reader_thread = _start_output_monitor(proc, monitor)

    stdout_chunks: list[str] = []

    def _stdout_reader() -> None:
        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                if isinstance(raw_line, bytes):
                    stdout_chunks.append(raw_line.decode("utf-8", errors="replace"))
                else:
                    stdout_chunks.append(raw_line)
        except (ValueError, OSError):
            pass

    stdout_thread = threading.Thread(target=_stdout_reader, daemon=True)
    stdout_thread.start()

    is_deadlocked = False
    try:
        while proc.poll() is None:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                is_deadlocked = monitor.stalled
                try:
                    os.killpg(proc.pid, 9)
                except OSError:
                    proc.kill()
                break
            time.sleep(0.5)
    except Exception:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            proc.kill()

    reader_thread.join(timeout=5)
    stdout_thread.join(timeout=5)

    duration = time.monotonic() - start_time
    exit_code = proc.returncode
    stderr = monitor.full_stderr
    stdout = "".join(stdout_chunks)
    combined_output = stdout + stderr

    crash_info = detect_crash(exit_code, stderr)
    if crash_info is not None:
        return IterationOutcome(
            index=iteration_index,
            status="crash",
            exit_code=exit_code,
            duration_s=duration,
            crash_signal=crash_info.signal_name,
            crash_signature=crash_info.signature,
            tsan_warnings=extract_tsan_warnings(stderr) if tsan_build else [],
            output_stalled=monitor.stalled,
            last_output_line=monitor.last_line[:200],
            stderr_tail=monitor.stderr_tail,
        )

    if exit_code is None or (duration >= timeout and exit_code in (None, -9, 137)):
        status = "deadlock" if is_deadlocked else "timeout"
        return IterationOutcome(
            index=iteration_index,
            status=status,
            exit_code=exit_code,
            duration_s=duration,
            output_stalled=is_deadlocked,
            last_output_line=monitor.last_line[:200],
            stderr_tail=monitor.stderr_tail,
        )

    test_results = parse_pytest_verbose(combined_output)
    test_counts = parse_pytest_summary(combined_output)

    tsan_warnings = extract_tsan_warnings(stderr) if tsan_build else []

    if exit_code == 0:
        status = "pass"
    else:
        status = "fail"

    return IterationOutcome(
        index=iteration_index,
        status=status,
        exit_code=exit_code,
        duration_s=duration,
        tsan_warnings=tsan_warnings,
        output_stalled=False,
        last_output_line=monitor.last_line[:200],
        stderr_tail=monitor.stderr_tail,
        test_results=test_results,
        tests_passed=test_counts.get("passed"),
        tests_failed=test_counts.get("failed"),
        tests_errored=test_counts.get("error"),
        tests_skipped=test_counts.get("skipped"),
    )


# ---------------------------------------------------------------------------
# Per-package execution
# ---------------------------------------------------------------------------


def run_package_ft(
    pkg: Any,
    config: FTRunConfig,
) -> FTPackageResult:
    """Run free-threading tests for a single package.

    1. Clone/pull the repo.
    2. Create venv and install the package.
    3. Probe extension GIL compatibility.
    4. Run N test iterations, capturing outcomes.
    5. Optionally run N iterations with GIL enabled for comparison.
    6. Compute aggregates and categorize.

    Args:
        pkg: PackageEntry from the registry.
        config: Run configuration.

    Returns:
        FTPackageResult with all iterations and classification.
    """
    result = FTPackageResult(package=pkg.package)

    # --- FT wheel trust check (before clone). ---
    _cached_metadata: dict[str, Any] | None = None

    if config.trust_ft_wheels:
        ft_check_version: tuple[int, int] | None = (
            None if config.trust_ft_wheels_any_version else config._target_python_version
        )
        _cached_metadata = fetch_pypi_metadata(pkg.package, timeout=10.0)
        if _cached_metadata is not None:
            urls = _cached_metadata.get("urls", [])
            pypi_version: str | None = None
            try:
                pypi_version = _cached_metadata["info"]["version"]
            except (KeyError, TypeError):
                pass

            if has_ft_wheel(urls, target_version=ft_check_version):
                version_desc = (
                    f" (Python {ft_check_version[0]}.{ft_check_version[1]})"
                    if ft_check_version
                    else " (any version)"
                )
                log.info(
                    "Skipping %s: free-threaded wheel found for %s%s",
                    pkg.package,
                    pypi_version or "latest",
                    version_desc,
                )
                result.category = FailureCategory.COMPATIBLE_BY_WHEEL
                result.ft_wheel_found = True
                result.ft_wheel_version = pypi_version
                result.install_from = "skipped"
                return result
            else:
                result.ft_wheel_found = False

    repo_dir = config.repos_dir / pkg.package
    venv_dir = config.venvs_dir / f"{pkg.package}-ft"

    # Step 1: Clone or pull repo.
    try:
        if repo_dir.exists():
            pull_repo(repo_dir)
        else:
            clone_repo(pkg.repo, repo_dir)
    except Exception as exc:
        log.error("Failed to clone/pull %s: %s", pkg.package, exc)
        result.install_ok = False
        result.install_error = f"Clone failed: {exc}"
        result.categorize()
        return result

    # Record current commit.
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            timeout=10,
        )
        if proc.returncode == 0:
            result.commit = proc.stdout.strip()
    except Exception:
        pass

    # Sdist version alignment.
    import_name = getattr(pkg, "import_name", None) or pkg.package.replace("-", "_")
    source_layout = "unknown"

    if config.install_from == "sdist":
        result.install_from = "sdist"
        # Reuse cached metadata from FT wheel check if available.
        if _cached_metadata is not None:
            try:
                sdist_version: str | None = _cached_metadata["info"]["version"]
            except (KeyError, TypeError):
                sdist_version = None
        else:
            sdist_version = fetch_latest_pypi_version(pkg.package)
        if sdist_version:
            log.info("PyPI latest version for %s: %s", pkg.package, sdist_version)
            commit_hash, matched_tag = checkout_matching_tag(repo_dir, pkg.package, sdist_version)
            if commit_hash:
                result.commit = commit_hash
                result.sdist_tag_matched = True
                log.info(
                    "Checked out tag %s for %s (commit %s)",
                    matched_tag,
                    pkg.package,
                    commit_hash[:12],
                )
            else:
                result.sdist_tag_matched = False
                log.warning(
                    "No matching tag for %s version %s, staying on HEAD",
                    pkg.package,
                    sdist_version,
                )
        else:
            sdist_version = None
            log.warning("Could not fetch PyPI version for %s", pkg.package)
        result.sdist_version = sdist_version
        source_layout = detect_source_layout(repo_dir, import_name)
        log.debug("Source layout for %s: %s", pkg.package, source_layout)
    else:
        result.install_from = "source"

    # Step 2: Create venv and install.
    try:
        create_venv(config.target_python, venv_dir)
    except Exception as exc:
        log.error("Venv creation failed for %s: %s", pkg.package, exc)
        result.install_ok = False
        result.install_error = f"Venv creation failed: {exc}"
        result.categorize()
        return result

    venv_python = venv_dir / "bin" / "python"

    env = clean_env(
        PYTHON_GIL="0",
        PYTHONFAULTHANDLER="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    if config.tsan_build:
        env["ASAN_OPTIONS"] = "detect_leaks=0"
        env["TSAN_OPTIONS"] = "exitcode=0"
    env.update(config.env_overrides)

    install_start = time.monotonic()
    raw_install_cmd = pkg.install_command or "pip install -e ."

    if config.install_from == "sdist":
        # Sdist mode: install from PyPI + deps from repo.
        sdist_cmd, deps_cmd = build_sdist_install_commands(pkg.package, raw_install_cmd)
        log.info("Installing %s from sdist: %s", pkg.package, sdist_cmd)
        try:
            install_result = install_package(
                venv_python,
                sdist_cmd,
                cwd=repo_dir,
                env=env,
                timeout=config.timeout,
            )
            if install_result.returncode != 0:
                result.install_ok = False
                result.install_error = (
                    f"Sdist install failed (exit {install_result.returncode}): "
                    f"{install_result.stderr.strip()[-500:]}"
                )
                result.install_duration_s = time.monotonic() - install_start
                result.categorize()
                return result
        except subprocess.TimeoutExpired:
            result.install_ok = False
            result.install_error = "Sdist install timed out"
            result.install_duration_s = time.monotonic() - install_start
            result.categorize()
            return result

        if deps_cmd:
            log.info("Installing test deps for %s: %s", pkg.package, deps_cmd)
            try:
                install_package(
                    venv_python, deps_cmd, cwd=repo_dir, env=env, timeout=config.timeout
                )
            except Exception as exc:
                log.warning("Test deps install failed for %s: %s (continuing)", pkg.package, exc)
    else:
        # Source mode: original behavior.
        install_cmd = raw_install_cmd
        try:
            install_result = install_package(
                venv_python,
                install_cmd,
                cwd=repo_dir,
                env=env,
                timeout=config.timeout,
            )
            if install_result.returncode != 0:
                result.install_ok = False
                result.install_error = (
                    f"Install failed (exit {install_result.returncode}): "
                    f"{install_result.stderr.strip()[-500:]}"
                )
                result.install_duration_s = time.monotonic() - install_start
                result.categorize()
                return result
        except subprocess.TimeoutExpired:
            result.install_ok = False
            result.install_error = "Install timed out"
            result.install_duration_s = time.monotonic() - install_start
            result.categorize()
            return result

    # Install extra deps.
    if config.extra_deps:
        try:
            extra_cmd = f"pip install {' '.join(config.extra_deps)}"
            install_package(
                venv_python,
                extra_cmd,
                cwd=repo_dir,
                env=env,
                timeout=config.timeout,
            )
        except Exception:
            pass

    result.install_duration_s = time.monotonic() - install_start

    # Shield source directory in sdist mode for import/compat checks and tests.
    _shield = (
        shield_source_dir(repo_dir, import_name, source_layout)
        if config.install_from == "sdist"
        else nullcontext()
    )
    with _shield:
        # Step 3: Extension compatibility check.
        if config.detect_extensions:
            try:
                compat = assess_extension_compat(
                    pkg.package,
                    venv_python=venv_python,
                    repo_dir=repo_dir if config.install_from != "sdist" else None,
                    import_name=getattr(pkg, "import_name", None),
                    env=env,
                )
                result.extension_compat = compat.to_dict()
                result.import_ok = compat.import_ok
                result.import_error = compat.import_error
            except Exception as exc:
                log.warning(
                    "Extension compat check failed for %s: %s",
                    pkg.package,
                    exc,
                )

            if not result.import_ok:
                result.categorize()
                return result

        # Step 4: Resolve test command.
        test_cmd = pkg.test_command or "python -m pytest"
        if config.test_command_override:
            test_cmd = config.test_command_override
        base_suffix = "-v"
        if config.test_command_suffix:
            base_suffix = f"{base_suffix} {config.test_command_suffix}"
        test_cmd = f"{test_cmd} {base_suffix}"

        # Step 5: Run iterations with GIL disabled.
        log.info(
            "Running %d iterations of %s (free-threaded)...",
            config.iterations,
            pkg.package,
        )

        for i in range(1, config.iterations + 1):
            iteration_env = dict(env)
            iteration_env["PYTHON_GIL"] = "0"

            outcome = run_single_iteration(
                venv_python=venv_python,
                test_command=test_cmd,
                cwd=repo_dir,
                env=iteration_env,
                timeout=config.timeout,
                stall_threshold=config.stall_threshold,
                iteration_index=i,
                tsan_build=config.tsan_build,
                stderr_tail_bytes=config.stderr_tail_bytes,
            )
            result.iterations.append(outcome)

            log.info(
                "  %s iteration %d/%d: %s (%.1fs)",
                pkg.package,
                i,
                config.iterations,
                outcome.status,
                outcome.duration_s,
            )

            if config.stop_on_first_pass and outcome.is_pass:
                log.info(
                    "  %s passed on iteration %d, stopping early.",
                    pkg.package,
                    i,
                )
                break

    # Step 6: GIL comparison (if enabled).
    if config.compare_with_gil:
        log.info(
            "Running %d iterations of %s (GIL enabled for comparison)...",
            config.iterations,
            pkg.package,
        )
        gil_iterations: list[IterationOutcome] = []

        for i in range(1, config.iterations + 1):
            iteration_env = dict(env)
            iteration_env["PYTHON_GIL"] = "1"

            outcome = run_single_iteration(
                venv_python=venv_python,
                test_command=test_cmd,
                cwd=repo_dir,
                env=iteration_env,
                timeout=config.timeout,
                stall_threshold=config.stall_threshold,
                iteration_index=i,
                tsan_build=False,
                stderr_tail_bytes=config.stderr_tail_bytes,
            )
            gil_iterations.append(outcome)

            log.info(
                "  %s (GIL) iteration %d/%d: %s (%.1fs)",
                pkg.package,
                i,
                config.iterations,
                outcome.status,
                outcome.duration_s,
            )

        result.gil_enabled_iterations = gil_iterations

    # Step 7: Compute aggregates and categorize.
    result.compute_aggregates()
    result.categorize()

    return result


# ---------------------------------------------------------------------------
# Full run orchestration
# ---------------------------------------------------------------------------


def run_ft(config: FTRunConfig) -> list[FTPackageResult]:
    """Execute a complete free-threading test run.

    1. Validate configuration.
    2. Profile the system and target Python.
    3. Optionally check system stability.
    4. Load packages from registry.
    5. Run each package.
    6. Save results.

    Args:
        config: Run configuration.

    Returns:
        List of FTPackageResult for all tested packages.
    """
    from labeille.bench.system import (
        capture_python_profile,
        capture_system_profile,
        check_stability,
    )
    from labeille.registry import load_index

    errors = config.validate()
    if errors:
        for e in errors:
            log.error("Config error: %s", e)
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

    sys_profile = capture_system_profile()
    py_profile = capture_python_profile(
        config.target_python,
        env={"PYTHON_GIL": "0"},
    )

    # Parse target Python version for FT wheel matching.
    config._target_python_version = _parse_python_version(py_profile.version)

    if not py_profile.gil_disabled:
        log.warning(
            "Target Python at %s does not appear to be a "
            "free-threaded build (GIL is not disabled). "
            "Results may not be meaningful.",
            config.target_python,
        )

    if config.check_stability:
        stability = check_stability()
        if not stability.stable:
            for err in stability.errors:
                log.error("Stability: %s", err)
            raise RuntimeError(
                "System is not stable enough for testing. Use --no-check-stability to override."
            )
        for warn in stability.warnings:
            log.warning("Stability: %s", warn)

    run_id = time.strftime("ft_%Y%m%d_%H%M%S")
    output_dir = config.results_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(config.registry_dir)
    packages = _select_packages(index, config)

    log.info(
        "Free-threading test run: %d packages, %d iterations each",
        len(packages),
        config.iterations,
    )
    if config.compare_with_gil:
        log.info("GIL comparison mode: will also run with GIL enabled")

    meta = FTRunMeta(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        system_profile=sys_profile.to_dict(),
        python_profile=py_profile.to_dict(),
        config={
            "target_python": str(config.target_python),
            "iterations": config.iterations,
            "timeout": config.timeout,
            "stall_threshold": config.stall_threshold,
            "compare_with_gil": config.compare_with_gil,
            "tsan_build": config.tsan_build,
            "stop_on_first_pass": config.stop_on_first_pass,
            "detect_extensions": config.detect_extensions,
            "extra_deps": config.extra_deps,
            "test_command_suffix": config.test_command_suffix,
            "test_command_override": config.test_command_override,
            "install_from": config.install_from,
            "trust_ft_wheels": config.trust_ft_wheels,
            "trust_ft_wheels_any_version": config.trust_ft_wheels_any_version,
        },
        cli_args=sys.argv[1:],
        packages_total=len(packages),
    )

    start_time = time.monotonic()
    results: list[FTPackageResult] = []

    for i, pkg in enumerate(packages, 1):
        log.info(
            "(%d/%d) Testing %s...",
            i,
            len(packages),
            pkg.package,
        )

        try:
            pkg_result = run_package_ft(pkg, config)
        except Exception as exc:
            log.error(
                "Unexpected error testing %s: %s",
                pkg.package,
                exc,
                exc_info=True,
            )
            pkg_result = FTPackageResult(
                package=pkg.package,
                install_ok=False,
                install_error=f"Unexpected error: {exc}",
            )
            pkg_result.categorize()

        results.append(pkg_result)

        append_ft_result(output_dir, pkg_result)

        log.info(
            "  %s \u2192 %s (pass rate: %.0f%%, %d iterations)",
            pkg.package,
            pkg_result.category.value,
            pkg_result.pass_rate * 100,
            pkg_result.iterations_completed,
        )

    meta.packages_completed = len(results)
    meta.total_iterations = sum(r.iterations_completed for r in results)
    meta.total_duration_s = time.monotonic() - start_time

    save_ft_run(output_dir, meta, results)

    log.info(
        "Free-threading run complete: %d packages in %.0fs \u2192 %s",
        len(results),
        meta.total_duration_s,
        output_dir,
    )

    return results


def _select_packages(index: Any, config: FTRunConfig) -> list[Any]:
    """Select and filter packages from the registry index.

    Applies --packages filter, --top N, and skips packages with
    ft_skip=True in the registry.
    """
    from labeille.registry import load_package

    # Load full PackageEntry objects for enriched packages.
    packages = []
    for entry in index.packages:
        if not entry.enriched:
            continue
        if entry.skip:
            continue
        try:
            pkg = load_package(entry.package, config.registry_dir)
            packages.append(pkg)
        except Exception:
            log.debug("Could not load package %s, skipping", entry.package)

    if config.packages_filter:
        names = set(config.packages_filter)
        packages = [p for p in packages if p.package in names]

    packages = [p for p in packages if not getattr(p, "ft_skip", False)]

    packages.sort(
        key=lambda p: getattr(p, "monthly_downloads", 0),
        reverse=True,
    )

    if config.top_n:
        packages = packages[: config.top_n]

    return packages
