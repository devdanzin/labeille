"""Run extension test suites under ThreadSanitizer-enabled free-threaded Python.

Builds a venv with a TSan-enabled free-threaded CPython, installs the
target extension, runs its test suite, and captures TSan race reports
from stderr.  The output is designed for use with ft-review-toolkit's
``tsan-report-analyzer`` agent.
"""

from __future__ import annotations

import importlib.resources
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from labeille.io_utils import append_jsonl, write_meta_json
from labeille.logging import get_logger
from labeille.runner import (
    InstallerBackend,
    clean_env,
    clone_repo,
    create_venv,
    install_package,
    pull_repo,
    resolve_installer,
    run_test_command,
)

log = get_logger("tsan_run")


# ---------------------------------------------------------------------------
# TSan report parsing
# ---------------------------------------------------------------------------

_RACE_HEADER_RE = re.compile(r"^WARNING: ThreadSanitizer: (.+?)(?:\s+\(pid=\d+\))?$", re.MULTILINE)


def parse_race_count(report: str) -> int:
    """Count the number of unique TSan race reports in *report*."""
    return len(_RACE_HEADER_RE.findall(report))


def parse_race_types(report: str) -> dict[str, int]:
    """Count TSan reports by type (e.g. ``data race``, ``thread leak``).

    Returns a mapping of race type to occurrence count.
    """
    counts: dict[str, int] = {}
    for match in _RACE_HEADER_RE.finditer(report):
        race_type = match.group(1).strip()
        counts[race_type] = counts.get(race_type, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Python validation
# ---------------------------------------------------------------------------


@dataclass
class PythonTsanInfo:
    """Information about a TSan-enabled Python build."""

    version: str
    is_free_threaded: bool
    is_tsan: bool
    is_debug: bool


def validate_tsan_python(python_path: Path) -> PythonTsanInfo:
    """Validate that *python_path* is a TSan-enabled free-threaded Python.

    Returns a :class:`PythonTsanInfo` with details about the build.

    Raises:
        RuntimeError: If the interpreter cannot be executed.
        ValueError: If the interpreter is not free-threaded or not TSan-enabled.
    """
    script = (
        "import sys, sysconfig\n"
        "v = sys.version\n"
        "ft = not sys._is_gil_enabled()\n"
        "abi = sysconfig.get_config_var('SOABI') or ''\n"
        "tsan = 'tsan' in (sysconfig.get_config_var('CONFIG_ARGS') or '').lower()\n"
        "# Also check CFLAGS for -fsanitize=thread\n"
        "cflags = (sysconfig.get_config_var('CFLAGS') or '').lower()\n"
        "tsan = tsan or 'fsanitize=thread' in cflags\n"
        "debug = hasattr(sys, 'gettotalrefcount')\n"
        "print(f'{v}|{ft}|{tsan}|{debug}|{abi}')\n"
    )
    try:
        proc = subprocess.run(
            [str(python_path), "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot execute {python_path}: {exc}") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"Python at {python_path} exited with code {proc.returncode}: "
            f"{(proc.stderr or '').strip()[:300]}"
        )

    parts = proc.stdout.strip().split("|")
    if len(parts) < 5:
        raise RuntimeError(f"Unexpected output from {python_path}: {proc.stdout.strip()!r}")

    version = parts[0]
    is_ft = parts[1] == "True"
    is_tsan = parts[2] == "True"
    is_debug = parts[3] == "True"

    if not is_ft:
        raise ValueError(
            f"Python at {python_path} is not free-threaded (GIL is enabled). "
            "TSan analysis requires --disable-gil. Build CPython with: "
            "./configure --disable-gil --with-thread-sanitizer"
        )

    if not is_tsan:
        log.warning(
            "Python at %s does not appear to have TSan instrumentation. "
            "Race detection may be incomplete. Build with: "
            "./configure --disable-gil --with-thread-sanitizer",
            python_path,
        )

    return PythonTsanInfo(
        version=version,
        is_free_threaded=is_ft,
        is_tsan=is_tsan,
        is_debug=is_debug,
    )


# ---------------------------------------------------------------------------
# Extension .so discovery
# ---------------------------------------------------------------------------


def find_extension_sos(venv_dir: Path, package_name: str) -> list[str]:
    """Find shared object files for *package_name* in the venv's site-packages.

    Looks for ``*.so`` files in directories matching the package name
    (normalised: lowercase, hyphens to underscores).
    """
    site_packages = venv_dir / "lib"
    if not site_packages.exists():
        return []

    # Find the python version subdir (e.g. python3.14t)
    python_dirs = list(site_packages.iterdir())
    if not python_dirs:
        return []

    sos: list[str] = []
    normalised = package_name.lower().replace("-", "_")
    for pydir in python_dirs:
        sp = pydir / "site-packages"
        if not sp.exists():
            continue
        for so_file in sp.rglob("*.so"):
            # Include .so files under directories matching the package name
            # or .so files whose name starts with the package name
            rel = so_file.relative_to(sp)
            top_dir = rel.parts[0].lower().replace("-", "_") if rel.parts else ""
            so_stem = so_file.stem.lower().replace("-", "_")
            if top_dir == normalised or so_stem.startswith(normalised):
                sos.append(str(so_file))

    return sorted(sos)


# ---------------------------------------------------------------------------
# Default suppressions
# ---------------------------------------------------------------------------


def get_default_suppressions_path() -> Path:
    """Return the path to the bundled TSan suppressions file."""
    ref = importlib.resources.files("labeille") / "data" / "tsan_suppressions.txt"
    # importlib.resources may return a Traversable — resolve to a real path.
    with importlib.resources.as_file(ref) as p:
        return Path(p)


# ---------------------------------------------------------------------------
# TSAN_OPTIONS builder
# ---------------------------------------------------------------------------


def build_tsan_options(
    *,
    suppressions_path: Path | None = None,
    log_path: str | None = None,
    history_size: int = 7,
    halt_on_error: bool = False,
    exitcode: int = 0,
) -> str:
    """Build a ``TSAN_OPTIONS`` environment variable string.

    Args:
        suppressions_path: Path to suppressions file (omitted if None).
        log_path: Write TSan output to ``<log_path>.<pid>`` instead of stderr.
        history_size: TSan history size (0-7, default 7 = 128 events).
        halt_on_error: Stop on first error (useful for quick checks).
        exitcode: Exit code override for TSan errors (0 = don't change).

    Returns:
        A colon-separated options string.
    """
    parts: list[str] = []
    if suppressions_path is not None:
        parts.append(f"suppressions={suppressions_path}")
    parts.append(f"history_size={history_size}")
    parts.append("second_deadlock_stack=1")
    parts.append("report_signal_unsafe=0")
    parts.append(f"halt_on_error={'1' if halt_on_error else '0'}")
    parts.append(f"exitcode={exitcode}")
    if log_path is not None:
        parts.append(f"log_path={log_path}")
    return ":".join(parts)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TsanRunConfig:
    """Configuration for a tsan-run invocation."""

    target_python: Path
    output_dir: Path = field(default_factory=lambda: Path("results"))
    registry_dir: Path | None = None
    repos_dir: Path | None = None
    venvs_dir: Path | None = None
    packages_filter: list[str] | None = None
    timeout: int = 600
    installer: str = "auto"
    suppressions: Path | None = None
    quick: bool = False
    stress: int = 1
    verbose: bool = False
    extra_deps: list[str] = field(default_factory=list)
    skip_if_exists: bool = True
    workers: int = 1
    test_script: Path | None = None


# ---------------------------------------------------------------------------
# Result data models
# ---------------------------------------------------------------------------


@dataclass
class TsanRunResult:
    """Result of running one extension's test suite under TSan."""

    package: str = ""
    status: str = ""  # ok, no_races, build_fail, install_error, test_fail,
    #                    no_repo, clone_error, timeout, skipped, error
    race_count: int = 0
    race_types: dict[str, int] = field(default_factory=dict)
    report_path: str = ""
    metadata_path: str = ""
    test_exit_code: int | None = None
    test_duration_s: float = 0.0
    install_duration_s: float = 0.0
    extension_so_paths: list[str] = field(default_factory=list)
    error_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL. Sparse output."""
        d: dict[str, Any] = {"package": self.package, "status": self.status}
        if self.race_count:
            d["race_count"] = self.race_count
        if self.race_types:
            d["race_types"] = self.race_types
        if self.report_path:
            d["report_path"] = self.report_path
        if self.metadata_path:
            d["metadata_path"] = self.metadata_path
        if self.test_exit_code is not None:
            d["test_exit_code"] = self.test_exit_code
        d["test_duration_s"] = round(self.test_duration_s, 2)
        if self.install_duration_s:
            d["install_duration_s"] = round(self.install_duration_s, 2)
        if self.extension_so_paths:
            d["extension_so_paths"] = self.extension_so_paths
        if self.error_summary:
            d["error_summary"] = self.error_summary
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TsanRunResult:
        """Deserialize from dict, ignoring unknown fields."""
        from labeille.io_utils import dataclass_from_dict

        return dataclass_from_dict(cls, data)


@dataclass
class TsanRunMeta:
    """Metadata for a tsan-run batch."""

    run_id: str
    target_python: str
    python_version: str
    is_free_threaded: bool = False
    is_tsan: bool = False
    started_at: str = ""
    finished_at: str = ""
    total_packages: int = 0
    packages_with_races: int = 0
    total_races: int = 0
    suppressions_used: str = ""
    quick_mode: bool = False
    stress_count: int = 1
    tsan_options: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {k: v for k, v in asdict(self).items() if v or isinstance(v, (bool, int))}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TsanRunMeta:
        """Deserialize from dict, ignoring unknown fields."""
        from labeille.io_utils import dataclass_from_dict

        return dataclass_from_dict(cls, data)


# ---------------------------------------------------------------------------
# Per-package TSan test run
# ---------------------------------------------------------------------------


def run_tsan_tests(
    pkg: Any,
    config: TsanRunConfig,
    pkg_output_dir: Path,
    *,
    installer: InstallerBackend = InstallerBackend.PIP,
    tsan_options: str = "",
    suppressions_path: Path | None = None,
) -> TsanRunResult:
    """Run a single package's test suite under TSan.

    Steps:
        1. Clone or update the repo.
        2. Create a venv with the TSan-enabled Python.
        3. Install the extension.
        4. Discover extension .so files.
        5. Run the test suite with TSAN_OPTIONS set.
        6. Capture stderr (TSan report), parse race count.
        7. Write tsan_report.txt and tsan_metadata.json.
    """
    result = TsanRunResult(package=pkg.package)

    # Check skip_if_exists.
    if config.skip_if_exists:
        existing_report = pkg_output_dir / "tsan_report.txt"
        if existing_report.exists():
            result.status = "skipped"
            return result

    # Step 1: Clone.
    repos_base = config.repos_dir or (config.output_dir / "_repos")
    repos_base.mkdir(parents=True, exist_ok=True)
    repo_dir = repos_base / pkg.package

    repo_url = getattr(pkg, "repo", None)
    if not repo_url:
        result.status = "no_repo"
        return result

    try:
        if repo_dir.exists() and (repo_dir / ".git").is_dir():
            pull_repo(repo_dir)
        else:
            clone_repo(repo_url, repo_dir)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        result.status = "clone_error"
        result.error_summary = str(exc)[:200]
        return result

    # Step 2: Create venv.
    venvs_base = config.venvs_dir or (config.output_dir / "_venvs")
    venvs_base.mkdir(parents=True, exist_ok=True)
    venv_dir = venvs_base / pkg.package

    try:
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
        create_venv(config.target_python, venv_dir, installer)
    except (subprocess.CalledProcessError, OSError) as exc:
        result.status = "install_error"
        result.error_summary = f"Venv creation failed: {exc}"
        return result

    venv_python = venv_dir / "bin" / "python"
    env = clean_env()

    # Step 3: Install the extension.
    install_cmd = getattr(pkg, "install_command", None) or "pip install -e ."
    install_start = time.monotonic()

    # Install extra deps first.
    all_extra_deps = list(config.extra_deps)
    pkg_deps = getattr(pkg, "dependencies", None) or []
    all_extra_deps.extend(pkg_deps)
    if all_extra_deps:
        deps_str = " ".join(f"'{d}'" for d in all_extra_deps)
        try:
            install_package(
                venv_python,
                f"pip install {deps_str}",
                cwd=repo_dir,
                env=env,
                timeout=config.timeout,
                installer=installer,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            log.warning("Extra deps install failed for %s: %s", pkg.package, exc)

    try:
        proc = install_package(
            venv_python,
            install_cmd,
            cwd=repo_dir,
            env=env,
            timeout=config.timeout,
            installer=installer,
        )
    except subprocess.TimeoutExpired:
        result.status = "timeout"
        result.install_duration_s = time.monotonic() - install_start
        return result
    except (subprocess.CalledProcessError, OSError) as exc:
        result.status = "install_error"
        result.error_summary = str(exc)[:200]
        result.install_duration_s = time.monotonic() - install_start
        return result

    result.install_duration_s = round(time.monotonic() - install_start, 2)

    if proc.returncode != 0:
        result.status = "install_error"
        result.error_summary = (proc.stderr or "")[-500:]
        return result

    # Step 4: Discover extension .so files.
    result.extension_so_paths = find_extension_sos(venv_dir, pkg.package)

    # Step 5: Run the test suite (or custom script) with TSAN_OPTIONS.
    if config.test_script is not None:
        test_cmd = f"python {config.test_script}"
        log.info("Using custom test script: %s", config.test_script)
    else:
        test_cmd = getattr(pkg, "test_command", None) or "python -m pytest tests/"

    # Build test environment with TSan settings.
    venv_bin = venv_python.parent
    test_env = {
        **env,
        "TSAN_OPTIONS": tsan_options,
        "PATH": f"{venv_bin}:{env.get('PATH', '')}",
        "PYTHONFAULTHANDLER": "1",
    }

    # Stress mode: repeat tests N times.
    stress_count = max(1, config.stress)
    all_stderr: list[str] = []
    last_exit_code = 0
    total_test_time = 0.0

    for iteration in range(stress_count):
        if stress_count > 1:
            log.info("  Stress iteration %d/%d for %s", iteration + 1, stress_count, pkg.package)

        test_start = time.monotonic()
        try:
            test_proc = run_test_command(
                venv_python,
                test_cmd,
                cwd=repo_dir,
                env=test_env,
                timeout=config.timeout,
            )
            last_exit_code = test_proc.returncode
            if test_proc.stderr:
                all_stderr.append(test_proc.stderr)
        except subprocess.TimeoutExpired as exc:
            result.status = "timeout"
            result.test_duration_s = time.monotonic() - test_start + total_test_time
            # Capture partial stderr if available.
            if hasattr(exc, "stderr") and exc.stderr:
                all_stderr.append(
                    exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode()
                )
            break
        except (subprocess.CalledProcessError, OSError) as exc:
            result.status = "error"
            result.error_summary = str(exc)[:200]
            result.test_duration_s = time.monotonic() - test_start + total_test_time
            break

        total_test_time += time.monotonic() - test_start

    if result.status in ("timeout", "error"):
        # Still save partial report if we got any stderr.
        if all_stderr:
            _save_report(pkg_output_dir, all_stderr, result, pkg, config, suppressions_path)
        return result

    result.test_exit_code = last_exit_code
    result.test_duration_s = round(total_test_time, 2)

    # Step 6: Parse TSan report from stderr.
    combined_stderr = "\n".join(all_stderr)
    race_count = parse_race_count(combined_stderr)
    result.race_count = race_count
    result.race_types = parse_race_types(combined_stderr)

    # Step 7: Write report and metadata.
    _save_report(pkg_output_dir, all_stderr, result, pkg, config, suppressions_path)

    if race_count > 0:
        result.status = "ok"
    else:
        result.status = "no_races"

    return result


def _save_report(
    pkg_output_dir: Path,
    stderr_parts: list[str],
    result: TsanRunResult,
    pkg: Any,
    config: TsanRunConfig,
    suppressions_path: Path | None,
) -> None:
    """Save TSan report and metadata files to *pkg_output_dir*."""
    pkg_output_dir.mkdir(parents=True, exist_ok=True)

    # Write raw TSan report.
    combined = "\n".join(stderr_parts)
    report_path = pkg_output_dir / "tsan_report.txt"
    try:
        report_path.write_text(combined, encoding="utf-8")
        result.report_path = str(report_path)
    except OSError as exc:
        log.warning("Could not save TSan report for %s: %s", pkg.package, exc)

    # Write test stdout (for context) — not captured separately, so skip.

    # Write metadata JSON.
    test_cmd = getattr(pkg, "test_command", None) or "python -m pytest tests/"
    metadata = {
        "extension": pkg.package,
        "extension_version": getattr(pkg, "version", None),
        "python_version": "",  # Filled by caller if available.
        "platform": _get_platform(),
        "test_command": test_cmd,
        "test_exit_code": result.test_exit_code,
        "test_duration_seconds": result.test_duration_s,
        "extension_so_paths": result.extension_so_paths,
        "source_root": str(config.repos_dir / pkg.package) if config.repos_dir else "",
        "race_count": result.race_count,
        "race_types": result.race_types,
        "suppressions": str(suppressions_path) if suppressions_path else "",
        "stress_iterations": config.stress,
    }
    metadata_path = pkg_output_dir / "tsan_metadata.json"
    try:
        write_meta_json(metadata_path, metadata)
        result.metadata_path = str(metadata_path)
    except OSError as exc:
        log.warning("Could not save metadata for %s: %s", pkg.package, exc)

    # Copy suppressions for reproducibility.
    if suppressions_path and suppressions_path.exists():
        supp_copy = pkg_output_dir / "tsan_suppressions.txt"
        if not supp_copy.exists():
            try:
                shutil.copy2(suppressions_path, supp_copy)
            except OSError:
                pass


def _get_platform() -> str:
    """Return a platform string like ``linux-x86_64``."""
    import platform

    return f"{platform.system().lower()}-{platform.machine()}"


# ---------------------------------------------------------------------------
# Full run orchestration
# ---------------------------------------------------------------------------


def _load_packages(config: TsanRunConfig) -> list[Any]:
    """Load and filter packages for TSan testing."""
    from labeille.registry import load_index, load_package

    if config.registry_dir is None and config.packages_filter is None:
        raise ValueError("Either --registry-dir or --packages must be provided.")

    packages: list[Any] = []

    if config.registry_dir:
        index = load_index(config.registry_dir)
        names = [e.name for e in index.packages]

        if config.packages_filter:
            filter_set = set(config.packages_filter)
            names = [n for n in names if n in filter_set]

        for name in names:
            try:
                pkg = load_package(name, config.registry_dir)
                if getattr(pkg, "skip", False):
                    continue
                packages.append(pkg)
            except (FileNotFoundError, OSError, ValueError) as exc:
                log.warning("Package %s not found in registry: %s", name, exc)

    elif config.packages_filter:
        from labeille.registry import PackageEntry
        from labeille.resolve import extract_repo_url, fetch_pypi_metadata

        for name in config.packages_filter:
            meta = fetch_pypi_metadata(name, timeout=10.0)
            repo_url = None
            if meta:
                repo_url = extract_repo_url(meta)
            packages.append(
                PackageEntry(
                    package=name,
                    repo=repo_url,
                )
            )

    if not packages:
        log.warning("No packages found after filtering.")

    return packages


def run_tsan_batch(
    config: TsanRunConfig,
) -> tuple[TsanRunMeta, list[TsanRunResult]]:
    """Execute a complete tsan-run batch.

    1. Validate the target Python (free-threaded, TSan-enabled).
    2. Build TSAN_OPTIONS string.
    3. Load packages.
    4. Run each package's test suite under TSan.
    5. Save metadata and results.
    """
    from labeille.io_utils import utc_now_iso

    # Validate Python.
    py_info = validate_tsan_python(config.target_python)
    log.info(
        "Python: %s (free-threaded=%s, tsan=%s, debug=%s)",
        py_info.version,
        py_info.is_free_threaded,
        py_info.is_tsan,
        py_info.is_debug,
    )

    # Resolve suppressions.
    suppressions_path = config.suppressions
    if suppressions_path is None:
        try:
            suppressions_path = get_default_suppressions_path()
            log.info("Using bundled suppressions: %s", suppressions_path)
        except (FileNotFoundError, TypeError):
            log.warning("No suppressions file found. TSan output may be noisy.")
            suppressions_path = None

    # Build TSAN_OPTIONS.
    tsan_opts = build_tsan_options(
        suppressions_path=suppressions_path,
        halt_on_error=config.quick,
        exitcode=66 if config.quick else 0,
    )
    log.info("TSAN_OPTIONS: %s", tsan_opts)

    # Build run ID and output dir.
    run_id = f"tsan_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = config.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata.
    meta = TsanRunMeta(
        run_id=run_id,
        target_python=str(config.target_python),
        python_version=py_info.version,
        is_free_threaded=py_info.is_free_threaded,
        is_tsan=py_info.is_tsan,
        started_at=utc_now_iso(),
        suppressions_used=str(suppressions_path) if suppressions_path else "",
        quick_mode=config.quick,
        stress_count=config.stress,
        tsan_options=tsan_opts,
    )

    # Load packages.
    packages = _load_packages(config)
    meta.total_packages = len(packages)

    # Write initial metadata.
    meta_path = output_dir / "tsan_meta.json"
    write_meta_json(meta_path, meta.to_dict())

    # Resolve installer.
    installer = resolve_installer(config.installer)

    # Results file.
    results_path = output_dir / "tsan_results.jsonl"

    # Run each package.
    results: list[TsanRunResult] = []
    for i, pkg in enumerate(packages, 1):
        log.info("[%d/%d] Testing %s under TSan...", i, len(packages), pkg.package)

        pkg_output_dir = output_dir / pkg.package
        pkg_output_dir.mkdir(parents=True, exist_ok=True)

        result = run_tsan_tests(
            pkg,
            config,
            pkg_output_dir,
            installer=installer,
            tsan_options=tsan_opts,
            suppressions_path=suppressions_path,
        )
        results.append(result)

        # Update metadata in result with Python version.
        if result.metadata_path:
            try:
                md_path = Path(result.metadata_path)
                md = json.loads(md_path.read_text(encoding="utf-8"))
                md["python_version"] = py_info.version
                write_meta_json(md_path, md)
            except (OSError, json.JSONDecodeError):
                pass

        # Append to JSONL.
        append_jsonl(results_path, result.to_dict())

        status_icon = "!" if result.race_count > 0 else "-" if result.status == "no_races" else "x"
        races_str = f" ({result.race_count} races)" if result.race_count else ""
        log.info(
            "  %s %s [%s]%s (%.1fs)",
            status_icon,
            pkg.package,
            result.status,
            races_str,
            result.test_duration_s,
        )

    # Finalize metadata.
    meta.finished_at = utc_now_iso()
    meta.packages_with_races = sum(1 for r in results if r.race_count > 0)
    meta.total_races = sum(r.race_count for r in results)
    write_meta_json(meta_path, meta.to_dict())

    return meta, results
