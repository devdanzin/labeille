"""Build C extension packages and generate compile_commands.json.

Wraps package builds with Bear to intercept compiler invocations and
produce compilation databases for clang-tidy analysis. Falls back to
build-system-specific mechanisms (Meson, CMake) when Bear is not
available.

Designed for integration with cext-review-toolkit's Tier 2 analysis.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

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
)

log = get_logger("cext_build")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CextBuildConfig:
    """Configuration for a cext-build run."""

    target_python: Path
    output_dir: Path = field(default_factory=lambda: Path("cext-builds"))
    registry_dir: Path | None = None
    repos_dir: Path | None = None
    venvs_dir: Path | None = None
    packages_filter: list[str] | None = None
    top_n: int | None = None
    timeout: int = 600
    installer: str = "auto"
    no_build_isolation: bool = True
    bear_path: str | None = None
    skip_if_exists: bool = True
    verbose: bool = False


# ---------------------------------------------------------------------------
# Result data models
# ---------------------------------------------------------------------------


@dataclass
class CextBuildResult:
    """Result of building a single C extension package."""

    package: str = ""
    status: str = ""  # ok, build_fail, no_compile_db, no_repo, clone_error,
    #                    timeout, skipped, pure_python
    compile_db_path: str | None = None
    compile_db_entries: int = 0
    compile_db_method: str = ""  # bear, meson, cmake, none
    build_duration_s: float = 0.0
    exit_code: int | None = None
    error_summary: str = ""
    repo_dir: str = ""
    build_system: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL. Sparse output."""
        d: dict[str, Any] = {"package": self.package, "status": self.status}
        if self.compile_db_path:
            d["compile_db_path"] = self.compile_db_path
        if self.compile_db_entries:
            d["compile_db_entries"] = self.compile_db_entries
        if self.compile_db_method:
            d["compile_db_method"] = self.compile_db_method
        d["build_duration_s"] = round(self.build_duration_s, 2)
        if self.exit_code is not None:
            d["exit_code"] = self.exit_code
        if self.error_summary:
            d["error_summary"] = self.error_summary
        if self.repo_dir:
            d["repo_dir"] = self.repo_dir
        if self.build_system:
            d["build_system"] = self.build_system
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CextBuildResult:
        """Deserialize from dict, ignoring unknown fields."""
        from labeille.io_utils import dataclass_from_dict

        return dataclass_from_dict(cls, data)


@dataclass
class CextBuildMeta:
    """Metadata for a cext-build run."""

    build_id: str
    target_python: str
    python_version: str
    started_at: str
    finished_at: str = ""
    total_packages: int = 0
    built_ok: int = 0
    compile_db_generated: int = 0
    bear_available: bool = False
    bear_version: str = ""
    no_build_isolation: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {k: v for k, v in asdict(self).items() if v or isinstance(v, (bool, int))}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CextBuildMeta:
        """Deserialize from dict, ignoring unknown fields."""
        from labeille.io_utils import dataclass_from_dict

        return dataclass_from_dict(cls, data)


# ---------------------------------------------------------------------------
# Bear detection
# ---------------------------------------------------------------------------


def detect_bear() -> tuple[str | None, str]:
    """Detect bear and return (path, version).

    Returns ``(None, "")`` if bear is not installed.
    """
    bear_path = shutil.which("bear")
    if bear_path is None:
        return None, ""
    try:
        proc = subprocess.run(
            [bear_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = (proc.stdout.strip() or proc.stderr.strip()).split("\n")[0]
        return bear_path, version
    except Exception:  # noqa: BLE001
        return bear_path, "unknown"


# ---------------------------------------------------------------------------
# Build system detection
# ---------------------------------------------------------------------------


def detect_build_system(repo_dir: Path) -> str:
    """Detect the build system from project files.

    Returns one of: ``"meson"``, ``"cmake"``, ``"setuptools"``,
    ``"flit"``, ``"hatch"``, ``"pdm"``, ``"unknown"``.
    """
    # Direct file checks take priority.
    if (repo_dir / "meson.build").exists():
        return "meson"
    if (repo_dir / "CMakeLists.txt").exists():
        return "cmake"

    # Check pyproject.toml build-backend.
    pyproject = repo_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            backend = data.get("build-system", {}).get("build-backend", "")
            if "mesonpy" in backend or "meson" in backend:
                return "meson"
            if "scikit_build" in backend or "skbuild" in backend:
                return "cmake"
            if "flit" in backend:
                return "flit"
            if "hatch" in backend:
                return "hatch"
            if "pdm" in backend:
                return "pdm"
            if "setuptools" in backend:
                return "setuptools"
        except (OSError, tomllib.TOMLDecodeError):
            pass

    # Fallback to setup.py/setup.cfg.
    if (repo_dir / "setup.py").exists() or (repo_dir / "setup.cfg").exists():
        return "setuptools"

    return "unknown"


def extract_build_requires(repo_dir: Path) -> list[str]:
    """Extract ``[build-system].requires`` from pyproject.toml.

    Returns a safe fallback if pyproject.toml is missing or unparseable.
    """
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.exists():
        return ["setuptools", "wheel"]
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        reqs = data.get("build-system", {}).get("requires", [])
        return reqs if reqs else ["setuptools", "wheel"]
    except (OSError, tomllib.TOMLDecodeError):
        return ["setuptools", "wheel"]


# ---------------------------------------------------------------------------
# Compile database helpers
# ---------------------------------------------------------------------------

_C_EXTENSIONS = {".c", ".cpp", ".cxx", ".cc", ".h", ".hpp"}


def find_compile_db(repo_dir: Path) -> Path | None:
    """Search for compile_commands.json in a repo and its build dirs.

    Checks the repo root first, then common build directories
    (``builddir/``, ``build/``, ``_skbuild/``, ``_cmake_build/``).
    """
    # Repo root.
    root_db = repo_dir / "compile_commands.json"
    if root_db.exists():
        return root_db

    # Known build directories.
    for subdir_name in ("builddir", "build", "_skbuild", "_cmake_build"):
        candidate = repo_dir / subdir_name / "compile_commands.json"
        if candidate.exists():
            return candidate

    # Search all immediate child directories.
    try:
        for child in repo_dir.iterdir():
            if child.is_dir():
                candidate = child / "compile_commands.json"
                if candidate.exists():
                    return candidate
    except OSError:
        pass

    return None


def postprocess_compile_db(compile_db_path: Path, repo_dir: Path) -> int:
    """Fix paths in compile_commands.json and count valid entries.

    Rewrites file paths that reference build temp directories to
    point to actual source files in *repo_dir* where possible.

    Returns the number of entries with valid (existing) source file paths.
    """
    try:
        entries = json.loads(compile_db_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0

    if not isinstance(entries, list):
        return 0

    valid = 0
    for entry in entries:
        file_path = entry.get("file", "")
        if not file_path:
            continue

        # Resolve relative to directory field.
        directory = Path(entry.get("directory", str(repo_dir)))
        resolved = Path(file_path)
        if not resolved.is_absolute():
            resolved = directory / resolved

        if resolved.exists():
            valid += 1
            continue

        # Try to find the file in repo_dir by basename.
        basename = resolved.name
        suffix = resolved.suffix
        if suffix not in _C_EXTENSIONS:
            continue

        matches = list(repo_dir.rglob(basename))
        if len(matches) == 1:
            entry["file"] = str(matches[0])
            entry["directory"] = str(matches[0].parent)
            valid += 1

    # Write back.
    try:
        compile_db_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        log.warning("Could not write corrected compile_commands.json: %s", exc)

    return valid


# ---------------------------------------------------------------------------
# Per-package build
# ---------------------------------------------------------------------------


def build_package_cext(
    pkg: Any,
    config: CextBuildConfig,
    pkg_output_dir: Path,
    *,
    bear_path: str | None = None,
    installer: InstallerBackend = InstallerBackend.PIP,
) -> CextBuildResult:
    """Build a single package and generate its compile database.

    Steps:
        1. Skip pure Python packages.
        2. Clone or update the repo.
        3. Detect the build system.
        4. Create a venv with the target Python.
        5. Install build dependencies (for --no-build-isolation).
        6. Build with bear (or fallback to build-system-specific method).
        7. Find and post-process compile_commands.json.
        8. Save build log and symlink repo into output.
    """
    result = CextBuildResult(package=pkg.package)

    # Step 1: Skip pure Python.
    ext_type = getattr(pkg, "extension_type", "unknown")
    if ext_type == "pure":
        result.status = "pure_python"
        return result

    # Check skip_if_exists.
    if config.skip_if_exists:
        existing_db = pkg_output_dir / "compile_commands.json"
        if existing_db.exists():
            result.status = "skipped"
            return result

    # Step 2: Clone.
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

    result.repo_dir = str(repo_dir)

    # Step 3: Detect build system.
    build_sys = detect_build_system(repo_dir)
    result.build_system = build_sys

    # Step 4: Create venv.
    venvs_base = config.venvs_dir or (config.output_dir / "_venvs")
    venvs_base.mkdir(parents=True, exist_ok=True)
    venv_dir = venvs_base / pkg.package

    try:
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
        create_venv(config.target_python, venv_dir, installer)
    except (subprocess.CalledProcessError, OSError) as exc:
        result.status = "build_fail"
        result.error_summary = f"Venv creation failed: {exc}"
        return result

    venv_python = venv_dir / "bin" / "python"
    env = clean_env(ASAN_OPTIONS="detect_leaks=0")

    # Step 5: Install build dependencies.
    if config.no_build_isolation:
        build_deps = extract_build_requires(repo_dir)
        if build_deps:
            deps_str = " ".join(f"'{d}'" for d in build_deps)
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
                log.warning(
                    "Build deps install failed for %s: %s (continuing anyway)",
                    pkg.package,
                    exc,
                )

    # Step 6: Build.
    install_cmd = getattr(pkg, "install_command", None) or "pip install -e ."

    if config.no_build_isolation and "--no-build-isolation" not in install_cmd:
        install_cmd = install_cmd.replace("pip install", "pip install --no-build-isolation", 1)

    start = time.monotonic()

    if bear_path:
        old_db = repo_dir / "compile_commands.json"
        if old_db.exists():
            old_db.unlink()
        full_cmd = f"{bear_path} -- {install_cmd}"
        log.info("Building %s with bear: %s", pkg.package, full_cmd)
    else:
        full_cmd = install_cmd
        if build_sys == "cmake":
            env["CMAKE_ARGS"] = env.get("CMAKE_ARGS", "") + " -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        log.info("Building %s (no bear): %s", pkg.package, full_cmd)

    try:
        proc = install_package(
            venv_python,
            full_cmd,
            cwd=repo_dir,
            env=env,
            timeout=config.timeout,
            installer=installer,
        )
        result.exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        result.status = "timeout"
        result.build_duration_s = time.monotonic() - start
        return result
    except (subprocess.CalledProcessError, OSError) as exc:
        result.status = "build_fail"
        result.error_summary = str(exc)[:200]
        result.build_duration_s = time.monotonic() - start
        return result

    result.build_duration_s = time.monotonic() - start

    # Save build log.
    log_text = ""
    if proc.stdout:
        log_text += proc.stdout
    if proc.stderr:
        log_text += "\n--- STDERR ---\n" + proc.stderr
    if log_text:
        try:
            (pkg_output_dir / "build.log").write_text(log_text, encoding="utf-8")
        except OSError as exc:
            log.warning("Could not save build log for %s: %s", pkg.package, exc)

    if proc.returncode != 0:
        result.status = "build_fail"
        result.error_summary = (proc.stderr or "")[-500:]
        return result

    # Step 7: Find and post-process compile database.
    compile_db = find_compile_db(repo_dir)

    if compile_db is not None:
        result.compile_db_method = "bear" if bear_path else build_sys
        valid_entries = postprocess_compile_db(compile_db, repo_dir)
        result.compile_db_entries = valid_entries

        output_db = pkg_output_dir / "compile_commands.json"
        if compile_db != output_db:
            shutil.copy2(compile_db, output_db)
        result.compile_db_path = str(output_db)

        if valid_entries == 0:
            result.status = "no_compile_db"
            result.compile_db_method = "none"
        else:
            result.status = "ok"
    else:
        result.status = "no_compile_db"
        result.compile_db_method = "none"

    # Step 8: Symlink repo into output directory.
    repo_link = pkg_output_dir / "repo"
    if not repo_link.exists():
        try:
            repo_link.symlink_to(repo_dir.resolve())
        except OSError:
            pass  # Symlinks may not work on all platforms/filesystems.

    return result


# ---------------------------------------------------------------------------
# Full run orchestration
# ---------------------------------------------------------------------------


def _load_packages(config: CextBuildConfig) -> list[Any]:
    """Load and filter packages for building.

    Filters to extension packages only (``extension_type != "pure"``).
    """
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
                packages.append(pkg)
            except (FileNotFoundError, OSError, ValueError) as exc:
                log.warning("Package %s not found in registry: %s", name, exc)

        if config.top_n and len(packages) > config.top_n:
            packages = packages[: config.top_n]
    elif config.packages_filter:
        from labeille.classifier import classify_from_urls
        from labeille.registry import PackageEntry
        from labeille.resolve import extract_repo_url, fetch_pypi_metadata

        for name in config.packages_filter:
            meta = fetch_pypi_metadata(name, timeout=10.0)
            repo_url = None
            ext_type: Literal["pure", "extensions", "unknown"] = "unknown"
            if meta:
                repo_url = extract_repo_url(meta)
                ext_type = classify_from_urls(meta.get("urls", []))
            packages.append(
                PackageEntry(
                    package=name,
                    repo=repo_url,
                    extension_type=ext_type,
                )
            )

    # Filter to extension packages.
    packages = [p for p in packages if getattr(p, "extension_type", "unknown") != "pure"]

    if not packages:
        log.warning("No C extension packages found after filtering.")

    return packages


def run_cext_builds(
    config: CextBuildConfig,
) -> tuple[CextBuildMeta, list[CextBuildResult]]:
    """Execute a complete cext-build run.

    1. Detect bear.
    2. Profile target Python.
    3. Load packages (from registry or inline list).
    4. Filter to C extension packages.
    5. Build each package and generate compile databases.
    6. Save metadata and results.
    """
    from labeille.bench.system import capture_python_profile
    from labeille.io_utils import utc_now_iso

    # Detect bear.
    bear_path_resolved, bear_version = detect_bear()
    if config.bear_path:
        bear_path_resolved = config.bear_path

    if bear_path_resolved:
        log.info("Bear found: %s (%s)", bear_path_resolved, bear_version)
    else:
        log.warning(
            "Bear not found. Will use build-system fallbacks "
            "(Meson/CMake only). Install bear for full coverage: "
            "apt install bear (Ubuntu) or brew install bear (macOS)."
        )

    # Profile target Python.
    py_profile = capture_python_profile(config.target_python)

    # Build run ID and output dir.
    build_id = f"cext_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = config.output_dir / build_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata.
    meta = CextBuildMeta(
        build_id=build_id,
        target_python=str(config.target_python),
        python_version=py_profile.version,
        started_at=utc_now_iso(),
        bear_available=bear_path_resolved is not None,
        bear_version=bear_version,
        no_build_isolation=config.no_build_isolation,
    )

    # Load packages.
    packages = _load_packages(config)
    meta.total_packages = len(packages)

    # Write initial metadata.
    meta_path = output_dir / "cext_meta.json"
    write_meta_json(meta_path, meta.to_dict())

    # Resolve installer.
    installer = resolve_installer(config.installer)

    # Results file.
    results_path = output_dir / "cext_results.jsonl"

    # Build each package.
    results: list[CextBuildResult] = []
    for i, pkg in enumerate(packages, 1):
        log.info("[%d/%d] Building %s...", i, len(packages), pkg.package)

        pkg_output_dir = output_dir / pkg.package
        pkg_output_dir.mkdir(parents=True, exist_ok=True)

        result = build_package_cext(
            pkg,
            config,
            pkg_output_dir,
            bear_path=bear_path_resolved,
            installer=installer,
        )
        results.append(result)

        # Append to JSONL.
        append_jsonl(results_path, result.to_dict())

        status_icon = "✓" if result.status == "ok" else "✗" if "fail" in result.status else "—"
        entries_str = f" ({result.compile_db_entries} TUs)" if result.compile_db_entries else ""
        log.info(
            "  %s %s [%s]%s (%.1fs)",
            status_icon,
            pkg.package,
            result.status,
            entries_str,
            result.build_duration_s,
        )

    # Finalize metadata.
    meta.finished_at = utc_now_iso()
    meta.built_ok = sum(1 for r in results if r.status == "ok")
    meta.compile_db_generated = sum(1 for r in results if r.compile_db_entries > 0)
    write_meta_json(meta_path, meta.to_dict())

    return meta, results
