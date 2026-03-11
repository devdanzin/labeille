"""C extension compatibility survey.

Attempts to build packages against a target Python, captures full build
output, classifies failures into fine-grained categories, and produces
clustered reports. Works on arbitrary PyPI packages, not just those in
the labeille registry.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from labeille.crash import detect_crash
from labeille.io_utils import (
    append_jsonl,
    generate_run_id,
    load_json_file,
    load_jsonl,
    utc_now_iso,
    write_meta_json,
)
from labeille.logging import get_logger
from labeille.runner import (
    InstallerBackend,
    clean_env,
    check_import,
    clone_repo,
    install_with_fallback,
    pull_repo,
    resolve_installer,
    validate_target_python,
)

log = get_logger("compat")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ErrorPattern:
    """A pattern that identifies a specific build failure category."""

    category: str
    subcategory: str
    pattern: re.Pattern[str]
    description: str
    since: str


@dataclass
class ErrorMatch:
    """A single matched error pattern in a build log."""

    category: str
    subcategory: str
    description: str
    since: str
    matched_line: str
    line_number: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "description": self.description,
            "since": self.since,
            "matched_line": self.matched_line,
            "line_number": self.line_number,
        }


@dataclass
class CompatPackageInput:
    """A package to survey, with optional metadata from the registry."""

    name: str
    repo_url: str | None = None
    install_command: str | None = None
    import_name: str | None = None
    extension_type: str = "unknown"
    source: str = ""


@dataclass
class CompatResult:
    """Result of attempting to build a single package."""

    package: str
    status: str
    exit_code: int | None = None
    duration_seconds: float = 0.0
    error_matches: list[ErrorMatch] = field(default_factory=list)
    primary_category: str = ""
    primary_subcategory: str = ""
    primary_description: str = ""
    import_error: str = ""
    crash_signature: str = ""
    extension_type: str = "unknown"
    source: str = ""
    from_mode: str = ""
    repo_url: str | None = None
    installer_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL. Sparse: omits None/empty/default values."""
        d: dict[str, Any] = {"package": self.package, "status": self.status}
        if self.exit_code is not None:
            d["exit_code"] = self.exit_code
        if self.duration_seconds:
            d["duration_seconds"] = self.duration_seconds
        if self.error_matches:
            d["error_matches"] = [m.to_dict() for m in self.error_matches]
        if self.primary_category:
            d["primary_category"] = self.primary_category
        if self.primary_subcategory:
            d["primary_subcategory"] = self.primary_subcategory
        if self.primary_description:
            d["primary_description"] = self.primary_description
        if self.import_error:
            d["import_error"] = self.import_error
        if self.crash_signature:
            d["crash_signature"] = self.crash_signature
        if self.extension_type != "unknown":
            d["extension_type"] = self.extension_type
        if self.source:
            d["source"] = self.source
        if self.from_mode:
            d["from_mode"] = self.from_mode
        if self.repo_url:
            d["repo_url"] = self.repo_url
        if self.installer_used:
            d["installer_used"] = self.installer_used
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompatResult:
        """Deserialize from JSONL dict."""
        matches = [ErrorMatch(**m) for m in data.get("error_matches", [])]
        return cls(
            package=data["package"],
            status=data["status"],
            exit_code=data.get("exit_code"),
            duration_seconds=data.get("duration_seconds", 0.0),
            error_matches=matches,
            primary_category=data.get("primary_category", ""),
            primary_subcategory=data.get("primary_subcategory", ""),
            primary_description=data.get("primary_description", ""),
            import_error=data.get("import_error", ""),
            crash_signature=data.get("crash_signature", ""),
            extension_type=data.get("extension_type", "unknown"),
            source=data.get("source", ""),
            from_mode=data.get("from_mode", ""),
            repo_url=data.get("repo_url"),
            installer_used=data.get("installer_used", ""),
        )


@dataclass
class CompatMeta:
    """Survey-level metadata."""

    survey_id: str
    target_python: str
    python_version: str
    from_mode: str
    no_binary_all: bool
    started_at: str
    finished_at: str
    total_packages: int
    installer_preference: str
    extra_patterns_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for JSON persistence."""
        d: dict[str, Any] = {
            "survey_id": self.survey_id,
            "target_python": self.target_python,
            "python_version": self.python_version,
            "from_mode": self.from_mode,
            "no_binary_all": self.no_binary_all,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_packages": self.total_packages,
            "installer_preference": self.installer_preference,
        }
        if self.extra_patterns_file:
            d["extra_patterns_file"] = self.extra_patterns_file
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompatMeta:
        """Deserialize from a dict loaded from JSON."""
        return cls(
            survey_id=data["survey_id"],
            target_python=data["target_python"],
            python_version=data["python_version"],
            from_mode=data["from_mode"],
            no_binary_all=data.get("no_binary_all", False),
            started_at=data["started_at"],
            finished_at=data.get("finished_at", ""),
            total_packages=data.get("total_packages", 0),
            installer_preference=data.get("installer_preference", "auto"),
            extra_patterns_file=data.get("extra_patterns_file"),
        )


@dataclass
class CompatSurvey:
    """Loaded survey with metadata and results."""

    meta: CompatMeta
    results: list[CompatResult]

    @property
    def by_status(self) -> dict[str, list[CompatResult]]:
        """Group results by status."""
        groups: dict[str, list[CompatResult]] = {}
        for r in self.results:
            groups.setdefault(r.status, []).append(r)
        return groups

    @property
    def by_category(self) -> dict[str, list[CompatResult]]:
        """Group build failures by primary_category."""
        groups: dict[str, list[CompatResult]] = {}
        for r in self.results:
            if r.primary_category:
                groups.setdefault(r.primary_category, []).append(r)
        return groups

    @property
    def by_subcategory(self) -> dict[str, list[CompatResult]]:
        """Group by 'category/subcategory' string."""
        groups: dict[str, list[CompatResult]] = {}
        for r in self.results:
            if r.primary_category and r.primary_subcategory:
                key = f"{r.primary_category}/{r.primary_subcategory}"
                groups.setdefault(key, []).append(r)
        return groups

    @property
    def summary_counts(self) -> dict[str, int]:
        """Status -> count."""
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Error classification engine — built-in patterns
# ---------------------------------------------------------------------------


def _p(category: str, subcategory: str, since: str, regex: str, description: str) -> ErrorPattern:
    """Shorthand for building ErrorPattern with compiled regex."""
    return ErrorPattern(
        category=category,
        subcategory=subcategory,
        pattern=re.compile(regex, re.IGNORECASE),
        description=description,
        since=since,
    )


_BUILTIN_PATTERNS: list[ErrorPattern] = [
    # --- python_header (environment issue) ---
    _p(
        "python_header",
        "python_h_missing",
        "",
        r"(?:fatal error.*Python\.h|Python\.h.*no such file)",
        "Python.h not found — Python dev headers missing",
    ),
    # --- removed_c_api ---
    _p(
        "removed_c_api",
        "PyUnicode_READY",
        "3.12",
        r"(?:error|implicit|undeclared|undefined).*\bPyUnicode_READY\b",
        "PyUnicode_READY removed in 3.12",
    ),
    _p(
        "removed_c_api",
        "_PyUnicode_Ready",
        "3.12",
        r"(?:error|implicit|undeclared|undefined).*\b_PyUnicode_Ready\b",
        "_PyUnicode_Ready (private) removed in 3.12",
    ),
    _p(
        "removed_c_api",
        "Py_UNICODE",
        "3.13",
        r"(?:error|unknown type).*\bPy_UNICODE\b",
        "Py_UNICODE type removed in 3.13",
    ),
    _p(
        "removed_c_api",
        "PyUnicode_AS_UNICODE",
        "3.13",
        r"(?:error|implicit|undeclared|undefined).*\bPyUnicode_AS_UNICODE\b",
        "PyUnicode_AS_UNICODE removed in 3.13",
    ),
    _p(
        "removed_c_api",
        "PyUnicode_GET_SIZE",
        "3.13",
        r"(?:error|implicit|undeclared|undefined).*\bPyUnicode_GET_SIZE\b",
        "PyUnicode_GET_SIZE removed in 3.13",
    ),
    _p(
        "removed_c_api",
        "tp_print",
        "3.12",
        r"(?:error|no member).*\btp_print\b",
        "tp_print slot removed in 3.12",
    ),
    _p(
        "removed_c_api",
        "PyEval_CallObject",
        "3.13",
        r"(?:error|implicit|undeclared|undefined).*\bPyEval_CallObject\b",
        "PyEval_CallObject removed in 3.13",
    ),
    _p(
        "removed_c_api",
        "PyEval_CallFunction",
        "3.13",
        r"(?:error|implicit|undeclared|undefined).*\bPyEval_CallFunction\b",
        "PyEval_CallFunction removed in 3.13",
    ),
    _p(
        "removed_c_api",
        "PyEval_CallMethod",
        "3.13",
        r"(?:error|implicit|undeclared|undefined).*\bPyEval_CallMethod\b",
        "PyEval_CallMethod removed in 3.13",
    ),
    _p(
        "removed_c_api",
        "PyObject_AsCharBuffer",
        "3.13",
        r"(?:error|implicit|undeclared|undefined).*\bPyObject_AsCharBuffer\b",
        "PyObject_AsCharBuffer removed in 3.13",
    ),
    _p(
        "removed_c_api",
        "PyObject_AsReadBuffer",
        "3.13",
        r"(?:error|implicit|undeclared|undefined).*\bPyObject_AsReadBuffer\b",
        "PyObject_AsReadBuffer removed in 3.13",
    ),
    # --- changed_struct ---
    _p(
        "changed_struct",
        "ob_type_assign",
        "3.13",
        r"(?:error|read-only|assignment).*\bob_type\b",
        "Direct ob_type assignment; use Py_SET_TYPE()",
    ),
    _p(
        "changed_struct",
        "ob_refcnt_assign",
        "3.13",
        r"(?:error|read-only|assignment).*\bob_refcnt\b",
        "Direct ob_refcnt assignment; use Py_SET_REFCNT()",
    ),
    _p(
        "changed_struct",
        "ob_size_assign",
        "3.13",
        r"(?:error|read-only|assignment).*\bob_size\b",
        "Direct ob_size assignment; use Py_SET_SIZE()",
    ),
    _p(
        "changed_struct",
        "PyFrameObject_fields",
        "3.13",
        r"(?:error|no member).*\b(?:f_code|f_back|f_locals|f_builtins|f_globals)\b",
        "Direct PyFrameObject field access; use accessor functions",
    ),
    # --- cython_incompatible ---
    _p(
        "cython_incompatible",
        "cython_too_old",
        "",
        r"cython.*(?:is not|does not|doesn't).*support.*python\s*3\.\d+",
        "Cython version does not support target Python",
    ),
    _p(
        "cython_incompatible",
        "cython_compile_error",
        "",
        r"(?:error.*cython|Cython\.Compiler).*(?:pyx|generated)",
        "Compilation error in Cython-generated code",
    ),
    _p(
        "cython_incompatible",
        "cython_deprecated",
        "",
        r"deprecated.*Cython|Cython.*deprecated",
        "Deprecated Cython API usage",
    ),
    # --- pyo3_incompatible ---
    _p(
        "pyo3_incompatible",
        "pyo3_version",
        "",
        r"pyo3.*(?:does not|doesn't).*support.*python",
        "PyO3 does not support target Python version",
    ),
    _p(
        "pyo3_incompatible",
        "maturin_version",
        "",
        r"maturin.*(?:error|unsupported|python)",
        "Maturin build error or unsupported Python",
    ),
    _p(
        "pyo3_incompatible",
        "rust_build_fail",
        "",
        r"error\[E\d+\]",
        "Rust compiler error (likely PyO3-related)",
    ),
    # --- numpy_c_api ---
    _p(
        "numpy_c_api",
        "numpy_api_version",
        "",
        r"numpy.*(?:api|abi).*(?:mismatch|version|incompatible)",
        "NumPy C API/ABI version mismatch",
    ),
    _p(
        "numpy_c_api",
        "numpy_deprecated_api",
        "",
        r"(?:NPY_NO_DEPRECATED_API|NPY_1_7_API_VERSION).*(?:error|warning)",
        "NumPy deprecated API usage",
    ),
    # --- missing_system_lib (with Python.h negative lookahead) ---
    _p(
        "missing_system_lib",
        "missing_header",
        "",
        r"fatal error:.*(?!Python\.h)\.h.*(?:no such file|not found)",
        "Missing system header file",
    ),
    _p(
        "missing_system_lib",
        "missing_library",
        "",
        r"(?:cannot find -l\w+|library not found for -l\w+)",
        "Missing system library",
    ),
    _p(
        "missing_system_lib",
        "pkg_config",
        "",
        r"(?:pkg-config.*not found|No package .* found)",
        "pkg-config dependency not satisfied",
    ),
    # --- setuptools_distutils ---
    _p(
        "setuptools_distutils",
        "distutils_removed",
        "3.12",
        r"(?:No module named|ModuleNotFoundError).*distutils",
        "distutils removed in 3.12; use setuptools",
    ),
    _p(
        "setuptools_distutils",
        "setuptools_version",
        "",
        r"(?:setuptools.*version|requires.*setuptools)",
        "setuptools version requirement not met",
    ),
    _p(
        "setuptools_distutils",
        "setup_py_error",
        "",
        r"error.*setup\.py|setup\.py.*(?:error|failed)",
        "setup.py execution error",
    ),
    _p(
        "setuptools_distutils",
        "pkg_resources",
        "3.12",
        r"(?:No module named|ModuleNotFoundError).*pkg_resources",
        "pkg_resources (from setuptools) not available",
    ),
    # --- build_backend ---
    _p(
        "build_backend",
        "backend_missing",
        "",
        r"backend.*(?:not available|not found)|build.*backend.*error",
        "PEP 517 build backend not available",
    ),
    _p(
        "build_backend",
        "mesonbuild",
        "",
        r"meson.*error|meson\.build.*error",
        "Meson build error",
    ),
    _p(
        "build_backend",
        "cmake_error",
        "",
        r"cmake.*error|CMakeLists.*error",
        "CMake build error",
    ),
    # --- compiler_error (catch-all, LAST) ---
    _p(
        "compiler_error",
        "compilation_error",
        "",
        r"error:.*(?:undeclared|undefined|incompatible|invalid|implicit declaration)",
        "Generic C/C++ compilation error",
    ),
    _p(
        "compiler_error",
        "linker_error",
        "",
        r"(?:undefined reference|undefined symbol|ld:.*error|ld returned)",
        "Linker error",
    ),
    # --- import_failure (only matched against import stderr) ---
    _p(
        "import_failure",
        "undefined_symbol",
        "",
        r"(?:undefined symbol|ImportError.*symbol).*Py\w+",
        "Undefined Python C API symbol at import time",
    ),
    _p(
        "import_failure",
        "abi_mismatch",
        "",
        r"(?:abi.*tag|not a supported wheel|cpython.*abi)",
        "ABI tag mismatch at import time",
    ),
]


_NO_SDIST_PATTERN = re.compile(
    r"no matching distribution.*--no-binary|"
    r"could not find a version.*--no-binary|"
    r"No files/directories .* sdist",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Classification engine
# ---------------------------------------------------------------------------


def classify_build_output(
    stderr: str,
    *,
    patterns: list[ErrorPattern] | None = None,
) -> list[ErrorMatch]:
    """Classify build stderr against known error patterns.

    Returns list of ErrorMatch instances ordered by line_number.
    The first match should be used as primary_category.
    """
    effective_patterns = patterns if patterns is not None else _BUILTIN_PATTERNS
    matches: list[ErrorMatch] = []
    for line_idx, line in enumerate(stderr.splitlines()):
        for pat in effective_patterns:
            if pat.pattern.search(line):
                matches.append(
                    ErrorMatch(
                        category=pat.category,
                        subcategory=pat.subcategory,
                        description=pat.description,
                        since=pat.since,
                        matched_line=line.strip(),
                        line_number=line_idx + 1,
                    )
                )
                break  # one match per line
    return matches


def load_patterns_from_yaml(path: Path) -> list[ErrorPattern]:
    """Load error patterns from a YAML file.

    Raises:
        ValueError: If the YAML structure is invalid or a regex fails to compile.
    """
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "patterns" not in data:
        raise ValueError(f"Expected top-level 'patterns' key in {path}")

    result: list[ErrorPattern] = []
    for entry in data["patterns"]:
        for key in ("category", "subcategory", "pattern", "description"):
            if key not in entry:
                raise ValueError(f"Missing required field '{key}' in pattern entry: {entry}")
        try:
            compiled = re.compile(entry["pattern"], re.IGNORECASE)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex in pattern {entry.get('subcategory', '?')!r}: "
                f"{entry['pattern']!r}: {exc}"
            ) from exc
        result.append(
            ErrorPattern(
                category=entry["category"],
                subcategory=entry["subcategory"],
                pattern=compiled,
                description=entry["description"],
                since=entry.get("since", ""),
            )
        )
    return result


def get_patterns(extra_patterns_file: Path | None = None) -> list[ErrorPattern]:
    """Return the effective pattern list.

    YAML patterns are inserted at the beginning and override built-in
    patterns with the same (category, subcategory).
    """
    if extra_patterns_file is None:
        return list(_BUILTIN_PATTERNS)
    yaml_patterns = load_patterns_from_yaml(extra_patterns_file)
    yaml_keys = {(p.category, p.subcategory) for p in yaml_patterns}
    filtered_builtins = [
        p for p in _BUILTIN_PATTERNS if (p.category, p.subcategory) not in yaml_keys
    ]
    return yaml_patterns + filtered_builtins


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def _read_packages_file(path: Path) -> list[str]:
    """Read package names from a file. Skips comments and blank lines."""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def resolve_compat_inputs(
    *,
    registry_dir: Path | None = None,
    extensions_only: bool = True,
    package_names: list[str] | None = None,
    packages_file: Path | None = None,
    top_n: int | None = None,
    target_python_version: str | None = None,
    pypi_timeout: float = 10.0,
) -> list[CompatPackageInput]:
    """Resolve and merge package inputs from all sources."""
    seen: dict[str, CompatPackageInput] = {}

    # Registry source.
    if registry_dir is not None:
        from labeille.registry import load_index, load_package

        index = load_index(registry_dir)
        entries = index.packages
        if extensions_only:
            entries = [e for e in entries if e.extension_type == "extensions"]
        entries = [e for e in entries if not e.skip]
        if target_python_version:
            entries = [e for e in entries if target_python_version not in e.skip_versions_keys]
        if top_n is not None:
            entries = sorted(entries, key=lambda e: e.download_count or 0, reverse=True)[:top_n]
        for entry in entries:
            pkg = load_package(entry.name, registry_dir)
            seen[entry.name.lower()] = CompatPackageInput(
                name=entry.name,
                repo_url=pkg.repo,
                install_command=pkg.install_command or None,
                import_name=pkg.import_name,
                extension_type=pkg.extension_type,
                source="registry",
            )

    # Inline and file sources share the same PyPI resolution logic.
    def _resolve_from_pypi(names: list[str], source: str) -> None:
        from labeille.classifier import classify_from_urls
        from labeille.resolve import extract_repo_url, fetch_pypi_metadata

        for name in names:
            if name.lower() in seen:
                continue
            meta = fetch_pypi_metadata(name, timeout=pypi_timeout)
            repo_url = None
            ext_type = "unknown"
            if meta:
                repo_url = extract_repo_url(meta)
                ext_type = classify_from_urls(meta.get("urls", []))
            seen[name.lower()] = CompatPackageInput(
                name=name,
                repo_url=repo_url,
                extension_type=ext_type,
                source=source,
            )

    if package_names:
        _resolve_from_pypi(package_names, "inline")

    if packages_file:
        file_names = _read_packages_file(packages_file)
        _resolve_from_pypi(file_names, "file")

    return list(seen.values())


# ---------------------------------------------------------------------------
# Core survey logic
# ---------------------------------------------------------------------------


def _survey_package(
    pkg: CompatPackageInput,
    target_python: Path,
    from_mode: str,
    output_dir: Path,
    timeout: int,
    repos_dir: Path | None,
    installer: InstallerBackend,
    patterns: list[ErrorPattern],
    no_binary_all: bool,
) -> CompatResult:
    """Attempt to build a single package and classify the outcome."""
    start = time.monotonic()
    result = CompatResult(
        package=pkg.name,
        status="skip",
        extension_type=pkg.extension_type,
        source=pkg.source,
        from_mode=from_mode,
        repo_url=pkg.repo_url,
    )

    with tempfile.TemporaryDirectory(prefix=f"compat-{pkg.name}-") as tmpdir:
        tmp_path = Path(tmpdir)
        venv_dir = tmp_path / "venv"

        # Build install command.
        if from_mode == "sdist":
            if no_binary_all:
                install_cmd = f"pip install --no-binary :all: {pkg.name}"
            else:
                install_cmd = f"pip install --no-binary {pkg.name} {pkg.name}"
            cwd = tmp_path
        else:
            # Source mode.
            if pkg.repo_url is None:
                result.status = "no_repo"
                result.duration_seconds = round(time.monotonic() - start, 2)
                return result
            repo_dir = repos_dir / pkg.name if repos_dir else tmp_path / "repo"
            try:
                if repo_dir.exists() and (repo_dir / ".git").exists():
                    pull_repo(repo_dir)
                else:
                    clone_repo(pkg.repo_url, repo_dir)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                result.status = "clone_error"
                result.duration_seconds = round(time.monotonic() - start, 2)
                log.warning("Clone failed for %s: %s", pkg.name, exc)
                return result
            install_cmd = pkg.install_command or "pip install -e ."
            cwd = repo_dir

        # Install.
        env = clean_env(ASAN_OPTIONS="detect_leaks=0")
        install_proc: subprocess.CompletedProcess[str] | None = None
        try:
            install_proc, actual_backend = install_with_fallback(
                python_path=target_python,
                venv_dir=venv_dir,
                install_command=install_cmd,
                cwd=cwd,
                env=env,
                timeout=timeout,
                installer=installer,
            )
            result.installer_used = actual_backend.value
        except subprocess.TimeoutExpired:
            result.status = "timeout"
            result.duration_seconds = round(time.monotonic() - start, 2)
            return result
        except (subprocess.CalledProcessError, OSError) as exc:
            result.status = "build_fail"
            result.duration_seconds = round(time.monotonic() - start, 2)
            log.warning("Build exception for %s: %s", pkg.name, exc)
            return result

        # Save build logs.
        logs_dir = output_dir / "build_logs"
        logs_dir.mkdir(exist_ok=True)
        if install_proc.stderr:
            (logs_dir / f"{pkg.name}.stderr").write_text(install_proc.stderr, encoding="utf-8")
        if install_proc.stdout:
            (logs_dir / f"{pkg.name}.stdout").write_text(install_proc.stdout, encoding="utf-8")

        # Check install result.
        result.exit_code = install_proc.returncode
        if install_proc.returncode != 0:
            stderr_text = install_proc.stderr or ""
            # Check for no-sdist case.
            if from_mode == "sdist" and _NO_SDIST_PATTERN.search(stderr_text):
                result.status = "no_sdist"
            else:
                result.status = "build_fail"
                matches = classify_build_output(stderr_text, patterns=patterns)
                result.error_matches = matches
                if matches:
                    result.primary_category = matches[0].category
                    result.primary_subcategory = matches[0].subcategory
                    result.primary_description = matches[0].description
            result.duration_seconds = round(time.monotonic() - start, 2)
            return result

        # Import check.
        import_name = pkg.import_name or pkg.name.replace("-", "_")
        venv_python = venv_dir / "bin" / "python"
        import_env = clean_env(PYTHON_JIT="0", ASAN_OPTIONS="detect_leaks=0")
        try:
            import_proc = check_import(venv_python, import_name, import_env)
        except subprocess.TimeoutExpired:
            result.status = "import_fail"
            result.import_error = "import timed out"
            result.duration_seconds = round(time.monotonic() - start, 2)
            return result

        if import_proc.returncode != 0:
            crash = detect_crash(import_proc.returncode, import_proc.stderr or "")
            if crash is not None:
                result.status = "import_crash"
                result.crash_signature = crash.signature
            else:
                result.status = "import_fail"
                result.import_error = (import_proc.stderr or "").strip()[-300:]
            # Classify import stderr.
            import_matches = classify_build_output(import_proc.stderr or "", patterns=patterns)
            result.error_matches = import_matches
            if import_matches:
                result.primary_category = import_matches[0].category
                result.primary_subcategory = import_matches[0].subcategory
                result.primary_description = import_matches[0].description
        else:
            result.status = "build_ok"

    result.duration_seconds = round(time.monotonic() - start, 2)
    return result


def run_compat_survey(
    packages: list[CompatPackageInput],
    target_python: Path,
    *,
    from_mode: str = "sdist",
    no_binary_all: bool = False,
    output_dir: Path,
    timeout: int = 600,
    workers: int = 1,
    repos_dir: Path | None = None,
    installer_preference: str = "auto",
    extra_patterns_file: Path | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> CompatSurvey:
    """Run a compatibility survey across packages."""
    python_version = validate_target_python(target_python)
    installer = resolve_installer(installer_preference)
    patterns = get_patterns(extra_patterns_file)

    survey_id = generate_run_id("compat")
    survey_dir = output_dir / survey_id
    survey_dir.mkdir(parents=True, exist_ok=True)
    (survey_dir / "build_logs").mkdir()

    meta = CompatMeta(
        survey_id=survey_id,
        target_python=str(target_python),
        python_version=python_version,
        from_mode=from_mode,
        no_binary_all=no_binary_all,
        started_at=utc_now_iso(),
        finished_at="",
        total_packages=len(packages),
        installer_preference=installer_preference,
        extra_patterns_file=str(extra_patterns_file) if extra_patterns_file else None,
    )

    results: list[CompatResult] = []
    jsonl_lock = threading.Lock()
    jsonl_path = survey_dir / "compat_results.jsonl"

    def _run_one(pkg: CompatPackageInput) -> CompatResult:
        r = _survey_package(
            pkg,
            target_python,
            from_mode,
            survey_dir,
            timeout,
            repos_dir,
            installer,
            patterns,
            no_binary_all,
        )
        with jsonl_lock:
            append_jsonl(jsonl_path, r.to_dict())
        if progress_callback:
            progress_callback(pkg.name, r.status)
        return r

    if workers <= 1:
        for pkg in packages:
            results.append(_run_one(pkg))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, pkg): pkg for pkg in packages}
            for future in as_completed(futures):
                results.append(future.result())

    meta.finished_at = utc_now_iso()
    write_meta_json(survey_dir / "compat_meta.json", meta.to_dict())

    return CompatSurvey(meta=meta, results=results)


def load_compat_survey(survey_dir: Path) -> CompatSurvey:
    """Load a saved survey from disk."""
    meta_path = survey_dir / "compat_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"compat_meta.json not found in {survey_dir}")
    meta = CompatMeta.from_dict(load_json_file(meta_path))

    jsonl_path = survey_dir / "compat_results.jsonl"
    results: list[CompatResult] = []
    if jsonl_path.exists():
        results = load_jsonl(jsonl_path, CompatResult.from_dict)

    return CompatSurvey(meta=meta, results=results)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@dataclass
class CompatDiffEntry:
    """A single package that changed between two surveys."""

    package: str
    status_a: str
    status_b: str
    category_a: str
    category_b: str
    description_a: str
    description_b: str

    @property
    def is_regression(self) -> bool:
        """True if the package built successfully in survey A but failed in B."""
        return self.status_a == "build_ok" and self.status_b != "build_ok"

    @property
    def is_fix(self) -> bool:
        """True if the package failed in survey A but built successfully in B."""
        return self.status_a != "build_ok" and self.status_b == "build_ok"


@dataclass
class CompatDiff:
    """Comparison between two surveys."""

    survey_a_id: str
    survey_b_id: str
    python_a: str
    python_b: str
    entries: list[CompatDiffEntry]

    @property
    def regressions(self) -> list[CompatDiffEntry]:
        """Packages that built in survey A but failed in survey B."""
        return [e for e in self.entries if e.is_regression]

    @property
    def fixes(self) -> list[CompatDiffEntry]:
        """Packages that failed in survey A but built in survey B."""
        return [e for e in self.entries if e.is_fix]

    @property
    def category_changes(self) -> list[CompatDiffEntry]:
        """Packages that changed category without regressing or being fixed."""
        return [
            e
            for e in self.entries
            if not e.is_regression and not e.is_fix and e.category_a != e.category_b
        ]


def diff_surveys(a: CompatSurvey, b: CompatSurvey) -> CompatDiff:
    """Compare two surveys. Only packages present in both are compared."""
    a_map = {r.package: r for r in a.results}
    b_map = {r.package: r for r in b.results}
    shared = sorted(set(a_map) & set(b_map))
    entries: list[CompatDiffEntry] = []
    for name in shared:
        ra, rb = a_map[name], b_map[name]
        if ra.status != rb.status or ra.primary_category != rb.primary_category:
            entries.append(
                CompatDiffEntry(
                    package=name,
                    status_a=ra.status,
                    status_b=rb.status,
                    category_a=ra.primary_category,
                    category_b=rb.primary_category,
                    description_a=ra.primary_description,
                    description_b=rb.primary_description,
                )
            )
    return CompatDiff(
        survey_a_id=a.meta.survey_id,
        survey_b_id=b.meta.survey_id,
        python_a=a.meta.python_version,
        python_b=b.meta.python_version,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Display and export
# ---------------------------------------------------------------------------

_STATUS_ORDER = [
    "build_ok",
    "build_fail",
    "import_fail",
    "import_crash",
    "no_sdist",
    "no_repo",
    "clone_error",
    "timeout",
    "skip",
]


def format_compat_report(survey: CompatSurvey) -> str:
    """Format a survey for terminal display."""
    from labeille.formatting import format_duration, format_percentage, format_table

    lines: list[str] = []
    meta = survey.meta
    n = len(survey.results)

    if n == 0:
        return "No packages surveyed."

    # Header.
    lines.append(f"Compatibility Survey: {meta.survey_id}")
    lines.append(f"  Python: {meta.python_version}")
    lines.append(f"  Mode: {meta.from_mode}")
    if meta.no_binary_all:
        lines.append("  --no-binary :all: enabled")
    total_dur = sum(r.duration_seconds for r in survey.results)
    lines.append(f"  Packages: {n}  Duration: {format_duration(total_dur)}")
    lines.append("")

    # Status overview.
    lines.append("Status overview:")
    counts = survey.summary_counts
    rows: list[list[str]] = []
    for status in _STATUS_ORDER:
        c = counts.get(status, 0)
        if c > 0:
            rows.append([status, str(c), format_percentage(c, n)])
    if rows:
        lines.append(format_table(["Status", "Count", "%"], rows))
    lines.append("")

    # Build failures by category.
    by_cat = survey.by_category
    if by_cat:
        fail_total = counts.get("build_fail", 0)
        lines.append("Build failures by category:")
        cat_rows: list[list[str]] = []
        for cat, pkgs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
            cat_rows.append([cat, str(len(pkgs)), format_percentage(len(pkgs), fail_total)])
        lines.append(format_table(["Category", "Count", "% of failures"], cat_rows))
        lines.append("")

    # Top subcategories.
    by_sub = survey.by_subcategory
    if by_sub:
        lines.append("Top failure subcategories:")
        sub_rows: list[list[str]] = []
        sorted_subs = sorted(by_sub.items(), key=lambda x: -len(x[1]))[:15]
        for sub_key, pkgs in sorted_subs:
            sample = ", ".join(r.package for r in pkgs[:5])
            if len(pkgs) > 5:
                sample += ", ..."
            since = pkgs[0].primary_description if pkgs else ""
            sub_rows.append([sub_key, str(len(pkgs)), since, sample])
        lines.append(format_table(["Subcategory", "Count", "Description", "Packages"], sub_rows))
        lines.append("")

    return "\n".join(lines)


def format_compat_diff(diff: CompatDiff) -> str:
    """Format a survey diff for terminal display."""
    from labeille.formatting import format_table

    lines: list[str] = []
    lines.append(f"Diff: {diff.survey_a_id} vs {diff.survey_b_id}")
    lines.append(f"  Python A: {diff.python_a}  Python B: {diff.python_b}")
    lines.append("")

    if not diff.entries:
        lines.append("No differences found.")
        return "\n".join(lines)

    lines.append(
        f"Summary: {len(diff.regressions)} regressions, "
        f"{len(diff.fixes)} fixes, {len(diff.category_changes)} category changes"
    )
    lines.append("")

    if diff.regressions:
        lines.append("Regressions (build_ok -> failure):")
        rows = [[e.package, e.status_b, e.category_b, e.description_b] for e in diff.regressions]
        lines.append(format_table(["Package", "New status", "Category", "Description"], rows))
        lines.append("")

    if diff.fixes:
        lines.append("Fixes (failure -> build_ok):")
        rows = [[e.package, e.status_a, e.category_a] for e in diff.fixes]
        lines.append(format_table(["Package", "Old status", "Old category"], rows))
        lines.append("")

    if diff.category_changes:
        lines.append("Category changes:")
        rows = [
            [e.package, e.category_a or "-", e.category_b or "-"] for e in diff.category_changes
        ]
        lines.append(format_table(["Package", "Old category", "New category"], rows))
        lines.append("")

    return "\n".join(lines)


def export_compat_markdown(survey: CompatSurvey) -> str:
    """Export a survey as a shareable markdown report."""
    lines: list[str] = []
    meta = survey.meta
    n = len(survey.results)

    lines.append(f"# Compatibility Survey: {meta.survey_id}")
    lines.append("")
    lines.append(f"- **Python:** {meta.python_version}")
    lines.append(f"- **Mode:** {meta.from_mode}")
    lines.append(f"- **Packages:** {n}")
    lines.append(f"- **Date:** {meta.started_at[:19]}")
    lines.append("")

    if n == 0:
        lines.append("No packages surveyed.")
        return "\n".join(lines)

    # Summary table.
    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count | % |")
    lines.append("|---|---:|---:|")
    counts = survey.summary_counts
    for status in _STATUS_ORDER:
        c = counts.get(status, 0)
        if c > 0:
            pct = f"{c / n * 100:.1f}%"
            lines.append(f"| {status} | {c} | {pct} |")
    lines.append("")

    # Failures by category.
    by_cat = survey.by_category
    if by_cat:
        lines.append("## Failures by Category")
        lines.append("")
        lines.append("| Category | Count | Packages |")
        lines.append("|---|---:|---|")
        for cat, pkgs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
            sample = ", ".join(r.package for r in pkgs[:8])
            if len(pkgs) > 8:
                sample += ", ..."
            lines.append(f"| {cat} | {len(pkgs)} | {sample} |")
        lines.append("")

    # Subcategory detail.
    by_sub = survey.by_subcategory
    if by_sub:
        lines.append("## Subcategory Detail")
        lines.append("")
        lines.append("| Subcategory | Since | Count | Packages |")
        lines.append("|---|---|---:|---|")
        for sub_key, pkgs in sorted(by_sub.items(), key=lambda x: -len(x[1])):
            since = pkgs[0].error_matches[0].since if pkgs and pkgs[0].error_matches else ""
            sample = ", ".join(r.package for r in pkgs[:5])
            if len(pkgs) > 5:
                sample += ", ..."
            lines.append(f"| {sub_key} | {since} | {len(pkgs)} | {sample} |")
        lines.append("")

    lines.append(f"*Generated by labeille compat on {meta.started_at[:19]}*")
    return "\n".join(lines)


def format_patterns_table(
    patterns: list[ErrorPattern],
    *,
    category_filter: str | None = None,
) -> str:
    """Format the pattern list for terminal display."""
    from labeille.formatting import format_table

    filtered = patterns
    if category_filter:
        filtered = [p for p in patterns if p.category == category_filter]
    rows = [[p.category, p.subcategory, p.since, p.description] for p in filtered]
    return format_table(["Category", "Subcategory", "Since", "Description"], rows)
