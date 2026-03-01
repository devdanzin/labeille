"""Extension module GIL compatibility detection.

Provides two complementary approaches to determining whether a
package's C extensions support free-threaded CPython:

1. Runtime probe: imports the package in a free-threaded interpreter
   and detects GIL fallback via sys._is_gil_enabled().
2. Source scan: searches C/C++ files for Py_mod_gil declarations.

The runtime probe tells you what actually happens. The source scan
tells you what the developer intended.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("labeille")


@dataclass
class ExtensionInfo:
    """Information about a single extension module's GIL compatibility."""

    module_name: str
    is_extension: bool = False
    import_ok: bool = True
    import_error: str | None = None
    triggered_gil_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtensionInfo:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ModGilDeclaration:
    """A Py_mod_gil declaration found in source code."""

    file: str
    line_number: int
    line_text: str
    is_not_used: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModGilDeclaration:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SourceScanResult:
    """Result of scanning source code for Py_mod_gil declarations."""

    files_scanned: int = 0
    files_with_mod_gil: int = 0
    declarations: list[ModGilDeclaration] = field(default_factory=list)

    @property
    def has_any_declaration(self) -> bool:
        return len(self.declarations) > 0

    @property
    def all_not_used(self) -> bool:
        """True if every declaration is Py_MOD_GIL_NOT_USED."""
        return len(self.declarations) > 0 and all(d.is_not_used for d in self.declarations)

    @property
    def has_required(self) -> bool:
        """True if any declaration is Py_MOD_GIL_USED."""
        return any(not d.is_not_used for d in self.declarations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "files_with_mod_gil": self.files_with_mod_gil,
            "declarations": [d.to_dict() for d in self.declarations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceScanResult:
        decls = [ModGilDeclaration.from_dict(d) for d in data.get("declarations", [])]
        return cls(
            files_scanned=data.get("files_scanned", 0),
            files_with_mod_gil=data.get("files_with_mod_gil", 0),
            declarations=decls,
        )


@dataclass
class ExtensionCompat:
    """Complete GIL compatibility assessment for a package."""

    package: str
    is_pure_python: bool = True
    extensions: list[ExtensionInfo] = field(default_factory=list)
    gil_fallback_active: bool = False
    all_extensions_compatible: bool = True
    import_ok: bool = True
    import_error: str | None = None
    source_scan: SourceScanResult | None = None
    probe_error: str | None = None

    @property
    def fully_compatible(self) -> bool:
        """True if the package is fully free-threading compatible.

        This means: importable, no GIL fallback, and either pure
        Python or all extensions declare Py_MOD_GIL_NOT_USED.
        """
        return self.import_ok and not self.gil_fallback_active and self.all_extensions_compatible

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "package": self.package,
            "is_pure_python": self.is_pure_python,
            "extensions": [e.to_dict() for e in self.extensions],
            "gil_fallback_active": self.gil_fallback_active,
            "all_extensions_compatible": self.all_extensions_compatible,
            "import_ok": self.import_ok,
            "import_error": self.import_error,
            "probe_error": self.probe_error,
        }
        if self.source_scan is not None:
            d["source_scan"] = self.source_scan.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtensionCompat:
        exts = [ExtensionInfo.from_dict(e) for e in data.get("extensions", [])]
        scan_data = data.get("source_scan")
        scan = SourceScanResult.from_dict(scan_data) if scan_data else None
        return cls(
            package=data["package"],
            is_pure_python=data.get("is_pure_python", True),
            extensions=exts,
            gil_fallback_active=data.get("gil_fallback_active", False),
            all_extensions_compatible=data.get("all_extensions_compatible", True),
            import_ok=data.get("import_ok", True),
            import_error=data.get("import_error"),
            source_scan=scan,
            probe_error=data.get("probe_error"),
        )


# ---------------------------------------------------------------------------
# Runtime GIL fallback probe
# ---------------------------------------------------------------------------

# The probe script runs inside the target free-threaded Python.
# It must be self-contained (no labeille imports).
_GIL_PROBE_SCRIPT = r'''
import importlib
import json
import sys

def probe_package(package_name):
    """Import a package and detect GIL fallback."""
    result = {
        "package": package_name,
        "import_ok": False,
        "import_error": None,
        "gil_enabled_before": None,
        "gil_enabled_after": None,
        "gil_fallback": False,
        "is_pure_python": True,
        "extensions_found": [],
    }

    # Check GIL state before import.
    try:
        result["gil_enabled_before"] = sys._is_gil_enabled()
    except AttributeError:
        # Not a free-threaded build.
        result["import_error"] = "Not a free-threaded Python build"
        print(json.dumps(result))
        return

    # Import the package.
    try:
        pkg = importlib.import_module(package_name)
        result["import_ok"] = True
    except ImportError as e:
        result["import_error"] = str(e)
        print(json.dumps(result))
        return
    except Exception as e:
        result["import_error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(result))
        return

    # Check GIL state after import.
    result["gil_enabled_after"] = sys._is_gil_enabled()
    result["gil_fallback"] = (
        not result["gil_enabled_before"]
        and result["gil_enabled_after"]
    )

    # Check if the package has extension modules.
    # Walk through the package's submodules if it has __path__.
    checked = set()
    to_check = [package_name]
    if hasattr(pkg, "__path__"):
        try:
            import pkgutil
            for importer, modname, ispkg in pkgutil.walk_packages(
                pkg.__path__, prefix=package_name + ".",
                onerror=lambda x: None,
            ):
                to_check.append(modname)
                if len(to_check) > 200:  # Safety limit.
                    break
        except Exception:
            pass

    for modname in to_check:
        if modname in checked:
            continue
        checked.add(modname)
        try:
            mod = importlib.import_module(modname)
            filepath = getattr(mod, "__file__", "") or ""
            if filepath.endswith((".so", ".pyd", ".dylib")):
                result["is_pure_python"] = False
                result["extensions_found"].append({
                    "module_name": modname,
                    "is_extension": True,
                    "file": filepath,
                })
        except Exception:
            pass  # Some submodules may not be importable.

    print(json.dumps(result))

probe_package(sys.argv[1])
'''


def probe_gil_fallback(
    package_name: str,
    venv_python: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 60,
) -> ExtensionCompat:
    """Probe a package's GIL fallback behavior in a venv.

    Imports the package in the target Python and checks whether
    the GIL was re-enabled after import.

    Args:
        package_name: The top-level import name of the package.
            This may differ from the PyPI name (e.g., "PIL" for
            "Pillow", "yaml" for "PyYAML").
        venv_python: Path to the Python executable in the venv
            where the package is installed.
        env: Environment variables for the subprocess. Should
            include PYTHON_GIL=0 to ensure free-threading is active.
        timeout: Timeout in seconds for the probe.

    Returns:
        ExtensionCompat with runtime probe results.
    """
    compat = ExtensionCompat(package=package_name)

    run_env = dict(os.environ)
    run_env["PYTHON_GIL"] = "0"
    run_env.pop("PYTHONHOME", None)
    run_env.pop("PYTHONPATH", None)
    run_env["ASAN_OPTIONS"] = "detect_leaks=0"
    if env:
        run_env.update(env)

    try:
        proc = subprocess.run(
            [str(venv_python), "-c", _GIL_PROBE_SCRIPT, package_name],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )

        if proc.returncode != 0:
            compat.import_ok = False
            compat.probe_error = (
                f"Probe exited with code {proc.returncode}: {proc.stderr.strip()[:500]}"
            )
            return compat

        if not proc.stdout.strip():
            compat.probe_error = "Probe produced no output"
            return compat

        data = json.loads(proc.stdout.strip())

        compat.import_ok = data.get("import_ok", False)
        compat.import_error = data.get("import_error")
        compat.gil_fallback_active = data.get("gil_fallback", False)
        compat.is_pure_python = data.get("is_pure_python", True)

        if not compat.is_pure_python:
            compat.all_extensions_compatible = not compat.gil_fallback_active

        for ext_data in data.get("extensions_found", []):
            compat.extensions.append(
                ExtensionInfo(
                    module_name=ext_data["module_name"],
                    is_extension=True,
                    import_ok=True,
                )
            )

    except subprocess.TimeoutExpired:
        compat.probe_error = f"Probe timed out after {timeout}s"
        compat.import_ok = False
    except json.JSONDecodeError as exc:
        compat.probe_error = f"Probe output not valid JSON: {exc}"
    except (FileNotFoundError, OSError) as exc:
        compat.probe_error = f"Could not run probe: {exc}"

    return compat


# ---------------------------------------------------------------------------
# Import name resolution
# ---------------------------------------------------------------------------

_IMPORT_NAME_OVERRIDES: dict[str, str] = {
    "pillow": "PIL",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
    "scikit-learn": "sklearn",
    "python-dateutil": "dateutil",
    "python-dotenv": "dotenv",
    "msgpack-python": "msgpack",
    "attrs": "attr",
    "protobuf": "google.protobuf",
    "opencv-python": "cv2",
    "ruamel.yaml": "ruamel.yaml",
    "pymongo": "pymongo",
    "ujson": "ujson",
}


def guess_import_name(pypi_name: str) -> str:
    """Guess the import name from a PyPI package name.

    Checks a known override table first, then falls back to
    replacing hyphens with underscores (covers most cases).

    Args:
        pypi_name: The PyPI package name (e.g., "scikit-learn").

    Returns:
        The guessed import name (e.g., "sklearn").
    """
    lower = pypi_name.lower()
    if lower in _IMPORT_NAME_OVERRIDES:
        return _IMPORT_NAME_OVERRIDES[lower]
    return lower.replace("-", "_")


# ---------------------------------------------------------------------------
# Source code scanning for Py_mod_gil
# ---------------------------------------------------------------------------

_SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".pyx",
    ".pxd",
}

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".tox",
    ".eggs",
    ".mypy_cache",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
}

_MOD_GIL_PATTERN = re.compile(
    r"""
    Py_mod_gil              # The slot identifier
    \s*[,\s]\s*             # Separator (comma or whitespace)
    (Py_MOD_GIL_NOT_USED    # The value we're looking for
    |Py_MOD_GIL_USED)       # Or the explicit GIL-required value
    """,
    re.VERBOSE,
)

_MOD_GIL_MENTION_PATTERN = re.compile(r"Py_MOD_GIL_NOT_USED|Py_MOD_GIL_USED|Py_mod_gil")


def scan_source_for_mod_gil(
    repo_dir: Path,
    *,
    max_files: int = 5000,
    max_file_size: int = 1_000_000,
) -> SourceScanResult:
    """Scan a repository's source files for Py_mod_gil declarations.

    Walks the repository looking for C/C++/Cython files and searches
    for Py_mod_gil slot declarations. This detects whether the
    developer has opted into free-threading support at the source
    level.

    Args:
        repo_dir: Path to the cloned repository root.
        max_files: Maximum number of source files to scan (safety
            limit for huge repos).
        max_file_size: Maximum size of a single file to scan in
            bytes.

    Returns:
        SourceScanResult with all declarations found.
    """
    result = SourceScanResult()
    files_checked = 0

    for source_file in _walk_source_files(repo_dir):
        if files_checked >= max_files:
            log.debug(
                "Source scan hit file limit (%d) in %s",
                max_files,
                repo_dir,
            )
            break

        files_checked += 1

        try:
            size = source_file.stat().st_size
            if size > max_file_size:
                continue
            content = source_file.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue

        result.files_scanned += 1
        file_has_declaration = False

        for line_num, line in enumerate(content.splitlines(), 1):
            match = _MOD_GIL_PATTERN.search(line)
            if match:
                value = match.group(1)
                result.declarations.append(
                    ModGilDeclaration(
                        file=str(source_file.relative_to(repo_dir)),
                        line_number=line_num,
                        line_text=line.strip()[:200],
                        is_not_used=(value == "Py_MOD_GIL_NOT_USED"),
                    )
                )
                file_has_declaration = True

        if file_has_declaration:
            result.files_with_mod_gil += 1

    return result


def _walk_source_files(repo_dir: Path) -> Iterator[Path]:
    """Yield C/C++/Cython source files in a repo, skipping junk dirs."""
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]

        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in _SOURCE_EXTENSIONS:
                yield Path(root) / filename


# ---------------------------------------------------------------------------
# Combined assessment
# ---------------------------------------------------------------------------


def assess_extension_compat(
    package_name: str,
    *,
    venv_python: Path | None = None,
    repo_dir: Path | None = None,
    import_name: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 60,
) -> ExtensionCompat:
    """Run both runtime probe and source scan for a package.

    Either or both of venv_python (for runtime probe) and repo_dir
    (for source scan) can be provided. If both are given, results
    are combined.

    Args:
        package_name: PyPI package name.
        venv_python: Path to Python in the package's venv (for probe).
        repo_dir: Path to the cloned repo (for source scan).
        import_name: Override for the import name. If None, guessed
            from package_name.
        env: Environment variables for the probe subprocess.
        timeout: Timeout for the runtime probe.

    Returns:
        ExtensionCompat with combined results.
    """
    actual_import_name = import_name or guess_import_name(package_name)

    if venv_python is not None:
        compat = probe_gil_fallback(
            actual_import_name,
            venv_python,
            env=env,
            timeout=timeout,
        )
    else:
        compat = ExtensionCompat(package=package_name)

    if repo_dir is not None:
        try:
            scan = scan_source_for_mod_gil(repo_dir)
            compat.source_scan = scan

            if venv_python is None:
                compat.is_pure_python = not scan.has_any_declaration
                if scan.has_any_declaration:
                    compat.all_extensions_compatible = scan.all_not_used
        except Exception as exc:
            log.warning("Source scan failed for %s: %s", package_name, exc)

    return compat


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------


def format_extension_compat(compat: ExtensionCompat) -> str:
    """Format extension compatibility for terminal display."""
    lines = [f"Extension compatibility: {compat.package}"]

    if not compat.import_ok:
        lines.append(f"  Import failed: {compat.import_error}")
        return "\n".join(lines)

    if compat.is_pure_python:
        lines.append("  Pure Python (no C extensions)")
    else:
        n_ext = len(compat.extensions)
        lines.append(f"  C extensions found: {n_ext}")
        if compat.gil_fallback_active:
            lines.append("  GIL fallback: ACTIVE (not free-threading safe)")
        else:
            lines.append("  GIL fallback: not triggered")

        for ext in compat.extensions:
            status = "ok" if ext.import_ok else f"FAILED: {ext.import_error}"
            lines.append(f"    {ext.module_name}: {status}")

    if compat.source_scan is not None:
        scan = compat.source_scan
        lines.append(
            f"  Source scan: {scan.files_scanned} files, "
            f"{len(scan.declarations)} Py_mod_gil declarations"
        )
        for decl in scan.declarations:
            kind = "NOT_USED" if decl.is_not_used else "USED"
            lines.append(f"    {decl.file}:{decl.line_number} \u2192 {kind}")

        if scan.has_any_declaration and scan.all_not_used:
            lines.append("  Source declares full free-threading support")
        elif scan.has_required:
            lines.append("  Source declares GIL requirement in some modules")
        elif not scan.has_any_declaration and not compat.is_pure_python:
            lines.append(
                "  No Py_mod_gil declarations found \u2014 extension may "
                "not have been updated for free-threading"
            )

    return "\n".join(lines)
