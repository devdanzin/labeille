"""Static test dependency scanner via AST-based import extraction.

Analyzes a package's source code to discover external test dependencies,
replacing the slow trial-and-error enrichment cycle with a single upfront scan.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from labeille.import_map import IMPORT_TO_PIP
from labeille.logging import get_logger
from labeille.registry import PackageEntry

log = get_logger("scan_deps")

# Directories always excluded from scanning.
_DEFAULT_EXCLUDE_DIRS: set[str] = {
    "__pycache__",
    ".git",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "node_modules",
    "vendor",
    "_vendor",
    "vendored",
    ".eggs",
    "*.egg-info",
    "build",
    "dist",
}

# Common test infrastructure module names that are local, not pip packages.
_TEST_INFRA_NAMES: set[str] = {
    "conftest",
    "fixtures",
    "helpers",
    "utils",
    "common",
    "base",
    "compat",
    "support",
    "testutils",
    "testing",
    "shared",
    "test_helpers",
    "test_utils",
    "test_support",
    "test_common",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ImportInfo:
    """A single import found in source code."""

    module: str  # top-level module name (e.g. 'pytest' from 'import pytest.mark')
    full_name: str  # complete import path (e.g. 'pytest.mark')
    source_file: str  # relative path where the import was found
    line_number: int
    is_conditional: bool  # True if inside try/except ImportError block


@dataclass
class ResolvedDep:
    """A resolved external dependency."""

    import_name: str  # what the code imports (e.g. 'yaml')
    pip_package: str  # what to pip install (e.g. 'PyYAML')
    source: str  # how resolved: 'identity', 'mapping', 'registry', 'unresolved'
    import_files: list[str]  # which files import it
    is_conditional: bool  # True if ALL occurrences are conditional
    note: str = ""  # warning for namespace packages or uncertain resolution


@dataclass
class ScanResult:
    """Complete result of a dependency scan."""

    package_name: str
    scan_dirs: list[str]
    total_files_scanned: int
    total_imports_found: int
    resolved: list[ResolvedDep]
    unresolved: list[ResolvedDep]
    already_installed: list[str]
    missing: list[str]
    suggested_install: str


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


def _is_import_error_handler(handler: ast.ExceptHandler) -> bool:
    """Return True if an except handler catches ImportError or ModuleNotFoundError."""
    if handler.type is None:
        # bare except: — treat as catching everything including ImportError
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in ("ImportError", "ModuleNotFoundError")
    if isinstance(handler.type, ast.Tuple):
        for elt in handler.type.elts:
            if isinstance(elt, ast.Name) and elt.id in (
                "ImportError",
                "ModuleNotFoundError",
            ):
                return True
    return False


def _find_conditional_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    """Find line ranges of try/except blocks that catch import errors.

    Returns a list of (start_line, end_line) tuples for the ``try`` bodies
    of try/except blocks where at least one handler catches ImportError
    or ModuleNotFoundError.
    """
    ranges: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            has_import_handler = any(_is_import_error_handler(h) for h in node.handlers)
            if has_import_handler and node.body:
                start = node.body[0].lineno
                end = node.body[-1].end_lineno or node.body[-1].lineno
                ranges.append((start, end))
    return ranges


def _in_conditional_range(line: int, ranges: list[tuple[int, int]]) -> bool:
    """Check if a line number falls within any conditional import range."""
    return any(start <= line <= end for start, end in ranges)


def extract_imports(file_path: Path, *, repo_root: Path | None = None) -> list[ImportInfo]:
    """Extract all imports from a single Python file using AST parsing.

    Handles:
    - ``import X`` -> module = 'X'
    - ``import X.Y.Z`` -> module = 'X'
    - ``from X import Y`` -> module = 'X'
    - ``from X.Y import Z`` -> module = 'X'
    - ``from . import X`` -> skip (relative import)
    - ``from .foo import X`` -> skip (relative import)

    Detects conditional imports: any import inside a try/except block
    where the except catches ImportError or ModuleNotFoundError is
    marked as conditional.

    If the file has a syntax error, logs a warning and returns an empty
    list.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("Could not read %s: %s", file_path, exc)
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        log.warning("Syntax error in %s, skipping", file_path)
        return []

    # Compute relative path for reporting.
    if repo_root is not None:
        try:
            rel_path = str(file_path.relative_to(repo_root))
        except ValueError:
            rel_path = str(file_path)
    else:
        rel_path = str(file_path)

    conditional_ranges = _find_conditional_ranges(tree)
    imports: list[ImportInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                is_cond = _in_conditional_range(node.lineno, conditional_ranges)
                imports.append(
                    ImportInfo(
                        module=top_level,
                        full_name=alias.name,
                        source_file=rel_path,
                        line_number=node.lineno,
                        is_conditional=is_cond,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports.
            if node.level and node.level > 0:
                continue
            if node.module is None:
                continue
            top_level = node.module.split(".")[0]
            is_cond = _in_conditional_range(node.lineno, conditional_ranges)
            imports.append(
                ImportInfo(
                    module=top_level,
                    full_name=node.module,
                    source_file=rel_path,
                    line_number=node.lineno,
                    is_conditional=is_cond,
                )
            )

    return imports


def _should_exclude_dir(dir_name: str, exclude_patterns: set[str]) -> bool:
    """Check if a directory name matches any exclusion pattern."""
    for pattern in exclude_patterns:
        if fnmatch(dir_name, pattern):
            return True
    return False


def extract_imports_from_directory(
    directory: Path,
    *,
    recursive: bool = True,
    exclude_patterns: list[str] | None = None,
    repo_root: Path | None = None,
) -> list[ImportInfo]:
    """Extract imports from all .py files in a directory tree.

    Args:
        directory: Root directory to scan.
        recursive: Whether to recurse into subdirectories.
        exclude_patterns: Glob patterns to skip. Default exclusions apply
            in addition to any provided patterns.
        repo_root: Repository root for computing relative paths.

    Returns:
        All imports found across all scanned files.
    """
    excludes = set(_DEFAULT_EXCLUDE_DIRS)
    if exclude_patterns:
        excludes.update(exclude_patterns)

    all_imports: list[ImportInfo] = []

    if not directory.is_dir():
        return all_imports

    if recursive:
        for py_file in sorted(directory.rglob("*.py")):
            # Check if any parent directory should be excluded.
            skip = False
            for part in py_file.relative_to(directory).parts[:-1]:
                if _should_exclude_dir(part, excludes):
                    skip = True
                    break
            if skip:
                continue
            all_imports.extend(extract_imports(py_file, repo_root=repo_root))
    else:
        for py_file in sorted(directory.glob("*.py")):
            all_imports.extend(extract_imports(py_file, repo_root=repo_root))

    return all_imports


# ---------------------------------------------------------------------------
# Import filtering
# ---------------------------------------------------------------------------


def get_stdlib_modules() -> set[str]:
    """Return the set of standard library module names.

    Uses ``sys.stdlib_module_names`` (Python 3.10+). Also includes
    well-known stdlib top-level names that might not appear in
    ``stdlib_module_names``.
    """
    names: set[str] = set()
    if hasattr(sys, "stdlib_module_names"):
        names.update(sys.stdlib_module_names)

    # Extra well-known stdlib names.
    names.update(
        {
            "__future__",
            "_thread",
            "_io",
            "_socket",
            "_ssl",
            "_collections",
            "_collections_abc",
            "_operator",
            "_abc",
            "_functools",
            "_weakref",
            "_signal",
            "_stat",
            "_string",
            "posixpath",
            "ntpath",
            "genericpath",
            "sre_compile",
            "sre_parse",
            "sre_constants",
            "encodings",
            # Common C extension modules in stdlib.
            "_ctypes",
            "_decimal",
            "_csv",
            "_json",
            "_pickle",
            "_struct",
            "_hashlib",
            "_lzma",
            "_bz2",
            "importlib",
            "test",
        }
    )
    return names


def get_local_modules(repo_root: Path, package_name: str) -> set[str]:
    """Identify local/internal modules that shouldn't be treated as external deps.

    Includes:
    - The package's own import name (and normalized variants).
    - Any top-level Python packages/modules found in ``src/`` or at repo root.
    - Common test infrastructure module names.
    - Any directory in the test directory that contains ``__init__.py``.
    """
    local: set[str] = set()

    # Package name and normalized variants.
    local.add(package_name)
    local.add(package_name.replace("-", "_"))
    local.add(package_name.replace("_", "-"))
    local.add(package_name.lower())
    local.add(package_name.lower().replace("-", "_"))

    # Scan src/ layout.
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        for child in src_dir.iterdir():
            if child.is_dir() and (child / "__init__.py").exists():
                local.add(child.name)
            elif child.is_file() and child.suffix == ".py" and child.name != "__init__.py":
                local.add(child.stem)

    # Scan repo root for top-level packages.
    for child in repo_root.iterdir():
        if child.is_dir() and (child / "__init__.py").exists():
            # Skip test directories — they're not the package itself.
            if child.name not in ("tests", "test", "testing"):
                local.add(child.name)
        elif child.is_file() and child.suffix == ".py" and child.name != "setup.py":
            local.add(child.stem)

    # Common test infrastructure names.
    local.update(_TEST_INFRA_NAMES)

    # Test directories with __init__.py are local packages.
    for test_dir_name in ("tests", "test", "testing"):
        test_dir = repo_root / test_dir_name
        if test_dir.is_dir():
            local.add(test_dir_name)
            for child in test_dir.iterdir():
                if child.is_dir() and (child / "__init__.py").exists():
                    local.add(child.name)

    return local


def filter_imports(
    imports: list[ImportInfo],
    *,
    stdlib: set[str],
    local_modules: set[str],
    include_conditional: bool = True,
) -> list[ImportInfo]:
    """Filter out stdlib and local imports, keeping only external deps.

    If ``include_conditional`` is False, also filters out imports inside
    try/except ImportError blocks.
    """
    result: list[ImportInfo] = []
    for imp in imports:
        if imp.module in stdlib:
            continue
        if imp.module in local_modules:
            continue
        if not include_conditional and imp.is_conditional:
            continue
        result.append(imp)
    return result


# ---------------------------------------------------------------------------
# Import-to-pip resolution
# ---------------------------------------------------------------------------


# Top-level import names that are namespace packages — the actual pip
# package depends on which sub-package is imported.  Resolution for
# these is uncertain; users should verify the specific package needed.
_NAMESPACE_PACKAGES: set[str] = {
    "google",
    "azure",
    "zope",
    "jaraco",
    "sphinxcontrib",
    "backports",
    "repoze",
}


def _is_plausible_pip_name(name: str) -> bool:
    """Check if a name looks like a plausible pip package for identity mapping.

    Filters out:
    - Names starting with ``_`` (private/internal)
    - Single character names (type variables)
    - All uppercase names (likely constants)
    - Common test infrastructure patterns
    - Names under 3 characters
    """
    if name.startswith("_"):
        return False
    if len(name) < 3:
        return False
    if name.isupper():
        return False
    if name in _TEST_INFRA_NAMES:
        return False
    return True


def resolve_imports(
    imports: list[ImportInfo],
    *,
    import_map: dict[str, str] | None = None,
    registry_entries: list[PackageEntry] | None = None,
) -> tuple[list[ResolvedDep], list[ResolvedDep]]:
    """Map import names to pip package names.

    Resolution order:
    1. Static mapping table (IMPORT_TO_PIP from import_map.py)
    2. Registry cross-reference (check import_name fields)
    3. Identity mapping (assume import_name == pip_name)
    4. If none of the above: mark as unresolved

    Deduplicates by import_name: if the same module is imported in
    multiple files, produces one ResolvedDep with all source files listed.
    Marks ``is_conditional=True`` only if ALL occurrences are conditional.

    Returns:
        Tuple of (resolved, unresolved).
    """
    if import_map is None:
        import_map = IMPORT_TO_PIP

    # Build registry lookup: import_name -> package name.
    registry_lookup: dict[str, str] = {}
    if registry_entries:
        for entry in registry_entries:
            if entry.import_name:
                registry_lookup[entry.import_name] = entry.package
                registry_lookup[entry.import_name.replace("-", "_")] = entry.package

    # Group imports by module name.
    grouped: dict[str, list[ImportInfo]] = {}
    for imp in imports:
        grouped.setdefault(imp.module, []).append(imp)

    resolved: list[ResolvedDep] = []
    unresolved: list[ResolvedDep] = []

    for module_name, module_imports in sorted(grouped.items()):
        # Normalize for lookup.
        normalized = module_name.replace("-", "_")
        import_files = sorted(set(imp.source_file for imp in module_imports))
        all_conditional = all(imp.is_conditional for imp in module_imports)

        # Collect unique full import paths (for namespace package sub-path lookup).
        full_names = sorted(set(imp.full_name for imp in module_imports))

        # 1a. Try full import paths in the static mapping (namespace sub-packages).
        pip_name: str | None = None
        for fn in full_names:
            pip_name = import_map.get(fn) or import_map.get(fn.replace("-", "_"))
            if pip_name:
                break

        # 1b. Fall back to top-level in static mapping.
        if not pip_name:
            pip_name = import_map.get(module_name) or import_map.get(normalized)

        if pip_name:
            note = ""
            if module_name in _NAMESPACE_PACKAGES:
                sub_pkgs = sorted(
                    set(imp.full_name for imp in module_imports if "." in imp.full_name)
                )
                if sub_pkgs:
                    note = (
                        f"namespace package \u2014 verify correct pip package "
                        f"(imports: {', '.join(sub_pkgs[:5])})"
                    )
            resolved.append(
                ResolvedDep(
                    import_name=module_name,
                    pip_package=pip_name,
                    source="mapping",
                    import_files=import_files,
                    is_conditional=all_conditional,
                    note=note,
                )
            )
            continue

        # 2. Registry cross-reference.
        pip_name = registry_lookup.get(module_name) or registry_lookup.get(normalized)
        if pip_name:
            resolved.append(
                ResolvedDep(
                    import_name=module_name,
                    pip_package=pip_name,
                    source="registry",
                    import_files=import_files,
                    is_conditional=all_conditional,
                )
            )
            continue

        # 3. Identity mapping (with validation).
        if _is_plausible_pip_name(module_name):
            note = ""
            if module_name in _NAMESPACE_PACKAGES:
                sub_pkgs = sorted(
                    set(imp.full_name for imp in module_imports if "." in imp.full_name)
                )
                note = (
                    f"namespace package \u2014 cannot determine pip package "
                    f"from top-level import (imports: {', '.join(sub_pkgs[:5])})"
                )
            resolved.append(
                ResolvedDep(
                    import_name=module_name,
                    pip_package=module_name,
                    source="identity",
                    import_files=import_files,
                    is_conditional=all_conditional,
                    note=note,
                )
            )
            continue

        # 4. Unresolved.
        unresolved.append(
            ResolvedDep(
                import_name=module_name,
                pip_package="",
                source="unresolved",
                import_files=import_files,
                is_conditional=all_conditional,
            )
        )

    return resolved, unresolved


# ---------------------------------------------------------------------------
# Install command comparison
# ---------------------------------------------------------------------------


# Extracts the base package name from a pip requirement string, stopping at
# the first version specifier, extras bracket, or environment marker.
_PIP_PKG_RE = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _normalize_pip_command(cmd: str) -> str | None:
    """Extract the part after ``pip install`` from various pip invocations.

    Handles ``pip install``, ``pip3 install``, ``python -m pip install``,
    ``python3 -m pip install``, and path-qualified pip (e.g.
    ``/tmp/venv/bin/pip install``).

    Returns the arguments string (everything after ``pip install``), or
    ``None`` if this isn't a pip install command.
    """
    for prefix in (
        "python -m pip install",
        "python3 -m pip install",
        "pip3 install",
        "pip install",
    ):
        if prefix in cmd:
            idx = cmd.index(prefix) + len(prefix)
            return cmd[idx:]
    return None


def _parse_install_packages(install_command: str) -> tuple[list[str], list[str]]:
    """Parse package names from an install command string.

    Returns:
        Tuple of (explicit_packages, extras_notes).
        explicit_packages: pip package names found in the command.
        extras_notes: notes about extras like ``[test]`` whose contents are unknown.
    """
    packages: list[str] = []
    extras: list[str] = []

    # Split on && to handle chained commands.
    for cmd in install_command.split("&&"):
        cmd = cmd.strip()
        args = _normalize_pip_command(cmd)
        if args is None:
            continue

        parts = ["pip", "install"] + args.split()
        skip_next = False
        for i, part in enumerate(parts):
            if skip_next:
                skip_next = False
                continue
            # Skip 'pip', 'install', and flags.
            if part in ("pip", "pip3", "install"):
                continue
            if part.startswith("-"):
                # Flags that take a value.
                if part in ("-e", "-r", "--requirement", "--editable", "--index-url", "-i"):
                    skip_next = True
                    # Check for extras on editable installs.
                    if part == "-e" and i + 1 < len(parts):
                        next_part = parts[i + 1]
                        if "[" in next_part:
                            extra = next_part[next_part.index("[") : next_part.index("]") + 1]
                            extras.append(f"{extra} (contents unknown)")
                continue
            # Check for extras syntax.
            if part.startswith(".") or part.startswith("'") or part.startswith('"'):
                cleaned = part.strip("'\"")
                if "[" in cleaned:
                    extra = cleaned[cleaned.index("[") : cleaned.index("]") + 1]
                    extras.append(f"{extra} (contents unknown)")
                continue
            # Extract base package name, ignoring version specifiers,
            # extras, and environment markers.
            base = part.split("[")[0].strip("'\"")
            match = _PIP_PKG_RE.match(base)
            if match:
                pkg_name = match.group(1)
                if pkg_name and not pkg_name.startswith("-"):
                    packages.append(pkg_name)

    return packages, extras


def compare_with_install_command(
    resolved: list[ResolvedDep],
    install_command: str,
) -> tuple[list[str], list[str]]:
    """Compare resolved deps against an existing install_command string.

    Returns:
        Tuple of (already_present, missing).
    """
    installed_pkgs, _ = _parse_install_packages(install_command)
    # Normalize for comparison: lowercase and replace hyphens/underscores.
    installed_normalized = {
        pkg.lower().replace("-", "_").replace(".", "_") for pkg in installed_pkgs
    }

    already_present: list[str] = []
    missing: list[str] = []

    for dep in resolved:
        dep_normalized = dep.pip_package.lower().replace("-", "_").replace(".", "_")
        if dep_normalized in installed_normalized:
            already_present.append(dep.pip_package)
        else:
            missing.append(dep.pip_package)

    return already_present, missing


# ---------------------------------------------------------------------------
# Full scan
# ---------------------------------------------------------------------------


def _auto_detect_test_dirs(repo_root: Path) -> list[Path]:
    """Auto-detect test directories by looking for common names."""
    candidates = ["tests", "test", "testing", "Tests"]
    found: list[Path] = []
    for name in candidates:
        d = repo_root / name
        if d.is_dir():
            found.append(d)
    return found


def build_scan_result(
    package_name: str,
    scan_dirs: list[str],
    total_files: int,
    imports: list[ImportInfo],
    resolved: list[ResolvedDep],
    unresolved: list[ResolvedDep],
    install_command: str | None = None,
) -> ScanResult:
    """Assemble the complete scan result with install comparison."""
    already_installed: list[str] = []
    missing: list[str] = []

    if install_command:
        already_installed, missing = compare_with_install_command(resolved, install_command)
    else:
        missing = [dep.pip_package for dep in resolved]

    suggested = ""
    if missing:
        suggested = "pip install " + " ".join(sorted(missing))

    return ScanResult(
        package_name=package_name,
        scan_dirs=scan_dirs,
        total_files_scanned=total_files,
        total_imports_found=len(imports),
        resolved=resolved,
        unresolved=unresolved,
        already_installed=already_installed,
        missing=missing,
        suggested_install=suggested,
    )


def scan_package_deps(
    repo_root: Path,
    package_name: str,
    *,
    test_dirs: list[str] | None = None,
    scan_source: bool = False,
    install_command: str | None = None,
    registry_entries: list[PackageEntry] | None = None,
    include_conditional: bool = True,
) -> ScanResult:
    """Scan a package repository for external test dependencies.

    Args:
        repo_root: Path to the cloned repository.
        package_name: The PyPI package name (for filtering local imports).
        test_dirs: Directories to scan (default: auto-detect).
        scan_source: Also scan the package source code (not just tests).
        install_command: Current install_command to compare against.
        registry_entries: Registry entries for cross-referencing import names.
        include_conditional: Include try/except ImportError imports.

    Returns:
        A ScanResult with resolved/unresolved deps and install comparison.
    """
    # Determine directories to scan.
    dirs_to_scan: list[Path] = []
    if test_dirs:
        for td in test_dirs:
            d = repo_root / td
            if d.is_dir():
                dirs_to_scan.append(d)
            else:
                log.warning("Test directory %s does not exist, skipping", d)
    else:
        dirs_to_scan = _auto_detect_test_dirs(repo_root)

    if scan_source:
        src = repo_root / "src"
        if src.is_dir():
            dirs_to_scan.append(src)
        else:
            dirs_to_scan.append(repo_root)

    # Fall back to repo root if no test dirs found.
    if not dirs_to_scan:
        log.info("No test directories found, scanning repo root")
        dirs_to_scan = [repo_root]

    scan_dir_names = [
        str(d.relative_to(repo_root)) if d != repo_root else "." for d in dirs_to_scan
    ]

    # Extract imports from all directories.
    all_imports: list[ImportInfo] = []
    files_scanned: set[str] = set()
    for scan_dir in dirs_to_scan:
        dir_imports = extract_imports_from_directory(scan_dir, repo_root=repo_root)
        all_imports.extend(dir_imports)
        files_scanned.update(imp.source_file for imp in dir_imports)

    # Filter imports.
    stdlib = get_stdlib_modules()
    local = get_local_modules(repo_root, package_name)
    filtered = filter_imports(
        all_imports,
        stdlib=stdlib,
        local_modules=local,
        include_conditional=include_conditional,
    )

    # Resolve imports to pip packages.
    resolved, unresolved = resolve_imports(
        filtered,
        registry_entries=registry_entries,
    )

    return build_scan_result(
        package_name=package_name,
        scan_dirs=scan_dir_names,
        total_files=len(files_scanned),
        imports=filtered,
        resolved=resolved,
        unresolved=unresolved,
        install_command=install_command,
    )
