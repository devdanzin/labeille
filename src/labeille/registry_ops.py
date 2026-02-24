"""Batch operations for the package registry.

Provides higher-level functions that orchestrate line-level YAML helpers
with file I/O, filtering, error handling, and reporting.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from labeille.logging import get_logger
from labeille.registry import (
    IndexEntry,
    load_index,
    load_package,
    save_index,
    sort_index,
    update_index_from_packages,
)
from labeille.yaml_lines import (
    format_yaml_value,
    has_field,
    insert_field_after,
    parse_default_value,
    remove_field as remove_field_lines,
    rename_field as rename_field_lines,
    set_field_value,
)

log = get_logger("registry_ops")

# Fields that cannot be removed from package YAML files.
PROTECTED_FIELDS = frozenset({"package", "repo", "pypi_url", "enriched"})

# Fields that cannot be removed from the index.
PROTECTED_INDEX_FIELDS = frozenset({"name", "skip"})


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class OpResult:
    """Result of a batch operation."""

    modified: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total: int = 0


@dataclass
class DryRunPreview:
    """Preview of what a batch operation would do."""

    affected_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    sample_diffs: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class ValidationIssue:
    """A single validation issue found in a package file."""

    filename: str
    level: str  # "error" or "warning"
    message: str


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@dataclass
class FieldFilter:
    """A single filter predicate for package selection."""

    field: str
    op: str  # "=", "~=", ":true", ":false", ":null", ":notnull"
    value: str | None


def parse_where(expr: str) -> FieldFilter:
    """Parse a ``--where`` expression into a :class:`FieldFilter`.

    Supported forms:
    - ``field=value`` — exact string match
    - ``field~=pattern`` — substring match
    - ``field:true`` / ``field:false`` — boolean match
    - ``field:null`` / ``field:notnull`` — presence/absence check
    """
    # Check for :true/:false/:null/:notnull first
    for suffix in (":true", ":false", ":null", ":notnull"):
        if expr.endswith(suffix):
            field_name = expr[: -len(suffix)]
            if not field_name:
                raise ValueError(f"Invalid --where expression: {expr!r}")
            return FieldFilter(field=field_name, op=suffix, value=None)

    # Check for ~= (substring)
    if "~=" in expr:
        field_name, _, pattern = expr.partition("~=")
        if not field_name:
            raise ValueError(f"Invalid --where expression: {expr!r}")
        return FieldFilter(field=field_name, op="~=", value=pattern)

    # Check for = (exact match)
    if "=" in expr:
        field_name, _, value = expr.partition("=")
        if not field_name:
            raise ValueError(f"Invalid --where expression: {expr!r}")
        return FieldFilter(field=field_name, op="=", value=value)

    raise ValueError(f"Invalid --where expression: {expr!r}")


def matches(entry: dict[str, Any], filters: list[FieldFilter]) -> bool:
    """Test whether a parsed YAML dict matches all filters (AND logic)."""
    for f in filters:
        val = entry.get(f.field)

        if f.op == ":null":
            if val is not None:
                return False
        elif f.op == ":notnull":
            if val is None:
                return False
        elif f.op == ":true":
            if not bool(val):
                return False
        elif f.op == ":false":
            if bool(val):
                return False
        elif f.op == "=":
            if str(val) != f.value:
                return False
        elif f.op == "~=":
            if f.value is None or f.value not in str(val):
                return False

    return True


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def _list_package_files(registry_dir: Path) -> list[Path]:
    """List all package YAML files in the registry, sorted by name."""
    packages_dir = registry_dir / "packages"
    if not packages_dir.is_dir():
        return []
    return sorted(packages_dir.glob("*.yaml"))


def _read_lines(path: Path) -> list[str]:
    """Read a file as a list of lines (preserving newlines)."""
    text = path.read_text(encoding="utf-8")
    # splitlines(True) keeps line endings
    return text.splitlines(True)


def _atomic_write(path: Path, lines: list[str]) -> None:
    """Write lines to a file atomically using a temp file and os.replace."""
    content = "".join(lines)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _filter_files(
    files: list[Path],
    filters: list[FieldFilter],
    packages_list: list[str] | None,
) -> list[Path]:
    """Filter package files by --where and --packages criteria."""
    result = files

    if packages_list is not None:
        name_set = set(packages_list)
        result = [f for f in result if f.stem in name_set]

    if filters:
        filtered = []
        for f in result:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and matches(data, filters):
                filtered.append(f)
        result = filtered

    return result


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


def batch_add_field(
    registry_dir: Path,
    field_name: str,
    field_type: str = "str",
    default: str | None = None,
    after: str | None = None,
    filters: list[FieldFilter] | None = None,
    packages_list: list[str] | None = None,
    dry_run: bool = True,
    lenient: bool = False,
) -> OpResult | DryRunPreview:
    """Add a new field to package YAML files.

    Args:
        registry_dir: Path to the registry directory.
        field_name: The YAML key to add.
        field_type: Field type (str, int, bool, list, dict).
        default: Default value string, or None for type default.
        after: Insert after this existing field. If None, append before last field.
        filters: Optional field filters.
        packages_list: Optional list of package names to operate on.
        dry_run: If True, return a preview without writing.
        lenient: If True, skip files that already have the field.

    Returns:
        A :class:`DryRunPreview` if *dry_run*, else an :class:`OpResult`.
    """
    files = _list_package_files(registry_dir)
    files = _filter_files(files, filters or [], packages_list)

    parsed_default = parse_default_value(default, field_type)
    value_text = format_yaml_value(parsed_default, field_type)

    modified: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    sample_diffs: list[tuple[str, str, str]] = []

    for f in files:
        lines = _read_lines(f)
        if has_field(lines, field_name):
            if lenient:
                skipped.append(f.name)
                continue
            else:
                errors.append(f.name)
                continue

        try:
            if after is not None:
                new_lines = insert_field_after(lines, after, field_name, value_text)
            else:
                # Append before the last field
                new_lines = _append_field(lines, field_name, value_text)
        except ValueError as e:
            errors.append(f"{f.name}: {e}")
            continue

        if dry_run:
            if len(sample_diffs) < 5:
                before_text = "".join(lines)
                after_text = "".join(new_lines)
                sample_diffs.append((f.name, before_text, after_text))
            modified.append(f.name)
        else:
            _atomic_write(f, new_lines)
            modified.append(f.name)

    if dry_run:
        return DryRunPreview(
            affected_count=len(modified),
            skipped_count=len(skipped),
            error_count=len(errors),
            sample_diffs=sample_diffs,
        )

    return OpResult(modified=modified, skipped=skipped, errors=errors, total=len(files))


def _append_field(lines: list[str], field_name: str, value_text: str) -> list[str]:
    """Append a field before the last top-level field in the YAML."""
    # Find the last top-level field
    last_field_idx = None
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and ":" in line and not line.startswith("#"):
            last_field_idx = i
    if last_field_idx is None:
        # No fields found, just append
        new_line = f"{field_name}: {value_text}\n"
        return lines + [new_line]

    new_line = f"{field_name}: {value_text}\n"
    return lines[:last_field_idx] + [new_line] + lines[last_field_idx:]


def batch_remove_field(
    registry_dir: Path,
    field_name: str,
    filters: list[FieldFilter] | None = None,
    packages_list: list[str] | None = None,
    dry_run: bool = True,
    lenient: bool = False,
) -> OpResult | DryRunPreview:
    """Remove a field from package YAML files.

    Args:
        registry_dir: Path to the registry directory.
        field_name: The YAML key to remove.
        filters: Optional field filters.
        packages_list: Optional list of package names to operate on.
        dry_run: If True, return a preview without writing.
        lenient: If True, skip files that don't have the field.

    Returns:
        A :class:`DryRunPreview` if *dry_run*, else an :class:`OpResult`.

    Raises:
        ValueError: If *field_name* is a protected field.
    """
    if field_name in PROTECTED_FIELDS:
        raise ValueError(f"Cannot remove protected field '{field_name}'")

    files = _list_package_files(registry_dir)
    files = _filter_files(files, filters or [], packages_list)

    modified: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    sample_diffs: list[tuple[str, str, str]] = []

    for f in files:
        lines = _read_lines(f)
        if not has_field(lines, field_name):
            if lenient:
                skipped.append(f.name)
                continue
            else:
                errors.append(f.name)
                continue

        new_lines = remove_field_lines(lines, field_name)

        if dry_run:
            if len(sample_diffs) < 5:
                sample_diffs.append((f.name, "".join(lines), "".join(new_lines)))
            modified.append(f.name)
        else:
            _atomic_write(f, new_lines)
            modified.append(f.name)

    if dry_run:
        return DryRunPreview(
            affected_count=len(modified),
            skipped_count=len(skipped),
            error_count=len(errors),
            sample_diffs=sample_diffs,
        )

    return OpResult(modified=modified, skipped=skipped, errors=errors, total=len(files))


def batch_rename_field(
    registry_dir: Path,
    old_name: str,
    new_name: str,
    filters: list[FieldFilter] | None = None,
    packages_list: list[str] | None = None,
    dry_run: bool = True,
    lenient: bool = False,
) -> OpResult | DryRunPreview:
    """Rename a field in package YAML files.

    Args:
        registry_dir: Path to the registry directory.
        old_name: The current field name.
        new_name: The new field name.
        filters: Optional field filters.
        packages_list: Optional list of package names to operate on.
        dry_run: If True, return a preview without writing.
        lenient: If True, skip files where preconditions aren't met.

    Returns:
        A :class:`DryRunPreview` if *dry_run*, else an :class:`OpResult`.
    """
    files = _list_package_files(registry_dir)
    files = _filter_files(files, filters or [], packages_list)

    modified: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    sample_diffs: list[tuple[str, str, str]] = []

    for f in files:
        lines = _read_lines(f)
        has_old = has_field(lines, old_name)
        has_new = has_field(lines, new_name)

        if not has_old or has_new:
            if lenient:
                skipped.append(f.name)
                continue
            else:
                if not has_old:
                    errors.append(f"{f.name}: field '{old_name}' not found")
                if has_new:
                    errors.append(f"{f.name}: field '{new_name}' already exists")
                continue

        new_lines = rename_field_lines(lines, old_name, new_name)

        if dry_run:
            if len(sample_diffs) < 5:
                sample_diffs.append((f.name, "".join(lines), "".join(new_lines)))
            modified.append(f.name)
        else:
            _atomic_write(f, new_lines)
            modified.append(f.name)

    if dry_run:
        return DryRunPreview(
            affected_count=len(modified),
            skipped_count=len(skipped),
            error_count=len(errors),
            sample_diffs=sample_diffs,
        )

    return OpResult(modified=modified, skipped=skipped, errors=errors, total=len(files))


def batch_set_field(
    registry_dir: Path,
    field_name: str,
    value_str: str,
    field_type: str | None = None,
    filters: list[FieldFilter] | None = None,
    packages_list: list[str] | None = None,
    require_all: bool = False,
    dry_run: bool = True,
) -> OpResult | DryRunPreview:
    """Set a field value on matching packages.

    Uses line-level manipulation to preserve YAML formatting.  PyYAML is
    used only for parsing (filter matching and type detection), never for
    serialisation.

    Args:
        registry_dir: Path to the registry directory.
        field_name: The YAML key to set.
        value_str: The value as a string (auto-parsed based on existing type).
        field_type: Explicit type override (str, int, bool, list, dict).
        filters: Optional field filters (required if *require_all* is False).
        packages_list: Optional list of package names to operate on.
        require_all: Whether ``--all`` was specified (required without filters).
        dry_run: If True, return a preview without writing.

    Returns:
        A :class:`DryRunPreview` if *dry_run*, else an :class:`OpResult`.
    """
    files = _list_package_files(registry_dir)
    if packages_list is not None:
        name_set = set(packages_list)
        files = [f for f in files if f.stem in name_set]

    modified: list[str] = []
    errors: list[str] = []
    sample_diffs: list[tuple[str, str, str]] = []

    for f in files:
        lines = _read_lines(f)
        # Parse for filtering and type detection only.
        raw = yaml.safe_load("".join(lines))
        if not isinstance(raw, dict):
            errors.append(f"{f.name}: invalid YAML")
            continue

        if filters and not matches(raw, filters):
            continue

        if field_name not in raw:
            errors.append(f"{f.name}: field '{field_name}' not found (use add-field first)")
            continue

        # Determine type from existing value or explicit --type.
        if field_type is None:
            existing = raw[field_name]
            if isinstance(existing, bool):
                field_type_resolved = "bool"
            elif isinstance(existing, int):
                field_type_resolved = "int"
            elif isinstance(existing, list):
                field_type_resolved = "list"
            elif isinstance(existing, dict):
                field_type_resolved = "dict"
            else:
                field_type_resolved = "str"
        else:
            field_type_resolved = field_type

        parsed_value = parse_default_value(value_str, field_type_resolved)

        old_value = raw[field_name]
        if old_value == parsed_value:
            continue  # No change needed.

        formatted = format_yaml_value(parsed_value, field_type_resolved)

        try:
            new_lines = set_field_value(lines, field_name, formatted)
        except ValueError as e:
            errors.append(f"{f.name}: {e}")
            continue

        if dry_run:
            if len(sample_diffs) < 5:
                sample_diffs.append((f.name, "".join(lines), "".join(new_lines)))
            modified.append(f.name)
        else:
            _atomic_write(f, new_lines)
            modified.append(f.name)

    if dry_run:
        return DryRunPreview(
            affected_count=len(modified),
            skipped_count=0,
            error_count=len(errors),
            sample_diffs=sample_diffs,
        )

    return OpResult(modified=modified, skipped=[], errors=errors, total=len(files))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Required fields that must be present.
_REQUIRED_FIELDS = {"package", "enriched"}

# Known PackageEntry fields (synced with the dataclass).
_KNOWN_FIELDS = {
    "package",
    "repo",
    "pypi_url",
    "extension_type",
    "python_versions",
    "install_method",
    "install_command",
    "test_command",
    "test_framework",
    "uses_xdist",
    "timeout",
    "skip",
    "skip_reason",
    "skip_versions",
    "notes",
    "enriched",
    "clone_depth",
    "import_name",
}

# Expected types for known fields.
_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "package": str,
    "repo": (str, type(None)),
    "pypi_url": str,
    "extension_type": str,
    "python_versions": list,
    "install_method": str,
    "install_command": str,
    "test_command": str,
    "test_framework": str,
    "uses_xdist": bool,
    "timeout": (int, type(None)),
    "skip": bool,
    "skip_reason": (str, type(None)),
    "skip_versions": dict,
    "notes": str,
    "enriched": bool,
    "clone_depth": (int, type(None)),
    "import_name": (str, type(None)),
}


def validate_registry(
    registry_dir: Path,
    strict: bool = False,
    filters: list[FieldFilter] | None = None,
    packages_list: list[str] | None = None,
) -> list[ValidationIssue]:
    """Validate package YAML files against the PackageEntry schema.

    Args:
        registry_dir: Path to the registry directory.
        strict: If True, warnings are promoted to errors.
        filters: Optional field filters.
        packages_list: Optional list of package names to validate.

    Returns:
        A list of :class:`ValidationIssue` instances.
    """
    files = _list_package_files(registry_dir)
    files = _filter_files(files, filters or [], packages_list)

    issues: list[ValidationIssue] = []

    for f in files:
        raw = yaml.safe_load(f.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            issues.append(ValidationIssue(f.name, "error", "file does not contain a YAML mapping"))
            continue

        # Check required fields
        for req in _REQUIRED_FIELDS:
            if req not in raw:
                issues.append(ValidationIssue(f.name, "error", f"missing required field '{req}'"))

        # Check unknown fields
        for key in raw:
            if key not in _KNOWN_FIELDS:
                level = "error" if strict else "warning"
                issues.append(ValidationIssue(f.name, level, f"unknown field '{key}'"))

        # Check type mismatches
        for key, expected in _FIELD_TYPES.items():
            if key in raw and raw[key] is not None:
                if not isinstance(raw[key], expected):
                    issues.append(
                        ValidationIssue(
                            f.name,
                            "error",
                            f"field '{key}' has type {type(raw[key]).__name__},"
                            f" expected {_type_name(expected)}",
                        )
                    )

        # Check skip_versions float keys
        sv = raw.get("skip_versions")
        if isinstance(sv, dict):
            for k in sv:
                if isinstance(k, float):
                    issues.append(
                        ValidationIssue(
                            f.name,
                            "warning",
                            f"skip_versions key '{k}' loaded as float",
                        )
                    )

        # Check non-skipped packages have commands
        if not raw.get("skip", False):
            if raw.get("enriched", False):
                if not raw.get("install_command"):
                    issues.append(
                        ValidationIssue(
                            f.name, "warning", "enriched package has empty install_command"
                        )
                    )
                if not raw.get("test_command"):
                    issues.append(
                        ValidationIssue(
                            f.name, "warning", "enriched package has empty test_command"
                        )
                    )

        # Check uses_xdist consistency.
        if not raw.get("skip", False):
            test_cmd = raw.get("test_command", "")
            if raw.get("uses_xdist", False):
                if test_cmd and "-p no:xdist" not in test_cmd:
                    issues.append(
                        ValidationIssue(
                            f.name,
                            "warning",
                            "uses_xdist is true but test_command does not include '-p no:xdist'",
                        )
                    )
            else:
                if test_cmd and "-p no:xdist" in test_cmd:
                    issues.append(
                        ValidationIssue(
                            f.name,
                            "warning",
                            "test_command includes '-p no:xdist' but uses_xdist is false",
                        )
                    )

    return issues


def _type_name(t: type | tuple[type, ...]) -> str:
    """Format a type or tuple of types as a readable string."""
    if isinstance(t, tuple):
        names = [x.__name__ for x in t]
        return " | ".join(names)
    return t.__name__


# ---------------------------------------------------------------------------
# Index operations
# ---------------------------------------------------------------------------


def rebuild_index(registry_dir: Path) -> int:
    """Rebuild index.yaml from all package YAML files.

    Returns:
        The count of packages indexed.
    """
    index = load_index(registry_dir)
    update_index_from_packages(index, registry_dir)

    # Also pick up any new packages not yet in the index
    packages_dir = registry_dir / "packages"
    if packages_dir.is_dir():
        existing_names = {e.name for e in index.packages}
        for f in sorted(packages_dir.glob("*.yaml")):
            name = f.stem
            if name not in existing_names:
                pkg = load_package(name, registry_dir)
                index.packages.append(
                    IndexEntry(
                        name=name,
                        extension_type=pkg.extension_type,
                        enriched=pkg.enriched,
                        skip=pkg.skip,
                        skip_versions_keys=sorted(pkg.skip_versions.keys()),
                    )
                )

    sort_index(index)
    save_index(index, registry_dir)
    return len(index.packages)


def add_index_field(
    registry_dir: Path,
    field_name: str,
    dry_run: bool = True,
) -> OpResult | DryRunPreview:
    """Add a field to every entry in the index.

    Reads the value from each package's YAML file.
    """
    index = load_index(registry_dir)
    modified_names: list[str] = [entry.name for entry in index.packages]

    if dry_run:
        return DryRunPreview(
            affected_count=len(modified_names),
            skipped_count=0,
            error_count=0,
            sample_diffs=[],
        )

    # Rebuild and save
    count = rebuild_index(registry_dir)
    return OpResult(modified=modified_names, skipped=[], errors=[], total=count)


def remove_index_field(
    registry_dir: Path,
    field_name: str,
    dry_run: bool = True,
) -> OpResult | DryRunPreview:
    """Remove a field from every entry in the index.

    Raises:
        ValueError: If *field_name* is a protected index field.
    """
    if field_name in PROTECTED_INDEX_FIELDS:
        raise ValueError(f"Cannot remove protected index field '{field_name}'")

    index = load_index(registry_dir)

    if dry_run:
        return DryRunPreview(
            affected_count=len(index.packages),
            skipped_count=0,
            error_count=0,
            sample_diffs=[],
        )

    # Remove the field from index entries by rebuilding
    # The field won't be present if it's not in IndexEntry
    count = rebuild_index(registry_dir)
    return OpResult(
        modified=[e.name for e in index.packages],
        skipped=[],
        errors=[],
        total=count,
    )
