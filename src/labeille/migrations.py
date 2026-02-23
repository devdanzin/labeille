"""Registry migration framework.

Migrations are named transformations applied to registry YAML files.
Each migration is a Python function registered with ``@register_migration``.
Migrations are logged to ``{registry_dir}/migrations.log`` to prevent
accidental re-application.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from labeille.logging import get_logger
from labeille.registry import _dict_to_package, _package_to_dict

log = get_logger("migrations")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MigrationResult:
    """Result of applying a migration to a single file."""

    package: str
    modified: bool
    description: str


@dataclass
class MigrationSpec:
    """A registered migration."""

    name: str
    description: str
    func: Callable[[Path, dict[str, Any]], MigrationResult]


@dataclass
class MigrationLogEntry:
    """A record of a migration that has been applied."""

    migration: str
    applied_at: str
    files_modified: int
    files_skipped: int


@dataclass
class MigrationDryRun:
    """Preview of what a migration would do."""

    migration: MigrationSpec
    affected_count: int
    skipped_count: int
    sample_results: list[MigrationResult] = field(default_factory=list)


@dataclass
class MigrationExecution:
    """Result of applying a migration."""

    migration: MigrationSpec
    modified_count: int
    skipped_count: int
    results: list[MigrationResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------

_MIGRATIONS: dict[str, MigrationSpec] = {}


def register_migration(
    name: str,
    description: str,
) -> Callable[
    [Callable[[Path, dict[str, Any]], MigrationResult]],
    Callable[[Path, dict[str, Any]], MigrationResult],
]:
    """Decorator to register a migration function."""

    def decorator(
        func: Callable[[Path, dict[str, Any]], MigrationResult],
    ) -> Callable[[Path, dict[str, Any]], MigrationResult]:
        _MIGRATIONS[name] = MigrationSpec(name=name, description=description, func=func)
        return func

    return decorator


def get_migration(name: str) -> MigrationSpec | None:
    """Look up a migration by name."""
    return _MIGRATIONS.get(name)


def list_migrations() -> list[MigrationSpec]:
    """Return all registered migrations in registration order."""
    return list(_MIGRATIONS.values())


# ---------------------------------------------------------------------------
# Migration log
# ---------------------------------------------------------------------------


def _log_path(registry_dir: Path) -> Path:
    """Return the path to the migration log file."""
    return registry_dir / "migrations.log"


def read_migration_log(registry_dir: Path) -> list[MigrationLogEntry]:
    """Read the migration log. Returns empty list if file doesn't exist."""
    path = _log_path(registry_dir)
    if not path.exists():
        return []
    entries: list[MigrationLogEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            entries.append(
                MigrationLogEntry(
                    migration=data["migration"],
                    applied_at=data["applied_at"],
                    files_modified=data["files_modified"],
                    files_skipped=data["files_skipped"],
                )
            )
        except (json.JSONDecodeError, KeyError):
            log.warning("Skipping malformed migration log entry: %s", line)
    return entries


def append_migration_log(registry_dir: Path, entry: MigrationLogEntry) -> None:
    """Append an entry to the migration log."""
    path = _log_path(registry_dir)
    data = {
        "migration": entry.migration,
        "applied_at": entry.applied_at,
        "files_modified": entry.files_modified,
        "files_skipped": entry.files_skipped,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def has_been_applied(registry_dir: Path, migration_name: str) -> bool:
    """Check if a migration has already been applied."""
    entries = read_migration_log(registry_dir)
    return any(e.migration == migration_name for e in entries)


def get_applied_date(registry_dir: Path, migration_name: str) -> str | None:
    """Return the date a migration was applied, or None if not applied."""
    for entry in read_migration_log(registry_dir):
        if entry.migration == migration_name:
            return entry.applied_at
    return None


# ---------------------------------------------------------------------------
# Migration execution engine
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, content: str) -> None:
    """Write content to a file atomically."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def execute_migration(
    migration: MigrationSpec,
    registry_dir: Path,
    *,
    dry_run: bool = True,
) -> MigrationDryRun | MigrationExecution:
    """Run a migration across all package YAML files.

    For each file in ``registry_dir/packages/``:
    1. Load the YAML as a dict.
    2. Call ``migration.func(file_path, data)``.
    3. If modified and not dry_run: re-serialize and write atomically.
    4. Return aggregate results.
    """
    packages_dir = registry_dir / "packages"
    if not packages_dir.is_dir():
        if dry_run:
            return MigrationDryRun(migration=migration, affected_count=0, skipped_count=0)
        return MigrationExecution(migration=migration, modified_count=0, skipped_count=0)

    files = sorted(packages_dir.glob("*.yaml"))

    modified_count = 0
    skipped_count = 0
    results: list[MigrationResult] = []

    for f in files:
        raw = yaml.safe_load(f.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            skipped_count += 1
            continue

        result = migration.func(f, raw)

        if result.modified:
            modified_count += 1
            results.append(result)

            if not dry_run:
                # Re-serialize through the registry module for consistent output.
                entry = _dict_to_package(raw)
                data = _package_to_dict(entry)
                text = yaml.dump(
                    data,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
                _atomic_write(f, text)
        else:
            skipped_count += 1

    if dry_run:
        return MigrationDryRun(
            migration=migration,
            affected_count=modified_count,
            skipped_count=skipped_count,
            sample_results=results[:5],
        )

    # Log the migration.
    log_entry = MigrationLogEntry(
        migration=migration.name,
        applied_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        files_modified=modified_count,
        files_skipped=skipped_count,
    )
    append_migration_log(registry_dir, log_entry)

    return MigrationExecution(
        migration=migration,
        modified_count=modified_count,
        skipped_count=skipped_count,
        results=results,
    )


# ---------------------------------------------------------------------------
# Built-in migration: skip-to-skip-versions
# ---------------------------------------------------------------------------


def _is_version_specific_skip(reason: str) -> bool:
    """Determine if a skip reason is specific to Python 3.15.

    Returns True for reasons related to:
    - PyO3 / maturin / Rust not supporting 3.15
    - Dependencies on pydantic-core, rpds-py (PyO3 packages)
    - Cython build failures on 3.15
    - JIT crashes during install (fixed by CPython rebuild)
    - 3.15-specific incompatibilities

    Returns False for structural reasons:
    - No test suite / stub package
    - Cloud credentials required
    - Monorepo / complex build
    - No repo URL
    - Deprecated / obsolete packages
    """
    r = reason.lower()

    # Structural reasons — don't convert even if they mention Rust/PyO3.
    structural_keywords = [
        "no test suite",
        "no meaningful test",
        "no python test suite",
        "no separate test suite",
        "no source test",
        "binary with no",
    ]
    if any(kw in r for kw in structural_keywords):
        return False

    # Version-specific keywords.
    version_specific_keywords = [
        "pyo3",
        "maturin",
        "rust",
        "pydantic-core",
        "rpds-py",
        "rpds",
        "doesn't support python 3.15",
        "doesn't support 3.15",
        "not support 3.15",
        "no 3.15 support",
        "no 3.15",
        "3.15",
        "cython build step fails",
        "cython build fails",
        "cython) editable build fails",
        "jit crash",
    ]

    return any(kw in r for kw in version_specific_keywords)


@register_migration(
    "skip-to-skip-versions",
    "Convert 3.15-specific skip:true to skip_versions",
)
def migrate_skip_to_skip_versions(
    file_path: Path,
    data: dict[str, Any],
) -> MigrationResult:
    """Move version-specific skip reasons from skip/skip_reason to skip_versions.

    Converts packages where:
    - ``skip`` is True, AND
    - ``skip_reason`` indicates a 3.15-specific issue

    Does NOT convert packages with structural skip reasons.
    """
    package = data.get("package", file_path.stem)

    if not data.get("skip", False):
        return MigrationResult(package=package, modified=False, description="not skipped")

    reason = data.get("skip_reason", "") or ""

    if not _is_version_specific_skip(reason):
        return MigrationResult(
            package=package, modified=False, description="structural skip (not version-specific)"
        )

    # Apply the transformation.
    data["skip"] = False
    data["skip_reason"] = None

    if "skip_versions" not in data:
        data["skip_versions"] = {}
    data["skip_versions"]["3.15"] = reason

    desc = reason[:60] + "..." if len(reason) > 60 else reason
    return MigrationResult(
        package=package,
        modified=True,
        description=f'skip:true -> skip_versions["3.15"]: {desc}',
    )
