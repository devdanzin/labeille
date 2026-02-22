"""Registry for package test configurations.

This module handles reading, writing, and validating the registry of packages
and their test configurations. The registry consists of an index file
(``registry/index.yaml``) listing all tracked packages, and per-package
configuration files (``registry/packages/{name}.yaml``) with detailed test
setup instructions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from labeille.logging import get_logger

log = get_logger("registry")


@dataclass
class PackageEntry:
    """Full configuration for a single package's test setup.

    Corresponds to a ``registry/packages/{name}.yaml`` file.
    """

    package: str
    repo: str | None = None
    pypi_url: str = ""
    extension_type: str = "unknown"  # pure | extensions | unknown
    python_versions: list[str] = field(default_factory=list)
    install_method: str = "pip"  # pip | pip-extras | custom
    install_command: str = ""
    test_command: str = ""
    test_framework: str = "pytest"  # pytest | unittest | custom
    uses_xdist: bool = False
    timeout: int | None = None
    skip: bool = False
    skip_reason: str | None = None
    notes: str = ""
    enriched: bool = False


@dataclass
class IndexEntry:
    """Summary entry for a package in the registry index.

    Corresponds to one item in the ``packages`` list in ``registry/index.yaml``.
    """

    name: str
    download_count: int | None = None
    extension_type: str = "unknown"
    enriched: bool = False
    skip: bool = False


@dataclass
class Index:
    """The full registry index.

    Corresponds to the ``registry/index.yaml`` file.
    """

    last_updated: str = ""
    packages: list[IndexEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _package_to_dict(entry: PackageEntry) -> dict[str, Any]:
    """Convert a PackageEntry to an ordered dict suitable for YAML output."""
    return asdict(entry)


def _dict_to_package(data: dict[str, Any]) -> PackageEntry:
    """Create a PackageEntry from a dict, tolerating missing/extra keys."""
    known = {f.name for f in fields(PackageEntry)}
    filtered = {k: v for k, v in data.items() if k in known}
    return PackageEntry(**filtered)


def _dict_to_index_entry(data: dict[str, Any]) -> IndexEntry:
    """Create an IndexEntry from a dict, tolerating missing/extra keys."""
    known = {f.name for f in fields(IndexEntry)}
    filtered = {k: v for k, v in data.items() if k in known}
    return IndexEntry(**filtered)


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------


def load_index(registry_path: Path) -> Index:
    """Load the registry index from disk.

    If the file does not exist, returns an empty index.

    Args:
        registry_path: Path to the registry directory.

    Returns:
        The parsed index.
    """
    index_file = registry_path / "index.yaml"
    if not index_file.exists():
        return Index()
    data = yaml.safe_load(index_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return Index()
    packages = [_dict_to_index_entry(p) for p in data.get("packages", []) if isinstance(p, dict)]
    return Index(last_updated=data.get("last_updated", ""), packages=packages)


def save_index(index: Index, registry_path: Path) -> None:
    """Save the registry index to disk.

    Entries are sorted by download_count descending, with ``None`` values last.

    Args:
        index: The index to save.
        registry_path: Path to the registry directory.
    """
    index.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    sort_index(index)

    data: dict[str, Any] = {
        "last_updated": index.last_updated,
        "packages": [asdict(e) for e in index.packages],
    }
    index_file = registry_path / "index.yaml"
    registry_path.mkdir(parents=True, exist_ok=True)
    index_file.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    log.debug("Saved index with %d packages to %s", len(index.packages), index_file)


def sort_index(index: Index) -> None:
    """Sort index entries by download_count descending (nulls last, then by name)."""
    index.packages.sort(
        key=lambda e: (
            0 if e.download_count is not None else 1,
            -(e.download_count or 0),
            e.name,
        )
    )


# ---------------------------------------------------------------------------
# Package I/O
# ---------------------------------------------------------------------------


def package_path(name: str, registry_path: Path) -> Path:
    """Return the filesystem path for a package YAML file."""
    return registry_path / "packages" / f"{name}.yaml"


def package_exists(name: str, registry_path: Path) -> bool:
    """Check whether a package YAML file exists in the registry."""
    return package_path(name, registry_path).exists()


def load_package(name: str, registry_path: Path) -> PackageEntry:
    """Load a package configuration from the registry.

    Args:
        name: The package name.
        registry_path: Path to the registry directory.

    Returns:
        The parsed package entry.

    Raises:
        FileNotFoundError: If the package YAML file does not exist.
    """
    p = package_path(name, registry_path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return PackageEntry(package=name)
    return _dict_to_package(data)


def save_package(entry: PackageEntry, registry_path: Path) -> None:
    """Save a package configuration to the registry.

    Args:
        entry: The package entry to save.
        registry_path: Path to the registry directory.
    """
    p = package_path(entry.package, registry_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _package_to_dict(entry)
    p.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    log.debug("Saved package %s to %s", entry.package, p)


def update_index_from_packages(index: Index, registry_path: Path) -> None:
    """Refresh index entries with current data from package YAML files.

    For each entry in the index that has a corresponding package file, updates
    ``extension_type``, ``enriched``, and ``skip`` from the package file.

    Args:
        index: The index to update (modified in place).
        registry_path: Path to the registry directory.
    """
    for entry in index.packages:
        if package_exists(entry.name, registry_path):
            pkg = load_package(entry.name, registry_path)
            entry.extension_type = pkg.extension_type
            entry.enriched = pkg.enriched
            entry.skip = pkg.skip
