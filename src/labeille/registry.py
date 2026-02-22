"""Registry for package test configurations.

This module handles reading, writing, and validating the registry of packages
and their test configurations. The registry consists of an index file
(``registry/index.yaml``) listing all tracked packages, and per-package
configuration files (``registry/packages/{name}.yaml``) with detailed test
setup instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PackageEntry:
    """Full configuration for a single package's test setup.

    Corresponds to a ``registry/packages/{name}.yaml`` file.
    """

    package: str
    repo: str
    pypi_url: str
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
    download_count: int = 0
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


def load_index(registry_path: Path) -> Index:
    """Load the registry index from disk.

    Args:
        registry_path: Path to the registry directory.

    Returns:
        The parsed index.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def save_index(index: Index, registry_path: Path) -> None:
    """Save the registry index to disk.

    Args:
        index: The index to save.
        registry_path: Path to the registry directory.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def load_package(name: str, registry_path: Path) -> PackageEntry:
    """Load a package configuration from the registry.

    Args:
        name: The package name.
        registry_path: Path to the registry directory.

    Returns:
        The parsed package entry.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def save_package(entry: PackageEntry, registry_path: Path) -> None:
    """Save a package configuration to the registry.

    Args:
        entry: The package entry to save.
        registry_path: Path to the registry directory.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError
