"""Classify packages as pure Python or C extension based on wheel tags.

This module inspects PyPI wheel metadata to determine whether a package
is pure Python, contains C extensions, or cannot be classified. This
classification helps prioritize packages for testing — C extension packages
are more likely to exercise the JIT through tight numerical loops and
native code interactions.
"""

from __future__ import annotations


def classify_package(package_name: str) -> str:
    """Classify a package as pure Python, C extension, or unknown.

    Inspects the available wheel files on PyPI to determine the package type
    based on platform and ABI tags.

    Args:
        package_name: The name of the package on PyPI.

    Returns:
        One of ``"pure"``, ``"extensions"``, or ``"unknown"``.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def is_pure_wheel(filename: str) -> bool:
    """Determine if a wheel filename indicates a pure Python package.

    Pure Python wheels use the ``py3-none-any`` tag pattern.

    Args:
        filename: The wheel filename to inspect.

    Returns:
        True if the wheel is pure Python.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def has_platform_wheel(filenames: list[str]) -> bool:
    """Determine if any wheel filename indicates platform-specific (C extension) code.

    Platform-specific wheels contain compiled extensions and will have
    platform tags like ``linux_x86_64`` or ``manylinux``.

    Args:
        filenames: The list of wheel filenames to inspect.

    Returns:
        True if any wheel is platform-specific.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError
