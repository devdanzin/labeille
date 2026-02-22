"""Classify packages as pure Python or C extension based on wheel tags.

This module inspects PyPI wheel metadata to determine whether a package
is pure Python, contains C extensions, or cannot be classified. This
classification helps prioritize packages for testing — C extension packages
are more likely to exercise the JIT through tight numerical loops and
native code interactions.
"""

from __future__ import annotations

import re
from typing import Any

from labeille.logging import get_logger

log = get_logger("classifier")

# Patterns that indicate a platform-specific (native extension) wheel.
_PLATFORM_INDICATORS = re.compile(
    r"manylinux|musllinux|macosx|win32|win_amd64|win_arm64", re.IGNORECASE
)


def classify_from_urls(urls: list[dict[str, Any]]) -> str:
    """Classify a package from its PyPI ``urls`` array.

    Examines the distribution files for the *latest* version to determine
    whether the package is pure Python, has C/Rust extensions, or cannot
    be classified.

    Args:
        urls: The ``urls`` array from the PyPI JSON API response.

    Returns:
        One of ``"pure"``, ``"extensions"``, or ``"unknown"``.
    """
    if not urls:
        return "unknown"

    filenames = [u.get("filename", "") for u in urls if u.get("filename")]
    wheel_filenames = [f for f in filenames if f.endswith(".whl")]

    if not wheel_filenames:
        # Only sdists — we can't tell from filenames alone.
        return "unknown"

    has_pure = any(is_pure_wheel(f) for f in wheel_filenames)
    has_platform = has_platform_wheel(wheel_filenames)

    if has_platform:
        return "extensions"
    if has_pure:
        return "pure"
    return "unknown"


def is_pure_wheel(filename: str) -> bool:
    """Determine if a wheel filename indicates a pure Python package.

    Pure Python wheels use tag patterns like ``py3-none-any`` or
    ``py2.py3-none-any``.

    Args:
        filename: The wheel filename to inspect.

    Returns:
        True if the wheel is pure Python.
    """
    lower = filename.lower()
    return "none-any.whl" in lower


def has_platform_wheel(filenames: list[str]) -> bool:
    """Determine if any wheel filename indicates platform-specific code.

    Platform-specific wheels contain compiled extensions and will have
    platform tags like ``manylinux``, ``macosx``, ``win``, or ``musllinux``.

    Args:
        filenames: The list of wheel filenames to inspect.

    Returns:
        True if any wheel is platform-specific.
    """
    return any(_PLATFORM_INDICATORS.search(f) for f in filenames)
