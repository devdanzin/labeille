"""Filesystem cache management for benchmark isolation.

Provides cache-dropping functionality for cold-cache benchmarking.
Cache dropping requires elevated privileges. Rather than running
labeille as root, we use a targeted sudoers rule that allows just
the cache-drop command without a password.

Setup (one-time, as root)::

    echo "$USER ALL=(root) NOPASSWD: /usr/local/bin/labeille-drop-caches" \
        | sudo tee /etc/sudoers.d/labeille-drop-caches
    sudo chmod 440 /etc/sudoers.d/labeille-drop-caches

The ``labeille-drop-caches`` script is a minimal wrapper::

    #!/bin/sh
    sync
    echo 3 > /proc/sys/vm/drop_caches  # Linux
    # or: purge                          # macOS

See ``generate_drop_caches_script()`` to auto-create the script.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from labeille.logging import get_logger

log = get_logger("bench.cache")


# Path to the cache-drop helper script.
DROP_CACHES_SCRIPT = "/usr/local/bin/labeille-drop-caches"


@dataclass
class CacheDropStatus:
    """Result of checking whether cache dropping is available."""

    available: bool  # True if cache dropping will work
    running_as_root: bool  # True if UID == 0
    script_exists: bool  # True if DROP_CACHES_SCRIPT exists
    sudo_works: bool  # True if passwordless sudo succeeds
    platform_supported: bool  # True if Linux or macOS
    message: str  # Human-readable explanation


def check_cache_drop_available() -> CacheDropStatus:
    """Check whether filesystem cache dropping is available.

    Checks (in order):
    1. Platform is Linux or macOS (not Windows, etc.).
    2. The labeille-drop-caches script exists.
    3. Passwordless sudo for the script works (dry-run via sudo -n).

    Does NOT check if we're running as root — that's handled by
    the caller (which should refuse to proceed if root without
    --run-dangerously-as-root).

    Returns:
        CacheDropStatus with detailed availability info.
    """
    running_as_root = os.getuid() == 0

    # 1. Platform check.
    if sys.platform not in ("linux", "darwin"):
        return CacheDropStatus(
            available=False,
            running_as_root=running_as_root,
            script_exists=False,
            sudo_works=False,
            platform_supported=False,
            message="Cache dropping is only supported on Linux and macOS.",
        )

    # 2. Script existence check.
    if not Path(DROP_CACHES_SCRIPT).exists():
        return CacheDropStatus(
            available=False,
            running_as_root=running_as_root,
            script_exists=False,
            sudo_works=False,
            platform_supported=True,
            message=(
                f"Cache-drop script not found at {DROP_CACHES_SCRIPT}.\n"
                f"Run 'labeille bench setup-cache-drop' for setup instructions."
            ),
        )

    # 3. Passwordless sudo check.
    try:
        result = subprocess.run(
            ["sudo", "-n", DROP_CACHES_SCRIPT, "--check"],
            capture_output=True,
            timeout=5,
        )
        sudo_works = result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        sudo_works = False

    if sudo_works:
        return CacheDropStatus(
            available=True,
            running_as_root=running_as_root,
            script_exists=True,
            sudo_works=True,
            platform_supported=True,
            message="Cache dropping is configured and ready.",
        )

    return CacheDropStatus(
        available=False,
        running_as_root=running_as_root,
        script_exists=True,
        sudo_works=False,
        platform_supported=True,
        message=(
            f"Passwordless sudo not configured for {DROP_CACHES_SCRIPT}.\n"
            f"Run 'labeille bench setup-cache-drop' for setup instructions."
        ),
    )


def drop_caches(*, allow_root: bool = False) -> bool:
    """Drop filesystem caches.

    Uses sudo to run the labeille-drop-caches script. Refuses to
    run if the current process is root (unless allow_root is True).

    Args:
        allow_root: If True, allow running as root. Required to be
            explicitly set — labeille should never accidentally run
            privileged operations.

    Returns:
        True if caches were dropped successfully, False otherwise.
    """
    is_root = os.getuid() == 0
    if is_root and not allow_root:
        log.error("Refusing to drop caches as root. Use --run-dangerously-as-root if intentional.")
        return False

    try:
        if is_root:
            # Running as root — no sudo needed.
            result = subprocess.run(
                [DROP_CACHES_SCRIPT],
                capture_output=True,
                timeout=10,
            )
        else:
            result = subprocess.run(
                ["sudo", "-n", DROP_CACHES_SCRIPT],
                capture_output=True,
                timeout=10,
            )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            log.warning("Cache drop failed (exit %d): %s", result.returncode, stderr)
            return False
        return True

    except subprocess.TimeoutExpired:
        log.warning("Cache drop timed out.")
        return False
    except OSError as exc:
        log.warning("Cache drop failed: %s", exc)
        return False


def check_not_root(*, allow_root: bool = False) -> None:
    """Refuse to run as root unless explicitly allowed.

    Args:
        allow_root: If True, skip the check.

    Raises:
        SystemExit: If running as root without allow_root.
    """
    if os.getuid() == 0 and not allow_root:
        raise SystemExit(
            "labeille should not run as root. Use --run-dangerously-as-root to override."
        )


def generate_drop_caches_script() -> str:
    """Generate the contents of the labeille-drop-caches helper script.

    Returns platform-appropriate script content.
    """
    return """\
#!/bin/sh
# labeille-drop-caches — drop filesystem caches for benchmark isolation
# Install: sudo install -m 755 <this-file> /usr/local/bin/labeille-drop-caches

set -e

case "$1" in
    --check)
        # No-op: used by labeille to test sudo access.
        exit 0
        ;;
esac

sync

case "$(uname -s)" in
    Linux)
        echo 3 > /proc/sys/vm/drop_caches
        ;;
    Darwin)
        purge
        ;;
    *)
        echo "Unsupported platform: $(uname -s)" >&2
        exit 1
        ;;
esac
"""


def format_setup_instructions() -> str:
    """Return human-readable setup instructions for cache dropping.

    Includes steps to create the script, install it, and add the
    sudoers rule.
    """
    return f"""\
Cache dropping setup
====================

1. Create the helper script:

    labeille bench setup-cache-drop --show-script > /tmp/labeille-drop-caches

2. Install it (requires root):

    sudo install -m 755 /tmp/labeille-drop-caches {DROP_CACHES_SCRIPT}

3. Add a passwordless sudoers rule:

    echo "$USER ALL=(root) NOPASSWD: {DROP_CACHES_SCRIPT}" \\
        | sudo tee /etc/sudoers.d/labeille-drop-caches
    sudo chmod 440 /etc/sudoers.d/labeille-drop-caches

4. Verify:

    sudo -n {DROP_CACHES_SCRIPT} --check && echo "OK"

Once configured, use --drop-caches or --warm-vs-cold with 'labeille bench run'.
"""
