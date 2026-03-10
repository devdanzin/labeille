"""Git operations, package spec parsing, and sdist helpers for the test runner.

Extracted from :mod:`labeille.runner` for clarity.  All names are
re-exported by :mod:`labeille.runner` so existing imports are unaffected.
"""

from __future__ import annotations

import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from labeille.logging import get_logger
from labeille.resolve import fetch_pypi_metadata

log = get_logger("repo_ops")


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------


def clone_repo(repo_url: str, dest: Path, clone_depth: int | None = None) -> str | None:
    """Clone a git repository and return the HEAD revision.

    Args:
        repo_url: The URL of the git repository.
        dest: The destination directory for the clone.
        clone_depth: Clone depth. ``None`` means shallow (depth=1).
            A positive integer uses that depth.

    Returns:
        The HEAD commit hash, or ``None`` on failure.

    Raises:
        subprocess.CalledProcessError: If the clone fails.
    """
    depth = clone_depth if clone_depth is not None else 1
    log.debug("Running: git clone --depth=%d %s %s", depth, repo_url, dest)
    clone_proc = subprocess.run(
        ["git", "clone", f"--depth={depth}", repo_url, str(dest)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    if clone_proc.stderr:
        log.debug("git clone stderr: %s", clone_proc.stderr.strip())

    # Fetch tags when using deeper clones (needed for setuptools-scm etc.).
    if clone_depth is not None and clone_depth > 1:
        log.debug("Fetching tags for %s (clone_depth=%d)", dest, clone_depth)
        fetch_proc = subprocess.run(
            ["git", "fetch", "--tags"],
            capture_output=True,
            text=True,
            cwd=str(dest),
            timeout=120,
            check=False,
        )
        if fetch_proc.returncode != 0:
            log.debug("git fetch --tags failed (non-fatal): %s", fetch_proc.stderr.strip())

    # Get the revision.
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(dest),
        timeout=10,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return None


def pull_repo(dest: Path) -> str | None:
    """Fetch latest changes and force-reset to upstream HEAD.

    Uses ``fetch`` + ``reset --hard`` + ``clean -fdx`` instead of
    ``pull --ff-only`` to handle dirty working trees left by test
    suites without needing a full re-clone.

    Args:
        dest: The directory containing the existing clone.

    Returns:
        The HEAD commit hash, or ``None`` on failure.

    Raises:
        subprocess.CalledProcessError: If the fetch fails.
    """
    log.debug("Fetching and resetting repo in %s", dest)

    # Fetch latest from origin.
    subprocess.run(
        ["git", "fetch", "origin"],
        capture_output=True,
        text=True,
        cwd=str(dest),
        timeout=120,
        check=True,
    )

    # Reset to the fetched HEAD, discarding any local modifications.
    # FETCH_HEAD works reliably with shallow clones where origin/HEAD
    # or origin/main might not be set.
    reset_proc = subprocess.run(
        ["git", "reset", "--hard", "FETCH_HEAD"],
        capture_output=True,
        text=True,
        cwd=str(dest),
        timeout=60,
        check=False,
    )
    if reset_proc.returncode != 0:
        log.debug(
            "git reset --hard failed (non-fatal): %s",
            reset_proc.stderr.strip(),
        )

    # Remove untracked files and directories left by test runs.
    clean_proc = subprocess.run(
        ["git", "clean", "-fdx"],
        capture_output=True,
        text=True,
        cwd=str(dest),
        timeout=60,
        check=False,
    )
    if clean_proc.returncode != 0:
        log.debug(
            "git clean failed (non-fatal): %s",
            clean_proc.stderr.strip(),
        )

    # Get the HEAD revision.
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(dest),
        timeout=10,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return None


# ---------------------------------------------------------------------------
# Package spec parsing and revision checkout
# ---------------------------------------------------------------------------


def parse_package_specs(
    raw: str,
) -> tuple[list[str], dict[str, str]]:
    """Parse a comma-separated package spec string.

    Each spec is either ``name`` or ``name@revision``.

    Returns:
        A tuple of (package_names, revision_overrides) where
        revision_overrides maps package name to revision string.

    Examples::

        "requests,click" → (["requests", "click"], {})
        "requests@abc123,click" → (["requests", "click"], {"requests": "abc123"})
        "numpy@HEAD~5" → (["numpy"], {"numpy": "HEAD~5"})
    """
    names: list[str] = []
    revisions: dict[str, str] = {}
    for spec in raw.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if "@" in spec:
            name, rev = spec.split("@", 1)
            name = name.strip()
            rev = rev.strip()
            if name and rev:
                names.append(name)
                revisions[name] = rev
            elif name:
                names.append(name)
        else:
            names.append(spec)
    return names, revisions


def parse_repo_overrides(raw: tuple[str, ...]) -> dict[str, str]:
    """Parse ``--repo-override`` arguments into a dict.

    Each argument is ``package_name=repo_url``.

    Raises:
        ValueError: If an argument is malformed (no ``=``, or empty name/URL).
    """
    overrides: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(
                f"Invalid --repo-override format: {item!r}. "
                f"Expected PKG=URL (e.g., 'requests=https://...')"
            )
        name, url = item.split("=", 1)
        name = name.strip()
        url = url.strip()
        if not name or not url:
            raise ValueError("Invalid --repo-override: both package name and URL required.")
        overrides[name] = url
    return overrides


def checkout_revision(repo_dir: Path, revision: str) -> str | None:
    """Checkout a specific git revision and return the resulting HEAD hash.

    Args:
        repo_dir: Path to the git repository.
        revision: Any git ref — commit hash, branch, tag, ``HEAD~N``.

    Returns:
        The full commit hash after checkout, or ``None`` on failure.
    """
    proc = subprocess.run(
        ["git", "checkout", revision],
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        log.warning(
            "git checkout %s failed in %s: %s. "
            "If using a revision beyond the clone depth, try --no-shallow.",
            revision,
            repo_dir,
            proc.stderr.strip(),
        )
        return None

    # Get the resolved commit hash.
    rev_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
        timeout=10,
    )
    if rev_proc.returncode == 0:
        return rev_proc.stdout.strip()
    return None


# ---------------------------------------------------------------------------
# Sdist-mode helpers
# ---------------------------------------------------------------------------


def fetch_latest_pypi_version(
    package_name: str,
    *,
    timeout: float = 10.0,
) -> str | None:
    """Fetch the latest release version of a package from PyPI.

    Returns:
        Version string (e.g. ``"2.1.0"``), or ``None`` on failure.
    """
    metadata = fetch_pypi_metadata(package_name, timeout=timeout)
    if metadata is None:
        return None
    try:
        return str(metadata["info"]["version"])
    except (KeyError, TypeError):
        return None


_TAG_PATTERNS: list[str] = [
    "v{version}",
    "{version}",
    "{package}-{version}",
    "release-{version}",
    "V{version}",
]


def checkout_matching_tag(
    repo_dir: Path,
    package_name: str,
    version: str,
) -> tuple[str | None, str | None]:
    """Attempt to check out the git tag matching a PyPI version.

    Tries several common tag naming conventions.  Fetches tags first
    for shallow clones.

    Returns:
        Tuple of ``(commit_hash, matched_tag)``.  Both ``None`` if no
        tag found.
    """
    # Best-effort fetch of tags for shallow clones.
    subprocess.run(
        ["git", "fetch", "--tags", "--depth=1", "origin"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
        timeout=60,
    )

    normalized = package_name.lower().replace("_", "-")

    for pattern in _TAG_PATTERNS:
        tag = pattern.format(version=version, package=normalized)
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{}}"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            timeout=10,
        )
        if proc.returncode == 0:
            commit = checkout_revision(repo_dir, tag)
            return (commit, tag)

    return (None, None)


def detect_source_layout(repo_dir: Path, import_name: str) -> str:
    """Detect whether a repo uses ``src/`` layout or flat layout.

    Returns:
        ``"src"`` if import_name exists under ``src/``, ``"flat"``
        if it exists at the repo root, or ``"unknown"`` if neither.
    """
    if (repo_dir / "src" / import_name).is_dir():
        return "src"
    if (repo_dir / import_name).is_dir():
        return "flat"
    return "unknown"


@contextmanager
def shield_source_dir(
    repo_dir: Path,
    import_name: str,
    layout: str,
) -> Iterator[None]:
    """Context manager that temporarily hides the source directory.

    For flat-layout packages, renames ``repo_dir/import_name`` to
    ``repo_dir/_labeille_shielded_import_name`` before yielding,
    and restores it after.  For src-layout or unknown, this is a
    no-op.
    """
    if layout != "flat":
        yield
        return
    src_path = repo_dir / import_name
    if not src_path.exists():
        yield
        return
    shield_path = repo_dir / f"_labeille_shielded_{import_name}"
    src_path.rename(shield_path)
    try:
        yield
    finally:
        shield_path.rename(src_path)


_SELF_INSTALL_RE = re.compile(
    r"""
    (?:pip\s+install|python\s+-m\s+pip\s+install)  # pip install variant
    \s+.*                                            # flags
    (?:
        -e\s+[.'"]                                   # editable: -e . or -e '.[dev]'
        | \.\s*$                                     # bare: pip install .
        | '\.\[                                      # pip install '.[extras]'
        | "\.\[                                      # pip install ".[extras]"
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _is_self_install_segment(segment: str) -> bool:
    """Return True if this command segment installs the package itself."""
    return bool(_SELF_INSTALL_RE.search(segment))


_EXTRAS_RE = re.compile(r"\.\[([^\]]+)\]")


def _extract_extras(segment: str) -> str | None:
    """Extract extras from an install segment.

    E.g. ``'.[dev,test]'`` -> ``'dev,test'``.
    """
    m = _EXTRAS_RE.search(segment)
    return m.group(1) if m else None


def split_install_command(
    install_command: str,
) -> tuple[list[str], list[str]]:
    """Split an install command into self-install and dependency segments.

    Segments are split on ``&&``.  A segment is classified as a
    "self-install" if it contains editable install markers (``-e .``,
    ``pip install .``, etc.).

    Returns:
        Tuple of ``(self_install_segments, other_segments)``.
    """
    if not install_command.strip():
        return ([], [])
    parts = re.split(r"\s*&&\s*", install_command)
    parts = [p.strip() for p in parts if p.strip()]
    self_install: list[str] = []
    other: list[str] = []
    for seg in parts:
        if _is_self_install_segment(seg):
            self_install.append(seg)
        else:
            other.append(seg)
    return (self_install, other)


def build_sdist_install_commands(
    package_name: str,
    install_command: str,
) -> tuple[str, str]:
    """Build sdist-mode install commands from a registry install_command.

    Replaces self-install segments with a ``pip install --no-binary``
    command.  If the self-install had extras, those are included.

    Returns:
        Tuple of ``(sdist_install_cmd, deps_install_cmd)``.
    """
    self_segments, other_segments = split_install_command(install_command)
    extras: str | None = None
    for seg in self_segments:
        extras = _extract_extras(seg)
        if extras:
            break
    if extras:
        sdist_cmd = f"pip install --no-binary {package_name} '{package_name}[{extras}]'"
    else:
        sdist_cmd = f"pip install --no-binary {package_name} {package_name}"
    deps_cmd = " && ".join(other_segments) if other_segments else ""
    return (sdist_cmd, deps_cmd)
