"""Resolve PyPI packages to their source repositories.

This module handles the discovery of source repositories for PyPI packages,
including fetching metadata from PyPI, extracting repository URLs, and
classifying packages by extension type. It orchestrates the full resolve
workflow: read inputs, query PyPI, update the registry.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from labeille import __version__
from labeille.classifier import classify_from_urls
from labeille.logging import get_logger
from labeille.registry import (
    IndexEntry,
    PackageEntry,
    load_index,
    load_package,
    package_exists,
    save_index,
    save_package,
    sort_index,
    update_index_from_packages,
)

log = get_logger("resolve")

_PYPI_URL = "https://pypi.org/pypi/{name}/json"
_USER_AGENT = f"labeille/{__version__} (https://github.com/devdanzin/labeille)"

# Keys in project_urls that likely point to a source repository,
# checked in priority order (case-insensitive).
_REPO_KEYS = [
    "source",
    "source code",
    "repository",
    "github",
    "code",
]
# "Homepage" is only accepted when it points to a known forge.
_FORGE_HOSTS = ("github.com", "gitlab.com")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PackageInput:
    """A package name with optional download count from the input source."""

    name: str
    download_count: int | None = None


@dataclass
class ResolveResult:
    """Result of resolving a single package."""

    name: str
    repo_url: str | None = None
    extension_type: str = "unknown"
    action: str = "skipped"  # created | updated | skipped | skipped_enriched | failed
    error: str | None = None


@dataclass
class ResolveSummary:
    """Aggregate summary of a resolve run."""

    total: int = 0
    resolved: int = 0
    skipped_enriched: int = 0
    failed: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0


# ---------------------------------------------------------------------------
# Input reading
# ---------------------------------------------------------------------------


def read_packages_from_args(names: tuple[str, ...] | list[str]) -> list[PackageInput]:
    """Create PackageInput entries from CLI positional arguments."""
    return [PackageInput(name=n.strip().lower()) for n in names if n.strip()]


def read_packages_from_file(path: Path) -> list[PackageInput]:
    """Read package names from a text file (one per line).

    Blank lines and lines starting with ``#`` are ignored.

    Args:
        path: Path to the text file.

    Returns:
        A list of PackageInput entries (download_count will be None).
    """
    entries: list[PackageInput] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(PackageInput(name=line.lower()))
    return entries


def read_packages_from_json(path: Path, top_n: int | None = None) -> list[PackageInput]:
    """Read packages from a PyPI top-packages JSON dump.

    Expects a JSON object with a ``rows`` array containing objects with
    a ``project`` key and optionally a ``download_count`` key.

    Args:
        path: Path to the JSON file.
        top_n: If given, only return the top N packages by download count.

    Returns:
        A list of PackageInput entries with download counts.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = data.get("rows", [])
    entries = [
        PackageInput(
            name=str(r["project"]).lower(),
            download_count=r.get("download_count"),
        )
        for r in rows
        if "project" in r
    ]
    # Sort by download count descending for --top.
    entries.sort(key=lambda e: -(e.download_count or 0))
    if top_n is not None:
        entries = entries[:top_n]
    return entries


def merge_inputs(*sources: list[PackageInput]) -> list[PackageInput]:
    """Merge multiple input sources, deduplicating by name.

    When the same package appears in multiple sources, the entry with a
    non-None download_count is preferred.

    Args:
        *sources: Variable number of PackageInput lists.

    Returns:
        A deduplicated list of PackageInput entries.
    """
    by_name: dict[str, PackageInput] = {}
    for source in sources:
        for pkg in source:
            existing = by_name.get(pkg.name)
            if existing is None:
                by_name[pkg.name] = pkg
            elif existing.download_count is None and pkg.download_count is not None:
                by_name[pkg.name] = pkg
    return list(by_name.values())


# ---------------------------------------------------------------------------
# PyPI metadata
# ---------------------------------------------------------------------------


def fetch_pypi_metadata(package_name: str, *, timeout: float = 10.0) -> dict[str, Any] | None:
    """Fetch package metadata from the PyPI JSON API.

    Args:
        package_name: The name of the package on PyPI.
        timeout: HTTP request timeout in seconds.

    Returns:
        The parsed JSON metadata, or ``None`` on failure.
    """
    url = _PYPI_URL.format(name=package_name)
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
    except requests.ConnectionError:
        log.error("Connection error fetching %s", package_name)
        return None
    except requests.Timeout:
        log.error("Timeout fetching %s", package_name)
        return None
    except requests.RequestException as exc:
        log.error("Request error fetching %s: %s", package_name, exc)
        return None

    if resp.status_code == 404:
        log.warning("Package %s not found on PyPI (404)", package_name)
        return None
    if resp.status_code == 429 or resp.status_code >= 500:
        log.warning("PyPI returned %d for %s, skipping", resp.status_code, package_name)
        return None
    if resp.status_code != 200:
        log.warning("Unexpected status %d for %s", resp.status_code, package_name)
        return None

    try:
        return resp.json()  # type: ignore[no-any-return]
    except (ValueError, requests.JSONDecodeError):
        log.error("Invalid JSON response for %s", package_name)
        return None


def extract_repo_url(metadata: dict[str, Any]) -> str | None:
    """Extract the source repository URL from PyPI metadata.

    Looks in ``info.project_urls`` for keys indicating a source repository,
    in priority order. Falls back to ``Homepage`` only if it points to a
    known code forge (github.com, gitlab.com).

    Args:
        metadata: The parsed PyPI JSON API response.

    Returns:
        The repository URL, or ``None`` if not found.
    """
    info = metadata.get("info", {})
    if not isinstance(info, dict):
        return None
    project_urls = info.get("project_urls")
    if not isinstance(project_urls, dict):
        return None

    # Normalise keys to lowercase for matching.
    normalised: dict[str, str] = {k.lower().strip(): v for k, v in project_urls.items()}

    # Try priority keys first.
    for key in _REPO_KEYS:
        if key in normalised:
            return normalised[key]

    # Fall back to homepage if it points to a forge.
    homepage = normalised.get("homepage", "")
    if homepage and any(host in homepage.lower() for host in _FORGE_HOSTS):
        return homepage

    return None


# ---------------------------------------------------------------------------
# Core resolve logic
# ---------------------------------------------------------------------------


def resolve_package(
    pkg: PackageInput,
    registry_path: Path,
    *,
    timeout: float = 10.0,
    dry_run: bool = False,
) -> ResolveResult:
    """Resolve a single package: fetch metadata, classify, and update registry.

    Args:
        pkg: The package input.
        registry_path: Path to the registry directory.
        timeout: HTTP timeout in seconds.
        dry_run: If True, do not write any files.

    Returns:
        A ResolveResult describing what happened.
    """
    result = ResolveResult(name=pkg.name)

    # Check for enriched file — never overwrite.
    if package_exists(pkg.name, registry_path):
        existing = load_package(pkg.name, registry_path)
        if existing.enriched:
            log.info("Skipping %s (already enriched)", pkg.name)
            result.action = "skipped_enriched"
            result.repo_url = existing.repo
            result.extension_type = existing.extension_type
            return result

    # Fetch metadata from PyPI.
    metadata = fetch_pypi_metadata(pkg.name, timeout=timeout)
    if metadata is None:
        result.action = "failed"
        result.error = "Failed to fetch PyPI metadata"
        return result

    # Extract repo URL and classify.
    result.repo_url = extract_repo_url(metadata)
    urls_data: list[dict[str, Any]] = metadata.get("urls", [])
    result.extension_type = classify_from_urls(urls_data)

    if dry_run:
        result.action = "skipped"
        log.info(
            "[dry-run] %s: repo=%s, type=%s",
            pkg.name,
            result.repo_url,
            result.extension_type,
        )
        return result

    # Create or update the package file.
    pypi_url = f"https://pypi.org/project/{pkg.name}/"

    if package_exists(pkg.name, registry_path):
        # Update existing non-enriched file, preserving manual edits.
        existing = load_package(pkg.name, registry_path)
        if result.repo_url is not None:
            existing.repo = result.repo_url
        existing.extension_type = result.extension_type
        existing.pypi_url = pypi_url
        save_package(existing, registry_path)
        result.action = "updated"
        log.info(
            "Updated %s: repo=%s, type=%s",
            pkg.name,
            existing.repo,
            existing.extension_type,
        )
    else:
        entry = PackageEntry(
            package=pkg.name,
            repo=result.repo_url,
            pypi_url=pypi_url,
            extension_type=result.extension_type,
        )
        save_package(entry, registry_path)
        result.action = "created"
        log.info(
            "Created %s: repo=%s, type=%s",
            pkg.name,
            result.repo_url,
            result.extension_type,
        )

    return result


def _tally_result(summary: ResolveSummary, result: ResolveResult) -> None:
    """Update summary counters from a single resolve result."""
    if result.action == "created":
        summary.created += 1
        summary.resolved += 1
    elif result.action == "updated":
        summary.updated += 1
        summary.resolved += 1
    elif result.action == "skipped_enriched":
        summary.skipped_enriched += 1
    elif result.action == "failed":
        summary.failed += 1
    else:
        summary.skipped += 1


def resolve_all(
    packages: list[PackageInput],
    registry_path: Path,
    *,
    timeout: float = 10.0,
    dry_run: bool = False,
    workers: int = 1,
) -> tuple[list[ResolveResult], ResolveSummary]:
    """Resolve a list of packages and update the registry index.

    When ``workers`` >= 2, resolution is parallelised with a thread pool.
    A 0.25-second delay is inserted between job submissions to avoid
    hammering PyPI.

    Args:
        packages: The packages to resolve.
        registry_path: Path to the registry directory.
        timeout: HTTP timeout in seconds.
        dry_run: If True, do not write any files.
        workers: Number of parallel resolution workers.

    Returns:
        A tuple of (results list, summary).
    """
    results: list[ResolveResult] = []
    summary = ResolveSummary(total=len(packages))

    # Build a lookup of download counts for index updates.
    download_counts: dict[str, int | None] = {p.name: p.download_count for p in packages}

    workers = max(1, workers)
    if workers == 1:
        for pkg in packages:
            result = resolve_package(pkg, registry_path, timeout=timeout, dry_run=dry_run)
            results.append(result)
            _tally_result(summary, result)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[Any, PackageInput] = {}
            for i, pkg in enumerate(packages):
                futures[
                    pool.submit(
                        resolve_package, pkg, registry_path, timeout=timeout, dry_run=dry_run
                    )
                ] = pkg
                # Rate-limit submissions to avoid hammering PyPI.
                if i < len(packages) - 1:
                    time.sleep(0.25)

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    pkg = futures[future]
                    log.error("Resolve worker exception for %s: %s", pkg.name, exc)
                    result = ResolveResult(
                        name=pkg.name, action="failed", error=f"Worker exception: {exc}"
                    )
                results.append(result)
                _tally_result(summary, result)

    # Update the index (unless dry-run).
    if not dry_run:
        index = load_index(registry_path)
        existing_names = {e.name for e in index.packages}

        for result in results:
            if result.action in ("created", "updated", "skipped_enriched"):
                if result.name not in existing_names:
                    index.packages.append(
                        IndexEntry(
                            name=result.name,
                            download_count=download_counts.get(result.name),
                            extension_type=result.extension_type,
                        )
                    )
                    existing_names.add(result.name)
                else:
                    # Update download count if we have one.
                    for entry in index.packages:
                        if entry.name == result.name:
                            dc = download_counts.get(result.name)
                            if dc is not None:
                                entry.download_count = dc
                            break

        update_index_from_packages(index, registry_path)
        sort_index(index)
        save_index(index, registry_path)

    return results, summary
