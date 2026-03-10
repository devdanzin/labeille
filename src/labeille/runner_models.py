"""Data models for the test runner.

Extracted from :mod:`labeille.runner` for clarity.  All names are
re-exported by :mod:`labeille.runner` so existing imports are unaffected.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Installer backend
# ---------------------------------------------------------------------------


class InstallerBackend(enum.Enum):
    """Package installer backend."""

    PIP = "pip"
    UV = "uv"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Top-level configuration for a test run."""

    target_python: Path
    registry_dir: Path
    results_dir: Path
    run_id: str
    timeout: int = 600
    top_n: int | None = None
    packages_filter: list[str] | None = None
    skip_extensions: bool = False
    skip_completed: bool = False
    stop_after_crash: int | None = None
    env_overrides: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    verbose: bool = False
    quiet: bool = False
    keep_work_dirs: bool = False
    repos_dir: Path | None = None
    venvs_dir: Path | None = None
    refresh_venvs: bool = False
    force_run: bool = False
    target_python_version: str = ""
    workers: int = 1
    cli_args: list[str] = field(default_factory=list)
    clone_depth_override: int | None = None
    revision_overrides: dict[str, str] = field(default_factory=dict)
    extra_deps: list[str] = field(default_factory=list)
    test_command_override: str | None = None
    test_command_suffix: str | None = None
    repo_overrides: dict[str, str] = field(default_factory=dict)
    installer: str = "auto"  # auto | uv | pip
    install_from: str = "source"  # source | sdist


@dataclass
class PackageResult:
    """Result of testing a single package."""

    package: str
    repo: str | None = None
    package_version: str | None = None
    git_revision: str | None = None
    status: Literal[
        "pass", "fail", "crash", "timeout", "install_error", "clone_error", "error"
    ] = "error"
    exit_code: int = -1
    signal: int | None = None
    crash_signature: str | None = None
    duration_seconds: float = 0.0
    install_duration_seconds: float = 0.0
    test_command: str = ""
    timeout_hit: bool = False
    stderr_tail: str = ""
    installed_dependencies: dict[str, str] = field(default_factory=dict)
    error_message: str | None = None
    requested_revision: str | None = None
    install_from: str = ""  # "source" or "sdist"
    sdist_version: str | None = None
    sdist_tag_matched: bool | None = None
    installer_backend: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        return asdict(self)


@dataclass
class RunSummary:
    """Aggregate summary of a test run."""

    total: int = 0
    tested: int = 0
    passed: int = 0
    failed: int = 0
    crashed: int = 0
    timed_out: int = 0
    install_errors: int = 0
    clone_errors: int = 0
    errors: int = 0
    skipped: int = 0
    version_skipped: int = 0


@dataclass
class RunOutput:
    """Complete output from a test run."""

    results: list[PackageResult]
    summary: RunSummary
    total_duration: float
    python_version: str
    jit_enabled: bool
    run_dir: Path = field(default_factory=lambda: Path("."))
