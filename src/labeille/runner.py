"""Test runner for executing package test suites against JIT-enabled CPython.

This module handles cloning repositories, installing packages into a
JIT-enabled Python environment, running their test suites, and capturing
the results including any crashes (segfaults, aborts, assertion failures).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    """Configuration for a single test suite run."""

    package_name: str
    repo_url: str
    python_path: Path
    test_command: str
    timeout: int | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a single test suite run."""

    package_name: str
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    crash_signal: int | None = None
    crash_signature: str | None = None
    duration_seconds: float = 0.0


def run_test_suite(config: RunConfig) -> RunResult:
    """Run a package's test suite and capture the result.

    Args:
        config: The run configuration.

    Returns:
        The result of the test run, including any crash information.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def clone_repo(repo_url: str, dest: Path) -> Path:
    """Clone a git repository to a local directory.

    Args:
        repo_url: The URL of the git repository.
        dest: The destination directory.

    Returns:
        The path to the cloned repository.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def install_package(repo_path: Path, python_path: Path, install_command: str) -> None:
    """Install a package from a local repository into a Python environment.

    Args:
        repo_path: The path to the local repository.
        python_path: The path to the Python interpreter.
        install_command: The command to install the package.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError
