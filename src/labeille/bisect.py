"""Automated crash bisection across git history.

Binary-searches a package's git history to find the first commit
where a crash appears (or disappears).
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from labeille.crash import detect_crash
from labeille.logging import get_logger
from labeille.runner import _clean_env, create_venv, install_package, run_test_command

log = get_logger("bisect")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BisectConfig:
    """Configuration for a bisect run."""

    package: str
    good_rev: str
    bad_rev: str
    target_python: Path
    registry_dir: Path | None = None
    timeout: int = 600
    test_command: str | None = None
    install_command: str | None = None
    crash_signature: str | None = None
    extra_deps: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)
    work_dir: Path | None = None
    verbose: bool = False


@dataclass
class BisectStep:
    """Result of testing a single commit during bisect."""

    commit: str
    commit_short: str
    status: str  # "good", "bad", "skip"
    detail: str = ""
    crash_signature: str | None = None
    duration_seconds: float = 0.0


@dataclass
class BisectResult:
    """Final result of a bisect operation."""

    package: str
    first_bad_commit: str | None
    first_bad_commit_short: str | None
    good_rev: str
    bad_rev: str
    steps: list[BisectStep]
    total_commits: int
    commits_tested: int

    @property
    def success(self) -> bool:
        """True if the first bad commit was found."""
        return self.first_bad_commit is not None


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def get_commit_range(repo_dir: Path, good_rev: str, bad_rev: str) -> list[str]:
    """Get the list of commit hashes between good and bad (exclusive of good).

    Returns commits in chronological order (oldest first).
    The good commit is NOT included; the bad commit IS included.
    """
    proc = subprocess.run(
        ["git", "rev-list", "--ancestry-path", f"{good_rev}..{bad_rev}"],
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
        timeout=30,
    )
    if proc.returncode != 0:
        log.error("git rev-list failed: %s", proc.stderr.strip())
        return []
    # rev-list returns newest first; reverse for chronological.
    commits = [c.strip() for c in proc.stdout.strip().split("\n") if c.strip()]
    commits.reverse()
    return commits


def _resolve_rev(repo_dir: Path, rev: str) -> str | None:
    """Resolve a revision spec to a full commit hash."""
    proc = subprocess.run(
        ["git", "rev-parse", rev],
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
        timeout=10,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return None


def _clone_full(repo_url: str, dest: Path) -> None:
    """Clone a repo at full depth (no --depth flag)."""
    subprocess.run(
        ["git", "clone", repo_url, str(dest)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )


def _get_commit_info(repo_dir: Path, commit: str) -> tuple[str, str, str] | None:
    """Get (author, date, subject) for a commit, or None on failure."""
    proc = subprocess.run(
        ["git", "log", "--format=%an%n%as%n%s", "-1", commit],
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
        timeout=10,
    )
    if proc.returncode == 0:
        lines = proc.stdout.strip().split("\n")
        if len(lines) >= 3:
            return (lines[0], lines[1], lines[2])
    return None


# ---------------------------------------------------------------------------
# Per-revision testing
# ---------------------------------------------------------------------------


def test_revision(repo_dir: Path, commit: str, config: BisectConfig) -> BisectStep:
    """Checkout a commit, build a venv, install, and run tests.

    Returns a BisectStep with status:

    - ``"good"``: tests ran without the target crash.
    - ``"bad"``: the target crash was detected.
    - ``"skip"``: build or install failed (can't test this commit).
    """
    short = commit[:7]
    start = time.monotonic()

    # Checkout the target revision.
    checkout = subprocess.run(
        ["git", "checkout", "--force", commit],
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
        timeout=60,
        check=False,
    )
    if checkout.returncode != 0:
        return BisectStep(
            commit=commit,
            commit_short=short,
            status="skip",
            detail=f"git checkout failed: {checkout.stderr.strip()[:200]}",
            duration_seconds=time.monotonic() - start,
        )
    # Clean untracked files from previous revisions.
    subprocess.run(
        ["git", "clean", "-fdx"],
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
        timeout=60,
        check=False,
    )

    # Create a fresh venv for this revision.
    with tempfile.TemporaryDirectory(prefix=f"bisect-{short}-") as venv_str:
        venv_path = Path(venv_str)
        try:
            create_venv(config.target_python, venv_path)
        except (subprocess.CalledProcessError, OSError) as exc:
            return BisectStep(
                commit=commit,
                commit_short=short,
                status="skip",
                detail=f"venv creation failed: {exc}",
                duration_seconds=time.monotonic() - start,
            )

        venv_python = venv_path / "bin" / "python"
        env = _clean_env(
            PYTHON_JIT="1",
            PYTHONFAULTHANDLER="1",
            PYTHONDONTWRITEBYTECODE="1",
            ASAN_OPTIONS="detect_leaks=0",
        )
        env.update(config.env_overrides)

        # Install the package.
        install_cmd = config.install_command or "pip install -e ."
        try:
            install_result = install_package(
                venv_python, install_cmd, cwd=repo_dir, env=env, timeout=config.timeout
            )
            if install_result.returncode != 0:
                stderr_tail = (install_result.stderr or "").strip()[-200:]
                return BisectStep(
                    commit=commit,
                    commit_short=short,
                    status="skip",
                    detail=f"install failed (exit {install_result.returncode}): {stderr_tail}",
                    duration_seconds=time.monotonic() - start,
                )
        except subprocess.TimeoutExpired:
            return BisectStep(
                commit=commit,
                commit_short=short,
                status="skip",
                detail="install timed out",
                duration_seconds=time.monotonic() - start,
            )

        # Install extra deps if specified (best-effort).
        if config.extra_deps:
            extra_cmd = f"pip install {' '.join(config.extra_deps)}"
            try:
                install_package(
                    venv_python, extra_cmd, cwd=repo_dir, env=env, timeout=config.timeout
                )
            except (subprocess.TimeoutExpired, OSError):
                pass

        # Run tests.
        test_cmd = config.test_command or "python -m pytest"
        try:
            test_result = run_test_command(
                venv_python, test_cmd, cwd=repo_dir, env=env, timeout=config.timeout
            )
        except subprocess.TimeoutExpired:
            return BisectStep(
                commit=commit,
                commit_short=short,
                status="skip",
                detail="tests timed out",
                duration_seconds=time.monotonic() - start,
            )

        elapsed = time.monotonic() - start

        # Check for crash.
        crash_info = detect_crash(test_result.returncode, test_result.stderr)
        if crash_info is None:
            return BisectStep(
                commit=commit,
                commit_short=short,
                status="good",
                detail=f"exit {test_result.returncode}, no crash",
                duration_seconds=elapsed,
            )

        # If a specific crash signature was requested, only count matching crashes.
        if config.crash_signature:
            if config.crash_signature.lower() not in crash_info.signature.lower():
                return BisectStep(
                    commit=commit,
                    commit_short=short,
                    status="good",
                    detail=f"crash detected but signature doesn't match (got: {crash_info.signature[:100]})",
                    crash_signature=crash_info.signature,
                    duration_seconds=elapsed,
                )

        return BisectStep(
            commit=commit,
            commit_short=short,
            status="bad",
            detail=crash_info.signature[:200],
            crash_signature=crash_info.signature,
            duration_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Main bisect algorithm
# ---------------------------------------------------------------------------


def run_bisect(config: BisectConfig) -> BisectResult:
    """Execute the bisect algorithm.

    1. Clone the repo at full depth.
    2. Verify good rev is good and bad rev is bad.
    3. Binary search for the first bad commit.
    4. Handle ``"skip"`` (unbuildable) commits by trying neighbors.
    """
    from labeille.registry import load_package

    steps: list[BisectStep] = []

    # Load package info from registry if available.
    if config.registry_dir:
        try:
            pkg_entry = load_package(config.package, config.registry_dir)
            if config.install_command is None:
                config.install_command = pkg_entry.install_command
            if config.test_command is None:
                config.test_command = pkg_entry.test_command
        except Exception:
            pass  # Registry is optional for bisect.

    # Determine repo URL.
    repo_url: str | None = None
    if config.registry_dir:
        try:
            pkg_entry = load_package(config.package, config.registry_dir)
            repo_url = pkg_entry.repo
        except Exception:
            pass
    if not repo_url:
        raise ValueError(
            f"No repo URL for {config.package}. "
            f"Provide --registry-dir with a registry entry for this package."
        )

    # Set up work directory.
    work_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
    if config.work_dir:
        repo_dir = config.work_dir / f"{config.package}-bisect"
    else:
        work_dir_ctx = tempfile.TemporaryDirectory(prefix="labeille-bisect-")
        repo_dir = Path(work_dir_ctx.name) / config.package

    try:
        # Clone at full depth (bisect needs the full history).
        log.info("Cloning %s (full depth) for bisect...", config.package)
        if repo_dir.exists():
            subprocess.run(
                ["git", "fetch", "origin"],
                capture_output=True,
                text=True,
                cwd=str(repo_dir),
                timeout=120,
                check=True,
            )
        else:
            _clone_full(repo_url, repo_dir)

        # Resolve symbolic refs to full hashes.
        good_hash = _resolve_rev(repo_dir, config.good_rev)
        bad_hash = _resolve_rev(repo_dir, config.bad_rev)
        if good_hash is None:
            raise ValueError(f"Could not resolve good revision: {config.good_rev}")
        if bad_hash is None:
            raise ValueError(f"Could not resolve bad revision: {config.bad_rev}")

        log.info("Bisecting %s: good=%s bad=%s", config.package, good_hash[:7], bad_hash[:7])

        # Get the commit range.
        commits = get_commit_range(repo_dir, good_hash, bad_hash)
        if not commits:
            raise ValueError(
                f"No commits found between {config.good_rev} and "
                f"{config.bad_rev}. Ensure good is an ancestor of bad."
            )

        total = len(commits)
        log.info("%d commits between good and bad", total)

        # Step 1: Verify the good revision.
        log.info("Verifying good revision %s...", good_hash[:7])
        good_step = test_revision(repo_dir, good_hash, config)
        steps.append(good_step)
        if good_step.status == "bad":
            log.error(
                "Good revision %s actually crashes! Provide an older known-good revision.",
                good_hash[:7],
            )
            return BisectResult(
                package=config.package,
                first_bad_commit=None,
                first_bad_commit_short=None,
                good_rev=config.good_rev,
                bad_rev=config.bad_rev,
                steps=steps,
                total_commits=total,
                commits_tested=1,
            )
        if good_step.status == "skip":
            log.error(
                "Good revision %s can't be built/tested: %s",
                good_hash[:7],
                good_step.detail,
            )
            return BisectResult(
                package=config.package,
                first_bad_commit=None,
                first_bad_commit_short=None,
                good_rev=config.good_rev,
                bad_rev=config.bad_rev,
                steps=steps,
                total_commits=total,
                commits_tested=1,
            )

        # Step 2: Verify the bad revision.
        log.info("Verifying bad revision %s...", bad_hash[:7])
        bad_step = test_revision(repo_dir, bad_hash, config)
        steps.append(bad_step)
        if bad_step.status != "bad":
            log.error(
                "Bad revision %s doesn't crash (status: %s).",
                bad_hash[:7],
                bad_step.status,
            )
            return BisectResult(
                package=config.package,
                first_bad_commit=None,
                first_bad_commit_short=None,
                good_rev=config.good_rev,
                bad_rev=config.bad_rev,
                steps=steps,
                total_commits=total,
                commits_tested=2,
            )

        # Step 3: Binary search.
        lo = 0  # Index into commits[] — currently good.
        hi = len(commits) - 1  # Index into commits[] — currently bad.

        while lo + 1 < hi:
            mid = (lo + hi) // 2
            mid_commit = commits[mid]
            remaining = hi - lo - 1
            log.info(
                "Bisect step: testing %s (%d commits remaining)",
                mid_commit[:7],
                remaining,
            )

            step = test_revision(repo_dir, mid_commit, config)
            steps.append(step)

            if step.status == "bad":
                hi = mid
            elif step.status == "good":
                lo = mid
            else:
                # Skip: can't test this commit. Try neighbors.
                log.info("Skipping %s (%s), trying neighbors...", mid_commit[:7], step.detail[:80])
                found_neighbor = _try_neighbors(commits, lo, hi, mid, repo_dir, config, steps)
                if found_neighbor is not None:
                    new_lo, new_hi = found_neighbor
                    lo, hi = new_lo, new_hi
                else:
                    log.warning(
                        "Could not find a testable commit near %s. "
                        "Bisect result may be imprecise.",
                        mid_commit[:7],
                    )
                    lo = mid  # Force progress to avoid infinite loop.

        first_bad = commits[hi]
        log.info("Bisect complete: first bad commit is %s", first_bad[:7])

        return BisectResult(
            package=config.package,
            first_bad_commit=first_bad,
            first_bad_commit_short=first_bad[:7],
            good_rev=config.good_rev,
            bad_rev=config.bad_rev,
            steps=steps,
            total_commits=total,
            commits_tested=len(steps),
        )

    finally:
        if work_dir_ctx is not None:
            work_dir_ctx.cleanup()


def _try_neighbors(
    commits: list[str],
    lo: int,
    hi: int,
    mid: int,
    repo_dir: Path,
    config: BisectConfig,
    steps: list[BisectStep],
) -> tuple[int, int] | None:
    """Try commits near mid when mid is a skip. Returns (new_lo, new_hi) or None."""
    tested = {s.commit for s in steps}
    for offset in range(1, min(5, hi - lo)):
        for candidate in [mid + offset, mid - offset]:
            if lo < candidate < hi:
                neighbor = commits[candidate]
                if neighbor in tested:
                    continue
                log.info("  Trying neighbor %s...", neighbor[:7])
                neighbor_step = test_revision(repo_dir, neighbor, config)
                steps.append(neighbor_step)
                if neighbor_step.status == "bad":
                    return (lo, candidate)
                if neighbor_step.status == "good":
                    return (candidate, hi)
    return None


def _log2(n: int) -> int:
    """Rough log base 2 for estimating bisect steps."""
    if n <= 0:
        return 0
    count = 0
    while n > 1:
        n >>= 1
        count += 1
    return count
