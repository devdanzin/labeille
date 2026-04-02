"""Command-line interface for labeille.

Provides the main CLI entry point and registers all subcommand groups
(``resolve``, ``run``, ``scan-deps``, ``registry``, ``analyze``, ``bench``,
``ft``, ``compat``, ``bisect``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from labeille.runner import RunnerConfig
    from labeille.scan_deps import ScanResult

from labeille import __version__
from labeille.cli_utils import parse_csv_list, parse_env_pairs
from labeille.logging import setup_logging
from labeille.resolve import (
    merge_inputs,
    read_packages_from_args,
    read_packages_from_file,
    read_packages_from_json,
    resolve_all,
)


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """labeille — Hunt for CPython JIT bugs by running real-world test suites."""


def _register_subcommands() -> None:
    """Register subcommand groups on the main CLI group."""
    from labeille.analyze_cli import analyze as analyze_group
    from labeille.bench_cli import bench as bench_group
    from labeille.cext_build_cli import cext_build as cext_build_cmd
    from labeille.compat_cli import compat as compat_group
    from labeille.ft_cli import ft as ft_group
    from labeille.registry_cli import registry as registry_group
    from labeille.tsan_run_cli import tsan_run as tsan_run_cmd

    main.add_command(registry_group)
    main.add_command(analyze_group)
    main.add_command(bench_group)
    main.add_command(ft_group)
    main.add_command(compat_group)
    main.add_command(cext_build_cmd)
    main.add_command(tsan_run_cmd)


_register_subcommands()


@main.command()
@click.argument("packages", nargs=-1)
@click.option(
    "--from-file",
    "from_file",
    type=click.Path(exists=True, path_type=Path),
    help="File with one package name per line.",
)
@click.option(
    "--from-json",
    "from_json",
    type=click.Path(exists=True, path_type=Path),
    help="JSON file with package metadata (e.g. top-pypi-packages.json).",
)
@click.option("--top", "top_n", type=int, default=None, help="Resolve only the top N packages.")
@click.option(
    "--registry-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Registry directory (default: ~/.local/share/labeille/registry/).",
)
@click.option("--dry-run", is_flag=True, help="Show what would be done without making changes.")
@click.option(
    "--timeout",
    type=float,
    default=10.0,
    show_default=True,
    help="PyPI API request timeout in seconds.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output.")
@click.option("-q", "--quiet", is_flag=True, help="Only show errors.")
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=Path("labeille-resolve.log"),
    show_default=True,
    help="Path to the log file.",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Number of parallel PyPI API requests.",
)
def resolve(
    packages: tuple[str, ...],
    from_file: Path | None,
    from_json: Path | None,
    top_n: int | None,
    registry_dir: Path | None,
    dry_run: bool,
    timeout: float,
    verbose: bool,
    quiet: bool,
    log_file: Path,
    workers: int,
) -> None:
    """Resolve PyPI packages to source repositories and build a test registry."""
    from labeille.registry import default_registry_dir

    setup_logging(verbose=verbose, quiet=quiet, log_file=log_file)
    if registry_dir is None:
        registry_dir = default_registry_dir()

    if top_n is not None and from_json is None:
        raise click.UsageError("--top requires --from-json")

    # Collect inputs from all sources.
    sources = []
    if packages:
        sources.append(read_packages_from_args(packages))
    if from_file is not None:
        sources.append(read_packages_from_file(from_file))
    if from_json is not None:
        sources.append(read_packages_from_json(from_json, top_n=top_n))

    if not sources:
        raise click.UsageError(
            "At least one of PACKAGES, --from-file, or --from-json must be provided."
        )

    merged = merge_inputs(*sources)
    if not merged:
        click.echo("No packages to resolve.")
        return

    click.echo(f"Resolving {len(merged)} package(s)...")
    if dry_run:
        click.echo("(dry-run mode — no files will be written)")

    if workers < 1:
        raise click.UsageError("--workers must be at least 1")

    results, summary = resolve_all(
        merged, registry_dir, timeout=timeout, dry_run=dry_run, workers=workers
    )

    # Print summary.
    click.echo("")
    click.echo(f"Resolved:          {summary.resolved}")
    click.echo(f"  New files:       {summary.created}")
    click.echo(f"  Updated:         {summary.updated}")
    click.echo(f"Skipped (enriched):{summary.skipped_enriched}")
    click.echo(f"Failed:            {summary.failed}")
    if dry_run:
        click.echo(f"Skipped (dry-run): {summary.skipped}")


def _build_run_config(  # noqa: PLR0913
    *,
    target_python: Path,
    python_version: str,
    registry_dir: Path,
    results_dir: Path,
    top_n: int | None,
    packages_csv: str | None,
    skip_extensions: bool,
    skip_completed: bool,
    force_run: bool,
    stop_after_crash: int | None,
    timeout: int,
    env_pairs: tuple[str, ...],
    run_id: str | None,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
    keep_work_dirs: bool,
    refresh_venvs: bool,
    repos_dir: Path | None,
    venvs_dir: Path | None,
    extra_deps: str | None,
    test_command_override: str | None,
    test_command_suffix: str | None,
    repo_overrides_raw: tuple[str, ...],
    clone_depth: int | None,
    no_shallow: bool,
    work_dir: Path | None,
    workers: int,
    installer: str,
    install_from: str,
) -> RunnerConfig:
    """Validate CLI inputs and build a RunnerConfig."""
    from labeille.io_utils import extract_minor_version, generate_run_id
    from labeille.runner import RunnerConfig, parse_package_specs, parse_repo_overrides

    env_overrides = parse_env_pairs(env_pairs)

    # Parse packages filter (supports name@revision syntax).
    packages_filter: list[str] | None = None
    revision_overrides: dict[str, str] = {}
    if packages_csv:
        packages_filter, revision_overrides = parse_package_specs(packages_csv)
        if not packages_filter:
            packages_filter = None

    # Resolve clone depth: --no-shallow wins, then --clone-depth.
    effective_clone_depth: int | None = None
    if no_shallow:
        effective_clone_depth = 0
    elif clone_depth is not None:
        effective_clone_depth = clone_depth

    # Warn if both test command override and suffix are set.
    if test_command_override and test_command_suffix:
        click.echo(
            "Warning: --test-command-override is set, --test-command-suffix will be ignored.",
            err=True,
        )

    # Parse repo overrides.
    try:
        repo_overrides = parse_repo_overrides(repo_overrides_raw)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    # Generate run ID.
    if run_id is None:
        run_id = generate_run_id("run")

    # --work-dir sets both --repos-dir and --venvs-dir as defaults.
    if work_dir is not None:
        if repos_dir is None:
            repos_dir = work_dir / "repos"
        if venvs_dir is None:
            venvs_dir = work_dir / "venvs"

    # Warn when only one of --repos-dir / --venvs-dir is set: the other
    # directory will use a temporary path and be cleaned up after the run.
    if (repos_dir is None) != (venvs_dir is None):
        missing = "--venvs-dir" if venvs_dir is None else "--repos-dir"
        click.echo(
            f"Warning: {missing} is not set; those directories will be temporary.",
            err=True,
        )

    return RunnerConfig(
        target_python=target_python,
        registry_dir=registry_dir,
        results_dir=results_dir,
        run_id=run_id,
        timeout=timeout,
        top_n=top_n,
        packages_filter=packages_filter,
        skip_extensions=skip_extensions,
        skip_completed=skip_completed,
        force_run=force_run,
        target_python_version=extract_minor_version(python_version),
        stop_after_crash=stop_after_crash,
        env_overrides=env_overrides,
        dry_run=dry_run,
        verbose=verbose,
        quiet=quiet,
        keep_work_dirs=keep_work_dirs,
        refresh_venvs=refresh_venvs,
        workers=workers,
        repos_dir=repos_dir,
        venvs_dir=venvs_dir,
        cli_args=sys.argv[1:],
        clone_depth_override=effective_clone_depth,
        revision_overrides=revision_overrides,
        extra_deps=parse_csv_list(extra_deps),
        test_command_override=test_command_override,
        test_command_suffix=test_command_suffix,
        repo_overrides=repo_overrides,
        installer=installer,  # type: ignore[arg-type]  # Click Choice returns str
        install_from=install_from,  # type: ignore[arg-type]  # Click Choice returns str
    )


@main.command("run")
@click.option(
    "--target-python",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the Python interpreter to test with.",
)
@click.option(
    "--registry-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Registry directory (default: ~/.local/share/labeille/registry/).",
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Output directory for test results.",
)
@click.option(
    "--top", "top_n", type=int, default=None, help="Test only the top N packages by downloads."
)
@click.option(
    "--packages",
    "packages_csv",
    type=str,
    default=None,
    help="Comma-separated list of package names (supports pkg@revision).",
)
@click.option("--skip-extensions", is_flag=True, help="Skip packages with C extensions.")
@click.option("--skip-completed", is_flag=True, help="Resume: skip already-tested packages.")
@click.option(
    "--force-run",
    is_flag=True,
    help="Override skip and skip_versions flags; run all selected packages.",
)
@click.option(
    "--stop-after-crash",
    type=int,
    default=None,
    help="Stop after N crashes are detected.",
)
@click.option(
    "--timeout",
    type=int,
    default=600,
    show_default=True,
    help="Test suite timeout in seconds.",
)
@click.option("--env", "env_pairs", type=str, multiple=True, help="KEY=VALUE env var.")
@click.option("--run-id", type=str, default=None, help="Custom run identifier.")
@click.option("--dry-run", is_flag=True, help="Show what would be done without running tests.")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output.")
@click.option("-q", "--quiet", is_flag=True, help="Only show crashes.")
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=Path("labeille-run.log"),
    show_default=True,
    help="Path to the log file.",
)
@click.option("--keep-work-dirs", is_flag=True, help="Don't clean up working directories.")
@click.option(
    "--refresh-venvs",
    is_flag=True,
    help="Delete and recreate existing venvs to pick up install command changes.",
)
@click.option(
    "--repos-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Persistent directory for repo clones. Reuses existing clones.",
)
@click.option(
    "--venvs-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Persistent directory for venvs. Reuses existing venvs.",
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Base directory for repos and venvs (sets --repos-dir and --venvs-dir).",
)
@click.option(
    "--extra-deps",
    type=str,
    default=None,
    help=(
        "Comma-separated list of additional packages to install in "
        "each venv after the package's own dependencies. "
        "Example: 'coverage,pytest-timeout'."
    ),
)
@click.option(
    "--test-command-override",
    type=str,
    default=None,
    help=(
        "Replace the test command for ALL packages in this run. "
        "For appending flags to existing commands, use --test-command-suffix instead."
    ),
)
@click.option(
    "--test-command-suffix",
    type=str,
    default=None,
    help=(
        "Append flags to each package's test command. "
        "Example: '--tb=long -v'. Ignored if --test-command-override is set."
    ),
)
@click.option(
    "--repo-override",
    "repo_overrides_raw",
    type=str,
    multiple=True,
    help=(
        "Override the repo URL for a package. Format: PKG=URL. Can be specified multiple times."
    ),
)
@click.option(
    "--clone-depth",
    type=int,
    default=None,
    help=(
        "Git clone depth (overrides per-package clone_depth). "
        "Use 0 or --no-shallow for full clones."
    ),
)
@click.option(
    "--no-shallow",
    is_flag=True,
    default=False,
    help="Disable shallow clones (equivalent to --clone-depth=0).",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Number of packages to test in parallel.",
)
@click.option(
    "--installer",
    type=click.Choice(["auto", "uv", "pip"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Package installer backend. 'auto' uses uv if available.",
)
@click.option(
    "--install-from",
    "install_from",
    type=click.Choice(["source", "sdist"], case_sensitive=False),
    default="source",
    show_default=True,
    help=(
        "Install packages from cloned source (default) or from PyPI sdist. "
        "sdist mode installs the released version via pip while running tests "
        "from the cloned repo."
    ),
)
@click.pass_context
def run_cmd(
    ctx: click.Context,
    target_python: Path,
    registry_dir: Path | None,
    results_dir: Path,
    top_n: int | None,
    packages_csv: str | None,
    skip_extensions: bool,
    skip_completed: bool,
    force_run: bool,
    stop_after_crash: int | None,
    timeout: int,
    env_pairs: tuple[str, ...],
    run_id: str | None,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
    log_file: Path,
    keep_work_dirs: bool,
    refresh_venvs: bool,
    repos_dir: Path | None,
    venvs_dir: Path | None,
    extra_deps: str | None,
    test_command_override: str | None,
    test_command_suffix: str | None,
    repo_overrides_raw: tuple[str, ...],
    clone_depth: int | None,
    no_shallow: bool,
    work_dir: Path | None,
    workers: int,
    installer: str,
    install_from: str,
) -> None:
    """Run test suites against a JIT-enabled Python build and detect crashes."""
    from labeille.registry import default_registry_dir
    from labeille.runner import run_all, validate_target_python
    from labeille.summary import format_summary

    setup_logging(verbose=verbose, quiet=quiet, log_file=log_file)

    if workers < 1:
        raise click.UsageError("--workers must be at least 1")

    if registry_dir is None:
        registry_dir = default_registry_dir()

    # Validate target python up front.
    try:
        python_version = validate_target_python(target_python)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Target Python: {python_version}")

    config = _build_run_config(
        target_python=target_python,
        python_version=python_version,
        registry_dir=registry_dir,
        results_dir=results_dir,
        top_n=top_n,
        packages_csv=packages_csv,
        skip_extensions=skip_extensions,
        skip_completed=skip_completed,
        force_run=force_run,
        stop_after_crash=stop_after_crash,
        timeout=timeout,
        env_pairs=env_pairs,
        run_id=run_id,
        dry_run=dry_run,
        verbose=verbose,
        quiet=quiet,
        keep_work_dirs=keep_work_dirs,
        refresh_venvs=refresh_venvs,
        repos_dir=repos_dir,
        venvs_dir=venvs_dir,
        extra_deps=extra_deps,
        test_command_override=test_command_override,
        test_command_suffix=test_command_suffix,
        repo_overrides_raw=repo_overrides_raw,
        clone_depth=clone_depth,
        no_shallow=no_shallow,
        work_dir=work_dir,
        workers=workers,
        installer=installer,
        install_from=install_from,
    )

    click.echo(f"Run ID: {config.run_id}")
    if dry_run:
        click.echo("(dry-run mode — no tests will be executed)")

    output = run_all(config)

    # Determine summary mode.
    if quiet:
        mode = "quiet"
    elif verbose:
        mode = "verbose"
    else:
        mode = "default"

    summary_text = format_summary(
        output.results,
        output.summary,
        config,
        output.python_version,
        output.jit_enabled,
        output.total_duration,
        run_dir=output.run_dir,
        mode=mode,
    )
    if summary_text:
        click.echo(summary_text)


@main.command("scan-deps")
@click.argument("repo_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--package-name",
    type=str,
    default=None,
    help="PyPI package name (default: inferred from repo directory name).",
)
@click.option(
    "--test-dirs",
    type=str,
    default=None,
    help="Comma-separated test directories to scan (default: auto-detect).",
)
@click.option(
    "--scan-source",
    is_flag=True,
    help="Also scan package source code (default: tests only).",
)
@click.option(
    "--install-command",
    type=str,
    default=None,
    help="Current install_command to compare against.",
)
@click.option(
    "--registry-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Registry directory for cross-referencing import_names.",
)
@click.option(
    "--include-conditional/--no-conditional",
    default=True,
    help="Include imports inside try/except ImportError (default: true).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["human", "json", "pip"]),
    default="human",
    show_default=True,
    help="Output format.",
)
def scan_deps_cmd(
    repo_path: Path,
    package_name: str | None,
    test_dirs: str | None,
    scan_source: bool,
    install_command: str | None,
    registry_dir: Path | None,
    include_conditional: bool,
    output_format: str,
) -> None:
    """Scan a repository for external test dependencies via AST import analysis."""
    import json as json_mod
    from dataclasses import asdict

    from labeille.scan_deps import scan_package_deps

    if package_name is None:
        package_name = repo_path.resolve().name

    parsed_test_dirs: list[str] | None = None
    if test_dirs is not None:
        parsed_test_dirs = [d.strip() for d in test_dirs.split(",") if d.strip()]

    # Load registry entries for cross-referencing if registry-dir is provided.
    registry_entries = None
    if registry_dir is not None:
        from labeille.registry import load_package

        packages_dir = registry_dir / "packages"
        if packages_dir.is_dir():
            registry_entries = []
            for yaml_file in sorted(packages_dir.glob("*.yaml")):
                try:
                    registry_entries.append(load_package(yaml_file.stem, registry_dir))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    click.echo(
                        f"Warning: could not load registry entry {yaml_file.stem}: {exc}",
                        err=True,
                    )

    result = scan_package_deps(
        repo_path.resolve(),
        package_name,
        test_dirs=parsed_test_dirs,
        scan_source=scan_source,
        install_command=install_command,
        registry_entries=registry_entries,
        include_conditional=include_conditional,
    )

    if output_format == "json":
        click.echo(json_mod.dumps(asdict(result), indent=2))
    elif output_format == "pip":
        _print_pip_format(result)
    else:
        _print_human_format(result, install_command)


def _print_pip_format(result: ScanResult) -> None:
    """Print just the pip install command for missing deps."""
    if result.missing:
        line = "pip install " + " ".join(sorted(result.missing))
        click.echo(line)
    if result.unresolved:
        names = ", ".join(dep.import_name for dep in result.unresolved)
        click.echo(f"# Unresolved: {names}")


def _print_human_format(result: ScanResult, install_command: str | None) -> None:
    """Print human-readable scan results."""
    click.echo(f"Scanning: {result.package_name}")
    click.echo(f"  Test directories: {', '.join(result.scan_dirs)}")
    click.echo(f"  Files scanned: {result.total_files_scanned}")
    click.echo(f"  Total imports found: {result.total_imports_found}")
    click.echo(f"  External dependencies: {len(result.resolved)}")
    click.echo()

    if result.resolved:
        click.echo(f"Resolved dependencies ({len(result.resolved)}):")
        for dep in result.resolved:
            files_str = ", ".join(dep.import_files[:3])
            if len(dep.import_files) > 3:
                files_str += f" (+{len(dep.import_files) - 3} more)"
            cond = "  [conditional]" if dep.is_conditional else ""
            name_col = f"  {dep.pip_package:<20s}"
            source_col = f"({dep.source})"
            click.echo(f"{name_col} {source_col:<12s} {files_str}{cond}")
            if dep.note:
                click.echo(f"  {'':20s} \u26a0 {dep.note}")
        click.echo()

    if result.unresolved:
        click.echo(f"Unresolved imports ({len(result.unresolved)}):")
        for dep in result.unresolved:
            files_str = ", ".join(dep.import_files[:3])
            if len(dep.import_files) > 3:
                files_str += f" (+{len(dep.import_files) - 3} more)"
            click.echo(f"  {dep.import_name:<20s} {files_str}")
        click.echo()

    if install_command:
        click.echo("Comparison with install_command:")
        click.echo(f"  install_command: {install_command}")
        # Show extras.
        from labeille.scan_deps import _parse_install_packages

        _, extras = _parse_install_packages(install_command)
        for extra in extras:
            click.echo(f"  Extras: {extra}")
        if result.already_installed:
            click.echo(f"  Already installed: {', '.join(sorted(result.already_installed))}")
        if result.missing:
            click.echo("  Missing (add to install_command):")
            click.echo(f"    {result.suggested_install}")
        else:
            click.echo("  All resolved dependencies are already installed.")
    elif result.suggested_install:
        click.echo("Suggested install command:")
        click.echo(f"  {result.suggested_install}")


@main.command("bisect")
@click.argument("package")
@click.option(
    "--good",
    "good_rev",
    required=True,
    help="Known-good git revision (no crash).",
)
@click.option(
    "--bad",
    "bad_rev",
    required=True,
    help="Known-bad git revision (crash present).",
)
@click.option(
    "--target-python",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the Python interpreter to test with.",
)
@click.option(
    "--registry-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Registry directory (default: ~/.local/share/labeille/registry/).",
)
@click.option(
    "--test-command",
    type=str,
    default=None,
    help="Override test command (default: from registry or 'python -m pytest').",
)
@click.option(
    "--install-command",
    type=str,
    default=None,
    help="Override install command (default: from registry or 'pip install -e .').",
)
@click.option(
    "--crash-signature",
    type=str,
    default=None,
    help="Only count crashes matching this substring.",
)
@click.option(
    "--timeout",
    type=int,
    default=600,
    show_default=True,
    help="Test suite timeout in seconds.",
)
@click.option(
    "--extra-deps",
    type=str,
    default=None,
    help="Comma-separated list of additional packages to install.",
)
@click.option("--env", "env_pairs", type=str, multiple=True, help="KEY=VALUE env var.")
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Persistent work directory for clones (default: temp dir).",
)
@click.option("-v", "--verbose", is_flag=True)
@click.option(
    "--installer",
    type=click.Choice(["auto", "uv", "pip"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Package installer backend. 'auto' uses uv if available.",
)
def bisect_cmd(
    package: str,
    good_rev: str,
    bad_rev: str,
    target_python: Path,
    registry_dir: Path | None,
    test_command: str | None,
    install_command: str | None,
    crash_signature: str | None,
    timeout: int,
    extra_deps: str | None,
    env_pairs: tuple[str, ...],
    work_dir: Path | None,
    verbose: bool,
    installer: str,
) -> None:
    """Bisect a package's git history to find the first commit that introduced a crash."""
    from labeille.bisect import BisectConfig, run_bisect
    from labeille.registry import default_registry_dir

    setup_logging(verbose=verbose)
    if registry_dir is None:
        registry_dir = default_registry_dir()

    env_overrides = parse_env_pairs(env_pairs)
    parsed_extra_deps = parse_csv_list(extra_deps)

    config = BisectConfig(
        package=package,
        good_rev=good_rev,
        bad_rev=bad_rev,
        target_python=target_python,
        registry_dir=registry_dir,
        timeout=timeout,
        test_command=test_command,
        install_command=install_command,
        crash_signature=crash_signature,
        extra_deps=parsed_extra_deps,
        env_overrides=env_overrides,
        work_dir=work_dir,
        verbose=verbose,
        installer=installer,  # type: ignore[arg-type]  # Click Choice returns str
    )

    click.echo(f"Bisecting {package}: good={good_rev} bad={bad_rev}")
    click.echo(f"Target Python: {target_python}")

    try:
        result = run_bisect(config)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # Print results.
    click.echo("")
    click.echo(
        f"Bisect complete: {result.commits_tested} commits tested "
        f"out of {result.total_commits} in range."
    )

    if result.success:
        click.echo(
            f"First bad commit: {result.first_bad_commit_short} ({result.first_bad_commit})"
        )
        # Show commit info if available.
        from labeille.bisect import _get_commit_info

        if work_dir:
            repo_dir = work_dir / f"{package}-bisect"
            if repo_dir.exists() and result.first_bad_commit:
                info = _get_commit_info(repo_dir, result.first_bad_commit)
                if info:
                    click.echo(f"  Author: {info[0]}")
                    click.echo(f"  Date:   {info[1]}")
                    click.echo(f"  Subject: {info[2]}")
    else:
        click.echo("Could not identify the first bad commit.")

    # Show step details in verbose mode.
    if verbose:
        click.echo("")
        click.echo("Steps:")
        for step in result.steps:
            click.echo(
                f"  {step.commit_short} [{step.status}] "
                f"({step.duration_seconds:.1f}s) {step.detail[:80]}"
            )
