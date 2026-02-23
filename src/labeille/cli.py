"""Command-line interface for labeille.

Provides the main CLI entry point with ``resolve`` and ``run`` subcommands.
"""

from __future__ import annotations

from pathlib import Path

import click

from labeille import __version__
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


@main.command()
@click.argument("packages", nargs=-1)
@click.option("--from-file", "from_file", type=click.Path(exists=True, path_type=Path))
@click.option("--from-json", "from_json", type=click.Path(exists=True, path_type=Path))
@click.option("--top", "top_n", type=int, default=None)
@click.option(
    "--registry-dir",
    type=click.Path(path_type=Path),
    default=Path("registry"),
    show_default=True,
)
@click.option("--dry-run", is_flag=True, help="Show what would be done without making changes.")
@click.option("--timeout", type=float, default=10.0, show_default=True)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output.")
@click.option("-q", "--quiet", is_flag=True, help="Only show errors.")
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=Path("labeille-resolve.log"),
    show_default=True,
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
    registry_dir: Path,
    dry_run: bool,
    timeout: float,
    verbose: bool,
    quiet: bool,
    log_file: Path,
    workers: int,
) -> None:
    """Resolve PyPI packages to source repositories and build a test registry."""
    setup_logging(verbose=verbose, quiet=quiet, log_file=log_file)

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
    default=Path("registry"),
    show_default=True,
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=Path("results"),
    show_default=True,
)
@click.option("--top", "top_n", type=int, default=None)
@click.option("--packages", "packages_csv", type=str, default=None)
@click.option("--skip-extensions", is_flag=True)
@click.option("--skip-completed", is_flag=True, help="Resume: skip already-tested packages.")
@click.option("--stop-after-crash", type=int, default=None)
@click.option("--timeout", type=int, default=600, show_default=True)
@click.option("--env", "env_pairs", type=str, multiple=True, help="KEY=VALUE env var.")
@click.option("--run-id", type=str, default=None)
@click.option("--dry-run", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
@click.option("-q", "--quiet", is_flag=True, help="Only show crashes.")
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=Path("labeille-run.log"),
    show_default=True,
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
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Number of packages to test in parallel.",
)
@click.pass_context
def run_cmd(
    ctx: click.Context,
    target_python: Path,
    registry_dir: Path,
    results_dir: Path,
    top_n: int | None,
    packages_csv: str | None,
    skip_extensions: bool,
    skip_completed: bool,
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
    work_dir: Path | None,
    workers: int,
) -> None:
    """Run test suites against a JIT-enabled Python build and detect crashes."""
    from datetime import datetime, timezone

    from labeille.runner import RunnerConfig, run_all, validate_target_python
    from labeille.summary import format_summary

    setup_logging(verbose=verbose, quiet=quiet, log_file=log_file)

    if workers < 1:
        raise click.UsageError("--workers must be at least 1")

    # Validate target python up front.
    try:
        python_version = validate_target_python(target_python)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Target Python: {python_version}")

    # Parse env pairs.
    env_overrides: dict[str, str] = {}
    for pair in env_pairs:
        if "=" not in pair:
            raise click.UsageError(f"Invalid --env format (expected KEY=VALUE): {pair}")
        key, _, value = pair.partition("=")
        env_overrides[key] = value

    # Parse packages filter.
    packages_filter: list[str] | None = None
    if packages_csv:
        packages_filter = [p.strip() for p in packages_csv.split(",") if p.strip()]

    # Generate run ID.
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    # --work-dir sets both --repos-dir and --venvs-dir as defaults.
    if work_dir is not None:
        if repos_dir is None:
            repos_dir = work_dir / "repos"
        if venvs_dir is None:
            venvs_dir = work_dir / "venvs"

    config = RunnerConfig(
        target_python=target_python,
        registry_dir=registry_dir,
        results_dir=results_dir,
        run_id=run_id,
        timeout=timeout,
        top_n=top_n,
        packages_filter=packages_filter,
        skip_extensions=skip_extensions,
        skip_completed=skip_completed,
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
        cli_args=list(ctx.params.keys()),
    )

    click.echo(f"Run ID: {run_id}")
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
