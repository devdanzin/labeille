"""CLI for ``labeille cext-build``."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from labeille.logging import setup_logging


@click.command("cext-build")
@click.option(
    "--target-python",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the Python interpreter to build against.",
)
@click.option(
    "--registry-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Registry directory for package metadata.",
)
@click.option(
    "--packages",
    "packages_csv",
    type=str,
    default=None,
    help="Comma-separated list of package names.",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=None,
    help="Top N extension packages by download count.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("cext-builds"),
    show_default=True,
    help="Output directory for build results and compile databases.",
)
@click.option(
    "--repos-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Persistent directory for repo clones.",
)
@click.option(
    "--venvs-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Persistent directory for build venvs.",
)
@click.option(
    "--timeout",
    type=int,
    default=600,
    show_default=True,
    help="Build timeout in seconds per package.",
)
@click.option(
    "--installer",
    type=click.Choice(["auto", "uv", "pip"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Package installer backend.",
)
@click.option(
    "--no-skip-existing",
    is_flag=True,
    default=False,
    help="Rebuild packages even if compile_commands.json already exists.",
)
@click.option(
    "--bear-path",
    type=str,
    default=None,
    help="Path to bear binary (default: auto-detect).",
)
@click.option(
    "--build-isolation/--no-build-isolation",
    default=False,
    help=(
        "Use pip's build isolation (default: off). "
        "Bear requires --no-build-isolation for reliable results."
    ),
)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output.")
def cext_build(
    target_python: Path,
    registry_dir: Path | None,
    packages_csv: str | None,
    top_n: int | None,
    output_dir: Path,
    repos_dir: Path | None,
    venvs_dir: Path | None,
    timeout: int,
    installer: str,
    no_skip_existing: bool,
    bear_path: str | None,
    build_isolation: bool,
    verbose: bool,
) -> None:
    """Build C extensions and generate compile_commands.json databases.

    Uses Bear to intercept compiler invocations during the build.
    When Bear is not available, falls back to build-system-specific
    mechanisms for Meson and CMake projects.

    The output is designed for use with cext-review-toolkit's
    clang-tidy analysis (Tier 2).

    \b
    Examples:
        # Build from registry (extension packages only)
        labeille cext-build --target-python ~/cpython/python \\
            --registry-dir registry --top 20

        # Build specific packages
        labeille cext-build --target-python ~/cpython/python \\
            --packages numpy,lxml,pyyaml

        # With persistent repos and venvs
        labeille cext-build --target-python ~/cpython/python \\
            --registry-dir registry --top 50 \\
            --repos-dir ~/cext-repos --venvs-dir ~/cext-venvs
    """
    setup_logging(verbose=verbose)

    from labeille.cext_build import CextBuildConfig, run_cext_builds

    packages_filter = (
        [p.strip() for p in packages_csv.split(",") if p.strip()] if packages_csv else None
    )

    if not registry_dir and not packages_filter:
        click.echo("Error: provide --registry-dir, --packages, or both.", err=True)
        sys.exit(1)

    config = CextBuildConfig(
        target_python=target_python,
        output_dir=output_dir,
        registry_dir=registry_dir,
        repos_dir=repos_dir,
        venvs_dir=venvs_dir,
        packages_filter=packages_filter,
        top_n=top_n,
        timeout=timeout,
        installer=installer,
        no_build_isolation=not build_isolation,
        bear_path=bear_path,
        skip_if_exists=not no_skip_existing,
        verbose=verbose,
    )

    meta, results = run_cext_builds(config)

    # Print summary.
    click.echo("")
    click.echo(f"Build complete: {meta.build_id}")
    click.echo(f"  Packages:              {meta.total_packages}")
    click.echo(f"  Built OK:              {meta.built_ok}")
    click.echo(f"  Compile DBs generated: {meta.compile_db_generated}")

    if meta.bear_available:
        click.echo(f"  Bear:                  {meta.bear_version}")
    else:
        click.echo("  Bear:                  not available (used fallbacks)")

    # Per-status breakdown.
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    for status, count in sorted(status_counts.items()):
        click.echo(f"  {status}: {count}")

    click.echo(f"\nOutput: {config.output_dir / meta.build_id}")
