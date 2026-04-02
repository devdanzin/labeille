"""CLI for ``labeille tsan-run``."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from labeille.logging import setup_logging


@click.command("tsan-run")
@click.option(
    "--target-python",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a TSan-enabled free-threaded Python interpreter.",
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
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Output directory for TSan reports and metadata.",
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
    help="Persistent directory for venvs.",
)
@click.option(
    "--timeout",
    type=int,
    default=600,
    show_default=True,
    help="Timeout in seconds per package (install + test).",
)
@click.option(
    "--installer",
    type=click.Choice(["auto", "uv", "pip"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Package installer backend.",
)
@click.option(
    "--suppressions",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="TSan suppressions file (default: bundled CPython suppressions).",
)
@click.option(
    "--quick",
    is_flag=True,
    default=False,
    help="Halt on first TSan error per package (fast CI check).",
)
@click.option(
    "--stress",
    type=int,
    default=1,
    show_default=True,
    help="Repeat each test suite N times to increase race detection.",
)
@click.option(
    "--extra-deps",
    type=str,
    default=None,
    help="Comma-separated extra pip packages to install in every venv.",
)
@click.option(
    "--test-script",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Custom Python script to run instead of the package's test_command.",
)
@click.option(
    "--no-skip-existing",
    is_flag=True,
    default=False,
    help="Re-run packages even if tsan_report.txt already exists.",
)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output.")
def tsan_run(
    target_python: Path,
    registry_dir: Path | None,
    packages_csv: str | None,
    output_dir: Path,
    repos_dir: Path | None,
    venvs_dir: Path | None,
    timeout: int,
    installer: str,
    suppressions: Path | None,
    quick: bool,
    stress: int,
    extra_deps: str | None,
    test_script: Path | None,
    no_skip_existing: bool,
    verbose: bool,
) -> None:
    """Run extension test suites under ThreadSanitizer.

    Installs extensions into a TSan-enabled free-threaded Python venv,
    runs their test suites, and captures data race reports from TSan.
    Output is designed for use with ft-review-toolkit's tsan-report-analyzer.

    \b
    Requirements:
        - A free-threaded Python built with --with-thread-sanitizer
        - llvm-symbolizer on PATH (for readable stack traces)

    \b
    Examples:
        # Test specific packages
        labeille tsan-run --target-python ~/tsan-python/python \\
            --packages numpy,lxml

        # Test from registry with stress mode
        labeille tsan-run --target-python ~/tsan-python/python \\
            --registry-dir ~/laruche --stress 3

        # Quick CI check (halt on first race)
        labeille tsan-run --target-python ~/tsan-python/python \\
            --packages myext --quick

        # Run a custom stress test script instead of the test suite
        labeille tsan-run --target-python ~/tsan-python/python \\
            --packages myext --test-script stress_test.py
    """
    setup_logging(verbose=verbose)

    from labeille.tsan_run import TsanRunConfig, run_tsan_batch

    packages_filter = (
        [p.strip() for p in packages_csv.split(",") if p.strip()] if packages_csv else None
    )

    if not registry_dir and not packages_filter:
        click.echo("Error: provide --registry-dir, --packages, or both.", err=True)
        sys.exit(1)

    parsed_extra_deps = (
        [d.strip() for d in extra_deps.split(",") if d.strip()] if extra_deps else []
    )

    config = TsanRunConfig(
        target_python=target_python,
        output_dir=output_dir,
        registry_dir=registry_dir,
        repos_dir=repos_dir,
        venvs_dir=venvs_dir,
        packages_filter=packages_filter,
        timeout=timeout,
        installer=installer,
        suppressions=suppressions,
        quick=quick,
        stress=stress,
        extra_deps=parsed_extra_deps,
        skip_if_exists=not no_skip_existing,
        verbose=verbose,
        test_script=test_script,
    )

    meta, results = run_tsan_batch(config)

    # Print summary.
    click.echo("")
    click.echo(f"TSan run complete: {meta.run_id}")
    click.echo(f"  Python:             {meta.python_version}")
    click.echo(f"  Free-threaded:      {meta.is_free_threaded}")
    click.echo(f"  TSan instrumented:  {meta.is_tsan}")
    click.echo(f"  Packages tested:    {meta.total_packages}")
    click.echo(f"  Packages with races: {meta.packages_with_races}")
    click.echo(f"  Total race reports: {meta.total_races}")

    if meta.quick_mode:
        click.echo("  Mode:               quick (halt on first error)")
    if meta.stress_count > 1:
        click.echo(f"  Stress iterations:  {meta.stress_count}")

    # Per-status breakdown.
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    for status, count in sorted(status_counts.items()):
        click.echo(f"  {status}: {count}")

    # List packages with races.
    race_results = [r for r in results if r.race_count > 0]
    if race_results:
        click.echo("")
        click.echo("Packages with data races:")
        for r in sorted(race_results, key=lambda x: -x.race_count):
            click.echo(f"  {r.package}: {r.race_count} reports -> {r.report_path}")

    click.echo(f"\nOutput: {config.output_dir / meta.run_id}")
