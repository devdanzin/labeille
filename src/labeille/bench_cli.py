"""CLI commands for labeille bench.

Subcommands:
    labeille bench run       Execute a benchmark
    labeille bench show      Display a benchmark's results
    labeille bench compare   Compare two or more benchmark results
    labeille bench system    Print system characterization
    labeille bench export    Export results to CSV/markdown
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from labeille.bench.results import BenchMeta, BenchPackageResult

log = logging.getLogger("labeille")


@click.group()
def bench() -> None:
    """Benchmark package test suites under different conditions."""


# ---------------------------------------------------------------------------
# bench run
# ---------------------------------------------------------------------------


@bench.command()
@click.option(
    "--profile",
    "profile_path",
    type=click.Path(exists=True),
    help="YAML profile defining benchmark conditions.",
)
@click.option(
    "--condition",
    "inline_conditions",
    type=str,
    multiple=True,
    help="Inline condition: 'name:key=value,...' (repeatable).",
)
@click.option(
    "--target-python",
    type=click.Path(exists=True),
    help="Default target Python interpreter.",
)
@click.option(
    "--iterations",
    type=int,
    default=None,
    help="Measured iterations (default: 5, min: 3).",
)
@click.option(
    "--warmup",
    type=int,
    default=None,
    help="Warm-up iterations (default: 1).",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Per-iteration timeout in seconds (default: 600).",
)
@click.option(
    "--alternate/--no-alternate",
    default=None,
    help="Alternate conditions per package.",
)
@click.option(
    "--interleave",
    is_flag=True,
    default=False,
    help="Interleave packages across iterations.",
)
@click.option(
    "--packages",
    type=str,
    default=None,
    help="Comma-separated package filter.",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=None,
    help="Top N packages by download count.",
)
@click.option(
    "--extra-deps",
    type=str,
    default=None,
    help="Comma-separated extra deps for all conditions.",
)
@click.option(
    "--test-command-suffix",
    type=str,
    default=None,
    help="Append to all test commands.",
)
@click.option(
    "--registry-dir",
    type=click.Path(exists=True),
    required=True,
    help="Registry directory.",
)
@click.option(
    "--repos-dir",
    type=click.Path(),
    default=None,
    help="Persistent repos directory.",
)
@click.option(
    "--venvs-dir",
    type=click.Path(),
    default=None,
    help="Persistent venvs directory.",
)
@click.option(
    "--work-dir",
    type=click.Path(),
    default=None,
    help="Sets both repos and venvs dirs.",
)
@click.option(
    "--results-dir",
    type=click.Path(),
    default="results",
    help="Results output directory.",
)
@click.option(
    "--name",
    type=str,
    default=None,
    help="Human-readable benchmark name.",
)
@click.option(
    "--check-stability",
    is_flag=True,
    default=False,
    help="Check system stability before starting.",
)
@click.option(
    "--wait-for-stability",
    is_flag=True,
    default=False,
    help="Wait for system to stabilize.",
)
@click.option(
    "--quick",
    is_flag=True,
    default=False,
    help="Quick mode: 3 iterations, no warmup, top 20.",
)
@click.option(
    "--env",
    "env_pairs",
    type=str,
    multiple=True,
    help="KEY=VALUE env var for all conditions (repeatable).",
)
@click.option("-v", "--verbose", is_flag=True, default=False)
def run(  # noqa: PLR0913
    profile_path: str | None,
    inline_conditions: tuple[str, ...],
    target_python: str | None,
    iterations: int | None,
    warmup: int | None,
    timeout: int | None,
    alternate: bool | None,
    interleave: bool,
    packages: str | None,
    top_n: int | None,
    extra_deps: str | None,
    test_command_suffix: str | None,
    registry_dir: str,
    repos_dir: str | None,
    venvs_dir: str | None,
    work_dir: str | None,
    results_dir: str,
    name: str | None,
    check_stability: bool,
    wait_for_stability: bool,
    quick: bool,
    env_pairs: tuple[str, ...],
    verbose: bool,
) -> None:
    """Run a benchmark across packages under specified conditions.

    Use --profile for a YAML profile or --condition for inline
    conditions.

    \b
    Examples:
        # From a YAML profile
        labeille bench run --profile bench-coverage.yaml \\
            --registry-dir ./registry --target-python /usr/bin/python3

        # Quick inline comparison
        labeille bench run \\
            --condition "baseline:" \\
            --condition "coverage:extra_deps=coverage,test_prefix=coverage run -m" \\
            --target-python /usr/bin/python3 \\
            --registry-dir ./registry --packages requests,click

        # Quick mode for development
        labeille bench run --profile bench-jit.yaml \\
            --registry-dir ./registry --quick
    """
    from labeille.bench.config import (
        BenchConfig,
        config_from_profile,
        load_profile,
        parse_inline_condition,
    )
    from labeille.bench.runner import BenchRunner, quick_config

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Build configuration.
    cli_overrides: dict[str, object] = {
        "target_python": target_python,
        "iterations": iterations,
        "warmup": warmup,
        "timeout": timeout,
        "registry_dir": registry_dir,
        "repos_dir": repos_dir or (work_dir and str(Path(work_dir) / "repos")),
        "venvs_dir": venvs_dir or (work_dir and str(Path(work_dir) / "venvs")),
        "results_dir": results_dir,
        "name": name,
        "check_stability": check_stability,
        "wait_for_stability": wait_for_stability,
        "alternate": alternate,
        "interleave": interleave,
        "packages_filter": (
            [p.strip() for p in packages.split(",") if p.strip()] if packages else None
        ),
        "top_n": top_n,
    }

    if profile_path:
        profile_data = load_profile(Path(profile_path))
        config = config_from_profile(profile_data, cli_overrides=cli_overrides)
    else:
        config = BenchConfig(
            name=name or "",
            iterations=iterations or 5,
            warmup=warmup if warmup is not None else 1,
            timeout=timeout or 600,
            default_target_python=target_python or "",
            registry_dir=Path(registry_dir),
        )
        if repos_dir or work_dir:
            config.repos_dir = Path(repos_dir or str(Path(work_dir) / "repos"))  # type: ignore[arg-type]
        if venvs_dir or work_dir:
            config.venvs_dir = Path(venvs_dir or str(Path(work_dir) / "venvs"))  # type: ignore[arg-type]
        config.results_dir = Path(results_dir)
        if packages:
            config.packages_filter = [p.strip() for p in packages.split(",") if p.strip()]
        if top_n:
            config.top_n = top_n

    # Add inline conditions.
    for spec in inline_conditions:
        cond = parse_inline_condition(spec)
        config.conditions[cond.name] = cond

    # Apply shared options.
    if extra_deps:
        config.default_extra_deps = [d.strip() for d in extra_deps.split(",") if d.strip()]
    if test_command_suffix:
        config.default_test_command_suffix = test_command_suffix

    # Parse env vars.
    for pair in env_pairs:
        if "=" in pair:
            k, v = pair.split("=", 1)
            config.default_env[k] = v

    config.check_stability = check_stability
    config.wait_for_stability = wait_for_stability
    config.cli_args = sys.argv[1:]

    if quick:
        config = quick_config(config)

    # Run.
    runner = BenchRunner(config)
    try:
        meta, results = runner.run()
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        click.echo("\nBenchmark interrupted.", err=True)
        raise SystemExit(130)  # noqa: B904

    # Display results.
    from labeille.bench.display import format_bench_show

    click.echo()
    click.echo(format_bench_show(meta, results))
    click.echo()
    click.echo(f"Results saved to: {config.output_dir}")


# ---------------------------------------------------------------------------
# bench show
# ---------------------------------------------------------------------------


@bench.command("show")
@click.argument("result_dir", type=click.Path(exists=True))
def show(result_dir: str) -> None:
    """Display results from a benchmark run.

    RESULT_DIR is the path to a benchmark output directory
    containing bench_meta.json and bench_results.jsonl.
    """
    from labeille.bench.display import format_bench_show
    from labeille.bench.results import load_bench_run

    meta, results = load_bench_run(Path(result_dir))
    click.echo(format_bench_show(meta, results))


# ---------------------------------------------------------------------------
# bench compare
# ---------------------------------------------------------------------------


@bench.command("compare")
@click.argument(
    "result_dirs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--baseline",
    type=str,
    default=None,
    help="Name of the baseline condition (default: first).",
)
@click.option(
    "--metric",
    type=click.Choice(["wall", "cpu", "rss"]),
    default="wall",
    help="Metric to compare.",
)
def compare(result_dirs: tuple[str, ...], baseline: str | None, metric: str) -> None:
    """Compare results from two or more benchmark runs.

    Accepts either:
    - Two result directories from different benchmark runs
    - One result directory with multiple conditions

    \b
    Examples:
        # Compare conditions within a single run
        labeille bench compare results/bench_20260227_143000

        # Compare two separate runs
        labeille bench compare results/bench_baseline results/bench_jit
    """
    from labeille.bench.display import (
        format_bench_show,
        format_comparison_summary,
    )
    from labeille.bench.results import load_bench_run

    if len(result_dirs) == 1:
        # Single directory: compare conditions within the run.
        meta, results = load_bench_run(Path(result_dirs[0]))
        conditions = list(meta.conditions.keys())
        if len(conditions) < 2:
            click.echo(
                "Error: Single benchmark run has only one condition. "
                "Provide two result directories for cross-run comparison.",
                err=True,
            )
            raise SystemExit(1)

        baseline_name = baseline or conditions[0]
        if baseline_name not in conditions:
            click.echo(
                f"Error: Baseline '{baseline_name}' not found. Available: {', '.join(conditions)}",
                err=True,
            )
            raise SystemExit(1)

        click.echo(format_bench_show(meta, results))
        click.echo()

        for cond_name in conditions:
            if cond_name == baseline_name:
                continue
            click.echo(f"\n{baseline_name} vs {cond_name}")
            click.echo("=" * (len(baseline_name) + len(cond_name) + 4))
            click.echo(format_comparison_summary(results, baseline_name, cond_name))
    else:
        # Multiple directories: cross-run comparison.
        all_runs: list[tuple[BenchMeta, list[BenchPackageResult]]] = []
        for rd in result_dirs:
            meta, results = load_bench_run(Path(rd))
            all_runs.append((meta, results))

        click.echo("Cross-run comparison:")
        click.echo()
        for i, (meta, results) in enumerate(all_runs):
            run_name = meta.name or meta.bench_id
            click.echo(f"  Run {i + 1}: {run_name}")
            click.echo(f"    Conditions: {', '.join(meta.conditions.keys())}")
            click.echo(f"    Packages: {meta.packages_completed}")
        click.echo()

        # Merge: create synthetic BenchPackageResults with conditions
        # named after each run.
        merged_results: dict[str, BenchPackageResult] = {}
        run_names: list[str] = []

        for meta, results in all_runs:
            run_name = meta.name or meta.bench_id
            run_names.append(run_name)
            first_cond = list(meta.conditions.keys())[0]

            for r in results:
                if r.skipped:
                    continue
                cond = r.conditions.get(first_cond)
                if not cond:
                    continue
                if r.package not in merged_results:
                    merged_results[r.package] = BenchPackageResult(
                        package=r.package,
                    )
                merged_results[r.package].conditions[run_name] = cond

        merged_list = list(merged_results.values())

        if len(run_names) >= 2:
            baseline_name = baseline or run_names[0]
            for treatment in run_names[1:]:
                click.echo(f"\n{baseline_name} vs {treatment}")
                click.echo("=" * (len(baseline_name) + len(treatment) + 4))
                click.echo(format_comparison_summary(merged_list, baseline_name, treatment))


# ---------------------------------------------------------------------------
# bench system
# ---------------------------------------------------------------------------


@bench.command("system")
@click.option(
    "--target-python",
    type=click.Path(exists=True),
    default=None,
    help="Also profile a target Python.",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def system_cmd(target_python: str | None, as_json: bool) -> None:
    """Print system characterization for benchmark documentation."""
    from labeille.bench.system import (
        capture_python_profile,
        capture_system_profile,
        format_python_profile,
        format_system_profile,
    )

    profile = capture_system_profile()

    if as_json:
        import json

        data = profile.to_dict()
        if target_python:
            pp = capture_python_profile(Path(target_python))
            data["target_python"] = pp.to_dict()
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(format_system_profile(profile))
        if target_python:
            click.echo()
            pp = capture_python_profile(Path(target_python))
            click.echo(format_python_profile(pp))


# ---------------------------------------------------------------------------
# bench export
# ---------------------------------------------------------------------------


@bench.command("export")
@click.argument("result_dir", type=click.Path(exists=True))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "csv-summary", "markdown"]),
    default="csv",
    help="Export format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file (default: stdout).",
)
def export(result_dir: str, fmt: str, output: str | None) -> None:
    """Export benchmark results to CSV or Markdown.

    \b
    Examples:
        labeille bench export results/bench_001 --format csv > data.csv
        labeille bench export results/bench_001 --format csv-summary > summary.csv
        labeille bench export results/bench_001 --format markdown -o report.md
    """
    from labeille.bench.export import export_csv, export_csv_summary, export_markdown
    from labeille.bench.results import load_bench_run

    meta, results = load_bench_run(Path(result_dir))

    if fmt == "csv":
        text = export_csv(meta, results)
    elif fmt == "csv-summary":
        text = export_csv_summary(meta, results)
    else:
        text = export_markdown(meta, results)

    if output:
        Path(output).write_text(text)
        click.echo(f"Exported to {output}")
    else:
        click.echo(text)
