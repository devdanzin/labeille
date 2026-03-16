"""CLI commands for labeille bench.

Subcommands:
    labeille bench run       Execute a benchmark
    labeille bench show      Display a benchmark's results
    labeille bench compare   Compare two or more benchmark results
    labeille bench system    Print system characterization
    labeille bench export    Export results to CSV/markdown
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from labeille.bench.config import BenchConfig

from labeille.bench.results import BenchMeta, BenchPackageResult
from labeille.cli_utils import parse_csv_list, parse_env_pairs
from labeille.logging import setup_logging


@click.group()
def bench() -> None:
    """Benchmark package test suites under different conditions."""


# ---------------------------------------------------------------------------
# bench run — helpers
# ---------------------------------------------------------------------------


def _build_base_config(ctx: click.Context) -> BenchConfig:
    """Build a BenchConfig from Click context parameters.

    Creates the initial config from either a YAML profile or CLI defaults,
    then applies inline conditions.
    """
    from labeille.bench.config import (
        BenchConfig,
        config_from_profile,
        load_profile,
        parse_inline_condition,
    )
    from labeille.registry import default_registry_dir

    p = ctx.params
    registry_dir: Path | None = p["registry_dir"]
    if registry_dir is None:
        registry_dir = default_registry_dir()

    repos_dir: Path | None = p["repos_dir"]
    venvs_dir: Path | None = p["venvs_dir"]
    work_dir: Path | None = p["work_dir"]
    target_python: Path | None = p["target_python"]
    packages: str | None = p["packages"]

    if p["profile_path"]:
        cli_overrides: dict[str, object] = {
            "target_python": str(target_python) if target_python else None,
            "iterations": p["iterations"],
            "warmup": p["warmup"],
            "timeout": p["timeout"],
            "registry_dir": registry_dir,
            "repos_dir": repos_dir or (work_dir and work_dir / "repos"),
            "venvs_dir": venvs_dir or (work_dir and work_dir / "venvs"),
            "results_dir": p["results_dir"],
            "name": p["name"],
            "check_stability": p["check_stability"],
            "wait_for_stability": p["wait_for_stability"],
            "alternate": p["alternate"],
            "interleave": p["interleave"],
            "packages_filter": parse_csv_list(packages) or None,
            "top_n": p["top_n"],
            "adaptive": p["adaptive"] or None,
            "adaptive_threshold": p["adaptive_threshold"],
            "adaptive_min_iterations": p["adaptive_min_iterations"],
        }
        profile_data = load_profile(p["profile_path"])
        config = config_from_profile(profile_data, cli_overrides=cli_overrides)
    else:
        config = BenchConfig(
            name=p["name"] or "",
            iterations=p["iterations"] or 5,
            warmup=p["warmup"] if p["warmup"] is not None else 1,
            timeout=p["timeout"] or 600,
            default_target_python=str(target_python) if target_python else "",
            registry_dir=registry_dir,
        )
        if repos_dir:
            config.repos_dir = repos_dir
        elif work_dir:
            config.repos_dir = work_dir / "repos"
        if venvs_dir:
            config.venvs_dir = venvs_dir
        elif work_dir:
            config.venvs_dir = work_dir / "venvs"
        config.results_dir = p["results_dir"]
        if packages:
            config.packages_filter = parse_csv_list(packages)
        if p["top_n"]:
            config.top_n = p["top_n"]

    for spec in p["inline_conditions"]:
        cond = parse_inline_condition(spec)
        config.conditions[cond.name] = cond

    return config


def _apply_config_overrides(config: BenchConfig, ctx: click.Context) -> BenchConfig:
    """Apply shared CLI overrides and finalize a BenchConfig.

    Handles test overrides, env vars, adaptive settings, resource
    constraints, and quick mode.
    """
    from labeille.bench.runner import quick_config

    p = ctx.params

    if p["extra_deps"]:
        config.default_extra_deps = parse_csv_list(p["extra_deps"])
    if p["test_command_suffix"]:
        config.default_test_command_suffix = p["test_command_suffix"]

    config.default_env.update(parse_env_pairs(p["env_pairs"]))

    config.installer = p["installer"]
    if p["adaptive"]:
        config.adaptive = True
    if p["adaptive_threshold"] is not None:
        config.adaptive_threshold = p["adaptive_threshold"]
    if p["adaptive_min_iterations"] is not None:
        config.adaptive_min_iterations = p["adaptive_min_iterations"]
    config.check_stability = p["check_stability"]
    config.wait_for_stability = p["wait_for_stability"]
    config.per_test_timing = p["per_test_timing"]
    config.drop_caches = p["drop_caches"] or p["warm_vs_cold"]
    config.warm_vs_cold = p["warm_vs_cold"]
    config.run_dangerously_as_root = p["run_dangerously_as_root"]

    if p["memory_limit"] or p["cpu_affinity"] or p["cpu_time_limit"]:
        from labeille.bench.constraints import ResourceConstraints

        cpu_affinity: str | None = p["cpu_affinity"]
        affinity_list = [int(c.strip()) for c in cpu_affinity.split(",")] if cpu_affinity else None
        config.default_constraints = ResourceConstraints(
            memory_limit_mb=p["memory_limit"],
            cpu_affinity=affinity_list,
            cpu_time_limit_s=p["cpu_time_limit"],
        )

    config.cli_args = sys.argv[1:]

    if p["quick"]:
        config = quick_config(config)

    return config


# ---------------------------------------------------------------------------
# bench run — command
# ---------------------------------------------------------------------------


# -- Profile and conditions
@bench.command()
@click.option(
    "--profile",
    "profile_path",
    type=click.Path(exists=True, path_type=Path),
    help="YAML profile defining benchmark conditions.",
)
@click.option(
    "--condition",
    "inline_conditions",
    type=str,
    multiple=True,
    help="Inline condition: 'name:key=value,...' (repeatable).",
)
# -- Execution
@click.option(
    "--target-python",
    type=click.Path(exists=True, path_type=Path),
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
    show_default=True,
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
# -- Package selection
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
# -- Test overrides
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
    "--env",
    "env_pairs",
    type=str,
    multiple=True,
    help="KEY=VALUE env var for all conditions (repeatable).",
)
# -- Paths
@click.option(
    "--registry-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Registry directory (default: ~/.local/share/labeille/registry/).",
)
@click.option(
    "--repos-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Persistent repos directory.",
)
@click.option(
    "--venvs-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Persistent venvs directory.",
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Sets both repos and venvs dirs.",
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default="results",
    help="Results output directory.",
)
# -- Run identity and modes
@click.option(
    "--name",
    type=str,
    default=None,
    help="Human-readable benchmark name.",
)
@click.option(
    "--quick",
    is_flag=True,
    default=False,
    help="Quick mode: 3 iterations, no warmup, top 20, adaptive.",
)
# -- Stability
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
# -- Adaptive convergence
@click.option(
    "--adaptive",
    is_flag=True,
    default=False,
    help="Stop iterating early when measurements converge (RSE below threshold).",
)
@click.option(
    "--adaptive-threshold",
    type=float,
    default=None,
    help="RSE convergence threshold (default: 0.005 = 0.5%%).",
)
@click.option(
    "--adaptive-min-iterations",
    type=int,
    default=None,
    help="Minimum measured iterations before convergence check (default: 5).",
)
# -- Advanced: timing, resources, caches
@click.option(
    "--per-test-timing",
    is_flag=True,
    default=False,
    help="Capture per-test timing via pytest --durations=0.",
)
@click.option(
    "--memory-limit",
    type=int,
    default=None,
    help="Memory limit in MB for all conditions (ulimit -v).",
)
@click.option(
    "--cpu-affinity",
    type=str,
    default=None,
    help="CPU core list (e.g. '0,1') for all conditions (taskset).",
)
@click.option(
    "--cpu-time-limit",
    type=int,
    default=None,
    help="CPU time limit in seconds for all conditions (ulimit -t).",
)
@click.option(
    "--drop-caches",
    is_flag=True,
    default=False,
    help="Drop filesystem caches between iterations (requires setup, see docs).",
)
@click.option(
    "--warm-vs-cold",
    is_flag=True,
    default=False,
    help="Run with and without cache dropping, compare results.",
)
@click.option(
    "--run-dangerously-as-root",
    is_flag=True,
    default=False,
    help="Allow running as root (for containers). Not recommended.",
)
@click.option(
    "--installer",
    type=click.Choice(["auto", "uv", "pip"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Package installer backend. 'auto' uses uv if available.",
)
@click.option("-v", "--verbose", is_flag=True, default=False)
@click.pass_context
def run(ctx: click.Context, /, **_kwargs: Any) -> None:
    """Run a benchmark across packages under specified conditions.

    Use --profile for a YAML profile or --condition for inline
    conditions.

    \b
    Examples:
        # From a YAML profile
        labeille bench run --profile bench-coverage.yaml \\
            --target-python /usr/bin/python3

        # Quick inline comparison
        labeille bench run \\
            --condition "baseline:" \\
            --condition "coverage:extra_deps=coverage,test_prefix=coverage run -m" \\
            --target-python /usr/bin/python3 --packages requests,click

        # Quick mode for development
        labeille bench run --profile bench-jit.yaml --quick
    """
    setup_logging(verbose=ctx.params["verbose"])

    config = _build_base_config(ctx)
    config = _apply_config_overrides(config, ctx)

    from labeille.bench.runner import BenchRunner

    runner = BenchRunner(config)
    try:
        meta, results = runner.run()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        click.echo("\nBenchmark interrupted.", err=True)
        raise SystemExit(130) from None

    from labeille.bench.display import format_bench_show

    click.echo()
    click.echo(format_bench_show(meta, results))
    click.echo()
    click.echo(f"Results saved to: {config.output_dir}")


# ---------------------------------------------------------------------------
# bench show
# ---------------------------------------------------------------------------


@bench.command("show")
@click.argument("result_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--anomalies", is_flag=True, default=False, help="Show measurement anomalies.")
@click.option(
    "--per-test",
    "per_test_package",
    type=str,
    default=None,
    help="Show per-test timing for a specific package.",
)
def show(result_dir: Path, anomalies: bool, per_test_package: str | None) -> None:
    """Display results from a benchmark run.

    RESULT_DIR is the path to a benchmark output directory
    containing bench_meta.json and bench_results.jsonl.
    """
    from labeille.bench.display import format_bench_show
    from labeille.bench.results import load_bench_run

    meta, results = load_bench_run(result_dir)
    click.echo(format_bench_show(meta, results))

    if per_test_package:
        from labeille.bench.display import format_per_test_summary

        pkg_result = next((r for r in results if r.package == per_test_package), None)
        if not pkg_result:
            click.echo(f"\nPackage '{per_test_package}' not found.", err=True)
        else:
            # Use the last measured iteration's timings.
            for cond_name, cond in pkg_result.conditions.items():
                for it in reversed(cond.measured_iterations):
                    if it.per_test_timings:
                        click.echo(f"\n{cond_name}:")
                        click.echo(format_per_test_summary(it.per_test_timings))
                        break

    if anomalies:
        from labeille.bench.anomaly import detect_anomalies
        from labeille.bench.display import format_anomaly_report

        report = detect_anomalies(results)
        text = format_anomaly_report(report)
        if text:
            click.echo()
            click.echo(text)


# ---------------------------------------------------------------------------
# bench compare
# ---------------------------------------------------------------------------


@bench.command("compare")
@click.argument(
    "result_dirs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
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
@click.option(
    "--per-test",
    "per_test_package",
    type=str,
    default=None,
    help="Show per-test overhead for a specific package.",
)
def compare(
    result_dirs: tuple[Path, ...],
    baseline: str | None,
    metric: str,
    per_test_package: str | None,
) -> None:
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
    if len(result_dirs) == 1:
        _compare_intra_run(result_dirs[0], baseline, per_test_package)
    else:
        _compare_cross_run(result_dirs, baseline)


def _compare_intra_run(
    result_dir: Path,
    baseline: str | None,
    per_test_package: str | None,
) -> None:
    """Compare conditions within a single benchmark run."""
    from labeille.bench.display import (
        format_bench_show,
        format_comparison_summary,
    )
    from labeille.bench.results import load_bench_run

    meta, results = load_bench_run(result_dir)
    conditions = list(meta.conditions.keys())
    if len(conditions) < 2:
        raise click.ClickException(
            "Single benchmark run has only one condition. "
            "Provide two result directories for cross-run comparison."
        )

    baseline_name = baseline or conditions[0]
    if baseline_name not in conditions:
        raise click.ClickException(
            f"Baseline '{baseline_name}' not found. Available: {', '.join(conditions)}"
        )

    click.echo(format_bench_show(meta, results))
    click.echo()

    for cond_name in conditions:
        if cond_name == baseline_name:
            continue
        click.echo(f"\n{baseline_name} vs {cond_name}")
        click.echo("=" * (len(baseline_name) + len(cond_name) + 4))
        click.echo(format_comparison_summary(results, baseline_name, cond_name))

    if per_test_package:
        from labeille.bench.compare import compare_per_test
        from labeille.bench.display import format_per_test_comparison

        for cond_name in conditions:
            if cond_name == baseline_name:
                continue
            overheads = compare_per_test(results, baseline_name, cond_name, per_test_package)
            if overheads:
                click.echo()
                click.echo(format_per_test_comparison(overheads))

    _report_anomalies(results)


def _compare_cross_run(
    result_dirs: tuple[Path, ...],
    baseline: str | None,
) -> None:
    """Compare results across multiple benchmark runs."""
    from labeille.bench.display import format_comparison_summary
    from labeille.bench.results import load_bench_run

    all_runs: list[tuple[BenchMeta, list[BenchPackageResult]]] = []
    for rd in result_dirs:
        meta, results = load_bench_run(rd)
        all_runs.append((meta, results))

    click.echo("Cross-run comparison:")
    click.echo()
    for i, (meta, results) in enumerate(all_runs):
        run_name = meta.name or meta.bench_id
        click.echo(f"  Run {i + 1}: {run_name}")
        click.echo(f"    Conditions: {', '.join(meta.conditions.keys())}")
        click.echo(f"    Packages: {meta.packages_completed}")
    click.echo()

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

        _report_anomalies(merged_list)


def _report_anomalies(results: list[BenchPackageResult]) -> None:
    """Print an anomaly summary if any anomalies are detected."""
    from labeille.bench.anomaly import detect_anomalies

    anomaly_report = detect_anomalies(results)
    if anomaly_report.anomalies:
        n_pkgs = len(anomaly_report.affected_packages)
        click.echo(
            f"\n\u26a0 {n_pkgs} package(s) have measurement anomalies "
            f"(use 'bench show --anomalies' for details)."
        )


# ---------------------------------------------------------------------------
# bench system
# ---------------------------------------------------------------------------


@bench.command("system")
@click.option(
    "--target-python",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Also profile a target Python.",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def system_cmd(target_python: Path | None, as_json: bool) -> None:
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
            pp = capture_python_profile(target_python)
            data["target_python"] = pp.to_dict()
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(format_system_profile(profile))
        if target_python:
            click.echo()
            pp = capture_python_profile(target_python)
            click.echo(format_python_profile(pp))


# ---------------------------------------------------------------------------
# bench export
# ---------------------------------------------------------------------------


@bench.command("export")
@click.argument("result_dir", type=click.Path(exists=True, path_type=Path))
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
    type=click.Path(path_type=Path),
    default=None,
    help="Output file (default: stdout).",
)
def export(result_dir: Path, fmt: str, output: Path | None) -> None:
    """Export benchmark results to CSV or Markdown.

    \b
    Examples:
        labeille bench export results/bench_001 --format csv > data.csv
        labeille bench export results/bench_001 --format csv-summary > summary.csv
        labeille bench export results/bench_001 --format markdown -o report.md
    """
    from labeille.bench.export import export_csv, export_csv_summary, export_markdown
    from labeille.bench.results import load_bench_run

    meta, results = load_bench_run(result_dir)

    if fmt == "csv":
        text = export_csv(meta, results)
    elif fmt == "csv-summary":
        text = export_csv_summary(meta, results)
    else:
        from labeille.bench.anomaly import detect_anomalies

        anomaly_report = detect_anomalies(results)
        text = export_markdown(meta, results, anomaly_report=anomaly_report)

    if output:
        output.write_text(text, encoding="utf-8")
        click.echo(f"Exported to {output}")
    else:
        click.echo(text)


# ---------------------------------------------------------------------------
# bench setup-cache-drop
# ---------------------------------------------------------------------------


@bench.command("setup-cache-drop")
@click.option(
    "--show-script",
    is_flag=True,
    default=False,
    help="Print the helper script contents (for piping to a file).",
)
def setup_cache_drop(show_script: bool) -> None:
    """Show setup instructions for filesystem cache dropping.

    Prints the helper script contents and sudoers configuration
    needed for --drop-caches to work.
    """
    from labeille.bench.cache import (
        check_cache_drop_available,
        format_setup_instructions,
        generate_drop_caches_script,
    )

    if show_script:
        click.echo(generate_drop_caches_script(), nl=False)
        return

    status = check_cache_drop_available()
    if status.available:
        click.echo("Cache dropping is already configured and working.")
    else:
        click.echo(format_setup_instructions())


# ---------------------------------------------------------------------------
# bench track (subgroup)
# ---------------------------------------------------------------------------


@bench.group("track")
def track() -> None:
    """Manage benchmark tracking series for longitudinal comparison."""


@track.command("init")
@click.argument("series_name")
@click.option("--description", "-d", default="", help="Series description.")
@click.option(
    "--tracking-dir",
    type=click.Path(path_type=Path),
    default="results/tracking",
    help="Parent directory for tracking series.",
)
def track_init(series_name: str, description: str, tracking_dir: Path) -> None:
    """Create a new benchmark tracking series."""
    from labeille.bench.tracking import init_series

    try:
        series = init_series(tracking_dir, series_name, description=description)
        click.echo(f"Created tracking series '{series.series_id}'.")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@track.command("add")
@click.argument("series_name")
@click.argument("bench_run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--notes", "-n", default="", help="Notes for this run.")
@click.option(
    "--commit",
    type=str,
    multiple=True,
    help="key=value commit info (repeatable).",
)
@click.option(
    "--tracking-dir",
    type=click.Path(path_type=Path),
    default="results/tracking",
    help="Parent directory for tracking series.",
)
def track_add(
    series_name: str,
    bench_run_dir: Path,
    notes: str,
    commit: tuple[str, ...],
    tracking_dir: Path,
) -> None:
    """Add a benchmark run to a tracking series."""
    from labeille.bench.tracking import add_run_to_series

    commit_info: dict[str, str] = {}
    for pair in commit:
        if "=" in pair:
            k, v = pair.split("=", 1)
            commit_info[k] = v

    series_dir = tracking_dir / series_name
    try:
        entry = add_run_to_series(
            series_dir,
            bench_run_dir,
            notes=notes,
            commit_info=commit_info,
        )
        click.echo(f"Added run '{entry.bench_id}' to series '{series_name}'.")
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


@track.command("show")
@click.argument("series_name")
@click.option("--last", "last_n", type=int, default=None, help="Show only last N runs.")
@click.option(
    "--tracking-dir",
    type=click.Path(path_type=Path),
    default="results/tracking",
    help="Parent directory for tracking series.",
)
def track_show(series_name: str, last_n: int | None, tracking_dir: Path) -> None:
    """Show runs in a tracking series."""
    from labeille.bench.tracking import load_series

    series_dir = tracking_dir / series_name
    try:
        series = load_series(series_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Series: {series.series_id}")
    if series.description:
        click.echo(f"  {series.description}")
    click.echo(f"  Created: {series.created}")
    click.echo(f"  Config fingerprint: {series.config_fingerprint or '(none yet)'}")
    if series.pinned_baseline_id:
        click.echo(f"  Pinned baseline: {series.pinned_baseline_id}")
    click.echo()

    runs = series.runs
    if last_n and last_n < len(runs):
        runs = runs[-last_n:]

    if not runs:
        click.echo("  No runs yet.")
        return

    # Table header.
    click.echo(f"  {'#':>3}  {'Date':20s}  {'Bench ID':30s}  {'Pkgs':>5}  Notes")
    click.echo(f"  {'---':>3}  {'----':20s}  {'--------':30s}  {'----':>5}  -----")
    for i, run in enumerate(runs, 1):
        date_str = run.timestamp[:19] if run.timestamp else "unknown"
        baseline_marker = " *" if run.bench_id == series.pinned_baseline_id else ""
        click.echo(
            f"  {i:3d}  {date_str:20s}  {run.bench_id:30s}  "
            f"{run.packages_completed:5d}  {run.notes}{baseline_marker}"
        )


@track.command("pin")
@click.argument("series_name")
@click.argument("bench_id")
@click.option(
    "--tracking-dir",
    type=click.Path(path_type=Path),
    default="results/tracking",
    help="Parent directory for tracking series.",
)
def track_pin(series_name: str, bench_id: str, tracking_dir: Path) -> None:
    """Pin a run as the baseline for trend analysis."""
    from labeille.bench.tracking import pin_baseline

    series_dir = tracking_dir / series_name
    try:
        pin_baseline(series_dir, bench_id)
        click.echo(f"Pinned '{bench_id}' as baseline for series '{series_name}'.")
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@track.command("unpin")
@click.argument("series_name")
@click.option(
    "--tracking-dir",
    type=click.Path(path_type=Path),
    default="results/tracking",
    help="Parent directory for tracking series.",
)
def track_unpin(series_name: str, tracking_dir: Path) -> None:
    """Remove the pinned baseline."""
    from labeille.bench.tracking import unpin_baseline

    series_dir = tracking_dir / series_name
    try:
        unpin_baseline(series_dir)
        click.echo(f"Unpinned baseline for series '{series_name}'.")
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


@track.command("list")
@click.option(
    "--tracking-dir",
    type=click.Path(path_type=Path),
    default="results/tracking",
    help="Parent directory for tracking series.",
)
def track_list(tracking_dir: Path) -> None:
    """List all tracking series."""
    from labeille.bench.tracking import list_series

    all_series = list_series(tracking_dir)
    if not all_series:
        click.echo("No tracking series found.")
        return

    click.echo(f"{'Name':25s}  {'Runs':>5}  {'Date Range':43s}  Description")
    click.echo(f"{'----':25s}  {'----':>5}  {'----------':43s}  -----------")
    for s in all_series:
        dr = s.date_range
        date_str = f"{dr[0][:19]} .. {dr[1][:19]}" if dr else "no runs"
        click.echo(f"{s.series_id:25s}  {s.n_runs:5d}  {date_str:43s}  {s.description}")


@track.command("trend")
@click.argument("series_name")
@click.option("--condition", type=str, default=None, help="Condition to analyze.")
@click.option(
    "--regression-threshold",
    type=float,
    default=0.02,
    help="Per-run change threshold for regression (fraction, default 0.02).",
)
@click.option(
    "--trend-threshold",
    type=float,
    default=0.05,
    help="Overall slope threshold for classification (fraction, default 0.05).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "csv", "markdown"]),
    default="table",
    help="Output format.",
)
@click.option(
    "--tracking-dir",
    type=click.Path(path_type=Path),
    default="results/tracking",
    help="Directory containing tracking series data.",
)
def track_trend(
    series_name: str,
    condition: str | None,
    regression_threshold: float,
    trend_threshold: float,
    fmt: str,
    tracking_dir: Path,
) -> None:
    """Analyze trends across runs in a tracking series."""
    from labeille.bench.tracking import load_series
    from labeille.bench.trends import analyze_series_trends

    series_dir = tracking_dir / series_name
    try:
        series = load_series(series_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    trend = analyze_series_trends(
        series,
        series_dir,
        condition=condition,
        regression_threshold=regression_threshold,
        trend_threshold=trend_threshold,
    )

    if fmt == "table":
        from labeille.bench.display import format_series_trend

        click.echo(format_series_trend(trend))
    elif fmt == "csv":
        from labeille.bench.export import export_trend_csv

        click.echo(export_trend_csv(trend))
    else:
        from labeille.bench.export import export_trend_markdown

        click.echo(export_trend_markdown(trend))


@track.command("alert")
@click.argument("series_name")
@click.option("--condition", type=str, default=None, help="Condition to analyze.")
@click.option(
    "--tracking-dir",
    type=click.Path(path_type=Path),
    default="results/tracking",
    help="Directory containing tracking series data.",
)
def track_alert(
    series_name: str,
    condition: str | None,
    tracking_dir: Path,
) -> None:
    """Show regression alerts for a tracking series.

    Compares the latest run against the baseline and previous run.
    Shows new regressions, sustained regressions, and recoveries.
    """
    from labeille.bench.display import format_regression_alerts
    from labeille.bench.tracking import load_series
    from labeille.bench.trends import analyze_series_trends

    series_dir = tracking_dir / series_name
    try:
        series = load_series(series_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    trend = analyze_series_trends(series, series_dir, condition=condition)

    if trend.alerts:
        click.echo(format_regression_alerts(trend.alerts))
    else:
        click.echo("No regression alerts.")
