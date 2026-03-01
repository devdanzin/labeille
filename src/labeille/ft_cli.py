"""CLI commands for free-threading compatibility testing."""

from __future__ import annotations

import logging
from pathlib import Path

import click

log = logging.getLogger("labeille")


@click.group()
def ft() -> None:
    """Free-threading compatibility testing.

    Test PyPI packages against free-threaded CPython builds to
    detect crashes, deadlocks, race conditions, and GIL fallback.
    """


# ---------------------------------------------------------------------------
# ft run
# ---------------------------------------------------------------------------


@ft.command()
@click.option(
    "--target-python",
    required=True,
    type=click.Path(exists=True),
    help="Path to free-threaded Python build.",
)
@click.option("--iterations", "-n", default=10, help="Runs per package (default: 10).")
@click.option("--timeout", default=600, help="Per-iteration timeout in seconds.")
@click.option(
    "--stall-threshold",
    default=60,
    help="Seconds without output before deadlock detection.",
)
@click.option("--packages", default=None, help="Comma-separated package filter.")
@click.option(
    "--top",
    "top_n",
    default=None,
    type=int,
    help="Test only top N packages by downloads.",
)
@click.option(
    "--compare-with-gil",
    is_flag=True,
    help="Also run with GIL enabled for comparison.",
)
@click.option(
    "--stop-on-first-pass",
    is_flag=True,
    help="Stop after first passing iteration per package.",
)
@click.option(
    "--detect-extensions/--no-detect-extensions",
    default=True,
    help="Check extension GIL compatibility.",
)
@click.option("--tsan", "tsan_build", is_flag=True, help="Parse TSAN warnings from stderr.")
@click.option(
    "--check-stability/--no-check-stability",
    default=False,
    help="Check system stability before starting.",
)
@click.option("--extra-deps", default=None, help="Comma-separated extra dependencies.")
@click.option("--test-command-suffix", default=None, help="Append to test commands.")
@click.option("--test-command-override", default=None, help="Override all test commands.")
@click.option(
    "--results-dir",
    type=click.Path(),
    default="results",
    help="Output directory for results.",
)
@click.option(
    "--registry-dir", type=click.Path(exists=True), default="registry", help="Registry directory."
)
@click.option("--repos-dir", type=click.Path(), default="repos", help="Repos directory.")
@click.option("--venvs-dir", type=click.Path(), default="venvs", help="Venvs directory.")
@click.option(
    "--env",
    "env_pairs",
    type=str,
    multiple=True,
    help="Extra env var as KEY=VALUE (repeatable).",
)
@click.option("-v", "--verbose", is_flag=True)
def run(
    target_python: str,
    iterations: int,
    timeout: int,
    stall_threshold: int,
    packages: str | None,
    top_n: int | None,
    compare_with_gil: bool,
    stop_on_first_pass: bool,
    detect_extensions: bool,
    tsan_build: bool,
    check_stability: bool,
    extra_deps: str | None,
    test_command_suffix: str | None,
    test_command_override: str | None,
    results_dir: str,
    registry_dir: str,
    repos_dir: str,
    venvs_dir: str,
    env_pairs: tuple[str, ...],
    verbose: bool,
) -> None:
    """Run free-threading compatibility tests.

    Tests each package's test suite multiple times under a
    free-threaded CPython build to detect crashes, deadlocks,
    race conditions, and GIL fallback behavior.

    \b
    Examples:
        # Basic run with 10 iterations
        labeille ft run --target-python /opt/cpython-ft/bin/python3

        # Quick survey (stop on first pass)
        labeille ft run --target-python /opt/cpython-ft/bin/python3 \\
            --stop-on-first-pass --top 50

        # With GIL comparison for precise classification
        labeille ft run --target-python /opt/cpython-ft/bin/python3 \\
            --compare-with-gil --iterations 5

        # With TSAN for race detection
        labeille ft run --target-python /opt/cpython-tsan/bin/python3 \\
            --tsan --iterations 5
    """
    from labeille.ft.runner import FTRunConfig, run_ft

    env_overrides: dict[str, str] = {}
    for pair in env_pairs:
        if "=" in pair:
            k, v = pair.split("=", 1)
            env_overrides[k] = v

    config = FTRunConfig(
        target_python=Path(target_python),
        iterations=iterations,
        timeout=timeout,
        stall_threshold=stall_threshold,
        packages_filter=(
            [p.strip() for p in packages.split(",") if p.strip()] if packages else None
        ),
        top_n=top_n,
        registry_dir=Path(registry_dir),
        repos_dir=Path(repos_dir),
        venvs_dir=Path(venvs_dir),
        results_dir=Path(results_dir),
        env_overrides=env_overrides,
        extra_deps=([d.strip() for d in extra_deps.split(",") if d.strip()] if extra_deps else []),
        test_command_suffix=test_command_suffix,
        test_command_override=test_command_override,
        detect_extensions=detect_extensions,
        stop_on_first_pass=stop_on_first_pass,
        tsan_build=tsan_build,
        compare_with_gil=compare_with_gil,
        check_stability=check_stability,
        verbose=verbose,
    )

    results = run_ft(config)

    # Print summary.
    from labeille.ft.display import format_compatibility_summary
    from labeille.ft.results import FTRunSummary

    summary = FTRunSummary.compute(results)
    click.echo()
    click.echo(format_compatibility_summary(summary.to_dict()))


# ---------------------------------------------------------------------------
# ft show
# ---------------------------------------------------------------------------


@ft.command()
@click.argument("result_dir", type=click.Path(exists=True))
@click.option(
    "--sort",
    "sort_by",
    type=click.Choice(["category", "pass_rate", "name"]),
    default="category",
)
@click.option("--limit", default=None, type=int, help="Maximum packages to show.")
def show(result_dir: str, sort_by: str, limit: int | None) -> None:
    """Display free-threading test results.

    Shows the compatibility summary, per-package table, and
    highlights packages that need investigation.
    """
    from labeille.ft.display import format_compatibility_summary, format_package_table
    from labeille.ft.results import FTRunSummary, load_ft_run

    meta, results = load_ft_run(Path(result_dir))
    summary = FTRunSummary.compute(results)

    # System and Python info.
    py_info = ""
    if meta.python_profile:
        py = meta.python_profile
        flags: list[str] = []
        if py.get("jit_enabled"):
            flags.append("JIT")
        if py.get("gil_disabled"):
            flags.append("free-threaded")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        py_info = f"Python: {py.get('version', '?')}{flag_str}"

    sys_info = ""
    if meta.system_profile:
        sp = meta.system_profile
        sys_info = (
            f"System: {sp.get('cpu_model', '?')}, "
            f"{sp.get('ram_total_gb', 0):.0f}GB RAM, "
            f"{sp.get('os_distro', '?')}"
        )

    click.echo(
        format_compatibility_summary(
            summary.to_dict(),
            python_info=py_info,
            system_info=sys_info,
        )
    )
    click.echo()
    click.echo(
        format_package_table(
            results,
            sort_by=sort_by,
            max_rows=limit,
        )
    )


# ---------------------------------------------------------------------------
# ft flaky
# ---------------------------------------------------------------------------


@ft.command()
@click.argument("result_dir", type=click.Path(exists=True))
@click.option("--package", default=None, help="Show flakiness for a specific package.")
def flaky(result_dir: str, package: str | None) -> None:
    """Analyze intermittent failures in detail.

    Shows which specific tests fail intermittently, failure patterns,
    and crash signature consistency.
    """
    from labeille.ft.analysis import analyze_flakiness
    from labeille.ft.display import format_flakiness_profile
    from labeille.ft.results import FailureCategory, load_ft_run

    _, results = load_ft_run(Path(result_dir))

    if package:
        targets = [r for r in results if r.package == package]
        if not targets:
            click.echo(f"Package '{package}' not found in results.")
            return
    else:
        targets = [
            r
            for r in results
            if r.category
            in (
                FailureCategory.INTERMITTENT,
                FailureCategory.CRASH,
            )
            and r.pass_count > 0
        ]

    if not targets:
        click.echo("No intermittent packages found.")
        return

    for r in targets:
        profile = analyze_flakiness(r)
        click.echo(format_flakiness_profile(profile))
        click.echo()


# ---------------------------------------------------------------------------
# ft compat
# ---------------------------------------------------------------------------


@ft.command()
@click.argument("result_dir", type=click.Path(exists=True))
@click.option("--extensions-only", is_flag=True, help="Only show packages with C extensions.")
def compat(result_dir: str, extensions_only: bool) -> None:
    """Show extension GIL compatibility details.

    Reports which packages have C extensions, whether they declare
    Py_mod_gil, and whether GIL fallback was triggered at runtime.
    """
    from labeille.ft.compat import ExtensionCompat, format_extension_compat
    from labeille.ft.results import load_ft_run

    _, results = load_ft_run(Path(result_dir))

    for r in sorted(results, key=lambda r: r.package):
        if r.extension_compat is None:
            continue

        ext = ExtensionCompat.from_dict(r.extension_compat)

        if extensions_only and ext.is_pure_python:
            continue

        click.echo(format_extension_compat(ext))
        click.echo()


# ---------------------------------------------------------------------------
# ft compare
# ---------------------------------------------------------------------------


@ft.command()
@click.argument("run_a", type=click.Path(exists=True))
@click.argument("run_b", type=click.Path(exists=True))
def compare(run_a: str, run_b: str) -> None:
    """Compare two free-threading test runs.

    Shows which packages improved, regressed, or stayed the same
    between runs. Useful for tracking compatibility progress across
    CPython versions.

    \b
    Examples:
        labeille ft compare results/ft_314a1 results/ft_314b2
    """
    # Implementation uses ft/compare.py (prompt 29).
    from labeille.ft.compare import compare_ft_runs
    from labeille.ft.display import format_ft_comparison
    from labeille.ft.results import load_ft_run

    meta_a, results_a = load_ft_run(Path(run_a))
    meta_b, results_b = load_ft_run(Path(run_b))

    comparison = compare_ft_runs(results_a, results_b)

    click.echo(
        format_ft_comparison(
            comparison,
            label_a=meta_a.run_id,
            label_b=meta_b.run_id,
        )
    )


# ---------------------------------------------------------------------------
# ft report
# ---------------------------------------------------------------------------


@ft.command()
@click.argument("result_dir", type=click.Path(exists=True))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["markdown", "text"]),
    default="markdown",
)
@click.option("--output", "-o", default=None, help="Output file (default: stdout).")
def report(result_dir: str, fmt: str, output: str | None) -> None:
    """Generate a comprehensive compatibility report.

    Produces a full report suitable for sharing with the CPython
    core team or the free-threading compatibility tracker.
    """
    # Implementation uses ft/export.py (prompt 29).
    from labeille.ft.export import generate_report
    from labeille.ft.results import load_ft_run

    meta, results = load_ft_run(Path(result_dir))
    report_text = generate_report(meta, results, format=fmt)

    if output:
        Path(output).write_text(report_text, encoding="utf-8")
        click.echo(f"Report written to {output}")
    else:
        click.echo(report_text)


# ---------------------------------------------------------------------------
# ft export
# ---------------------------------------------------------------------------


@ft.command()
@click.argument("result_dir", type=click.Path(exists=True))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "json"]),
    default="csv",
)
@click.option("--output", "-o", default=None, help="Output file (default: stdout).")
def export(result_dir: str, fmt: str, output: str | None) -> None:
    """Export results for external analysis.

    CSV format includes one row per package with category, pass rate,
    crash count, and other key metrics. Suitable for pandas, R,
    or spreadsheet analysis.
    """
    # Implementation uses ft/export.py (prompt 29).
    from labeille.ft.export import export_csv, export_json
    from labeille.ft.results import load_ft_run

    _, results = load_ft_run(Path(result_dir))

    if fmt == "csv":
        text = export_csv(results)
    else:
        text = export_json(results)

    if output:
        Path(output).write_text(text, encoding="utf-8")
        click.echo(f"Exported to {output}")
    else:
        click.echo(text)
