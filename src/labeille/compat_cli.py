"""CLI commands for ``labeille compat``.

Subcommands:
    labeille compat survey    Run a C extension compatibility survey
    labeille compat show      Display results from a saved survey
    labeille compat compare   Compare two surveys
    labeille compat patterns  List built-in error classification patterns
"""

from __future__ import annotations

from pathlib import Path

import click

from labeille.cli_utils import parse_csv_list
from labeille.logging import setup_logging


@click.group()
def compat() -> None:
    """C extension compatibility survey.

    Build packages against a target Python and classify failures
    into fine-grained categories (removed C API, Cython/PyO3
    incompatibility, missing headers, etc.).
    """


# ---------------------------------------------------------------------------
# compat survey
# ---------------------------------------------------------------------------


@compat.command()
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
    "--packages-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="File with one package name per line.",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=None,
    help="Survey only the top N packages by download count (requires --registry-dir).",
)
@click.option(
    "--from",
    "from_mode",
    type=click.Choice(["sdist", "source"]),
    default="sdist",
    show_default=True,
    help="Build from sdist (PyPI) or source (git clone).",
)
@click.option(
    "--no-binary-all",
    is_flag=True,
    help="Use pip --no-binary :all: (force compile everything).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("compat-results"),
    show_default=True,
    help="Output directory for survey results.",
)
@click.option("--timeout", type=int, default=600, show_default=True)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Number of packages to build in parallel.",
)
@click.option(
    "--repos-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Persistent directory for repo clones (source mode).",
)
@click.option(
    "--installer",
    type=click.Choice(["auto", "uv", "pip"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Package installer backend.",
)
@click.option(
    "--extra-patterns",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML file with additional error classification patterns.",
)
@click.option(
    "--extensions-only/--all-types",
    default=True,
    help="Only survey packages with C extensions (default: true).",
)
@click.option("-v", "--verbose", is_flag=True)
@click.option("-q", "--quiet", is_flag=True, help="Only show errors.")
@click.option(
    "--export-markdown",
    type=click.Path(path_type=Path),
    default=None,
    help="Also export results as a markdown file.",
)
def survey(
    target_python: Path,
    registry_dir: Path | None,
    packages_csv: str | None,
    packages_file: Path | None,
    top_n: int | None,
    from_mode: str,
    no_binary_all: bool,
    output_dir: Path,
    timeout: int,
    workers: int,
    repos_dir: Path | None,
    installer: str,
    extra_patterns: Path | None,
    extensions_only: bool,
    verbose: bool,
    quiet: bool,
    export_markdown: Path | None,
) -> None:
    """Run a compatibility survey against packages."""
    from labeille.compat import (
        export_compat_markdown,
        format_compat_report,
        resolve_compat_inputs,
        run_compat_survey,
    )
    from labeille.runner import validate_target_python

    setup_logging(verbose=verbose, quiet=quiet)

    # Validate target python.
    try:
        python_version = validate_target_python(target_python)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    if workers < 1:
        raise click.UsageError("--workers must be at least 1")

    # Must have at least one package source.
    if not registry_dir and not packages_csv and not packages_file:
        raise click.UsageError(
            "At least one of --registry-dir, --packages, or --packages-file is required."
        )

    click.echo(f"Target Python: {python_version}")

    # Parse inline packages.
    package_names: list[str] | None = None
    if packages_csv:
        package_names = parse_csv_list(packages_csv)

    # Extract minor version for skip_versions filtering.
    from labeille.runner import extract_python_minor_version

    minor_ver = extract_python_minor_version(python_version)

    # Resolve inputs.
    packages = resolve_compat_inputs(
        registry_dir=registry_dir,
        extensions_only=extensions_only,
        package_names=package_names,
        packages_file=packages_file,
        top_n=top_n,
        target_python_version=minor_ver,
    )

    if not packages:
        click.echo("No packages to survey.")
        return

    click.echo(f"Surveying {len(packages)} package(s)...")

    # Progress callback.
    completed = [0]

    def _on_progress(name: str, status: str) -> None:
        completed[0] += 1
        if not quiet:
            click.echo(f"  [{completed[0]}/{len(packages)}] {name}: {status}")

    result = run_compat_survey(
        packages,
        target_python,
        from_mode=from_mode,
        no_binary_all=no_binary_all,
        output_dir=output_dir,
        timeout=timeout,
        workers=workers,
        repos_dir=repos_dir,
        installer_preference=installer,
        extra_patterns_file=extra_patterns,
        progress_callback=_on_progress,
    )

    # Display report.
    report = format_compat_report(result)
    click.echo("")
    click.echo(report)

    # Export markdown if requested.
    if export_markdown:
        md = export_compat_markdown(result)
        export_markdown.parent.mkdir(parents=True, exist_ok=True)
        export_markdown.write_text(md, encoding="utf-8")
        click.echo(f"Markdown exported to {export_markdown}")

    click.echo(f"Results saved to {output_dir}/{result.meta.survey_id}/")


# ---------------------------------------------------------------------------
# compat show
# ---------------------------------------------------------------------------


@compat.command()
@click.argument("survey_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "markdown"]),
    default="table",
    show_default=True,
)
@click.option(
    "--category",
    type=str,
    default=None,
    help="Filter results to a specific failure category.",
)
@click.option(
    "--status",
    type=str,
    default=None,
    help="Filter results to a specific status (e.g. build_fail, build_ok).",
)
def show(
    survey_dir: Path,
    fmt: str,
    category: str | None,
    status: str | None,
) -> None:
    """Display results from a saved compatibility survey."""
    from labeille.compat import (
        CompatSurvey,
        export_compat_markdown,
        format_compat_report,
        load_compat_survey,
    )

    try:
        survey = load_compat_survey(survey_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    # Apply filters if requested.
    if category or status:
        filtered = survey.results
        if category:
            filtered = [r for r in filtered if r.primary_category == category]
        if status:
            filtered = [r for r in filtered if r.status == status]
        survey = CompatSurvey(meta=survey.meta, results=filtered)
        click.echo(f"Filtered to {len(filtered)} result(s).")

    if fmt == "markdown":
        click.echo(export_compat_markdown(survey))
    else:
        click.echo(format_compat_report(survey))


# ---------------------------------------------------------------------------
# compat compare
# ---------------------------------------------------------------------------


@compat.command()
@click.argument("survey_a", type=click.Path(exists=True, path_type=Path))
@click.argument("survey_b", type=click.Path(exists=True, path_type=Path))
def compare(survey_a: Path, survey_b: Path) -> None:
    """Compare two compatibility surveys (A = baseline, B = new)."""
    from labeille.compat import (
        diff_surveys,
        format_compat_diff,
        load_compat_survey,
    )

    try:
        a = load_compat_survey(survey_a)
        b = load_compat_survey(survey_b)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    result = diff_surveys(a, b)
    click.echo(format_compat_diff(result))


# ---------------------------------------------------------------------------
# compat patterns
# ---------------------------------------------------------------------------


@compat.command()
@click.option(
    "--category",
    type=str,
    default=None,
    help="Filter to a specific category.",
)
@click.option(
    "--extra-patterns",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Include patterns from a YAML file.",
)
def patterns(category: str | None, extra_patterns: Path | None) -> None:
    """List built-in error classification patterns."""
    from labeille.compat import format_patterns_table, get_patterns

    try:
        pattern_list = get_patterns(extra_patterns)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(format_patterns_table(pattern_list, category_filter=category))
