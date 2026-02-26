"""CLI commands for the ``labeille analyze`` subgroup.

Provides five subcommands for analyzing registry composition and run results:
``registry``, ``run``, ``compare``, ``history``, and ``package``.
"""

from __future__ import annotations

from pathlib import Path

import click

from labeille.analyze import (
    ComparisonResult,
    HistoryAnalysis,
    PackageHistory,
    RegistryStats,
    ResultsStore,
    RunAnalysis,
    RunData,
    StatusChange,
    analyze_history,
    analyze_package,
    analyze_registry,
    analyze_run,
    build_reproduce_command,
    categorize_install_errors,
    compare_runs,
    extract_minor_version,
    result_detail,
)
from labeille.formatting import (
    format_duration,
    format_histogram,
    format_percentage,
    format_section_header,
    format_signal_name,
    format_sparkline,
    format_status_icon,
    format_table,
    truncate,
)
from labeille.registry import PackageEntry, load_package, package_exists


@click.group()
def analyze() -> None:
    """Analyze registry composition and run results."""


_results_dir_option = click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=Path("results"),
    show_default=True,
)
_registry_dir_option = click.option(
    "--registry-dir",
    type=click.Path(path_type=Path),
    default=Path("registry"),
    show_default=True,
)


# ---------------------------------------------------------------------------
# analyze registry
# ---------------------------------------------------------------------------


@analyze.command("registry")
@_registry_dir_option
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["counts", "table"]),
    default="counts",
    show_default=True,
)
@click.option("--where", "where_exprs", type=str, multiple=True)
@click.option("--python-version", type=str, default=None)
def registry_cmd(
    registry_dir: Path,
    fmt: str,
    where_exprs: tuple[str, ...],
    python_version: str | None,
) -> None:
    """Analyze registry composition."""
    packages = _load_all_packages(registry_dir, where_exprs)

    if fmt == "table":
        _print_registry_table(packages, python_version)
    else:
        stats = analyze_registry(packages, target_python_version=python_version)
        _print_registry_counts(stats)


def _load_all_packages(
    registry_dir: Path, where_exprs: tuple[str, ...] = ()
) -> list[PackageEntry]:
    """Load all package entries from the registry, optionally filtered."""
    from labeille.registry_ops import FieldFilter, matches, parse_where

    packages_dir = registry_dir / "packages"
    if not packages_dir.is_dir():
        return []

    filters: list[FieldFilter] = [parse_where(e) for e in where_exprs]
    packages: list[PackageEntry] = []

    import yaml

    for f in sorted(packages_dir.glob("*.yaml")):
        if filters:
            raw = yaml.safe_load(f.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not matches(raw, filters):
                continue
        pkg = load_package(f.stem, registry_dir)
        packages.append(pkg)

    return packages


def _print_registry_counts(stats: RegistryStats) -> None:
    """Print the counts format for registry analysis."""
    click.echo(
        f"Registry: {stats.total} packages ({stats.active} active, {stats.skipped} skipped)"
    )
    click.echo()

    # By extension type.
    click.echo("By extension type:")
    for ext_type in sorted(stats.by_extension_type.keys()):
        active, skipped = stats.by_extension_type[ext_type]
        total = active + skipped
        click.echo(f"  {ext_type:<15s} {total:3d}  ({active:3d} active, {skipped:3d} skipped)")
    click.echo()

    # By skip reason.
    if stats.by_skip_category:
        click.echo(f"By skip reason ({stats.skipped} skipped):")
        for cat, count in sorted(stats.by_skip_category.items(), key=lambda x: -x[1]):
            click.echo(f"  {cat:<30s} {count:3d}")
        click.echo()

    # By test framework.
    if stats.by_test_framework:
        click.echo(f"By test framework ({stats.active} active):")
        for fw, count in sorted(stats.by_test_framework.items(), key=lambda x: -x[1]):
            click.echo(f"  {fw:<15s} {count:3d}")
        click.echo()

    # Notable.
    if stats.notable:
        click.echo("Notable:")
        for label, count in sorted(stats.notable.items()):
            click.echo(f"  {label + ':':<20s} {count:3d} packages")
        click.echo()

    # Quality warnings.
    if stats.quality_warnings:
        click.echo(f"Quality warnings ({len(stats.quality_warnings)}):")
        for pkg_name, warning in stats.quality_warnings[:20]:
            click.echo(f"  {pkg_name}: {warning}")


def _print_registry_table(packages: list[PackageEntry], python_version: str | None) -> None:
    """Print the table format for registry analysis."""
    headers = ["Package", "Type", "Status", "Framework", "Timeout", "Notes"]
    rows: list[list[str]] = []

    for pkg in sorted(packages, key=lambda p: p.package):
        is_skipped = pkg.skip
        if (
            not is_skipped
            and python_version
            and pkg.skip_versions
            and python_version in pkg.skip_versions
        ):
            is_skipped = True

        status = "skip" if is_skipped else "active"
        timeout_str = str(pkg.timeout) if pkg.timeout is not None else ""
        notes = truncate(pkg.notes or "", 30)

        rows.append(
            [
                pkg.package,
                pkg.extension_type,
                status,
                pkg.test_framework,
                timeout_str,
                notes,
            ]
        )

    click.echo(
        format_table(
            headers,
            rows,
            alignments=["l", "l", "l", "l", "r", "l"],
            max_col_width={0: 25, 5: 30},
        )
    )


# ---------------------------------------------------------------------------
# analyze run
# ---------------------------------------------------------------------------

_STATUS_ORDER: dict[str, int] = {
    "crash": 0,
    "timeout": 1,
    "fail": 2,
    "install_error": 3,
    "clone_error": 4,
    "error": 5,
    "pass": 6,
}


@analyze.command("run")
@_results_dir_option
@_registry_dir_option
@click.argument("run_id", default="latest")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["summary", "table", "full"]),
    default="summary",
    show_default=True,
)
@click.option("-q", "--quiet", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
@click.option("--no-histogram", is_flag=True)
@click.option("--no-reproduce", is_flag=True)
def run_cmd(
    results_dir: Path,
    registry_dir: Path,
    run_id: str,
    fmt: str,
    quiet: bool,
    verbose: bool,
    no_histogram: bool,
    no_reproduce: bool,
) -> None:
    """Analyze a single run."""
    store = ResultsStore(results_dir)

    run = store.get(run_id)
    if run is None:
        raise click.ClickException(f"Run not found: {run_id}")

    # Find previous run with same Python minor version.

    all_runs = store.list_runs()
    current_pv = extract_minor_version(run.meta.python_version)
    previous: RunData | None = None
    found_current = False
    for r in all_runs:
        if r.run_id == run.run_id:
            found_current = True
            continue
        if found_current and extract_minor_version(r.meta.python_version) == current_pv:
            previous = r
            break

    analysis = analyze_run(run, previous_run=previous)

    if quiet:
        _print_run_quiet(analysis, run)
        return

    if fmt == "table":
        _print_run_table(analysis, verbose=True)
        return

    if fmt == "full":
        _print_run_table(analysis, verbose=True)
        _print_run_stderr(analysis)
        return

    # summary format
    _print_run_summary(
        analysis,
        run,
        registry_dir,
        verbose=verbose,
        no_histogram=no_histogram,
        no_reproduce=no_reproduce,
    )


def _print_run_quiet(analysis: RunAnalysis, run: RunData) -> None:
    """Print quiet mode: only crashes + one-liner."""
    crashes = analysis.crashes
    if not crashes:
        return

    click.echo(format_section_header("Crashes"))
    for r in crashes:
        sig_name = format_signal_name(r.signal)
        signature = r.crash_signature or "unknown"
        click.echo(f"  {r.package}: {sig_name}: {signature}")
        if run.run_dir:
            click.echo(f"    Stderr: {run.run_dir / 'crashes' / f'{r.package}.stderr'}")
        click.echo(f"    Test command: {r.test_command}")
        click.echo()
    click.echo(format_section_header(""))

    total_dur = analysis.total_duration
    tested = len(run.results)
    crash_word = "crash" if len(crashes) == 1 else "crashes"
    click.echo(
        f"{len(crashes)} {crash_word} found in {tested} packages tested "
        f"({format_duration(total_dur)})"
    )


def _print_run_summary(
    analysis: RunAnalysis,
    run: RunData,
    registry_dir: Path,
    *,
    verbose: bool = False,
    no_histogram: bool = False,
    no_reproduce: bool = False,
) -> None:
    """Print the summary format for run analysis."""
    meta = run.meta

    # 1. Run header.
    click.echo()
    click.echo(f"Run ID: {run.run_id}")
    click.echo(f"Python: {meta.python_version}")
    click.echo(f"JIT enabled: {'yes' if meta.jit_enabled else 'no'}")
    click.echo(f"Duration: {format_duration(analysis.total_duration)}")
    click.echo()

    # 2. Per-package table.
    results = run.results
    if verbose:
        show_results = results
    else:
        show_results = [r for r in results if r.status != "pass"]

    if show_results:
        show_results = sorted(
            show_results,
            key=lambda r: (_STATUS_ORDER.get(r.status, 99), r.package),
        )

        # Build change tags.
        change_tags: dict[str, str] = {}
        if analysis.status_changes is not None:
            for sc in analysis.status_changes:
                if sc.new_status == "pass" and sc.old_status in ("crash", "fail"):
                    change_tags[sc.package] = "[FIXED]"
                elif sc.old_status == "pass" and sc.new_status in ("crash", "fail"):
                    change_tags[sc.package] = "[REGRESSED]"

        headers = ["Package", "Status", "Duration", "Signal", "Detail"]
        rows: list[list[str]] = []
        for r in show_results:
            status_str = format_status_icon(r.status)
            tag = change_tags.get(r.package, "")
            if tag:
                status_str += f" {tag}"
            sig = format_signal_name(r.signal)
            detail = result_detail(r)
            rows.append(
                [
                    r.package,
                    status_str,
                    format_duration(r.duration_seconds),
                    sig,
                    truncate(detail, 60),
                ]
            )

        click.echo(
            format_table(
                headers,
                rows,
                alignments=["l", "l", "r", "l", "l"],
                max_col_width={0: 20, 4: 60},
            )
        )
        click.echo()

    # 2b. Status changes with commit context (vs previous run).
    if analysis.status_changes:
        click.echo(f"Status changes vs previous run ({len(analysis.status_changes)}):")
        for sc in analysis.status_changes:
            old_icon = format_status_icon(sc.old_status).split()[0]
            new_icon = format_status_icon(sc.new_status).split()[0]
            click.echo(
                f"  {old_icon}\u2192{new_icon}  {sc.package:<20s} "
                f"{sc.old_status.upper()} \u2192 {sc.new_status.upper()}"
            )
            click.echo(f"    {_format_commit_context(sc)}")
        click.echo()

    # 3. Aggregate summary.
    tested = len(results)
    total_pkgs = meta.packages_tested + meta.packages_skipped
    if total_pkgs == 0:
        total_pkgs = tested

    counts = analysis.status_counts

    def _p(n: int) -> str:
        return format_percentage(n, tested)

    passed = counts.get("pass", 0)
    failed = counts.get("fail", 0)
    crashed = counts.get("crash", 0)

    left = [
        f"Packages tested: {tested} / {total_pkgs}",
        f"  Passed:        {passed:3d} ({_p(passed)})",
        f"  Failed:        {failed:3d} ({_p(failed)})",
        f"  Crashed:       {crashed:3d} ({_p(crashed)})",
        f"  Timed out:     {counts.get('timeout', 0):3d}",
        f"  Install errors:{counts.get('install_error', 0):3d}",
        f"  Clone errors:  {counts.get('clone_error', 0):3d}",
        f"  Other errors:  {counts.get('error', 0):3d}",
    ]

    right = [
        f"Total time: {format_duration(analysis.total_duration)}",
        f"Avg per package: {format_duration(analysis.avg_duration)}",
    ]
    if analysis.fastest:
        right.append(
            f"Fastest: {analysis.fastest.package} "
            f"({format_duration(analysis.fastest.duration_seconds)})"
        )
    if analysis.slowest:
        right.append(
            f"Slowest: {analysis.slowest.package} "
            f"({format_duration(analysis.slowest.duration_seconds)})"
        )

    pad = max(len(line) for line in left) + 4
    for i in range(max(len(left), len(right))):
        l_text = left[i] if i < len(left) else ""
        r_text = right[i] if i < len(right) else ""
        click.echo(f"{l_text:<{pad}s}{r_text}")
    click.echo()

    # 4. Duration histogram.
    if not no_histogram and analysis.duration_buckets:
        click.echo(f"Duration distribution ({tested} packages):")
        click.echo(format_histogram(analysis.duration_buckets, total=tested))
        click.echo()

    # 5. Install error analysis.
    install_errors = [r for r in results if r.status == "install_error"]
    if install_errors:
        cats = categorize_install_errors(results)
        click.echo("Install errors by type:")
        for cat, pkgs in sorted(cats.items(), key=lambda x: -len(x[1])):
            pkg_list = ", ".join(pkgs[:5])
            if len(pkgs) > 5:
                pkg_list += ", ..."
            click.echo(f"  {cat + ':':<20s} {len(pkgs):2d}  ({pkg_list})")
        click.echo()

    # 6. Crash detail.
    if analysis.crashes:
        click.echo(format_section_header("Crashes"))
        for r in analysis.crashes:
            sig = format_signal_name(r.signal)
            signature = r.crash_signature or "unknown"
            click.echo(f"  {r.package}: {sig}: {signature}")
            if run.run_dir:
                click.echo(f"    Stderr: {run.run_dir / 'crashes' / f'{r.package}.stderr'}")
            click.echo(f"    Test command: {r.test_command}")
            click.echo()
        click.echo(format_section_header(""))

    # 7. Reproduce blocks.
    if not no_reproduce and analysis.crashes:
        click.echo(format_section_header("Reproduce"))
        for r in analysis.crashes:
            entry: PackageEntry | None = None
            if package_exists(r.package, registry_dir):
                entry = load_package(r.package, registry_dir)

            if entry is not None:
                sig = format_signal_name(r.signal)
                click.echo(f"# {r.package} ({sig}):")
                cmd = build_reproduce_command(r, entry, str(meta.target_python))
                click.echo(cmd)
                click.echo()
        click.echo(format_section_header(""))


def _print_run_table(analysis: RunAnalysis, *, verbose: bool = False) -> None:
    """Print the table format for run analysis."""
    results = analysis.run.results
    headers = [
        "Package",
        "Status",
        "Duration",
        "Install",
        "Signal",
        "Exit Code",
        "Timed Out",
        "Detail",
    ]
    rows: list[list[str]] = []
    for r in sorted(results, key=lambda r: (_STATUS_ORDER.get(r.status, 99), r.package)):
        rows.append(
            [
                r.package,
                format_status_icon(r.status),
                format_duration(r.duration_seconds),
                format_duration(r.install_duration_seconds),
                format_signal_name(r.signal),
                str(r.exit_code),
                "yes" if r.timeout_hit else "",
                truncate(result_detail(r), 50),
            ]
        )

    click.echo(
        format_table(
            headers,
            rows,
            alignments=["l", "l", "r", "r", "l", "r", "l", "l"],
            max_col_width={0: 20, 7: 50},
        )
    )


def _print_run_stderr(analysis: RunAnalysis) -> None:
    """Print stderr for non-passing packages (full format)."""
    for r in analysis.run.results:
        if r.status == "pass":
            continue
        if r.stderr_tail:
            click.echo(f"\n  --- {r.package} stderr (last lines) ---")
            for line in r.stderr_tail.splitlines()[-10:]:
                click.echo(f"    {line}")


# ---------------------------------------------------------------------------
# analyze compare
# ---------------------------------------------------------------------------


@analyze.command("compare")
@_results_dir_option
@click.argument("run_a")
@click.argument("run_b")
@click.option("--only-changes", is_flag=True)
@click.option("--no-timing", is_flag=True)
def compare_cmd(
    results_dir: Path,
    run_a: str,
    run_b: str,
    only_changes: bool,
    no_timing: bool,
) -> None:
    """Compare two runs."""
    store = ResultsStore(results_dir)

    ra = store.get(run_a)
    rb = store.get(run_b)
    if ra is None:
        raise click.ClickException(f"Run not found: {run_a}")
    if rb is None:
        raise click.ClickException(f"Run not found: {run_b}")

    comparison = compare_runs(ra, rb)
    _print_comparison(comparison, only_changes=only_changes, no_timing=no_timing)


def _commit_annotation(
    old_status: str,
    new_status: str,
    commit_changed: bool,
    commit_known: bool,
) -> str:
    """Return a brief annotation about the likely cause of a status change."""
    if not commit_known:
        return ""
    if old_status == "pass" and new_status == "crash":
        if not commit_changed:
            return " — likely a CPython/JIT regression"
        return ""
    if old_status == "crash" and new_status == "pass":
        if not commit_changed:
            return " — likely a CPython/JIT fix"
        return ""
    return ""


def _format_commit_context(change: StatusChange) -> str:
    """Format commit info for a status change."""
    old = (change.old_commit or "unknown")[:7]
    new = (change.new_commit or "unknown")[:7]
    commit_known = change.old_commit is not None and change.new_commit is not None
    if old == "unknown" or new == "unknown":
        label = f"Repo: {old} \u2192 {new}"
    elif old == new:
        annotation = _commit_annotation(change.old_status, change.new_status, False, commit_known)
        label = f"Repo: {old} (unchanged{annotation})"
    else:
        annotation = _commit_annotation(change.old_status, change.new_status, True, commit_known)
        label = f"Repo: {old} \u2192 {new} (changed{annotation})"
    return label


def _print_comparison(
    comp: ComparisonResult,
    *,
    only_changes: bool = False,
    no_timing: bool = False,
) -> None:
    """Print comparison results."""
    click.echo(f"Comparing: {comp.run_a.run_id} \u2192 {comp.run_b.run_id}")

    pv_a = comp.run_a.meta.python_version
    pv_b = comp.run_b.meta.python_version
    if pv_a or pv_b:
        pv_a_short = truncate(pv_a, 40)
        pv_b_short = truncate(pv_b, 40)
        click.echo(f"  Python: {pv_a_short} \u2192 {pv_b_short}")

    click.echo(f"  Packages in common: {comp.packages_in_common}")
    if comp.packages_only_in_a:
        click.echo(
            f"  Only in first run:  {len(comp.packages_only_in_a)}  "
            f"({', '.join(comp.packages_only_in_a[:5])})"
        )
    if comp.packages_only_in_b:
        click.echo(
            f"  Only in second run: {len(comp.packages_only_in_b)}  "
            f"({', '.join(comp.packages_only_in_b[:5])})"
        )
    click.echo()

    # Status changes.
    if comp.status_changes:
        click.echo(f"Status changes ({len(comp.status_changes)}):")
        for sc in comp.status_changes:
            old_icon = format_status_icon(sc.old_status).split()[0]
            new_icon = format_status_icon(sc.new_status).split()[0]
            detail = sc.new_detail
            if detail:
                click.echo(
                    f"  {old_icon}\u2192{new_icon}  {sc.package:<20s} "
                    f"{sc.old_status.upper()} \u2192 {sc.new_status.upper()}  "
                    f"  {truncate(detail, 40)}"
                )
            else:
                click.echo(
                    f"  {old_icon}\u2192{new_icon}  {sc.package:<20s} "
                    f"{sc.old_status.upper()} \u2192 {sc.new_status.upper()}"
                )
            click.echo(f"    {_format_commit_context(sc)}")
        click.echo()

        # Summary statistics for new crashes.
        new_crashes = [
            c for c in comp.status_changes if c.old_status == "pass" and c.new_status == "crash"
        ]
        if new_crashes:
            repo_unchanged = sum(
                1
                for c in new_crashes
                if c.old_commit and c.new_commit and c.old_commit == c.new_commit
            )
            repo_changed = sum(
                1
                for c in new_crashes
                if c.old_commit and c.new_commit and c.old_commit != c.new_commit
            )
            repo_unknown = len(new_crashes) - repo_unchanged - repo_changed
            click.echo(f"  New crashes: {len(new_crashes)}")
            if repo_unchanged:
                click.echo(f"    Repo unchanged: {repo_unchanged}")
            if repo_changed:
                click.echo(f"    Repo changed: {repo_changed}")
            if repo_unknown:
                click.echo(f"    Commit unknown: {repo_unknown}")
            click.echo()
    else:
        click.echo("No status changes.")
        click.echo()

    # Crash signature changes.
    if comp.signature_changes:
        click.echo(f"Crash signature changes ({len(comp.signature_changes)}):")
        for pkg, old_sig, new_sig in comp.signature_changes:
            if old_sig is None and new_sig is not None:
                click.echo(f"  {pkg}: NEW crash \u2014 {truncate(new_sig, 60)}")
            elif old_sig is not None and new_sig is None:
                click.echo(f"  {pkg}: crash RESOLVED (was: {truncate(old_sig, 60)})")
            elif old_sig != new_sig:
                click.echo(
                    f"  {pkg}: {truncate(old_sig or '', 30)} \u2192 {truncate(new_sig or '', 30)}"
                )
        click.echo()

    # Unchanged.
    if not only_changes and comp.unchanged_counts:
        parts = [
            f"{count} {status}"
            for status, count in sorted(comp.unchanged_counts.items(), key=lambda x: -x[1])
        ]
        total_unchanged = sum(comp.unchanged_counts.values())
        click.echo(f"Unchanged: {total_unchanged} ({', '.join(parts)})")
        click.echo()

    # Timing changes.
    if not no_timing and not only_changes and comp.timing_changes:
        click.echo(f"Timing changes (>{20}% and >{30}s):")
        headers = ["Package", "Before", "After", "Change"]
        rows: list[list[str]] = []
        for tc in sorted(comp.timing_changes, key=lambda x: -abs(x.change_pct)):
            sign = "+" if tc.change_pct > 0 else ""
            rows.append(
                [
                    tc.package,
                    format_duration(tc.old_seconds),
                    format_duration(tc.new_seconds),
                    f"{sign}{tc.change_pct:.0f}%",
                ]
            )
        click.echo(format_table(headers, rows, alignments=["l", "r", "r", "r"]))


# ---------------------------------------------------------------------------
# analyze history
# ---------------------------------------------------------------------------


@analyze.command("history")
@_results_dir_option
@click.option("--last", "last_n", type=int, default=10, show_default=True)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "timeline"]),
    default="table",
    show_default=True,
)
@click.option("--python-version", type=str, default=None)
def history_cmd(
    results_dir: Path,
    last_n: int,
    fmt: str,
    python_version: str | None,
) -> None:
    """View run history and trends."""
    store = ResultsStore(results_dir)

    runs = store.list_runs(python_version=python_version)[:last_n]

    if not runs:
        click.echo("No runs found.")
        return

    history = analyze_history(runs)

    if fmt == "timeline":
        _print_history_timeline(history)
    else:
        _print_history_table(history)


def _print_history_table(history: HistoryAnalysis) -> None:
    """Print the table format for history analysis."""
    click.echo(f"Run history (last {len(history.runs)}):")
    click.echo()

    headers = [
        "Run ID",
        "Python",
        "Tested",
        "Pass",
        "Fail",
        "Crash",
        "Timeout",
        "Error",
        "Duration",
    ]
    rows: list[list[str]] = []

    for run in history.runs:
        results = run.results
        by_status: dict[str, int] = {}
        for r in results:
            by_status[r.status] = by_status.get(r.status, 0) + 1

        pv = truncate(run.meta.python_version.split("(")[0].strip(), 20)
        total_dur = sum(r.duration_seconds for r in results)

        rows.append(
            [
                run.run_id,
                pv,
                str(len(results)),
                str(by_status.get("pass", 0)),
                str(by_status.get("fail", 0)),
                str(by_status.get("crash", 0)),
                str(by_status.get("timeout", 0)),
                str(
                    by_status.get("error", 0)
                    + by_status.get("install_error", 0)
                    + by_status.get("clone_error", 0)
                ),
                format_duration(total_dur),
            ]
        )

    click.echo(
        format_table(
            headers,
            rows,
            alignments=["l", "l", "r", "r", "r", "r", "r", "r", "r"],
        )
    )
    click.echo()

    # Crash summary.
    click.echo("Crash summary:")
    click.echo(f"  Total unique crash signatures: {history.total_unique_crashes}")
    click.echo(f"  Currently reproducing (latest run): {history.currently_reproducing}")
    click.echo(f"  Likely fixed (in earlier runs, not in latest): {history.likely_fixed}")
    click.echo()

    # Flaky packages.
    if history.flaky_packages:
        click.echo("Flaky packages (inconsistent across runs with same Python version):")
        for pkg, statuses in history.flaky_packages:
            icons = " ".join(format_status_icon(s).split()[0] for s in statuses)
            oscillations = sum(
                1
                for i in range(1, len(statuses))
                if statuses[i] != statuses[i - 1] and statuses[i] != "crash"
            )
            click.echo(f"  {pkg}: {icons}  ({oscillations} oscillations)")


def _print_history_timeline(history: HistoryAnalysis) -> None:
    """Print the timeline format for history analysis."""
    if history.crash_trend:
        spark = format_sparkline([float(x) for x in history.crash_trend], width=10)
        crash_first = history.crash_trend[0]
        crash_last = history.crash_trend[-1]
        click.echo(
            f"Crash trend (last {len(history.runs)} runs):   "
            f"{spark}  {crash_first} \u2192 {crash_last}"
        )

    if history.pass_rate_trend:
        spark = format_sparkline(history.pass_rate_trend, width=10)
        rate_first = f"{history.pass_rate_trend[0]:.0f}%"
        rate_last = f"{history.pass_rate_trend[-1]:.0f}%"
        click.echo(f"Pass rate trend:               {spark}  {rate_first} \u2192 {rate_last}")

    click.echo()

    # Group by Python version.

    by_version: dict[str, list[RunData]] = {}
    for run in history.runs:
        pv = extract_minor_version(run.meta.python_version)
        by_version.setdefault(pv, []).append(run)

    if len(by_version) > 1:
        click.echo("By Python version:")
        for pv, vruns in sorted(by_version.items()):
            first_crashes = sum(1 for r in vruns[-1].results if r.status == "crash")
            last_crashes = sum(1 for r in vruns[0].results if r.status == "crash")
            first_total = len(vruns[-1].results)
            last_total = len(vruns[0].results)
            first_pass = sum(1 for r in vruns[-1].results if r.status == "pass")
            last_pass = sum(1 for r in vruns[0].results if r.status == "pass")
            first_rate = f"{first_pass / first_total * 100:.0f}%" if first_total else "-"
            last_rate = f"{last_pass / last_total * 100:.0f}%" if last_total else "-"
            click.echo(
                f"  {pv}  ({len(vruns)} runs): "
                f"crashes {first_crashes}\u2192{last_crashes}, "
                f"pass rate {first_rate}\u2192{last_rate}"
            )


# ---------------------------------------------------------------------------
# analyze package
# ---------------------------------------------------------------------------


@analyze.command("package")
@_results_dir_option
@_registry_dir_option
@click.argument("package_name")
@click.option("--last", "last_n", type=int, default=None)
def package_cmd(
    results_dir: Path,
    registry_dir: Path,
    package_name: str,
    last_n: int | None,
) -> None:
    """Deep dive on a specific package's history."""
    store = ResultsStore(results_dir)

    entry: PackageEntry | None = None
    if package_exists(package_name, registry_dir):
        entry = load_package(package_name, registry_dir)

    history = analyze_package(package_name, store, registry_entry=entry)

    if last_n is not None:
        history.run_results = history.run_results[:last_n]

    _print_package_history(history)


def _print_package_history(history: PackageHistory) -> None:
    """Print package history."""
    click.echo(f"Package: {history.package}")

    entry = history.registry_entry
    if entry is not None:
        click.echo(f"  Repo: {entry.repo or 'N/A'}")
        timeout_str = f"{entry.timeout}s" if entry.timeout is not None else "default"
        click.echo(
            f"  Type: {entry.extension_type} | "
            f"Framework: {entry.test_framework} | "
            f"Timeout: {timeout_str}"
        )
    click.echo()

    # Run history.
    if history.run_results:
        click.echo(f"Run history ({len(history.run_results)} runs):")
        for run, result in history.run_results:
            pv = extract_minor_version(run.meta.python_version)
            rev = ""
            if result.git_revision:
                rev = f" ({result.git_revision[:7]})"
            status = format_status_icon(result.status)
            dur = format_duration(result.duration_seconds)
            detail = ""
            if result.status == "crash":
                detail = f"  {result.crash_signature or ''}"
            elif result.status == "pass":
                install_dur = format_duration(result.install_duration_seconds)
                detail = f"  (install: {install_dur})"

            date = run.run_id[:10] if len(run.run_id) >= 10 else run.run_id
            click.echo(f"  {date}  {pv}{rev}  {status}  {dur:>8s}{detail}")
        click.echo()
    else:
        click.echo("No run history found.")
        click.echo()

    # Crash signatures.
    if history.crash_signatures:
        click.echo("Crash signatures seen:")
        for sig, count in sorted(history.crash_signatures.items(), key=lambda x: -x[1]):
            status_note = ""
            if history.likely_fixed:
                status_note = " \u2014 not in latest run (likely fixed)"
            click.echo(f"  {sig}")
            click.echo(f"    Occurrences: {count}")
            if history.latest_crash_date:
                click.echo(f"    Last seen: {history.latest_crash_date}")
            if status_note:
                click.echo(f"    Status: {status_note.strip()}")
        click.echo()

    # Duration trend.
    if len(history.run_results) >= 2:
        durations = [format_duration(r.duration_seconds) for _, r in reversed(history.run_results)]
        click.echo(f"Duration trend: {' \u2192 '.join(durations)}")
        click.echo()

    # Dependency changes.
    if history.dependency_changes:
        click.echo("Dependency changes at status transitions:")
        for run_id, old_deps, new_deps in history.dependency_changes[:3]:
            click.echo(f"  Run {run_id}:")
            all_keys = set(old_deps.keys()) | set(new_deps.keys())
            for key in sorted(all_keys):
                old_v = old_deps.get(key)
                new_v = new_deps.get(key)
                if old_v is None and new_v is not None:
                    click.echo(f"    + {key} {new_v}")
                elif old_v is not None and new_v is None:
                    click.echo(f"    - {key} {old_v}")
                elif old_v != new_v:
                    click.echo(f"    ~ {key} {old_v} \u2192 {new_v}")
