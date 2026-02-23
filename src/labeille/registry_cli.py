"""CLI commands for batch registry management.

Provides the ``labeille registry`` subgroup with commands for adding,
removing, renaming, and setting fields across all package YAML files,
plus validation and index management.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from labeille.registry_ops import (
    PROTECTED_FIELDS,
    PROTECTED_INDEX_FIELDS,
    DryRunPreview,
    FieldFilter,
    OpResult,
    batch_add_field,
    batch_remove_field,
    batch_rename_field,
    batch_set_field,
    parse_where,
    rebuild_index,
    remove_index_field as remove_index_field_op,
    validate_registry,
)
from labeille.yaml_lines import format_yaml_value, parse_default_value


def _auto_detect_registry(registry_dir: Path | None) -> Path:
    """Resolve the registry directory, defaulting to ``registry/``."""
    if registry_dir is not None:
        return registry_dir
    cwd = Path.cwd()
    if (cwd / "registry" / "packages").is_dir():
        return cwd / "registry"
    return Path("registry")


def _parse_filters(where_exprs: tuple[str, ...]) -> list[FieldFilter]:
    """Parse multiple --where expressions."""
    filters: list[FieldFilter] = []
    for expr in where_exprs:
        try:
            filters.append(parse_where(expr))
        except ValueError as e:
            raise click.UsageError(str(e)) from e
    return filters


def _print_dry_run(
    command: str,
    description: str,
    preview: DryRunPreview,
) -> None:
    """Print a dry-run preview."""
    click.echo(f"DRY RUN — {command} {description}")
    click.echo("")

    if preview.error_count > 0:
        click.echo(f"Errors: {preview.error_count} files with precondition failures.")
        click.echo("Use --lenient to skip these files and process the rest.")
        click.echo("")

    if preview.skipped_count > 0:
        click.echo(f"Would skip {preview.skipped_count} files (precondition not met).")

    click.echo(f"Would modify {preview.affected_count} files.")

    if preview.sample_diffs:
        count = len(preview.sample_diffs)
        total = preview.affected_count
        click.echo(f"\nSample changes ({count} of {total}):\n")
        for filename, before, after in preview.sample_diffs:
            click.echo(f"  {filename}:")
            _print_diff_summary(before, after)
            click.echo("")

    click.echo("Re-run with --apply to write changes.")


def _print_diff_summary(before: str, after: str) -> None:
    """Print a brief diff between before and after text."""
    before_lines = before.splitlines()
    after_lines = after.splitlines()

    # Find added lines
    before_set = set(before_lines)
    after_set = set(after_lines)

    for line in after_lines:
        if line not in before_set:
            click.echo(f"    + {line.strip()}")

    for line in before_lines:
        if line not in after_set:
            click.echo(f"    - {line.strip()}")


def _print_result(result: OpResult, update_index: bool, registry_dir: Path) -> None:
    """Print the result of an applied operation."""
    if result.modified:
        click.echo(f"Modified {len(result.modified)} files.")
    if result.skipped:
        click.echo(
            f"Skipped {len(result.skipped)} files (precondition not met),"
            f" modified {len(result.modified)} files."
        )
    if result.errors:
        click.echo(f"Errors in {len(result.errors)} files:")
        for e in result.errors[:10]:
            click.echo(f"  {e}")
        if len(result.errors) > 10:
            click.echo(f"  ... and {len(result.errors) - 10} more")

    if update_index and result.modified:
        count = rebuild_index(registry_dir)
        click.echo(f"Index updated ({count} packages).")

    if result.modified:
        click.echo(f"Review with: git diff {registry_dir / 'packages/'}")


# ---------------------------------------------------------------------------
# Shared click options
# ---------------------------------------------------------------------------

_apply_option = click.option(
    "--apply", is_flag=True, help="Actually write changes (without this, dry-run only)."
)
_lenient_option = click.option(
    "--lenient",
    is_flag=True,
    help="Skip files where the precondition isn't met instead of erroring.",
)
_registry_dir_option = click.option(
    "--registry-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to registry directory (default: auto-detect).",
)
_where_option = click.option(
    "--where",
    "where_exprs",
    type=str,
    multiple=True,
    help="Filter packages (repeatable, combined with AND).",
)
_packages_option = click.option(
    "--packages",
    "packages_csv",
    type=str,
    default=None,
    help="Comma-separated package names.",
)
_update_index_option = click.option(
    "--update-index/--no-update-index",
    default=True,
    help="Rebuild the index after applying changes (default: true).",
)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group("registry")
def registry() -> None:
    """Batch management of the package registry."""


# ---------------------------------------------------------------------------
# add-field
# ---------------------------------------------------------------------------


@registry.command("add-field")
@click.argument("field_name")
@click.option(
    "--type",
    "field_type",
    type=click.Choice(["str", "int", "bool", "list", "dict"]),
    default="str",
    show_default=True,
    help="Field type.",
)
@click.option("--default", "default_value", type=str, default=None, help="Default value.")
@click.option("--after", type=str, default=None, help="Insert after this existing field.")
@_where_option
@_packages_option
@_apply_option
@_lenient_option
@_registry_dir_option
@_update_index_option
def add_field_cmd(
    field_name: str,
    field_type: str,
    default_value: str | None,
    after: str | None,
    where_exprs: tuple[str, ...],
    packages_csv: str | None,
    apply: bool,
    lenient: bool,
    registry_dir: Path | None,
    update_index: bool,
) -> None:
    """Add a new field to package YAML files."""
    registry_dir = _auto_detect_registry(registry_dir)
    filters = _parse_filters(where_exprs)
    packages_list = (
        [p.strip() for p in packages_csv.split(",") if p.strip()] if packages_csv else None
    )

    parsed_default = parse_default_value(default_value, field_type)
    value_text = format_yaml_value(parsed_default, field_type)

    result = batch_add_field(
        registry_dir,
        field_name,
        field_type=field_type,
        default=default_value,
        after=after,
        filters=filters,
        packages_list=packages_list,
        dry_run=not apply,
        lenient=lenient,
    )

    if isinstance(result, DryRunPreview):
        _print_dry_run(
            "add-field",
            f"'{field_name}' ({field_type}, default: {value_text})",
            result,
        )
        if result.error_count > 0 and not lenient:
            sys.exit(1)
    else:
        _print_result(result, update_index, registry_dir)
        if result.errors and not lenient:
            sys.exit(1)
        if not result.modified and result.skipped:
            click.echo("Nothing to do — all files were skipped.")
            sys.exit(1)
        if result.modified:
            click.echo(f"Remember to add '{field_name}' to PackageEntry in registry.py")


# ---------------------------------------------------------------------------
# remove-field
# ---------------------------------------------------------------------------


@registry.command("remove-field")
@click.argument("field_name")
@_where_option
@_packages_option
@_apply_option
@_lenient_option
@_registry_dir_option
@_update_index_option
def remove_field_cmd(
    field_name: str,
    where_exprs: tuple[str, ...],
    packages_csv: str | None,
    apply: bool,
    lenient: bool,
    registry_dir: Path | None,
    update_index: bool,
) -> None:
    """Remove a field from package YAML files."""
    registry_dir = _auto_detect_registry(registry_dir)

    if field_name in PROTECTED_FIELDS:
        click.echo(f"Error: '{field_name}' is a protected field and cannot be removed.")
        sys.exit(1)

    filters = _parse_filters(where_exprs)
    packages_list = (
        [p.strip() for p in packages_csv.split(",") if p.strip()] if packages_csv else None
    )

    result = batch_remove_field(
        registry_dir,
        field_name,
        filters=filters,
        packages_list=packages_list,
        dry_run=not apply,
        lenient=lenient,
    )

    if isinstance(result, DryRunPreview):
        _print_dry_run("remove-field", f"'{field_name}'", result)
        if result.error_count > 0 and not lenient:
            sys.exit(1)
    else:
        _print_result(result, update_index, registry_dir)
        if result.errors and not lenient:
            sys.exit(1)
        if not result.modified and result.skipped:
            click.echo("Nothing to do — all files were skipped.")
            sys.exit(1)


# ---------------------------------------------------------------------------
# rename-field
# ---------------------------------------------------------------------------


@registry.command("rename-field")
@click.argument("old_name")
@click.argument("new_name")
@_where_option
@_packages_option
@_apply_option
@_lenient_option
@_registry_dir_option
@_update_index_option
def rename_field_cmd(
    old_name: str,
    new_name: str,
    where_exprs: tuple[str, ...],
    packages_csv: str | None,
    apply: bool,
    lenient: bool,
    registry_dir: Path | None,
    update_index: bool,
) -> None:
    """Rename a field in package YAML files."""
    registry_dir = _auto_detect_registry(registry_dir)
    filters = _parse_filters(where_exprs)
    packages_list = (
        [p.strip() for p in packages_csv.split(",") if p.strip()] if packages_csv else None
    )

    result = batch_rename_field(
        registry_dir,
        old_name,
        new_name,
        filters=filters,
        packages_list=packages_list,
        dry_run=not apply,
        lenient=lenient,
    )

    if isinstance(result, DryRunPreview):
        _print_dry_run("rename-field", f"'{old_name}' → '{new_name}'", result)
        if result.error_count > 0 and not lenient:
            sys.exit(1)
    else:
        _print_result(result, update_index, registry_dir)
        if result.errors and not lenient:
            sys.exit(1)
        if not result.modified and result.skipped:
            click.echo("Nothing to do — all files were skipped.")
            sys.exit(1)
        if result.modified:
            click.echo(
                f"Remember to rename '{old_name}' to '{new_name}' in PackageEntry in registry.py"
            )


# ---------------------------------------------------------------------------
# set-field
# ---------------------------------------------------------------------------


@registry.command("set-field")
@click.argument("field_name")
@click.argument("value")
@click.option(
    "--type",
    "field_type",
    type=click.Choice(["str", "int", "bool", "list", "dict"]),
    default=None,
    help="Explicit type override.",
)
@click.option("--all", "select_all", is_flag=True, help="Apply to all packages.")
@_where_option
@_packages_option
@_apply_option
@_registry_dir_option
@_update_index_option
def set_field_cmd(
    field_name: str,
    value: str,
    field_type: str | None,
    select_all: bool,
    where_exprs: tuple[str, ...],
    packages_csv: str | None,
    apply: bool,
    registry_dir: Path | None,
    update_index: bool,
) -> None:
    """Set a field to a specific value on matching packages."""
    registry_dir = _auto_detect_registry(registry_dir)
    filters = _parse_filters(where_exprs)
    packages_list = (
        [p.strip() for p in packages_csv.split(",") if p.strip()] if packages_csv else None
    )

    if not select_all and not filters and not packages_list:
        raise click.UsageError(
            "set-field requires --all, --where, or --packages to prevent accidental mass updates."
        )

    result = batch_set_field(
        registry_dir,
        field_name,
        value,
        field_type=field_type,
        filters=filters,
        packages_list=packages_list,
        require_all=select_all,
        dry_run=not apply,
    )

    if isinstance(result, DryRunPreview):
        _print_dry_run("set-field", f"'{field_name}' = '{value}'", result)
        if result.error_count > 0:
            sys.exit(1)
    else:
        _print_result(result, update_index, registry_dir)
        if result.errors:
            sys.exit(1)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@registry.command("validate")
@click.option("--strict", is_flag=True, help="Treat warnings as errors.")
@_where_option
@_packages_option
@_registry_dir_option
def validate_cmd(
    strict: bool,
    where_exprs: tuple[str, ...],
    packages_csv: str | None,
    registry_dir: Path | None,
) -> None:
    """Check YAML files against the PackageEntry schema."""
    registry_dir = _auto_detect_registry(registry_dir)
    filters = _parse_filters(where_exprs)
    packages_list = (
        [p.strip() for p in packages_csv.split(",") if p.strip()] if packages_csv else None
    )

    from labeille.registry_ops import _list_package_files, _filter_files

    files = _list_package_files(registry_dir)
    files = _filter_files(files, filters, packages_list)

    click.echo(f"Checking {len(files)} packages...")
    click.echo("")

    issues = validate_registry(
        registry_dir,
        strict=strict,
        filters=filters,
        packages_list=packages_list,
    )

    error_count = 0
    warning_count = 0

    for issue in issues:
        prefix = "ERROR" if issue.level == "error" else "WARN"
        click.echo(f"  {issue.filename}: {prefix} {issue.message}")
        if issue.level == "error":
            error_count += 1
        else:
            warning_count += 1

    ok_count = len(files) - len({i.filename for i in issues})
    click.echo("")
    click.echo(f"Results: {ok_count} OK, {warning_count} warnings, {error_count} errors")

    if error_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# add-index-field
# ---------------------------------------------------------------------------


@registry.command("add-index-field")
@click.argument("field_name")
@_apply_option
@_registry_dir_option
def add_index_field_cmd(
    field_name: str,
    apply: bool,
    registry_dir: Path | None,
) -> None:
    """Add a field to the registry index."""
    registry_dir = _auto_detect_registry(registry_dir)

    from labeille.registry_ops import add_index_field

    result = add_index_field(registry_dir, field_name, dry_run=not apply)

    if isinstance(result, DryRunPreview):
        click.echo(f"DRY RUN — add-index-field '{field_name}'")
        click.echo(f"Would update {result.affected_count} index entries.")
        click.echo("Re-run with --apply to write changes.")
    else:
        click.echo(f"Index updated ({len(result.modified)} packages).")


# ---------------------------------------------------------------------------
# remove-index-field
# ---------------------------------------------------------------------------


@registry.command("remove-index-field")
@click.argument("field_name")
@_apply_option
@_registry_dir_option
def remove_index_field_cmd(
    field_name: str,
    apply: bool,
    registry_dir: Path | None,
) -> None:
    """Remove a field from the registry index."""
    registry_dir = _auto_detect_registry(registry_dir)

    if field_name in PROTECTED_INDEX_FIELDS:
        click.echo(f"Error: '{field_name}' is a protected index field and cannot be removed.")
        sys.exit(1)

    result = remove_index_field_op(registry_dir, field_name, dry_run=not apply)

    if isinstance(result, DryRunPreview):
        click.echo(f"DRY RUN — remove-index-field '{field_name}'")
        click.echo(f"Would update {result.affected_count} index entries.")
        click.echo("Re-run with --apply to write changes.")
    else:
        click.echo(f"Index updated ({len(result.modified)} packages).")


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


@registry.command("migrate")
@click.argument("migration_name", required=False)
@click.option("--list", "list_flag", is_flag=True, help="List available migrations.")
@_apply_option
@_registry_dir_option
@_update_index_option
def migrate_cmd(
    migration_name: str | None,
    list_flag: bool,
    apply: bool,
    registry_dir: Path | None,
    update_index: bool,
) -> None:
    """Apply a named registry migration."""
    from labeille.migrations import (
        MigrationDryRun,
        execute_migration,
        get_applied_date,
        get_migration,
        has_been_applied,
        list_migrations,
    )

    registry_dir = _auto_detect_registry(registry_dir)

    if list_flag:
        migrations = list_migrations()
        if not migrations:
            click.echo("No migrations registered.")
            return
        click.echo("Available migrations:\n")
        for m in migrations:
            applied = has_been_applied(registry_dir, m.name)
            status = "applied" if applied else "not applied"
            date = get_applied_date(registry_dir, m.name)
            if date:
                status = f"applied on {date}"
            click.echo(f"  {m.name:<30s} {m.description}")
            click.echo(f"  {'':30s} Status: {status}")
            click.echo()
        return

    if migration_name is None:
        click.echo("Error: missing MIGRATION_NAME. Use --list to see available migrations.")
        sys.exit(1)

    migration = get_migration(migration_name)
    if migration is None:
        click.echo(f"Error: unknown migration '{migration_name}'.")
        available = list_migrations()
        if available:
            click.echo("Available migrations:")
            for m in available:
                click.echo(f"  {m.name}")
        sys.exit(1)

    if has_been_applied(registry_dir, migration_name):
        date = get_applied_date(registry_dir, migration_name)
        click.echo(f"Error: migration '{migration_name}' was already applied on {date}.")
        click.echo("To re-run, manually remove the entry from registry/migrations.log.")
        sys.exit(1)

    result = execute_migration(migration, registry_dir, dry_run=not apply)

    if isinstance(result, MigrationDryRun):
        click.echo(f"DRY RUN — migration '{migration.name}'")
        click.echo(f"Description: {migration.description}")
        click.echo()
        click.echo(
            f"Would modify {result.affected_count} files, skip {result.skipped_count} files."
        )

        if result.sample_results:
            count = len(result.sample_results)
            click.echo(f"\nSample changes ({count} of {result.affected_count}):\n")
            for mr in result.sample_results:
                click.echo(f"  {mr.package}: {mr.description}")
            click.echo()

        click.echo("Re-run with --apply to execute.")
    else:
        click.echo(
            f"Applied migration '{migration.name}': "
            f"modified {result.modified_count} files, skipped {result.skipped_count} files."
        )

        if update_index and result.modified_count > 0:
            count = rebuild_index(registry_dir)
            click.echo(f"Index updated ({count} packages).")
