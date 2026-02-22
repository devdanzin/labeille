"""Command-line interface for labeille.

Provides the main CLI entry point with ``resolve`` and ``run`` subcommands.
"""

import click

from labeille import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """labeille — Hunt for CPython JIT bugs by running real-world test suites."""


@main.command()
def resolve() -> None:
    """Resolve PyPI packages to source repositories and build a test registry."""
    click.echo("resolve: not yet implemented")


@main.command()
def run() -> None:
    """Run test suites against a JIT-enabled Python build and detect crashes."""
    click.echo("run: not yet implemented")
