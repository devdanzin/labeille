"""Shared CLI parsing helpers.

Small utilities used by multiple CLI modules (``cli.py``, ``registry_cli.py``,
``bench_cli.py``, ``ft_cli.py``, ``compat_cli.py``).  Lives in its own module
to avoid circular imports — ``cli.py`` registers subcommand groups from those
modules at import time.
"""

from __future__ import annotations

import click


def parse_env_pairs(env_pairs: tuple[str, ...]) -> dict[str, str]:
    """Parse KEY=VALUE environment variable pairs from CLI --env options."""
    result: dict[str, str] = {}
    for pair in env_pairs:
        if "=" not in pair:
            raise click.UsageError(f"Invalid --env format (expected KEY=VALUE): {pair}")
        key, _, value = pair.partition("=")
        result[key] = value
    return result


def parse_csv_list(csv: str | None) -> list[str]:
    """Parse a comma-separated string into a list, stripping whitespace."""
    if not csv:
        return []
    return [item.strip() for item in csv.split(",") if item.strip()]
