"""Cross-run comparison for free-threading results.

Stub module — full implementation in prompt 29.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FTRunComparison:
    """Comparison between two free-threading runs."""

    improved: list[dict[str, Any]] = field(default_factory=list)
    regressed: list[dict[str, Any]] = field(default_factory=list)
    unchanged: list[dict[str, Any]] = field(default_factory=list)


def compare_ft_runs(
    results_a: list[Any],
    results_b: list[Any],
) -> FTRunComparison:
    """Compare two sets of free-threading results.

    Stub — returns empty comparison. Full implementation in prompt 29.
    """
    return FTRunComparison()
