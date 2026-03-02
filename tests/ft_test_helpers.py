"""Shared test helpers for free-threading test modules."""

from __future__ import annotations

from typing import Any

from labeille.ft.results import (
    FTPackageResult,
    IterationOutcome,
)


def make_iteration(
    index: int = 1,
    status: str = "pass",
    exit_code: int = 0,
    duration_s: float = 10.0,
    **kwargs: Any,
) -> IterationOutcome:
    """Create an IterationOutcome with sensible defaults."""
    return IterationOutcome(
        index=index,
        status=status,
        exit_code=exit_code,
        duration_s=duration_s,
        **kwargs,
    )


def make_package_result(
    package: str = "test-pkg",
    statuses: list[str] | None = None,
    **kwargs: Any,
) -> FTPackageResult:
    """Create an FTPackageResult with specified iteration statuses.

    Example: make_package_result("foo", ["pass", "pass", "crash"])
    """
    if statuses is None:
        statuses = ["pass"] * 5

    iterations = [
        make_iteration(
            index=i + 1,
            status=s,
            exit_code=0 if s == "pass" else 1,
        )
        for i, s in enumerate(statuses)
    ]

    result = FTPackageResult(
        package=package,
        iterations=iterations,
        **kwargs,
    )
    result.compute_aggregates()
    result.categorize()
    return result
