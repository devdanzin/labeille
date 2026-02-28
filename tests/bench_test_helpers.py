"""Shared test fixtures for benchmark tests."""

from __future__ import annotations

from labeille.bench.results import (
    BenchConditionResult,
    BenchIteration,
    BenchMeta,
    BenchPackageResult,
    ConditionDef,
)
from labeille.bench.system import SystemProfile


def make_iterations(
    wall_times: list[float],
    *,
    warmup_count: int = 0,
    status: str = "ok",
) -> list[BenchIteration]:
    """Create a list of BenchIterations from wall times."""
    iterations = []
    for i, wt in enumerate(wall_times):
        iterations.append(
            BenchIteration(
                index=i + 1,
                warmup=i < warmup_count,
                wall_time_s=wt,
                user_time_s=wt * 0.8,
                sys_time_s=wt * 0.1,
                peak_rss_mb=256.0,
                exit_code=0 if status == "ok" else 1,
                status=status,
            )
        )
    return iterations


def make_condition_result(
    name: str,
    wall_times: list[float],
    *,
    warmup_count: int = 0,
    status: str = "ok",
) -> BenchConditionResult:
    """Create a BenchConditionResult with computed stats."""
    cond = BenchConditionResult(condition_name=name)
    cond.iterations = make_iterations(
        wall_times,
        warmup_count=warmup_count,
        status=status,
    )
    cond.compute_stats()
    return cond


def make_package_result(
    name: str,
    conditions: dict[str, list[float]],
    *,
    warmup_count: int = 0,
) -> BenchPackageResult:
    """Create a BenchPackageResult from condition name -> wall times."""
    pkg = BenchPackageResult(package=name)
    for cond_name, wall_times in conditions.items():
        pkg.conditions[cond_name] = make_condition_result(
            cond_name,
            wall_times,
            warmup_count=warmup_count,
        )
    return pkg


def make_meta(
    *,
    name: str = "Test Benchmark",
    conditions: list[str] | None = None,
) -> BenchMeta:
    """Create a minimal BenchMeta."""
    cond_names = conditions or ["baseline"]
    return BenchMeta(
        bench_id="bench_test_001",
        name=name,
        system=SystemProfile(
            cpu_model="Test CPU",
            cpu_cores_physical=4,
            cpu_cores_logical=8,
            ram_total_gb=16.0,
            ram_available_gb=12.0,
            os_distro="Test Linux",
            hostname="testhost",
        ),
        conditions={n: ConditionDef(name=n) for n in cond_names},
        config={
            "iterations": 5,
            "warmup": 1,
        },
        packages_total=3,
        packages_completed=3,
    )
