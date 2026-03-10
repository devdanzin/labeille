"""Benchmark execution engine.

Orchestrates:
1. System profiling and stability checks
2. Package selection and repo cloning
3. Venv creation and package installation (per condition)
4. Iteration execution with timing capture
5. Incremental result writing
6. Progress reporting

Execution strategies:
- Block (default for single condition): run all iterations of
  each package before moving to the next.
- Alternating (default for multi-condition): for each package,
  alternate between conditions per iteration.
- Interleaved: run iteration 1 of all packages, then iteration 2
  of all packages, etc.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from labeille.bench.config import (
    BenchConfig,
    resolve_constraints,
    resolve_env,
    resolve_extra_deps,
    resolve_target_python,
    resolve_test_command,
    validate_config,
)
from labeille.bench.stats import relative_standard_error
from labeille.bench.results import (
    BenchConditionResult,
    BenchIteration,
    BenchMeta,
    BenchPackageResult,
    append_package_result,
    save_bench_run,
)
from labeille.bench.system import (
    SystemSnapshot,
    capture_python_profile,
    capture_system_profile,
    check_stability,
    format_python_profile,
    format_system_profile,
)
from labeille.bench.cache import check_cache_drop_available, check_not_root, drop_caches
from labeille.bench.constraints import (
    apply_constraints_to_command,
    check_constraints_available,
    detect_oom_from_result,
)
from labeille.bench.timing import (
    parse_pytest_durations,
    prepare_per_test_command,
    run_timed_in_venv,
)
from labeille.io_utils import utc_now_iso
from labeille.logging import get_logger

log = get_logger("bench.runner")


# ---------------------------------------------------------------------------
# Install environment
# ---------------------------------------------------------------------------


def _build_install_env(
    condition_env: dict[str, str],
    venv_dir: Path,
) -> dict[str, str]:
    """Build a complete environment for package installation.

    Starts from the inherited ``os.environ`` (so tools like ``git``,
    ``cc``, etc. are found), strips Python-specific pollution, prepends
    the venv ``bin/`` to ``PATH``, and layers condition-specific
    variables on top.
    """
    from labeille.runner import clean_env

    env = clean_env()

    venv_bin = str(venv_dir / "bin")
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(venv_dir)

    env.update(condition_env)
    return env


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


@dataclass
class BenchProgress:
    """Progress info passed to the callback."""

    phase: str  # "setup", "warmup", "measure", "done"
    package: str
    condition: str
    iteration: int  # 1-based
    total_iterations: int
    packages_done: int
    packages_total: int
    wall_time_s: float = 0.0
    status: str = ""
    detail: str = ""


# Type alias for the progress callback.
ProgressCallback = Any  # Callable[[BenchProgress], None] | None


# ---------------------------------------------------------------------------
# BenchRunner
# ---------------------------------------------------------------------------


class BenchRunner:
    """Executes a benchmark run according to a BenchConfig.

    Usage::

        config = BenchConfig(...)
        runner = BenchRunner(config)
        meta, results = runner.run()
    """

    def __init__(
        self,
        config: BenchConfig,
        progress_callback: ProgressCallback = None,
    ) -> None:
        self.config = config
        self.progress: Any = progress_callback or self._default_progress
        self._meta: BenchMeta | None = None
        self._results: list[BenchPackageResult] = []

    def run(self) -> tuple[BenchMeta, list[BenchPackageResult]]:
        """Execute the full benchmark.

        Returns:
            Tuple of (BenchMeta, list of BenchPackageResult).

        Raises:
            ValueError: If configuration is invalid.
            SystemExit: If stability check fails and wait is not set.
        """
        # Phase 1: Validate configuration.
        errors = validate_config(self.config)
        fatal = [e for e in errors if e.severity == "error"]
        warnings = [e for e in errors if e.severity == "warning"]
        for w in warnings:
            log.warning("Config warning: %s: %s", w.field, w.message)
        if fatal:
            messages = [f"  {e.field}: {e.message}" for e in fatal]
            raise ValueError("Invalid benchmark configuration:\n" + "\n".join(messages))

        # Phase 2: System profiling.
        log.info("Capturing system profile...")
        system_profile = capture_system_profile()
        log.info("\n%s", format_system_profile(system_profile))

        # Profile each unique target Python.
        python_profiles: dict[str, Any] = {}
        seen_pythons: dict[str, str] = {}  # path → condition name
        for name, cond in self.config.conditions.items():
            python_path = resolve_target_python(cond, self.config.default_target_python)
            path_str = str(python_path)
            if path_str not in seen_pythons:
                env = resolve_env(cond, self.config.default_env)
                pp = capture_python_profile(python_path, env=env)
                python_profiles[name] = pp
                seen_pythons[path_str] = name
                log.info(
                    "Condition '%s':\n  %s",
                    name,
                    format_python_profile(pp).replace("\n", "\n  "),
                )
            else:
                # Same Python as another condition — reference it.
                python_profiles[name] = python_profiles[seen_pythons[path_str]]

        # Phase 3: Stability check.
        if self.config.check_stability or self.config.wait_for_stability:
            stability = check_stability(
                max_load=self.config.max_load,
                min_available_ram_gb=self.config.min_available_ram_gb,
            )
            for warn_msg in stability.warnings:
                log.warning("Stability: %s", warn_msg)
            if not stability.stable:
                for e in stability.errors:
                    log.error("Stability: %s", e)
                if self.config.wait_for_stability:
                    self._wait_for_stability()
                else:
                    raise SystemExit(
                        "System is not stable for benchmarking. "
                        "Use --wait-for-stability to wait, or "
                        "reduce --max-load threshold."
                    )

        # Phase 3b: Root check and cache-drop validation.
        check_not_root(allow_root=self.config.run_dangerously_as_root)
        if self.config.drop_caches:
            status = check_cache_drop_available()
            if not status.available:
                raise SystemExit(status.message)
            log.info("Cache dropping enabled.")

        # Phase 3c: Validate resource constraints.
        for cond_name, cond in self.config.conditions.items():
            resolved = resolve_constraints(cond, self.config.default_constraints)
            if resolved and resolved.has_any:
                constraint_warnings = check_constraints_available(resolved)
                for cw in constraint_warnings:
                    log.warning("Constraints [%s]: %s", cond_name, cw)

        # Phase 4: Prepare output directory and metadata.
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "bench_results.jsonl"

        meta = BenchMeta(
            bench_id=self.config.bench_id,
            name=self.config.name,
            description=self.config.description,
            system=system_profile,
            python_profiles=python_profiles,
            conditions={name: cond for name, cond in self.config.conditions.items()},
            config={
                "iterations": self.config.iterations,
                "warmup": self.config.warmup,
                "timeout": self.config.timeout,
                "alternate": self.config.should_alternate,
                "interleave": self.config.interleave,
                "packages_filter": self.config.packages_filter,
                "top_n": self.config.top_n,
                "adaptive": self.config.adaptive,
                "adaptive_threshold": self.config.adaptive_threshold,
                "adaptive_min_iterations": self.config.adaptive_min_iterations,
            },
            cli_args=self.config.cli_args,
            start_time=utc_now_iso(),
        )
        self._meta = meta

        # Phase 5: Load packages from registry.
        packages = self._load_packages()
        meta.packages_total = len(packages)
        log.info("Benchmarking %d packages", len(packages))

        # Write initial metadata (will be updated at the end).
        meta_path = output_dir / "bench_meta.json"
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8")

        # Phase 6: Execute benchmarks.
        if self.config.interleave:
            results = self._run_interleaved(packages, results_path)
        else:
            results = self._run_sequential(packages, results_path)

        # Phase 7: Finalize.
        meta.end_time = utc_now_iso()
        meta.packages_completed = sum(1 for r in results if not r.skipped)
        meta.packages_skipped = sum(1 for r in results if r.skipped)

        # Rewrite metadata with final stats.
        save_bench_run(output_dir, meta, results)
        log.info("Benchmark complete: %s", output_dir)

        self._meta = meta
        self._results = results
        return meta, results

    def _load_packages(self) -> list[Any]:
        """Load package entries from the registry.

        Applies package filters and top_n selection.
        """
        from labeille.registry import load_index, load_package

        if not self.config.registry_dir:
            raise ValueError(
                "Registry directory is required for benchmarking. "
                "Use --registry-dir to specify it."
            )

        index = load_index(self.config.registry_dir)
        packages = []
        index_names = {entry.name for entry in index.packages}

        # Apply package filter.
        if self.config.packages_filter:
            names = list(self.config.packages_filter)
        else:
            names = list(index_names)

        # Apply skip filter.
        if self.config.skip_packages:
            skip_set = set(self.config.skip_packages)
            names = [n for n in names if n not in skip_set]

        # Load each package.
        for name in sorted(names):
            if name not in index_names:
                log.warning("Package '%s' not in registry, skipping", name)
                continue
            try:
                entry = load_package(name, self.config.registry_dir)
                packages.append(entry)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to load package '%s': %s", name, exc)

        # Apply top_n.
        if self.config.top_n and len(packages) > self.config.top_n:
            packages = packages[: self.config.top_n]

        return packages

    def _run_sequential(
        self,
        packages: list[Any],
        results_path: Path,
    ) -> list[BenchPackageResult]:
        """Run benchmarks sequentially: all iterations of one package
        before moving to the next.

        If should_alternate is True, alternates conditions within
        each package's iterations.
        """
        results: list[BenchPackageResult] = []

        for pkg_idx, pkg in enumerate(packages):
            pkg_result = self._benchmark_package(pkg, pkg_idx, len(packages))
            results.append(pkg_result)
            append_package_result(results_path, pkg_result)

        return results

    def _run_interleaved(
        self,
        packages: list[Any],
        results_path: Path,
    ) -> list[BenchPackageResult]:
        """Run benchmarks interleaved: iteration 1 of all packages,
        then iteration 2, etc.
        """
        # Pre-setup: clone repos and create venvs for all packages.
        log.info("Pre-setup phase: cloning repos and creating venvs...")
        pkg_setups: list[dict[str, Any]] = []
        pkg_results: list[BenchPackageResult] = []

        for pkg in packages:
            setup = self._setup_package(pkg)
            if setup is None:
                pkg_result = BenchPackageResult(
                    package=pkg.package,
                    skipped=True,
                    skip_reason="setup failed",
                )
                pkg_results.append(pkg_result)
                pkg_setups.append({})
                continue

            pkg_result = BenchPackageResult(
                package=pkg.package,
                clone_duration_s=setup.get("clone_duration", 0.0),
            )
            for cond_name in self.config.conditions:
                pkg_result.conditions[cond_name] = BenchConditionResult(
                    condition_name=cond_name,
                    install_duration_s=setup.get(f"install_{cond_name}", 0.0),
                    venv_setup_duration_s=setup.get(f"venv_{cond_name}", 0.0),
                )
            pkg_results.append(pkg_result)
            pkg_setups.append(setup)

        # Now run iterations.
        total_iter = self.config.total_iterations
        condition_names = list(self.config.conditions.keys())

        # Track which packages have converged (for adaptive mode).
        converged_packages: set[int] = set()

        for iter_idx in range(total_iter):
            is_warmup = iter_idx < self.config.warmup
            phase = "warmup" if is_warmup else "measure"
            log.info("Iteration %d/%d (%s)...", iter_idx + 1, total_iter, phase)

            for pkg_idx, (pkg, setup, pkg_result) in enumerate(
                zip(packages, pkg_setups, pkg_results),
            ):
                if pkg_result.skipped or not setup or pkg_idx in converged_packages:
                    continue

                for cond_name in condition_names:
                    cond = self.config.conditions[cond_name]
                    iteration = self._run_iteration(
                        pkg=pkg,
                        cond=cond,
                        setup=setup,
                        iter_index=iter_idx + 1,
                        is_warmup=is_warmup,
                    )
                    pkg_result.conditions[cond_name].iterations.append(iteration)

                    self.progress(
                        BenchProgress(
                            phase=phase,
                            package=pkg.package,
                            condition=cond_name,
                            iteration=iter_idx + 1,
                            total_iterations=total_iter,
                            packages_done=pkg_idx,
                            packages_total=len(packages),
                            wall_time_s=iteration.wall_time_s,
                            status=iteration.status,
                        )
                    )

                # Adaptive convergence: check after each full round of conditions.
                if self.config.adaptive and not is_warmup:
                    all_converged = all(
                        self._check_convergence(pkg_result.conditions[cn])
                        for cn in condition_names
                    )
                    if all_converged:
                        for cn in condition_names:
                            pkg_result.conditions[cn].converged_early = True
                        converged_packages.add(pkg_idx)
                        log.info(
                            "Adaptive: %s converged after %d measured iterations",
                            pkg.package,
                            iter_idx + 1 - self.config.warmup,
                        )

        # Compute stats and write results.
        results: list[BenchPackageResult] = []
        for pkg_result in pkg_results:
            for cond_result in pkg_result.conditions.values():
                cond_result.compute_stats()
            results.append(pkg_result)
            append_package_result(results_path, pkg_result)

        return results

    def _benchmark_package(
        self,
        pkg: Any,
        pkg_idx: int,
        pkg_total: int,
    ) -> BenchPackageResult:
        """Benchmark a single package across all conditions."""
        pkg_result = BenchPackageResult(package=pkg.package)

        # Setup phase.
        setup = self._setup_package(pkg)
        if setup is None:
            pkg_result.skipped = True
            pkg_result.skip_reason = "setup failed"
            return pkg_result

        pkg_result.clone_duration_s = setup.get("clone_duration", 0.0)

        # Initialize condition results.
        for cond_name in self.config.conditions:
            pkg_result.conditions[cond_name] = BenchConditionResult(
                condition_name=cond_name,
                install_duration_s=setup.get(f"install_{cond_name}", 0.0),
                venv_setup_duration_s=setup.get(f"venv_{cond_name}", 0.0),
            )

        # Run iterations.
        total_iter = self.config.total_iterations
        condition_names = list(self.config.conditions.keys())

        if self.config.should_alternate and len(condition_names) > 1:
            # Alternating: A1, B1, A2, B2, ...
            for iter_idx in range(total_iter):
                is_warmup = iter_idx < self.config.warmup
                phase = "warmup" if is_warmup else "measure"

                for cond_name in condition_names:
                    cond = self.config.conditions[cond_name]
                    iteration = self._run_iteration(
                        pkg=pkg,
                        cond=cond,
                        setup=setup,
                        iter_index=iter_idx + 1,
                        is_warmup=is_warmup,
                    )
                    pkg_result.conditions[cond_name].iterations.append(iteration)

                    self.progress(
                        BenchProgress(
                            phase=phase,
                            package=pkg.package,
                            condition=cond_name,
                            iteration=iter_idx + 1,
                            total_iterations=total_iter,
                            packages_done=pkg_idx,
                            packages_total=pkg_total,
                            wall_time_s=iteration.wall_time_s,
                            status=iteration.status,
                        )
                    )

                # Adaptive convergence: check after each full round of conditions.
                if self.config.adaptive and not is_warmup:
                    all_converged = all(
                        self._check_convergence(pkg_result.conditions[cn])
                        for cn in condition_names
                    )
                    if all_converged:
                        for cn in condition_names:
                            pkg_result.conditions[cn].converged_early = True
                        log.info(
                            "Adaptive: %s converged after %d measured iterations",
                            pkg.package,
                            iter_idx + 1 - self.config.warmup,
                        )
                        break
        else:
            # Block: all iterations of condition A, then B.
            for cond_name in condition_names:
                cond = self.config.conditions[cond_name]
                for iter_idx in range(total_iter):
                    is_warmup = iter_idx < self.config.warmup
                    phase = "warmup" if is_warmup else "measure"

                    iteration = self._run_iteration(
                        pkg=pkg,
                        cond=cond,
                        setup=setup,
                        iter_index=iter_idx + 1,
                        is_warmup=is_warmup,
                    )
                    pkg_result.conditions[cond_name].iterations.append(iteration)

                    self.progress(
                        BenchProgress(
                            phase=phase,
                            package=pkg.package,
                            condition=cond_name,
                            iteration=iter_idx + 1,
                            total_iterations=total_iter,
                            packages_done=pkg_idx,
                            packages_total=pkg_total,
                            wall_time_s=iteration.wall_time_s,
                            status=iteration.status,
                        )
                    )

                    # Adaptive convergence: check after each measured iteration.
                    if (
                        self.config.adaptive
                        and not is_warmup
                        and self._check_convergence(pkg_result.conditions[cond_name])
                    ):
                        pkg_result.conditions[cond_name].converged_early = True
                        log.info(
                            "Adaptive: %s/%s converged after %d measured iterations",
                            pkg.package,
                            cond_name,
                            iter_idx + 1 - self.config.warmup,
                        )
                        break

        # Compute stats.
        for cond_result in pkg_result.conditions.values():
            cond_result.compute_stats()

        return pkg_result

    def _setup_package(self, pkg: Any) -> dict[str, Any] | None:
        """Clone repo and create venvs for all conditions.

        Returns a setup dict with repo_dir, venvs, and durations.
        Returns None if setup fails.
        """
        from labeille.runner import (
            clone_repo,
            create_venv,
            install_package,
            install_with_fallback,
            resolve_installer,
        )

        setup: dict[str, Any] = {}

        # Clone the repo.
        if not getattr(pkg, "repo", None):
            log.warning("Package '%s' has no repo URL, skipping", pkg.package)
            return None

        repos_dir = self.config.repos_dir or (self.config.output_dir / "repos")
        repos_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = repos_dir / pkg.package

        clone_start = time.monotonic()
        try:
            if repo_dir.exists():
                from labeille.runner import pull_repo

                pull_repo(repo_dir)
            else:
                clone_repo(pkg.repo, repo_dir)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to clone %s: %s", pkg.package, exc)
            return None
        setup["clone_duration"] = time.monotonic() - clone_start
        setup["repo_dir"] = repo_dir

        # Resolve installer backend.
        installer = resolve_installer(self.config.installer)

        # Create a venv per condition.
        venvs: dict[str, Path] = {}
        venvs_base = self.config.venvs_dir or (self.config.output_dir / "venvs")
        venvs_base.mkdir(parents=True, exist_ok=True)

        for cond_name, cond in self.config.conditions.items():
            venv_dir = venvs_base / f"{pkg.package}_{cond_name}"
            python_path = resolve_target_python(cond, self.config.default_target_python)

            # Create venv.
            venv_start = time.monotonic()
            try:
                if venv_dir.exists():
                    import shutil

                    shutil.rmtree(venv_dir)
                create_venv(python_path, venv_dir, installer)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failed to create venv for %s/%s: %s",
                    pkg.package,
                    cond_name,
                    exc,
                )
                return None
            setup[f"venv_{cond_name}"] = time.monotonic() - venv_start

            # Install the package.
            venv_python = venv_dir / "bin" / "python"
            install_env = _build_install_env(
                resolve_env(cond, self.config.default_env),
                venv_dir,
            )
            install_cmd = (
                cond.install_command or getattr(pkg, "install_command", None) or "pip install -e ."
            )

            install_start = time.monotonic()
            try:
                result, actual_backend = install_with_fallback(
                    python_path,
                    venv_dir,
                    install_cmd,
                    repo_dir,
                    install_env,
                    self.config.timeout,
                    installer,
                )
                venv_python = venv_dir / "bin" / "python"  # refresh after possible recreate
                if hasattr(result, "returncode") and result.returncode != 0:
                    log.error(
                        "Install failed for %s/%s (exit %d)",
                        pkg.package,
                        cond_name,
                        result.returncode,
                    )
                    return None
            except Exception as exc:  # noqa: BLE001
                log.error("Install failed for %s/%s: %s", pkg.package, cond_name, exc)
                return None
            setup[f"install_{cond_name}"] = time.monotonic() - install_start

            # Install extra deps.
            extra_deps = resolve_extra_deps(cond, self.config.default_extra_deps)
            if extra_deps:
                try:
                    deps_str = " ".join(extra_deps)
                    install_package(
                        venv_python,
                        f"pip install {deps_str}",
                        cwd=repo_dir,
                        env=install_env,
                        timeout=self.config.timeout,
                        installer=actual_backend,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Extra deps install warning for %s/%s: %s",
                        pkg.package,
                        cond_name,
                        exc,
                    )

            venvs[cond_name] = venv_dir

        setup["venvs"] = venvs
        return setup

    def _run_iteration(
        self,
        *,
        pkg: Any,
        cond: Any,
        setup: dict[str, Any],
        iter_index: int,
        is_warmup: bool,
    ) -> BenchIteration:
        """Execute a single timed iteration."""
        repo_dir: Path = setup["repo_dir"]
        venvs: dict[str, Path] = setup["venvs"]
        venv_path = venvs[cond.name]

        # Resolve the test command.
        registry_cmd = getattr(pkg, "test_command", None)
        test_cmd = resolve_test_command(
            registry_cmd,
            cond,
            default_suffix=self.config.default_test_command_suffix,
        )

        # Per-test timing: modify command if enabled.
        per_test_enabled = False
        if self.config.per_test_timing:
            test_framework = getattr(pkg, "test_framework", "") or ""
            test_cmd, per_test_enabled = prepare_per_test_command(test_cmd, test_framework)

        # Apply resource constraints.
        constraints = resolve_constraints(cond, self.config.default_constraints)
        constraints_applied = constraints is not None and constraints.has_any
        if constraints_applied:
            test_cmd = apply_constraints_to_command(test_cmd, constraints)

        # Build environment.
        env = resolve_env(cond, self.config.default_env)
        env.setdefault("PYTHONFAULTHANDLER", "1")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

        # Drop caches if enabled.
        caches_dropped = False
        if self.config.drop_caches:
            caches_dropped = drop_caches(
                allow_root=self.config.run_dangerously_as_root,
            )
            if not caches_dropped:
                log.warning("Cache drop failed for iteration %d, continuing.", iter_index)

        # Capture pre-iteration system state.
        snap_before = SystemSnapshot.capture()

        # Run the timed iteration.
        timed = run_timed_in_venv(
            venv_path=venv_path,
            test_command=test_cmd,
            cwd=repo_dir,
            env=env,
            timeout=self.config.timeout,
        )

        # Capture post-iteration system state.
        snap_after = SystemSnapshot.capture()

        # Classify the result.
        status: Literal["ok", "fail", "timeout", "error", "oom"]
        if timed.timed_out:
            status = "timeout"
        elif timed.exit_code == 0:
            status = "ok"
        elif timed.exit_code < 0:
            status = "error"
        else:
            status = "fail"

        # OOM detection.
        oom_detected = detect_oom_from_result(timed.exit_code, timed.stderr, constraints)
        if oom_detected:
            status = "oom"

        # Parse per-test timings if enabled.
        per_test_timings = None
        if per_test_enabled:
            per_test_timings = parse_pytest_durations(timed.stdout)

        return BenchIteration(
            index=iter_index,
            warmup=is_warmup,
            wall_time_s=timed.wall_time_s,
            user_time_s=timed.user_time_s,
            sys_time_s=timed.sys_time_s,
            peak_rss_mb=timed.peak_rss_mb,
            exit_code=timed.exit_code,
            status=status,
            load_avg_start=snap_before.load_avg_1m,
            load_avg_end=snap_after.load_avg_1m,
            ram_available_start_gb=snap_before.ram_available_gb,
            caches_dropped=caches_dropped,
            constraints_applied=constraints_applied,
            oom_detected=oom_detected,
            per_test_timings=per_test_timings,
        )

    def _check_convergence(
        self,
        cond_result: BenchConditionResult,
    ) -> bool:
        """Check if wall-time measurements have converged.

        Convergence is reached when the Relative Standard Error (RSE)
        of measured wall times drops below the adaptive threshold.

        Args:
            cond_result: The condition result to check.

        Returns:
            True if RSE is below the threshold.
        """
        measured = [it for it in cond_result.iterations if not it.warmup]
        n = len(measured)
        if n < self.config.adaptive_min_iterations:
            return False
        wall_times = [it.wall_time_s for it in measured]
        rse = relative_standard_error(wall_times)
        if rse != rse:  # NaN check
            return False
        return rse < self.config.adaptive_threshold

    def _wait_for_stability(self) -> None:
        """Block until the system meets stability criteria."""
        log.info(
            "Waiting for system stability (load < %.1f, RAM > %.1f GB)...",
            self.config.max_load,
            self.config.min_available_ram_gb,
        )
        max_wait = 300  # 5 minutes max.
        interval = 10  # Check every 10 seconds.
        waited = 0

        while waited < max_wait:
            stability = check_stability(
                max_load=self.config.max_load,
                min_available_ram_gb=self.config.min_available_ram_gb,
            )
            if stability.stable:
                log.info("System stabilized after %d seconds.", waited)
                return
            time.sleep(interval)
            waited += interval
            if waited % 60 == 0:
                log.info("Still waiting... (%d/%d seconds)", waited, max_wait)

        log.warning(
            "System did not stabilize within %d seconds. Proceeding anyway.",
            max_wait,
        )

    @staticmethod
    def _default_progress(progress: BenchProgress) -> None:
        """Default progress callback: log to stderr."""
        if progress.phase == "warmup":
            marker = "W"
        elif progress.phase == "measure":
            marker = "M"
        else:
            marker = " "

        pkg_progress = f"[{progress.packages_done + 1}/{progress.packages_total}]"

        line = (
            f"  {pkg_progress} {progress.package:30s} "
            f"{progress.condition:15s} "
            f"{marker}{progress.iteration}/{progress.total_iterations} "
        )
        if progress.wall_time_s:
            line += f"{progress.wall_time_s:8.2f}s "
        if progress.status:
            line += f"[{progress.status}]"

        log.info(line)


# ---------------------------------------------------------------------------
# Quick mode helper
# ---------------------------------------------------------------------------


def quick_config(config: BenchConfig) -> BenchConfig:
    """Apply quick mode settings for rapid iteration.

    Reduces iterations to 3, warmup to 0, limits to top 20
    packages, and enables adaptive convergence.  Useful during
    development and testing.
    """
    config.iterations = 3
    config.warmup = 0
    config.top_n = min(config.top_n or 20, 20)
    config.adaptive = True
    config.name = f"{config.name} (quick)" if config.name else "Quick benchmark"
    return config
