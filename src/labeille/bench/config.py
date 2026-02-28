"""Benchmark configuration and condition profile loading.

Handles:
- Loading benchmark profiles from YAML files.
- Parsing inline condition definitions from CLI arguments.
- Merging CLI options with profile defaults.
- Resolving test commands per condition (override > prefix > suffix > registry).
- Validating the final configuration before execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labeille.bench.results import ConditionDef

log = logging.getLogger("labeille")


# ---------------------------------------------------------------------------
# BenchConfig
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    """Resolved configuration for a benchmark run."""

    # Identity
    bench_id: str = ""  # Auto-generated if empty
    name: str = ""  # Human-readable name
    description: str = ""

    # Conditions to compare
    conditions: dict[str, ConditionDef] = field(default_factory=dict)

    # Iteration control
    iterations: int = 5  # Number of measured iterations
    warmup: int = 1  # Number of warm-up iterations
    timeout: int = 600  # Per-iteration timeout in seconds

    # Execution strategy
    alternate: bool | None = None  # None = auto (True if multi-condition)
    interleave: bool = False

    # Package selection (same semantics as labeille run)
    packages_filter: list[str] | None = None
    top_n: int | None = None
    skip_packages: list[str] | None = None

    # Shared options (applied to ALL conditions unless overridden)
    default_target_python: str = ""
    default_env: dict[str, str] = field(default_factory=dict)
    default_extra_deps: list[str] = field(default_factory=list)
    default_test_command_suffix: str | None = None

    # Paths
    registry_dir: Path | None = None
    repos_dir: Path | None = None
    venvs_dir: Path | None = None
    results_dir: Path = field(default_factory=lambda: Path("results"))

    # Stability
    check_stability: bool = False
    wait_for_stability: bool = False
    max_load: float = 1.0
    min_available_ram_gb: float = 2.0

    # CLI provenance
    cli_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.bench_id:
            self.bench_id = f"bench_{time.strftime('%Y%m%d_%H%M%S')}"

    @property
    def total_iterations(self) -> int:
        """Total iterations per package per condition (warmup + measured)."""
        return self.warmup + self.iterations

    @property
    def should_alternate(self) -> bool:
        """Whether to alternate conditions per package."""
        if self.alternate is not None:
            return self.alternate
        # Auto: alternate if there are multiple conditions.
        return len(self.conditions) > 1

    @property
    def output_dir(self) -> Path:
        """The output directory for this benchmark run."""
        return self.results_dir / self.bench_id


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationError:
    """A single configuration validation error."""

    field: str
    message: str
    severity: str = "error"  # "error" or "warning"


def validate_config(config: BenchConfig) -> list[ValidationError]:
    """Validate a benchmark configuration.

    Returns a list of validation errors.  Empty list means valid.
    """
    errors: list[ValidationError] = []

    # Must have at least one condition.
    if not config.conditions:
        errors.append(
            ValidationError(
                field="conditions",
                message=(
                    "No benchmark conditions defined. "
                    "Use --profile or --condition to define at least one."
                ),
            )
        )

    # Each condition must have a target Python.
    for name, cond in config.conditions.items():
        python_path = cond.target_python or config.default_target_python
        if not python_path:
            errors.append(
                ValidationError(
                    field=f"conditions.{name}.target_python",
                    message=(
                        f"Condition '{name}' has no target Python. "
                        f"Set --target-python or specify it in the condition."
                    ),
                )
            )
        else:
            python = Path(python_path)
            if not python.exists():
                errors.append(
                    ValidationError(
                        field=f"conditions.{name}.target_python",
                        message=(
                            f"Target Python for condition '{name}' does not exist: {python_path}"
                        ),
                    )
                )
            elif not python.is_file():
                errors.append(
                    ValidationError(
                        field=f"conditions.{name}.target_python",
                        message=(
                            f"Target Python for condition '{name}' is not a file: {python_path}"
                        ),
                    )
                )

    # Iterations must be >= 3.
    if config.iterations < 3:
        errors.append(
            ValidationError(
                field="iterations",
                message=(
                    f"Need at least 3 measured iterations for "
                    f"meaningful statistics (got {config.iterations})."
                ),
            )
        )

    # Warmup must be >= 0.
    if config.warmup < 0:
        errors.append(
            ValidationError(
                field="warmup",
                message=f"Warmup iterations cannot be negative (got {config.warmup}).",
            )
        )

    # Registry dir must exist if specified.
    if config.registry_dir and not config.registry_dir.exists():
        errors.append(
            ValidationError(
                field="registry_dir",
                message=f"Registry directory does not exist: {config.registry_dir}",
            )
        )

    # Timeout must be positive.
    if config.timeout <= 0:
        errors.append(
            ValidationError(
                field="timeout",
                message=f"Timeout must be positive (got {config.timeout}).",
            )
        )

    # Interleave and alternate are mutually exclusive.
    if config.interleave and config.should_alternate:
        errors.append(
            ValidationError(
                field="interleave",
                message=(
                    "--interleave and --alternate are mutually exclusive. "
                    "Choose one execution strategy."
                ),
                severity="warning",
            )
        )

    # Condition names must be non-empty.
    for name in config.conditions:
        if not name or not name.strip():
            errors.append(
                ValidationError(
                    field="conditions",
                    message="Condition names must be non-empty.",
                )
            )

    return errors


# ---------------------------------------------------------------------------
# YAML profile loading
# ---------------------------------------------------------------------------


def load_profile(profile_path: Path) -> dict[str, Any]:
    """Load a benchmark profile from a YAML file.

    Profile format::

        name: "benchmark name"
        description: "optional description"
        iterations: 5
        warmup: 1
        timeout: 600

        conditions:
          baseline:
            description: "No coverage"
            target_python: "/usr/bin/python3.15"
            env:
              PYTHON_JIT: "1"
          coverage:
            description: "With coverage.py"
            extra_deps: ["coverage"]
            test_command_prefix: "coverage run -m"

    Returns:
        The parsed YAML as a dict.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for loading benchmark profiles. "
            "Install it with: pip install pyyaml"
        ) from exc

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    text = profile_path.read_text()
    data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Profile must be a YAML mapping, got {type(data).__name__}")

    return data


def config_from_profile(
    profile_data: dict[str, Any],
    *,
    cli_overrides: dict[str, Any] | None = None,
) -> BenchConfig:
    """Build a BenchConfig from a parsed YAML profile.

    CLI overrides take precedence over profile values for:
    iterations, warmup, timeout, target_python, registry_dir,
    repos_dir, venvs_dir, results_dir, packages_filter.

    Args:
        profile_data: Parsed YAML profile dict.
        cli_overrides: Dict of CLI option values that override
            profile defaults.  Keys match BenchConfig field names.

    Returns:
        BenchConfig with conditions and settings populated.
    """
    cli = cli_overrides or {}

    config = BenchConfig(
        name=cli.get("name") or profile_data.get("name", ""),
        description=profile_data.get("description", ""),
        iterations=(cli.get("iterations") or profile_data.get("iterations", 5)),
        warmup=(cli["warmup"] if cli.get("warmup") is not None else profile_data.get("warmup", 1)),
        timeout=cli.get("timeout") or profile_data.get("timeout", 600),
        default_target_python=(cli.get("target_python") or profile_data.get("target_python", "")),
    )

    # Parse conditions.
    conditions_data = profile_data.get("conditions", {})
    if not isinstance(conditions_data, dict):
        raise ValueError("Profile 'conditions' must be a mapping of condition_name -> definition")

    for name, cond_data in conditions_data.items():
        if cond_data is None:
            cond_data = {}
        if not isinstance(cond_data, dict):
            raise ValueError(
                f"Condition '{name}' must be a mapping, got {type(cond_data).__name__}"
            )

        config.conditions[name] = ConditionDef(
            name=name,
            description=cond_data.get("description", ""),
            target_python=cond_data.get("target_python", ""),
            env=cond_data.get("env", {}),
            extra_deps=cond_data.get("extra_deps", []),
            test_command_override=cond_data.get("test_command_override"),
            test_command_prefix=cond_data.get("test_command_prefix"),
            test_command_suffix=cond_data.get("test_command_suffix"),
            install_command=cond_data.get("install_command"),
        )

    # Apply CLI path overrides.
    if cli.get("registry_dir"):
        config.registry_dir = Path(cli["registry_dir"])
    elif profile_data.get("registry_dir"):
        config.registry_dir = Path(profile_data["registry_dir"])

    if cli.get("repos_dir"):
        config.repos_dir = Path(cli["repos_dir"])
    if cli.get("venvs_dir"):
        config.venvs_dir = Path(cli["venvs_dir"])
    if cli.get("results_dir"):
        config.results_dir = Path(cli["results_dir"])

    if cli.get("packages_filter"):
        config.packages_filter = cli["packages_filter"]
    if cli.get("top_n"):
        config.top_n = cli["top_n"]

    if cli.get("check_stability"):
        config.check_stability = True
    if cli.get("wait_for_stability"):
        config.wait_for_stability = True

    if cli.get("alternate") is not None:
        config.alternate = cli["alternate"]
    if cli.get("interleave"):
        config.interleave = True

    return config


# ---------------------------------------------------------------------------
# Inline condition parsing
# ---------------------------------------------------------------------------


def parse_inline_condition(spec: str) -> ConditionDef:
    """Parse an inline condition specification from CLI.

    Format: ``"name:key=value,key=value,..."``
    or just ``"name:"`` for defaults.

    Supported keys:
      target_python, env.KEY, extra_deps, test_command_override,
      test_command_prefix, test_prefix, test_command_suffix,
      test_suffix, install_command, description

    Examples::

        "baseline:"
        "coverage:extra_deps=coverage+pytest-cov,test_prefix=coverage run -m"
        "jit:target_python=/opt/python-jit/bin/python3,env.PYTHON_JIT=1"

    Returns:
        ConditionDef with parsed values.
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid condition spec: '{spec}'. Expected format: 'name:key=value,...'"
        )

    name, rest = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError("Condition name cannot be empty.")

    cond = ConditionDef(name=name)

    if not rest.strip():
        return cond  # Just a name, all defaults.

    pairs = _split_condition_pairs(rest.strip())

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid key=value pair in condition '{name}': '{pair}'")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "target_python":
            cond.target_python = value
        elif key == "extra_deps":
            cond.extra_deps = [d.strip() for d in value.split("+") if d.strip()]
        elif key == "test_command_override":
            cond.test_command_override = value
        elif key in ("test_prefix", "test_command_prefix"):
            cond.test_command_prefix = value
        elif key in ("test_suffix", "test_command_suffix"):
            cond.test_command_suffix = value
        elif key == "install_command":
            cond.install_command = value
        elif key == "description":
            cond.description = value
        elif key.startswith("env."):
            env_key = key[4:]
            cond.env[env_key] = value
        else:
            raise ValueError(
                f"Unknown condition key '{key}' in condition '{name}'. "
                f"Valid keys: target_python, extra_deps, test_command_override, "
                f"test_prefix, test_suffix, install_command, description, env.KEY"
            )

    return cond


def _split_condition_pairs(text: str) -> list[str]:
    """Split condition key=value pairs on commas.

    Values may contain spaces.  We split on commas and rejoin segments
    that don't contain ``=`` with the preceding segment (they are part
    of a value that contained a comma).
    """
    raw_parts = text.split(",")
    pairs: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if "=" in part and (not pairs or "=" in pairs[-1]):
            pairs.append(part)
        elif pairs:
            # Continuation of the previous value.
            pairs[-1] += "," + part
        else:
            pairs.append(part)
    return pairs


# ---------------------------------------------------------------------------
# Test command resolution
# ---------------------------------------------------------------------------


def resolve_test_command(
    registry_command: str | None,
    condition: ConditionDef,
    *,
    default_suffix: str | None = None,
) -> str:
    """Resolve the test command for a package under a condition.

    Resolution order (first match wins):

    1. ``condition.test_command_override`` — use as-is
    2. ``condition.test_command_prefix`` — prefix + registry command
    3. ``condition.test_command_suffix`` — registry command + suffix
    4. ``default_suffix`` — registry command + default suffix
    5. Registry command as-is
    6. Fallback: ``"python -m pytest"``

    Args:
        registry_command: The package's test command from the registry.
        condition: The condition being applied.
        default_suffix: Global suffix from BenchConfig.

    Returns:
        The resolved test command string.
    """
    base = registry_command or "python -m pytest"

    if condition.test_command_override:
        return condition.test_command_override

    if condition.test_command_prefix:
        prefix = condition.test_command_prefix
        if base.startswith("python -m "):
            rest = base[len("python -m ") :]
            return f"{prefix} {rest}"
        if base.startswith("python "):
            rest = base[len("python ") :]
            return f"{prefix} {rest}"
        return f"{prefix} {base}"

    cmd = base
    if condition.test_command_suffix:
        cmd = f"{cmd} {condition.test_command_suffix}"
    elif default_suffix:
        cmd = f"{cmd} {default_suffix}"

    return cmd


def resolve_target_python(
    condition: ConditionDef,
    default: str,
) -> Path:
    """Resolve the target Python for a condition.

    Returns:
        Path to the Python executable.

    Raises:
        ValueError: If no target Python can be determined.
    """
    python_str = condition.target_python or default
    if not python_str:
        raise ValueError(
            f"No target Python for condition '{condition.name}'. "
            f"Set --target-python or specify target_python in the condition definition."
        )
    return Path(python_str)


def resolve_env(
    condition: ConditionDef,
    default_env: dict[str, str],
) -> dict[str, str]:
    """Merge default env with condition-specific env.

    Condition env takes precedence over defaults.
    """
    env = dict(default_env)
    env.update(condition.env)
    return env


def resolve_extra_deps(
    condition: ConditionDef,
    default_deps: list[str],
) -> list[str]:
    """Merge default extra deps with condition-specific deps.

    Returns a deduplicated list, preserving order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for dep in default_deps + condition.extra_deps:
        if dep not in seen:
            seen.add(dep)
            result.append(dep)
    return result
