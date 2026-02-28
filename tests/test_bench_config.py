"""Tests for labeille.bench.config — benchmark configuration and condition profiles."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from labeille.bench.config import (
    BenchConfig,
    _split_condition_pairs,
    config_from_profile,
    load_profile,
    parse_inline_condition,
    resolve_env,
    resolve_extra_deps,
    resolve_target_python,
    resolve_test_command,
    validate_config,
)
from labeille.bench.results import ConditionDef


# ---------------------------------------------------------------------------
# BenchConfig tests
# ---------------------------------------------------------------------------


class TestBenchConfig(unittest.TestCase):
    """Tests for BenchConfig dataclass."""

    def test_config_defaults(self) -> None:
        """Default config values should be sensible."""
        config = BenchConfig()
        self.assertEqual(config.iterations, 5)
        self.assertEqual(config.warmup, 1)
        self.assertEqual(config.timeout, 600)
        self.assertFalse(config.interleave)
        self.assertIsNone(config.alternate)

    def test_config_bench_id_auto(self) -> None:
        """Auto-generated bench_id should start with 'bench_'."""
        config = BenchConfig()
        self.assertTrue(config.bench_id.startswith("bench_"))

    def test_config_bench_id_explicit(self) -> None:
        """Explicit bench_id should be preserved."""
        config = BenchConfig(bench_id="my_bench")
        self.assertEqual(config.bench_id, "my_bench")

    def test_config_total_iterations(self) -> None:
        """total_iterations = warmup + iterations."""
        config = BenchConfig(iterations=5, warmup=2)
        self.assertEqual(config.total_iterations, 7)

    def test_config_should_alternate_auto_single(self) -> None:
        """Single condition: should_alternate auto-resolves to False."""
        config = BenchConfig()
        config.conditions["baseline"] = ConditionDef(name="baseline")
        self.assertFalse(config.should_alternate)

    def test_config_should_alternate_auto_multi(self) -> None:
        """Multiple conditions: should_alternate auto-resolves to True."""
        config = BenchConfig()
        config.conditions["baseline"] = ConditionDef(name="baseline")
        config.conditions["jit"] = ConditionDef(name="jit")
        self.assertTrue(config.should_alternate)

    def test_config_should_alternate_explicit(self) -> None:
        """Explicit alternate=False overrides auto."""
        config = BenchConfig(alternate=False)
        config.conditions["a"] = ConditionDef(name="a")
        config.conditions["b"] = ConditionDef(name="b")
        self.assertFalse(config.should_alternate)

    def test_config_output_dir(self) -> None:
        """output_dir should be results_dir / bench_id."""
        config = BenchConfig(bench_id="x", results_dir=Path("/tmp"))
        self.assertEqual(config.output_dir, Path("/tmp/x"))


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateConfig(unittest.TestCase):
    """Tests for validate_config()."""

    def _valid_config(self) -> BenchConfig:
        """Create a minimal valid config using sys.executable."""
        config = BenchConfig(bench_id="test")
        config.conditions["baseline"] = ConditionDef(
            name="baseline",
            target_python=sys.executable,
        )
        return config

    def test_validate_no_conditions(self) -> None:
        """Empty conditions should produce an error."""
        config = BenchConfig(bench_id="test")
        errors = validate_config(config)
        self.assertTrue(any(e.field == "conditions" for e in errors))

    def test_validate_missing_python(self) -> None:
        """Condition without target_python and no default should error."""
        config = BenchConfig(bench_id="test")
        config.conditions["base"] = ConditionDef(name="base")
        errors = validate_config(config)
        self.assertTrue(any("target python" in e.message.lower() for e in errors))

    def test_validate_python_not_found(self) -> None:
        """Nonexistent target Python should error."""
        config = BenchConfig(bench_id="test")
        config.conditions["base"] = ConditionDef(
            name="base",
            target_python="/nonexistent/python3.99",
        )
        errors = validate_config(config)
        self.assertTrue(any("does not exist" in e.message for e in errors))

    def test_validate_iterations_too_low(self) -> None:
        """iterations < 3 should produce an error."""
        config = self._valid_config()
        config.iterations = 2
        errors = validate_config(config)
        self.assertTrue(any(e.field == "iterations" for e in errors))

    def test_validate_negative_warmup(self) -> None:
        """Negative warmup should produce an error."""
        config = self._valid_config()
        config.warmup = -1
        errors = validate_config(config)
        self.assertTrue(any(e.field == "warmup" for e in errors))

    def test_validate_negative_timeout(self) -> None:
        """Zero timeout should produce an error."""
        config = self._valid_config()
        config.timeout = 0
        errors = validate_config(config)
        self.assertTrue(any(e.field == "timeout" for e in errors))

    def test_validate_valid_config(self) -> None:
        """Well-formed config should produce no errors."""
        config = self._valid_config()
        errors = validate_config(config)
        self.assertEqual(errors, [])

    def test_validate_registry_dir_missing(self) -> None:
        """Nonexistent registry dir should error."""
        config = self._valid_config()
        config.registry_dir = Path("/nonexistent/registry")
        errors = validate_config(config)
        self.assertTrue(any(e.field == "registry_dir" for e in errors))

    def test_validate_returns_warnings(self) -> None:
        """interleave + alternate should produce a warning."""
        config = self._valid_config()
        config.conditions["jit"] = ConditionDef(name="jit", target_python=sys.executable)
        config.interleave = True
        config.alternate = True
        errors = validate_config(config)
        warnings = [e for e in errors if e.severity == "warning"]
        self.assertTrue(len(warnings) > 0)


# ---------------------------------------------------------------------------
# Profile loading tests
# ---------------------------------------------------------------------------


class TestLoadProfile(unittest.TestCase):
    """Tests for load_profile()."""

    def test_load_profile_valid(self) -> None:
        """Valid YAML profile should load correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "name: test bench\n"
                "iterations: 7\n"
                "conditions:\n"
                "  baseline:\n"
                "    description: No JIT\n"
                "  jit:\n"
                "    description: With JIT\n"
                "    env:\n"
                "      PYTHON_JIT: '1'\n"
            )
            path = Path(f.name)
        try:
            data = load_profile(path)
            self.assertEqual(data["name"], "test bench")
            self.assertEqual(data["iterations"], 7)
            self.assertIn("baseline", data["conditions"])
            self.assertIn("jit", data["conditions"])
        finally:
            path.unlink()

    def test_load_profile_missing_file(self) -> None:
        """Nonexistent file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_profile(Path("/nonexistent/profile.yaml"))

    def test_load_profile_invalid_yaml(self) -> None:
        """Invalid YAML should raise an error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: [unterminated\n")
            path = Path(f.name)
        try:
            with self.assertRaises(Exception):
                load_profile(path)
        finally:
            path.unlink()

    def test_load_profile_not_a_mapping(self) -> None:
        """YAML list instead of mapping should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError, msg="must be a YAML mapping"):
                load_profile(path)
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# config_from_profile tests
# ---------------------------------------------------------------------------


class TestConfigFromProfile(unittest.TestCase):
    """Tests for config_from_profile()."""

    def test_config_from_profile_basic(self) -> None:
        """Profile with name, 2 conditions, iterations=7."""
        data = {
            "name": "overhead test",
            "iterations": 7,
            "conditions": {
                "baseline": {"description": "No JIT"},
                "jit": {"description": "With JIT", "env": {"PYTHON_JIT": "1"}},
            },
        }
        config = config_from_profile(data)
        self.assertEqual(config.name, "overhead test")
        self.assertEqual(config.iterations, 7)
        self.assertEqual(len(config.conditions), 2)
        self.assertIn("baseline", config.conditions)
        self.assertIn("jit", config.conditions)

    def test_config_from_profile_condition_fields(self) -> None:
        """Condition fields should be populated from profile."""
        data = {
            "conditions": {
                "cov": {
                    "target_python": "/opt/python",
                    "env": {"COV": "1"},
                    "extra_deps": ["coverage"],
                    "test_command_prefix": "coverage run -m",
                },
            },
        }
        config = config_from_profile(data)
        cond = config.conditions["cov"]
        self.assertEqual(cond.target_python, "/opt/python")
        self.assertEqual(cond.env, {"COV": "1"})
        self.assertEqual(cond.extra_deps, ["coverage"])
        self.assertEqual(cond.test_command_prefix, "coverage run -m")

    def test_config_from_profile_cli_overrides_iterations(self) -> None:
        """CLI iterations should override profile iterations."""
        data = {"iterations": 5, "conditions": {"a": None}}
        config = config_from_profile(data, cli_overrides={"iterations": 10})
        self.assertEqual(config.iterations, 10)

    def test_config_from_profile_cli_overrides_target_python(self) -> None:
        """CLI target_python should override profile's."""
        data = {"target_python": "/old/python", "conditions": {"a": None}}
        config = config_from_profile(data, cli_overrides={"target_python": "/new/python"})
        self.assertEqual(config.default_target_python, "/new/python")

    def test_config_from_profile_empty_condition(self) -> None:
        """Condition value of None should be treated as empty dict."""
        data = {"conditions": {"empty": None}}
        config = config_from_profile(data)
        self.assertIn("empty", config.conditions)
        self.assertEqual(config.conditions["empty"].name, "empty")

    def test_config_from_profile_invalid_conditions_type(self) -> None:
        """conditions as a list should raise ValueError."""
        data = {"conditions": ["a", "b"]}
        with self.assertRaises(ValueError):
            config_from_profile(data)

    def test_config_from_profile_cli_overrides_paths(self) -> None:
        """CLI path overrides should be applied."""
        data = {"conditions": {"a": None}}
        config = config_from_profile(
            data,
            cli_overrides={
                "registry_dir": "/tmp/reg",
                "repos_dir": "/tmp/repos",
                "venvs_dir": "/tmp/venvs",
                "results_dir": "/tmp/results",
            },
        )
        self.assertEqual(config.registry_dir, Path("/tmp/reg"))
        self.assertEqual(config.repos_dir, Path("/tmp/repos"))
        self.assertEqual(config.venvs_dir, Path("/tmp/venvs"))
        self.assertEqual(config.results_dir, Path("/tmp/results"))

    def test_config_from_profile_warmup_zero(self) -> None:
        """CLI warmup=0 should override profile's warmup (not be falsy-skipped)."""
        data = {"warmup": 3, "conditions": {"a": None}}
        config = config_from_profile(data, cli_overrides={"warmup": 0})
        self.assertEqual(config.warmup, 0)


# ---------------------------------------------------------------------------
# Inline condition parsing tests
# ---------------------------------------------------------------------------


class TestParseInlineCondition(unittest.TestCase):
    """Tests for parse_inline_condition()."""

    def test_parse_inline_defaults(self) -> None:
        """'baseline:' should produce defaults."""
        cond = parse_inline_condition("baseline:")
        self.assertEqual(cond.name, "baseline")
        self.assertEqual(cond.target_python, "")
        self.assertEqual(cond.env, {})
        self.assertEqual(cond.extra_deps, [])

    def test_parse_inline_target_python(self) -> None:
        """target_python key should be parsed."""
        cond = parse_inline_condition("jit:target_python=/opt/python")
        self.assertEqual(cond.target_python, "/opt/python")

    def test_parse_inline_extra_deps(self) -> None:
        """extra_deps should split on '+'."""
        cond = parse_inline_condition("cov:extra_deps=coverage+pytest-cov")
        self.assertEqual(cond.extra_deps, ["coverage", "pytest-cov"])

    def test_parse_inline_env(self) -> None:
        """env.KEY=VALUE should populate env dict."""
        cond = parse_inline_condition("jit:env.PYTHON_JIT=1,env.FOO=bar")
        self.assertEqual(cond.env, {"PYTHON_JIT": "1", "FOO": "bar"})

    def test_parse_inline_test_prefix(self) -> None:
        """test_prefix should set test_command_prefix."""
        cond = parse_inline_condition("cov:test_prefix=coverage run -m")
        self.assertEqual(cond.test_command_prefix, "coverage run -m")

    def test_parse_inline_multiple_keys(self) -> None:
        """Multiple keys should all be parsed."""
        cond = parse_inline_condition("full:target_python=/opt/py,extra_deps=cov,env.JIT=1")
        self.assertEqual(cond.name, "full")
        self.assertEqual(cond.target_python, "/opt/py")
        self.assertEqual(cond.extra_deps, ["cov"])
        self.assertEqual(cond.env, {"JIT": "1"})

    def test_parse_inline_no_colon(self) -> None:
        """Missing colon should raise ValueError."""
        with self.assertRaises(ValueError):
            parse_inline_condition("baseline")

    def test_parse_inline_empty_name(self) -> None:
        """Empty name should raise ValueError."""
        with self.assertRaises(ValueError):
            parse_inline_condition(":key=val")

    def test_parse_inline_unknown_key(self) -> None:
        """Unknown key should raise ValueError."""
        with self.assertRaises(ValueError):
            parse_inline_condition("x:badkey=val")

    def test_parse_inline_no_equals(self) -> None:
        """Missing '=' in pair should raise ValueError."""
        with self.assertRaises(ValueError):
            parse_inline_condition("x:badformat")

    def test_parse_inline_test_suffix(self) -> None:
        """test_suffix should set test_command_suffix."""
        cond = parse_inline_condition("v:test_suffix=-v --tb=short")
        self.assertEqual(cond.test_command_suffix, "-v --tb=short")

    def test_parse_inline_description(self) -> None:
        """description key should be parsed."""
        cond = parse_inline_condition("base:description=No JIT enabled")
        self.assertEqual(cond.description, "No JIT enabled")


# ---------------------------------------------------------------------------
# _split_condition_pairs tests
# ---------------------------------------------------------------------------


class TestSplitConditionPairs(unittest.TestCase):
    """Tests for _split_condition_pairs() helper."""

    def test_split_simple(self) -> None:
        """Simple comma-separated pairs."""
        self.assertEqual(_split_condition_pairs("a=1,b=2"), ["a=1", "b=2"])

    def test_split_value_with_space(self) -> None:
        """Values containing spaces should not be split."""
        result = _split_condition_pairs("cmd=coverage run -m,b=2")
        self.assertEqual(result, ["cmd=coverage run -m", "b=2"])

    def test_split_empty(self) -> None:
        """Empty string should produce empty list."""
        self.assertEqual(_split_condition_pairs(""), [])

    def test_split_trailing_comma(self) -> None:
        """Trailing comma should be ignored."""
        self.assertEqual(_split_condition_pairs("a=1,"), ["a=1"])

    def test_split_single_pair(self) -> None:
        """Single pair without comma."""
        self.assertEqual(_split_condition_pairs("key=val"), ["key=val"])


# ---------------------------------------------------------------------------
# Test command resolution tests
# ---------------------------------------------------------------------------


class TestResolveTestCommand(unittest.TestCase):
    """Tests for resolve_test_command()."""

    def test_resolve_no_overrides(self) -> None:
        """Registry command returned as-is when no overrides."""
        cond = ConditionDef(name="base")
        result = resolve_test_command("python -m pytest tests/", cond)
        self.assertEqual(result, "python -m pytest tests/")

    def test_resolve_override(self) -> None:
        """test_command_override wins over everything."""
        cond = ConditionDef(name="base", test_command_override="custom")
        result = resolve_test_command("python -m pytest", cond)
        self.assertEqual(result, "custom")

    def test_resolve_prefix(self) -> None:
        """Prefix replaces 'python -m' in the command."""
        cond = ConditionDef(name="cov", test_command_prefix="coverage run -m")
        result = resolve_test_command("python -m pytest tests/", cond)
        self.assertEqual(result, "coverage run -m pytest tests/")

    def test_resolve_prefix_no_python_m(self) -> None:
        """Prefix prepends when base doesn't start with 'python -m'."""
        cond = ConditionDef(name="cov", test_command_prefix="coverage run -m")
        result = resolve_test_command("pytest tests/", cond)
        self.assertEqual(result, "coverage run -m pytest tests/")

    def test_resolve_suffix(self) -> None:
        """Condition suffix appended to base command."""
        cond = ConditionDef(name="v", test_command_suffix="-v")
        result = resolve_test_command("python -m pytest", cond)
        self.assertEqual(result, "python -m pytest -v")

    def test_resolve_default_suffix(self) -> None:
        """Default suffix appended when condition has no suffix."""
        cond = ConditionDef(name="base")
        result = resolve_test_command("python -m pytest", cond, default_suffix="-v")
        self.assertEqual(result, "python -m pytest -v")

    def test_resolve_condition_suffix_beats_default(self) -> None:
        """Condition suffix takes precedence over default suffix."""
        cond = ConditionDef(name="v", test_command_suffix="--tb=long")
        result = resolve_test_command("python -m pytest", cond, default_suffix="--tb=short")
        self.assertEqual(result, "python -m pytest --tb=long")

    def test_resolve_override_beats_all(self) -> None:
        """Override wins even if prefix and suffix are also set."""
        cond = ConditionDef(
            name="all",
            test_command_override="override cmd",
            test_command_prefix="prefix",
            test_command_suffix="suffix",
        )
        result = resolve_test_command("python -m pytest", cond, default_suffix="-v")
        self.assertEqual(result, "override cmd")

    def test_resolve_no_registry_command(self) -> None:
        """None registry command falls back to 'python -m pytest'."""
        cond = ConditionDef(name="base")
        result = resolve_test_command(None, cond)
        self.assertEqual(result, "python -m pytest")

    def test_resolve_prefix_with_python_space(self) -> None:
        """Prefix replaces 'python ' when base starts with it."""
        cond = ConditionDef(name="cov", test_command_prefix="coverage run")
        result = resolve_test_command("python test_suite.py", cond)
        self.assertEqual(result, "coverage run test_suite.py")


# ---------------------------------------------------------------------------
# resolve_target_python tests
# ---------------------------------------------------------------------------


class TestResolveTargetPython(unittest.TestCase):
    """Tests for resolve_target_python()."""

    def test_resolve_condition_python(self) -> None:
        """Condition's python takes precedence."""
        cond = ConditionDef(name="a", target_python="/opt/python")
        result = resolve_target_python(cond, "/default/python")
        self.assertEqual(result, Path("/opt/python"))

    def test_resolve_default_python(self) -> None:
        """Falls back to default when condition has no python."""
        cond = ConditionDef(name="a")
        result = resolve_target_python(cond, "/default/python")
        self.assertEqual(result, Path("/default/python"))

    def test_resolve_no_python(self) -> None:
        """Neither set raises ValueError."""
        cond = ConditionDef(name="a")
        with self.assertRaises(ValueError):
            resolve_target_python(cond, "")


# ---------------------------------------------------------------------------
# resolve_env tests
# ---------------------------------------------------------------------------


class TestResolveEnv(unittest.TestCase):
    """Tests for resolve_env()."""

    def test_resolve_env_merge(self) -> None:
        """Default and condition env should be merged."""
        cond = ConditionDef(name="a", env={"B": "2"})
        result = resolve_env(cond, {"A": "1"})
        self.assertEqual(result, {"A": "1", "B": "2"})

    def test_resolve_env_override(self) -> None:
        """Condition env overrides default for same key."""
        cond = ConditionDef(name="a", env={"A": "2"})
        result = resolve_env(cond, {"A": "1"})
        self.assertEqual(result, {"A": "2"})

    def test_resolve_env_empty(self) -> None:
        """Both empty should produce empty dict."""
        cond = ConditionDef(name="a")
        result = resolve_env(cond, {})
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# resolve_extra_deps tests
# ---------------------------------------------------------------------------


class TestResolveExtraDeps(unittest.TestCase):
    """Tests for resolve_extra_deps()."""

    def test_resolve_deps_merge(self) -> None:
        """Default and condition deps should be merged."""
        cond = ConditionDef(name="a", extra_deps=["b"])
        result = resolve_extra_deps(cond, ["a"])
        self.assertEqual(result, ["a", "b"])

    def test_resolve_deps_dedup(self) -> None:
        """Duplicates should be removed, preserving order."""
        cond = ConditionDef(name="a", extra_deps=["b", "c"])
        result = resolve_extra_deps(cond, ["a", "b"])
        self.assertEqual(result, ["a", "b", "c"])

    def test_resolve_deps_empty(self) -> None:
        """Both empty should produce empty list."""
        cond = ConditionDef(name="a")
        result = resolve_extra_deps(cond, [])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
