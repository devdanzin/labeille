"""Tests for labeille.bench.constraints — resource limits for benchmarking."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from labeille.bench.constraints import (
    ResourceConstraints,
    apply_constraints_to_command,
    build_taskset_prefix,
    build_ulimit_prefix,
    check_constraints_available,
    detect_oom_from_result,
)
from labeille.bench.results import BenchIteration, ConditionDef


# ---------------------------------------------------------------------------
# TestResourceConstraints
# ---------------------------------------------------------------------------


class TestResourceConstraints(unittest.TestCase):
    """Tests for ResourceConstraints dataclass."""

    def test_has_any_with_memory(self) -> None:
        c = ResourceConstraints(memory_limit_mb=1024)
        self.assertTrue(c.has_any)

    def test_has_any_empty(self) -> None:
        c = ResourceConstraints()
        self.assertFalse(c.has_any)

    def test_has_any_with_affinity(self) -> None:
        c = ResourceConstraints(cpu_affinity=[0])
        self.assertTrue(c.has_any)

    def test_to_dict_sparse(self) -> None:
        c = ResourceConstraints(memory_limit_mb=1024)
        d = c.to_dict()
        self.assertEqual(d, {"memory_limit_mb": 1024})
        self.assertNotIn("cpu_time_limit_s", d)
        self.assertNotIn("file_size_limit_mb", d)
        self.assertNotIn("cpu_affinity", d)

    def test_to_dict_roundtrip(self) -> None:
        c = ResourceConstraints(
            memory_limit_mb=1024,
            cpu_time_limit_s=300,
            file_size_limit_mb=100,
            cpu_affinity=[0, 1, 2],
        )
        d = c.to_dict()
        restored = ResourceConstraints.from_dict(d)
        self.assertEqual(restored.memory_limit_mb, 1024)
        self.assertEqual(restored.cpu_time_limit_s, 300)
        self.assertEqual(restored.file_size_limit_mb, 100)
        self.assertEqual(restored.cpu_affinity, [0, 1, 2])

    def test_from_dict_ignores_unknown(self) -> None:
        d = {"memory_limit_mb": 1024, "unknown_field": "ignored"}
        c = ResourceConstraints.from_dict(d)
        self.assertEqual(c.memory_limit_mb, 1024)
        self.assertFalse(hasattr(c, "unknown_field"))

    def test_has_cpu_affinity(self) -> None:
        self.assertTrue(ResourceConstraints(cpu_affinity=[0]).has_cpu_affinity)
        self.assertFalse(ResourceConstraints(cpu_affinity=[]).has_cpu_affinity)
        self.assertFalse(ResourceConstraints().has_cpu_affinity)


# ---------------------------------------------------------------------------
# TestBuildUlimitPrefix
# ---------------------------------------------------------------------------


class TestBuildUlimitPrefix(unittest.TestCase):
    """Tests for build_ulimit_prefix()."""

    def test_memory_only(self) -> None:
        c = ResourceConstraints(memory_limit_mb=1024)
        result = build_ulimit_prefix(c)
        self.assertIn("ulimit -v 1048576", result)  # 1024 * 1024

    def test_cpu_time_only(self) -> None:
        c = ResourceConstraints(cpu_time_limit_s=300)
        result = build_ulimit_prefix(c)
        self.assertIn("ulimit -t 300", result)

    def test_file_size_only(self) -> None:
        c = ResourceConstraints(file_size_limit_mb=100)
        result = build_ulimit_prefix(c)
        self.assertIn("ulimit -f 204800", result)  # 100 * 2048

    def test_multiple_constraints(self) -> None:
        c = ResourceConstraints(
            memory_limit_mb=1024,
            cpu_time_limit_s=300,
            file_size_limit_mb=100,
        )
        result = build_ulimit_prefix(c)
        self.assertIn("ulimit -v 1048576", result)
        self.assertIn("ulimit -t 300", result)
        self.assertIn("ulimit -f 204800", result)

    def test_no_constraints(self) -> None:
        c = ResourceConstraints()
        result = build_ulimit_prefix(c)
        self.assertEqual(result, "")

    def test_ends_with_exec(self) -> None:
        c = ResourceConstraints(memory_limit_mb=1024)
        result = build_ulimit_prefix(c)
        self.assertTrue(result.endswith("; exec "))


# ---------------------------------------------------------------------------
# TestBuildTasksetPrefix
# ---------------------------------------------------------------------------


class TestBuildTasksetPrefix(unittest.TestCase):
    """Tests for build_taskset_prefix()."""

    @patch("labeille.bench.constraints.shutil.which", return_value="/usr/bin/taskset")
    @patch("labeille.bench.constraints.sys.platform", "linux")
    def test_linux_with_taskset(self, mock_which: object) -> None:
        c = ResourceConstraints(cpu_affinity=[0, 1])
        result = build_taskset_prefix(c)
        self.assertEqual(result, "taskset -c 0,1 ")

    @patch("labeille.bench.constraints.sys.platform", "darwin")
    def test_macos_ignored(self) -> None:
        c = ResourceConstraints(cpu_affinity=[0, 1])
        result = build_taskset_prefix(c)
        self.assertEqual(result, "")

    @patch("labeille.bench.constraints.shutil.which", return_value=None)
    @patch("labeille.bench.constraints.sys.platform", "linux")
    def test_no_taskset_binary(self, mock_which: object) -> None:
        c = ResourceConstraints(cpu_affinity=[0, 1])
        result = build_taskset_prefix(c)
        self.assertEqual(result, "")

    def test_no_affinity(self) -> None:
        c = ResourceConstraints()
        result = build_taskset_prefix(c)
        self.assertEqual(result, "")

    @patch("labeille.bench.constraints.shutil.which", return_value="/usr/bin/taskset")
    @patch("labeille.bench.constraints.sys.platform", "linux")
    def test_single_core(self, mock_which: object) -> None:
        c = ResourceConstraints(cpu_affinity=[2])
        result = build_taskset_prefix(c)
        self.assertEqual(result, "taskset -c 2 ")


# ---------------------------------------------------------------------------
# TestApplyConstraintsToCommand
# ---------------------------------------------------------------------------


class TestApplyConstraintsToCommand(unittest.TestCase):
    """Tests for apply_constraints_to_command()."""

    def test_no_constraints(self) -> None:
        result = apply_constraints_to_command("python -m pytest", None)
        self.assertEqual(result, "python -m pytest")

    def test_ulimit_wrapping(self) -> None:
        c = ResourceConstraints(memory_limit_mb=1024)
        result = apply_constraints_to_command("python -m pytest", c)
        self.assertIn("bash -c '", result)
        self.assertIn("ulimit -v 1048576", result)
        self.assertIn("python -m pytest", result)

    @patch("labeille.bench.constraints.shutil.which", return_value="/usr/bin/taskset")
    @patch("labeille.bench.constraints.sys.platform", "linux")
    def test_taskset_prefixed(self, mock_which: object) -> None:
        c = ResourceConstraints(cpu_affinity=[0, 1])
        result = apply_constraints_to_command("python -m pytest", c)
        self.assertTrue(result.startswith("taskset -c 0,1 "))

    @patch("labeille.bench.constraints.shutil.which", return_value="/usr/bin/taskset")
    @patch("labeille.bench.constraints.sys.platform", "linux")
    def test_both_ulimit_and_taskset(self, mock_which: object) -> None:
        c = ResourceConstraints(memory_limit_mb=1024, cpu_affinity=[0, 1])
        result = apply_constraints_to_command("python -m pytest", c)
        self.assertTrue(result.startswith("taskset -c 0,1 "))
        self.assertIn("bash -c '", result)
        self.assertIn("ulimit -v", result)

    def test_empty_constraints(self) -> None:
        c = ResourceConstraints()
        result = apply_constraints_to_command("python -m pytest", c)
        self.assertEqual(result, "python -m pytest")


# ---------------------------------------------------------------------------
# TestCheckConstraintsAvailable
# ---------------------------------------------------------------------------


class TestCheckConstraintsAvailable(unittest.TestCase):
    """Tests for check_constraints_available()."""

    @patch("labeille.bench.constraints.shutil.which", return_value="/usr/bin/taskset")
    @patch("labeille.bench.constraints.sys.platform", "linux")
    def test_all_available_linux(self, mock_which: object) -> None:
        c = ResourceConstraints(memory_limit_mb=1024, cpu_affinity=[0])
        warnings = check_constraints_available(c)
        self.assertEqual(warnings, [])

    @patch("labeille.bench.constraints.sys.platform", "darwin")
    def test_affinity_on_macos(self) -> None:
        c = ResourceConstraints(cpu_affinity=[0, 1])
        warnings = check_constraints_available(c)
        self.assertTrue(any("only available on Linux" in w for w in warnings))

    @patch("labeille.bench.constraints.shutil.which", return_value=None)
    @patch("labeille.bench.constraints.sys.platform", "linux")
    def test_taskset_missing(self, mock_which: object) -> None:
        c = ResourceConstraints(cpu_affinity=[0])
        warnings = check_constraints_available(c)
        self.assertTrue(any("taskset not found" in w for w in warnings))

    @patch("labeille.bench.constraints.os.cpu_count", return_value=4)
    @patch("labeille.bench.constraints.shutil.which", return_value="/usr/bin/taskset")
    @patch("labeille.bench.constraints.sys.platform", "linux")
    def test_invalid_core_index(self, mock_which: object, mock_cpu: object) -> None:
        c = ResourceConstraints(cpu_affinity=[0, 8])
        warnings = check_constraints_available(c)
        self.assertTrue(any("8" in w for w in warnings))

    def test_no_constraints(self) -> None:
        c = ResourceConstraints()
        warnings = check_constraints_available(c)
        self.assertEqual(warnings, [])


# ---------------------------------------------------------------------------
# TestDetectOomFromResult
# ---------------------------------------------------------------------------


class TestDetectOomFromResult(unittest.TestCase):
    """Tests for detect_oom_from_result()."""

    def test_sigkill_with_memory_limit(self) -> None:
        c = ResourceConstraints(memory_limit_mb=1024)
        self.assertTrue(detect_oom_from_result(-9, "", c))

    def test_sigsegv_with_memory_limit(self) -> None:
        c = ResourceConstraints(memory_limit_mb=1024)
        self.assertTrue(detect_oom_from_result(-11, "", c))

    def test_memory_error_in_stderr(self) -> None:
        c = ResourceConstraints(memory_limit_mb=1024)
        self.assertTrue(detect_oom_from_result(1, "MemoryError: out of memory", c))

    def test_cannot_allocate_in_stderr(self) -> None:
        c = ResourceConstraints(memory_limit_mb=1024)
        self.assertTrue(detect_oom_from_result(1, "Cannot allocate memory", c))

    def test_sigkill_without_memory_limit(self) -> None:
        c = ResourceConstraints(cpu_time_limit_s=300)
        self.assertFalse(detect_oom_from_result(-9, "", c))

    def test_normal_exit(self) -> None:
        c = ResourceConstraints(memory_limit_mb=1024)
        self.assertFalse(detect_oom_from_result(0, "", c))

    def test_no_constraints(self) -> None:
        self.assertFalse(detect_oom_from_result(-9, "", None))


# ---------------------------------------------------------------------------
# TestConditionDefConstraints
# ---------------------------------------------------------------------------


class TestConditionDefConstraints(unittest.TestCase):
    """Tests for ConditionDef constraints field."""

    def test_condition_with_constraints_roundtrip(self) -> None:
        cond = ConditionDef(
            name="constrained",
            constraints=ResourceConstraints(memory_limit_mb=2048, cpu_affinity=[0, 1]),
        )
        d = cond.to_dict()
        self.assertIn("constraints", d)
        self.assertEqual(d["constraints"]["memory_limit_mb"], 2048)

        restored = ConditionDef.from_dict(d)
        assert restored.constraints is not None
        self.assertEqual(restored.constraints.memory_limit_mb, 2048)
        self.assertEqual(restored.constraints.cpu_affinity, [0, 1])

    def test_condition_without_constraints(self) -> None:
        cond = ConditionDef(name="baseline")
        d = cond.to_dict()
        self.assertNotIn("constraints", d)

    def test_condition_from_old_format(self) -> None:
        d = {"name": "old_baseline", "target_python": "/usr/bin/python3"}
        cond = ConditionDef.from_dict(d)
        self.assertIsNone(cond.constraints)


# ---------------------------------------------------------------------------
# TestBenchIterationConstraintFields
# ---------------------------------------------------------------------------


class TestBenchIterationConstraintFields(unittest.TestCase):
    """Tests for BenchIteration constraint-related fields."""

    def test_defaults(self) -> None:
        it = BenchIteration(
            index=1,
            warmup=False,
            wall_time_s=5.0,
            user_time_s=4.0,
            sys_time_s=0.5,
            peak_rss_mb=256.0,
            exit_code=0,
            status="ok",
        )
        self.assertFalse(it.constraints_applied)
        self.assertFalse(it.oom_detected)

    def test_constraints_applied_serialization(self) -> None:
        it = BenchIteration(
            index=1,
            warmup=False,
            wall_time_s=5.0,
            user_time_s=4.0,
            sys_time_s=0.5,
            peak_rss_mb=256.0,
            exit_code=0,
            status="ok",
            constraints_applied=True,
            oom_detected=True,
        )
        d = it.to_dict()
        self.assertTrue(d["constraints_applied"])
        self.assertTrue(d["oom_detected"])
        restored = BenchIteration.from_dict(d)
        self.assertTrue(restored.constraints_applied)
        self.assertTrue(restored.oom_detected)

    def test_false_fields_not_in_dict(self) -> None:
        it = BenchIteration(
            index=1,
            warmup=False,
            wall_time_s=5.0,
            user_time_s=4.0,
            sys_time_s=0.5,
            peak_rss_mb=256.0,
            exit_code=0,
            status="ok",
        )
        d = it.to_dict()
        self.assertNotIn("constraints_applied", d)
        self.assertNotIn("oom_detected", d)

    def test_backward_compat_missing_keys(self) -> None:
        d = {
            "index": 1,
            "warmup": False,
            "wall_time_s": 5.0,
            "user_time_s": 4.0,
            "sys_time_s": 0.5,
            "peak_rss_mb": 256.0,
            "exit_code": 0,
            "status": "ok",
        }
        it = BenchIteration.from_dict(d)
        self.assertFalse(it.constraints_applied)
        self.assertFalse(it.oom_detected)


if __name__ == "__main__":
    unittest.main()
