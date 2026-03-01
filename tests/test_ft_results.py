"""Tests for labeille.ft.results — free-threading result dataclasses."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ft_test_helpers import make_iteration, make_package_result

from labeille.ft.results import (
    FailureCategory,
    FTPackageResult,
    FTRunMeta,
    FTRunSummary,
    IterationOutcome,
    append_ft_result,
    categorize_package,
    load_ft_run,
    load_ft_summary,
    save_ft_run,
)


# ---------------------------------------------------------------------------
# FailureCategory tests
# ---------------------------------------------------------------------------


class TestFailureCategory(unittest.TestCase):
    def test_category_values(self) -> None:
        values = [c.value for c in FailureCategory]
        self.assertEqual(len(values), len(set(values)))

    def test_category_is_usable(self) -> None:
        usable = {
            FailureCategory.COMPATIBLE,
            FailureCategory.COMPATIBLE_GIL_FALLBACK,
            FailureCategory.INTERMITTENT,
            FailureCategory.TSAN_WARNINGS,
        }
        for cat in FailureCategory:
            if cat in usable:
                self.assertTrue(cat.is_usable, f"{cat} should be usable")
            else:
                self.assertFalse(cat.is_usable, f"{cat} should not be usable")

    def test_category_severity_order(self) -> None:
        self.assertLess(
            FailureCategory.COMPATIBLE.severity,
            FailureCategory.INTERMITTENT.severity,
        )
        self.assertLess(
            FailureCategory.INTERMITTENT.severity,
            FailureCategory.CRASH.severity,
        )
        self.assertLess(
            FailureCategory.CRASH.severity,
            FailureCategory.DEADLOCK.severity,
        )
        self.assertLess(
            FailureCategory.DEADLOCK.severity,
            FailureCategory.INSTALL_FAILURE.severity,
        )

    def test_category_symbols(self) -> None:
        for cat in FailureCategory:
            self.assertTrue(len(cat.symbol) > 0, f"{cat} has empty symbol")

    def test_category_from_string(self) -> None:
        self.assertEqual(FailureCategory("compatible"), FailureCategory.COMPATIBLE)

    def test_category_invalid_string(self) -> None:
        with self.assertRaises(ValueError):
            FailureCategory("not_a_category")


# ---------------------------------------------------------------------------
# IterationOutcome tests
# ---------------------------------------------------------------------------


class TestIterationOutcome(unittest.TestCase):
    def test_iteration_pass(self) -> None:
        it = IterationOutcome(index=1, status="pass", exit_code=0)
        self.assertTrue(it.is_pass)
        self.assertFalse(it.is_crash)
        self.assertFalse(it.is_deadlock)
        self.assertFalse(it.is_timeout)

    def test_iteration_crash(self) -> None:
        it = IterationOutcome(index=1, status="crash", exit_code=-11)
        self.assertTrue(it.is_crash)
        self.assertFalse(it.is_pass)

    def test_iteration_deadlock(self) -> None:
        it = IterationOutcome(index=1, status="deadlock")
        self.assertTrue(it.is_deadlock)
        self.assertTrue(it.is_timeout)

    def test_iteration_timeout(self) -> None:
        it = IterationOutcome(index=1, status="timeout")
        self.assertTrue(it.is_timeout)
        self.assertFalse(it.is_deadlock)

    def test_iteration_tsan(self) -> None:
        it = IterationOutcome(index=1, status="pass", tsan_warnings=["data race"])
        self.assertTrue(it.has_tsan_warnings)

    def test_iteration_serialization_roundtrip(self) -> None:
        it = IterationOutcome(
            index=3,
            status="crash",
            exit_code=-11,
            duration_s=5.5,
            crash_signal="SIGSEGV",
            crash_signature="_PyEval_EvalFrame+0x1234",
            tsan_warnings=["data race in foo"],
            output_stalled=True,
            last_output_line="FAILED test_bar",
            stderr_tail="segfault at 0x0",
            test_results={"test_bar": "FAILED"},
            tests_passed=10,
            tests_failed=1,
            tests_errored=0,
            tests_skipped=2,
        )
        d = it.to_dict()
        restored = IterationOutcome.from_dict(d)
        self.assertEqual(restored.index, 3)
        self.assertEqual(restored.status, "crash")
        self.assertEqual(restored.crash_signal, "SIGSEGV")
        self.assertEqual(restored.tsan_warnings, ["data race in foo"])
        self.assertEqual(restored.test_results, {"test_bar": "FAILED"})
        self.assertEqual(restored.tests_passed, 10)

    def test_iteration_serialization_omits_empty_test_results(self) -> None:
        it = IterationOutcome(index=1, status="pass")
        d = it.to_dict()
        self.assertNotIn("test_results", d)

    def test_iteration_from_dict_missing_fields(self) -> None:
        d = {"index": 1, "status": "pass"}
        it = IterationOutcome.from_dict(d)
        self.assertEqual(it.index, 1)
        self.assertIsNone(it.exit_code)
        self.assertEqual(it.duration_s, 0.0)
        self.assertEqual(it.tsan_warnings, [])


# ---------------------------------------------------------------------------
# FTPackageResult tests
# ---------------------------------------------------------------------------


class TestFTPackageResult(unittest.TestCase):
    def test_compute_aggregates_all_pass(self) -> None:
        result = FTPackageResult(
            package="pkg",
            iterations=[make_iteration(i, "pass") for i in range(5)],
        )
        result.compute_aggregates()
        self.assertEqual(result.pass_count, 5)
        self.assertEqual(result.pass_rate, 1.0)
        self.assertEqual(result.crash_count, 0)
        self.assertEqual(result.iterations_completed, 5)

    def test_compute_aggregates_mixed(self) -> None:
        result = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass"),
                make_iteration(1, "pass"),
                make_iteration(2, "pass"),
                make_iteration(3, "crash", exit_code=-11),
                make_iteration(4, "fail", exit_code=1),
            ],
        )
        result.compute_aggregates()
        self.assertAlmostEqual(result.pass_rate, 0.6)
        self.assertEqual(result.crash_count, 1)
        self.assertEqual(result.fail_count, 1)

    def test_compute_aggregates_empty(self) -> None:
        result = FTPackageResult(package="pkg")
        result.compute_aggregates()
        self.assertEqual(result.iterations_completed, 0)
        self.assertEqual(result.pass_rate, 0.0)

    def test_compute_aggregates_mean_duration(self) -> None:
        result = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass", duration_s=10.0),
                make_iteration(1, "pass", duration_s=12.0),
                make_iteration(2, "pass", duration_s=11.0),
            ],
        )
        result.compute_aggregates()
        self.assertAlmostEqual(result.mean_duration_s, 11.0)

    def test_compute_aggregates_failure_signatures(self) -> None:
        result = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "crash", crash_signature="sig_a"),
                make_iteration(1, "crash", crash_signature="sig_a"),
                make_iteration(2, "crash", crash_signature="sig_b"),
            ],
        )
        result.compute_aggregates()
        self.assertEqual(result.failure_signatures, ["sig_a", "sig_b"])

    def test_compute_aggregates_tsan_types(self) -> None:
        result = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass", tsan_warnings=["data race", "data race"]),
                make_iteration(1, "pass", tsan_warnings=["thread leak"]),
            ],
        )
        result.compute_aggregates()
        self.assertEqual(result.tsan_warning_types, ["data race", "thread leak"])
        self.assertEqual(result.tsan_warning_iterations, 2)

    def test_compute_aggregates_flaky_tests(self) -> None:
        result = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(0, "pass", test_results={"test_foo": "PASSED"}),
                make_iteration(1, "fail", test_results={"test_foo": "FAILED"}),
                make_iteration(2, "pass", test_results={"test_foo": "PASSED"}),
                make_iteration(3, "fail", test_results={"test_foo": "FAILED"}),
                make_iteration(4, "pass", test_results={"test_foo": "PASSED"}),
            ],
        )
        result.compute_aggregates()
        self.assertIn("test_foo", result.flaky_tests)
        self.assertEqual(result.flaky_tests["test_foo"], 2)

    def test_compute_aggregates_no_flaky_if_always_fails(self) -> None:
        result = FTPackageResult(
            package="pkg",
            iterations=[
                make_iteration(i, "fail", test_results={"test_bar": "FAILED"}) for i in range(5)
            ],
        )
        result.compute_aggregates()
        self.assertNotIn("test_bar", result.flaky_tests)

    def test_compute_aggregates_gil_comparison(self) -> None:
        gil_iters = [make_iteration(i, "pass" if i < 4 else "fail") for i in range(5)]
        result = FTPackageResult(
            package="pkg",
            iterations=[make_iteration(i, "pass") for i in range(5)],
            gil_enabled_iterations=gil_iters,
        )
        result.compute_aggregates()
        self.assertIsNotNone(result.gil_enabled_pass_rate)
        self.assertAlmostEqual(result.gil_enabled_pass_rate, 0.8)  # type: ignore[arg-type]

    def test_serialization_roundtrip(self) -> None:
        result = FTPackageResult(
            package="mypkg",
            iterations=[
                make_iteration(0, "pass", duration_s=10.0),
                make_iteration(1, "crash", exit_code=-11, crash_signal="SIGSEGV"),
            ],
            extension_compat={"is_pure_python": False, "gil_fallback_active": True},
            install_duration_s=5.5,
            commit="abc123",
            gil_enabled_iterations=[make_iteration(0, "pass")],
        )
        result.compute_aggregates()
        result.categorize()

        d = result.to_dict()
        restored = FTPackageResult.from_dict(d)
        self.assertEqual(restored.package, "mypkg")
        self.assertEqual(restored.category, result.category)
        self.assertEqual(len(restored.iterations), 2)
        self.assertIsNotNone(restored.extension_compat)
        self.assertEqual(restored.commit, "abc123")
        self.assertIsNotNone(restored.gil_enabled_iterations)
        assert restored.gil_enabled_iterations is not None
        self.assertEqual(len(restored.gil_enabled_iterations), 1)

    def test_serialization_without_optional_fields(self) -> None:
        result = FTPackageResult(package="minimal")
        result.compute_aggregates()
        result.categorize()
        d = result.to_dict()
        self.assertNotIn("extension_compat", d)
        self.assertNotIn("gil_enabled_pass_rate", d)
        self.assertNotIn("gil_enabled_iterations", d)


# ---------------------------------------------------------------------------
# Categorization tests
# ---------------------------------------------------------------------------


class TestCategorizePackage(unittest.TestCase):
    def test_categorize_install_failure(self) -> None:
        result = FTPackageResult(package="pkg", install_ok=False)
        self.assertEqual(categorize_package(result), FailureCategory.INSTALL_FAILURE)

    def test_categorize_import_failure(self) -> None:
        result = FTPackageResult(package="pkg", import_ok=False)
        self.assertEqual(categorize_package(result), FailureCategory.IMPORT_FAILURE)

    def test_categorize_no_iterations(self) -> None:
        result = FTPackageResult(package="pkg", iterations_completed=0)
        self.assertEqual(categorize_package(result), FailureCategory.UNKNOWN)

    def test_categorize_compatible(self) -> None:
        r = make_package_result("pkg", ["pass"] * 5)
        self.assertEqual(r.category, FailureCategory.COMPATIBLE)

    def test_categorize_compatible_gil_fallback(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[make_iteration(i, "pass") for i in range(5)],
            extension_compat={"gil_fallback_active": True},
        )
        r.compute_aggregates()
        cat = r.categorize()
        self.assertEqual(cat, FailureCategory.COMPATIBLE_GIL_FALLBACK)

    def test_categorize_tsan_warnings(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[make_iteration(i, "pass", tsan_warnings=["data race"]) for i in range(5)],
        )
        r.compute_aggregates()
        cat = r.categorize()
        self.assertEqual(cat, FailureCategory.TSAN_WARNINGS)

    def test_categorize_tsan_takes_priority_over_gil_fallback(self) -> None:
        r = FTPackageResult(
            package="pkg",
            iterations=[make_iteration(i, "pass", tsan_warnings=["data race"]) for i in range(5)],
            extension_compat={"gil_fallback_active": True},
        )
        r.compute_aggregates()
        cat = r.categorize()
        self.assertEqual(cat, FailureCategory.TSAN_WARNINGS)

    def test_categorize_crash(self) -> None:
        r = make_package_result("pkg", ["pass", "pass", "crash"])
        self.assertEqual(r.category, FailureCategory.CRASH)

    def test_categorize_deadlock(self) -> None:
        r = make_package_result("pkg", ["pass", "deadlock", "pass"])
        self.assertEqual(r.category, FailureCategory.DEADLOCK)

    def test_categorize_deadlock_takes_priority_over_crash(self) -> None:
        r = make_package_result("pkg", ["deadlock", "crash", "pass"])
        self.assertEqual(r.category, FailureCategory.DEADLOCK)

    def test_categorize_incompatible(self) -> None:
        r = make_package_result("pkg", ["fail"] * 5)
        self.assertEqual(r.category, FailureCategory.INCOMPATIBLE)

    def test_categorize_intermittent(self) -> None:
        r = make_package_result("pkg", ["pass", "pass", "fail"])
        self.assertEqual(r.category, FailureCategory.INTERMITTENT)

    def test_categorize_crash_with_some_passes(self) -> None:
        r = make_package_result("pkg", ["pass"] * 7 + ["crash"] * 3)
        self.assertEqual(r.category, FailureCategory.CRASH)


# ---------------------------------------------------------------------------
# FTRunSummary tests
# ---------------------------------------------------------------------------


class TestFTRunSummary(unittest.TestCase):
    def test_summary_category_counts(self) -> None:
        results = [
            make_package_result("a", ["pass"] * 5),
            make_package_result("b", ["pass"] * 5),
            make_package_result("c", ["pass"] * 5),
            make_package_result("d", ["pass", "fail", "pass"]),
            make_package_result("e", ["pass", "fail", "pass"]),
            make_package_result("f", ["pass", "crash", "pass"]),
        ]
        summary = FTRunSummary.compute(results)
        self.assertEqual(summary.categories.get("compatible"), 3)
        self.assertEqual(summary.categories.get("intermittent"), 2)
        self.assertEqual(summary.categories.get("crash"), 1)

    def test_summary_pass_rate_distribution(self) -> None:
        results = [
            make_package_result("a", ["pass"] * 5),  # 100%
            make_package_result("b", ["pass"] * 9 + ["fail"]),  # 90%
            make_package_result("c", ["pass"] * 5 + ["fail"] * 5),  # 50%
            make_package_result("d", ["pass"] + ["fail"] * 9),  # 10%
            make_package_result("e", ["fail"] * 5),  # 0%
        ]
        summary = FTRunSummary.compute(results)
        self.assertEqual(summary.pass_rate_distribution["100%"], 1)
        self.assertEqual(summary.pass_rate_distribution["90-99%"], 1)
        self.assertEqual(summary.pass_rate_distribution["50-89%"], 1)
        self.assertEqual(summary.pass_rate_distribution["1-49%"], 1)
        self.assertEqual(summary.pass_rate_distribution["0%"], 1)

    def test_summary_pure_python_vs_extension(self) -> None:
        results = [
            make_package_result("pure1", ["pass"] * 5),
            make_package_result(
                "ext1",
                ["pass"] * 5,
                extension_compat={"is_pure_python": False},
            ),
            make_package_result(
                "ext2",
                ["fail"] * 5,
                extension_compat={"is_pure_python": False},
            ),
        ]
        summary = FTRunSummary.compute(results)
        self.assertEqual(summary.pure_python_count, 1)
        self.assertEqual(summary.extension_count, 2)
        self.assertEqual(summary.pure_python_compatible_pct, 100.0)
        self.assertEqual(summary.extension_compatible_pct, 50.0)

    def test_summary_serialization_roundtrip(self) -> None:
        results = [
            make_package_result("a", ["pass"] * 5),
            make_package_result("b", ["crash", "pass"]),
        ]
        summary = FTRunSummary.compute(results)
        d = summary.to_dict()
        restored = FTRunSummary.from_dict(d)
        self.assertEqual(restored.total_packages, 2)
        self.assertEqual(restored.categories, summary.categories)
        self.assertEqual(restored.pass_rate_distribution, summary.pass_rate_distribution)


# ---------------------------------------------------------------------------
# I/O tests
# ---------------------------------------------------------------------------


class TestIO(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            meta = FTRunMeta(run_id="test-1", timestamp="2026-03-01T00:00:00")
            results = [
                make_package_result("a", ["pass"] * 3),
                make_package_result("b", ["pass", "crash"]),
                make_package_result("c", ["fail"] * 3),
            ]
            save_ft_run(d, meta, results)

            loaded_meta, loaded_results = load_ft_run(d)
            self.assertEqual(loaded_meta.run_id, "test-1")
            self.assertEqual(loaded_meta.packages_completed, 3)
            self.assertEqual(len(loaded_results), 3)
            self.assertEqual(loaded_results[0].package, "a")
            self.assertEqual(loaded_results[1].package, "b")

    def test_append_ft_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            # Create empty JSONL.
            (d / "ft_results.jsonl").write_text("")
            (d / "ft_meta.json").write_text(
                json.dumps(FTRunMeta(run_id="x", timestamp="t").to_dict())
            )

            for i in range(3):
                r = make_package_result(f"pkg{i}", ["pass"])
                append_ft_result(d, r)

            _, loaded = load_ft_run(d)
            self.assertEqual(len(loaded), 3)

    def test_append_then_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "ft_results.jsonl").write_text("")
            (d / "ft_meta.json").write_text(
                json.dumps(FTRunMeta(run_id="x", timestamp="t").to_dict())
            )

            append_ft_result(d, make_package_result("alpha", ["pass", "pass"]))
            append_ft_result(d, make_package_result("beta", ["fail"]))

            _, loaded = load_ft_run(d)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].package, "alpha")
            self.assertEqual(loaded[1].package, "beta")

    def test_load_nonexistent_dir(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_ft_run(Path("/nonexistent/dir"))

    def test_load_empty_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "ft_meta.json").write_text(
                json.dumps(FTRunMeta(run_id="x", timestamp="t").to_dict())
            )
            (d / "ft_results.jsonl").write_text("")
            _, loaded = load_ft_run(d)
            self.assertEqual(loaded, [])

    def test_load_ft_summary_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            results = [make_package_result("a", ["pass"] * 5)]
            meta = FTRunMeta(run_id="test", timestamp="t")
            save_ft_run(d, meta, results)

            summary = load_ft_summary(d)
            self.assertEqual(summary.total_packages, 1)
            self.assertEqual(summary.categories.get("compatible"), 1)

    def test_load_ft_summary_recomputes_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            results = [
                make_package_result("a", ["pass"] * 5),
                make_package_result("b", ["crash", "pass"]),
            ]
            meta = FTRunMeta(run_id="test", timestamp="t")
            save_ft_run(d, meta, results)

            # Delete summary file.
            (d / "ft_summary.json").unlink()

            summary = load_ft_summary(d)
            self.assertEqual(summary.total_packages, 2)


# ---------------------------------------------------------------------------
# FTRunMeta tests
# ---------------------------------------------------------------------------


class TestFTRunMeta(unittest.TestCase):
    def test_meta_serialization_roundtrip(self) -> None:
        meta = FTRunMeta(
            run_id="run-42",
            timestamp="2026-03-01T12:00:00",
            packages_total=10,
            packages_completed=8,
            total_iterations=40,
            total_duration_s=1234.5,
            cli_args=["--packages", "foo", "--iterations", "5"],
        )
        d = meta.to_dict()
        restored = FTRunMeta.from_dict(d)
        self.assertEqual(restored.run_id, "run-42")
        self.assertEqual(restored.packages_total, 10)
        self.assertEqual(restored.cli_args, ["--packages", "foo", "--iterations", "5"])

    def test_meta_from_dict_missing_fields(self) -> None:
        d = {"run_id": "x", "timestamp": "t"}
        meta = FTRunMeta.from_dict(d)
        self.assertEqual(meta.run_id, "x")
        self.assertEqual(meta.packages_total, 0)
        self.assertEqual(meta.cli_args, [])


if __name__ == "__main__":
    unittest.main()
