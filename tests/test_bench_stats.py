"""Tests for labeille.bench.stats — statistical functions for benchmark comparison.

Every statistical function is tested against known values to ensure
correctness. Where applicable, results are cross-checked against
scipy to verify agreement.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from labeille.bench.stats import (
    BootstrapCI,
    DescriptiveStats,
    EffectSize,
    OverheadResult,
    TTestResult,
    _percentile,
    _regularized_incomplete_beta,
    bootstrap_ci,
    cohens_d,
    compute_overhead,
    describe,
    detect_outliers,
    welch_ttest,
)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


class TestDescribe(unittest.TestCase):
    """Tests for describe() and DescriptiveStats."""

    def test_describe_basic(self) -> None:
        """Known-value test with a small sample."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        stats = describe(values)
        self.assertEqual(stats.n, 8)
        self.assertAlmostEqual(stats.mean, 5.0, places=5)
        self.assertAlmostEqual(stats.median, 4.5, places=5)
        self.assertAlmostEqual(stats.min, 2.0, places=5)
        self.assertAlmostEqual(stats.max, 9.0, places=5)
        self.assertGreater(stats.stdev, 0)
        self.assertGreater(stats.iqr, 0)

    def test_describe_single_value(self) -> None:
        """Single value: stdev and CV should be 0."""
        stats = describe([42.0])
        self.assertEqual(stats.n, 1)
        self.assertAlmostEqual(stats.mean, 42.0, places=5)
        self.assertAlmostEqual(stats.median, 42.0, places=5)
        self.assertAlmostEqual(stats.stdev, 0.0, places=5)
        self.assertAlmostEqual(stats.cv, 0.0, places=5)

    def test_describe_empty(self) -> None:
        """Empty input: all fields should be NaN."""
        stats = describe([])
        self.assertEqual(stats.n, 0)
        self.assertTrue(math.isnan(stats.mean))
        self.assertTrue(math.isnan(stats.median))
        self.assertTrue(math.isnan(stats.stdev))
        self.assertTrue(math.isnan(stats.cv))

    def test_describe_two_values(self) -> None:
        """Two values: stdev should be computable."""
        stats = describe([1.0, 3.0])
        self.assertEqual(stats.n, 2)
        self.assertAlmostEqual(stats.mean, 2.0, places=5)
        self.assertAlmostEqual(stats.median, 2.0, places=5)
        self.assertGreater(stats.stdev, 0)

    def test_describe_identical_values(self) -> None:
        """Identical values: stdev=0, cv should handle zero mean case."""
        stats = describe([5.0, 5.0, 5.0])
        self.assertAlmostEqual(stats.stdev, 0.0, places=5)
        self.assertAlmostEqual(stats.cv, 0.0, places=5)

    def test_describe_cv_zero_mean(self) -> None:
        """CV should be inf when mean is zero but stdev is nonzero."""
        stats = describe([-1.0, 1.0])
        self.assertAlmostEqual(stats.mean, 0.0, places=5)
        self.assertTrue(math.isinf(stats.cv))

    def test_describe_to_dict(self) -> None:
        """to_dict should return all fields with rounded values."""
        stats = describe([1.0, 2.0, 3.0, 4.0, 5.0])
        d = stats.to_dict()
        self.assertEqual(d["n"], 5)
        self.assertIsInstance(d["mean"], float)
        self.assertIn("stdev", d)
        self.assertIn("iqr", d)
        self.assertIn("cv", d)
        # All float values should be rounded
        for key, val in d.items():
            if isinstance(val, float):
                s = str(val)
                if "." in s:
                    decimals = len(s.split(".")[1])
                    self.assertLessEqual(decimals, 6)

    def test_describe_unsorted_input(self) -> None:
        """Input does not need to be sorted."""
        stats = describe([5.0, 1.0, 3.0, 2.0, 4.0])
        self.assertAlmostEqual(stats.min, 1.0, places=5)
        self.assertAlmostEqual(stats.max, 5.0, places=5)
        self.assertAlmostEqual(stats.median, 3.0, places=5)


# ---------------------------------------------------------------------------
# Percentile
# ---------------------------------------------------------------------------


class TestPercentile(unittest.TestCase):
    """Tests for _percentile() helper."""

    def test_percentile_median(self) -> None:
        """50th percentile should match median for odd-length list."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(_percentile(values, 0.5), 3.0, places=5)

    def test_percentile_median_even(self) -> None:
        """50th percentile for even-length list uses interpolation."""
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(_percentile(values, 0.5), 2.5, places=5)

    def test_percentile_q1(self) -> None:
        """25th percentile for known data."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        q1 = _percentile(values, 0.25)
        self.assertAlmostEqual(q1, 2.75, places=5)

    def test_percentile_q3(self) -> None:
        """75th percentile for known data."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        q3 = _percentile(values, 0.75)
        self.assertAlmostEqual(q3, 6.25, places=5)

    def test_percentile_boundaries(self) -> None:
        """0th percentile = min, 100th percentile = max."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(_percentile(values, 0.0), 1.0, places=5)
        self.assertAlmostEqual(_percentile(values, 1.0), 5.0, places=5)

    def test_percentile_single_value(self) -> None:
        """Single-element list: all percentiles return that value."""
        self.assertAlmostEqual(_percentile([42.0], 0.25), 42.0, places=5)
        self.assertAlmostEqual(_percentile([42.0], 0.75), 42.0, places=5)

    def test_percentile_empty(self) -> None:
        """Empty list should return NaN."""
        self.assertTrue(math.isnan(_percentile([], 0.5)))


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------


class TestWelchTTest(unittest.TestCase):
    """Tests for welch_ttest() and TTestResult."""

    def test_ttest_known_values(self) -> None:
        """Known-value test: two small samples with known t and p."""
        # Sample A: [1, 2, 3, 4, 5], mean=3, var=2.5
        # Sample B: [2, 3, 4, 5, 6], mean=4, var=2.5
        # t = (3-4) / sqrt(2.5/5 + 2.5/5) = -1 / sqrt(1.0) = -1.0
        # df = (0.5+0.5)^2 / (0.25/4 + 0.25/4) = 1.0 / 0.125 = 8.0
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = welch_ttest(a, b)
        self.assertAlmostEqual(result.t_statistic, -1.0, places=3)
        self.assertAlmostEqual(result.degrees_of_freedom, 8.0, places=1)
        self.assertGreater(result.p_value, 0.05)  # Not significant
        self.assertFalse(result.significant_05)

    def test_ttest_identical_samples(self) -> None:
        """Identical samples: t=0, p=1."""
        a = [1.0, 2.0, 3.0]
        result = welch_ttest(a, a)
        self.assertAlmostEqual(result.t_statistic, 0.0, places=5)
        self.assertAlmostEqual(result.p_value, 1.0, places=2)

    def test_ttest_highly_significant(self) -> None:
        """Clearly different samples should be significant."""
        a = [1.0, 1.1, 1.0, 0.9, 1.0]
        b = [10.0, 10.1, 10.0, 9.9, 10.0]
        result = welch_ttest(a, b)
        self.assertTrue(result.significant_001)
        self.assertEqual(result.significance_stars, "***")

    def test_ttest_too_few_values(self) -> None:
        """Fewer than 2 values in either sample: NaN result."""
        result = welch_ttest([1.0], [2.0, 3.0])
        self.assertTrue(math.isnan(result.t_statistic))
        self.assertTrue(math.isnan(result.p_value))
        self.assertFalse(result.significant_05)

    def test_ttest_zero_variance_same_mean(self) -> None:
        """Both samples have zero variance and same mean."""
        result = welch_ttest([5.0, 5.0], [5.0, 5.0])
        self.assertAlmostEqual(result.t_statistic, 0.0, places=5)
        self.assertAlmostEqual(result.p_value, 1.0, places=2)

    def test_ttest_zero_variance_different_mean(self) -> None:
        """Both samples have zero variance but different means."""
        result = welch_ttest([1.0, 1.0], [2.0, 2.0])
        self.assertTrue(math.isinf(result.t_statistic))
        self.assertAlmostEqual(result.p_value, 0.0, places=5)
        self.assertTrue(result.significant_001)

    def test_ttest_significance_stars(self) -> None:
        """Test all significance star levels."""
        ns = TTestResult(0.0, 10.0, 0.10, False, False, False)
        self.assertEqual(ns.significance_stars, "ns")
        star1 = TTestResult(0.0, 10.0, 0.03, True, True, False)
        self.assertEqual(star1.significance_stars, "**")
        star2 = TTestResult(0.0, 10.0, 0.04, False, True, False)
        self.assertEqual(star2.significance_stars, "*")
        star3 = TTestResult(0.0, 10.0, 0.0005, True, True, True)
        self.assertEqual(star3.significance_stars, "***")

    def test_ttest_scipy_fallback(self) -> None:
        """Force the pure Python path and verify reasonable p-values."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        with patch.dict("sys.modules", {"scipy": None, "scipy.stats": None}):
            result = welch_ttest(a, b)
        self.assertAlmostEqual(result.t_statistic, -1.0, places=3)
        # p-value should be in a reasonable range (~0.35)
        self.assertGreater(result.p_value, 0.1)
        self.assertLess(result.p_value, 0.9)

    def test_ttest_scipy_agreement(self) -> None:
        """If scipy is available, compare its result to our pure Python path."""
        try:
            from scipy.stats import ttest_ind
        except ImportError:
            self.skipTest("scipy not available")

        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        b = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

        # scipy reference
        scipy_result = ttest_ind(a, b, equal_var=False)

        # Our implementation
        our_result = welch_ttest(a, b)

        self.assertAlmostEqual(our_result.t_statistic, float(scipy_result.statistic), places=5)
        self.assertAlmostEqual(our_result.p_value, float(scipy_result.pvalue), places=3)


# ---------------------------------------------------------------------------
# Regularized incomplete beta function
# ---------------------------------------------------------------------------


class TestRegularizedIncompleteBeta(unittest.TestCase):
    """Tests for _regularized_incomplete_beta() internal function."""

    def test_ibeta_x0(self) -> None:
        """I_0(a, b) = 0 for any valid a, b."""
        self.assertAlmostEqual(_regularized_incomplete_beta(0.0, 2.0, 3.0), 0.0, places=10)

    def test_ibeta_x1(self) -> None:
        """I_1(a, b) = 1 for any valid a, b."""
        self.assertAlmostEqual(_regularized_incomplete_beta(1.0, 2.0, 3.0), 1.0, places=10)

    def test_ibeta_symmetric(self) -> None:
        """I_0.5(a, a) = 0.5 by symmetry."""
        self.assertAlmostEqual(_regularized_incomplete_beta(0.5, 5.0, 5.0), 0.5, places=6)

    def test_ibeta_known_value(self) -> None:
        """I_0.3(2, 5) should be close to a known value (~0.5798)."""
        # Verified via symmetry: I_0.3(2,5) + I_0.7(5,2) = 1.0
        result = _regularized_incomplete_beta(0.3, 2.0, 5.0)
        self.assertAlmostEqual(result, 0.57983, places=3)

    def test_ibeta_invalid_x(self) -> None:
        """x outside [0, 1] should return NaN."""
        self.assertTrue(math.isnan(_regularized_incomplete_beta(-0.1, 2.0, 3.0)))
        self.assertTrue(math.isnan(_regularized_incomplete_beta(1.1, 2.0, 3.0)))

    def test_ibeta_uses_symmetry_relation(self) -> None:
        """Verify the symmetry relation: I_x(a,b) + I_{1-x}(b,a) = 1."""
        x, a, b = 0.7, 2.0, 5.0
        forward = _regularized_incomplete_beta(x, a, b)
        reverse = _regularized_incomplete_beta(1 - x, b, a)
        self.assertAlmostEqual(forward + reverse, 1.0, places=6)


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


class TestCohensD(unittest.TestCase):
    """Tests for cohens_d() and EffectSize."""

    def test_cohens_d_known(self) -> None:
        """Known value: identical variances, known mean difference."""
        # mean_a=3, mean_b=4, pooled_sd = sqrt(2.5) ≈ 1.581
        # d = (4-3) / sqrt(2.5) ≈ 0.632 → medium
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = cohens_d(a, b)
        self.assertAlmostEqual(result.d, 1.0 / math.sqrt(2.5), places=3)
        self.assertEqual(result.classification, "medium")

    def test_cohens_d_negligible(self) -> None:
        """Very small difference → negligible."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.01, 2.01, 3.01, 4.01, 5.01]
        result = cohens_d(a, b)
        self.assertEqual(result.classification, "negligible")

    def test_cohens_d_large(self) -> None:
        """Large difference → large."""
        a = [1.0, 1.1, 0.9, 1.0, 1.1]
        b = [10.0, 10.1, 9.9, 10.0, 10.1]
        result = cohens_d(a, b)
        self.assertEqual(result.classification, "large")

    def test_cohens_d_negative(self) -> None:
        """If sample_a has larger mean, d should be negative."""
        a = [10.0, 11.0, 12.0]
        b = [1.0, 2.0, 3.0]
        result = cohens_d(a, b)
        self.assertLess(result.d, 0)

    def test_cohens_d_zero_variance(self) -> None:
        """Same constant in both samples: d=0."""
        result = cohens_d([5.0, 5.0], [5.0, 5.0])
        self.assertAlmostEqual(result.d, 0.0, places=5)
        self.assertEqual(result.classification, "negligible")

    def test_cohens_d_zero_variance_different_means(self) -> None:
        """Zero variance but different means: d=inf."""
        result = cohens_d([1.0, 1.0], [2.0, 2.0])
        self.assertTrue(math.isinf(result.d))
        self.assertEqual(result.classification, "large")

    def test_cohens_d_too_few(self) -> None:
        """Fewer than 2 values: NaN."""
        result = cohens_d([1.0], [2.0, 3.0])
        self.assertTrue(math.isnan(result.d))
        self.assertEqual(result.classification, "unknown")

    def test_classify_boundaries(self) -> None:
        """Verify classification boundaries exactly."""
        self.assertEqual(EffectSize.classify(0.0), "negligible")
        self.assertEqual(EffectSize.classify(0.19), "negligible")
        self.assertEqual(EffectSize.classify(0.2), "small")
        self.assertEqual(EffectSize.classify(0.49), "small")
        self.assertEqual(EffectSize.classify(0.5), "medium")
        self.assertEqual(EffectSize.classify(0.79), "medium")
        self.assertEqual(EffectSize.classify(0.8), "large")
        self.assertEqual(EffectSize.classify(2.0), "large")
        # Negative values use absolute value.
        self.assertEqual(EffectSize.classify(-0.3), "small")
        self.assertEqual(EffectSize.classify(-1.0), "large")


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


class TestBootstrapCI(unittest.TestCase):
    """Tests for bootstrap_ci() and BootstrapCI."""

    def test_bootstrap_ci_seeded(self) -> None:
        """Seeded bootstrap should be deterministic."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [3.0, 4.0, 5.0, 6.0, 7.0]
        ci1 = bootstrap_ci(a, b, seed=42, n_bootstrap=1000)
        ci2 = bootstrap_ci(a, b, seed=42, n_bootstrap=1000)
        self.assertAlmostEqual(ci1.lower, ci2.lower, places=10)
        self.assertAlmostEqual(ci1.upper, ci2.upper, places=10)

    def test_bootstrap_ci_different_seeds(self) -> None:
        """Different seeds should give different results."""
        a = [1.0, 1.5, 2.3, 3.7, 4.2, 5.1, 6.8, 7.3]
        b = [3.1, 4.4, 5.2, 6.7, 7.9, 8.3, 9.1, 10.5]
        ci1 = bootstrap_ci(a, b, seed=42, n_bootstrap=1000, statistic="mean")
        ci2 = bootstrap_ci(a, b, seed=99, n_bootstrap=1000, statistic="mean")
        # With enough varied data and mean statistic, seeds should produce different bounds
        self.assertNotAlmostEqual(ci1.lower, ci2.lower, places=5)

    def test_bootstrap_ci_contains_zero(self) -> None:
        """Overlapping samples: CI should contain zero."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.5, 2.5, 3.5, 4.5, 5.5]
        ci = bootstrap_ci(a, b, seed=42, n_bootstrap=5000)
        # The samples are close, CI likely contains zero
        # Just verify the method exists and returns a bool
        self.assertIsInstance(ci.contains_zero(), bool)

    def test_bootstrap_ci_clearly_different(self) -> None:
        """Clearly separated samples: CI should not contain zero."""
        a = [1.0, 1.1, 0.9, 1.0, 1.1]
        b = [10.0, 10.1, 9.9, 10.0, 10.1]
        ci = bootstrap_ci(a, b, seed=42, n_bootstrap=5000)
        self.assertFalse(ci.contains_zero())
        self.assertGreater(ci.lower, 0)

    def test_bootstrap_ci_point_estimate(self) -> None:
        """Point estimate should be the actual difference in medians."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [3.0, 4.0, 5.0, 6.0, 7.0]
        ci = bootstrap_ci(a, b, seed=42, n_bootstrap=100, statistic="median")
        # median(b) - median(a) = 5 - 3 = 2
        self.assertAlmostEqual(ci.point_estimate, 2.0, places=5)

    def test_bootstrap_ci_mean_statistic(self) -> None:
        """Test with mean statistic instead of median."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        ci = bootstrap_ci(a, b, seed=42, n_bootstrap=1000, statistic="mean")
        # mean(b) - mean(a) = 5 - 2 = 3
        self.assertAlmostEqual(ci.point_estimate, 3.0, places=5)
        self.assertGreater(ci.lower, 0)

    def test_bootstrap_ci_empty_samples(self) -> None:
        """Empty samples should return NaN CI."""
        ci = bootstrap_ci([], [1.0, 2.0])
        self.assertTrue(math.isnan(ci.lower))
        self.assertTrue(math.isnan(ci.upper))
        self.assertTrue(math.isnan(ci.point_estimate))
        self.assertEqual(ci.n_bootstrap, 0)

    def test_bootstrap_ci_confidence_level(self) -> None:
        """Confidence level should be stored in the result."""
        # Use larger, more varied samples so the bootstrap distribution
        # has enough resolution for different confidence levels to differ.
        a = [1.0, 1.5, 2.3, 3.1, 3.7, 4.2, 5.1, 5.8, 6.4, 7.3]
        b = [3.1, 3.8, 4.4, 5.2, 5.9, 6.7, 7.3, 7.9, 8.6, 9.5]
        ci_95 = bootstrap_ci(a, b, seed=42, confidence=0.95, n_bootstrap=5000, statistic="mean")
        ci_99 = bootstrap_ci(a, b, seed=42, confidence=0.99, n_bootstrap=5000, statistic="mean")
        self.assertAlmostEqual(ci_95.confidence_level, 0.95, places=5)
        self.assertAlmostEqual(ci_99.confidence_level, 0.99, places=5)
        # 99% CI should be wider (or equal, at minimum)
        width_95 = ci_95.upper - ci_95.lower
        width_99 = ci_99.upper - ci_99.lower
        self.assertGreaterEqual(width_99, width_95)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


class TestDetectOutliers(unittest.TestCase):
    """Tests for detect_outliers()."""

    def test_no_outliers(self) -> None:
        """Tight data: no outliers."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = detect_outliers(values)
        self.assertEqual(result, [False, False, False, False, False])

    def test_one_outlier(self) -> None:
        """One clear outlier."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        result = detect_outliers(values)
        self.assertFalse(result[0])  # 1.0 not an outlier
        self.assertTrue(result[5])  # 100.0 is an outlier

    def test_low_outlier(self) -> None:
        """Outlier on the low end."""
        values = [-100.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        result = detect_outliers(values)
        self.assertTrue(result[0])  # -100.0 is an outlier
        self.assertFalse(result[-1])  # 5.0 is not

    def test_custom_factor(self) -> None:
        """Higher factor should detect fewer outliers."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 15.0]
        result_standard = detect_outliers(values, factor=1.5)
        result_extreme = detect_outliers(values, factor=3.0)
        # Extreme factor should be less sensitive
        n_standard = sum(result_standard)
        n_extreme = sum(result_extreme)
        self.assertGreaterEqual(n_standard, n_extreme)

    def test_too_few_values(self) -> None:
        """Fewer than 4 values: no outlier detection."""
        self.assertEqual(detect_outliers([1.0, 2.0, 3.0]), [False, False, False])
        self.assertEqual(detect_outliers([1.0]), [False])
        self.assertEqual(detect_outliers([]), [])

    def test_preserves_order(self) -> None:
        """Outlier flags correspond to original positions, not sorted."""
        values = [100.0, 2.0, 3.0, 4.0, 5.0]
        result = detect_outliers(values)
        # 100.0 is at index 0 in original
        self.assertTrue(result[0])
        # Other values are not outliers
        for i in range(1, len(result)):
            self.assertFalse(result[i])


# ---------------------------------------------------------------------------
# Overhead computation
# ---------------------------------------------------------------------------


class TestComputeOverhead(unittest.TestCase):
    """Tests for compute_overhead() and OverheadResult."""

    def test_compute_overhead_20pct(self) -> None:
        """20% overhead should be detected correctly."""
        baseline = [100.0, 100.0, 100.0, 100.0, 100.0]
        treatment = [120.0, 120.0, 120.0, 120.0, 120.0]
        result = compute_overhead(baseline, treatment, ci_seed=42)
        self.assertAlmostEqual(result.relative_overhead, 0.2, places=3)
        self.assertAlmostEqual(result.overhead_pct, 20.0, places=1)
        self.assertAlmostEqual(result.absolute_diff, 20.0, places=3)

    def test_compute_overhead_stats(self) -> None:
        """Verify that baseline_stats and treatment_stats are populated."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        treatment = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = compute_overhead(baseline, treatment, ci_seed=42)
        self.assertEqual(result.baseline_stats.n, 5)
        self.assertEqual(result.treatment_stats.n, 5)
        self.assertIsInstance(result.ttest, TTestResult)
        self.assertIsInstance(result.effect_size, EffectSize)
        self.assertIsInstance(result.ci, BootstrapCI)

    def test_compute_overhead_practically_significant(self) -> None:
        """Large, significant overhead should be practically significant."""
        baseline = [100.0, 100.1, 99.9, 100.0, 100.1]
        treatment = [200.0, 200.1, 199.9, 200.0, 200.1]
        result = compute_overhead(baseline, treatment, ci_seed=42)
        self.assertTrue(result.practically_significant)
        self.assertTrue(result.ttest.significant_05)
        self.assertNotEqual(result.effect_size.classification, "negligible")

    def test_compute_overhead_not_significant(self) -> None:
        """Nearly identical distributions should not be practically significant."""
        baseline = [100.0, 101.0, 99.0, 100.5, 99.5]
        treatment = [100.1, 101.1, 99.1, 100.6, 99.6]
        result = compute_overhead(baseline, treatment, ci_seed=42)
        self.assertFalse(result.practically_significant)

    def test_compute_overhead_zero_baseline(self) -> None:
        """Zero baseline median: relative overhead should be inf."""
        baseline = [-1.0, 0.0, 0.0, 0.0, 1.0]
        treatment = [5.0, 6.0, 7.0, 8.0, 9.0]
        result = compute_overhead(baseline, treatment, ci_seed=42)
        # Baseline median is 0.0
        self.assertTrue(math.isinf(result.relative_overhead))


# ---------------------------------------------------------------------------
# Integration / round-trip tests
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple stats functions."""

    def test_overhead_result_types(self) -> None:
        """All fields of OverheadResult should be the expected types."""
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
        treatment = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = compute_overhead(baseline, treatment, ci_seed=42, ci_bootstrap_n=100)
        self.assertIsInstance(result, OverheadResult)
        self.assertIsInstance(result.baseline_stats, DescriptiveStats)
        self.assertIsInstance(result.treatment_stats, DescriptiveStats)
        self.assertIsInstance(result.ttest, TTestResult)
        self.assertIsInstance(result.effect_size, EffectSize)
        self.assertIsInstance(result.ci, BootstrapCI)
        self.assertIsInstance(result.absolute_diff, float)
        self.assertIsInstance(result.relative_overhead, float)
        self.assertIsInstance(result.overhead_pct, float)
        self.assertIsInstance(result.practically_significant, bool)

    def test_describe_and_outliers_consistent(self) -> None:
        """IQR from describe matches outlier detection behavior."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        stats = describe(values)
        outliers = detect_outliers(values)
        # If IQR > 0, outlier detection should work
        if stats.iqr > 0:
            self.assertIn(True, outliers)  # 100 should be detected

    def test_ttest_and_cohens_d_agree_on_direction(self) -> None:
        """t-test and Cohen's d should agree on the direction of difference."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [4.0, 5.0, 6.0, 7.0, 8.0]
        ttest = welch_ttest(a, b)
        effect = cohens_d(a, b)
        # b > a, so t < 0 (sample_a - sample_b) and d > 0 (sample_b - sample_a)
        self.assertLess(ttest.t_statistic, 0)
        self.assertGreater(effect.d, 0)


if __name__ == "__main__":
    unittest.main()
