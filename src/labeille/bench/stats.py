"""Statistical functions for benchmark comparison.

Provides Welch's t-test, bootstrap confidence intervals, Cohen's d
effect size, outlier detection, and summary statistics — all in pure
Python with no external dependencies.

If scipy.stats is available, it is used for the t-test p-value
calculation (more precise). Otherwise, a pure Python approximation
of the Student's t CDF is used.

References:
    Welch's t-test: Welch, B. L. (1947). "The generalization of
        'Student's' problem when several different population
        variances are involved." Biometrika 34(1-2): 28-35.
    Cohen's d: Cohen, J. (1988). "Statistical Power Analysis for
        the Behavioral Sciences." 2nd ed.
    Bootstrap CI: Efron, B. & Tibshirani, R. J. (1993). "An
        Introduction to the Bootstrap."
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


@dataclass
class DescriptiveStats:
    """Summary statistics for a sample."""

    n: int
    mean: float
    median: float
    stdev: float
    min: float
    max: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    iqr: float  # interquartile range
    cv: float  # coefficient of variation (stdev/mean)

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to a dict with rounded values."""
        return {
            "n": self.n,
            "mean": round(self.mean, 6),
            "median": round(self.median, 6),
            "stdev": round(self.stdev, 6),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
            "q1": round(self.q1, 6),
            "q3": round(self.q3, 6),
            "iqr": round(self.iqr, 6),
            "cv": round(self.cv, 6),
        }


def describe(values: Sequence[float]) -> DescriptiveStats:
    """Compute descriptive statistics for a sample.

    Args:
        values: A sequence of numeric values. Must have at least 1
            element for basic stats, at least 2 for stdev/CV.

    Returns:
        DescriptiveStats with all fields populated. If n < 2,
        stdev and CV are 0.0.
    """
    if not values:
        return DescriptiveStats(
            n=0,
            mean=float("nan"),
            median=float("nan"),
            stdev=float("nan"),
            min=float("nan"),
            max=float("nan"),
            q1=float("nan"),
            q3=float("nan"),
            iqr=float("nan"),
            cv=float("nan"),
        )

    sorted_v = sorted(values)
    n = len(sorted_v)
    mean = statistics.mean(sorted_v)
    median = statistics.median(sorted_v)

    if n >= 2:
        stdev = statistics.stdev(sorted_v)
        cv = stdev / mean if mean != 0 else float("inf")
    else:
        stdev = 0.0
        cv = 0.0

    q1 = _percentile(sorted_v, 0.25)
    q3 = _percentile(sorted_v, 0.75)

    return DescriptiveStats(
        n=n,
        mean=mean,
        median=median,
        stdev=stdev,
        min=sorted_v[0],
        max=sorted_v[-1],
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        cv=cv,
    )


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile using linear interpolation.

    Equivalent to numpy.percentile with interpolation='linear'.
    Assumes sorted_values is already sorted in ascending order.
    """
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_values[0]

    # Linear interpolation between data points.
    k = (n - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d = k - f
    return sorted_values[int(f)] * (1 - d) + sorted_values[int(c)] * d


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------


@dataclass
class TTestResult:
    """Result of Welch's t-test comparing two independent samples."""

    t_statistic: float
    degrees_of_freedom: float
    p_value: float
    significant_01: bool  # p < 0.01
    significant_05: bool  # p < 0.05
    significant_001: bool  # p < 0.001

    @property
    def significance_stars(self) -> str:
        """Return significance stars: ***, **, *, or ns."""
        if self.significant_001:
            return "***"
        if self.significant_01:
            return "**"
        if self.significant_05:
            return "*"
        return "ns"


def welch_ttest(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> TTestResult:
    """Perform Welch's t-test for two independent samples.

    Tests the null hypothesis that the two populations have equal
    means, without assuming equal variances.

    Args:
        sample_a: First sample (at least 2 values).
        sample_b: Second sample (at least 2 values).

    Returns:
        TTestResult with t-statistic, degrees of freedom, and
        two-tailed p-value.

    If either sample has fewer than 2 values or zero variance,
    returns a result with NaN values and no significance.
    """
    na, nb = len(sample_a), len(sample_b)

    if na < 2 or nb < 2:
        return TTestResult(
            t_statistic=float("nan"),
            degrees_of_freedom=float("nan"),
            p_value=float("nan"),
            significant_01=False,
            significant_05=False,
            significant_001=False,
        )

    mean_a = statistics.mean(sample_a)
    mean_b = statistics.mean(sample_b)
    var_a = statistics.variance(sample_a)
    var_b = statistics.variance(sample_b)

    # Handle zero variance.
    if var_a == 0 and var_b == 0:
        # Both samples have zero variance. If means are equal,
        # the difference is exactly zero (not testable).
        if mean_a == mean_b:
            return TTestResult(0.0, float("inf"), 1.0, False, False, False)
        else:
            # Means differ with zero variance → infinite t-statistic.
            return TTestResult(float("inf"), 0.0, 0.0, True, True, True)

    # Welch's t-statistic.
    se_a = var_a / na
    se_b = var_b / nb
    se_diff = math.sqrt(se_a + se_b)

    if se_diff == 0:
        return TTestResult(0.0, float("inf"), 1.0, False, False, False)

    t = (mean_a - mean_b) / se_diff

    # Welch-Satterthwaite degrees of freedom.
    numerator = (se_a + se_b) ** 2
    denominator = (se_a**2 / (na - 1)) + (se_b**2 / (nb - 1))
    if denominator == 0:
        df = float("inf")
    else:
        df = numerator / denominator

    # Two-tailed p-value.
    p = _t_cdf_two_tailed(abs(t), df)

    return TTestResult(
        t_statistic=t,
        degrees_of_freedom=df,
        p_value=p,
        significant_01=p < 0.01,
        significant_05=p < 0.05,
        significant_001=p < 0.001,
    )


def _t_cdf_two_tailed(t: float, df: float) -> float:
    """Compute two-tailed p-value for Student's t-distribution.

    Tries scipy first; falls back to a pure Python approximation
    using the regularized incomplete beta function.

    The p-value is P(|T| > t) = 2 * P(T > t) for a t-distribution
    with ``df`` degrees of freedom.
    """
    if math.isinf(t):
        return 0.0
    if math.isnan(t) or math.isnan(df):
        return float("nan")
    if df <= 0:
        return float("nan")

    try:
        from scipy.stats import t as t_dist  # type: ignore[import-untyped]

        return float(2.0 * t_dist.sf(t, df))
    except ImportError:
        pass

    # Pure Python fallback using the regularized incomplete beta
    # function. The CDF of the t-distribution is:
    # P(T <= t) = 1 - 0.5 * I_x(df/2, 1/2)
    # where x = df / (df + t^2).
    # Two-tailed: p = I_x(df/2, 1/2)
    x = df / (df + t * t)
    p = _regularized_incomplete_beta(x, df / 2.0, 0.5)
    return min(max(p, 0.0), 1.0)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Compute the regularized incomplete beta function I_x(a, b).

    Uses the continued fraction expansion (Lentz's method).
    Accurate to ~10 decimal places for typical benchmark parameters
    (df between 2 and 100).

    Reference: Numerical Recipes, Chapter 6.4.
    """
    if x < 0 or x > 1:
        return float("nan")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Use the symmetry relation if x > (a+1)/(a+b+2) for faster
    # convergence.
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(1 - x, b, a)

    # Log of the prefactor: x^a * (1-x)^b / (a * B(a,b))
    # B(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    prefactor = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta - math.log(a))

    # Continued fraction expansion (Lentz's method).
    max_iter = 200
    epsilon = 1e-14
    tiny = 1e-30

    # Initialize Lentz's algorithm.
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even term: d_{2m}
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        f *= c * d

        # Odd term: d_{2m+1}
        num = -((a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < epsilon:
            break

    return prefactor * f


# ---------------------------------------------------------------------------
# Cohen's d effect size
# ---------------------------------------------------------------------------


@dataclass
class EffectSize:
    """Cohen's d effect size with classification."""

    d: float
    classification: str  # negligible, small, medium, large

    @staticmethod
    def classify(d: float) -> str:
        """Classify effect size per Cohen's conventions.

        |d| < 0.2: negligible
        |d| < 0.5: small
        |d| < 0.8: medium
        |d| >= 0.8: large
        """
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        if d_abs < 0.5:
            return "small"
        if d_abs < 0.8:
            return "medium"
        return "large"


def cohens_d(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> EffectSize:
    """Compute Cohen's d effect size for two independent samples.

    Uses the pooled standard deviation:
    d = (mean_b - mean_a) / s_pooled

    where s_pooled = sqrt(((n_a-1)*var_a + (n_b-1)*var_b) / (n_a+n_b-2))

    A positive d means sample_b has a larger mean than sample_a.

    Returns EffectSize with d=NaN if either sample has < 2 values.
    """
    na, nb = len(sample_a), len(sample_b)

    if na < 2 or nb < 2:
        return EffectSize(d=float("nan"), classification="unknown")

    mean_a = statistics.mean(sample_a)
    mean_b = statistics.mean(sample_b)
    var_a = statistics.variance(sample_a)
    var_b = statistics.variance(sample_b)

    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    if pooled_var == 0:
        if mean_a == mean_b:
            return EffectSize(d=0.0, classification="negligible")
        return EffectSize(
            d=float("inf") if mean_b > mean_a else float("-inf"),
            classification="large",
        )

    d = (mean_b - mean_a) / math.sqrt(pooled_var)
    return EffectSize(d=d, classification=EffectSize.classify(d))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for the difference in means."""

    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int
    point_estimate: float  # observed difference in statistic

    def contains_zero(self) -> bool:
        """True if the CI includes zero (no significant difference)."""
        return self.lower <= 0 <= self.upper


def bootstrap_ci(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "median",
    seed: int | None = None,
) -> BootstrapCI:
    """Compute a bootstrap confidence interval for the difference.

    Computes the difference statistic(B) - statistic(A) for
    n_bootstrap resamples and returns the percentile CI.

    Args:
        sample_a: Baseline sample.
        sample_b: Treatment sample.
        confidence: Confidence level (default 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap resamples.
        statistic: "mean" or "median".
        seed: Random seed for reproducibility.

    Returns:
        BootstrapCI with lower and upper bounds.
    """
    if not sample_a or not sample_b:
        return BootstrapCI(
            lower=float("nan"),
            upper=float("nan"),
            confidence_level=confidence,
            n_bootstrap=0,
            point_estimate=float("nan"),
        )

    rng = random.Random(seed)
    list_a = list(sample_a)
    list_b = list(sample_b)

    stat_fn = statistics.median if statistic == "median" else statistics.mean

    point = stat_fn(list_b) - stat_fn(list_a)

    differences: list[float] = []
    for _ in range(n_bootstrap):
        resample_a = rng.choices(list_a, k=len(list_a))
        resample_b = rng.choices(list_b, k=len(list_b))
        diff = stat_fn(resample_b) - stat_fn(resample_a)
        differences.append(diff)

    differences.sort()

    alpha = 1 - confidence
    lower_idx = int(math.floor(alpha / 2 * n_bootstrap))
    upper_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    return BootstrapCI(
        lower=differences[lower_idx],
        upper=differences[upper_idx],
        confidence_level=confidence,
        n_bootstrap=n_bootstrap,
        point_estimate=point,
    )


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def detect_outliers(
    values: Sequence[float],
    *,
    factor: float = 1.5,
) -> list[bool]:
    """Detect outliers using the IQR method.

    A value is an outlier if it falls below Q1 - factor*IQR or
    above Q3 + factor*IQR.

    Args:
        values: The data points.
        factor: IQR multiplier (default 1.5 for standard outliers,
                use 3.0 for extreme outliers).

    Returns:
        A list of booleans, True for outlier positions.
    """
    if len(values) < 4:
        return [False] * len(values)

    sorted_v = sorted(values)
    q1 = _percentile(sorted_v, 0.25)
    q3 = _percentile(sorted_v, 0.75)
    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    return [v < lower or v > upper for v in values]


# ---------------------------------------------------------------------------
# Overhead computation
# ---------------------------------------------------------------------------


@dataclass
class OverheadResult:
    """Result of comparing two conditions (overhead analysis)."""

    baseline_stats: DescriptiveStats
    treatment_stats: DescriptiveStats
    absolute_diff: float  # median_treatment - median_baseline
    relative_overhead: float  # as a fraction (0.20 = 20%)
    overhead_pct: float  # as a percentage (20.0)
    ttest: TTestResult
    effect_size: EffectSize
    ci: BootstrapCI

    @property
    def practically_significant(self) -> bool:
        """True if the difference is both statistically significant
        and has a non-negligible effect size."""
        return self.ttest.significant_05 and self.effect_size.classification != "negligible"


def compute_overhead(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    ci_confidence: float = 0.95,
    ci_bootstrap_n: int = 10000,
    ci_seed: int | None = None,
) -> OverheadResult:
    """Compute the overhead of treatment relative to baseline.

    Produces descriptive stats, t-test, effect size, and bootstrap
    confidence interval in one call.

    Args:
        baseline: Timing measurements for the baseline condition.
        treatment: Timing measurements for the treatment condition.
        ci_confidence: Confidence level for the bootstrap CI.
        ci_bootstrap_n: Number of bootstrap resamples.
        ci_seed: Random seed for reproducibility.
    """
    baseline_stats = describe(baseline)
    treatment_stats = describe(treatment)

    abs_diff = treatment_stats.median - baseline_stats.median
    if baseline_stats.median > 0:
        rel = abs_diff / baseline_stats.median
    else:
        rel = float("inf") if abs_diff > 0 else 0.0

    return OverheadResult(
        baseline_stats=baseline_stats,
        treatment_stats=treatment_stats,
        absolute_diff=abs_diff,
        relative_overhead=rel,
        overhead_pct=rel * 100,
        ttest=welch_ttest(list(baseline), list(treatment)),
        effect_size=cohens_d(list(baseline), list(treatment)),
        ci=bootstrap_ci(
            list(baseline),
            list(treatment),
            confidence=ci_confidence,
            n_bootstrap=ci_bootstrap_n,
            seed=ci_seed,
        ),
    )
