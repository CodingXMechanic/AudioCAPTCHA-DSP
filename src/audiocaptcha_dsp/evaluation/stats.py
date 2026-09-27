from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.stats import mannwhitneyu, ttest_ind

# ---------------------------------------------------------------------------
# Metadata blocklist for metric discovery
# ---------------------------------------------------------------------------
# These keys appear in result dicts (from ExperimentResult.to_dict() or from
# the runner's condition_metrics flat dicts) but are bookkeeping fields, not
# scientific measurements. They are excluded from metric discovery in
# summarize_condition() and compare_conditions() via _discover_metrics().
#
# MAINTENANCE: When adding new fields to ExperimentResult (core/types.py) or
# to the runner's per-sample dict (experiments/runner.py), add the key here
# if the field is numeric and should NOT be treated as a metric.
# ---------------------------------------------------------------------------
_METADATA_KEYS: frozenset[str] = frozenset({
    "condition_index",
    "condition_params",
    "sample_index",
    "sample_id",
    "experiment_id",
    "seed",
    "duration_seconds",
    "transform_chain",
    "metrics",
})


def _discover_metrics(result: dict[str, Any]) -> list[str]:
    """Discover metric keys in a result dict, excluding metadata.

    Returns sorted list of key names whose values are numeric and are not
    in the _METADATA_KEYS blocklist.
    """
    keys: list[str] = []
    for key, val in result.items():
        if key in _METADATA_KEYS:
            continue
        if isinstance(val, (int, float, np.integer, np.floating)):
            keys.append(key)
    return sorted(keys)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic(sample)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_stats, alpha))
    upper = float(np.quantile(boot_stats, 1.0 - alpha))
    point = float(statistic(data))
    return (point, lower, upper)


def _bca_via_scipy(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float] | None:
    """Adapter to scipy.stats.bootstrap(method='BCa').

    Returns (point, lower, upper) on success, or None if BCa fails
    (degenerate data, NaN bounds, or exception). Caller should fall
    back to percentile bootstrap when None is returned.
    """
    from scipy.stats import bootstrap as scipy_bootstrap

    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n < 2:
        return None

    point = float(statistic(data))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = scipy_bootstrap(
                (data,),
                statistic,
                n_resamples=n_bootstrap,
                confidence_level=ci,
                method="BCa",
                random_state=seed,
            )
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
        if not (np.isfinite(low) and np.isfinite(high)):
            return None
        return (point, low, high)
    except Exception:
        return None


def bootstrap_ci_bca(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """BCa bootstrap confidence interval via scipy, with percentile fallback.

    Delegates to scipy.stats.bootstrap(method='BCa'). If BCa fails
    (degenerate data, NaN bounds, numerical issues), falls back to the
    percentile bootstrap (bootstrap_ci).

    Returns (point_estimate, lower_bound, upper_bound).
    """
    result = _bca_via_scipy(data, statistic, n_bootstrap, ci, seed)
    if result is not None:
        return result
    return bootstrap_ci(data, statistic, n_bootstrap, ci, seed)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size between two independent groups.

    Uses pooled standard deviation.
    Returns 0.0 if either group has fewer than 2 samples or if
    pooled standard deviation is zero.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_diff = np.mean(a) - np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-15:
        return 0.0
    return float(mean_diff / pooled_std)


def hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Hedges' g effect size (bias-corrected Cohen's d).

    Applies the small-sample correction factor J(df).
    For df > 150, returns Cohen's d directly (correction ~ 1.0).
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    d = cohens_d(a, b)
    df = n_a + n_b - 2
    if df < 1:
        return float(d)
    if df > 150:
        return float(d)
    try:
        correction = gamma_func(df / 2.0) / (
            np.sqrt(df / 2.0) * gamma_func((df - 1) / 2.0)
        )
        if np.isfinite(correction):
            return float(d * correction)
    except (OverflowError, ValueError, ZeroDivisionError):
        pass
    return float(d)


# ---------------------------------------------------------------------------
# Multiple comparison corrections
# ---------------------------------------------------------------------------

def bonferroni_correction(
    p_values: list[float], alpha: float = 0.05
) -> tuple[list[bool], float]:
    """Bonferroni correction for multiple comparisons.

    Returns (rejections list, corrected alpha).
    """
    n = len(p_values)
    if n == 0:
        return ([], alpha)
    corrected_alpha = alpha / n
    rejections = [p < corrected_alpha for p in p_values]
    return (rejections, corrected_alpha)


def benjamini_hochberg_correction(
    p_values: list[float], alpha: float = 0.05
) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Returns (rejections list, adjusted p-values).
    """
    n = len(p_values)
    if n == 0:
        return ([], [])
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n

    # Compute adjusted p-values from largest to smallest
    prev = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        raw_adj = p * n / (rank_idx + 1)
        adjusted[orig_idx] = min(raw_adj, prev)
        prev = adjusted[orig_idx]

    # Step-up procedure for rejections
    rejections = [False] * n
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        if p <= alpha * (rank_idx + 1) / n:
            for j in range(rank_idx + 1):
                rejections[indexed[j][0]] = True
            break

    return (rejections, adjusted)


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def mann_whitney_u(
    group_a: np.ndarray, group_b: np.ndarray
) -> tuple[float, float]:
    """Mann-Whitney U test for two independent groups.

    Returns (U_statistic, p_value).
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return (0.0, 1.0)
    result = mannwhitneyu(a, b, alternative="two-sided")
    return (float(result.statistic), float(result.pvalue))


def welch_ttest(
    group_a: np.ndarray, group_b: np.ndarray
) -> tuple[float, float]:
    """Welch's t-test (unequal variance) for two independent groups.

    Returns (t_statistic, p_value).
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return (0.0, 1.0)
    result = ttest_ind(a, b, equal_var=False)
    return (float(result.statistic), float(result.pvalue))


# ---------------------------------------------------------------------------
# Condition summary
# ---------------------------------------------------------------------------

@dataclass
class ConditionSummary:
    condition_index: int
    condition_params: dict
    n_samples: int
    metric_summaries: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "condition_index": self.condition_index,
            "condition_params": self.condition_params,
            "n_samples": self.n_samples,
            "metric_summaries": self.metric_summaries,
        }


def summarize_condition(
    results: list[dict[str, Any]],
    ci: float = 0.95,
    seed: int = 42,
    method: str = "percentile",
) -> ConditionSummary:
    """Summarize per-condition results with confidence intervals.

    Parameters
    ----------
    results : list of dict
        Per-sample result records, each containing numeric metric fields.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for bootstrap resampling.
    method : str
        Bootstrap method: "percentile" (default, backward-compatible)
        or "bca" (BCa via scipy with percentile fallback).
    """
    if not results:
        return ConditionSummary(condition_index=-1, condition_params={}, n_samples=0)
    numeric_keys = _discover_metrics(results[0])
    summary: dict[str, dict[str, float]] = {}
    for metric_name in numeric_keys:
        raw = [r[metric_name] for r in results if metric_name in r]
        values = np.array([v for v in raw if np.isfinite(v)], dtype=np.float64)
        if len(values) == 0:
            continue
        n_boot = min(1000, max(100, len(values) * 10))
        if method == "bca":
            point, lower, upper = bootstrap_ci_bca(
                values, n_bootstrap=n_boot, ci=ci, seed=seed
            )
        else:
            point, lower, upper = bootstrap_ci(
                values, n_bootstrap=n_boot, ci=ci, seed=seed
            )
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "ci_lower": lower,
            "ci_upper": upper,
            "n": len(values),
        }
    return ConditionSummary(
        condition_index=results[0].get("condition_index", -1),
        condition_params=results[0].get("condition_params", {}),
        n_samples=len(results),
        metric_summaries=summary,
    )


# ---------------------------------------------------------------------------
# Condition comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of comparing two experimental conditions on a single metric."""

    metric_name: str
    condition_a_index: int
    condition_b_index: int
    condition_a_params: dict
    condition_b_params: dict
    mean_a: float
    mean_b: float
    mean_diff: float
    relative_diff_pct: float
    cohens_d: float
    hedges_g: float
    mann_whitney_u: float
    mann_whitney_p: float
    welch_t: float
    welch_p: float
    ci_lower: float
    ci_upper: float
    n_a: int
    n_b: int

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "condition_a_index": self.condition_a_index,
            "condition_b_index": self.condition_b_index,
            "condition_a_params": self.condition_a_params,
            "condition_b_params": self.condition_b_params,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_diff": self.mean_diff,
            "relative_diff_pct": self.relative_diff_pct,
            "cohens_d": self.cohens_d,
            "hedges_g": self.hedges_g,
            "mann_whitney_u": self.mann_whitney_u,
            "mann_whitney_p": self.mann_whitney_p,
            "welch_t": self.welch_t,
            "welch_p": self.welch_p,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_a": self.n_a,
            "n_b": self.n_b,
        }


def compare_conditions(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
    condition_a_index: int = -1,
    condition_b_index: int = -1,
    condition_a_params: dict | None = None,
    condition_b_params: dict | None = None,
    ci: float = 0.95,
    seed: int = 42,
) -> list[ComparisonResult]:
    """Compare two conditions across all shared numeric metrics.

    For each shared metric, computes descriptive statistics, effect sizes,
    frequentist test p-values, and a bootstrap CI on the difference of means.

    Parameters
    ----------
    results_a, results_b : list of dict
        Per-sample result records for each condition.
    condition_a_index, condition_b_index : int
        Condition indices for labeling.
    condition_a_params, condition_b_params : dict, optional
        Condition parameter dicts. Extracted from results if not provided.
    ci : float
        Confidence level for the bootstrap CI on the mean difference.
    seed : int
        Random seed for bootstrap.

    Returns
    -------
    list of ComparisonResult
        One per shared numeric metric.
    """
    if not results_a or not results_b:
        return []

    keys_a = set(_discover_metrics(results_a[0]))
    keys_b = set(_discover_metrics(results_b[0]))
    shared_keys = sorted(keys_a & keys_b)

    a_params = (
        condition_a_params
        if condition_a_params is not None
        else (results_a[0].get("condition_params", {}) if results_a else {})
    )
    b_params = (
        condition_b_params
        if condition_b_params is not None
        else (results_b[0].get("condition_params", {}) if results_b else {})
    )

    comparisons: list[ComparisonResult] = []
    for metric_name in shared_keys:
        values_a = np.array(
            [
                r[metric_name]
                for r in results_a
                if metric_name in r and np.isfinite(r[metric_name])
            ],
            dtype=np.float64,
        )
        values_b = np.array(
            [
                r[metric_name]
                for r in results_b
                if metric_name in r and np.isfinite(r[metric_name])
            ],
            dtype=np.float64,
        )
        if len(values_a) < 2 or len(values_b) < 2:
            continue

        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        mean_diff = mean_a - mean_b
        relative_diff = (mean_diff / mean_a * 100.0) if abs(mean_a) > 1e-15 else 0.0

        # Bootstrap CI on the difference of means
        rng = np.random.default_rng(seed)
        n_boot = 1000
        boot_diffs = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            sa = rng.choice(values_a, size=len(values_a), replace=True)
            sb = rng.choice(values_b, size=len(values_b), replace=True)
            boot_diffs[i] = np.mean(sa) - np.mean(sb)
        alpha = (1.0 - ci) / 2.0
        ci_lower = float(np.quantile(boot_diffs, alpha))
        ci_upper = float(np.quantile(boot_diffs, 1.0 - alpha))

        d = cohens_d(values_a, values_b)
        g = hedges_g(values_a, values_b)
        mw_u, mw_p = mann_whitney_u(values_a, values_b)
        wt_t, wt_p = welch_ttest(values_a, values_b)

        comparisons.append(
            ComparisonResult(
                metric_name=metric_name,
                condition_a_index=condition_a_index,
                condition_b_index=condition_b_index,
                condition_a_params=a_params if isinstance(a_params, dict) else {},
                condition_b_params=b_params if isinstance(b_params, dict) else {},
                mean_a=mean_a,
                mean_b=mean_b,
                mean_diff=mean_diff,
                relative_diff_pct=relative_diff,
                cohens_d=d,
                hedges_g=g,
                mann_whitney_u=mw_u,
                mann_whitney_p=mw_p,
                welch_t=wt_t,
                welch_p=wt_p,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_a=len(values_a),
                n_b=len(values_b),
            )
        )

    return comparisons


# ---------------------------------------------------------------------------
# Paired Non-Parametric & Permutation Tests
# ---------------------------------------------------------------------------

def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired two-sided permutation test for difference of means.
    
    Returns (observed_mean_diff, p_value).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("Paired test requires equal-length arrays.")
    if len(a) < 2:
        return (0.0, 1.0)
        
    diffs = a - b
    obs_diff = float(np.mean(diffs))
    
    rng = np.random.default_rng(seed)
    # Random sign flips for paired difference
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(diffs)))
    perm_diffs = np.mean(signs * diffs, axis=1)
    
    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
    return (obs_diff, p_value)


def wilcoxon_signed_rank(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test.
    
    Returns (statistic, p_value).
    """
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("Wilcoxon test requires equal-length arrays.")
    if len(a) < 2 or np.allclose(a, b):
        return (0.0, 1.0)
    try:
        res = wilcoxon(a, b)
        return (float(res.statistic), float(res.pvalue))
    except Exception:
        return (0.0, 1.0)


# ---------------------------------------------------------------------------
# Multi-Objective Pareto Frontier Analysis
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    candidates: list[dict[str, Any]],
    objectives: dict[str, str],
) -> list[dict[str, Any]]:
    """Identify Pareto-efficient candidates across multiple objectives.
    
    A candidate X is dominated by Y if Y is at least as good as X across all
    objectives and strictly better than X in at least one objective.
    
    Parameters
    ----------
    candidates : list of dict
        Each dict must contain metrics for all keys in `objectives`.
    objectives : dict[str, str]
        Mapping from metric_name to direction: 'max' or 'min'.
        e.g. {'human_success_rate': 'max', 'wer': 'max', 'mbsd': 'min', 'duration_seconds': 'min'}
        
    Returns
    -------
    list of dict
        Copy of candidates with boolean field 'is_pareto' and 'dominating_count' added.
    """
    if not candidates:
        return []
        
    n = len(candidates)
    results = [dict(c) for c in candidates]
    
    # Extract numerical arrays normalized such that higher is always better
    norm_vals = np.zeros((n, len(objectives)), dtype=np.float64)
    obj_keys = list(objectives.keys())
    
    for j, (key, direction) in enumerate(objectives.items()):
        for i, c in enumerate(candidates):
            val = float(c.get(key, 0.0))
            norm_vals[i, j] = val if direction.lower() == "max" else -val
            
    is_dominated = np.zeros(n, dtype=bool)
    dominating_count = np.zeros(n, dtype=int)
    
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            # Check if k dominates i: k >= i in all objectives and k > i in at least one
            greater_equal = norm_vals[k] >= norm_vals[i]
            strictly_greater = norm_vals[k] > norm_vals[i]
            if np.all(greater_equal) and np.any(strictly_greater):
                is_dominated[i] = True
                dominating_count[i] += 1
                
    for i in range(n):
        results[i]["is_pareto"] = bool(not is_dominated[i])
        results[i]["dominance_rank"] = int(dominating_count[i])
        
    return results

