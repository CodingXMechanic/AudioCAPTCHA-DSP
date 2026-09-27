from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.evaluation.stats import (
    bootstrap_ci,
    bootstrap_ci_bca,
    cohens_d,
    hedges_g,
    bonferroni_correction,
    benjamini_hochberg_correction,
    mann_whitney_u,
    welch_ttest,
    compare_conditions,
    ComparisonResult,
    summarize_condition,
)


# ---------------------------------------------------------------------------
# bootstrap_ci (percentile — existing function, regression tests)
# ---------------------------------------------------------------------------
class TestBootstrapCi:
    def test_returns_three_floats(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        point, lower, upper = bootstrap_ci(data, seed=42)
        assert isinstance(point, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_lower_le_point_le_upper(self) -> None:
        data = np.random.default_rng(7).normal(5.0, 2.0, size=50)
        point, lower, upper = bootstrap_ci(data, seed=42)
        assert lower <= point <= upper

    def test_symmetric_data_symmetric_ci(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        point, lower, upper = bootstrap_ci(data, seed=42)
        assert abs((point - lower) - (upper - point)) < 1.0

    def test_empty_data_returns_nan(self) -> None:
        point, lower, upper = bootstrap_ci(np.array([]))
        assert np.isnan(point)
        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_reproducible_with_seed(self) -> None:
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        r1 = bootstrap_ci(data, seed=99)
        r2 = bootstrap_ci(data, seed=99)
        assert r1 == r2

    def test_different_seeds_differ_slightly(self) -> None:
        data = np.random.default_rng(0).normal(0, 1, size=100)
        r1 = bootstrap_ci(data, seed=1)
        r2 = bootstrap_ci(data, seed=2)
        assert r1 != r2

    def test_point_estimate_is_mean(self) -> None:
        data = np.array([2.0, 4.0, 6.0, 8.0])
        point, _, _ = bootstrap_ci(data, seed=42)
        assert point == pytest.approx(5.0)

    def test_ci_widens_with_smaller_sample(self) -> None:
        rng = np.random.default_rng(42)
        wide = bootstrap_ci(rng.normal(0, 1, size=10), seed=42)
        narrow = bootstrap_ci(rng.normal(0, 1, size=1000), seed=42)
        wide_width = wide[2] - wide[1]
        narrow_width = narrow[2] - narrow[1]
        assert wide_width > narrow_width


# ---------------------------------------------------------------------------
# bootstrap_ci_bca (new — scipy adapter with fallback)
# ---------------------------------------------------------------------------
class TestBootstrapCiBca:
    def test_returns_three_floats(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        point, lower, upper = bootstrap_ci_bca(data, seed=42)
        assert isinstance(point, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_lower_le_point_le_upper(self) -> None:
        data = np.random.default_rng(7).normal(5.0, 2.0, size=50)
        point, lower, upper = bootstrap_ci_bca(data, seed=42)
        assert lower <= point <= upper

    def test_finite_for_normal_data(self) -> None:
        data = np.random.default_rng(42).normal(0, 1, size=50)
        point, lower, upper = bootstrap_ci_bca(data, seed=42)
        assert np.isfinite(point)
        assert np.isfinite(lower)
        assert np.isfinite(upper)

    def test_reproducible_with_seed(self) -> None:
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        r1 = bootstrap_ci_bca(data, seed=99)
        r2 = bootstrap_ci_bca(data, seed=99)
        assert r1 == r2

    def test_zero_variance_falls_back_to_percentile(self) -> None:
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        point, lower, upper = bootstrap_ci_bca(data, seed=42)
        assert np.isfinite(point)
        assert np.isfinite(lower)
        assert np.isfinite(upper)
        assert point == pytest.approx(5.0)

    def test_n2_falls_back_gracefully(self) -> None:
        data = np.array([1.0, 2.0])
        point, lower, upper = bootstrap_ci_bca(data, seed=42)
        assert np.isfinite(point)
        assert np.isfinite(lower)
        assert np.isfinite(upper)

    def test_point_estimate_is_mean(self) -> None:
        data = np.array([2.0, 4.0, 6.0, 8.0])
        point, _, _ = bootstrap_ci_bca(data, seed=42)
        assert point == pytest.approx(5.0)

    def test_skewed_data_asymmetric_ci(self) -> None:
        data = np.random.default_rng(42).exponential(2.0, size=100)
        point, lower, upper = bootstrap_ci_bca(data, seed=42)
        lower_dist = point - lower
        upper_dist = upper - point
        # For exponential, upper tail should be longer
        assert upper_dist > lower_dist * 0.5  # rough asymmetry check

    def test_bca_similar_to_percentile_for_symmetric(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        pct = bootstrap_ci(data, seed=42)
        bca = bootstrap_ci_bca(data, seed=42)
        # Both should have similar point estimates
        assert pct[0] == pytest.approx(bca[0])
        # CI widths should be in the same ballpark
        pct_width = pct[2] - pct[1]
        bca_width = bca[2] - bca[1]
        assert abs(pct_width - bca_width) < 2.0


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------
class TestCohensD:
    def test_identical_groups_returns_zero(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cohens_d(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_known_effect_size(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0.0, 1.0, size=1000)
        b = rng.normal(1.0, 1.0, size=1000)
        d = cohens_d(a, b)
        assert d == pytest.approx(-1.0, abs=0.15)

    def test_zero_variance_returns_zero(self) -> None:
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert cohens_d(a, b) == 0.0

    def test_single_element_returns_zero(self) -> None:
        assert cohens_d(np.array([1.0]), np.array([2.0])) == 0.0

    def test_negative_when_b_larger(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        assert cohens_d(a, b) < 0.0

    def test_antisymmetry(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, size=50)
        b = rng.normal(1, 1.5, size=50)
        assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a))

    def test_positive_when_a_larger(self) -> None:
        a = np.array([10.0, 11.0, 12.0, 13.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert cohens_d(a, b) > 0.0


# ---------------------------------------------------------------------------
# Hedges' g
# ---------------------------------------------------------------------------
class TestHedgesG:
    def test_approximately_cohens_d_for_large_df(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=200)
        b = rng.normal(0.5, 1, size=200)
        d = cohens_d(a, b)
        g = hedges_g(a, b)
        assert g == pytest.approx(d, abs=0.01)

    def test_smaller_than_cohens_d_for_small_df(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=5)
        b = rng.normal(1, 1, size=5)
        d = cohens_d(a, b)
        g = hedges_g(a, b)
        assert abs(g) < abs(d)

    def test_df_less_than_one_returns_d(self) -> None:
        a = np.array([1.0])
        b = np.array([2.0])
        # n_a=1, n_b=1 → df=0 → should return d (which is 0.0 for single elements)
        assert hedges_g(a, b) == pytest.approx(cohens_d(a, b))

    def test_large_df_no_overflow(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=200)
        b = rng.normal(0.5, 1, size=200)
        g = hedges_g(a, b)
        assert np.isfinite(g)

    def test_same_sign_as_d(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=20)
        b = rng.normal(1, 1, size=20)
        d = cohens_d(a, b)
        g = hedges_g(a, b)
        assert np.sign(d) == np.sign(g)

    def test_identical_groups_returns_zero(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert hedges_g(a, a) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Bonferroni correction
# ---------------------------------------------------------------------------
class TestBonferroniCorrection:
    def test_all_pass(self) -> None:
        p = [0.001, 0.002, 0.003]
        rej, alpha_c = bonferroni_correction(p, alpha=0.05)
        assert all(rej)
        assert alpha_c == pytest.approx(0.05 / 3)

    def test_all_fail(self) -> None:
        p = [0.5, 0.6, 0.7]
        rej, _ = bonferroni_correction(p, alpha=0.05)
        assert not any(rej)

    def test_empty_input(self) -> None:
        rej, alpha_c = bonferroni_correction([], alpha=0.05)
        assert rej == []
        assert alpha_c == 0.05

    def test_single_test_unchanged_alpha(self) -> None:
        rej, alpha_c = bonferroni_correction([0.03], alpha=0.05)
        assert rej == [True]
        assert alpha_c == 0.05

    def test_mixed_rejections(self) -> None:
        p = [0.001, 0.01, 0.04, 0.06]
        rej, alpha_c = bonferroni_correction(p, alpha=0.05)
        corrected = 0.05 / 4  # 0.0125
        expected = [p_i < corrected for p_i in p]
        assert rej == expected


# ---------------------------------------------------------------------------
# Benjamini-Hochberg correction
# ---------------------------------------------------------------------------
class TestBenjaminiHochbergCorrection:
    def test_all_pass(self) -> None:
        p = [0.001, 0.002, 0.003]
        rej, adjusted = benjamini_hochberg_correction(p, alpha=0.05)
        assert all(rej)
        assert all(a >= raw for a, raw in zip(adjusted, p))

    def test_all_fail(self) -> None:
        p = [0.5, 0.6, 0.7]
        rej, _ = benjamini_hochberg_correction(p, alpha=0.05)
        assert not any(rej)

    def test_empty_input(self) -> None:
        rej, adjusted = benjamini_hochberg_correction([], alpha=0.05)
        assert rej == []
        assert adjusted == []

    def test_adjusted_ge_raw(self) -> None:
        p = [0.01, 0.04, 0.03, 0.02]
        _, adjusted = benjamini_hochberg_correction(p, alpha=0.05)
        for adj, raw in zip(adjusted, p):
            assert adj >= raw - 1e-12  # float tolerance

    def test_adjusted_monotonic_when_sorted(self) -> None:
        p = [0.01, 0.04, 0.03, 0.02, 0.06]
        _, adjusted = benjamini_hochberg_correction(p, alpha=0.05)
        sorted_adj = sorted(adjusted)
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-12

    def test_classic_example(self) -> None:
        # p = [0.01, 0.04, 0.03, 0.005], alpha=0.05
        # Sorted: 0.005, 0.01, 0.03, 0.04
        # Thresholds: 0.0125, 0.025, 0.0375, 0.05
        # 0.04 <= 0.05 → reject all 4
        p = [0.01, 0.04, 0.03, 0.005]
        rej, _ = benjamini_hochberg_correction(p, alpha=0.05)
        assert all(rej)

    def test_bh_more_powerful_than_bonferroni(self) -> None:
        p = [0.01, 0.02, 0.03, 0.04, 0.05]
        bh_rej, _ = benjamini_hochberg_correction(p, alpha=0.05)
        bo_rej, _ = bonferroni_correction(p, alpha=0.05)
        assert sum(bh_rej) >= sum(bo_rej)


# ---------------------------------------------------------------------------
# Mann-Whitney U
# ---------------------------------------------------------------------------
class TestMannWhitneyU:
    def test_different_groups_low_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=50)
        b = rng.normal(3, 1, size=50)
        _, p = mann_whitney_u(a, b)
        assert p < 0.001

    def test_identical_groups_high_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=30)
        _, p = mann_whitney_u(a, a)
        assert p > 0.9

    def test_single_element_guard(self) -> None:
        u, p = mann_whitney_u(np.array([1.0]), np.array([2.0]))
        assert u == 0.0
        assert p == 1.0

    def test_reproducible(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=20)
        b = rng.normal(0.5, 1, size=20)
        r1 = mann_whitney_u(a, b)
        r2 = mann_whitney_u(a, b)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------
class TestWelchTTest:
    def test_different_groups_low_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=50)
        b = rng.normal(3, 1, size=50)
        _, p = welch_ttest(a, b)
        assert p < 0.001

    def test_identical_groups_high_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=30)
        _, p = welch_ttest(a, a)
        assert p > 0.9

    def test_single_element_guard(self) -> None:
        t, p = welch_ttest(np.array([1.0]), np.array([2.0]))
        assert t == 0.0
        assert p == 1.0

    def test_unequal_variances_different_df(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=50)
        b = rng.normal(0, 10, size=50)
        t1, p1 = welch_ttest(a, b)
        assert np.isfinite(t1)
        assert np.isfinite(p1)


# ---------------------------------------------------------------------------
# compare_conditions
# ---------------------------------------------------------------------------
class TestCompareConditions:
    def _make_results(
        self, n_samples: int, metric_values: list[float], cond_idx: int = 0
    ) -> list[dict]:
        """Helper to build per-sample result dicts with realistic metadata."""
        results = []
        for i, v in enumerate(metric_values):
            results.append({
                "experiment_id": "exp_test",
                "condition_index": cond_idx,
                "condition_params": {"param": cond_idx},
                "sample_index": i,
                "sample_id": f"sample_{i:04d}",
                "seed": 42,
                "duration_seconds": 0.01,
                "transform_chain": ["temporal.jitter"],
                "metric_a": v,
                "metric_b": v * 0.5,
            })
        return results

    def test_different_conditions_produces_comparisons(self) -> None:
        rng = np.random.default_rng(42)
        a = self._make_results(20, list(rng.normal(10, 1, 20)), cond_idx=0)
        b = self._make_results(20, list(rng.normal(5, 1, 20)), cond_idx=1)
        comps = compare_conditions(a, b, condition_a_index=0, condition_b_index=1)
        metric_comps = [c for c in comps if c.metric_name in ("metric_a", "metric_b")]
        assert len(metric_comps) > 0
        for c in metric_comps:
            assert isinstance(c, ComparisonResult)
            assert c.condition_a_index == 0
            assert c.condition_b_index == 1

    def test_identical_conditions_near_zero_effects(self) -> None:
        rng = np.random.default_rng(42)
        vals = list(rng.normal(10, 1, 20))
        a = self._make_results(20, vals, cond_idx=0)
        b = self._make_results(20, vals, cond_idx=0)
        comps = compare_conditions(a, b)
        # Only check actual metrics, not condition_params
        metric_comps = [c for c in comps if c.metric_name in ("metric_a", "metric_b")]
        for c in metric_comps:
            assert abs(c.mean_diff) < 0.01
            assert abs(c.cohens_d) < 0.01

    def test_empty_input_returns_empty(self) -> None:
        assert compare_conditions([], []) == []
        assert compare_conditions([], [{"a": 1.0}]) == []
        assert compare_conditions([{"a": 1.0}], []) == []

    def test_single_element_groups_skipped(self) -> None:
        a = [{"metric": 1.0}]
        b = [{"metric": 2.0}]
        comps = compare_conditions(a, b)
        assert comps == []

    def test_all_output_fields_present(self) -> None:
        rng = np.random.default_rng(42)
        a = self._make_results(10, list(rng.normal(10, 1, 10)))
        b = self._make_results(10, list(rng.normal(5, 1, 10)))
        comps = compare_conditions(a, b)
        metric_comps = [c for c in comps if c.metric_name == "metric_a"]
        assert len(metric_comps) > 0
        c = metric_comps[0]
        assert hasattr(c, "metric_name")
        assert hasattr(c, "mean_a")
        assert hasattr(c, "mean_b")
        assert hasattr(c, "mean_diff")
        assert hasattr(c, "relative_diff_pct")
        assert hasattr(c, "cohens_d")
        assert hasattr(c, "hedges_g")
        assert hasattr(c, "mann_whitney_u")
        assert hasattr(c, "mann_whitney_p")
        assert hasattr(c, "welch_t")
        assert hasattr(c, "welch_p")
        assert hasattr(c, "ci_lower")
        assert hasattr(c, "ci_upper")
        assert hasattr(c, "n_a")
        assert hasattr(c, "n_b")

    def test_to_dict_is_json_serializable(self) -> None:
        rng = np.random.default_rng(42)
        a = self._make_results(10, list(rng.normal(10, 1, 10)))
        b = self._make_results(10, list(rng.normal(5, 1, 10)))
        comps = compare_conditions(a, b)
        metric_comps = [c for c in comps if c.metric_name == "metric_a"]
        d = metric_comps[0].to_dict()
        import json
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["metric_name"] == comps[0].metric_name

    def test_relative_diff_pct_correct(self) -> None:
        a = [{"metric": 10.0} for _ in range(5)]
        b = [{"metric": 8.0} for _ in range(5)]
        comps = compare_conditions(a, b)
        metric_c = [c for c in comps if c.metric_name == "metric"][0]
        # mean_a=10, mean_b=8, diff=2, relative=2/10*100=20%
        assert metric_c.relative_diff_pct == pytest.approx(20.0, abs=0.1)

    def test_reproducible_with_seed(self) -> None:
        rng = np.random.default_rng(42)
        a = self._make_results(10, list(rng.normal(10, 1, 10)))
        b = self._make_results(10, list(rng.normal(5, 1, 10)))
        r1 = compare_conditions(a, b, seed=42)
        r2 = compare_conditions(a, b, seed=42)
        assert len(r1) == len(r2)
        metric_r1 = sorted([c for c in r1 if c.metric_name in ("metric_a", "metric_b")], key=lambda c: c.metric_name)
        metric_r2 = sorted([c for c in r2 if c.metric_name in ("metric_a", "metric_b")], key=lambda c: c.metric_name)
        for c1, c2 in zip(metric_r1, metric_r2):
            assert c1.mean_diff == pytest.approx(c2.mean_diff)
            assert c1.cohens_d == pytest.approx(c2.cohens_d)
            assert c1.mann_whitney_p == pytest.approx(c2.mann_whitney_p)
            assert c1.ci_lower == pytest.approx(c2.ci_lower)

    def test_significant_pvalues_for_different_groups(self) -> None:
        rng = np.random.default_rng(42)
        a = self._make_results(30, list(rng.normal(10, 1, 30)))
        b = self._make_results(30, list(rng.normal(5, 1, 30)))
        comps = compare_conditions(a, b)
        metric_comps = [c for c in comps if c.metric_name in ("metric_a", "metric_b")]
        for c in metric_comps:
            assert c.mann_whitney_p < 0.05
            assert c.welch_p < 0.05

    def test_ci_width_positive(self) -> None:
        rng = np.random.default_rng(42)
        a = self._make_results(10, list(rng.normal(10, 1, 10)))
        b = self._make_results(10, list(rng.normal(5, 1, 10)))
        comps = compare_conditions(a, b)
        metric_comps = [c for c in comps if c.metric_name in ("metric_a", "metric_b")]
        for c in metric_comps:
            assert c.ci_upper > c.ci_lower

    def test_metadata_keys_excluded_from_comparison(self) -> None:
        """condition_index, sample_index, seed, etc. must not appear as metrics."""
        rng = np.random.default_rng(42)
        a = self._make_results(10, list(rng.normal(10, 1, 10)), cond_idx=0)
        b = self._make_results(10, list(rng.normal(5, 1, 10)), cond_idx=1)
        comps = compare_conditions(a, b)
        metric_names = {c.metric_name for c in comps}
        assert "condition_index" not in metric_names
        assert "sample_index" not in metric_names
        assert "seed" not in metric_names
        assert "duration_seconds" not in metric_names
        assert "experiment_id" not in metric_names
        assert "sample_id" not in metric_names
        assert "transform_chain" not in metric_names
        assert "condition_params" not in metric_names
        assert "metrics" not in metric_names

    def test_only_real_metrics_in_comparison(self) -> None:
        """With realistic metadata, only metric_a and metric_b should compare."""
        rng = np.random.default_rng(42)
        a = self._make_results(10, list(rng.normal(10, 1, 10)))
        b = self._make_results(10, list(rng.normal(5, 1, 10)))
        comps = compare_conditions(a, b)
        metric_names = {c.metric_name for c in comps}
        assert metric_names == {"metric_a", "metric_b"}


# ---------------------------------------------------------------------------
# summarize_condition (regression + method parameter)
# ---------------------------------------------------------------------------
class TestSummarizeCondition:
    def test_default_method_is_percentile(self) -> None:
        """Default method='percentile' preserves backward compatibility."""
        results = [
            {"condition_index": 0, "condition_params": {}, "snr_db": float(i)}
            for i in range(8)
        ]
        summary = summarize_condition(results)
        assert summary.condition_index == 0
        assert summary.n_samples == 8
        assert "snr_db" in summary.metric_summaries

    def test_explicit_percentile(self) -> None:
        results = [
            {"condition_index": 0, "condition_params": {}, "rmse": 0.1 * i}
            for i in range(8)
        ]
        summary = summarize_condition(results, method="percentile")
        assert "rmse" in summary.metric_summaries
        s = summary.metric_summaries["rmse"]
        assert s["mean"] == pytest.approx(0.35, abs=0.01)
        assert s["ci_lower"] <= s["mean"] <= s["ci_upper"]

    def test_explicit_bca(self) -> None:
        results = [
            {"condition_index": 0, "condition_params": {}, "rmse": 0.1 * i}
            for i in range(8)
        ]
        summary = summarize_condition(results, method="bca")
        assert "rmse" in summary.metric_summaries
        s = summary.metric_summaries["rmse"]
        assert s["mean"] == pytest.approx(0.35, abs=0.01)
        assert s["ci_lower"] <= s["mean"] <= s["ci_upper"]
        assert np.isfinite(s["ci_lower"])
        assert np.isfinite(s["ci_upper"])

    def test_bca_and_percentile_produce_similar_means(self) -> None:
        rng = np.random.default_rng(42)
        vals = list(rng.normal(5.0, 2.0, 20))
        results = [
            {"condition_index": 0, "condition_params": {}, "metric": v}
            for v in vals
        ]
        pct = summarize_condition(results, method="percentile")
        bca = summarize_condition(results, method="bca")
        pct_mean = pct.metric_summaries["metric"]["mean"]
        bca_mean = bca.metric_summaries["metric"]["mean"]
        assert pct_mean == pytest.approx(bca_mean)

    def test_empty_results(self) -> None:
        summary = summarize_condition([])
        assert summary.condition_index == -1
        assert summary.n_samples == 0

    def test_non_numeric_keys_ignored(self) -> None:
        results = [
            {"condition_index": 0, "condition_params": {"a": 1}, "metric": 1.5},
            {"condition_index": 0, "condition_params": {"a": 1}, "metric": 2.5},
        ]
        summary = summarize_condition(results)
        assert "metric" in summary.metric_summaries
        # condition_params is a dict, not numeric — should be ignored
        assert "condition_params" not in summary.metric_summaries

    def test_to_dict_serializable(self) -> None:
        results = [
            {"condition_index": 0, "condition_params": {"amp": 10}, "rmse": 0.5},
            {"condition_index": 0, "condition_params": {"amp": 10}, "rmse": 0.6},
        ]
        summary = summarize_condition(results)
        d = summary.to_dict()
        import json
        json.dumps(d)  # should not raise

    def test_metadata_keys_excluded_from_summary(self) -> None:
        """condition_index, sample_index, seed, etc. must not appear as metrics."""
        results = [
            {
                "condition_index": 0,
                "condition_params": {"amp": 10},
                "sample_index": i,
                "sample_id": f"sample_{i:04d}",
                "seed": 42,
                "duration_seconds": 0.01,
                "transform_chain": ["temporal.jitter"],
                "rmse": 0.1 * i,
                "snr_db": float(i),
            }
            for i in range(8)
        ]
        summary = summarize_condition(results)
        assert "condition_index" not in summary.metric_summaries
        assert "sample_index" not in summary.metric_summaries
        assert "seed" not in summary.metric_summaries
        assert "duration_seconds" not in summary.metric_summaries
        assert "experiment_id" not in summary.metric_summaries
        assert "sample_id" not in summary.metric_summaries
        assert "transform_chain" not in summary.metric_summaries
        # Real metrics should still be present
        assert "rmse" in summary.metric_summaries
        assert "snr_db" in summary.metric_summaries


# ---------------------------------------------------------------------------
# Permutation Test, Wilcoxon & Pareto Frontier Tests
# ---------------------------------------------------------------------------
class TestAdvancedStats:
    def test_paired_permutation_test_identical(self) -> None:
        from audiocaptcha_dsp.evaluation.stats import paired_permutation_test
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        diff, p = paired_permutation_test(x, x)
        assert diff == 0.0
        assert p >= 0.9

    def test_paired_permutation_test_different(self) -> None:
        from audiocaptcha_dsp.evaluation.stats import paired_permutation_test
        x = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        diff, p = paired_permutation_test(x, y)
        assert diff == pytest.approx(9.0)
        assert p < 0.01

    def test_wilcoxon_signed_rank_identical(self) -> None:
        from audiocaptcha_dsp.evaluation.stats import wilcoxon_signed_rank
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stat, p = wilcoxon_signed_rank(x, x)
        assert p == 1.0

    def test_wilcoxon_signed_rank_different(self) -> None:
        from audiocaptcha_dsp.evaluation.stats import wilcoxon_signed_rank
        x = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        stat, p = wilcoxon_signed_rank(x, y)
        assert p < 0.01

    def test_compute_pareto_frontier(self) -> None:
        from audiocaptcha_dsp.evaluation.stats import compute_pareto_frontier
        candidates = [
            {"id": "A", "hsr": 0.9, "wer": 0.8, "cost": 0.1},  # high hsr, high wer, low cost -> Pareto
            {"id": "B", "hsr": 0.5, "wer": 0.4, "cost": 0.5},  # strictly dominated by A
            {"id": "C", "hsr": 0.95, "wer": 0.7, "cost": 0.2}, # highest hsr -> Pareto
            {"id": "D", "hsr": 0.8, "wer": 0.95, "cost": 0.3}, # highest wer -> Pareto
        ]
        objectives = {"hsr": "max", "wer": "max", "cost": "min"}
        results = compute_pareto_frontier(candidates, objectives)
        
        pareto_ids = {r["id"] for r in results if r["is_pareto"]}
        assert "A" in pareto_ids
        assert "C" in pareto_ids
        assert "D" in pareto_ids
        assert "B" not in pareto_ids  # B is dominated

