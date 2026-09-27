from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from audiocaptcha_dsp.evaluation.stats import ComparisonResult
from audiocaptcha_dsp.experiments.benchmark import (
    BenchmarkComparison,
    BenchmarkRunner,
    BenchmarkSummary,
    ExperimentManifest,
    MultiBenchmark,
    MultiBenchmarkResult,
)


# ---------------------------------------------------------------------------
# Fixtures: generate synthetic manifest data
# ---------------------------------------------------------------------------


def _make_manifest_dict(
    experiment_id: str,
    n_conditions: int = 3,
    n_samples: int = 5,
    seed: int = 42,
    metric_offset: float = 0.0,
) -> dict:
    """Generate a synthetic manifest dict with known structure."""
    rng = np.random.default_rng(seed)
    results = []
    per_condition = []

    for cond_idx in range(n_conditions):
        cond_results = []
        metric_accum: dict[str, list[float]] = {}

        for sample_idx in range(n_samples):
            metrics = {
                "snr_db": float(rng.normal(15.0 + metric_offset, 2.0)),
                "rmse": float(rng.uniform(0.01, 0.05 + metric_offset * 0.01)),
                "mse": float(rng.uniform(0.0001, 0.002 + metric_offset * 0.0001)),
            }
            for k, v in metrics.items():
                metric_accum.setdefault(k, []).append(v)

            results.append({
                "experiment_id": experiment_id,
                "condition_index": cond_idx,
                "condition_params": {"amplitude_ms": cond_idx * 10},
                "sample_index": sample_idx,
                "sample_id": f"sample_{sample_idx:04d}",
                "metrics": metrics,
                "transform_chain": ["temporal.jitter"],
                "duration_seconds": float(rng.uniform(0.01, 0.05)),
                "seed": seed,
            })
            cond_results.append(metrics)

        # Build condition summary
        metric_summaries = {}
        for metric_name, vals in metric_accum.items():
            arr = np.array(vals)
            metric_summaries[metric_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "ci_lower": float(np.mean(arr) - 0.5),
                "ci_upper": float(np.mean(arr) + 0.5),
                "n": len(arr),
            }

        per_condition.append({
            "condition_index": cond_idx,
            "condition_params": {"amplitude_ms": cond_idx * 10},
            "n_samples": n_samples,
            "metric_summaries": metric_summaries,
        })

    return {
        "manifest_version": "1.0",
        "experiment_id": experiment_id,
        "timestamp": 1700000000.0,
        "summary": {
            "experiment_id": experiment_id,
            "total_conditions": n_conditions,
            "total_samples": n_conditions * n_samples,
            "total_duration_seconds": 1.0,
            "per_condition_summaries": per_condition,
        },
        "results": results,
    }


@pytest.fixture
def manifest_dir(tmp_path):
    """Create two manifest files and return their paths."""
    manifest_a = _make_manifest_dict("exp_A", n_conditions=3, n_samples=5, seed=42)
    manifest_b = _make_manifest_dict("exp_B", n_conditions=3, n_samples=5, seed=99, metric_offset=5.0)

    path_a = tmp_path / "exp_A_manifest.json"
    path_b = tmp_path / "exp_B_manifest.json"

    path_a.write_text(json.dumps(manifest_a), encoding="utf-8")
    path_b.write_text(json.dumps(manifest_b), encoding="utf-8")

    return path_a, path_b


@pytest.fixture
def manifest_paths_three(tmp_path):
    """Create three manifest files for multi-benchmark."""
    paths = []
    for i, seed in enumerate([10, 20, 30]):
        m = _make_manifest_dict(f"exp_{i}", n_conditions=2, n_samples=4, seed=seed, metric_offset=float(i))
        p = tmp_path / f"exp_{i}_manifest.json"
        p.write_text(json.dumps(m), encoding="utf-8")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# ExperimentManifest
# ---------------------------------------------------------------------------
class TestExperimentManifest:
    def test_load_from_path(self, manifest_dir) -> None:
        path_a, _ = manifest_dir
        m = ExperimentManifest.load(path_a)
        assert m.experiment_id == "exp_A"
        assert m.manifest_version == "1.0"
        assert m.total_conditions == 3
        assert m.total_samples == 15

    def test_results_for_condition(self, manifest_dir) -> None:
        path_a, _ = manifest_dir
        m = ExperimentManifest.load(path_a)
        cond0 = m.results_for_condition(0)
        assert len(cond0) == 5
        assert all(r["condition_index"] == 0 for r in cond0)

    def test_metric_names(self, manifest_dir) -> None:
        path_a, _ = manifest_dir
        m = ExperimentManifest.load(path_a)
        names = m.metric_names()
        assert "snr_db" in names
        assert "rmse" in names
        assert "mse" in names
        # Metadata should be excluded
        assert "condition_index" not in names
        assert "sample_index" not in names
        assert "seed" not in names
        assert "duration_seconds" not in names

    def test_condition_params(self, manifest_dir) -> None:
        path_a, _ = manifest_dir
        m = ExperimentManifest.load(path_a)
        params = m.condition_params(0)
        assert params == {"amplitude_ms": 0}

    def test_empty_results(self, tmp_path) -> None:
        m = {
            "manifest_version": "1.0",
            "experiment_id": "empty",
            "timestamp": 0.0,
            "summary": {"per_condition_summaries": []},
            "results": [],
        }
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(m))
        loaded = ExperimentManifest.load(p)
        assert loaded.total_samples == 0
        assert loaded.metric_names() == []


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------
class TestBenchmarkRunner:
    def test_compare_returns_benchmark_comparison(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        runner = BenchmarkRunner(path_a, path_b)
        result = runner.compare()
        assert isinstance(result, BenchmarkComparison)

    def test_experiment_ids(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        runner = BenchmarkRunner(path_a, path_b)
        result = runner.compare()
        assert result.experiment_a_id == "exp_A"
        assert result.experiment_b_id == "exp_B"

    def test_common_metrics(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        runner = BenchmarkRunner(path_a, path_b)
        result = runner.compare()
        assert "snr_db" in result.common_metrics
        assert "rmse" in result.common_metrics
        assert "mse" in result.common_metrics

    def test_condition_comparisons_populated(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        runner = BenchmarkRunner(path_a, path_b)
        result = runner.compare()
        # 3 shared conditions → 3 comparison sets
        assert len(result.condition_comparisons) == 3
        for idx in range(3):
            assert idx in result.condition_comparisons
            assert len(result.condition_comparisons[idx]) > 0

    def test_comparison_results_are_typed(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        runner = BenchmarkRunner(path_a, path_b)
        result = runner.compare()
        for comps in result.condition_comparisons.values():
            for c in comps:
                assert isinstance(c, ComparisonResult)
                assert isinstance(c.mean_a, float)
                assert isinstance(c.mean_b, float)
                assert isinstance(c.cohens_d, float)
                assert isinstance(c.mann_whitney_p, float)

    def test_best_conditions_present(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        runner = BenchmarkRunner(path_a, path_b)
        result = runner.compare()
        assert "snr_db" in result.best_conditions
        for metric, info in result.best_conditions.items():
            assert "best_condition_index" in info
            assert "best_mean" in info
            assert "params" in info

    def test_overall_summary(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        runner = BenchmarkRunner(path_a, path_b)
        result = runner.compare()
        overall = result.summary["overall"]
        assert "snr_db" in overall
        assert "mean_diff" in overall["snr_db"]
        assert "mean_cohens_d" in overall["snr_db"]
        assert "n_comparisons" in overall["snr_db"]

    def test_to_dict_serializable(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        runner = BenchmarkRunner(path_a, path_b)
        result = runner.compare()
        d = result.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["experiment_a_id"] == "exp_A"

    def test_summary_returns_benchmark_summary(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        runner = BenchmarkRunner(path_a, path_b)
        s = runner.summary()
        assert isinstance(s, BenchmarkSummary)
        assert s.experiment_a_id == "exp_A"
        assert s.n_conditions_a == 3
        assert s.n_conditions_b == 3

    def test_reproducible_with_seed(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        r1 = BenchmarkRunner(path_a, path_b).compare(seed=42)
        r2 = BenchmarkRunner(path_a, path_b).compare(seed=42)
        for idx in r1.condition_comparisons:
            for c1, c2 in zip(r1.condition_comparisons[idx], r2.condition_comparisons[idx]):
                assert c1.mean_diff == pytest.approx(c2.mean_diff)
                assert c1.cohens_d == pytest.approx(c2.cohens_d)

    def test_different_metric_offsets_produce_different_results(self, manifest_dir) -> None:
        """exp_B has metric_offset=5.0, so SNR should be systematically higher."""
        path_a, path_b = manifest_dir
        runner = BenchmarkRunner(path_a, path_b)
        result = runner.compare()
        snr_overall = result.summary["overall"].get("snr_db", {})
        # exp_B has higher SNR → mean_diff (A - B) should be negative
        if snr_overall:
            assert snr_overall["mean_diff"] < 0.0

    def test_accepts_manifest_objects(self, manifest_dir) -> None:
        path_a, path_b = manifest_dir
        ma = ExperimentManifest.load(path_a)
        mb = ExperimentManifest.load(path_b)
        runner = BenchmarkRunner(ma, mb)
        result = runner.compare()
        assert result.experiment_a_id == "exp_A"


# ---------------------------------------------------------------------------
# MultiBenchmark
# ---------------------------------------------------------------------------
class TestMultiBenchmark:
    def test_compare_all_returns_result(self, manifest_paths_three) -> None:
        mb = MultiBenchmark(manifest_paths_three)
        result = mb.compare_all()
        assert isinstance(result, MultiBenchmarkResult)

    def test_pairwise_keys(self, manifest_paths_three) -> None:
        mb = MultiBenchmark(manifest_paths_three)
        result = mb.compare_all()
        # 3 experiments → 3 pairwise: 0vs1, 0vs2, 1vs2
        assert len(result.pairwise) == 3
        keys = set(result.pairwise.keys())
        assert "exp_0_vs_exp_1" in keys
        assert "exp_0_vs_exp_2" in keys
        assert "exp_1_vs_exp_2" in keys

    def test_pairwise_values_are_comparisons(self, manifest_paths_three) -> None:
        mb = MultiBenchmark(manifest_paths_three)
        result = mb.compare_all()
        for key, comp in result.pairwise.items():
            assert isinstance(comp, BenchmarkComparison)

    def test_ranking_present(self, manifest_paths_three) -> None:
        mb = MultiBenchmark(manifest_paths_three)
        result = mb.compare_all()
        assert "snr_db" in result.ranking
        for metric, ranked in result.ranking.items():
            assert len(ranked) == 3  # 3 experiments
            assert all("experiment_id" in r for r in ranked)
            assert all("mean_value" in r for r in ranked)

    def test_ranking_ordered_descending(self, manifest_paths_three) -> None:
        mb = MultiBenchmark(manifest_paths_three)
        result = mb.compare_all()
        for metric, ranked in result.ranking.items():
            for i in range(len(ranked) - 1):
                assert ranked[i]["mean_value"] >= ranked[i + 1]["mean_value"]

    def test_to_dict_serializable(self, manifest_paths_three) -> None:
        mb = MultiBenchmark(manifest_paths_three)
        result = mb.compare_all()
        d = result.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_experiments_list(self, manifest_paths_three) -> None:
        mb = MultiBenchmark(manifest_paths_three)
        result = mb.compare_all()
        assert result.experiments == ["exp_0", "exp_1", "exp_2"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestBenchmarkEdgeCases:
    def test_no_shared_conditions_compares_aggregate(self, tmp_path) -> None:
        """When condition indices don't overlap, falls back to all-vs-all."""
        ma = _make_manifest_dict("exp_A", n_conditions=2, n_samples=3, seed=10)
        # Shift condition indices to 10, 11 so they don't overlap with 0, 1
        for r in ma["results"]:
            r["condition_index"] += 10
        for pc in ma["summary"]["per_condition_summaries"]:
            pc["condition_index"] += 10

        mb = _make_manifest_dict("exp_B", n_conditions=2, n_samples=3, seed=20)

        pa = tmp_path / "a.json"
        pb = tmp_path / "b.json"
        pa.write_text(json.dumps(ma))
        pb.write_text(json.dumps(mb))

        runner = BenchmarkRunner(pa, pb)
        result = runner.compare()
        # Should still produce a comparison (aggregate)
        assert len(result.condition_comparisons) > 0

    def test_empty_manifest(self, tmp_path) -> None:
        """Empty results list should not crash."""
        ma = _make_manifest_dict("exp_A", n_conditions=0, n_samples=0)
        mb = _make_manifest_dict("exp_B", n_conditions=1, n_samples=3, seed=20)

        pa = tmp_path / "a.json"
        pb = tmp_path / "b.json"
        pa.write_text(json.dumps(ma))
        pb.write_text(json.dumps(mb))

        runner = BenchmarkRunner(pa, pb)
        result = runner.compare()
        assert isinstance(result, BenchmarkComparison)
