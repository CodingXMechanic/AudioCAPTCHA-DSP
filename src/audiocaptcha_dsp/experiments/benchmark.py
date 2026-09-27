from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from audiocaptcha_dsp.evaluation.stats import (
    ComparisonResult,
    bootstrap_ci_bca,
    cohens_d,
    compare_conditions,
    hedges_g,
    mann_whitney_u,
    welch_ttest,
    _discover_metrics,
)

logger = logging.getLogger(__name__)


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a result dict so metrics are at the top level.

    If the result contains a nested "metrics" dict (as produced by
    ExperimentResult.to_dict()), the metric key-value pairs are merged
    into the top-level dict. The nested "metrics" key itself is removed
    so it is not treated as a metric by _discover_metrics().
    """
    normalized = dict(result)
    nested = normalized.pop("metrics", None)
    if nested and isinstance(nested, dict):
        # Merge nested metrics into top level (nested values don't override
        # top-level keys — top-level metadata takes precedence).
        for k, v in nested.items():
            if k not in normalized:
                normalized[k] = v
    return normalized


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class ExperimentManifest:
    """Loaded and parsed experiment manifest with convenience accessors."""

    experiment_id: str
    manifest_version: str
    timestamp: float
    raw: dict[str, Any]
    results: list[dict[str, Any]]
    per_condition: list[dict[str, Any]]

    @classmethod
    def load(cls, path: Path | str) -> ExperimentManifest:
        """Load a manifest JSON file.

        Normalizes result dicts so that metric values are always at the top
        level. Handles both the nested format (ExperimentResult.to_dict() with
        a "metrics" sub-dict) and the flat format (runner's condition_metrics).
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        results = raw.get("results", [])
        results = [_normalize_result(r) for r in results]
        return cls(
            experiment_id=raw.get("experiment_id", path.stem),
            manifest_version=raw.get("manifest_version", "unknown"),
            timestamp=raw.get("timestamp", 0.0),
            raw=raw,
            results=results,
            per_condition=raw.get("summary", {}).get("per_condition_summaries", []),
        )

    @property
    def total_conditions(self) -> int:
        return len(self.per_condition)

    @property
    def total_samples(self) -> int:
        return len(self.results)

    def results_for_condition(self, condition_index: int) -> list[dict[str, Any]]:
        """Return per-sample result dicts for a given condition."""
        return [r for r in self.results if r.get("condition_index") == condition_index]

    def metric_names(self) -> list[str]:
        """Return the list of metric names (shared across all results)."""
        if not self.results:
            return []
        return _discover_metrics(self.results[0])

    def condition_params(self, condition_index: int) -> dict:
        """Return the parameter dict for a given condition."""
        results = self.results_for_condition(condition_index)
        if results:
            return results[0].get("condition_params", {})
        if condition_index < len(self.per_condition):
            return self.per_condition[condition_index].get("condition_params", {})
        return {}


# ---------------------------------------------------------------------------
# Cross-experiment comparison
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkComparison:
    """Result of benchmarking two experiments against each other."""

    experiment_a_id: str
    experiment_b_id: str
    experiment_a_path: Path
    experiment_b_path: Path
    timestamp: float
    common_metrics: list[str]
    condition_comparisons: dict[int, list[ComparisonResult]]
    best_conditions: dict[str, dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "experiment_a_id": self.experiment_a_id,
            "experiment_b_id": self.experiment_b_id,
            "experiment_a_path": str(self.experiment_a_path),
            "experiment_b_path": str(self.experiment_b_path),
            "timestamp": self.timestamp,
            "common_metrics": self.common_metrics,
            "condition_comparisons": {
                str(k): [c.to_dict() for c in v]
                for k, v in self.condition_comparisons.items()
            },
            "best_conditions": self.best_conditions,
            "summary": self.summary,
        }


@dataclass
class BenchmarkSummary:
    """Aggregated summary across all conditions of two experiments."""

    experiment_a_id: str
    experiment_b_id: str
    n_conditions_a: int
    n_conditions_b: int
    common_metrics: list[str]
    overall: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "experiment_a_id": self.experiment_a_id,
            "experiment_b_id": self.experiment_b_id,
            "n_conditions_a": self.n_conditions_a,
            "n_conditions_b": self.n_conditions_b,
            "common_metrics": self.common_metrics,
            "overall": self.overall,
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Compare two experiments across all shared conditions and metrics."""

    def __init__(
        self,
        manifest_a: Path | str | ExperimentManifest,
        manifest_b: Path | str | ExperimentManifest,
    ):
        if isinstance(manifest_a, ExperimentManifest):
            self.manifest_a = manifest_a
        else:
            self.manifest_a = ExperimentManifest.load(manifest_a)

        if isinstance(manifest_b, ExperimentManifest):
            self.manifest_b = manifest_b
        else:
            self.manifest_b = ExperimentManifest.load(manifest_b)

        logger.info(
            "Benchmark: %s (%d conditions, %d samples) vs %s (%d conditions, %d samples)",
            self.manifest_a.experiment_id,
            self.manifest_a.total_conditions,
            self.manifest_a.total_samples,
            self.manifest_b.experiment_id,
            self.manifest_b.total_conditions,
            self.manifest_b.total_samples,
        )

    def compare(
        self,
        ci: float = 0.95,
        seed: int = 42,
    ) -> BenchmarkComparison:
        """Run the full comparison between the two experiments.

        Compares each condition in experiment A against the corresponding
        condition in experiment B (matched by condition_index). For
        conditions that don't match, all-vs-all pairwise comparisons are
        computed.
        """
        common_metrics = sorted(
            set(self.manifest_a.metric_names()) & set(self.manifest_b.metric_names())
        )
        logger.info("Common metrics: %s", common_metrics)

        condition_comparisons: dict[int, list[ComparisonResult]] = {}

        # Match conditions by index (paired comparison)
        cond_indices_a = {
            r.get("condition_index", -1) for r in self.manifest_a.results
        }
        cond_indices_b = {
            r.get("condition_index", -1) for r in self.manifest_b.results
        }
        shared_indices = sorted(cond_indices_a & cond_indices_b)

        if shared_indices:
            for idx in shared_indices:
                results_a = self.manifest_a.results_for_condition(idx)
                results_b = self.manifest_b.results_for_condition(idx)
                if results_a and results_b:
                    comps = compare_conditions(
                        results_a,
                        results_b,
                        condition_a_index=idx,
                        condition_b_index=idx,
                        ci=ci,
                        seed=seed,
                    )
                    condition_comparisons[idx] = comps
        else:
            # No shared indices: compare aggregate (all A vs all B)
            logger.warning(
                "No shared condition indices. Comparing all A vs all B."
            )
            comps = compare_conditions(
                self.manifest_a.results,
                self.manifest_b.results,
                condition_a_index=0,
                condition_b_index=0,
                ci=ci,
                seed=seed,
            )
            condition_comparisons[0] = comps

        # Determine best conditions per metric
        best_conditions = self._find_best_conditions(condition_comparisons, common_metrics)

        # Build overall summary
        overall = self._compute_overall(condition_comparisons, common_metrics)

        summary = {
            "n_compared_conditions": len(condition_comparisons),
            "n_common_metrics": len(common_metrics),
            "paired_comparison": bool(shared_indices),
            "overall": overall,
        }

        return BenchmarkComparison(
            experiment_a_id=self.manifest_a.experiment_id,
            experiment_b_id=self.manifest_b.experiment_id,
            experiment_a_path=Path(self.manifest_a.raw.get("_path", "")),
            experiment_b_path=Path(self.manifest_b.raw.get("_path", "")),
            timestamp=time.time(),
            common_metrics=common_metrics,
            condition_comparisons=condition_comparisons,
            best_conditions=best_conditions,
            summary=summary,
        )

    def summary(self) -> BenchmarkSummary:
        """Compute an aggregated summary across all conditions."""
        comp = self.compare()
        return BenchmarkSummary(
            experiment_a_id=comp.experiment_a_id,
            experiment_b_id=comp.experiment_b_id,
            n_conditions_a=self.manifest_a.total_conditions,
            n_conditions_b=self.manifest_b.total_conditions,
            common_metrics=comp.common_metrics,
            overall=comp.summary["overall"],
        )

    def _find_best_conditions(
        self,
        condition_comparisons: dict[int, list[ComparisonResult]],
        metrics: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Find the best condition per metric (highest mean_a)."""
        best: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            best_mean = -np.inf
            best_idx = -1
            for idx, comps in condition_comparisons.items():
                for c in comps:
                    if c.metric_name == metric and c.mean_a > best_mean:
                        best_mean = c.mean_a
                        best_idx = idx
            if best_idx >= 0:
                best[metric] = {
                    "best_condition_index": best_idx,
                    "best_mean": best_mean,
                    "params": self.manifest_a.condition_params(best_idx),
                }
        return best

    def _compute_overall(
        self,
        condition_comparisons: dict[int, list[ComparisonResult]],
        metrics: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute overall aggregate statistics across all conditions."""
        overall: dict[str, dict[str, float]] = {}
        for metric in metrics:
            diffs: list[float] = []
            effect_sizes: list[float] = []
            for comps in condition_comparisons.values():
                for c in comps:
                    if c.metric_name == metric:
                        diffs.append(c.mean_diff)
                        effect_sizes.append(c.cohens_d)
            if diffs:
                diff_arr = np.array(diffs)
                es_arr = np.array(effect_sizes)
                overall[metric] = {
                    "mean_diff": float(np.mean(diff_arr)),
                    "std_diff": float(np.std(diff_arr)),
                    "min_diff": float(np.min(diff_arr)),
                    "max_diff": float(np.max(diff_arr)),
                    "mean_cohens_d": float(np.mean(es_arr)),
                    "std_cohens_d": float(np.std(es_arr)),
                    "n_comparisons": len(diffs),
                }
        return overall


# ---------------------------------------------------------------------------
# Multi-experiment benchmark
# ---------------------------------------------------------------------------

@dataclass
class MultiBenchmarkResult:
    """Result of benchmarking multiple experiments."""

    experiments: list[str]
    timestamp: float
    pairwise: dict[str, BenchmarkComparison]
    ranking: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict:
        return {
            "experiments": self.experiments,
            "timestamp": self.timestamp,
            "pairwise": {k: v.to_dict() for k, v in self.pairwise.items()},
            "ranking": self.ranking,
        }


class MultiBenchmark:
    """Compare multiple experiments pairwise."""

    def __init__(self, manifests: list[Path | str]):
        self.manifests: list[ExperimentManifest] = []
        for m in manifests:
            manifest = ExperimentManifest.load(m)
            # Store path for reference
            manifest.raw["_path"] = str(m)
            self.manifests.append(manifest)
        logger.info("MultiBenchmark: loaded %d experiments", len(self.manifests))

    def compare_all(self, ci: float = 0.95, seed: int = 42) -> MultiBenchmarkResult:
        """Run all pairwise comparisons."""
        pairwise: dict[str, BenchmarkComparison] = {}
        for i in range(len(self.manifests)):
            for j in range(i + 1, len(self.manifests)):
                key = f"{self.manifests[i].experiment_id}_vs_{self.manifests[j].experiment_id}"
                runner = BenchmarkRunner(self.manifests[i], self.manifests[j])
                pairwise[key] = runner.compare(ci=ci, seed=seed)

        ranking = self._rank_experiments(pairwise)

        return MultiBenchmarkResult(
            experiments=[m.experiment_id for m in self.manifests],
            timestamp=time.time(),
            pairwise=pairwise,
            ranking=ranking,
        )

    def _rank_experiments(
        self, pairwise: dict[str, BenchmarkComparison]
    ) -> dict[str, list[dict[str, Any]]]:
        """Rank experiments by mean performance per metric."""
        # Collect all metric means per experiment
        exp_metrics: dict[str, dict[str, list[float]]] = {}
        for comp in pairwise.values():
            for idx, comps in comp.condition_comparisons.items():
                for c in comps:
                    if c.metric_name not in exp_metrics:
                        exp_metrics[c.metric_name] = {}
                    if comp.experiment_a_id not in exp_metrics[c.metric_name]:
                        exp_metrics[c.metric_name][comp.experiment_a_id] = []
                    if comp.experiment_b_id not in exp_metrics[c.metric_name]:
                        exp_metrics[c.metric_name][comp.experiment_b_id] = []
                    exp_metrics[c.metric_name][comp.experiment_a_id].append(c.mean_a)
                    exp_metrics[c.metric_name][comp.experiment_b_id].append(c.mean_b)

        ranking: dict[str, list[dict[str, Any]]] = {}
        for metric, exp_vals in exp_metrics.items():
            ranked = sorted(
                [
                    {
                        "experiment_id": exp_id,
                        "mean_value": float(np.mean(vals)),
                        "n_conditions": len(vals),
                    }
                    for exp_id, vals in exp_vals.items()
                ],
                key=lambda x: x["mean_value"],
                reverse=True,
            )
            ranking[metric] = ranked

        return ranking
