from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
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


def summarize_condition(results: list[dict[str, Any]], ci: float = 0.95, seed: int = 42) -> ConditionSummary:
    if not results:
        return ConditionSummary(condition_index=-1, condition_params={}, n_samples=0)
    numeric_keys: list[str] = []
    for key in results[0]:
        val = results[0][key]
        if isinstance(val, (int, float, np.integer, np.floating)):
            numeric_keys.append(key)
    summary: dict[str, dict[str, float]] = {}
    for metric_name in numeric_keys:
        raw = [r[metric_name] for r in results if metric_name in r]
        values = np.array([v for v in raw if np.isfinite(v)], dtype=np.float64)
        if len(values) == 0:
            continue
        point, lower, upper = bootstrap_ci(values, n_bootstrap=min(1000, max(100, len(values) * 10)), ci=ci, seed=seed)
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
