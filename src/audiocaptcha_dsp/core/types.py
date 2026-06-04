from __future__ import annotations

import itertools
from typing import Any, Protocol
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class BarkScaleMethod(str, Enum):
    TRAUNMULLER = "traunmuller"
    ZWICKER = "zwicker"
    WANG = "wang"


class MaskingModel(str, Enum):
    SIMULTANEOUS = "simultaneous"
    TEMPORAL = "temporal"
    COMBINED = "combined"


class ThresholdModel(str, Enum):
    ISO226 = "iso226"
    ISO389 = "iso389"
    CUSTOM = "custom"


@dataclass
class TranscriptionResult:
    text: str
    confidence: float | None = None
    segments: list[dict] = field(default_factory=list)
    language: str = "en"


class Transform(Protocol):
    def __call__(self, signal: "Signal") -> "Signal": ...


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    name: str
    description: str
    dataset_name: str
    transform_specs: list["TransformSpec"]
    num_samples: int
    seed: int
    output_dir: Path
    parameters: dict[str, Any] = field(default_factory=dict)
    composite_conditions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_composite(self) -> bool:
        return len(self.transform_specs) > 1

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        import yaml
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        exp = raw.get("experiment", {})
        transform_specs = []
        parameters: dict[str, Any] = {}
        composite_conditions: list[dict[str, Any]] = []

        if "transform" in exp and exp["transform"]:
            transform_specs.append(TransformSpec.from_dict(exp["transform"], exp.get("parameters", {})))
            parameters = exp.get("parameters", {})
        elif "transforms" in exp and exp["transforms"]:
            for tdict in exp["transforms"]:
                transform_specs.append(TransformSpec.from_dict(tdict.get("type", ""), tdict.get("parameters", {})))
            composite_conditions = cls._expand_composite_conditions(exp["transforms"])
            if not composite_conditions:
                composite_conditions = [{}]

        return cls(
            experiment_id=exp.get("id", path.stem),
            name=exp.get("name", path.stem),
            description=exp.get("description", ""),
            dataset_name=exp.get("dataset", "synthetic"),
            transform_specs=transform_specs,
            num_samples=exp.get("num_samples", 10),
            seed=exp.get("seed", 42),
            output_dir=Path(exp.get("output_dir", "results")),
            parameters=parameters,
            composite_conditions=composite_conditions,
        )

    @staticmethod
    def _expand_composite_conditions(transforms: list[dict]) -> list[dict[str, Any]]:
        per_transform_param_lists: list[list[dict[str, Any]]] = []
        for t in transforms:
            params = t.get("parameters", {})
            if not params:
                per_transform_param_lists.append([{}])
                continue
            keys = sorted(params.keys())
            values_lists = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]
            combos = list(itertools.product(*values_lists))
            per_transform_param_lists.append([dict(zip(keys, combo)) for combo in combos])

        cross = list(itertools.product(*per_transform_param_lists))
        result: list[dict[str, Any]] = []
        for combo in cross:
            merged: dict[str, Any] = {}
            for d in combo:
                merged.update(d)
            result.append(merged)
        return result


@dataclass(frozen=True)
class TransformSpec:
    transform_type: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, transform_type: str, params: dict[str, Any] | None) -> TransformSpec:
        return cls(transform_type=transform_type, parameters=params or {})


@dataclass
class ExperimentResult:
    experiment_id: str
    condition_index: int
    condition_params: dict[str, Any]
    sample_index: int
    sample_id: str
    metrics: dict[str, float]
    transform_chain: list[str]
    duration_seconds: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "condition_index": self.condition_index,
            "condition_params": self.condition_params,
            "sample_index": self.sample_index,
            "sample_id": self.sample_id,
            "metrics": self.metrics,
            "transform_chain": self.transform_chain,
            "duration_seconds": self.duration_seconds,
            "seed": self.seed,
        }


@dataclass
class BenchmarkResult:
    experiment_id: str
    total_conditions: int
    total_samples: int
    total_duration_seconds: float
    per_condition: list[dict[str, Any]]
    results: list[ExperimentResult]

    def summary(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "total_conditions": self.total_conditions,
            "total_samples": self.total_samples,
            "total_duration_seconds": self.total_duration_seconds,
            "per_condition_summaries": self.per_condition,
        }
