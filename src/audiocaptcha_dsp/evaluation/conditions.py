from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConditions:
    parameters: dict[str, Any] = field(default_factory=dict)
    _grid: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def build_grid(self) -> list[dict[str, Any]]:
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        self._grid = [dict(zip(keys, combo)) for combo in product(*values)]
        return self._grid

    @classmethod
    def from_config(cls, config_path: Path | str) -> ExperimentConditions:
        config_path = Path(config_path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        exp = config.get("experiment", {})
        params = exp.get("parameters", {})
        condition_sets = {k: v if isinstance(v, list) else [v] for k, v in params.items()}
        return cls(parameters=condition_sets)
