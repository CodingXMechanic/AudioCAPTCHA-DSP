from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRegistry:
    experiments_dir: Path = Path("configs/experiments")
    _registry: dict[str, Path] = field(default_factory=dict, repr=False)

    def discover(self) -> dict[str, Path]:
        if not self.experiments_dir.exists():
            logger.warning("Experiments directory not found: %s", self.experiments_dir)
            return {}
        for path in sorted(self.experiments_dir.glob("*.yaml")):
            self._registry[path.stem] = path
        logger.info("Discovered %d experiments", len(self._registry))
        return self._registry

    def get(self, name: str) -> Path:
        if name not in self._registry:
            raise KeyError(f"Experiment '{name}' not found. Available: {list(self._registry.keys())}")
        return self._registry[name]
