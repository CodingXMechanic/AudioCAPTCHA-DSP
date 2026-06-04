from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.types import Transform

logger = logging.getLogger(__name__)


@dataclass
class Pipeline:
    transforms: list[Transform] = field(default_factory=list)
    name: str = "pipeline"

    def add(self, transform: Transform) -> Pipeline:
        self.transforms.append(transform)
        return self

    def apply(self, signal: Signal) -> Signal:
        result = signal
        for transform in self.transforms:
            logger.debug("Applying %s to signal", transform)
            result = transform(result)
        return result

    @classmethod
    def from_config(cls, config_path: Path | str) -> Pipeline:
        config_path = Path(config_path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded pipeline config from %s", config_path)
        return cls(name=config.get("name", config_path.stem))
