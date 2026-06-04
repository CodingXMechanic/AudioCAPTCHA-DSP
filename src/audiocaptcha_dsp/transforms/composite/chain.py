from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.types import Transform
from audiocaptcha_dsp.transforms.base import BaseTransform


@dataclass
class TransformChain(BaseTransform):
    transforms: list[Transform] = field(default_factory=list)
    name: str = "transform_chain"

    def __call__(self, signal: Signal) -> Signal:
        result = signal
        for transform in self.transforms:
            result = transform(result)
        return result
