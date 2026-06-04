from __future__ import annotations

from dataclasses import dataclass

from audiocaptcha_dsp.core.signal import Signal


@dataclass
class BaseTransform:
    name: str = "base_transform"

    def __call__(self, signal: Signal) -> Signal:
        raise NotImplementedError
