from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform


class Resampler(BaseTransform):
    def __init__(self, factor: float, name: str = "resampler") -> None:
        super().__init__(name=name)
        if factor <= 0:
            raise ValueError(f"Resample factor must be positive, got {factor}")
        self.factor = factor

    def __call__(self, signal: Signal) -> Signal:
        if self.factor == 1.0:
            return signal.clone()

        orig_len = signal.waveform.shape[-1]
        new_len = int(round(orig_len / self.factor))

        up = new_len
        down = orig_len
        gcd = np.gcd(up, down)
        up //= gcd
        down //= gcd

        if signal.waveform.ndim == 1:
            resampled = resample_poly(signal.waveform, up, down).astype(np.float64)
        else:
            channels = []
            for ch in range(signal.waveform.shape[0]):
                r = resample_poly(signal.waveform[ch], up, down).astype(np.float64)
                channels.append(r)
            resampled = np.stack(channels, axis=0)

        output = Signal(
            waveform=resampled,
            sample_rate=signal.sample_rate,
            metadata={
                **signal.metadata,
                "resample_factor": self.factor,
                "original_length": orig_len,
                "resampled_length": resampled.shape[-1],
            },
        )
        return output
