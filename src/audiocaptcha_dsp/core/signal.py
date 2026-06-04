from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Signal:
    waveform: np.ndarray
    sample_rate: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.waveform.ndim == 1:
            self.waveform = self.waveform.astype(np.float64)
        elif self.waveform.ndim == 2:
            self.waveform = self.waveform.astype(np.float64)
        else:
            raise ValueError(f"Waveform must be 1D or 2D, got {self.waveform.ndim}D")

    @property
    def duration_seconds(self) -> float:
        return float(self.waveform.shape[-1]) / self.sample_rate

    @property
    def num_channels(self) -> int:
        return 1 if self.waveform.ndim == 1 else self.waveform.shape[0]

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.waveform.astype(np.float64) ** 2)))

    def clone(self) -> Signal:
        return Signal(
            waveform=self.waveform.copy(),
            sample_rate=self.sample_rate,
            metadata=self.metadata.copy(),
        )

    def to_mono(self) -> Signal:
        if self.waveform.ndim == 1:
            return self.clone()
        return Signal(
            waveform=self.waveform.mean(axis=0),
            sample_rate=self.sample_rate,
            metadata={**self.metadata, "mono_mixed_from_channels": self.num_channels},
        )
