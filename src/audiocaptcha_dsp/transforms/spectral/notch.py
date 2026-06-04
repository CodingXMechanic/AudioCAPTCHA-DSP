from __future__ import annotations

import numpy as np
from scipy.signal import iirnotch, sosfilt, tf2sos

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform


class SpectralNotch(BaseTransform):
    def __init__(
        self,
        frequency_hz: float = 1000.0,
        bandwidth_hz: float = 100.0,
        name: str = "spectral_notch",
    ) -> None:
        super().__init__(name=name)
        if frequency_hz <= 0:
            raise ValueError(f"Notch frequency must be positive, got {frequency_hz}")
        if bandwidth_hz <= 0:
            raise ValueError(f"Bandwidth must be positive, got {bandwidth_hz}")
        self.frequency_hz = frequency_hz
        self.bandwidth_hz = bandwidth_hz

    def __call__(self, signal: Signal) -> Signal:
        sr = signal.sample_rate
        nyquist = sr / 2.0
        if self.frequency_hz >= nyquist:
            raise ValueError(
                f"Notch frequency {self.frequency_hz} Hz must be below Nyquist {nyquist} Hz"
            )

        q = self.frequency_hz / self.bandwidth_hz
        b, a = iirnotch(self.frequency_hz, q, fs=sr)
        sos = tf2sos(b, a)

        if signal.waveform.ndim == 1:
            filtered = sosfilt(sos, signal.waveform).astype(np.float64)
        else:
            channels = []
            for ch in range(signal.waveform.shape[0]):
                f = sosfilt(sos, signal.waveform[ch]).astype(np.float64)
                channels.append(f)
            filtered = np.stack(channels, axis=0)

        output = Signal(
            waveform=filtered,
            sample_rate=sr,
            metadata={
                **signal.metadata,
                "notch_applied": True,
                "notch_frequency_hz": self.frequency_hz,
                "notch_bandwidth_hz": self.bandwidth_hz,
                "notch_q_factor": q,
            },
        )
        return output
