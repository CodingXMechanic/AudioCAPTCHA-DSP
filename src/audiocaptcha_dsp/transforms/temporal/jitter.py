from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import resample_poly

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform


class TemporalJitter(BaseTransform):
    def __init__(
        self,
        amplitude_ms: float = 10.0,
        frequency_hz: float = 5.0,
        name: str = "temporal_jitter",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if amplitude_ms < 0:
            raise ValueError(f"Amplitude must be non-negative, got {amplitude_ms}")
        if frequency_hz <= 0:
            raise ValueError(f"Frequency must be positive, got {frequency_hz}")
        self.amplitude_ms = amplitude_ms
        self.frequency_hz = frequency_hz
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.amplitude_ms == 0.0:
            output = signal.clone()
            output.metadata["jitter_applied"] = False
            return output

        rng = np.random.default_rng(self.seed)
        n = signal.waveform.shape[-1]
        sr = signal.sample_rate
        duration_s = n / sr

        amplitude_samples = int(round(self.amplitude_ms / 1000.0 * sr))
        num_cycles = max(1, int(np.ceil(self.frequency_hz * duration_s)))

        num_control_points = max(4, 2 * num_cycles)
        control_times = np.linspace(0, duration_s, num_control_points)
        control_displacements = rng.uniform(-1.0, 1.0, size=num_control_points)
        control_displacements[0] = 0.0
        control_displacements[-1] = 0.0

        control_samples = (control_times * sr).astype(np.float64)

        base_times = np.arange(n, dtype=np.float64)
        spline = CubicSpline(control_samples, control_displacements)
        displacements = spline(base_times)
        max_abs = np.max(np.abs(displacements)) if np.max(np.abs(displacements)) > 0 else 1.0
        displacement_signal = (displacements / max_abs) * amplitude_samples

        warped_times = base_times + displacement_signal
        warped_times = np.clip(warped_times, 0, n - 1)
        warped_times = np.sort(warped_times)

        orig_mono = signal.waveform
        if orig_mono.ndim == 2:
            orig_mono = orig_mono.mean(axis=0)

        jittered = np.interp(warped_times, base_times, orig_mono).astype(np.float64)

        if jittered.size != n:
            jittered = np.interp(
                np.linspace(0, jittered.size - 1, n),
                np.arange(jittered.size, dtype=np.float64),
                jittered,
            ).astype(np.float64)

        if signal.waveform.ndim == 2:
            jittered = np.broadcast_to(jittered, signal.waveform.shape).copy()

        output = Signal(
            waveform=jittered,
            sample_rate=sr,
            metadata={
                **signal.metadata,
                "jitter_applied": True,
                "jitter_amplitude_ms": self.amplitude_ms,
                "jitter_frequency_hz": self.frequency_hz,
                "jitter_amplitude_samples": amplitude_samples,
                "jitter_num_control_points": num_control_points,
                "jitter_seed": self.seed,
            },
        )
        return output
