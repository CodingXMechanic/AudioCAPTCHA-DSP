from __future__ import annotations

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform
from audiocaptcha_dsp.psychoacoustics.bark_scale import hz_to_bark, bark_to_hz


class MaskingInjection(BaseTransform):
    def __init__(
        self,
        masker_level_db: float = 20.0,
        bandwidth_bark: float = 2.0,
        center_bark: float = 6.0,
        name: str = "masking_injection",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.masker_level_db = masker_level_db
        self.bandwidth_bark = bandwidth_bark
        self.center_bark = center_bark
        self.seed = seed

    def _make_masker_spectrum(self, n_freq: int, sr: int, rng: np.random.Generator) -> np.ndarray:
        spectrum = np.zeros(n_freq, dtype=np.complex128)
        freqs = np.linspace(0, sr / 2.0, n_freq)
        bark_freqs = hz_to_bark(freqs)
        center_hz = bark_to_hz(np.array([self.center_bark]))[0]

        low_bark = self.center_bark - self.bandwidth_bark / 2.0
        high_bark = self.center_bark + self.bandwidth_bark / 2.0
        low_hz = bark_to_hz(np.array([low_bark]))[0]
        high_hz = bark_to_hz(np.array([high_bark]))[0]

        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        magnitude = 10.0 ** (self.masker_level_db / 20.0)
        phase = rng.uniform(0, 2.0 * np.pi, size=n_freq)
        spectrum[band_mask] = magnitude * np.exp(1j * phase[band_mask])

        return spectrum

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        mono = signal.to_mono().waveform
        n = mono.shape[0]
        sr = signal.sample_rate

        n_fft = int(2 ** np.ceil(np.log2(n)))
        n_freq = n_fft // 2 + 1

        signal_spectrum = np.fft.rfft(mono, n=n_fft)
        masker_spectrum = self._make_masker_spectrum(n_freq, sr, rng)

        combined = signal_spectrum + masker_spectrum
        output_waveform = np.fft.irfft(combined, n=n_fft)[:n].astype(np.float64)

        if signal.waveform.ndim == 2:
            orig_rms = np.sqrt(np.mean(signal.waveform ** 2))
            new_rms = np.sqrt(np.mean(output_waveform ** 2))
            if new_rms > 0:
                output_waveform = output_waveform * (orig_rms / new_rms)
            output_waveform = np.broadcast_to(output_waveform, signal.waveform.shape).copy()

        output = Signal(
            waveform=output_waveform,
            sample_rate=sr,
            metadata={
                **signal.metadata,
                "masker_applied": True,
                "masker_level_db": self.masker_level_db,
                "masker_bandwidth_bark": self.bandwidth_bark,
                "masker_center_bark": self.center_bark,
                "masker_center_hz": float(bark_to_hz(np.array([self.center_bark]))[0]),
                "masker_seed": self.seed,
            },
        )
        return output
