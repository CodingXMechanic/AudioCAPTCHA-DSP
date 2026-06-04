from __future__ import annotations

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform
from audiocaptcha_dsp.psychoacoustics.bark_scale import hz_to_bark, bark_to_hz


class FrequencyWarping(BaseTransform):
    def __init__(
        self,
        alpha: float = 0.0,
        name: str = "frequency_warping",
    ) -> None:
        super().__init__(name=name)
        self.alpha = alpha

    def _warp_spectrum(self, spectrum: np.ndarray, freqs: np.ndarray, alpha: float) -> np.ndarray:
        n_freq = len(freqs)
        if alpha == 0.0 or n_freq < 2:
            return spectrum.copy()

        bark = hz_to_bark(freqs)
        bark_min = bark[0] if bark[0] > 0 else 1e-6
        bark_max = bark[-1]

        warped_bark = bark * (1.0 + alpha * (bark - bark_min) / max(bark_max - bark_min, 1e-6))
        warped_hz = bark_to_hz(warped_bark)
        warped_hz = np.clip(warped_hz, freqs[0], freqs[-1])

        magnitudes = np.abs(spectrum)
        phases = np.angle(spectrum)

        warped_magnitudes = np.interp(freqs, warped_hz, magnitudes)
        warped_phases = np.interp(freqs, warped_hz, phases)

        return (warped_magnitudes * np.exp(1j * warped_phases)).astype(np.complex128)

    def __call__(self, signal: Signal) -> Signal:
        if self.alpha == 0.0:
            output = signal.clone()
            output.metadata["warping_applied"] = False
            return output

        mono = signal.to_mono().waveform
        n = mono.shape[0]
        sr = signal.sample_rate

        n_fft = int(2 ** np.ceil(np.log2(n)))
        n_freq = n_fft // 2 + 1
        freqs = np.linspace(0, sr / 2.0, n_freq)

        spectrum = np.fft.rfft(mono, n=n_fft)
        warped = self._warp_spectrum(spectrum, freqs, self.alpha)
        output_waveform = np.fft.irfft(warped, n=n_fft)[:n].astype(np.float64)

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
                "warping_applied": True,
                "warping_alpha": self.alpha,
            },
        )
        return output
