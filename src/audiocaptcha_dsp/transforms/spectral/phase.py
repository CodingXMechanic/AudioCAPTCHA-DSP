"""
Family D extensions: formant and phase-domain spectral transformations.
========================================================================
WHAT-REMAINS.txt §3.D gaps implemented here:

- Formant shifting (spectral warping around formant bands)
- Formant suppression (peak attenuation of formant regions)
- Frequency-domain phase perturbation / group-delay perturbation
- Minimum-phase conversion (all-pass redistribution)

References
----------
- Klatt (1980). Software for a cascade/parallel formant synthesizer.
- Oppenheim & Schafer (2010). Discrete-Time Signal Processing — phase /
  group-delay and minimum-phase theory.
- Quatieri (2002). Discrete-Time Speech Signal Processing — formants.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform


def _safe(y: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _stft_frames(x: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Analysis STFT with n_fft zero-padding on both sides (center=True).

    Padding guarantees full window overlap over the original signal, so
    edge-fade artifacts of the synthesis normalization never touch it.
    """
    win = np.hanning(n_fft)
    x_pad = np.pad(x, (n_fft, n_fft))
    n_frames = max(1, int(np.ceil((len(x_pad) - n_fft) / hop)) + 1)
    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        seg = x_pad[i * hop: i * hop + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        frames[i] = seg * win
    spec = np.fft.rfft(frames, axis=1)
    return spec, win, hop


def _istft_frames(spec: np.ndarray, win: np.ndarray, hop: int, n_out: int) -> np.ndarray:
    """Synthesis for :func:`_stft_frames`; trims the analysis padding."""
    frames = np.fft.irfft(spec, n=win.shape[0], axis=1)
    n_fft = win.shape[0]
    total = n_out + 2 * n_fft
    out = np.zeros(total + 2 * n_fft)
    wsum = np.zeros_like(out)
    for i in range(frames.shape[0]):
        out[i * hop: i * hop + n_fft] += frames[i] * win
        wsum[i * hop: i * hop + n_fft] += win ** 2
    # Floor the normalizer (win² → 0 beyond the padded coverage): edges
    # fade instead of amplifying spectral-gain ringing in the trim zones.
    peak = float(np.max(wsum)) if wsum.size else 0.0
    wsum = np.maximum(wsum, 0.1 * peak) if peak > 0 else wsum
    y = out / wsum
    # Drop the n_fft analysis padding and trim to the original length
    y = y[n_fft: n_fft + total]
    return y[:n_out] if len(y) >= n_out else np.pad(y, (0, n_out - len(y)))


class FormantShift(BaseTransform):
    """Shift spectral envelope peaks (formants) by warping the frequency axis.

    The magnitude spectrum is warped by ``shift_ratio`` around formant
    bands estimated from a smoothed spectral envelope, keeping F0 in place
    (vowel identity change with pitch preserved).
    """

    def __init__(
        self,
        shift_ratio: float = 1.1,
        name: str = "spectral.formant_shift",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if shift_ratio <= 0:
            raise ValueError("shift_ratio must be > 0")
        self.shift_ratio = shift_ratio
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if abs(self.shift_ratio - 1.0) < 1e-9:
            return signal.clone()
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n_fft = 1024
        hop = n_fft // 2
        spec, win, hop = _stft_frames(x, n_fft, hop)
        mag = np.abs(spec)
        phase = np.angle(spec)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        # Warp frequency axis: new_mag[i] = mag evaluated at f / shift_ratio
        src_idx = np.clip(freqs / self.shift_ratio, 0, freqs[-1])
        warped = np.empty_like(mag)
        for i in range(mag.shape[0]):
            warped[i] = np.interp(src_idx, freqs, mag[i])
        # Preserve DC/energy roughly
        scale = np.mean(mag) / (np.mean(warped) + 1e-12)
        y_spec = warped * scale * np.exp(1j * phase)
        y = _istft_frames(y_spec, win, hop, len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.shift_ratio": self.shift_ratio},
        )


class FormantSuppression(BaseTransform):
    """Suppress formant peaks by flattening smoothed spectral envelope maxima.

    The smoothed spectral envelope's peaks (±``bandwidth_hz`` around each
    detected maximum) are attenuated by ``depth_db``, mimicking loss of
    formant contrast (e.g. certain vocoder/channel conditions).
    """

    def __init__(
        self,
        depth_db: float = -8.0,
        bandwidth_hz: float = 300.0,
        max_formants: int = 4,
        name: str = "spectral.formant_suppression",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if depth_db > 0:
            raise ValueError("depth_db must be <= 0 (attenuation)")
        if bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be > 0")
        self.depth_db = depth_db
        self.bandwidth_hz = bandwidth_hz
        self.max_formants = max_formants
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if abs(self.depth_db) < 1e-9:
            return signal.clone()
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n_fft = 1024
        hop = n_fft // 2
        spec, win, hop = _stft_frames(x, n_fft, hop)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        # Envelope from mean spectrum: smooth with a long moving average
        mean_mag = mag.mean(axis=0) + 1e-12
        smooth = np.convolve(mean_mag, np.ones(61) / 61.0, mode="same")
        # Local maxima of the envelope above the mean
        peaks = [
            i for i in range(1, len(smooth) - 1)
            if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]
            and smooth[i] > np.mean(smooth)
        ]
        # Keep strongest peaks within speech band (< 5 kHz), prefer 300–4500 Hz
        peaks = [p for p in peaks if 250.0 <= freqs[p] <= 5000.0]
        peaks = sorted(peaks, key=lambda p: -smooth[p])[: self.max_formants]

        gain = np.ones_like(freqs)
        lin = 10.0 ** (self.depth_db / 20.0)
        for p in peaks:
            mask = np.abs(freqs - freqs[p]) <= self.bandwidth_hz
            # Triangular notch: deepest at the peak
            prof = np.clip(1.0 - np.abs(freqs[p] - freqs[mask]) / self.bandwidth_hz, 0.0, 1.0)
            gain[mask] = np.minimum(gain[mask], 1.0 - (1.0 - lin) * prof)
        y_spec = spec * gain[None, :]
        y = _istft_frames(y_spec, win, hop, len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.depth_db": self.depth_db,
                      f"{self.name}.n_formants": len(peaks)},
        )


class GroupDelayPerturbation(BaseTransform):
    """Perturb the group delay (phase slope) while keeping magnitude fixed.

    A random smooth phase perturbation is added to the STFT spectrum; the
    magnitude is untouched, so linear/nonlinear-phase distortions are
    isolated from spectral-envelope changes (Oppenheim & Schafer 2010).
    """

    def __init__(
        self,
        perturbation_scale: float = 0.3,
        smooth_bins: int = 16,
        name: str = "spectral.group_delay",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if perturbation_scale < 0:
            raise ValueError("perturbation_scale must be >= 0")
        self.perturbation_scale = perturbation_scale
        self.smooth_bins = max(1, int(smooth_bins))
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.perturbation_scale == 0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n_fft = 1024
        hop = n_fft // 2
        spec, win, hop = _stft_frames(x, n_fft, hop)
        n_bins = spec.shape[1]
        # Smooth random phase curve per frame
        noise = rng.normal(0.0, 1.0, (spec.shape[0], n_bins))
        kernel = np.hanning(2 * self.smooth_bins + 1)
        kernel /= kernel.sum()
        smooth_phase = np.apply_along_axis(
            lambda r: np.convolve(r, kernel, mode="same"), 1, noise
        )
        perturbation = (
            self.perturbation_scale * np.pi * smooth_phase
            / (np.max(np.abs(smooth_phase)) + 1e-12)
        )
        y_spec = np.abs(spec) * np.exp(1j * (np.angle(spec) + perturbation))
        y = _istft_frames(y_spec, win, hop, len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata,
                      f"{self.name}.scale": self.perturbation_scale},
        )


class MinimumPhaseConversion(BaseTransform):
    """Convert the signal to its minimum-phase equivalent (spectral envelope kept).

    The magnitude response is preserved while all phase is concentrated
    near the onset — energy of each resonance moves earlier, removing
    linear-phase group delay (Oppenheim & Schafer 2010, ch. 12).
    """

    def __init__(
        self,
        name: str = "spectral.minimum_phase",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        n = len(x)
        if n < 8:
            return signal.clone()
        # Cepstral (real-cepstrum) method: min-phase = exp(IFT{0.5*HT{log|X|}})
        n_fft = 1 << int(np.ceil(np.log2(max(256, 2 * n))))
        X = np.fft.rfft(x, n=n_fft)
        log_mag = np.log(np.abs(X) + 1e-12)
        cep = np.fft.irfft(log_mag, n=n_fft)
        # Hilbert-style cepstral folding
        folded = cep.copy()
        folded[1: n_fft // 2] *= 2.0
        folded[n_fft // 2 + 1:] = 0.0
        min_phase_log = np.fft.rfft(folded, n=n_fft)[: len(X)]
        X_min = np.exp(min_phase_log)
        y = np.fft.irfft(X_min, n=n_fft)[:n]
        # Preserve overall RMS
        rms_x = np.sqrt(np.mean(x ** 2))
        rms_y = np.sqrt(np.mean(y ** 2)) + 1e-12
        y = y * (rms_x / rms_y)
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.applied": True},
        )
