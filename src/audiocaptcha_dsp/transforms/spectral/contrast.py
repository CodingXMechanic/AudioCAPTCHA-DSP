"""
Family D extensions: spectral contrast, equalization, sharpening, holes, T-F masking.
======================================================================================
WHAT-REMAINS.txt §3.D gaps implemented here:

- Spectral contrast modification (peaks vs valleys per band)
- Random spectral equalization (multi-band random gain curve)
- Spectral sharpening (spectral peak emphasis / de-emphasis of valleys)
- Spectral hole creation (broadband band rejection)
- Time-frequency masking (T-F unit masking à la SpecAugment in the STFT)

References
----------
- Huang et al. (2011). Spectral-temporal features for speaker recognition
  (spectral contrast). IEEE TASLP.
- Park et al. (2019). SpecAugment (time-frequency masking). Interspeech.
- Klatt (1982). Pitch and frequency demodulation (spectral peak structure).
"""
from __future__ import annotations

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform


def _safe(y: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _stft(x: np.ndarray, n_fft: int, hop: int):
    """Analysis STFT with n_fft zero-padding on both sides (center=True)."""
    win = np.hanning(n_fft)
    x_pad = np.pad(x, (n_fft, n_fft))
    n_frames = max(1, int(np.ceil((len(x_pad) - n_fft) / hop)) + 1)
    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        seg = x_pad[i * hop: i * hop + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        frames[i] = seg * win
    return np.fft.rfft(frames, axis=1), win


def _istft(spec: np.ndarray, win: np.ndarray, hop: int, n_out: int) -> np.ndarray:
    """Synthesis for :func:`_stft`; trims the analysis padding."""
    frames = np.fft.irfft(spec, n=win.shape[0], axis=1)
    n_fft = win.shape[0]
    total = n_out + 2 * n_fft
    out = np.zeros(total + 2 * n_fft)
    wsum = np.zeros_like(out)
    for i in range(frames.shape[0]):
        out[i * hop: i * hop + n_fft] += frames[i] * win
        wsum[i * hop: i * hop + n_fft] += win ** 2
    # Floor the normalizer (win² → 0 beyond padded coverage): trim-zone
    # edges fade instead of amplifying spectral-gain ringing.
    peak = float(np.max(wsum)) if wsum.size else 0.0
    wsum = np.maximum(wsum, 0.1 * peak) if peak > 0 else wsum
    y = out / wsum
    y = y[n_fft: n_fft + total]
    return y[:n_out] if len(y) >= n_out else np.pad(y, (0, n_out - len(y)))


class SpectralContrastModification(BaseTransform):
    """Modify the contrast between spectral peaks and valleys per frequency band.

    Bands are taken as contiguous Bark-like groups; per band the deviation
    of the spectrum from its band mean is scaled by ``contrast_factor``.
    Values >1 exaggerate peak/valley structure, <1 flatten it.
    """

    def __init__(
        self,
        contrast_factor: float = 1.5,
        n_bands: int = 16,
        name: str = "spectral.contrast",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if contrast_factor < 0:
            raise ValueError("contrast_factor must be >= 0")
        if n_bands < 2:
            raise ValueError("n_bands must be >= 2")
        self.contrast_factor = contrast_factor
        self.n_bands = n_bands
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if abs(self.contrast_factor - 1.0) < 1e-9:
            return signal.clone()
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n_fft, hop = 1024, 512
        spec, win = _stft(x, n_fft, hop)
        mag = np.abs(spec)
        phase = np.angle(spec)
        n_bins = mag.shape[1]
        edges = np.linspace(0, n_bins, self.n_bands + 1, dtype=int)
        out_mag = np.empty_like(mag)
        for b in range(self.n_bands):
            lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
            band = mag[:, lo:hi]
            mean = band.mean(axis=1, keepdims=True)
            out_mag[:, lo:hi] = mean + (band - mean) * self.contrast_factor
        out_mag = np.maximum(out_mag, 0.0)
        y = _istft(out_mag * np.exp(1j * phase), win, hop, len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.contrast_factor": self.contrast_factor},
        )


class RandomSpectralEqualization(BaseTransform):
    """Apply a random smooth multi-band equalization curve.

    ``n_bands`` band gains are drawn uniformly in ±``max_gain_db`` and
    interpolated smoothly across frequency — models unknown playback EQ /
    room coloration and randomized channel response.
    """

    def __init__(
        self,
        max_gain_db: float = 6.0,
        n_bands: int = 8,
        name: str = "spectral.random_eq",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if max_gain_db < 0:
            raise ValueError("max_gain_db must be >= 0")
        if n_bands < 2:
            raise ValueError("n_bands must be >= 2")
        self.max_gain_db = max_gain_db
        self.n_bands = n_bands
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.max_gain_db == 0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n_fft, hop = 1024, 512
        spec, win = _stft(x, n_fft, hop)
        n_bins = spec.shape[1]
        band_centers = np.linspace(0, n_bins - 1, self.n_bands)
        gains_db = rng.uniform(-self.max_gain_db, self.max_gain_db, self.n_bands)
        bins = np.arange(n_bins)
        gain_curve_db = np.interp(bins, band_centers, gains_db)
        gain = 10.0 ** (gain_curve_db / 20.0)
        y = _istft(spec * gain[None, :], win, hop, len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.max_gain_db": self.max_gain_db},
        )


class SpectralSharpening(BaseTransform):
    """Sharpen the spectrum: amplify peaks relative to local valleys.

    Applies spectral whitening-style emphasis: mag' = mag * (mag/local_mean)**k
    with smoothing width ``smooth_bins``. k>0 sharpens harmonic structure.
    """

    def __init__(
        self,
        strength: float = 0.5,
        smooth_bins: int = 9,
        name: str = "spectral.sharpening",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if strength < 0:
            raise ValueError("strength must be >= 0")
        self.strength = strength
        self.smooth_bins = max(1, int(smooth_bins))
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.strength == 0:
            return signal.clone()
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n_fft, hop = 1024, 512
        spec, win = _stft(x, n_fft, hop)
        mag = np.abs(spec)
        phase = np.angle(spec)
        kernel = np.ones(2 * self.smooth_bins + 1) / (2 * self.smooth_bins + 1)
        local_mean = np.apply_along_axis(
            lambda r: np.convolve(r, kernel, mode="same"), 1, mag + 1e-12
        )
        ratio = np.clip(mag / local_mean, 1e-3, 1e3)
        out_mag = mag * ratio ** self.strength
        # Energy normalization per frame relative to input
        scale = (mag.sum(axis=1, keepdims=True) + 1e-12) / (out_mag.sum(axis=1, keepdims=True) + 1e-12)
        out_mag = out_mag * scale
        y = _istft(out_mag * np.exp(1j * phase), win, hop, len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.strength": self.strength},
        )


class SpectralHole(BaseTransform):
    """Carve a broadband spectral hole (notch band) across the whole signal.

    Unlike a static notch filter, the hole is created in the STFT domain
    with a smooth-shouldered bandstop of width ``width_hz`` around
    ``center_hz`` plus spectral interpolation, modeling codec band dropout.
    """

    def __init__(
        self,
        center_hz: float = 2000.0,
        width_hz: float = 800.0,
        depth_db: float = -30.0,
        name: str = "spectral.hole",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if width_hz <= 0 or depth_db > 0:
            raise ValueError("width_hz > 0 and depth_db <= 0 required")
        self.center_hz = center_hz
        self.width_hz = width_hz
        self.depth_db = depth_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n_fft, hop = 2048, 1024
        spec, win = _stft(x, n_fft, hop)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        half = self.width_hz / 2.0
        dist = np.abs(freqs - self.center_hz)
        # Smooth-shouldered attenuation (raised-cosine notch)
        gain = np.ones_like(freqs)
        inside = dist <= half
        gain[inside] = 10.0 ** (self.depth_db / 20.0)
        shoulder = (dist > half) & (dist <= 1.5 * half)
        t = (dist[shoulder] - half) / (0.5 * half)
        gain[shoulder] = 10.0 ** (self.depth_db / 20.0) + (1.0 - 10.0 ** (self.depth_db / 20.0)) * (0.5 * (1 + np.cos(np.pi * t)))
        y = _istft(spec * gain[None, :], win, hop, len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.center_hz": self.center_hz,
                      f"{self.name}.width_hz": self.width_hz},
        )


class TimeFrequencyMasking(BaseTransform):
    """SpecAugment-style masking directly in the STFT time-frequency plane.

    ``n_masks`` rectangular T-F regions are zeroed with dimensions drawn
    uniformly up to ``max_freq_mask`` (fractions of bins) and
    ``max_time_mask`` (fractions of frames) — models intermittent
    narrowband interference or human auditory attentional masking.
    """

    def __init__(
        self,
        n_masks: int = 2,
        max_freq_mask: float = 0.15,
        max_time_mask: float = 0.15,
        name: str = "spectral.tf_masking",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if n_masks < 0:
            raise ValueError("n_masks must be >= 0")
        if not (0 <= max_freq_mask <= 1) or not (0 <= max_time_mask <= 1):
            raise ValueError("mask fractions must be within [0, 1]")
        self.n_masks = n_masks
        self.max_freq_mask = max_freq_mask
        self.max_time_mask = max_time_mask
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.n_masks == 0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n_fft, hop = 1024, 512
        spec, win = _stft(x, n_fft, hop)
        spec = spec.copy()
        n_t, n_f = spec.shape
        for _ in range(self.n_masks):
            f_width = int(rng.uniform(0, self.max_freq_mask) * n_f)
            t_width = int(rng.uniform(0, self.max_time_mask) * n_t)
            if f_width > 0:
                f0 = int(rng.integers(0, n_f - f_width + 1))
                spec[:, f0: f0 + f_width] = 0.0
            if t_width > 0:
                t0 = int(rng.integers(0, n_t - t_width + 1))
                spec[t0: t0 + t_width, :] = 0.0
        y = _istft(spec, win, hop, len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.n_masks": self.n_masks},
        )
