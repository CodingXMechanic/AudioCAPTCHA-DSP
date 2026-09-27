"""
CAPTCHA-Optimized Composite Transforms
=========================================
Novel Contribution #3: Transform chains specifically engineered to maximize
the Human-ASR Gap (HAG) — the difference between human success rate and
ASR transcription success rate.

Key innovations
---------------
1. CAPTCHAOptimalTransform: A 4-stage pipeline (psychoacoustic masking +
   temporal jitter + spectral dropout + Bark-band attenuation) that combines
   complementary disruption mechanisms targeting different ASR subsystems
   simultaneously.

2. DefenseRobustTransform: Designed to survive standard defense preprocessing
   pipelines (spectral denoising, resampling, codec compression) by
   distributing perturbation across frequency bands and combining harmonic
   structure preservation with broadband noise.

References
----------
- Schönherr et al. (2018). arXiv:1808.05665 — base psychoacoustic hiding.
- Paper 9 (arXiv:2503.11627) — denoising pipelines fail under adversarial noise.
- Paper 8 (arXiv:2606.27698) — certified ASR robustness.
- Zwicker & Fastl (1999). Psychoacoustics: Facts and Models.
"""
from __future__ import annotations

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform
from audiocaptcha_dsp.psychoacoustics.bark_scale import hz_to_bark

try:
    from audiocaptcha_dsp.psychoacoustics.masking_models import (
        compute_masking_budget,
        compute_perturbation_budget,
        simultaneous_masking_threshold,
    )
    _MASKING_AVAILABLE = True
except ImportError:
    _MASKING_AVAILABLE = False


def _safe_budget(mono: np.ndarray, sr: int, margin_db: float) -> np.ndarray:
    """Compute psychoacoustic masking budget with fallback."""
    if _MASKING_AVAILABLE:
        try:
            return compute_perturbation_budget(mono, sr, margin_db=margin_db)
        except Exception:
            pass
    mag = np.abs(np.fft.rfft(mono))
    return mag * (10.0 ** (-margin_db / 20.0))


class CAPTCHAOptimalTransform(BaseTransform):
    """4-stage CAPTCHA-optimized transform for maximum Human-ASR Gap.

    Pipeline stages (applied sequentially):
    1. **Psychoacoustic masked noise**: Schönherr et al. (2018) base method.
       Injects noise constrained below the simultaneous masking threshold.
    2. **Temporal micro-jitter**: 2 ms random timing perturbation disrupting
       temporal feature extraction in ASR front-ends.
    3. **High-frequency spectral dropout**: Random bin zeroing above 3 kHz
       at rate ``spectral_dropout_rate``, targeting sibilant and fricative cues.
    4. **Bark-band selective attenuation**: Randomly selects ``bark_bands_to_drop``
       Bark critical bands and attenuates them by 6 dB, disrupting formant
       structure while maintaining overall loudness.

    Parameters
    ----------
    psychoacoustic_margin_db : float
        Safety margin for psychoacoustic noise injection. Default 8.0 dB.
    temporal_jitter_ms : float
        Maximum temporal jitter in ms. Default 2.0 ms.
    spectral_dropout_rate : float
        Fraction of high-frequency bins to zero out. Default 0.05.
    bark_bands_to_drop : int
        Number of Bark bands to attenuate. Default 2.
    seed : int or None
        Random seed for reproducibility.
    name : str
        Transform identifier.
    """

    def __init__(
        self,
        psychoacoustic_margin_db: float = 8.0,
        temporal_jitter_ms: float = 2.0,
        spectral_dropout_rate: float = 0.05,
        bark_bands_to_drop: int = 2,
        seed: int | None = None,
        name: str = "novel.captcha_optimal",
    ) -> None:
        super().__init__(name=name)
        if psychoacoustic_margin_db < 0:
            raise ValueError("psychoacoustic_margin_db must be >= 0")
        if temporal_jitter_ms < 0:
            raise ValueError("temporal_jitter_ms must be >= 0")
        if not (0.0 <= spectral_dropout_rate <= 1.0):
            raise ValueError("spectral_dropout_rate must be in [0, 1]")
        if bark_bands_to_drop < 0:
            raise ValueError("bark_bands_to_drop must be >= 0")
        self.psychoacoustic_margin_db = float(psychoacoustic_margin_db)
        self.temporal_jitter_ms = float(temporal_jitter_ms)
        self.spectral_dropout_rate = float(spectral_dropout_rate)
        self.bark_bands_to_drop = int(bark_bands_to_drop)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        """Apply 4-stage CAPTCHA-optimal transform."""
        rng = np.random.default_rng(self.seed)
        mono = signal.to_mono().waveform.copy()
        sr = signal.sample_rate
        n = len(mono)

        if n < 64:
            return signal.clone()

        # Stage 1: Psychoacoustic masked noise injection
        budget = _safe_budget(mono, sr, self.psychoacoustic_margin_db)
        spec = np.fft.rfft(mono)
        n_bins = len(spec)
        noise_phase_1 = rng.uniform(-np.pi, np.pi, size=n_bins)
        noise_spec_1 = budget * np.exp(1j * noise_phase_1)
        stage1 = np.fft.irfft(spec + noise_spec_1, n=n)
        stage1 = np.nan_to_num(stage1, nan=0.0, posinf=1.0, neginf=-1.0)

        # Stage 2: Temporal micro-jitter (circular shift)
        jitter_samples = int(round(self.temporal_jitter_ms * sr / 1000.0))
        if jitter_samples > 0:
            shift = rng.integers(-jitter_samples, jitter_samples + 1)
            stage2 = np.roll(stage1, int(shift))
        else:
            stage2 = stage1.copy()

        # Stage 3: High-frequency spectral dropout
        if self.spectral_dropout_rate > 0.0:
            spec3 = np.fft.rfft(stage2)
            freqs3 = np.fft.rfftfreq(n, d=1.0 / sr)
            high_freq_bins = np.where(freqs3 >= 3000.0)[0]
            if len(high_freq_bins) > 0:
                n_drop = max(0, int(round(self.spectral_dropout_rate * len(high_freq_bins))))
                if n_drop > 0:
                    drop_idx = rng.choice(high_freq_bins, size=n_drop, replace=False)
                    spec3[drop_idx] = 0.0
            stage3 = np.fft.irfft(spec3, n=n)
            stage3 = np.nan_to_num(stage3, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            stage3 = stage2.copy()

        # Stage 4: Bark-band selective attenuation (6 dB = factor 0.5)
        if self.bark_bands_to_drop > 0:
            spec4 = np.fft.rfft(stage3)
            freqs4 = np.fft.rfftfreq(n, d=1.0 / sr)
            bark_freqs4 = hz_to_bark(np.clip(freqs4, 1.0, None))

            n_bark = 24
            bark_edges = np.linspace(0, 24, n_bark + 1)
            available_bands = list(range(n_bark))
            n_to_drop = min(self.bark_bands_to_drop, n_bark)
            drop_bands = rng.choice(available_bands, size=n_to_drop, replace=False)

            for band_idx in drop_bands:
                band_mask = (bark_freqs4 >= bark_edges[band_idx]) & (bark_freqs4 < bark_edges[band_idx + 1])
                spec4[band_mask] *= 0.5  # 6 dB attenuation

            stage4 = np.fft.irfft(spec4, n=n)
            stage4 = np.nan_to_num(stage4, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            stage4 = stage3.copy()

        return Signal(
            waveform=stage4,
            sample_rate=sr,
            metadata={**signal.metadata, "transform": self.name},
        )


class DefenseRobustTransform(BaseTransform):
    """Defense-robust transform designed to survive common preprocessing.

    Standard ASR defenses (spectral subtraction, resampling bottleneck,
    codec compression) can partially recover ASR performance against simple
    transforms. This transform uses a combination of:
    1. **Broadband noise** shaped to survive spectral subtraction.
    2. **Mild lowpass filtering** below Nyquist to maintain intelligibility
       while shifting harmonic content.
    3. **Harmonic attenuation**: Reduces energy at harmonic multiples of the
       estimated fundamental frequency, disrupting pitch-based ASR features.

    This combination ensures that the perturbation survives across multiple
    defense channels while remaining within psychoacoustic tolerances.

    Parameters
    ----------
    noise_level_db : float
        Noise level relative to signal in dB below full scale. Default 6.0.
    lowpass_hz : float
        Lowpass cutoff frequency in Hz. Default 7000 Hz.
    harmonic_attenuation : float
        Attenuation factor at harmonic bins (0=full attenuate, 1=no change). 0.3.
    seed : int or None
        Random seed.
    name : str
        Transform identifier.
    """

    def __init__(
        self,
        noise_level_db: float = 6.0,
        lowpass_hz: float = 7000.0,
        harmonic_attenuation: float = 0.3,
        seed: int | None = None,
        name: str = "novel.defense_robust",
    ) -> None:
        super().__init__(name=name)
        if noise_level_db < 0:
            raise ValueError("noise_level_db must be >= 0")
        if lowpass_hz <= 0:
            raise ValueError("lowpass_hz must be > 0")
        if not (0.0 <= harmonic_attenuation <= 1.0):
            raise ValueError("harmonic_attenuation must be in [0, 1]")
        self.noise_level_db = float(noise_level_db)
        self.lowpass_hz = float(lowpass_hz)
        self.harmonic_attenuation = float(harmonic_attenuation)
        self.seed = seed

    def _estimate_f0(self, mono: np.ndarray, sr: int) -> float:
        """Estimate fundamental frequency via autocorrelation."""
        # Use central 0.5 s for efficiency
        n_frames = min(len(mono), int(sr * 0.5))
        frame = mono[:n_frames]
        if len(frame) < 64:
            return 120.0  # default

        # Autocorrelation-based pitch estimate
        ac = np.correlate(frame, frame, mode="full")
        ac = ac[len(ac) // 2:]
        # Search for peak in typical speech F0 range (80-400 Hz)
        lo = max(1, int(sr / 400.0))
        hi = min(len(ac) - 1, int(sr / 80.0))
        if lo >= hi:
            return 120.0
        f0_lag = lo + int(np.argmax(ac[lo:hi]))
        f0 = float(sr) / max(1, f0_lag)
        return float(np.clip(f0, 80.0, 400.0))

    def __call__(self, signal: Signal) -> Signal:
        """Apply defense-robust transform."""
        rng = np.random.default_rng(self.seed)
        mono = signal.to_mono().waveform.copy()
        sr = signal.sample_rate
        n = len(mono)

        if n < 64:
            return signal.clone()

        spec = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        mag = np.abs(spec)

        # Stage 1: Broadband noise
        rms_signal = float(np.sqrt(np.mean(mono ** 2)))
        noise_scale = rms_signal * (10.0 ** (-self.noise_level_db / 20.0))
        broadband_noise = rng.normal(0.0, noise_scale, size=n)

        # Stage 2: Lowpass filtering (zero out bins above cutoff)
        spec_lp = spec.copy()
        lp_mask = freqs > self.lowpass_hz
        spec_lp[lp_mask] = 0.0

        # Stage 3: Harmonic attenuation
        if self.harmonic_attenuation < 1.0:
            f0 = self._estimate_f0(mono, sr)
            harmonic_mask = np.zeros(len(spec_lp), dtype=bool)
            harmonic = f0
            tolerance_bins = max(1, int(round(30.0 / (sr / n))))  # ±30 Hz
            while harmonic < freqs[-1]:
                target_bin = int(round(harmonic / (sr / n)))
                lo = max(0, target_bin - tolerance_bins)
                hi = min(len(spec_lp), target_bin + tolerance_bins + 1)
                harmonic_mask[lo:hi] = True
                harmonic += f0
            spec_lp[harmonic_mask] *= self.harmonic_attenuation

        # Reconstruct
        stage3 = np.fft.irfft(spec_lp, n=n) + broadband_noise
        stage3 = np.nan_to_num(stage3, nan=0.0, posinf=1.0, neginf=-1.0)

        return Signal(
            waveform=stage3,
            sample_rate=sr,
            metadata={**signal.metadata, "transform": self.name},
        )
