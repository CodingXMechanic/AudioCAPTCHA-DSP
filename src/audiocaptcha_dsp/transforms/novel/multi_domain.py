"""
Multi-Domain Perturbation Transforms
=======================================
Novel Contribution #2: Coordinated time-frequency domain perturbation.

Applies simultaneous perturbation in both time and frequency domains using
psychoacoustic constraints, and adaptive formant-region perturbation.

Key innovations over Schönherr et al. (2018)
--------------------------------------------
The base paper operates exclusively in the frequency domain. This module
introduces:
1. MultiDomainPerturbation: coordinated temporal + spectral perturbation
   such that both attack vectors are applied simultaneously, making defense
   via single-domain enhancement (e.g., spectral subtraction) less effective.
2. AdaptiveFormantPerturbation: detects formant-like spectral peaks and
   applies targeted frequency-domain shifts to disrupt vowel identity in ASR
   while remaining perceptually natural.

References
----------
- Schönherr et al. (2018). arXiv:1808.05665.
- Zwicker & Fastl (1999). Psychoacoustics: Facts and Models.
- Paper 9 (arXiv:2503.11627) — denoising pipelines fail under adversarial noise.
"""
from __future__ import annotations

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform

try:
    from audiocaptcha_dsp.psychoacoustics.masking_models import (
        compute_masking_budget,
        compute_perturbation_budget,
        simultaneous_masking_threshold,
    )
    _MASKING_AVAILABLE = True
except ImportError:
    _MASKING_AVAILABLE = False


def _safe_budget(signal_frame: np.ndarray, sr: int, margin_db: float) -> np.ndarray:
    """Compute masking budget with fallback."""
    if _MASKING_AVAILABLE:
        try:
            return compute_perturbation_budget(signal_frame, sr, margin_db=margin_db)
        except Exception:
            pass
    mag = np.abs(np.fft.rfft(signal_frame))
    return mag * (10.0 ** (-margin_db / 20.0))


class MultiDomainPerturbation(BaseTransform):
    """Coordinated time-frequency domain psychoacoustic perturbation.

    Simultaneously applies:
    1. **Temporal component**: time-domain noise shaped by the envelope of
       the psychoacoustic masking budget.
    2. **Spectral component**: frequency-domain perturbation constrained by
       the masking threshold.

    The combination makes single-domain defense strategies (e.g., spectral
    subtraction, time-domain smoothing) insufficient to fully recover the
    original signal's ASR decodability.

    Parameters
    ----------
    temporal_scale : float
        Scale factor for time-domain noise (relative to masking envelope). 0.3.
    spectral_scale : float
        Scale factor for frequency-domain noise. 0.3.
    margin_db : float
        Safety margin below masking threshold in dB. Default 6.0.
    seed : int or None
        Random seed.
    name : str
        Transform identifier.
    """

    def __init__(
        self,
        temporal_scale: float = 0.3,
        spectral_scale: float = 0.3,
        margin_db: float = 6.0,
        seed: int | None = None,
        name: str = "novel.multi_domain_perturbation",
    ) -> None:
        super().__init__(name=name)
        if temporal_scale < 0 or spectral_scale < 0 or margin_db < 0:
            raise ValueError("All scale and margin parameters must be >= 0")
        self.temporal_scale = float(temporal_scale)
        self.spectral_scale = float(spectral_scale)
        self.margin_db = float(margin_db)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        """Apply multi-domain perturbation."""
        rng = np.random.default_rng(self.seed)
        mono = signal.to_mono().waveform.copy()
        sr = signal.sample_rate
        n = len(mono)

        # --- Spectral component ---
        budget = _safe_budget(mono, sr, self.margin_db)
        spec = np.fft.rfft(mono)
        n_bins = len(spec)
        noise_phase = rng.uniform(-np.pi, np.pi, size=n_bins)
        spectral_noise_td = np.fft.irfft(
            self.spectral_scale * budget * np.exp(1j * noise_phase),
            n=n,
        )

        # --- Temporal component ---
        # Envelope from masking budget (back to time domain as amplitude envelope)
        budget_envelope = np.fft.irfft(budget, n=n)
        budget_envelope = np.abs(budget_envelope)
        # Normalize envelope
        env_max = float(np.max(budget_envelope))
        if env_max > 1e-12:
            budget_envelope /= env_max
        temporal_noise = self.temporal_scale * budget_envelope * rng.normal(0.0, 1.0, size=n)

        # --- Combine ---
        output = mono + spectral_noise_td + temporal_noise
        output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return Signal(
            waveform=output,
            sample_rate=sr,
            metadata={**signal.metadata, "transform": self.name},
        )


class AdaptiveFormantPerturbation(BaseTransform):
    """Formant-adaptive frequency perturbation.

    Detects the dominant spectral peaks (formant-like resonances) in the
    magnitude spectrum via peak-picking on a smoothed envelope. Applies
    a controlled frequency shift (``±formant_shift_hz``) to energy in the
    vicinity of detected formants. This disrupts vowel identity in ASR
    feature extraction (Wav2Vec 2.0, HuBERT formant encoding) while
    remaining perceptually close to the original.

    Parameters
    ----------
    formant_shift_hz : float
        Maximum frequency shift applied near formant peaks, in Hz. Default 50.
    num_formants : int
        Number of formant peaks to target. Default 3.
    smoothing_bins : int
        Window size for spectral smoothing (peak detection). Default 7.
    seed : int or None
        Random seed.
    name : str
        Transform identifier.
    """

    def __init__(
        self,
        formant_shift_hz: float = 50.0,
        num_formants: int = 3,
        smoothing_bins: int = 7,
        seed: int | None = None,
        name: str = "novel.adaptive_formant_perturbation",
    ) -> None:
        super().__init__(name=name)
        if formant_shift_hz < 0:
            raise ValueError("formant_shift_hz must be >= 0")
        if num_formants < 1:
            raise ValueError("num_formants must be >= 1")
        self.formant_shift_hz = float(formant_shift_hz)
        self.num_formants = num_formants
        self.smoothing_bins = max(1, smoothing_bins)
        self.seed = seed

    def _find_formant_peaks(
        self,
        mag: np.ndarray,
        freqs: np.ndarray,
        f_min: float = 200.0,
        f_max: float = 4000.0,
    ) -> list[int]:
        """Find top-N spectral peaks in speech formant range."""
        # Restrict to formant frequency range
        mask = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(mask):
            return []

        # Smooth magnitude
        win = np.ones(self.smoothing_bins) / self.smoothing_bins
        smoothed = np.convolve(mag, win, mode="same")

        # Peak picking (local maxima in formant range)
        candidate_bins = np.where(mask)[0]
        peaks = []
        for bi in candidate_bins:
            lo = max(0, bi - 2)
            hi = min(len(smoothed) - 1, bi + 2)
            if smoothed[bi] == np.max(smoothed[lo:hi + 1]):
                peaks.append(bi)

        if not peaks:
            # Fall back to top-N bins in range
            restricted_mag = np.where(mask, smoothed, 0.0)
            top_idx = np.argsort(restricted_mag)[-self.num_formants:]
            return sorted(top_idx.tolist())

        # Sort by magnitude and take top num_formants
        peaks_sorted = sorted(peaks, key=lambda k: smoothed[k], reverse=True)
        return peaks_sorted[: self.num_formants]

    def __call__(self, signal: Signal) -> Signal:
        """Apply adaptive formant perturbation."""
        rng = np.random.default_rng(self.seed)
        mono = signal.to_mono().waveform.copy()
        sr = signal.sample_rate
        n = len(mono)

        if n < 128:
            return signal.clone()

        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        spec = np.fft.rfft(mono)
        mag = np.abs(spec)
        phase = np.angle(spec)

        formant_bins = self._find_formant_peaks(mag, freqs)

        if not formant_bins:
            return signal.clone()

        # Apply shift around each formant peak
        shift_bins_max = max(1, int(round(self.formant_shift_hz / (sr / n))))
        mag_perturbed = mag.copy()

        for fb in formant_bins:
            # Random shift direction and magnitude
            shift = rng.integers(-shift_bins_max, shift_bins_max + 1)
            if shift == 0:
                continue
            # Shift energy near this formant
            region = slice(
                max(0, fb - shift_bins_max - 2),
                min(len(mag), fb + shift_bins_max + 3),
            )
            region_len = mag[region].shape[0]
            shifted_region = np.zeros(region_len, dtype=np.float64)
            for k in range(region_len):
                src_k = k - shift
                if 0 <= src_k < region_len:
                    shifted_region[k] = mag[region][src_k]
            mag_perturbed[region] = shifted_region

        # Reconstruct with original phase
        spec_new = mag_perturbed * np.exp(1j * phase)
        output = np.fft.irfft(spec_new, n=n)
        output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return Signal(
            waveform=output,
            sample_rate=sr,
            metadata={**signal.metadata, "transform": self.name},
        )
