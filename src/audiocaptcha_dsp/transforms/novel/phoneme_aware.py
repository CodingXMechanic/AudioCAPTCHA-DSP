"""
Phoneme-Aware Perturbation Transforms
========================================
Novel contribution: Differentially allocates masking-constrained perturbation
across energy-segmented phoneme regions. High-energy frames (vowels, voiced
consonants) receive stronger perturbation; low-energy frames (fricatives,
silence) receive weaker perturbation. This exploits the empirical finding
(Paper 21: arXiv 2406.08619) that SSL representations are more phonetic than
semantic, making phoneme-boundary distortions maximally disruptive to ASR.

Key innovation over Schönherr et al. (2018)
--------------------------------------------
The base paper applies psychoacoustic masking uniformly across the entire
waveform. This module introduces *phoneme-granular allocation* — the
perturbation budget is differentially weighted based on the energy profile
of each overlapping analysis frame, approximating vowel vs. consonant regions
without requiring a forced aligner. This produces stronger disruption at
phonetically critical boundaries while maintaining overall perceptual quality.

References
----------
- Schönherr et al. (2018). arXiv:1808.05665 — psychoacoustic masking base.
- Baevski et al. (2020). NeurIPS — wav2vec 2.0 phonetic sensitivity.
- Hsu et al. (2021). TASLP — HuBERT phoneme-level masked prediction.
- arXiv 2406.08619 (2024) — SSL representations are more phonetic than semantic.
- arXiv 2307.12498 (2023) — WavAugment phoneme adversarial training.
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


def _safe_masking_budget(
    frame: np.ndarray,
    sr: int,
    margin_db: float,
) -> np.ndarray:
    """Compute masking budget with fallback to simple spectral envelope."""
    if _MASKING_AVAILABLE:
        try:
            return compute_perturbation_budget(frame, sr, margin_db=margin_db)
        except Exception:
            pass
    # Fallback: flat budget proportional to spectral magnitude
    spec = np.fft.rfft(frame)
    mag = np.abs(spec)
    scale = 10.0 ** (-margin_db / 20.0)
    return mag * scale


class PhonemeAwarePerturbation(BaseTransform):
    """Phoneme-granular psychoacoustically-constrained perturbation.

    Segments the input waveform into overlapping frames and classifies each
    frame as 'vowel-like' (high energy) or 'consonant-like' (low energy)
    using frame RMS relative to the median RMS. The psychoacoustic masking
    budget (Schönherr et al. 2018) is then scaled by ``vowel_scale`` for
    high-energy frames and ``consonant_scale`` for low-energy frames before
    applying random-phase spectral perturbation.

    Parameters
    ----------
    frame_duration_ms : float
        Analysis frame length in milliseconds. Default 25 ms.
    hop_duration_ms : float
        Frame hop size in milliseconds. Default 10 ms.
    vowel_scale : float
        Perturbation scale for vowel-like (high-energy) frames. Default 1.0.
    consonant_scale : float
        Perturbation scale for consonant-like (low-energy) frames. Default 0.3.
    margin_db : float
        Safety margin below masking threshold in dB. Default 6.0.
    seed : int or None
        Random seed for reproducibility.
    name : str
        Unique transform identifier.
    """

    def __init__(
        self,
        frame_duration_ms: float = 25.0,
        hop_duration_ms: float = 10.0,
        vowel_scale: float = 1.0,
        consonant_scale: float = 0.3,
        margin_db: float = 6.0,
        seed: int | None = None,
        name: str = "novel.phoneme_aware_perturbation",
    ) -> None:
        super().__init__(name=name)
        if frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be > 0")
        if hop_duration_ms <= 0:
            raise ValueError("hop_duration_ms must be > 0")
        if vowel_scale < 0 or consonant_scale < 0:
            raise ValueError("vowel_scale and consonant_scale must be non-negative")
        if margin_db < 0:
            raise ValueError("margin_db must be >= 0")
        self.frame_duration_ms = frame_duration_ms
        self.hop_duration_ms = hop_duration_ms
        self.vowel_scale = float(vowel_scale)
        self.consonant_scale = float(consonant_scale)
        self.margin_db = float(margin_db)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        """Apply phoneme-aware perturbation to the signal."""
        rng = np.random.default_rng(self.seed)
        sr = signal.sample_rate
        data = signal.to_mono().waveform.copy()

        frame_len = max(1, int(round(self.frame_duration_ms * sr / 1000.0)))
        hop_len = max(1, int(round(self.hop_duration_ms * sr / 1000.0)))
        n = len(data)

        if n < frame_len:
            # Signal shorter than one frame — fall back to identity
            return signal.clone()

        # Compute per-frame RMS for phoneme classification
        n_frames = max(1, 1 + (n - frame_len) // hop_len)
        frame_rms = np.zeros(n_frames, dtype=np.float64)
        for fi in range(n_frames):
            start = fi * hop_len
            frame_rms[fi] = float(np.sqrt(np.mean(data[start:start + frame_len] ** 2)))

        rms_median = float(np.median(frame_rms))

        # Overlap-add output accumulator
        output = np.zeros(n, dtype=np.float64)
        norm_acc = np.zeros(n, dtype=np.float64)
        window = np.hanning(frame_len)

        for fi in range(n_frames):
            start = fi * hop_len
            end = min(start + frame_len, n)
            actual_len = end - start
            frame = np.zeros(frame_len, dtype=np.float64)
            frame[:actual_len] = data[start:end]

            # Classify frame
            is_vowel = frame_rms[fi] > rms_median
            scale = self.vowel_scale if is_vowel else self.consonant_scale

            # Psychoacoustic budget
            budget = _safe_masking_budget(frame, sr, self.margin_db)

            # Spectral perturbation
            spec = np.fft.rfft(frame)
            n_bins = len(spec)
            noise_phase = rng.uniform(-np.pi, np.pi, size=n_bins)
            noise_spec = scale * budget * np.exp(1j * noise_phase)
            perturbed = np.fft.irfft(spec + noise_spec, n=frame_len)

            # Overlap-add with Hanning window
            win_frame = perturbed * window
            output[start:end] += win_frame[:actual_len]
            norm_acc[start:end] += window[:actual_len]

        # Normalize by overlap-add window accumulation
        nonzero = norm_acc > 1e-12
        output[nonzero] /= norm_acc[nonzero]

        # NaN/Inf guard
        output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return Signal(
            waveform=output,
            sample_rate=sr,
            metadata={**signal.metadata, "transform": self.name},
        )


class PhonemeSegmentDropout(BaseTransform):
    """Energy-segmented phoneme dropout transform.

    Segments the waveform into overlapping analysis frames, classifies each
    frame as low-energy (fricative/silence-like) or high-energy (vowel-like),
    and stochastically zeros out frames — with higher dropout probability for
    low-energy regions. This targets fricative and silence regions where ASR
    models depend on subtle spectral cues.

    Parameters
    ----------
    drop_prob : float
        Base dropout probability. Default 0.15 (15% of frames).
    frame_duration_ms : float
        Analysis frame length in ms. Default 20 ms.
    hop_duration_ms : float
        Frame hop size in ms. Default 10 ms.
    seed : int or None
        Random seed.
    name : str
        Transform identifier.
    """

    def __init__(
        self,
        drop_prob: float = 0.15,
        frame_duration_ms: float = 20.0,
        hop_duration_ms: float = 10.0,
        seed: int | None = None,
        name: str = "novel.phoneme_segment_dropout",
    ) -> None:
        super().__init__(name=name)
        if not (0.0 <= drop_prob <= 1.0):
            raise ValueError("drop_prob must be in [0, 1]")
        if frame_duration_ms <= 0 or hop_duration_ms <= 0:
            raise ValueError("Durations must be > 0")
        self.drop_prob = float(drop_prob)
        self.frame_duration_ms = frame_duration_ms
        self.hop_duration_ms = hop_duration_ms
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        """Apply phoneme segment dropout."""
        rng = np.random.default_rng(self.seed)
        sr = signal.sample_rate
        data = signal.to_mono().waveform.copy()
        n = len(data)

        frame_len = max(1, int(round(self.frame_duration_ms * sr / 1000.0)))
        hop_len = max(1, int(round(self.hop_duration_ms * sr / 1000.0)))

        if n < frame_len:
            return signal.clone()

        # Per-frame RMS for classification
        n_frames = max(1, 1 + (n - frame_len) // hop_len)
        frame_rms = np.zeros(n_frames, dtype=np.float64)
        for fi in range(n_frames):
            start = fi * hop_len
            frame_rms[fi] = float(np.sqrt(np.mean(data[start:start + frame_len] ** 2)))

        rms_median = float(np.median(frame_rms))

        # Overlap-add output
        output = np.zeros(n, dtype=np.float64)
        norm_acc = np.zeros(n, dtype=np.float64)
        window = np.hanning(frame_len)

        for fi in range(n_frames):
            start = fi * hop_len
            end = min(start + frame_len, n)
            actual_len = end - start
            frame = data[start:end]

            is_low_energy = frame_rms[fi] <= rms_median
            # Low-energy frames get 1.5x dropout probability; high-energy get 0.5x
            effective_prob = (
                min(1.0, self.drop_prob * 1.5)
                if is_low_energy
                else self.drop_prob * 0.5
            )
            frame_out = np.zeros(actual_len) if rng.random() < effective_prob else frame.copy()

            output[start:end] += frame_out * window[:actual_len]
            norm_acc[start:end] += window[:actual_len]

        nonzero = norm_acc > 1e-12
        output[nonzero] /= norm_acc[nonzero]
        output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return Signal(
            waveform=output,
            sample_rate=sr,
            metadata={**signal.metadata, "transform": self.name},
        )
