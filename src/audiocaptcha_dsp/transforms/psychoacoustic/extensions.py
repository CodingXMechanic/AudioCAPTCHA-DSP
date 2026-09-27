"""
Family F extensions: ERB/temporal/combined masking and margin-controlled perturbation.
=======================================================================================
WHAT-REMAINS.txt §3.F gaps implemented here:

- ERB-scale masking (Glasberg & Moore ERB-rate bands)
- Temporal masking: forward (and practical backward) masking budgets
- Combined temporal + spectral masking budgets
- Signal-dependent masking thresholds with explicit margin λ controls
- Per-frame masking budgets (per-frame allocation)
- Loudness-preserving perturbation (short-time loudness envelope matched)
- Speech-aware masking (voiced/unvoiced gated budgets)

Terminology: all transforms here are described as *psychoacoustically
constrained* — never as "imperceptible" — pending human listening
validation (WHAT-REMAINS.txt §3.F).

References
----------
- Glasberg & Moore (1990). Derivation of auditory filter shapes from
  notched-noise data (ERB rate). Hearing Research.
- Moore & Glasberg (1989). Suggested formulae for deriving loudness-related
  measures; masking temporal dynamics.
- Zwicker & Fastl (1999). Psychoacoustics: Facts and Models, Ch. 4.
- Schönherr et al. (2018). arXiv:1808.05665 — hearing-threshold-constrained
  perturbation (λ margin), MP3 model.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform
from audiocaptcha_dsp.psychoacoustics.masking_models import (
    simultaneous_masking_threshold,
    temporal_masking_threshold,
)

try:
    from audiocaptcha_dsp.psychoacoustics.masking_models import (
        compute_masking_budget,
        compute_perturbation_budget,
    )
except ImportError:  # pragma: no cover
    def compute_masking_budget(signal, sr, margin_db=6.0):
        mag = np.abs(np.fft.rfft(np.asarray(signal, dtype=np.float64)))
        return (10.0 ** ((20 * np.log10(mag + 1e-12) - 30.0 - margin_db) / 10.0))

    # Fallback is already in signal (FFT-bin) units — ATH mixing not present
    compute_perturbation_budget = compute_masking_budget


def _safe(y: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _hz_to_erb_rate(freq_hz: np.ndarray) -> np.ndarray:
    """Glasberg & Moore (1990) ERB-rate: ERB_N = 21.4 log10(1 + 0.00437 f)."""
    freq_hz = np.clip(np.asarray(freq_hz, dtype=np.float64), 0.0, None)
    return 21.4 * np.log10(1.0 + 0.00437 * freq_hz)


def _erb_band_budget(
    x: np.ndarray,
    sr: int,
    margin_db: float,
    n_bands: int = 32,
) -> np.ndarray:
    """Per-rfft-bin perturbation power budget derived from ERB-rate bands.

    1. Accumulate power into ``n_bands`` rectangular ERB-rate bands
       (raw ``|rfft|**2`` units, i.e. the same reference as the spectrum
       the consumer perturbs).
    2. Apply a symmetric spreading function in ERB space (−27 / −5 dB/ERB).
    3. Distribute each band's threshold power over the rfft bins inside the
       band (band-integrated → per-bin), then enforce a tiny positive floor
       so the budget stays finite and strictly positive.
    """
    n_fft = len(x)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    power = np.abs(np.fft.rfft(x)) ** 2

    erb = _hz_to_erb_rate(freqs)
    edges = np.linspace(erb.min(), erb.max(), n_bands + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    band_power = np.zeros(n_bands)
    idx = np.clip(np.digitize(erb, edges) - 1, 0, n_bands - 1)
    np.add.at(band_power, idx, power)
    bin_counts = np.maximum(np.bincount(idx, minlength=n_bands), 1)

    eps = np.finfo(float).tiny
    band_db = 10.0 * np.log10(band_power + eps)

    # Symmetric spreading in ERB space (upper skirt shallower, per Zwicker)
    spread_db = band_db.copy()
    for i in range(n_bands):
        for j in range(n_bands):
            d = centers[j] - centers[i]
            if d < 0:
                slope = -27.0
            else:
                slope = -5.0
            spread_db[j] = np.maximum(spread_db[j], band_db[i] + slope * abs(d))

    # Margin + tiny positive floor (bands with no masker contribution stay
    # effectively zero; a −2 dB-style floor would inject ~0.6 power per bin)
    thresh_db = spread_db - margin_db
    thresh_db = np.maximum(thresh_db, -3000.0)

    band_thresh_power = 10.0 ** (thresh_db / 10.0)
    # Band-integrated threshold → per-bin budget
    budget = band_thresh_power[idx] / bin_counts[idx]
    return budget


class ERBScalePerturbation(BaseTransform):
    """Perturbation allocated along an ERB-rate masking budget.

    Equivalent of :class:`BarkScalePerturbation` on the Glasberg–Moore
    ERB-rate scale, which better matches auditory-filter bandwidths at
    high frequencies (Glasberg & Moore 1990).
    """

    def __init__(
        self,
        perturbation_scale: float = 0.1,
        margin_db: float = 6.0,
        n_bands: int = 32,
        name: str = "psychoacoustic.erb_perturbation",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if perturbation_scale < 0 or margin_db < 0:
            raise ValueError("perturbation_scale and margin_db must be >= 0")
        if n_bands < 4:
            raise ValueError("n_bands must be >= 4")
        self.perturbation_scale = perturbation_scale
        self.margin_db = margin_db
        self.n_bands = n_bands
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        if x.ndim != 1:
            x = x.mean(axis=0)
        budget = _erb_band_budget(x, signal.sample_rate, self.margin_db, self.n_bands)
        spec = np.fft.rfft(x)
        if len(budget) != len(spec):
            n = min(len(budget), len(spec))
            tmp = np.zeros(len(spec), dtype=np.float64)
            tmp[:n] = budget[:n]
            budget = tmp
        amp = np.sqrt(np.maximum(budget, 0.0)) * self.perturbation_scale
        phase = rng.uniform(-np.pi, np.pi, len(spec))
        y = np.fft.irfft(spec + amp * np.exp(1j * phase), n=len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata,
                      f"{self.name}.scale": self.perturbation_scale,
                      f"{self.name}.margin_db": self.margin_db},
        )


class TemporalMaskingPerturbation(BaseTransform):
    """Forward (and short backward) temporal-masking constrained perturbation.

    Uses the per-frame forward temporal masking threshold
    (:func:`temporal_masking_threshold`) as a per-frame spectral budget:
    each STFT frame receives random-phase noise whose magnitude never
    exceeds the frame's temporal threshold minus ``margin_db``. Frames
    immediately after loud maskers receive a higher allowed budget than
    quiet frames (Moore & Glasberg; Zwicker & Fastl 1999).
    """

    def __init__(
        self,
        margin_db: float = 6.0,
        frame_duration: float = 0.025,
        hop_duration: float = 0.010,
        name: str = "psychoacoustic.temporal_masking",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if margin_db < 0:
            raise ValueError("margin_db must be >= 0")
        if frame_duration <= 0 or hop_duration <= 0:
            raise ValueError("frame and hop durations must be > 0")
        self.margin_db = margin_db
        self.frame_duration = frame_duration
        self.hop_duration = hop_duration
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        if x.ndim != 1:
            x = x.mean(axis=0)
        sr = signal.sample_rate

        # Per-frame temporal masking thresholds in dB SPL
        thresh_db = temporal_masking_threshold(
            x, sr, frame_duration=self.frame_duration, hop_duration=self.hop_duration
        )  # (n_frames, n_bins)

        frame_len = int(round(self.frame_duration * sr))
        hop_len = int(round(self.hop_duration * sr))
        n_fft = frame_len
        win = np.hanning(n_fft)

        # Frame the signal consistently with the threshold computation
        pad = max(0, (thresh_db.shape[0] - 1) * hop_len + n_fft - len(x))
        x_pad = np.pad(x, (0, pad))
        n_frames = thresh_db.shape[0]

        out = np.zeros_like(x_pad)
        wsum = np.zeros_like(x_pad)
        for i in range(n_frames):
            start = i * hop_len
            frame = x_pad[start: start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spec = np.fft.rfft(frame)
            mag = np.abs(spec)
            # Per-bin allowed power from temporal threshold (dB SPL proxy)
            allowed_db = thresh_db[i] - self.margin_db
            # Reference the threshold against the frame's own spectrum peak so
            # that the dB-SPL proxy behaves as a relative constraint.
            frame_ref_db = 20.0 * np.log10(mag.max() + 1e-12)
            allowed_mag = 10.0 ** ((allowed_db - frame_ref_db) / 20.0)
            allowed_mag = np.minimum(allowed_mag, mag * 10.0 ** (-self.margin_db / 20.0))
            allowed_mag = np.maximum(allowed_mag, 0.0)
            phase = rng.uniform(-np.pi, np.pi, len(spec))
            noisy = (mag + allowed_mag * np.exp(1j * phase)) * np.exp(1j * np.angle(spec))
            y_frame = np.fft.irfft(noisy, n=n_fft)
            out[start: start + n_fft] += y_frame * win
            wsum[start: start + n_fft] += win

        wsum[wsum < 1e-8] = 1.0
        y = out / wsum
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)]
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.margin_db": self.margin_db},
        )


class CombinedTemporalSpectralMasking(BaseTransform):
    """Joint temporal + spectral masking budget (min of both thresholds).

    Per-frame perturbation power budget = min(simultaneous threshold,
    forward-temporal threshold) − margin: the strictest of the two
    constraints, as in MP3's two-stage psychoacoustic stage.
    """

    def __init__(
        self,
        margin_db: float = 6.0,
        name: str = "psychoacoustic.combined_masking",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if margin_db < 0:
            raise ValueError("margin_db must be >= 0")
        self.margin_db = margin_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        if x.ndim != 1:
            x = x.mean(axis=0)
        sr = signal.sample_rate

        spec = np.fft.rfft(x)
        mag = np.abs(spec)

        # Whole-signal simultaneous threshold (dB), aligned to rfft grid
        sim_db = simultaneous_masking_threshold(x, sr)
        if len(sim_db) != len(spec):
            n = min(len(sim_db), len(spec))
            tmp = np.full(len(spec), 60.0)
            tmp[:n] = sim_db[:n]
            sim_db = tmp

        # Frame-wise forward temporal threshold: use max over time (conservative)
        temporal_db = temporal_masking_threshold(x, sr)
        temp_max_db = temporal_db.max(axis=0)
        if len(temp_max_db) != len(spec):
            n = min(len(temp_max_db), len(spec))
            tmp = np.full(len(spec), 60.0)
            tmp[:n] = temp_max_db[:n]
            temp_max_db = tmp

        joint_db = np.minimum(sim_db, temp_max_db) - self.margin_db
        frame_ref_db = 20.0 * np.log10(mag.max() + 1e-12)
        allowed_mag = 10.0 ** ((joint_db - frame_ref_db) / 20.0)
        allowed_mag = np.clip(allowed_mag, 0.0, mag * 10.0 ** (-self.margin_db / 20.0))

        phase = rng.uniform(-np.pi, np.pi, len(spec))
        y = np.fft.irfft(mag * np.exp(1j * np.angle(spec)) + allowed_mag * np.exp(1j * phase), n=len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.margin_db": self.margin_db},
        )


class LoudnessPreservingPerturbation(BaseTransform):
    """Add masking-budget noise while preserving the short-time loudness.

    After perturbation, the short-time loudness envelope (A-weighted RMS
    proxy, 20 ms frames / 10 ms hop) is rescaled to match the original
    envelope, so overall and dynamic loudness remain stable while the
    spectral fine structure changes.
    """

    def __init__(
        self,
        margin_db: float = 6.0,
        perturbation_scale: float = 1.0,
        name: str = "psychoacoustic.loudness_preserving",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if margin_db < 0 or perturbation_scale < 0:
            raise ValueError("margin_db and perturbation_scale must be >= 0")
        self.margin_db = margin_db
        self.perturbation_scale = perturbation_scale
        self.seed = seed

    @staticmethod
    def _envelope(x: np.ndarray, sr: int) -> np.ndarray:
        frame = max(1, int(0.020 * sr))
        hop = max(1, int(0.010 * sr))
        n_frames = max(1, 1 + (len(x) - frame) // hop)
        env = np.empty(n_frames)
        for i in range(n_frames):
            seg = x[i * hop: i * hop + frame]
            env[i] = np.sqrt(np.mean(seg ** 2) + 1e-12)
        return env

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        if x.ndim != 1:
            x = x.mean(axis=0)
        sr = signal.sample_rate
        spec = np.fft.rfft(x)
        budget = compute_perturbation_budget(x, sr, margin_db=self.margin_db)
        budget = np.asarray(budget, dtype=np.float64)
        if len(budget) != len(spec):
            n = min(len(budget), len(spec))
            tmp = np.zeros(len(spec), dtype=np.float64)
            tmp[:n] = budget[:n]
            budget = tmp
        amp = np.sqrt(np.maximum(budget, 0.0)) * self.perturbation_scale
        phase = rng.uniform(-np.pi, np.pi, len(spec))
        y = np.fft.irfft(spec + amp * np.exp(1j * phase), n=len(x))

        # Match short-time loudness envelope to the original
        env_x = self._envelope(x, sr)
        env_y = self._envelope(y, sr)
        hop = max(1, int(0.010 * sr))
        # Wide clip: in AM troughs the input envelope is near zero while the
        # perturbation keeps a floor, so matching demands large attenuations;
        # bounds only guard against division blow-ups.
        gains = np.clip(env_x / (env_y + 1e-12), 1e-2, 1e2)
        # Continuous hop-rate gain curve: piecewise-constant 10 ms gains
        # would step discontinuously at frame boundaries (clicks).
        centers = np.arange(len(gains)) * hop + hop / 2.0
        gain_curve = np.interp(
            np.arange(len(y)), centers, gains,
            left=gains[0], right=gains[-1],
        )
        y = y * gain_curve
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata,
                      f"{self.name}.margin_db": self.margin_db,
                      f"{self.name}.scale": self.perturbation_scale},
        )


class PerFrameMaskingBudget(BaseTransform):
    """Per-frame masking budget allocation with an explicit allocation exponent.

    Each STFT frame receives a budget derived from its own energy
    (loud frames mask more); ``allocation_exponent`` skews the budget
    toward loud (p>1) or quiet (p<1) frames — a per-frame masking-budget
    control (WHAT-REMAINS.txt §3.F "per-frame masking budgets").
    """

    def __init__(
        self,
        margin_db: float = 6.0,
        allocation_exponent: float = 1.0,
        name: str = "psychoacoustic.frame_budget",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if margin_db < 0:
            raise ValueError("margin_db must be >= 0")
        if allocation_exponent <= 0:
            raise ValueError("allocation_exponent must be > 0")
        self.margin_db = margin_db
        self.allocation_exponent = allocation_exponent
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        if x.ndim != 1:
            x = x.mean(axis=0)
        sr = signal.sample_rate
        n_fft, hop = 1024, 512
        if len(x) < n_fft:
            return signal.clone()
        win = np.hanning(n_fft)
        n_frames = 1 + (len(x) - n_fft) // hop

        out = np.zeros(len(x) + n_fft)
        wsum = np.zeros_like(out)
        for i in range(n_frames):
            seg = x[i * hop: i * hop + n_fft] * win
            spec = np.fft.rfft(seg)
            mag = np.abs(spec)
            frame_power = np.mean(mag ** 2)
            # Per-frame budget: frame-length rfft units match spec exactly,
            # so the threshold stays in the same power reference as the frame.
            budget = compute_perturbation_budget(seg, sr, margin_db=self.margin_db)
            budget = np.asarray(budget, dtype=np.float64)
            total_power = float(np.mean(budget)) + 1e-18
            allocation = float((frame_power / (total_power + 1e-18)) ** self.allocation_exponent)
            # The budget is a hard cap: never allocate more than 100% of it
            allocation = float(np.clip(allocation, 1e-3, 1.0))
            amp = np.sqrt(np.maximum(budget, 0.0)) * allocation
            phase = rng.uniform(-np.pi, np.pi, len(spec))
            y_frame = np.fft.irfft(spec + amp * np.exp(1j * phase), n=n_fft)
            out[i * hop: i * hop + n_fft] += y_frame * win
            wsum[i * hop: i * hop + n_fft] += win ** 2

        wsum = np.maximum(wsum, 1e-3 * float(wsum.max()))
        y = out / wsum
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)]
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata,
                      f"{self.name}.margin_db": self.margin_db,
                      f"{self.name}.exponent": self.allocation_exponent},
        )


class SignalDependentThreshold(BaseTransform):
    """Enforce |D(f)| ≤ H(f) directly (Schönherr-style threshold projection).

    A random-phase perturbation is generated, its spectrum compared against
    the masking threshold H, and any bin exceeding H − margin_db is clipped
    back — the same *difference-vs-threshold* enforcement used in
    arXiv:1808.05665 Eqs. (3)–(4), with λ = ``margin_db``.
    """

    def __init__(
        self,
        margin_db: float = 10.0,
        perturbation_scale: float = 1.0,
        name: str = "psychoacoustic.signal_threshold",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if margin_db < 0 or perturbation_scale < 0:
            raise ValueError("margin_db and perturbation_scale must be >= 0")
        self.margin_db = margin_db
        self.perturbation_scale = perturbation_scale
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        if x.ndim != 1:
            x = x.mean(axis=0)
        sr = signal.sample_rate
        spec = np.fft.rfft(x)
        mag = np.abs(spec)

        # Threshold below the margin, in the signal's own FFT-bin power units.
        # Bins governed only by the ATH (no masking from the signal) receive a
        # zero budget — see compute_perturbation_budget for the SPL/units
        # rationale. 10*log10(B) == simultaneous_masking_threshold - margin_db.
        budget = np.asarray(
            compute_perturbation_budget(x, sr, margin_db=self.margin_db),
            dtype=np.float64,
        )
        if len(budget) != len(spec):
            n = min(len(budget), len(spec))
            tmp = np.zeros(len(spec), dtype=np.float64)
            tmp[:n] = budget[:n]
            budget = tmp
        thresh_db = 10.0 * np.log10(np.maximum(budget, np.finfo(np.float64).tiny))

        ref_db = 20.0 * np.log10(mag.max() + 1e-12)
        allowed_mag = 10.0 ** ((thresh_db - ref_db) / 20.0)
        allowed_mag = np.clip(allowed_mag, 0.0, None)

        # Generate candidate perturbation and project onto the feasible set
        phase = rng.uniform(-np.pi, np.pi, len(spec))
        cand = np.maximum(mag, allowed_mag) * 10.0 ** (-self.margin_db / 20.0) * phase / np.pi
        cand_mag = np.abs(cand)
        over = cand_mag > allowed_mag
        cand[over] = cand[over] * (allowed_mag[over] / (cand_mag[over] + 1e-18))
        cand *= self.perturbation_scale

        y = np.fft.irfft(spec + cand, n=len(x))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata,
                      f"{self.name}.margin_db": self.margin_db,
                      f"{self.name}.clipped_bins": int(np.count_nonzero(over))},
        )


class SpeechAwareMasking(BaseTransform):
    """Speech-aware masking: tighter budgets in voiced regions.

    Voiced frames (low zero-crossing rate + high periodicity) receive a
    ``voiced_margin_db`` budget; unvoiced frames receive the looser
    ``unvoiced_margin_db``. Speech-aware allocation exploits the higher
    masking power of stationary voiced maskers vs the greater robustness
    of fricatives (WHAT-REMAINS.txt §3.F "speech-aware masking").
    """

    def __init__(
        self,
        voiced_margin_db: float = 8.0,
        unvoiced_margin_db: float = 3.0,
        perturbation_scale: float = 0.8,
        name: str = "psychoacoustic.speech_aware",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if voiced_margin_db < 0 or unvoiced_margin_db < 0:
            raise ValueError("margins must be >= 0")
        if perturbation_scale < 0:
            raise ValueError("perturbation_scale must be >= 0")
        self.voiced_margin_db = voiced_margin_db
        self.unvoiced_margin_db = unvoiced_margin_db
        self.perturbation_scale = perturbation_scale
        self.seed = seed

    @staticmethod
    def _voiced_mask(x: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 512) -> np.ndarray:
        """Per-frame voiced/unvoiced decision via ZCR + energy ratio."""
        if len(x) < n_fft:
            return np.array([True])
        n_frames = 1 + (len(x) - n_fft) // hop
        mask = np.zeros(n_frames, dtype=bool)
        eps = 1e-12
        for i in range(n_frames):
            seg = x[i * hop: i * hop + n_fft]
            zcr = np.mean(np.abs(np.diff(np.signbit(seg).astype(int)))) + eps
            spectrum = np.abs(np.fft.rfft(seg)) ** 2
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            low = spectrum[(freqs > 0) & (freqs < 400)].sum()
            total = spectrum.sum() + eps
            rms = np.sqrt(np.mean(seg ** 2))
            mask[i] = (zcr < 0.15) and (low / total > 0.4) and (rms > 0.01 * np.max(np.abs(x) + eps))
        return mask

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        if x.ndim != 1:
            x = x.mean(axis=0)
        sr = signal.sample_rate
        n_fft, hop = 1024, 512
        if len(x) < n_fft:
            return signal.clone()
        win = np.hanning(n_fft)
        n_frames = 1 + (len(x) - n_fft) // hop
        voiced = self._voiced_mask(x, sr, n_fft, hop)

        out = np.zeros(len(x) + n_fft)
        wsum = np.zeros_like(out)
        for i in range(n_frames):
            seg = x[i * hop: i * hop + n_fft] * win
            spec = np.fft.rfft(seg)
            # Per-frame budget in frame-length rfft units (matches spec);
            # voiced frames get the tight margin, unvoiced the loose one.
            margin = self.voiced_margin_db if voiced[i] else self.unvoiced_margin_db
            budget = np.asarray(
                compute_perturbation_budget(seg, sr, margin_db=margin),
                dtype=np.float64,
            )
            amp = np.sqrt(np.maximum(budget, 0.0)) * self.perturbation_scale
            phase = rng.uniform(-np.pi, np.pi, len(spec))
            y_frame = np.fft.irfft(spec + amp * np.exp(1j * phase), n=n_fft)
            out[i * hop: i * hop + n_fft] += y_frame * win
            wsum[i * hop: i * hop + n_fft] += win ** 2

        wsum = np.maximum(wsum, 1e-3 * float(wsum.max()))
        y = out / wsum
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)]
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata,
                      f"{self.name}.voiced_margin_db": self.voiced_margin_db,
                      f"{self.name}.unvoiced_margin_db": self.unvoiced_margin_db,
                      f"{self.name}.n_voiced_frames": int(np.count_nonzero(voiced))},
        )
