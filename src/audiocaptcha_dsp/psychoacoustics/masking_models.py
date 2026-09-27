"""
Psychoacoustic Masking Models
==============================
Implements simultaneous and temporal masking threshold estimation for use in
perceptual audio quality assessment and adversarial audio generation research.

The spreading-function model follows Zwicker & Fastl (1999) and is used in
psychoacoustically-informed adversarial audio attacks described by Schönherr
et al. (2018, 2019).

References
----------
- Zwicker, E. & Fastl, H. (1999). *Psychoacoustics: Facts and Models*, 2nd ed.
  Springer. ISBN 978-3-540-65063-5.
- Schönherr, L., Kohls, K., Zeiler, S., Holz, T., & Kolossa, D. (2018).
  "Adversarial attacks against automatic speech recognition systems via
  psychoacoustic hiding." arXiv:1808.05665.
- Schönherr, L., Zeiler, S., Kolossa, D., et al. (2019). "Imperio: Robust
  over-the-air adversarial examples for automatic speech recognition systems."
  arXiv:1901.11167.
- Painter, T. & Spanias, A. (2000). "Perceptual coding of digital audio."
  *Proceedings of the IEEE*, 88(4), 451–515.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import get_window

from audiocaptcha_dsp.core.types import MaskingModel
from audiocaptcha_dsp.psychoacoustics.bark_scale import hz_to_bark
from audiocaptcha_dsp.psychoacoustics.thresholds import absolute_threshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rfft_power_spectrum(signal: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    """Return one-sided power spectrum via rfft."""
    if n_fft is None:
        n_fft = len(signal)
    spectrum = np.fft.rfft(signal, n=n_fft)
    power = np.abs(spectrum) ** 2
    return power


def _rfft_freqs(n_fft: int, sr: int) -> np.ndarray:
    """Return frequency axis for np.fft.rfft output."""
    return np.fft.rfftfreq(n_fft, d=1.0 / sr)


def _bark_power_bands(power: np.ndarray, freqs: np.ndarray, n_bark: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate FFT power into Bark-scale critical bands.

    Parameters
    ----------
    power : np.ndarray, shape (N,)
        One-sided power spectrum.
    freqs : np.ndarray, shape (N,)
        Frequency axis in Hz.
    n_bark : int
        Number of Bark bands (24 covers the audible range 0–24 Bark).

    Returns
    -------
    bark_power : np.ndarray, shape (n_bark,)
        Summed power per Bark band.
    bark_centers : np.ndarray, shape (n_bark,)
        Centre frequency of each Bark band in Bark.
    """
    bark_centers = np.arange(0.5, n_bark + 0.5)  # 0.5, 1.5, ..., 23.5
    bark_freqs = hz_to_bark(freqs)
    bark_power = np.zeros(n_bark, dtype=np.float64)
    for i, bc in enumerate(bark_centers):
        mask = (bark_freqs >= bc - 0.5) & (bark_freqs < bc + 0.5)
        bark_power[i] = power[mask].sum() if mask.any() else 0.0
    return bark_power, bark_centers


def _spreading_function(bark_centers: np.ndarray, bark_power_db: np.ndarray) -> np.ndarray:
    """Apply Zwicker-style triangular spreading function in the Bark domain.

    The spreading (excitation) pattern of a masker at Bark position z_m is
    approximated by a triangular function:

    * Below the masker (z < z_m):  slope = −27 dB / Bark
    * Above the masker (z > z_m):  slope = −(27 + 0.37 · max(L_m − 40, 0)) dB / Bark
      simplified here to −5 dB / Bark (conservative, per Schönherr 2018).

    Parameters
    ----------
    bark_centers : np.ndarray, shape (B,)
        Bark centre positions of each band.
    bark_power_db : np.ndarray, shape (B,)
        Power of each Bark band in dB.

    Returns
    -------
    spread_db : np.ndarray, shape (B,)
        Spread masking threshold (dB) at each Bark position, taking the
        maximum contribution over all maskers.
    """
    n_bands = len(bark_centers)
    spread_db = np.full(n_bands, -np.inf)

    for m in range(n_bands):
        lm = bark_power_db[m]
        for z in range(n_bands):
            dz = bark_centers[z] - bark_centers[m]
            if dz < 0:
                # Below masker: steep lower skirt
                contribution = lm + 27.0 * dz  # dz negative → decreases
            elif dz == 0:
                contribution = lm
            else:
                # Above masker: shallower upper skirt (simplified)
                contribution = lm - 5.0 * dz
            spread_db[z] = max(spread_db[z], contribution)

    return spread_db


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simultaneous_masking_threshold(
    signal: np.ndarray,
    sr: int,
    model: MaskingModel | str = MaskingModel.SIMULTANEOUS,
) -> np.ndarray:
    """Compute the simultaneous (spectro-temporal) masking threshold.

    Implements the Zwicker critical-band masking model with a triangular
    spreading function.  The threshold is computed in the Bark domain and
    then interpolated back to the FFT frequency grid.

    Algorithm
    ---------
    1. Compute one-sided power spectrum via ``np.fft.rfft``.
    2. Accumulate power into 24 Bark critical bands.
    3. Convert to dB; apply the triangular spreading function
       (−27 dB/Bark below, −5 dB/Bark above the masker).
    4. Add absolute threshold floor (ISO 226 ATH).
    5. Interpolate back to the FFT frequency grid.

    Parameters
    ----------
    signal : array_like, shape (N,)
        Time-domain waveform (mono, float).  Silence (all-zero) is handled
        gracefully and returns the ATH.
    sr : int
        Sample rate in Hz. Must be positive.
    model : MaskingModel or str, optional
        Masking model variant.  Only ``'simultaneous'`` is fully implemented;
        ``'temporal'`` and ``'combined'`` forward to the appropriate dedicated
        functions.

    Returns
    -------
    np.ndarray, shape (N//2 + 1,)
        Masking threshold in dB SPL for each rfft frequency bin.  All values
        are finite and ≥ ATH at the corresponding frequency.

    Raises
    ------
    ValueError
        If *signal* is not 1-D, or if *sr* ≤ 0.
    ValueError
        If *signal* contains NaN or Inf.

    Notes
    -----
    The spreading function slopes follow Schönherr et al. (2018) Section 3:
    lower skirt −27 dB/Bark, upper skirt −5 dB/Bark (conservative).  A
    safety clip at the ATH ensures the returned threshold never falls below
    the absolute threshold.

    References
    ----------
    Zwicker, E. & Fastl, H. (1999). *Psychoacoustics: Facts and Models*.
    Schönherr et al. (2018). arXiv:1808.05665.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}.")
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}.")
    if np.any(~np.isfinite(signal)):
        raise ValueError("signal contains NaN or Inf values.")

    model = MaskingModel(model) if isinstance(model, str) else model

    n_fft = len(signal)
    freqs = _rfft_freqs(n_fft, sr)
    power = _rfft_power_spectrum(signal, n_fft)

    # --- ATH floor ---
    ath_db = absolute_threshold(freqs)

    if np.all(signal == 0.0):
        # Silence: return ATH directly
        return ath_db.copy()

    # --- Bark-domain spreading ---
    bark_power, bark_centers = _bark_power_bands(power, freqs, n_bark=24)

    eps = np.finfo(np.float64).tiny
    bark_power_db = 10.0 * np.log10(bark_power + eps)

    spread_db = _spreading_function(bark_centers, bark_power_db)

    # --- Interpolate Bark-domain threshold to FFT bins ---
    # Map FFT freqs to Bark, then interpolate the spread_db curve
    fft_bark = hz_to_bark(np.clip(freqs, 1.0, None))  # avoid bark(0)
    # bark_centers are 0.5, 1.5, ..., 23.5
    # spread_db holds *band-integrated* levels (Zwicker masking pattern).
    # A per-bin threshold must be the band's power distributed across the
    # FFT bins inside that band (i.e. a power density); otherwise every bin
    # of a wide Bark band would receive the whole band's power and any
    # budget derived from this threshold would exceed the signal itself.
    n_bands = len(bark_centers)
    band_idx = np.clip(np.floor(fft_bark).astype(int), 0, n_bands - 1)
    band_bins = np.bincount(band_idx, minlength=n_bands).astype(np.float64)
    spread_per_bin_db = spread_db - 10.0 * np.log10(np.maximum(band_bins, 1.0))
    masking_db = np.interp(fft_bark, bark_centers, spread_per_bin_db)

    # --- Enforce ATH floor ---
    masking_db = np.maximum(masking_db, ath_db)

    # --- Sanity: replace non-finite with ATH ---
    if not np.all(np.isfinite(masking_db)):
        warnings.warn(
            "Non-finite masking threshold values replaced with ATH.",
            RuntimeWarning,
            stacklevel=2,
        )
        masking_db = np.where(np.isfinite(masking_db), masking_db, ath_db)

    return masking_db


def temporal_masking_threshold(
    signal: np.ndarray,
    sr: int,
    frame_duration: float = 0.025,
    hop_duration: float = 0.010,
) -> np.ndarray:
    """Compute per-frame forward temporal masking thresholds.

    Forward masking occurs when a masker raises the threshold of a
    subsequent sound.  Duration is typically up to ~150 ms with an
    exponential decay governed by:

    .. math::

        T_{\\text{fwd}}(t) = T_0 \\cdot 2^{-t / t_{\\mathrm{half}}}

    where :math:`t_{\\mathrm{half}} \\approx 10\\,\\text{ms}` (corresponding
    to −12 dB per doubling of time after masker offset), and :math:`T_0` is
    the simultaneous masking threshold at the masker frame.

    Parameters
    ----------
    signal : array_like, shape (N,)
        Mono time-domain waveform.
    sr : int
        Sample rate in Hz.
    frame_duration : float, optional
        Analysis frame length in seconds. Default 25 ms.
    hop_duration : float, optional
        Frame hop size in seconds. Default 10 ms.

    Returns
    -------
    np.ndarray, shape (n_frames, n_fft_bins)
        Forward temporal masking threshold in dB SPL for each frame and
        rfft frequency bin.  Each frame's threshold is the maximum of its
        own simultaneous masking threshold and the decayed threshold
        propagated from all preceding frames within 150 ms.

    Raises
    ------
    ValueError
        If *signal* is not 1-D, if *sr* ≤ 0, or if durations are non-positive.
    ValueError
        If *signal* contains NaN or Inf.

    Notes
    -----
    Forward masking parameters follow:
    - Zwicker & Fastl (1999), Chapter 4.
    - Moore (2012), "An Introduction to the Psychology of Hearing", Ch. 3.
    - Schönherr et al. (2018) arXiv:1808.05665 employ a similar model for
      psychoacoustically-constrained adversarial audio.

    The backward masking component is intentionally omitted (it is very
    brief, ~5 ms, and practically irrelevant for most applications).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}.")
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}.")
    if frame_duration <= 0:
        raise ValueError(f"frame_duration must be positive, got {frame_duration}.")
    if hop_duration <= 0:
        raise ValueError(f"hop_duration must be positive, got {hop_duration}.")
    if np.any(~np.isfinite(signal)):
        raise ValueError("signal contains NaN or Inf values.")

    frame_len = int(round(frame_duration * sr))
    hop_len = int(round(hop_duration * sr))
    frame_len = max(frame_len, 1)
    hop_len = max(hop_len, 1)

    # Pad signal so that all samples are covered
    n_frames = max(1, 1 + (len(signal) - frame_len + hop_len - 1) // hop_len)
    pad_len = (n_frames - 1) * hop_len + frame_len - len(signal)
    if pad_len > 0:
        signal = np.pad(signal, (0, pad_len), mode="constant")

    n_fft = frame_len
    n_bins = n_fft // 2 + 1

    # Compute per-frame simultaneous masking thresholds
    sim_thresh = np.zeros((n_frames, n_bins), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        frame = signal[start: start + frame_len]
        sim_thresh[i] = simultaneous_masking_threshold(frame, sr)

    # Forward temporal masking: exponential decay
    # Half-life ≈ 10 ms → -12 dB per doubling of time
    t_half_s = 0.010  # 10 ms
    max_forward_ms = 0.150  # 150 ms
    max_forward_frames = int(np.ceil(max_forward_ms / hop_duration))

    temporal_thresh = sim_thresh.copy()
    for i in range(1, n_frames):
        for lag in range(1, min(i, max_forward_frames) + 1):
            elapsed_s = lag * hop_duration
            # Exponential decay: −12 dB per doubling of time after masker offset
            # (Zwicker & Fastl 1999, Ch. 4; t_half ≈ 10 ms)
            with np.errstate(divide="ignore"):
                decay_db = -12.0 * np.log2(elapsed_s / t_half_s + 1e-12)
            decayed = sim_thresh[i - lag] + decay_db
            temporal_thresh[i] = np.maximum(temporal_thresh[i], decayed)

    return temporal_thresh


def compute_masking_budget(
    signal: np.ndarray,
    sr: int,
    margin_db: float = 6.0,
) -> np.ndarray:
    """Compute the psychoacoustic energy budget for imperceptible perturbation.

    Returns the per-frequency-bin power budget (linear, not dB) below which
    an additive perturbation would lie beneath the simultaneous masking
    threshold — and therefore be psychoacoustically inaudible.

    The budget is computed as:

    .. math::

        B[k] = 10^{(M[k] - \\Delta) / 10}

    where :math:`M[k]` is the simultaneous masking threshold in dB at bin
    *k* and :math:`\\Delta` is the safety margin in dB.

    Parameters
    ----------
    signal : array_like, shape (N,)
        Mono time-domain waveform.
    sr : int
        Sample rate in Hz.
    margin_db : float, optional
        Safety margin in dB below the masking threshold. Default 6 dB
        (i.e., perturbation is at most half the masked power). Must be
        non-negative.

    Returns
    -------
    np.ndarray, shape (N//2 + 1,)
        Maximum perturbation power (linear) per rfft frequency bin that
        would remain psychoacoustically masked.  All values are positive
        and finite.

    Raises
    ------
    ValueError
        If *margin_db* < 0, or if *signal*/*sr* are invalid.

    Notes
    -----
    This budget corresponds to the "PsychoacousticModel" used in
    Schönherr et al. (2018) arXiv:1808.05665 and the Carlini & Wagner (2018)
    audio attack adapted version (imperceptible CW attack).

    References
    ----------
    Schönherr, L. et al. (2018). arXiv:1808.05665.
    Zwicker, E. & Fastl, H. (1999). *Psychoacoustics: Facts and Models*.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if margin_db < 0:
        raise ValueError(f"margin_db must be non-negative, got {margin_db}.")

    masking_db = simultaneous_masking_threshold(signal, sr)
    budget_db = masking_db - margin_db
    budget_linear = 10.0 ** (budget_db / 10.0)

    # Sanity: ensure all values are finite and positive
    if not np.all(np.isfinite(budget_linear)):
        warnings.warn(
            "Non-finite masking budget values detected; clamping to small positive.",
            RuntimeWarning,
            stacklevel=2,
        )
        budget_linear = np.where(
            np.isfinite(budget_linear), budget_linear, np.finfo(np.float64).tiny
        )
    budget_linear = np.clip(budget_linear, np.finfo(np.float64).tiny, None)
    return budget_linear


def compute_perturbation_budget(
    signal: np.ndarray,
    sr: int,
    margin_db: float = 6.0,
) -> np.ndarray:
    """Signal-referenced perturbation budget below the masking threshold.

    Identical to :func:`compute_masking_budget`, except that frequency bins
    whose threshold is governed *solely* by the absolute threshold of
    hearing (ATH) — i.e. regions where the signal itself provides no
    masking — receive a budget of zero.

    Rationale
    ---------
    The ATH is expressed in dB SPL, while the masker-spread part of
    :func:`simultaneous_masking_threshold` is in FFT-bin power units of the
    input signal.  Without a playback-level reference the two cannot be
    mixed: the raw ATH value (up to 120 dB SPL below 20 Hz) would be
    misread as a per-bin power budget of 10¹¹ and dominate the injected
    energy.  The conservative, unit-safe choice is therefore to inject
    perturbation only where the signal's own masking spread authorizes it
    (the hearing-threshold constraint of Schönherr et al. 2018 with λ as
    the safety margin still applies to every remaining bin).

    Parameters
    ----------
    signal : array_like, shape (N,)
        Mono time-domain waveform.
    sr : int
        Sample rate in Hz.
    margin_db : float, optional
        Safety margin λ below the masking threshold (default 6 dB).

    Returns
    -------
    np.ndarray, shape (N//2 + 1,)
        Maximum perturbation power (linear) per rfft bin; bins with no
        signal-provided masking receive exactly 0.
    """
    budget = compute_masking_budget(signal, sr, margin_db=margin_db)
    signal = np.asarray(signal, dtype=np.float64)
    freqs = _rfft_freqs(len(signal), sr)
    ath_db = absolute_threshold(freqs)
    # Reconstruct the threshold from the budget: M = 10*log10(B) + λ.
    # M is floored at the ATH inside simultaneous_masking_threshold, so
    # M ≈ ATH ⟺ the ATH governs (masker spread is below the floor there).
    m_db = 10.0 * np.log10(budget) + margin_db
    quiet = m_db <= ath_db + 1e-6
    return np.where(quiet, 0.0, budget)
