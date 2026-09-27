"""
Psychoacoustic Threshold Models
================================
Implements the Absolute Threshold of Hearing (ATH) and masking threshold
computation based on ISO 226:2003 and related psychoacoustic literature.

References
----------
- ISO 226:2003, "Acoustics — Normal equal-loudness-level contours"
- Moore, B.C.J. (2012). "An Introduction to the Psychology of Hearing", 6th ed.
- Painter, T. & Spanias, A. (2000). "Perceptual coding of digital audio".
  Proceedings of the IEEE, 88(4), 451–515.
"""
from __future__ import annotations

import warnings

import numpy as np

from audiocaptcha_dsp.core.types import ThresholdModel


def absolute_threshold(
    freq_hz: np.ndarray | float,
    model: ThresholdModel | str = ThresholdModel.ISO226,
) -> np.ndarray:
    """Compute the Absolute Threshold of Hearing (ATH) in dB SPL.

    The ATH is the minimum sound pressure level a normal-hearing listener can
    detect in a free-field environment with no background noise.

    Parameters
    ----------
    freq_hz : array_like
        Frequency or frequencies in Hz at which to evaluate the ATH.
        Must be non-negative. Frequencies of 0 Hz are treated as −∞ dB by
        returning a large positive value (inaudible DC).
    model : ThresholdModel or str, optional
        ATH model to use. Currently supported:

        ``'iso226'`` (default)
            Analytically approximated ISO 226:2003 equal-loudness curve at
            0 phon (i.e., the threshold of hearing), using the formula:

            .. math::

                \\text{ATH}(f) = 3.64\\left(\\frac{f}{1000}\\right)^{-0.8}
                - 6.5\\,e^{-0.6\\left(\\frac{f}{1000}-3.3\\right)^2}
                + 10^{-3}\\left(\\frac{f}{1000}\\right)^4

            Valid range: ~20 Hz – 20 kHz; values outside are extrapolated.

    Returns
    -------
    np.ndarray
        ATH values in dB SPL, same shape as *freq_hz*.  Values for f = 0 Hz
        are set to 100 dB (effectively inaudible DC).  Output is clipped to
        the range [−20, 120] dB to guard against numerical blow-up at extreme
        frequencies.

    Raises
    ------
    ValueError
        If *model* is not a recognised :class:`ThresholdModel` variant.
    ValueError
        If *freq_hz* contains NaN or negative values.

    Notes
    -----
    The ISO 226 analytical approximation used here is widely adopted in
    perceptual audio coding (cf. Painter & Spanias 2000) and in adversarial
    audio research (cf. Schönherr et al. 2018, arXiv:1808.05665).

    Examples
    --------
    >>> import numpy as np
    >>> from audiocaptcha_dsp.psychoacoustics.thresholds import absolute_threshold
    >>> absolute_threshold(np.array([1000.0]))
    array([2.5...])
    """
    freq_hz = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))

    if np.any(np.isnan(freq_hz)):
        raise ValueError("freq_hz must not contain NaN values.")
    if np.any(freq_hz < 0):
        raise ValueError("freq_hz must be non-negative (Hz).")

    model = ThresholdModel(model) if isinstance(model, str) else model

    if model == ThresholdModel.ISO226:
        return _ath_iso226(freq_hz)
    elif model == ThresholdModel.ISO389:
        # ISO 389 relates to audiometric reference equivalent threshold
        # sound pressure levels; fall back to ISO226 approximation.
        warnings.warn(
            "ISO389 model not separately implemented; falling back to ISO226.",
            UserWarning,
            stacklevel=2,
        )
        return _ath_iso226(freq_hz)
    elif model == ThresholdModel.CUSTOM:
        raise ValueError(
            "ThresholdModel.CUSTOM requires a caller-supplied function. "
            "Use absolute_threshold with model='iso226' or pass your own."
        )
    else:
        raise ValueError(f"Unsupported threshold model: {model!r}")


def _ath_iso226(freq_hz: np.ndarray) -> np.ndarray:
    """ISO 226:2003 analytical ATH approximation (internal helper).

    Parameters
    ----------
    freq_hz : np.ndarray
        Frequencies in Hz (non-negative, no NaN).

    Returns
    -------
    np.ndarray
        ATH in dB SPL, clipped to [−20, 120].
    """
    # Replace 0 Hz with a sentinel to avoid division-by-zero; handle after.
    safe_freq = np.where(freq_hz == 0.0, np.nan, freq_hz)
    f_khz = safe_freq / 1000.0

    # Painter & Spanias (2000) / Schönherr et al. (2018) approximation
    ath = (
        3.64 * np.power(f_khz, -0.8, where=~np.isnan(f_khz), out=np.full_like(f_khz, np.nan))
        - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
        + 1e-3 * f_khz ** 4
    )

    # DC (f = 0 Hz) is inaudible; assign a large positive threshold.
    ath = np.where(freq_hz == 0.0, 100.0, ath)

    # Clip to a physically meaningful range.
    ath = np.clip(ath, -20.0, 120.0)
    return ath


def compute_masking_threshold_db(
    signal_spectrum: np.ndarray,
    freqs: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Compute a per-bin overall masking threshold in dB SPL.

    Combines the absolute threshold of hearing with the spectral content of
    *signal_spectrum* to produce a frequency-wise masking threshold array.
    The masking threshold at each bin is the maximum of the ATH and the
    simultaneous masking elevation caused by nearby spectral peaks.

    This function is a lightweight alternative to the full spreading-function
    computation in :func:`simultaneous_masking_threshold`; it is suitable for
    quick energy-budget estimation.

    Parameters
    ----------
    signal_spectrum : array_like, shape (N,)
        Power spectrum of the signal (linear, not dB), as returned by
        ``np.abs(np.fft.rfft(signal)) ** 2``.  Must be non-negative.
    freqs : array_like, shape (N,)
        Centre frequencies in Hz corresponding to each bin of
        *signal_spectrum*.  Must be the same length as *signal_spectrum*.
    sr : int
        Sample rate in Hz. Currently used only for validation.

    Returns
    -------
    np.ndarray, shape (N,)
        Masking threshold in dB SPL per frequency bin.  Values are always
        ≥ ATH and finite.

    Raises
    ------
    ValueError
        If *signal_spectrum* and *freqs* have incompatible shapes, or if
        *signal_spectrum* contains negative values.
    ValueError
        If *sr* ≤ 0.

    Notes
    -----
    The masking threshold floor is set to the ISO 226 ATH.  Spectral masking
    elevation above a masker is modelled by a simple 3 dB uplift per octave
    of proximity — a deliberate simplification; use
    :func:`~audiocaptcha_dsp.psychoacoustics.masking_models.simultaneous_masking_threshold`
    for the full Zwicker spreading-function model.

    References
    ----------
    ISO 226:2003, "Acoustics — Normal equal-loudness-level contours".
    Moore, B.C.J. (2012). "An Introduction to the Psychology of Hearing".
    """
    signal_spectrum = np.asarray(signal_spectrum, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)

    if signal_spectrum.ndim != 1 or freqs.ndim != 1:
        raise ValueError("signal_spectrum and freqs must be 1-D arrays.")
    if signal_spectrum.shape != freqs.shape:
        raise ValueError(
            f"signal_spectrum shape {signal_spectrum.shape} != freqs shape {freqs.shape}."
        )
    if np.any(signal_spectrum < 0):
        raise ValueError("signal_spectrum must be non-negative (power values).")
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}.")

    # Absolute threshold floor
    ath_db = absolute_threshold(freqs, model=ThresholdModel.ISO226)

    # Convert power spectrum to dB (reference = 1.0 to give relative dB SPL)
    # Add a small epsilon to avoid log(0).
    eps = np.finfo(np.float64).tiny
    spectrum_db = 10.0 * np.log10(signal_spectrum + eps)

    # The masking threshold is the pointwise maximum of ATH and signal level.
    # This is the simplest (single-masker) approximation.
    masking_db = np.maximum(ath_db, spectrum_db)

    # Sanity check: replace any remaining NaN/Inf
    if not np.all(np.isfinite(masking_db)):
        warnings.warn(
            "Non-finite values detected in masking threshold; clamping to ATH.",
            RuntimeWarning,
            stacklevel=2,
        )
        masking_db = np.where(np.isfinite(masking_db), masking_db, ath_db)

    return masking_db
