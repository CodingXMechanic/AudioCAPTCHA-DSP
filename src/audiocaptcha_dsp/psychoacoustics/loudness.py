"""
Psychoacoustic Loudness Models
================================
Implements loudness computation, RMS-to-dBSPL conversion, and signal
normalisation based on Steven's power law and the ISO 226 equal-loudness
framework.

References
----------
- Stevens, S. S. (1957). "On the psychophysical law." *Psychological Review*,
  64(3), 153–181.
- Zwicker, E. & Fastl, H. (1999). *Psychoacoustics: Facts and Models*, 2nd ed.
  Springer.
- Moore, B.C.J. (2012). *An Introduction to the Psychology of Hearing*, 6th ed.
"""
from __future__ import annotations

import warnings

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.psychoacoustics.bark_scale import hz_to_bark


# ---------------------------------------------------------------------------
# Existing function (preserved)
# ---------------------------------------------------------------------------

def specific_loudness(
    power_spectrum: np.ndarray,
    freq_bins: np.ndarray,
) -> np.ndarray:
    """Compute specific loudness (sone/Bark) per critical band.

    Applies Steven's power law to power accumulated in Bark critical bands.
    Specific loudness :math:`N'(z)` at Bark position *z* is defined as:

    .. math::

        N'(z) = k \\cdot E(z)^{0.23}

    where :math:`E(z)` is the excitation in the *z*-th critical band and
    :math:`k` is a normalisation constant chosen so that a 1 kHz, 40 dB SPL
    tone yields 1 sone.

    Parameters
    ----------
    power_spectrum : array_like, shape (N,)
        Power spectrum (linear, reference = 1 Pa²) of the signal.
    freq_bins : array_like, shape (N,)
        Centre frequencies in Hz for each bin of *power_spectrum*.

    Returns
    -------
    np.ndarray, shape (24,)
        Specific loudness in sone per Bark band (24 critical bands).

    Raises
    ------
    ValueError
        If *power_spectrum* contains negative values or if shapes mismatch.

    Notes
    -----
    This approximation follows Stevens (1957) and Zwicker & Fastl (1999),
    Section 8.1.  The constant ``k = 0.0635`` is the Zwicker normalisation
    factor (see eq. 8.1 in Zwicker & Fastl).
    """
    power_spectrum = np.asarray(power_spectrum, dtype=np.float64)
    freq_bins = np.asarray(freq_bins, dtype=np.float64)

    if power_spectrum.shape != freq_bins.shape:
        raise ValueError(
            f"power_spectrum shape {power_spectrum.shape} != freq_bins shape {freq_bins.shape}."
        )
    if np.any(power_spectrum < 0):
        raise ValueError("power_spectrum must be non-negative.")

    n_bark = 24
    bark_edges = np.arange(0, n_bark + 1, dtype=np.float64)  # 0,1,...,24
    bark_freqs = hz_to_bark(np.clip(freq_bins, 1.0, None))

    # k: Zwicker normalisation so that 1 sone ≡ 40 dB SPL at 1 kHz
    k = 0.0635
    specific = np.zeros(n_bark, dtype=np.float64)
    for i in range(n_bark):
        mask = (bark_freqs >= bark_edges[i]) & (bark_freqs < bark_edges[i + 1])
        excitation = power_spectrum[mask].sum() if mask.any() else 0.0
        specific[i] = k * (excitation ** 0.23)

    return specific


# ---------------------------------------------------------------------------
# New public functions
# ---------------------------------------------------------------------------

def compute_loudness_sone(signal: np.ndarray, sr: int) -> float:
    """Compute total loudness of a signal in sone using Steven's power law.

    Integrates :func:`specific_loudness` over all 24 Bark critical bands.

    .. math::

        N = \\sum_{z=1}^{24} N'(z)

    The total loudness is in sone; 1 sone corresponds perceptually to a pure
    tone of 1 kHz at 40 dB SPL.

    Parameters
    ----------
    signal : array_like, shape (N,)
        Mono time-domain waveform (float, arbitrary amplitude normalisation).
    sr : int
        Sample rate in Hz.

    Returns
    -------
    float
        Total loudness in sone (≥ 0).

    Raises
    ------
    ValueError
        If *signal* is not 1-D or if *sr* ≤ 0.
    ValueError
        If *signal* contains NaN or Inf.

    Notes
    -----
    Steven's power law exponent 0.23 gives the perceptual loudness growth;
    the Zwicker model also accounts for a masking threshold floor but that
    correction is omitted here for simplicity.  For a full Zwicker loudness
    model see ISO 532-1.

    References
    ----------
    Stevens, S.S. (1957). "On the psychophysical law." *Psychological Review*.
    Zwicker, E. & Fastl, H. (1999). *Psychoacoustics: Facts and Models*.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}.")
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}.")
    if np.any(~np.isfinite(signal)):
        raise ValueError("signal contains NaN or Inf.")

    n_fft = len(signal)
    spectrum = np.fft.rfft(signal, n=n_fft)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    spec_loud = specific_loudness(power, freqs)
    total = float(np.sum(spec_loud))
    return max(total, 0.0)


def rms_to_dbspl(rms_value: float | np.ndarray, reference: float = 2e-5) -> float | np.ndarray:
    """Convert RMS amplitude to dB SPL.

    .. math::

        L = 20 \\log_{10}\\!\\left(\\frac{p_{\\text{rms}}}{p_0}\\right)

    where :math:`p_0 = 2 \\times 10^{-5}\\,\\text{Pa}` is the standard
    acoustic reference pressure for airborne sound (ISO 1683).

    Parameters
    ----------
    rms_value : float or array_like
        RMS pressure amplitude (Pa).  Must be strictly positive.
    reference : float, optional
        Reference pressure in Pa. Default 2×10⁻⁵ Pa (ISO 1683).

    Returns
    -------
    float or np.ndarray
        Sound pressure level in dB SPL.  Returns −∞ for rms_value = 0.

    Raises
    ------
    ValueError
        If *rms_value* contains negative values.
    ValueError
        If *reference* ≤ 0.

    Notes
    -----
    The standard acoustic reference for airborne sound is
    :math:`p_0 = 20\\,\\mu\\text{Pa}` (ISO 1683:2015).

    Examples
    --------
    >>> rms_to_dbspl(2e-5)   # 0 dB SPL
    0.0
    >>> rms_to_dbspl(2e-4)   # 20 dB SPL
    20.0
    """
    scalar_input = np.ndim(rms_value) == 0
    rms_value = np.atleast_1d(np.asarray(rms_value, dtype=np.float64))
    if reference <= 0:
        raise ValueError(f"reference must be positive, got {reference}.")
    if np.any(rms_value < 0):
        raise ValueError("rms_value must be non-negative.")

    with np.errstate(divide="ignore"):
        dbspl = 20.0 * np.log10(rms_value / reference)

    result = dbspl if not scalar_input else float(dbspl[0])
    return result


def normalize_to_dbspl(
    signal: "Signal",
    target_dbspl: float,
    reference: float = 2e-5,
) -> "Signal":
    """Normalise a Signal to a target dB SPL level.

    Scales the signal waveform so that its RMS amplitude corresponds to
    *target_dbspl* dB re *reference* Pa.

    Parameters
    ----------
    signal : Signal
        Input signal.  The waveform may be 1-D or 2-D (channels × samples);
        RMS is computed globally across all samples.
    target_dbspl : float
        Target sound pressure level in dB SPL.
    reference : float, optional
        Acoustic reference pressure in Pa. Default 2×10⁻⁵ Pa.

    Returns
    -------
    Signal
        New Signal with waveform scaled to *target_dbspl*.  Metadata is
        propagated; ``'normalized_dbspl'`` key is added.

    Raises
    ------
    ValueError
        If *signal* is silent (RMS = 0) — cannot normalise silence.
    ValueError
        If *reference* ≤ 0 or *target_dbspl* is not finite.

    Notes
    -----
    The target RMS amplitude :math:`p_{\\text{target}}` is:

    .. math::

        p_{\\text{target}} = p_0 \\cdot 10^{L_{\\text{target}} / 20}

    The waveform is then scaled by :math:`p_{\\text{target}} / p_{\\text{rms}}`.

    Examples
    --------
    >>> import numpy as np
    >>> from audiocaptcha_dsp.core.signal import Signal
    >>> from audiocaptcha_dsp.psychoacoustics.loudness import normalize_to_dbspl
    >>> sig = Signal(waveform=np.random.randn(16000), sample_rate=16000)
    >>> out = normalize_to_dbspl(sig, target_dbspl=70.0)
    """
    if reference <= 0:
        raise ValueError(f"reference must be positive, got {reference}.")
    if not np.isfinite(target_dbspl):
        raise ValueError(f"target_dbspl must be finite, got {target_dbspl}.")

    waveform = signal.waveform.astype(np.float64)
    current_rms = float(np.sqrt(np.mean(waveform ** 2)))
    if current_rms == 0.0:
        raise ValueError("Cannot normalise a silent signal (RMS = 0).")

    target_rms = reference * 10.0 ** (target_dbspl / 20.0)
    scale = target_rms / current_rms
    normalised_waveform = waveform * scale

    # Warn if clipping might occur (for informational purposes only)
    if np.any(np.abs(normalised_waveform) > 1.0):
        warnings.warn(
            f"normalize_to_dbspl: waveform values exceed ±1.0 after "
            f"normalisation to {target_dbspl} dB SPL. Consider clipping.",
            UserWarning,
            stacklevel=2,
        )

    new_metadata = {**signal.metadata, "normalized_dbspl": target_dbspl}
    return Signal(
        waveform=normalised_waveform,
        sample_rate=signal.sample_rate,
        metadata=new_metadata,
    )
