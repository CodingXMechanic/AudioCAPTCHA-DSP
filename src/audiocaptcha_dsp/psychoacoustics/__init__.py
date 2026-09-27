"""
AudioCAPTCHA-DSP Psychoacoustics Package
==========================================
Public API for psychoacoustic transforms, masking models, and loudness
computation used in the AudioCAPTCHA-DSP research framework.
"""
from audiocaptcha_dsp.psychoacoustics.bark_scale import hz_to_bark, bark_to_hz
from audiocaptcha_dsp.psychoacoustics.thresholds import (
    absolute_threshold,
    compute_masking_threshold_db,
)
from audiocaptcha_dsp.psychoacoustics.masking_models import (
    simultaneous_masking_threshold,
    temporal_masking_threshold,
    compute_masking_budget,
)
from audiocaptcha_dsp.psychoacoustics.loudness import (
    specific_loudness,
    compute_loudness_sone,
    rms_to_dbspl,
    normalize_to_dbspl,
)

__all__ = [
    # Bark scale
    "hz_to_bark",
    "bark_to_hz",
    # Thresholds
    "absolute_threshold",
    "compute_masking_threshold_db",
    # Masking models
    "simultaneous_masking_threshold",
    "temporal_masking_threshold",
    "compute_masking_budget",
    # Loudness
    "specific_loudness",
    "compute_loudness_sone",
    "rms_to_dbspl",
    "normalize_to_dbspl",
]
