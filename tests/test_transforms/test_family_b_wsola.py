import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.temporal.wsola import (
    WSOLATimeStretch,
    PSOLAPitchShift,
    ProsodyPerturbation,
    RhythmPerturbation,
    PauseInsertion,
    SegmentDisplacement,
    MicroTimingVariation,
)


@pytest.fixture
def speech_signal() -> Signal:
    """Harmonic speech-like signal with formant-ish envelope, 1.5 s @ 16 kHz."""
    sr = 16000
    t = np.arange(int(1.5 * sr)) / sr
    f0 = 130.0
    wave = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
        + 0.1 * np.sin(2 * np.pi * 4 * f0 * t)
    )
    # Syllabic-like AM to create energy minima for pause insertion
    wave *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    return Signal(waveform=wave, sample_rate=sr, metadata={"transcript": "test"})


# ---------------------------------------------------------------- family B


def test_wsola_rate_gt_1_shortens(speech_signal):
    out = WSOLATimeStretch(rate=1.5)(speech_signal)
    assert len(out.waveform) < len(speech_signal.waveform)
    assert out.sample_rate == speech_signal.sample_rate


def test_wsola_rate_lt_1_lengthens(speech_signal):
    out = WSOLATimeStretch(rate=0.8)(speech_signal)
    assert len(out.waveform) > len(speech_signal.waveform)


def test_wsola_identity_rate(speech_signal):
    out = WSOLATimeStretch(rate=1.0)(speech_signal)
    assert np.allclose(out.waveform, speech_signal.waveform, atol=1e-9)


def test_wsola_invalid_rate_raises(speech_signal):
    with pytest.raises(ValueError):
        WSOLATimeStretch(rate=0.0)


def test_psola_preserves_duration(speech_signal):
    out = PSOLAPitchShift(n_steps=3.0)(speech_signal)
    assert len(out.waveform) == len(speech_signal.waveform)
    assert np.all(np.isfinite(out.waveform))


def test_psola_zero_steps_identity(speech_signal):
    out = PSOLAPitchShift(n_steps=0.0)(speech_signal)
    assert np.allclose(out.waveform, speech_signal.waveform, atol=1e-9)


def test_psola_negative_shift(speech_signal):
    out = PSOLAPitchShift(n_steps=-4.0)(speech_signal)
    assert len(out.waveform) == len(speech_signal.waveform)
    assert not np.allclose(out.waveform, speech_signal.waveform)


def test_prosody_preserves_duration_and_changes_signal(speech_signal):
    out = ProsodyPerturbation(semitone_range=2.0, n_segments=6, seed=7)(speech_signal)
    assert len(out.waveform) == len(speech_signal.waveform)
    assert not np.allclose(out.waveform, speech_signal.waveform)
    assert np.all(np.isfinite(out.waveform))


def test_prosody_zero_range_identity(speech_signal):
    out = ProsodyPerturbation(semitone_range=0.0)(speech_signal)
    assert np.allclose(out.waveform, speech_signal.waveform, atol=1e-9)


def test_rhythm_preserves_duration(speech_signal):
    out = RhythmPerturbation(rate_sigma=0.2, n_segments=8, seed=3)(speech_signal)
    assert len(out.waveform) == len(speech_signal.waveform)
    assert np.all(np.isfinite(out.waveform))


def test_rhythm_zero_sigma_identity(speech_signal):
    out = RhythmPerturbation(rate_sigma=0.0)(speech_signal)
    assert np.allclose(out.waveform, speech_signal.waveform, atol=1e-9)


def test_pause_insertion_increases_length(speech_signal):
    out = PauseInsertion(n_pauses=3, pause_ms=120.0, seed=1)(speech_signal)
    assert len(out.waveform) > len(speech_signal.waveform)
    # Expected increase ≈ 3 × 120 ms (allow boundary skips)
    added = (len(out.waveform) - len(speech_signal.waveform)) / speech_signal.sample_rate
    assert added > 0.15


def test_pause_insertion_zero_count_identity(speech_signal):
    out = PauseInsertion(n_pauses=0)(speech_signal)
    assert len(out.waveform) == len(speech_signal.waveform)


def test_segment_displacement_preserves_duration(speech_signal):
    out = SegmentDisplacement(segment_ms=200.0, max_shift_ms=60.0, n_moves=3, seed=5)(
        speech_signal
    )
    assert len(out.waveform) == len(speech_signal.waveform)
    assert not np.allclose(out.waveform, speech_signal.waveform)


def test_segment_displacement_zero_moves_identity(speech_signal):
    out = SegmentDisplacement(n_moves=0)(speech_signal)
    assert np.allclose(out.waveform, speech_signal.waveform, atol=1e-9)


def test_micro_timing_preserves_duration(speech_signal):
    out = MicroTimingVariation(max_shift_ms=8.0, seed=11)(speech_signal)
    assert len(out.waveform) == len(speech_signal.waveform)
    assert not np.allclose(out.waveform, speech_signal.waveform)


def test_micro_timing_zero_shift_identity(speech_signal):
    out = MicroTimingVariation(max_shift_ms=0.0)(speech_signal)
    assert np.allclose(out.waveform, speech_signal.waveform, atol=1e-9)


def test_all_family_b_no_nan_inf(speech_signal):
    transforms = [
        WSOLATimeStretch(rate=1.3),
        PSOLAPitchShift(n_steps=2.0),
        ProsodyPerturbation(semitone_range=1.5, seed=2),
        RhythmPerturbation(rate_sigma=0.15, seed=2),
        PauseInsertion(n_pauses=2, seed=2),
        SegmentDisplacement(n_moves=2, seed=2),
        MicroTimingVariation(max_shift_ms=5.0, seed=2),
    ]
    for tr in transforms:
        out = tr(speech_signal)
        assert out.sample_rate == speech_signal.sample_rate
        assert len(out.waveform) > 0
        assert not np.any(np.isnan(out.waveform))
        assert not np.any(np.isinf(out.waveform))
