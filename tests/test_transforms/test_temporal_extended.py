import pytest
import numpy as np
from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.temporal.stretching import (
    TimeStretch, PitchShift, SpeedPerturbation,
    TimeMasking, TemporalDropout, LocalTimeWarping
)

@pytest.fixture
def dummy_signal():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = np.sin(2 * np.pi * 440 * t)
    return Signal(waveform=data, sample_rate=sr)

def test_time_stretch_changes_duration(dummy_signal):
    transform = TimeStretch(rate=2.0)
    out = transform(dummy_signal)
    assert len(out.waveform) < len(dummy_signal.waveform)

def test_time_stretch_factor_1_is_near_identity(dummy_signal):
    transform = TimeStretch(rate=1.0)
    out = transform(dummy_signal)
    assert np.allclose(out.waveform, dummy_signal.waveform, atol=1e-5)

def test_pitch_shift_zero_steps_is_near_identity(dummy_signal):
    transform = PitchShift(n_steps=0.0)
    out = transform(dummy_signal)
    assert np.allclose(out.waveform, dummy_signal.waveform, atol=1e-5)

def test_speed_perturbation_changes_duration(dummy_signal):
    transform = SpeedPerturbation(factor=2.0)
    out = transform(dummy_signal)
    assert len(out.waveform) < len(dummy_signal.waveform)

def test_time_masking_zeroes_out_region(dummy_signal):
    transform = TimeMasking(max_mask_ms=100.0, num_masks=1)
    out = transform(dummy_signal)
    assert np.any(out.waveform == 0.0)

def test_temporal_dropout_reduces_energy(dummy_signal):
    transform = TemporalDropout(dropout_rate=0.5, segment_ms=10.0)
    out = transform(dummy_signal)
    assert np.sum(out.waveform**2) < np.sum(dummy_signal.waveform**2)

def test_local_time_warping_preserves_length(dummy_signal):
    transform = LocalTimeWarping(warp_factor=0.1)
    out = transform(dummy_signal)
    assert len(out.waveform) == len(dummy_signal.waveform)

def test_all_temporal_transforms_preserve_sample_rate_attribute(dummy_signal):
    transforms = [
        TimeStretch(rate=1.5), PitchShift(n_steps=2), SpeedPerturbation(factor=1.5),
        TimeMasking(), TemporalDropout(), LocalTimeWarping()
    ]
    for t in transforms:
        out = t(dummy_signal)
        assert out.sample_rate == dummy_signal.sample_rate

def test_all_temporal_transforms_no_nan_inf(dummy_signal):
    transforms = [
        TimeStretch(rate=1.5), PitchShift(n_steps=2), SpeedPerturbation(factor=1.5),
        TimeMasking(), TemporalDropout(), LocalTimeWarping()
    ]
    for t in transforms:
        out = t(dummy_signal)
        assert not np.any(np.isnan(out.waveform))
        assert not np.any(np.isinf(out.waveform))
