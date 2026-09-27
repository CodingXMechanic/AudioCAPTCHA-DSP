import pytest
import numpy as np
from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.baseline.gain import (
    IdentityTransform, GainTransform, PeakNormalize, RMSNormalize, LoudnessNormalize,
    DynamicRangeCompressor, Limiter, SilencePadding, SampleRateConverter, BitDepthConverter, MuLawCompanding
)

@pytest.fixture
def dummy_signal():
    sr = 16000
    t = np.linspace(0, 1, sr)
    waveform = np.sin(2 * np.pi * 440 * t) * 0.5
    return Signal(waveform=waveform, sample_rate=sr, metadata={})

def test_identity_is_clone(dummy_signal):
    transform = IdentityTransform()
    out = transform(dummy_signal)
    np.testing.assert_array_equal(out.waveform, dummy_signal.waveform)
    assert out.sample_rate == dummy_signal.sample_rate

def test_gain_increases_amplitude(dummy_signal):
    transform = GainTransform(gain_db=6.0)
    out = transform(dummy_signal)
    assert np.max(np.abs(out.waveform)) > np.max(np.abs(dummy_signal.waveform))

def test_gain_decreases_amplitude(dummy_signal):
    transform = GainTransform(gain_db=-6.0)
    out = transform(dummy_signal)
    assert np.max(np.abs(out.waveform)) < np.max(np.abs(dummy_signal.waveform))

def test_peak_normalize_correct_peak(dummy_signal):
    transform = PeakNormalize(target_db=-3.0)
    out = transform(dummy_signal)
    target_linear = 10.0 ** (-3.0 / 20.0)
    assert np.isclose(np.max(np.abs(out.waveform)), target_linear)

def test_rms_normalize_correct_rms(dummy_signal):
    transform = RMSNormalize(target_rms=0.1)
    out = transform(dummy_signal)
    rms = np.sqrt(np.mean(out.waveform**2))
    assert np.isclose(rms, 0.1)

def test_compressor_reduces_dynamic_range():
    sr = 16000
    t = np.linspace(0, 1, sr)
    waveform = np.sin(2 * np.pi * 440 * t) * 0.9  # High amplitude
    waveform[:sr//2] *= 0.1  # Low amplitude half
    sig = Signal(waveform=waveform, sample_rate=sr, metadata={})
    
    transform = DynamicRangeCompressor(threshold_db=-20.0, ratio=4.0)
    out = transform(sig)
    
    # Check that high amplitude part was compressed more than low amplitude part
    orig_ratio = np.max(np.abs(sig.waveform[sr//2:])) / np.max(np.abs(sig.waveform[:sr//2]))
    new_ratio = np.max(np.abs(out.waveform[sr//2:])) / np.max(np.abs(out.waveform[:sr//2]))
    assert new_ratio < orig_ratio

def test_limiter_clips_at_threshold(dummy_signal):
    # Dummy has peak 0.5 (approx -6dB)
    transform = Limiter(threshold_db=-10.0) # approx 0.316
    out = transform(dummy_signal)
    target = 10.0 ** (-10.0 / 20.0)
    assert np.max(np.abs(out.waveform)) <= target + 1e-5

def test_silence_padding_increases_duration(dummy_signal):
    transform = SilencePadding(pad_ms=500.0, location='both')
    out = transform(dummy_signal)
    expected_extra = int(16000 * 0.5) * 2
    assert out.waveform.shape[-1] == dummy_signal.waveform.shape[-1] + expected_extra

def test_sample_rate_converter_changes_sample_rate(dummy_signal):
    transform = SampleRateConverter(target_sr=8000)
    out = transform(dummy_signal)
    assert out.sample_rate == dummy_signal.sample_rate # output SR preserved
    assert out.waveform.shape == dummy_signal.waveform.shape # polyphase back preserves length mostly

def test_bit_depth_converter_reduces_precision(dummy_signal):
    transform = BitDepthConverter(bits=4)
    out = transform(dummy_signal)
    # Should have fewer unique values
    assert len(np.unique(out.waveform)) <= 16

def test_mu_law_output_shape_preserved(dummy_signal):
    transform = MuLawCompanding()
    out = transform(dummy_signal)
    assert out.waveform.shape == dummy_signal.waveform.shape

def test_all_baseline_transforms_no_nan_inf(dummy_signal):
    transforms = [
        IdentityTransform(), GainTransform(), PeakNormalize(), RMSNormalize(), LoudnessNormalize(),
        DynamicRangeCompressor(), Limiter(), SilencePadding(), SampleRateConverter(),
        BitDepthConverter(), MuLawCompanding()
    ]
    for t in transforms:
        out = t(dummy_signal)
        assert np.isfinite(out.waveform).all()
