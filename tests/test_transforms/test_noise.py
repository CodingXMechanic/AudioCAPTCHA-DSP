import pytest
import numpy as np
from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.noise.additive import (
    WhiteNoise, PinkNoise, BrownNoise, BandLimitedNoise, SpeechShapedNoise, ImpulsiveNoise, TonalInterference
)
from audiocaptcha_dsp.transforms.noise.reverberation import Reverberation, SimpleEcho

@pytest.fixture
def dummy_signal():
    sr = 16000
    t = np.linspace(0, 1, sr)
    waveform = np.sin(2 * np.pi * 440 * t) * 0.5
    return Signal(waveform=waveform, sample_rate=sr, metadata={})

@pytest.fixture
def silent_signal():
    sr = 16000
    waveform = np.zeros(sr)
    return Signal(waveform=waveform, sample_rate=sr, metadata={})

def test_white_noise_output_shape_preserved(dummy_signal):
    transform = WhiteNoise(snr_db=20.0, seed=42)
    out = transform(dummy_signal)
    assert out.waveform.shape == dummy_signal.waveform.shape

def test_white_noise_snr_approximately_correct(dummy_signal):
    transform = WhiteNoise(snr_db=20.0, seed=42)
    out = transform(dummy_signal)
    
    sig_rms = np.sqrt(np.mean(dummy_signal.waveform**2))
    noise = out.waveform - dummy_signal.waveform
    noise_rms = np.sqrt(np.mean(noise**2))
    
    actual_snr = 20 * np.log10(sig_rms / noise_rms)
    assert np.isclose(actual_snr, 20.0, atol=3.0)

def test_pink_noise_has_1_over_f_spectrum(silent_signal):
    transform = PinkNoise(snr_db=20.0, seed=42)
    out = transform(silent_signal)
    
    noise = out.waveform
    X = np.abs(np.fft.rfft(noise))
    # low freq energy > high freq energy
    assert np.sum(X[1:100]) > np.sum(X[-100:])

def test_brown_noise_has_steeper_slope_than_pink(silent_signal):
    transform_pink = PinkNoise(snr_db=20.0, seed=42)
    transform_brown = BrownNoise(snr_db=20.0, seed=42)
    
    pink = transform_pink(silent_signal).waveform
    brown = transform_brown(silent_signal).waveform
    
    X_pink = np.abs(np.fft.rfft(pink))
    X_brown = np.abs(np.fft.rfft(brown))
    
    pink_ratio = np.sum(X_pink[1:100]) / (np.sum(X_pink[-100:]) + 1e-12)
    brown_ratio = np.sum(X_brown[1:100]) / (np.sum(X_brown[-100:]) + 1e-12)
    
    assert brown_ratio > pink_ratio

def test_impulsive_noise_sparsity(silent_signal):
    transform = ImpulsiveNoise(rate=0.01, seed=42)
    out = transform(silent_signal)
    
    non_zeros = np.count_nonzero(out.waveform)
    # 0.01 of 16000 is 160
    assert non_zeros < 500

def test_tonal_interference_has_peak_at_frequency(silent_signal):
    transform = TonalInterference(frequency_hz=1000.0, snr_db=20.0, seed=42)
    out = transform(silent_signal)
    
    X = np.abs(np.fft.rfft(out.waveform))
    freqs = np.fft.rfftfreq(len(out.waveform), 1.0/16000)
    
    peak_idx = np.argmax(X)
    assert np.isclose(freqs[peak_idx], 1000.0, atol=10.0)

def test_reverberation_output_shape(dummy_signal):
    transform = Reverberation()
    out = transform(dummy_signal)
    assert out.waveform.shape == dummy_signal.waveform.shape

def test_reverberation_wet_level_affects_output(dummy_signal):
    t_dry = Reverberation(wet_level=0.0, dry_level=1.0)
    t_wet = Reverberation(wet_level=1.0, dry_level=0.0)
    
    out_dry = t_dry(dummy_signal)
    out_wet = t_wet(dummy_signal)
    
    assert not np.allclose(out_dry.waveform, out_wet.waveform)

def test_echo_creates_delayed_copy(silent_signal):
    # Put a spike at t=0
    silent_signal.waveform[0] = 1.0
    transform = SimpleEcho(delay_ms=10.0, gain=0.5)
    out = transform(silent_signal)
    
    delay_samples = int(16000 * 0.01)
    assert out.waveform[0] == 1.0
    assert out.waveform[delay_samples] == 0.5

def test_all_transforms_preserve_sample_rate(dummy_signal):
    transforms = [
        WhiteNoise(), PinkNoise(), BrownNoise(), BandLimitedNoise(), SpeechShapedNoise(),
        ImpulsiveNoise(), TonalInterference(), Reverberation(), SimpleEcho()
    ]
    for t in transforms:
        out = t(dummy_signal)
        assert out.sample_rate == dummy_signal.sample_rate

def test_all_transforms_no_nan_inf_in_output(dummy_signal):
    transforms = [
        WhiteNoise(), PinkNoise(), BrownNoise(), BandLimitedNoise(), SpeechShapedNoise(),
        ImpulsiveNoise(), TonalInterference(), Reverberation(), SimpleEcho()
    ]
    for t in transforms:
        out = t(dummy_signal)
        assert np.isfinite(out.waveform).all()

def test_all_transforms_with_silent_input(silent_signal):
    transforms = [
        WhiteNoise(), PinkNoise(), BrownNoise(), BandLimitedNoise(), SpeechShapedNoise(),
        ImpulsiveNoise(), TonalInterference(), Reverberation(), SimpleEcho()
    ]
    for t in transforms:
        out = t(silent_signal)
        assert np.isfinite(out.waveform).all()
