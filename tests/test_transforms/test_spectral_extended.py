import pytest
import numpy as np
from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.spectral.filtering import (
    BandpassFilter, LowpassFilter, HighpassFilter, SpectralTilt,
    SpectralSmoothing, FrequencyBinDropout, PhaseRandomization,
    CombFilter, HarmonicAttenuation
)
from audiocaptcha_dsp.transforms.spectral.mel_masking import (
    MelBandMasking, BarkBandMasking, CriticalBandAttenuation
)

@pytest.fixture
def dummy_signal():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)
    return Signal(waveform=data, sample_rate=sr)

def test_bandpass_attenuates_out_of_band_frequencies(dummy_signal):
    transform = BandpassFilter(low_hz=500, high_hz=2000, order=4)
    out = transform(dummy_signal)
    spec_in = np.abs(np.fft.rfft(dummy_signal.waveform))
    spec_out = np.abs(np.fft.rfft(out.waveform))
    freqs = np.fft.rfftfreq(len(dummy_signal.waveform), 1/dummy_signal.sample_rate)
    idx_5000 = np.argmin(np.abs(freqs - 5000))
    assert spec_out[idx_5000] < spec_in[idx_5000] * 0.1

def test_lowpass_attenuates_high_frequencies(dummy_signal):
    transform = LowpassFilter(cutoff_hz=2000, order=4)
    out = transform(dummy_signal)
    spec_in = np.abs(np.fft.rfft(dummy_signal.waveform))
    spec_out = np.abs(np.fft.rfft(out.waveform))
    freqs = np.fft.rfftfreq(len(dummy_signal.waveform), 1/dummy_signal.sample_rate)
    idx_5000 = np.argmin(np.abs(freqs - 5000))
    assert spec_out[idx_5000] < spec_in[idx_5000] * 0.1

def test_highpass_attenuates_low_frequencies(dummy_signal):
    transform = HighpassFilter(cutoff_hz=500, order=4)
    out = transform(dummy_signal)
    spec_in = np.abs(np.fft.rfft(dummy_signal.waveform))
    spec_out = np.abs(np.fft.rfft(out.waveform))
    freqs = np.fft.rfftfreq(len(dummy_signal.waveform), 1/dummy_signal.sample_rate)
    idx_100 = np.argmin(np.abs(freqs - 100))
    assert spec_out[idx_100] < spec_in[idx_100] * 0.1

def test_spectral_tilt_changes_spectrum_shape(dummy_signal):
    transform = SpectralTilt(tilt_db_per_octave=6.0, pivot_hz=1000.0)
    out = transform(dummy_signal)
    assert not np.allclose(out.waveform, dummy_signal.waveform)

def test_phase_randomization_preserves_magnitude(dummy_signal):
    transform = PhaseRandomization(randomization_strength=1.0)
    out = transform(dummy_signal)
    mag_in = np.abs(np.fft.rfft(dummy_signal.waveform))
    mag_out = np.abs(np.fft.rfft(out.waveform))
    assert np.allclose(mag_in, mag_out, atol=1e-5)

def test_phase_randomization_zero_strength_is_identity(dummy_signal):
    transform = PhaseRandomization(randomization_strength=0.0)
    out = transform(dummy_signal)
    assert np.allclose(out.waveform, dummy_signal.waveform, atol=1e-5)

def test_comb_filter_output_shape(dummy_signal):
    transform = CombFilter(delay_ms=10.0, gain=0.5)
    out = transform(dummy_signal)
    assert out.waveform.shape == dummy_signal.waveform.shape

def test_mel_masking_reduces_energy(dummy_signal):
    transform = MelBandMasking(max_mask_bands=2, n_mels=80)
    out = transform(dummy_signal)
    assert np.sum(out.waveform**2) <= np.sum(dummy_signal.waveform**2) + 1e-5

def test_bark_masking_reduces_energy(dummy_signal):
    transform = BarkBandMasking(max_mask_bands=2)
    out = transform(dummy_signal)
    assert np.sum(out.waveform**2) <= np.sum(dummy_signal.waveform**2) + 1e-5

def test_critical_band_attenuation_reduces_energy(dummy_signal):
    transform = CriticalBandAttenuation(band_indices=[5, 10, 15], attenuation_db=20.0)
    out = transform(dummy_signal)
    assert np.sum(out.waveform**2) <= np.sum(dummy_signal.waveform**2) + 1e-5

def test_freq_dropout_creates_zero_bins(dummy_signal):
    transform = FrequencyBinDropout(dropout_rate=0.5)
    out = transform(dummy_signal)
    spec_out = np.abs(np.fft.rfft(out.waveform))
    assert np.any(np.isclose(spec_out, 0.0, atol=1e-10))

def test_spectral_smoothing_reduces_spectral_contrast(dummy_signal):
    transform = SpectralSmoothing(smoothing_bins=10)
    out = transform(dummy_signal)
    assert not np.allclose(out.waveform, dummy_signal.waveform)

def test_all_spectral_transforms_preserve_sample_rate(dummy_signal):
    transforms = [
        BandpassFilter(), LowpassFilter(), HighpassFilter(), SpectralTilt(),
        SpectralSmoothing(), FrequencyBinDropout(), PhaseRandomization(),
        CombFilter(), HarmonicAttenuation(), MelBandMasking(),
        BarkBandMasking(), CriticalBandAttenuation()
    ]
    for t in transforms:
        out = t(dummy_signal)
        assert out.sample_rate == dummy_signal.sample_rate

def test_all_spectral_transforms_no_nan_inf(dummy_signal):
    dummy_signal.waveform[10:20] = 0.0
    transforms = [
        BandpassFilter(), LowpassFilter(), HighpassFilter(), SpectralTilt(),
        SpectralSmoothing(), FrequencyBinDropout(), PhaseRandomization(),
        CombFilter(), HarmonicAttenuation(), MelBandMasking(),
        BarkBandMasking(), CriticalBandAttenuation()
    ]
    for t in transforms:
        out = t(dummy_signal)
        assert not np.any(np.isnan(out.waveform))
        assert not np.any(np.isinf(out.waveform))
