import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.multirate import (
    IntegerResampling,
    RationalResampling,
    NonIntegerResampling,
    TimeVaryingResampling,
    SampleRateDrift,
    BandwidthLimitation,
    AntiAliasVariation,
    ControlledAliasing,
    NarrowbandTelephone,
    WidebandToNarrowband,
    CodecMultirateArtifacts,
)


@pytest.fixture
def wideband_signal() -> Signal:
    """16 kHz wideband signal with content up to ~7 kHz."""
    sr = 16000
    t = np.arange(int(1.0 * sr)) / sr
    wave = (
        0.4 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 3000 * t)
        + 0.2 * np.sin(2 * np.pi * 6500 * t)
    )
    wave *= 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    return Signal(waveform=wave, sample_rate=sr)


ALL_TRANSFORMS = [
    IntegerResampling(down_factor=2),
    RationalResampling(up_factor=3, down_factor=2),
    NonIntegerResampling(target_sr=11025),
    TimeVaryingResampling(rate_start=1.0, rate_end=1.1),
    SampleRateDrift(drift_ppm=500.0),
    BandwidthLimitation(cutoff_hz=7000.0),
    AntiAliasVariation(down_factor=2, alias_filter_order=1),
    ControlledAliasing(down_factor=3),
    NarrowbandTelephone(),
    WidebandToNarrowband(),
    CodecMultirateArtifacts(block_size_ms=1.5),
]


def test_multirate_preserves_duration_and_rate(wideband_signal):
    for tr in ALL_TRANSFORMS:
        out = tr(wideband_signal)
        assert out.sample_rate == wideband_signal.sample_rate
        assert len(out.waveform) == len(wideband_signal.waveform)
        assert np.all(np.isfinite(out.waveform))


def test_integer_resampling_antialiases(wideband_signal):
    """Down/up with anti-aliasing must attenuate the 6.5 kHz component."""
    out = IntegerResampling(down_factor=2, anti_alias=True)(wideband_signal)
    spec_in = np.abs(np.fft.rfft(wideband_signal.waveform))
    spec_out = np.abs(np.fft.rfft(out.waveform))
    freqs = np.fft.rfftfreq(len(wideband_signal.waveform), 1 / wideband_signal.sample_rate)
    band = (freqs > 6000) & (freqs < 7000)
    assert spec_out[band].mean() < 0.5 * spec_in[band].mean()


def test_controlled_aliasing_folds_energy(wideband_signal):
    """Aliased decimation must move energy to inaudible-band fold-over."""
    out = ControlledAliasing(down_factor=3)(wideband_signal)
    assert not np.allclose(out.waveform, wideband_signal.waveform, atol=1e-6)


def test_bandwidth_limitation_removes_high_freq(wideband_signal):
    out = BandwidthLimitation(cutoff_hz=4000.0)(wideband_signal)
    spec = np.abs(np.fft.rfft(out.waveform))
    freqs = np.fft.rfftfreq(len(out.waveform), 1 / wideband_signal.sample_rate)
    high = freqs > 5500
    low = (freqs > 300) & (freqs < 3500)
    assert spec[high].mean() < 0.2 * spec[low].mean()


def test_narrowband_telephone_passes_speech_band(wideband_signal):
    out = NarrowbandTelephone()(wideband_signal)
    spec = np.abs(np.fft.rfft(out.waveform))
    freqs = np.fft.rfftfreq(len(out.waveform), 1 / wideband_signal.sample_rate)
    speech_band = (freqs > 500) & (freqs < 3000)
    high_band = freqs > 5000
    assert spec[speech_band].mean() > 3.0 * spec[high_band].mean()


def test_sample_rate_drift_small_changes(wideband_signal):
    out = SampleRateDrift(drift_ppm=500.0)(wideband_signal)
    assert len(out.waveform) == len(wideband_signal.waveform)
    diff = np.abs(out.waveform - wideband_signal.waveform)
    # 500 ppm = subtle change, not destruction
    assert diff.mean() < 0.5 * np.abs(wideband_signal.waveform).mean()


def test_zero_drift_is_identity(wideband_signal):
    out = SampleRateDrift(drift_ppm=0.0)(wideband_signal)
    assert np.allclose(out.waveform, wideband_signal.waveform, atol=1e-9)


def test_time_varying_resampling_changes_signal(wideband_signal):
    out = TimeVaryingResampling(rate_start=1.0, rate_end=1.1)(wideband_signal)
    assert len(out.waveform) == len(wideband_signal.waveform)
    assert not np.allclose(out.waveform, wideband_signal.waveform, atol=1e-6)


def test_codec_artifacts_add_noise_floor(wideband_signal):
    out = CodecMultirateArtifacts(block_size_ms=1.5, quantization_bits=3.0)(wideband_signal)
    assert not np.allclose(out.waveform, wideband_signal.waveform, atol=1e-6)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        IntegerResampling(down_factor=0)
    with pytest.raises(ValueError):
        BandwidthLimitation(cutoff_hz=-1.0)
    with pytest.raises(ValueError):
        AntiAliasVariation(down_factor=2, alias_filter_order=-1)
