import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.noise.interference import (
    BabbleNoise,
    HarmonicInterference,
    TransientMasking,
    NonstationaryNoise,
    DelayAndAddInterference,
    RoomImpulseResponse,
    synthetic_rir,
)


@pytest.fixture
def speech_like() -> Signal:
    sr = 16000
    t = np.arange(int(1.5 * sr)) / sr
    wave = 0.5 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 360 * t)
    wave *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return Signal(waveform=wave, sample_rate=sr)


def test_babble_adds_noise_at_snr(speech_like):
    out = BabbleNoise(snr_db=15.0, n_talkers=8, seed=1)(speech_like)
    assert len(out.waveform) == len(speech_like.waveform)
    residual = out.waveform - speech_like.waveform
    snr = 10 * np.log10(
        np.mean(speech_like.waveform ** 2) / (np.mean(residual ** 2) + 1e-18)
    )
    assert abs(snr - 15.0) < 1.0


def test_babble_different_seed_different_noise(speech_like):
    a = BabbleNoise(snr_db=15.0, seed=1)(speech_like)
    b = BabbleNoise(snr_db=15.0, seed=2)(speech_like)
    assert not np.allclose(a.waveform, b.waveform)


def test_harmonic_interference_adds_harmonics(speech_like):
    out = HarmonicInterference(f0_hz=150.0, n_harmonics=8, snr_db=15.0, seed=3)(speech_like)
    residual = out.waveform - speech_like.waveform
    spec = np.abs(np.fft.rfft(residual))
    freqs = np.fft.rfftfreq(len(residual), 1 / speech_like.sample_rate)
    # Energy should concentrate near multiples of 150 Hz
    peak_bin = np.argmax(spec[1:]) + 1
    nearest_harmonic = round(freqs[peak_bin] / 150.0) * 150.0
    assert abs(freqs[peak_bin] - nearest_harmonic) < 5.0


def test_transient_masking_hits_onsets(speech_like):
    out = TransientMasking(level_db=8.0, n_transients=4, seed=2)(speech_like)
    assert len(out.waveform) == len(speech_like.waveform)
    assert not np.allclose(out.waveform, speech_like.waveform)


def test_nonstationary_noise_at_snr(speech_like):
    out = NonstationaryNoise(snr_db=12.0, segment_s=0.3, seed=5)(speech_like)
    residual = out.waveform - speech_like.waveform
    snr = 10 * np.log10(
        np.mean(speech_like.waveform ** 2) / (np.mean(residual ** 2) + 1e-18)
    )
    assert abs(snr - 12.0) < 1.5


def test_nonstationary_variance_changes_over_time(speech_like):
    out = NonstationaryNoise(snr_db=5.0, segment_s=0.2, seed=7)(speech_like)
    residual = out.waveform - speech_like.waveform
    n = len(residual)
    first = np.var(residual[: n // 3])
    last = np.var(residual[2 * n // 3 :])
    # Nonstationary ⇒ segment energies are not near-identical
    ratio = max(first, last) / (min(first, last) + 1e-18)
    assert ratio > 1.05


def test_delay_add_comb_filters(speech_like):
    out = DelayAndAddInterference(n_taps=3, max_delay_ms=30.0, tap_gain=0.4, seed=4)(
        speech_like
    )
    assert len(out.waveform) == len(speech_like.waveform)
    assert not np.allclose(out.waveform, speech_like.waveform)


def test_synthetic_rir_shape_and_decay():
    sr = 16000
    ir = synthetic_rir(sr, rt60_s=0.4, seed=1)
    assert len(ir) == int(0.4 * sr) + int(0.05 * sr)
    assert np.max(np.abs(ir)) == pytest.approx(1.0, abs=1e-6)
    # Direct path first (or at least early energy dominates late energy)
    e_early = np.mean(ir[: int(0.05 * sr)] ** 2)
    e_late = np.mean(ir[int(0.25 * sr) :] ** 2)
    assert e_early > e_late


def test_rir_convolution_is_time_variant_and_safe(speech_like):
    out = RoomImpulseResponse(rt60_s=0.5, wet_db=-3.0, seed=6)(speech_like)
    assert len(out.waveform) == len(speech_like.waveform)
    assert np.all(np.isfinite(out.waveform))
    assert not np.allclose(out.waveform, speech_like.waveform, atol=1e-6)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        BabbleNoise(n_talkers=0)
    with pytest.raises(ValueError):
        HarmonicInterference(f0_hz=-10.0, n_harmonics=4)
    with pytest.raises(ValueError):
        RoomImpulseResponse(rt60_s=0.0)
    with pytest.raises(ValueError):
        NonstationaryNoise(segment_s=0.0)
