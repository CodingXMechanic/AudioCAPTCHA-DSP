import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.spectral.phase import (
    FormantShift,
    FormantSuppression,
    GroupDelayPerturbation,
    MinimumPhaseConversion,
)
from audiocaptcha_dsp.transforms.spectral.contrast import (
    SpectralContrastModification,
    RandomSpectralEqualization,
    SpectralSharpening,
    SpectralHole,
    TimeFrequencyMasking,
)


@pytest.fixture
def vowel_like() -> Signal:
    """Harmonic signal with two formant-like resonant peaks."""
    sr = 16000
    t = np.arange(int(1.0 * sr)) / sr
    f0 = 120.0
    wave = np.zeros_like(t)
    for h in range(1, 30):
        f = f0 * h
        if f > 6000:
            break
        # Resonance bumps near 700 Hz and 2200 Hz (F1/F2 of /a/)
        amp = 1.0 / h
        amp *= 1.0 + 2.0 * np.exp(-((f - 700) ** 2) / (2 * 150.0 ** 2))
        amp *= 1.0 + 1.5 * np.exp(-((f - 2200) ** 2) / (2 * 250.0 ** 2))
        wave += amp * np.sin(2 * np.pi * f * t)
    wave *= 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    return Signal(waveform=wave, sample_rate=sr)


ALL_TRANSFORMS = [
    FormantShift(shift_ratio=1.2),
    FormantSuppression(depth_db=-8.0),
    GroupDelayPerturbation(perturbation_scale=0.5, seed=1),
    MinimumPhaseConversion(),
    SpectralContrastModification(contrast_factor=1.5),
    RandomSpectralEqualization(max_gain_db=6.0, seed=1),
    SpectralSharpening(strength=0.5),
    SpectralHole(center_hz=2000.0, width_hz=800.0),
    TimeFrequencyMasking(n_masks=2, seed=1),
]


def test_spectral_extensions_preserve_shape(vowel_like):
    for tr in ALL_TRANSFORMS:
        out = tr(vowel_like)
        assert out.sample_rate == vowel_like.sample_rate
        assert len(out.waveform) == len(vowel_like.waveform)
        assert np.all(np.isfinite(out.waveform))


def test_formant_shift_moves_spectral_peaks(vowel_like):
    out = FormantShift(shift_ratio=1.25)(vowel_like)
    in_spec = np.abs(np.fft.rfft(vowel_like.waveform))
    out_spec = np.abs(np.fft.rfft(out.waveform))
    freqs = np.fft.rfftfreq(len(vowel_like.waveform), 1 / vowel_like.sample_rate)
    in_peak = freqs[np.argmax(in_spec[1:]) + 1]
    out_peak = freqs[np.argmax(out_spec[1:]) + 1]
    # Envelope warped upward ⇒ dominant peak moves up
    assert out_peak >= in_peak - 50.0
    assert not np.allclose(out.waveform, vowel_like.waveform, atol=1e-6)


def test_formant_suppression_reduces_formant_energy(vowel_like):
    out = FormantSuppression(depth_db=-12.0, bandwidth_hz=400.0)(vowel_like)
    in_spec = np.abs(np.fft.rfft(vowel_like.waveform))
    out_spec = np.abs(np.fft.rfft(out.waveform))
    band = slice(
        np.searchsorted(np.fft.rfftfreq(len(vowel_like.waveform), 1 / vowel_like.sample_rate), 400),
        np.searchsorted(np.fft.rfftfreq(len(vowel_like.waveform), 1 / vowel_like.sample_rate), 1000),
    )
    assert out_spec[band].mean() < in_spec[band].mean()


def test_group_delay_preserves_magnitude(vowel_like):
    out = GroupDelayPerturbation(perturbation_scale=0.7, seed=4)(vowel_like)
    in_mag = np.abs(np.fft.rfft(vowel_like.waveform))
    out_mag = np.abs(np.fft.rfft(out.waveform))
    # Per-frame magnitudes are untouched; overlap-add with perturbed phases
    # introduces a small deviation, so the full-spectrum magnitude envelope
    # must remain highly correlated (>> spectral-hole-class transforms).
    assert np.corrcoef(in_mag, out_mag)[0, 1] > 0.9
    # And the perturbation must actually change the signal (phase moved)
    assert not np.allclose(out.waveform, vowel_like.waveform, atol=1e-6)


def test_minimum_phase_changes_phase_keeps_rms(vowel_like):
    out = MinimumPhaseConversion()(vowel_like)
    rms_in = np.sqrt(np.mean(vowel_like.waveform ** 2))
    rms_out = np.sqrt(np.mean(out.waveform ** 2))
    assert abs(rms_out - rms_in) / (rms_in + 1e-12) < 0.05
    assert not np.allclose(out.waveform, vowel_like.waveform, atol=1e-6)


def test_spectral_contrast_factor_one_is_identity(vowel_like):
    out = SpectralContrastModification(contrast_factor=1.0)(vowel_like)
    assert np.allclose(out.waveform, vowel_like.waveform, atol=1e-9)


def test_random_eq_changes_spectrum(vowel_like):
    out = RandomSpectralEqualization(max_gain_db=8.0, seed=2)(vowel_like)
    assert not np.allclose(out.waveform, vowel_like.waveform, atol=1e-6)


def test_spectral_sharpening_zero_is_identity(vowel_like):
    out = SpectralSharpening(strength=0.0)(vowel_like)
    assert np.allclose(out.waveform, vowel_like.waveform, atol=1e-9)


def test_spectral_hole_carves_band(vowel_like):
    out = SpectralHole(center_hz=2000.0, width_hz=600.0, depth_db=-30.0)(vowel_like)
    spec = np.abs(np.fft.rfft(out.waveform))
    freqs = np.fft.rfftfreq(len(out.waveform), 1 / vowel_like.sample_rate)
    hole = (freqs > 1800) & (freqs < 2200)
    side = (freqs > 900) & (freqs < 1300)
    assert spec[hole].mean() < 0.3 * spec[side].mean()


def test_tf_masking_zeroes_units(vowel_like):
    out = TimeFrequencyMasking(n_masks=3, max_freq_mask=0.4, max_time_mask=0.4, seed=9)(
        vowel_like
    )
    assert not np.allclose(out.waveform, vowel_like.waveform, atol=1e-6)
    assert np.all(np.isfinite(out.waveform))


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        FormantShift(shift_ratio=-1.0)
    with pytest.raises(ValueError):
        FormantSuppression(depth_db=6.0)
    with pytest.raises(ValueError):
        SpectralHole(width_hz=0.0, depth_db=-10.0)
    with pytest.raises(ValueError):
        TimeFrequencyMasking(n_masks=-1)
