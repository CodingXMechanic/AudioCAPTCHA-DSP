import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.psychoacoustic.constrained import (
    PsychoacousticNoiseInjection,
    BarkScalePerturbation,
)
from audiocaptcha_dsp.transforms.psychoacoustic.extensions import (
    ERBScalePerturbation,
    TemporalMaskingPerturbation,
    CombinedTemporalSpectralMasking,
    LoudnessPreservingPerturbation,
    PerFrameMaskingBudget,
    SignalDependentThreshold,
    SpeechAwareMasking,
)


@pytest.fixture
def speech_like() -> Signal:
    sr = 16000
    t = np.arange(int(1.2 * sr)) / sr
    wave = (
        0.5 * np.sin(2 * np.pi * 160 * t)
        + 0.3 * np.sin(2 * np.pi * 320 * t)
        + 0.2 * np.sin(2 * np.pi * 900 * t)
    )
    wave *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    return Signal(waveform=wave, sample_rate=sr, metadata={"transcript": "test"})


ALL_TRANSFORMS = [
    PsychoacousticNoiseInjection(margin_db=10.0, seed=1),
    BarkScalePerturbation(perturbation_scale=0.2, seed=1),
    ERBScalePerturbation(perturbation_scale=0.2, seed=1),
    TemporalMaskingPerturbation(margin_db=6.0, seed=1),
    CombinedTemporalSpectralMasking(margin_db=6.0, seed=1),
    LoudnessPreservingPerturbation(margin_db=6.0, seed=1),
    PerFrameMaskingBudget(margin_db=6.0, allocation_exponent=1.0, seed=1),
    SignalDependentThreshold(margin_db=10.0, seed=1),
    SpeechAwareMasking(voiced_margin_db=8.0, unvoiced_margin_db=3.0, seed=1),
]


def test_family_f_shape_and_finiteness(speech_like):
    for tr in ALL_TRANSFORMS:
        out = tr(speech_like)
        assert out.sample_rate == speech_like.sample_rate
        assert len(out.waveform) == len(speech_like.waveform)
        assert np.all(np.isfinite(out.waveform))
        assert not np.allclose(out.waveform, speech_like.waveform, atol=1e-12)


def test_family_f_perturbation_stays_small(speech_like):
    """Psychoacoustic transforms must perturb, but stay well below full-scale."""
    for tr in ALL_TRANSFORMS:
        out = tr(speech_like)
        delta = out.waveform - speech_like.waveform
        # Perturbation is bounded relative to signal RMS (margins constrain it)
        snr = 10 * np.log10(
            np.mean(speech_like.waveform ** 2) / (np.mean(delta ** 2) + 1e-18)
        )
        assert snr > 0.0, f"{type(tr).__name__} perturbation too large (SNR={snr:.1f} dB)"


def test_larger_margin_reduces_perturbation(speech_like):
    """λ margin ordering: margin 20 dB ⇒ less added energy than margin 2 dB."""
    small = PsychoacousticNoiseInjection(margin_db=2.0, seed=42)(speech_like)
    large = PsychoacousticNoiseInjection(margin_db=20.0, seed=42)(speech_like)
    d_small = np.sum((small.waveform - speech_like.waveform) ** 2)
    d_large = np.sum((large.waveform - speech_like.waveform) ** 2)
    assert d_large < d_small


def test_signal_dependent_threshold_records_clipped_bins(speech_like):
    out = SignalDependentThreshold(margin_db=15.0, seed=2)(speech_like)
    key = "psychoacoustic.signal_threshold.clipped_bins"
    assert key in out.metadata
    assert out.metadata[key] >= 0


def test_loudness_preserving_keeps_envelope(speech_like):
    out = LoudnessPreservingPerturbation(margin_db=6.0, seed=3)(speech_like)
    frame = int(0.02 * speech_like.sample_rate)

    def env(x):
        n = len(x) // frame
        return np.array([np.sqrt(np.mean(x[i * frame : (i + 1) * frame] ** 2)) for i in range(n)])

    e_in = env(speech_like.waveform)
    e_out = env(out.waveform)
    # Short-time loudness within ±3 dB of original
    ratio_db = 20 * np.log10((e_out + 1e-12) / (e_in + 1e-12))
    assert np.mean(np.abs(ratio_db)) < 3.0


def test_speech_aware_uses_voiced_frames(speech_like):
    out = SpeechAwareMasking(voiced_margin_db=8.0, unvoiced_margin_db=3.0, seed=4)(speech_like)
    key = "psychoacoustic.speech_aware.n_voiced_frames"
    assert key in out.metadata
    assert out.metadata[key] > 0


def test_seed_determinism(speech_like):
    a = ERBScalePerturbation(perturbation_scale=0.3, seed=7)(speech_like)
    b = ERBScalePerturbation(perturbation_scale=0.3, seed=7)(speech_like)
    c = ERBScalePerturbation(perturbation_scale=0.3, seed=8)(speech_like)
    assert np.allclose(a.waveform, b.waveform)
    assert not np.allclose(a.waveform, c.waveform)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        PsychoacousticNoiseInjection(margin_db=-1.0)
    with pytest.raises(ValueError):
        ERBScalePerturbation(n_bands=2)
    with pytest.raises(ValueError):
        PerFrameMaskingBudget(allocation_exponent=0.0)
    with pytest.raises(ValueError):
        SpeechAwareMasking(voiced_margin_db=-2.0)
