from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.novel import (
    PhonemeAwarePerturbation,
    PhonemeSegmentDropout,
    MultiDomainPerturbation,
    AdaptiveFormantPerturbation,
    CAPTCHAOptimalTransform,
    DefenseRobustTransform,
)


@pytest.fixture
def speech_like_signal() -> Signal:
    """Generate a 1-second speech-like harmonic signal at 16 kHz."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Fundamental + harmonics with amplitude envelope
    f0 = 150.0
    waveform = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    envelope = np.sin(np.pi * t) ** 2  # smooth onset/offset
    waveform *= envelope
    return Signal(waveform=waveform, sample_rate=sr, metadata={"transcript": "test speech"})


class TestNovelTransforms:
    def test_phoneme_aware_perturbation(self, speech_like_signal: Signal) -> None:
        transform = PhonemeAwarePerturbation(
            vowel_scale=0.8,
            consonant_scale=0.2,
            margin_db=6.0,
            seed=42,
        )
        out = transform(speech_like_signal)
        assert isinstance(out, Signal)
        assert out.sample_rate == speech_like_signal.sample_rate
        assert len(out.waveform) == len(speech_like_signal.waveform)
        assert not np.any(np.isnan(out.waveform))
        assert not np.any(np.isinf(out.waveform))
        # Signal was modified
        assert not np.allclose(out.waveform, speech_like_signal.waveform)

    def test_phoneme_segment_dropout(self, speech_like_signal: Signal) -> None:
        transform = PhonemeSegmentDropout(drop_prob=0.3, seed=42)
        out = transform(speech_like_signal)
        assert isinstance(out, Signal)
        assert out.sample_rate == speech_like_signal.sample_rate
        assert len(out.waveform) == len(speech_like_signal.waveform)
        assert not np.any(np.isnan(out.waveform))
        assert not np.any(np.isinf(out.waveform))

    def test_multi_domain_perturbation(self, speech_like_signal: Signal) -> None:
        transform = MultiDomainPerturbation(
            temporal_scale=0.2,
            spectral_scale=0.2,
            margin_db=6.0,
            seed=42,
        )
        out = transform(speech_like_signal)
        assert isinstance(out, Signal)
        assert out.sample_rate == speech_like_signal.sample_rate
        assert len(out.waveform) == len(speech_like_signal.waveform)
        assert not np.any(np.isnan(out.waveform))
        assert not np.any(np.isinf(out.waveform))
        assert not np.allclose(out.waveform, speech_like_signal.waveform)

    def test_adaptive_formant_perturbation(self, speech_like_signal: Signal) -> None:
        transform = AdaptiveFormantPerturbation(
            formant_shift_hz=40.0,
            num_formants=2,
            seed=42,
        )
        out = transform(speech_like_signal)
        assert isinstance(out, Signal)
        assert out.sample_rate == speech_like_signal.sample_rate
        assert len(out.waveform) == len(speech_like_signal.waveform)
        assert not np.any(np.isnan(out.waveform))
        assert not np.any(np.isinf(out.waveform))

    def test_captcha_optimal_transform(self, speech_like_signal: Signal) -> None:
        transform = CAPTCHAOptimalTransform(
            psychoacoustic_margin_db=6.0,
            temporal_jitter_ms=1.5,
            spectral_dropout_rate=0.04,
            bark_bands_to_drop=2,
            seed=42,
        )
        out = transform(speech_like_signal)
        assert isinstance(out, Signal)
        assert out.sample_rate == speech_like_signal.sample_rate
        assert len(out.waveform) == len(speech_like_signal.waveform)
        assert not np.any(np.isnan(out.waveform))
        assert not np.any(np.isinf(out.waveform))
        assert not np.allclose(out.waveform, speech_like_signal.waveform)

    def test_defense_robust_transform(self, speech_like_signal: Signal) -> None:
        transform = DefenseRobustTransform(
            noise_level_db=12.0,
            lowpass_hz=6000.0,
            harmonic_attenuation=0.5,
            seed=42,
        )
        out = transform(speech_like_signal)
        assert isinstance(out, Signal)
        assert out.sample_rate == speech_like_signal.sample_rate
        assert len(out.waveform) == len(speech_like_signal.waveform)
        assert not np.any(np.isnan(out.waveform))
        assert not np.any(np.isinf(out.waveform))
        assert not np.allclose(out.waveform, speech_like_signal.waveform)

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError):
            PhonemeAwarePerturbation(vowel_scale=-1.0)
        with pytest.raises(ValueError):
            PhonemeSegmentDropout(drop_prob=1.5)
        with pytest.raises(ValueError):
            MultiDomainPerturbation(temporal_scale=-0.1)
        with pytest.raises(ValueError):
            AdaptiveFormantPerturbation(formant_shift_hz=-10.0)
        with pytest.raises(ValueError):
            CAPTCHAOptimalTransform(psychoacoustic_margin_db=-5.0)
        with pytest.raises(ValueError):
            DefenseRobustTransform(noise_level_db=-1.0)
