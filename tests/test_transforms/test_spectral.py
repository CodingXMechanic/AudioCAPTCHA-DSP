from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.spectral.masking import MaskingInjection
from audiocaptcha_dsp.transforms.spectral.notch import SpectralNotch
from audiocaptcha_dsp.transforms.spectral.warping import FrequencyWarping


class TestMaskingInjection:
    def test_call_returns_signal(self, clean_signal: Signal) -> None:
        transform = MaskingInjection(masker_level_db=20.0, bandwidth_bark=2.0)
        result = transform(clean_signal)
        assert isinstance(result, Signal)

    def test_very_low_level_preserves_signal(self, clean_signal: Signal) -> None:
        transform = MaskingInjection(masker_level_db=-80.0, bandwidth_bark=2.0, seed=42)
        result = transform(clean_signal)
        np.testing.assert_array_almost_equal(result.waveform, clean_signal.waveform, decimal=4)

    def test_nonzero_level_changes_signal(self, clean_signal: Signal) -> None:
        transform = MaskingInjection(masker_level_db=30.0, bandwidth_bark=2.0, seed=42)
        result = transform(clean_signal)
        assert not np.array_equal(result.waveform, clean_signal.waveform)

    def test_output_length_preserved(self, clean_signal: Signal) -> None:
        transform = MaskingInjection(masker_level_db=20.0, bandwidth_bark=3.0)
        result = transform(clean_signal)
        assert result.waveform.shape == clean_signal.waveform.shape

    def test_preserves_sample_rate(self, clean_signal: Signal) -> None:
        transform = MaskingInjection()
        result = transform(clean_signal)
        assert result.sample_rate == clean_signal.sample_rate

    def test_reproducible_with_seed(self, clean_signal: Signal) -> None:
        t1 = MaskingInjection(masker_level_db=25.0, bandwidth_bark=2.0, seed=55)
        t2 = MaskingInjection(masker_level_db=25.0, bandwidth_bark=2.0, seed=55)
        r1 = t1(clean_signal)
        r2 = t2(clean_signal)
        np.testing.assert_array_equal(r1.waveform, r2.waveform)

    def test_different_seeds_differ(self, clean_signal: Signal) -> None:
        t1 = MaskingInjection(masker_level_db=25.0, bandwidth_bark=2.0, seed=1)
        t2 = MaskingInjection(masker_level_db=25.0, bandwidth_bark=2.0, seed=2)
        r1 = t1(clean_signal)
        r2 = t2(clean_signal)
        assert not np.array_equal(r1.waveform, r2.waveform)

    def test_higher_level_increases_distortion(self, clean_signal: Signal) -> None:
        t_low = MaskingInjection(masker_level_db=10.0, bandwidth_bark=2.0, seed=42)
        t_high = MaskingInjection(masker_level_db=40.0, bandwidth_bark=2.0, seed=42)
        r_low = t_low(clean_signal)
        r_high = t_high(clean_signal)
        diff_low = np.sqrt(np.mean((r_low.waveform - clean_signal.waveform) ** 2))
        diff_high = np.sqrt(np.mean((r_high.waveform - clean_signal.waveform) ** 2))
        assert diff_high > diff_low

    def test_center_bark_affects_spectrum(self, clean_signal: Signal) -> None:
        t_low = MaskingInjection(masker_level_db=30.0, bandwidth_bark=1.0, center_bark=2.0, seed=42)
        t_high = MaskingInjection(masker_level_db=30.0, bandwidth_bark=1.0, center_bark=12.0, seed=42)
        r_low = t_low(clean_signal)
        r_high = t_high(clean_signal)
        assert not np.array_equal(r_low.waveform, r_high.waveform)

    def test_metadata_fields(self, clean_signal: Signal) -> None:
        transform = MaskingInjection(masker_level_db=25.0, bandwidth_bark=3.0, center_bark=6.0)
        result = transform(clean_signal)
        assert result.metadata["masker_applied"] is True
        assert result.metadata["masker_level_db"] == 25.0
        assert result.metadata["masker_bandwidth_bark"] == 3.0
        assert result.metadata["masker_center_bark"] == 6.0
        assert "masker_center_hz" in result.metadata

    def test_stereo_preserved(self, stereo_signal: Signal) -> None:
        transform = MaskingInjection(masker_level_db=20.0, bandwidth_bark=2.0, seed=42)
        result = transform(stereo_signal)
        assert result.waveform.ndim == 2


class TestSpectralNotch:
    def test_call_returns_signal(self, clean_signal: Signal) -> None:
        transform = SpectralNotch(frequency_hz=1000.0, bandwidth_hz=100.0)
        result = transform(clean_signal)
        assert isinstance(result, Signal)

    def test_attenuates_notch_frequency(self, sample_rate: int) -> None:
        freq = 1000.0
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        waveform = np.sin(2.0 * np.pi * freq * t)
        signal = Signal(waveform=waveform, sample_rate=sample_rate)
        transform = SpectralNotch(frequency_hz=freq, bandwidth_hz=50.0)
        result = transform(signal)
        assert result.rms < signal.rms * 0.5

    def test_preserves_outside_notch_frequency(self) -> None:
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        waveform = np.sin(2.0 * np.pi * 440.0 * t)
        signal = Signal(waveform=waveform, sample_rate=sr)
        transform = SpectralNotch(frequency_hz=4000.0, bandwidth_hz=100.0)
        result = transform(signal)
        ratio = result.rms / signal.rms
        assert ratio > 0.8

    def test_output_length_preserved(self, clean_signal: Signal) -> None:
        transform = SpectralNotch(frequency_hz=500.0, bandwidth_hz=50.0)
        result = transform(clean_signal)
        assert result.waveform.shape == clean_signal.waveform.shape

    def test_preserves_sample_rate(self, clean_signal: Signal) -> None:
        transform = SpectralNotch()
        result = transform(clean_signal)
        assert result.sample_rate == clean_signal.sample_rate

    def test_metadata_fields(self, clean_signal: Signal) -> None:
        transform = SpectralNotch(frequency_hz=2000.0, bandwidth_hz=200.0)
        result = transform(clean_signal)
        assert result.metadata["notch_applied"] is True
        assert result.metadata["notch_frequency_hz"] == 2000.0
        assert result.metadata["notch_bandwidth_hz"] == 200.0
        assert result.metadata["notch_q_factor"] == 10.0

    def test_invalid_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SpectralNotch(frequency_hz=0.0, bandwidth_hz=100.0)

    def test_invalid_bandwidth_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SpectralNotch(frequency_hz=1000.0, bandwidth_hz=0.0)

    def test_nyquist_violation_raises(self, clean_signal: Signal) -> None:
        nyquist = clean_signal.sample_rate / 2.0
        with pytest.raises(ValueError, match="Nyquist"):
            SpectralNotch(frequency_hz=nyquist, bandwidth_hz=100.0)(clean_signal)

    def test_stereo_notch_all_channels(self, stereo_signal: Signal) -> None:
        transform = SpectralNotch(frequency_hz=500.0, bandwidth_hz=50.0)
        result = transform(stereo_signal)
        assert result.waveform.ndim == 2
        assert result.waveform.shape[0] == stereo_signal.waveform.shape[0]

    def test_silence_stays_silent(self, silent_signal: Signal) -> None:
        transform = SpectralNotch(frequency_hz=500.0, bandwidth_hz=50.0)
        result = transform(silent_signal)
        np.testing.assert_array_almost_equal(result.waveform, np.zeros_like(result.waveform))


class TestFrequencyWarping:
    def test_call_returns_signal(self, clean_signal: Signal) -> None:
        transform = FrequencyWarping(alpha=0.0)
        result = transform(clean_signal)
        assert isinstance(result, Signal)

    def test_zero_alpha_preserves_signal(self, clean_signal: Signal) -> None:
        transform = FrequencyWarping(alpha=0.0)
        result = transform(clean_signal)
        np.testing.assert_array_almost_equal(result.waveform, clean_signal.waveform)
        assert result.metadata["warping_applied"] is False

    def test_nonzero_alpha_changes_signal(self, clean_signal: Signal) -> None:
        transform = FrequencyWarping(alpha=0.3)
        result = transform(clean_signal)
        assert not np.array_equal(result.waveform, clean_signal.waveform)
        assert result.metadata["warping_applied"] is True

    def test_negative_alpha_changes_signal(self, clean_signal: Signal) -> None:
        transform = FrequencyWarping(alpha=-0.3)
        result = transform(clean_signal)
        assert not np.array_equal(result.waveform, clean_signal.waveform)

    def test_output_length_preserved(self, clean_signal: Signal) -> None:
        transform = FrequencyWarping(alpha=0.5)
        result = transform(clean_signal)
        assert result.waveform.shape == clean_signal.waveform.shape

    def test_preserves_sample_rate(self, clean_signal: Signal) -> None:
        transform = FrequencyWarping(alpha=0.2)
        result = transform(clean_signal)
        assert result.sample_rate == clean_signal.sample_rate

    def test_nonzero_alpha_changes_spectrum(self, sample_rate: int) -> None:
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        waveform = (
            0.4 * np.sin(2 * np.pi * 300 * t)
            + 0.3 * np.sin(2 * np.pi * 1200 * t)
            + 0.2 * np.sin(2 * np.pi * 3000 * t)
            + 0.1 * np.sin(2 * np.pi * 6000 * t)
        )
        signal = Signal(waveform=waveform, sample_rate=sample_rate)
        t_identity = FrequencyWarping(alpha=0.0)
        t_warp = FrequencyWarping(alpha=0.3)
        r_identity = t_identity(signal)
        r_warp = t_warp(signal)
        np.testing.assert_array_almost_equal(
            r_identity.waveform, signal.waveform, decimal=10
        )
        assert not np.allclose(r_warp.waveform, signal.waveform)

    def test_metadata_fields(self, clean_signal: Signal) -> None:
        transform = FrequencyWarping(alpha=0.4)
        result = transform(clean_signal)
        assert result.metadata["warping_alpha"] == 0.4

    def test_stereo_handled(self, stereo_signal: Signal) -> None:
        transform = FrequencyWarping(alpha=0.2)
        result = transform(stereo_signal)
        assert isinstance(result, Signal)
        assert result.waveform.ndim == 2

    def test_silence_stays_silent(self, silent_signal: Signal) -> None:
        transform = FrequencyWarping(alpha=0.5)
        result = transform(silent_signal)
        np.testing.assert_array_almost_equal(result.waveform, np.zeros_like(result.waveform))
