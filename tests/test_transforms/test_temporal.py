from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.temporal.resampler import Resampler
from audiocaptcha_dsp.transforms.temporal.jitter import TemporalJitter
from audiocaptcha_dsp.transforms.temporal.overlap_add import PerturbedOverlapAdd


class TestResampler:
    def test_call_returns_signal(self, clean_signal: Signal) -> None:
        transform = Resampler(factor=1.0)
        result = transform(clean_signal)
        assert isinstance(result, Signal)

    def test_metadata_set(self, clean_signal: Signal) -> None:
        transform = Resampler(factor=0.95)
        result = transform(clean_signal)
        assert result.metadata["resample_factor"] == 0.95

    def test_identity_preserves_waveform(self, clean_signal: Signal) -> None:
        transform = Resampler(factor=1.0)
        result = transform(clean_signal)
        np.testing.assert_array_almost_equal(result.waveform, clean_signal.waveform)

    def test_upscale_increases_length(self, clean_signal: Signal) -> None:
        transform = Resampler(factor=0.5)
        result = transform(clean_signal)
        assert result.waveform.shape[-1] > clean_signal.waveform.shape[-1]

    def test_downscale_decreases_length(self, clean_signal: Signal) -> None:
        transform = Resampler(factor=2.0)
        result = transform(clean_signal)
        assert result.waveform.shape[-1] < clean_signal.waveform.shape[-1]

    def test_output_length_ratio(self, clean_signal: Signal) -> None:
        factor = 0.8
        transform = Resampler(factor=factor)
        result = transform(clean_signal)
        expected_len = int(round(clean_signal.waveform.shape[-1] / factor))
        assert abs(result.waveform.shape[-1] - expected_len) <= 2

    def test_preserves_sample_rate(self, clean_signal: Signal) -> None:
        transform = Resampler(factor=1.1)
        result = transform(clean_signal)
        assert result.sample_rate == clean_signal.sample_rate

    def test_stereo_resamples_all_channels(self, stereo_signal: Signal) -> None:
        transform = Resampler(factor=0.9)
        result = transform(stereo_signal)
        assert result.waveform.ndim == 2
        assert result.waveform.shape[0] == stereo_signal.waveform.shape[0]

    def test_invalid_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Resampler(factor=0.0)
        with pytest.raises(ValueError, match="positive"):
            Resampler(factor=-1.0)

    def test_original_length_in_metadata(self, clean_signal: Signal) -> None:
        transform = Resampler(factor=0.7)
        result = transform(clean_signal)
        assert result.metadata["original_length"] == clean_signal.waveform.shape[-1]


class TestTemporalJitter:
    def test_call_returns_signal(self, clean_signal: Signal) -> None:
        transform = TemporalJitter(amplitude_ms=10.0, frequency_hz=5.0)
        result = transform(clean_signal)
        assert isinstance(result, Signal)

    def test_metadata_set(self, clean_signal: Signal) -> None:
        transform = TemporalJitter(amplitude_ms=20.0, frequency_hz=10.0)
        result = transform(clean_signal)
        assert result.metadata["jitter_amplitude_ms"] == 20.0
        assert result.metadata["jitter_frequency_hz"] == 10.0

    def test_zero_amplitude_returns_same(self, clean_signal: Signal) -> None:
        transform = TemporalJitter(amplitude_ms=0.0, frequency_hz=5.0)
        result = transform(clean_signal)
        np.testing.assert_array_almost_equal(result.waveform, clean_signal.waveform)
        assert result.metadata["jitter_applied"] is False

    def test_nonzero_amplitude_changes_waveform(self, clean_signal: Signal) -> None:
        transform = TemporalJitter(amplitude_ms=20.0, frequency_hz=5.0, seed=42)
        result = transform(clean_signal)
        assert not np.array_equal(result.waveform, clean_signal.waveform)
        assert result.metadata["jitter_applied"] is True

    def test_output_length_preserved(self, clean_signal: Signal) -> None:
        transform = TemporalJitter(amplitude_ms=15.0, frequency_hz=8.0, seed=42)
        result = transform(clean_signal)
        assert result.waveform.shape == clean_signal.waveform.shape

    def test_preserves_sample_rate(self, clean_signal: Signal) -> None:
        transform = TemporalJitter(amplitude_ms=10.0, frequency_hz=5.0)
        result = transform(clean_signal)
        assert result.sample_rate == clean_signal.sample_rate

    def test_reproducible_with_seed(self, clean_signal: Signal) -> None:
        t1 = TemporalJitter(amplitude_ms=15.0, frequency_hz=5.0, seed=99)
        t2 = TemporalJitter(amplitude_ms=15.0, frequency_hz=5.0, seed=99)
        r1 = t1(clean_signal)
        r2 = t2(clean_signal)
        np.testing.assert_array_equal(r1.waveform, r2.waveform)

    def test_different_seeds_differ(self, clean_signal: Signal) -> None:
        t1 = TemporalJitter(amplitude_ms=15.0, frequency_hz=5.0, seed=1)
        t2 = TemporalJitter(amplitude_ms=15.0, frequency_hz=5.0, seed=2)
        r1 = t1(clean_signal)
        r2 = t2(clean_signal)
        assert not np.array_equal(r1.waveform, r2.waveform)

    def test_negative_amplitude_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            TemporalJitter(amplitude_ms=-5.0, frequency_hz=5.0)

    def test_nonpositive_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TemporalJitter(amplitude_ms=10.0, frequency_hz=0.0)

    def test_mean_stays_similar(self, clean_signal: Signal) -> None:
        transform = TemporalJitter(amplitude_ms=10.0, frequency_hz=5.0, seed=42)
        result = transform(clean_signal)
        orig_mean = np.mean(clean_signal.waveform)
        new_mean = np.mean(result.waveform)
        assert abs(orig_mean - new_mean) < 0.1


class TestPerturbedOverlapAdd:
    def test_call_returns_signal(self, clean_signal: Signal) -> None:
        transform = PerturbedOverlapAdd(window="hann", hop_ratio=0.25)
        result = transform(clean_signal)
        assert isinstance(result, Signal)

    def test_output_length_matches_input(self, clean_signal: Signal) -> None:
        transform = PerturbedOverlapAdd(window_size=512, hop_ratio=0.25)
        result = transform(clean_signal)
        assert result.waveform.shape[-1] == clean_signal.waveform.shape[-1]

    def test_preserves_sample_rate(self, clean_signal: Signal) -> None:
        transform = PerturbedOverlapAdd()
        result = transform(clean_signal)
        assert result.sample_rate == clean_signal.sample_rate

    def test_jitter_changes_output(self, clean_signal: Signal) -> None:
        t_no_jitter = PerturbedOverlapAdd(window_size=512, hop_ratio=0.25, jitter_ms=0.0)
        t_jitter = PerturbedOverlapAdd(window_size=512, hop_ratio=0.25, jitter_ms=5.0, seed=42)
        r1 = t_no_jitter(clean_signal)
        r2 = t_jitter(clean_signal)
        assert not np.array_equal(r1.waveform, r2.waveform)

    def test_reproducible_with_seed(self, clean_signal: Signal) -> None:
        t1 = PerturbedOverlapAdd(window_size=256, hop_ratio=0.25, jitter_ms=5.0, seed=7)
        t2 = PerturbedOverlapAdd(window_size=256, hop_ratio=0.25, jitter_ms=5.0, seed=7)
        r1 = t1(clean_signal)
        r2 = t2(clean_signal)
        np.testing.assert_array_equal(r1.waveform, r2.waveform)

    def test_invalid_window_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Window size"):
            PerturbedOverlapAdd(window_size=1)

    def test_invalid_hop_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="Hop ratio"):
            PerturbedOverlapAdd(hop_ratio=0.0)
        with pytest.raises(ValueError, match="Hop ratio"):
            PerturbedOverlapAdd(hop_ratio=1.5)

    def test_stereo_handled(self, stereo_signal: Signal) -> None:
        transform = PerturbedOverlapAdd(window_size=256, hop_ratio=0.25)
        result = transform(stereo_signal)
        assert isinstance(result, Signal)

    def test_silence_stays_silent(self, silent_signal: Signal) -> None:
        transform = PerturbedOverlapAdd(window_size=256, hop_ratio=0.25)
        result = transform(silent_signal)
        np.testing.assert_array_almost_equal(result.waveform, np.zeros_like(result.waveform))
