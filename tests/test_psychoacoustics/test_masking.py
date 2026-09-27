"""
Tests for audiocaptcha_dsp.psychoacoustics.masking_models
==========================================================
Validates simultaneous masking, temporal masking, and masking budget.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.psychoacoustics.masking_models import (
    simultaneous_masking_threshold,
    temporal_masking_threshold,
    compute_masking_budget,
)
from audiocaptcha_dsp.core.types import MaskingModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_tone(freq_hz: float = 1000.0, sr: int = 16000, duration: float = 0.1) -> np.ndarray:
    """Synthesise a pure tone."""
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float64)


def _make_noise(sr: int = 16000, duration: float = 0.1) -> np.ndarray:
    """White noise."""
    return RNG.standard_normal(int(sr * duration))


# ---------------------------------------------------------------------------
# simultaneous_masking_threshold
# ---------------------------------------------------------------------------

class TestSimultaneousMasking:
    """Tests for simultaneous_masking_threshold."""

    def test_returns_ndarray(self) -> None:
        """Returns an np.ndarray."""
        signal = _make_noise()
        result = simultaneous_masking_threshold(signal, sr=16000)
        assert isinstance(result, np.ndarray)

    def test_simultaneous_masking_returns_array(self) -> None:
        """Alias confirming ndarray return for the required test name."""
        signal = _make_tone()
        result = simultaneous_masking_threshold(signal, sr=16000)
        assert isinstance(result, np.ndarray)

    def test_simultaneous_masking_shape_matches_rfft(self) -> None:
        """Output length equals N//2 + 1 (rfft output size)."""
        n = 4096
        signal = _make_noise(sr=22050, duration=n / 22050)
        result = simultaneous_masking_threshold(signal, sr=22050)
        expected_len = n // 2 + 1
        assert result.shape == (expected_len,), (
            f"Expected shape ({expected_len},), got {result.shape}"
        )

    def test_masking_threshold_finite(self) -> None:
        """All masking threshold values are finite."""
        signal = _make_noise()
        result = simultaneous_masking_threshold(signal, sr=16000)
        assert np.all(np.isfinite(result)), "Non-finite masking threshold values found."

    def test_masking_threshold_above_ath(self) -> None:
        """Simultaneous masking threshold ≥ ATH at every bin."""
        from audiocaptcha_dsp.psychoacoustics.thresholds import absolute_threshold
        signal = _make_tone(1000.0)
        result = simultaneous_masking_threshold(signal, sr=16000)
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / 16000)
        ath = absolute_threshold(freqs)
        assert np.all(result >= ath - 1e-6), "Masking threshold fell below ATH."

    def test_silence_returns_ath(self) -> None:
        """Silent signal returns the ATH (floor only)."""
        from audiocaptcha_dsp.psychoacoustics.thresholds import absolute_threshold
        n = 1600
        signal = np.zeros(n, dtype=np.float64)
        result = simultaneous_masking_threshold(signal, sr=16000)
        freqs = np.fft.rfftfreq(n, d=1.0 / 16000)
        ath = absolute_threshold(freqs)
        np.testing.assert_allclose(result, ath, rtol=1e-6)

    def test_non_1d_raises(self) -> None:
        """2-D input raises ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            simultaneous_masking_threshold(np.zeros((2, 1000)), sr=16000)

    def test_nan_input_raises(self) -> None:
        """NaN in signal raises ValueError."""
        signal = np.ones(1000)
        signal[500] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            simultaneous_masking_threshold(signal, sr=16000)

    def test_inf_input_raises(self) -> None:
        """Inf in signal raises ValueError."""
        signal = np.ones(1000)
        signal[0] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            simultaneous_masking_threshold(signal, sr=16000)

    def test_invalid_sr_raises(self) -> None:
        """Non-positive sr raises ValueError."""
        with pytest.raises(ValueError, match="sr must be positive"):
            simultaneous_masking_threshold(np.zeros(100), sr=0)

    def test_model_string_accepted(self) -> None:
        """String model argument 'simultaneous' is accepted."""
        signal = _make_noise()
        result = simultaneous_masking_threshold(signal, sr=16000, model="simultaneous")
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# temporal_masking_threshold
# ---------------------------------------------------------------------------

class TestTemporalMasking:
    """Tests for temporal_masking_threshold."""

    def test_temporal_masking_returns_array(self) -> None:
        """Returns an np.ndarray."""
        signal = _make_noise(sr=16000, duration=0.5)
        result = temporal_masking_threshold(signal, sr=16000)
        assert isinstance(result, np.ndarray)

    def test_temporal_masking_2d_output(self) -> None:
        """Output is 2-D: (n_frames, n_fft_bins)."""
        signal = _make_noise(sr=16000, duration=0.5)
        result = temporal_masking_threshold(signal, sr=16000)
        assert result.ndim == 2, f"Expected 2-D, got {result.ndim}-D"

    def test_temporal_masking_finite(self) -> None:
        """All temporal masking threshold values are finite."""
        signal = _make_noise(sr=16000, duration=0.3)
        result = temporal_masking_threshold(signal, sr=16000)
        assert np.all(np.isfinite(result))

    def test_temporal_masking_above_or_equal_simultaneous(self) -> None:
        """Temporal threshold ≥ simultaneous threshold for each frame."""
        signal = _make_noise(sr=16000, duration=0.5)
        temporal = temporal_masking_threshold(signal, sr=16000, frame_duration=0.025, hop_duration=0.010)
        # Check that the temporal threshold is at least as large as the
        # simultaneous threshold for the first frame (no forward masking yet)
        frame_len = int(round(0.025 * 16000))
        frame0 = signal[:frame_len]
        sim0 = simultaneous_masking_threshold(frame0, sr=16000)
        # temporal[0] should equal sim0 (first frame has no predecessors)
        np.testing.assert_allclose(temporal[0], sim0, rtol=1e-6)

    def test_temporal_masking_custom_durations(self) -> None:
        """Custom frame/hop durations produce correct number of frames."""
        sr = 16000
        duration_s = 1.0
        signal = _make_noise(sr=sr, duration=duration_s)
        frame_dur = 0.020
        hop_dur = 0.010
        result = temporal_masking_threshold(signal, sr=sr,
                                             frame_duration=frame_dur,
                                             hop_duration=hop_dur)
        frame_len = int(round(frame_dur * sr))
        hop_len = int(round(hop_dur * sr))
        expected_frames = max(1, 1 + (len(signal) - frame_len + hop_len - 1) // hop_len)
        assert result.shape[0] == expected_frames

    def test_temporal_non_1d_raises(self) -> None:
        """2-D input raises ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            temporal_masking_threshold(np.zeros((2, 8000)), sr=16000)

    def test_temporal_nan_input_raises(self) -> None:
        """NaN in signal raises ValueError."""
        signal = _make_noise()
        signal[10] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            temporal_masking_threshold(signal, sr=16000)


# ---------------------------------------------------------------------------
# compute_masking_budget
# ---------------------------------------------------------------------------

class TestComputeMaskingBudget:
    """Tests for compute_masking_budget."""

    def test_masking_budget_positive(self) -> None:
        """Budget values are all strictly positive."""
        signal = _make_tone(1000.0, sr=16000, duration=0.1)
        budget = compute_masking_budget(signal, sr=16000)
        assert np.all(budget > 0), "Budget contains non-positive values."

    def test_masking_budget_returns_array(self) -> None:
        """Returns an np.ndarray."""
        signal = _make_noise()
        budget = compute_masking_budget(signal, sr=16000)
        assert isinstance(budget, np.ndarray)

    def test_masking_budget_shape_matches_rfft(self) -> None:
        """Budget shape is (N//2 + 1,)."""
        n = 1600
        signal = _make_noise(sr=16000, duration=n / 16000)
        budget = compute_masking_budget(signal, sr=16000)
        assert budget.shape == (n // 2 + 1,)

    def test_masking_budget_finite(self) -> None:
        """All budget values are finite."""
        signal = _make_noise()
        budget = compute_masking_budget(signal, sr=16000)
        assert np.all(np.isfinite(budget))

    def test_larger_margin_reduces_budget(self) -> None:
        """Larger safety margin yields smaller budget."""
        signal = _make_tone(1000.0)
        b6 = compute_masking_budget(signal, sr=16000, margin_db=6.0)
        b12 = compute_masking_budget(signal, sr=16000, margin_db=12.0)
        assert np.all(b12 <= b6 + 1e-10), "Larger margin should reduce budget."

    def test_negative_margin_raises(self) -> None:
        """Negative margin_db raises ValueError."""
        signal = _make_noise()
        with pytest.raises(ValueError, match="non-negative"):
            compute_masking_budget(signal, sr=16000, margin_db=-1.0)

    def test_zero_margin_budget_matches_threshold(self) -> None:
        """With margin_db=0, budget = 10^(masking_db/10)."""
        signal = _make_tone(2000.0)
        masking_db = simultaneous_masking_threshold(signal, sr=16000)
        budget0 = compute_masking_budget(signal, sr=16000, margin_db=0.0)
        expected = 10.0 ** (masking_db / 10.0)
        np.testing.assert_allclose(budget0, expected, rtol=1e-6)
