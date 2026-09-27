"""
Tests for audiocaptcha_dsp.psychoacoustics.thresholds
======================================================
Validates the ISO 226 ATH curve and compute_masking_threshold_db.
"""
from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.psychoacoustics.thresholds import (
    absolute_threshold,
    compute_masking_threshold_db,
)
from audiocaptcha_dsp.core.types import ThresholdModel


# ---------------------------------------------------------------------------
# absolute_threshold tests
# ---------------------------------------------------------------------------

class TestAbsoluteThreshold:
    """Tests for the absolute_threshold (ATH) function."""

    def test_returns_array(self) -> None:
        """absolute_threshold returns an ndarray for array input."""
        freqs = np.array([250.0, 1000.0, 4000.0])
        result = absolute_threshold(freqs)
        assert isinstance(result, np.ndarray)

    def test_ath_returns_array_same_shape(self) -> None:
        """Output shape equals input shape."""
        freqs = np.linspace(20.0, 20000.0, 200)
        result = absolute_threshold(freqs)
        assert result.shape == freqs.shape

    def test_ath_scalar_input(self) -> None:
        """Scalar input is accepted and returns a 1-element array."""
        result = absolute_threshold(1000.0)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_ath_has_minimum_near_4khz(self) -> None:
        """ATH is lowest in the 2–5 kHz region (peak human hearing sensitivity)."""
        freqs = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0,
                          5000.0, 8000.0, 12000.0, 16000.0])
        ath = absolute_threshold(freqs)
        min_idx = np.argmin(ath)
        # Minimum should be in the 2–5 kHz range
        assert 2000.0 <= freqs[min_idx] <= 6000.0, (
            f"ATH minimum at {freqs[min_idx]} Hz, expected 2–6 kHz. "
            f"ATH values: {ath}"
        )

    def test_ath_is_higher_at_very_low_freq(self) -> None:
        """ATH at very low frequencies (< 200 Hz) should be much higher than at 3–4 kHz."""
        low_ath = absolute_threshold(np.array([50.0]))[0]
        mid_ath = absolute_threshold(np.array([3500.0]))[0]
        assert low_ath > mid_ath, (
            f"ATH at 50 Hz ({low_ath:.1f} dB) should exceed ATH at 3500 Hz ({mid_ath:.1f} dB)."
        )

    def test_ath_is_higher_at_very_high_freq(self) -> None:
        """ATH at very high frequencies (> 15 kHz) should be much higher than at 3–4 kHz."""
        high_ath = absolute_threshold(np.array([18000.0]))[0]
        mid_ath = absolute_threshold(np.array([3500.0]))[0]
        assert high_ath > mid_ath, (
            f"ATH at 18 kHz ({high_ath:.1f} dB) should exceed ATH at 3500 Hz ({mid_ath:.1f} dB)."
        )

    def test_ath_at_zero_hz(self) -> None:
        """Zero Hz (DC) should return a large positive value (inaudible)."""
        result = absolute_threshold(np.array([0.0]))
        assert result[0] >= 50.0, f"ATH at 0 Hz should be large, got {result[0]:.1f} dB."

    def test_ath_output_is_finite(self) -> None:
        """All ATH values for standard audio frequencies should be finite."""
        freqs = np.logspace(np.log10(20), np.log10(20000), 100)
        result = absolute_threshold(freqs)
        assert np.all(np.isfinite(result)), "ATH returned non-finite values."

    def test_ath_iso226_model_string(self) -> None:
        """ThresholdModel string 'iso226' is accepted."""
        freqs = np.array([1000.0])
        result = absolute_threshold(freqs, model="iso226")
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_ath_invalid_model_raises(self) -> None:
        """Invalid model string raises ValueError."""
        with pytest.raises(ValueError):
            absolute_threshold(np.array([1000.0]), model="bogus_model")  # type: ignore[arg-type]

    def test_ath_negative_freq_raises(self) -> None:
        """Negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            absolute_threshold(np.array([-100.0]))

    def test_ath_nan_input_raises(self) -> None:
        """NaN in freq_hz raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            absolute_threshold(np.array([np.nan]))

    def test_ath_clipped_to_valid_range(self) -> None:
        """ATH output is clipped to [−20, 120] dB."""
        freqs = np.array([1.0, 50.0, 1000.0, 19000.0])
        result = absolute_threshold(freqs)
        assert np.all(result >= -20.0)
        assert np.all(result <= 120.0)


# ---------------------------------------------------------------------------
# compute_masking_threshold_db tests
# ---------------------------------------------------------------------------

class TestComputeMaskingThresholdDb:
    """Tests for compute_masking_threshold_db."""

    def _make_spectrum_freqs(self, n: int = 512, sr: int = 16000):
        """Helper: create a simple power spectrum and frequency axis."""
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        # Flat-ish power spectrum with a peak at 1 kHz
        power = np.ones(len(freqs)) * 1e-4
        return power, freqs

    def test_masking_threshold_shape(self) -> None:
        """Output shape matches input spectrum shape."""
        power, freqs = self._make_spectrum_freqs(512, 16000)
        result = compute_masking_threshold_db(power, freqs, sr=16000)
        assert result.shape == freqs.shape

    def test_masking_threshold_above_ath(self) -> None:
        """Masking threshold is everywhere ≥ ATH."""
        power, freqs = self._make_spectrum_freqs(512, 16000)
        masking = compute_masking_threshold_db(power, freqs, sr=16000)
        ath = absolute_threshold(freqs)
        assert np.all(masking >= ath - 1e-6), (
            "Masking threshold fell below ATH at some frequencies."
        )

    def test_masking_threshold_finite(self) -> None:
        """All masking threshold values are finite."""
        power, freqs = self._make_spectrum_freqs(1024, 22050)
        result = compute_masking_threshold_db(power, freqs, sr=22050)
        assert np.all(np.isfinite(result))

    def test_masking_threshold_silence(self) -> None:
        """Zero power spectrum yields masking threshold equal to ATH."""
        power = np.zeros(257)
        freqs = np.fft.rfftfreq(512, d=1.0 / 16000)
        result = compute_masking_threshold_db(power, freqs, sr=16000)
        ath = absolute_threshold(freqs)
        # For silence, spectrum_db is very low → threshold should equal ATH
        np.testing.assert_allclose(result, ath, rtol=1e-5)

    def test_masking_threshold_shape_mismatch_raises(self) -> None:
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError):
            compute_masking_threshold_db(np.ones(10), np.ones(20), sr=16000)

    def test_masking_threshold_negative_power_raises(self) -> None:
        """Negative power spectrum raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            compute_masking_threshold_db(np.array([-1.0, 1.0, 0.5]),
                                         np.array([100.0, 1000.0, 5000.0]),
                                         sr=16000)
