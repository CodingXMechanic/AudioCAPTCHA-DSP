from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.evaluation.metrics import (
    compute_wer,
    compute_cer,
    compute_detailed_wer,
    compute_snr,
    compute_si_sdr,
    compute_rmse,
    compute_mse,
    compute_psnr,
    compute_spectral_convergence,
    compute_normalized_cross_correlation,
    compute_log_spectral_distance,
    compute_mbsd,
    compute_stoi_proxy,
    compute_human_asr_gap,
    compute_metrics,
)


class TestMetrics:
    def test_identical_wer_is_zero(self) -> None:
        result = compute_wer("hello world", "hello world")
        assert result == 0.0

    def test_wer_detects_substitutions(self) -> None:
        result = compute_wer("hello world", "hello there")
        assert 0.0 < result <= 1.0

    def test_wer_detects_deletions(self) -> None:
        result = compute_wer("hello world", "hello")
        assert result > 0.0

    def test_identical_cer_is_zero(self) -> None:
        result = compute_cer("hello", "hello")
        assert result == 0.0

    def test_cer_detects_errors(self) -> None:
        result = compute_cer("hello", "hallo")
        assert 0.0 < result <= 1.0

    def test_cer_longer_reference(self) -> None:
        result = compute_cer("hello", "hi")
        assert result > 0.0

    def test_detailed_wer(self) -> None:
        res = compute_detailed_wer("the quick brown fox", "the slow brown fox jumps")
        assert "wer" in res
        assert "substitutions" in res
        assert "insertions" in res
        assert "deletions" in res
        assert res["wer"] > 0.0

    def test_snr_identical_returns_inf(self) -> None:
        x = np.random.randn(1000)
        assert compute_snr(x, x) == float("inf")

    def test_si_sdr_identical_returns_inf(self) -> None:
        x = np.random.randn(1000)
        assert compute_si_sdr(x, x) == float("inf")

    def test_si_sdr_scale_invariant(self) -> None:
        x = np.random.randn(1000)
        # Scaled version should have infinite SI-SDR
        val = compute_si_sdr(x, 2.5 * x)
        assert val > 100.0 or val == float("inf")

    def test_rmse_and_mse(self) -> None:
        x = np.ones(100)
        y = np.zeros(100)
        assert np.isclose(compute_rmse(x, y), 1.0)
        assert np.isclose(compute_mse(x, y), 1.0)

    def test_psnr(self) -> None:
        x = np.ones(100)
        y = np.ones(100)
        assert compute_psnr(x, y) == float("inf")

    def test_spectral_convergence(self) -> None:
        x = np.random.randn(1000)
        assert compute_spectral_convergence(x, x) == 0.0

    def test_normalized_cross_correlation(self) -> None:
        x = np.random.randn(1000)
        assert np.isclose(compute_normalized_cross_correlation(x, x), 1.0)

    def test_lsd_identical_is_zero(self) -> None:
        x = np.random.randn(16000)
        assert np.isclose(compute_log_spectral_distance(x, x, sr=16000), 0.0, atol=1e-5)

    def test_mbsd_identical_is_zero(self) -> None:
        x = np.random.randn(16000)
        assert np.isclose(compute_mbsd(x, x, sr=16000), 0.0, atol=1e-5)

    def test_stoi_proxy_identical_is_one(self) -> None:
        x = np.random.randn(16000)
        assert np.isclose(compute_stoi_proxy(x, x, sr=16000), 1.0, atol=1e-3)

    def test_human_asr_gap(self) -> None:
        # High human success (0.95), low ASR success (0.10) -> large gap (0.85)
        gap = compute_human_asr_gap(0.95, 0.10)
        assert np.isclose(gap, 0.85)

    def test_compute_metrics_dict(self) -> None:
        s1 = Signal(waveform=np.random.randn(16000), sample_rate=16000)
        s2 = Signal(waveform=s1.waveform + 0.01 * np.random.randn(16000), sample_rate=16000)
        metrics = compute_metrics(s1, s2)
        assert "snr_db" in metrics
        assert "si_sdr_db" in metrics
        assert "log_spectral_distance" in metrics
        assert "mbsd" in metrics
        assert "stoi_proxy" in metrics
        assert all(np.isfinite(v) for v in metrics.values())
