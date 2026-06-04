from __future__ import annotations

from audiocaptcha_dsp.evaluation.metrics import compute_wer, compute_cer


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
