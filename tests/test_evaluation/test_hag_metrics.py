from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.evaluation.hag_metrics import (
    compute_hsr,
    compute_asr_sr,
    compute_hag,
    compute_defense_survival_rate,
    compute_preprocessing_recovery_rate,
    compute_cross_model_failure_rate,
    compute_perceptual_acceptability_rate,
    compute_transformation_robustness_score,
    compute_captcha_security_score,
    SecurityEvaluator,
    SecurityEvaluation,
    rank_transforms_multiobjective,
)


class TestHAGMetrics:
    def test_compute_hsr(self) -> None:
        assert compute_hsr([True, True, True, False]) == 0.75
        assert compute_hsr([]) == 0.0

    def test_compute_asr_sr(self) -> None:
        # wer < 0.3 counts as success
        wers = [0.1, 0.2, 0.4, 0.8]
        assert compute_asr_sr(wers, wer_threshold=0.3) == 0.5

    def test_compute_hag(self) -> None:
        # HAG = HSR - ASR-SR
        assert compute_hag(0.9, 0.2) == pytest.approx(0.7)
        assert compute_hag(0.5, 0.5) == pytest.approx(0.0)
        assert compute_hag(0.2, 0.8) == pytest.approx(-0.6)

    def test_defense_survival_rate(self) -> None:
        # orig wer = 0.0, trans wer = 0.8, post-defense = 0.6
        dsr = compute_defense_survival_rate(0.0, 0.8, 0.6)
        assert dsr == pytest.approx(0.75)

        # Full recovery: post-defense = orig
        dsr_recovered = compute_defense_survival_rate(0.0, 0.8, 0.0)
        assert dsr_recovered == pytest.approx(0.0)

    def test_cross_model_failure_rate(self) -> None:
        per_model = {
            "m1": [0.6, 0.7, 0.2],
            "m2": [0.8, 0.4, 0.1],
        }
        # Utterance 0: m1=0.6, m2=0.8 -> both fail (> 0.5)
        # Utterance 1: m1=0.7, m2=0.4 -> only m1 fails
        # Utterance 2: m1=0.2, m2=0.1 -> neither fails
        cmfr = compute_cross_model_failure_rate(per_model, wer_threshold=0.5)
        assert cmfr == pytest.approx(1.0 / 3.0)

    def test_perceptual_acceptability_rate(self) -> None:
        stoi = [0.8, 0.6, 0.9]
        mbsd = [3.0, 6.0, 4.0]
        snr = [20.0, 10.0, 25.0]
        # sample 0: stoi 0.8>=0.7, mbsd 3<=5, snr 20>=15 -> PASS
        # sample 1: fails all
        # sample 2: passes all
        par = compute_perceptual_acceptability_rate(
            stoi, mbsd, snr, stoi_threshold=0.7, mbsd_threshold=5.0, snr_threshold=15.0
        )
        assert par == pytest.approx(2.0 / 3.0)

    def test_security_evaluator(self) -> None:
        evaluator = SecurityEvaluator()
        dsp_metrics = [
            {"stoi_proxy": 0.85, "mbsd": 3.2, "snr_db": 18.5},
            {"stoi_proxy": 0.90, "mbsd": 2.8, "snr_db": 22.0},
        ]
        asr_wers = {"whisper": [0.6, 0.8]}
        ev = evaluator.evaluate(
            transform_name="test_transform",
            condition_params={"strength": 0.5},
            dsp_metrics=dsp_metrics,
            asr_results=asr_wers,
            human_responses=[True, True],
        )
        assert ev.hsr == 1.0
        assert ev.asr_sr == 0.0  # both wers > 0.3
        assert ev.hag == 1.0
        assert ev.security_label == "EXCELLENT"

    def test_pareto_ranking(self) -> None:
        evaluator = SecurityEvaluator()
        e1 = SecurityEvaluation(transform_name="t1", hag=0.8, par=0.9, dsr=0.9, cmfr=0.8, css=0.85)
        e2 = SecurityEvaluation(transform_name="t2", hag=0.3, par=0.3, dsr=0.3, cmfr=0.3, css=0.3)
        ranked = evaluator.mark_pareto_efficient([e1, e2])
        assert ranked[0].is_pareto_efficient is True
        assert ranked[1].is_pareto_efficient is False

        pareto_ordered = rank_transforms_multiobjective(ranked, ranking_mode="pareto")
        assert pareto_ordered[0].transform_name == "t1"
