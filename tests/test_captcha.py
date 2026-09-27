from __future__ import annotations

import time
import pytest

from audiocaptcha_dsp.captcha.generator import (
    CAPTCHAChallenge,
    CAPTCHAGenerator,
    HumanStudyProtocol,
)


class TestCAPTCHALayer:
    def test_challenge_creation_and_verification(self) -> None:
        challenge = CAPTCHAChallenge(
            challenge_id="ch_001",
            prompt="Listen and enter the NATO phonetic codeword heard:",
            correct_answer="bravo",
            answer_choices=["alpha", "bravo", "charlie", "delta"],
            transform_name="novel.captcha_optimal",
            transform_params={"margin_db": 6.0},
            created_at=time.time(),
            expires_at=time.time() + 300.0,
            replay_limit=3,
        )
        assert challenge.is_expired() is False
        assert challenge.verify_answer("bravo") is True
        assert challenge.verify_answer("  BRAVO  ") is True
        assert challenge.verify_answer("charlie") is False

        # Serialization
        d = challenge.to_dict()
        assert "correct_answer" not in d  # secure export
        rd = challenge.to_research_dict()
        assert rd["correct_answer"] == "bravo"

    def test_generator_batch(self) -> None:
        gen = CAPTCHAGenerator(seed=42, expiry_seconds=120.0, n_distractors=3)
        challenges = gen.generate_batch(5)
        assert len(challenges) == 5
        for ch in challenges:
            assert len(ch.answer_choices) == 4
            assert ch.correct_answer in ch.answer_choices

    def test_human_study_protocol(self) -> None:
        protocol = HumanStudyProtocol(
            study_id="pilot_01",
            n_practice_trials=2,
            n_main_trials=10,
            n_attention_checks=2,
            seed=42,
        )
        conditions = ["baseline.identity", "psychoacoustic.masked_noise", "novel.captcha_optimal"]
        order = protocol.generate_trial_order(conditions, n_participants=4)
        assert len(order) == 4
        # Each participant gets n_main_trials trials
        assert len(order[0]) == protocol.n_main_trials

        # Attention check
        good_responses = [{"is_attention_check": True, "correct": True}]
        bad_responses = [{"is_attention_check": True, "correct": False}]
        assert protocol.check_attention(good_responses) is True
        assert protocol.check_attention(bad_responses) is False


class TestPhase10DataModel:
    def test_log_response_anonymized_fields(self) -> None:
        """§10 data model: pseudonymous ID, confidence, ratings, device."""
        gen = CAPTCHAGenerator(seed=42)
        ch = gen.generate_batch(1)[0]
        rec = gen.log_response(
            ch, ch.correct_answer, completion_time_s=3.2, replay_count=1,
            participant_id="P-007", confidence=0.9, difficulty=2.0,
            naturalness=4.0, device="lab_headphones",
        )
        assert rec["participant_id"] == "P-007"
        assert rec["confidence"] == 0.9
        assert rec["difficulty"] == 2.0
        assert rec["naturalness"] == 4.0
        assert rec["device"] == "lab_headphones"
        assert rec["is_correct"] is True
        # Backward-compatible defaults (no participant data required)
        rec2 = gen.log_response(ch, "wrong", completion_time_s=1.0)
        assert rec2["participant_id"] is None
        assert rec2["is_correct"] is False

    def test_export_responses_jsonl(self, tmp_path) -> None:
        records = [{"a": 1}, {"b": "x"}]
        out = HumanStudyProtocol.export_responses(records, tmp_path / "resp.jsonl")
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert '"a"' in lines[0]


class TestPhase11Hooks:
    def test_rate_limit_disabled_by_default(self) -> None:
        gen = CAPTCHAGenerator(seed=42)
        assert all(gen.check_rate_limit(f"k{i}") for i in range(50))

    def test_builtin_rate_limit_window(self) -> None:
        gen = CAPTCHAGenerator(seed=42, rate_limit_per_minute=2)
        now = 1_000_000.0
        assert gen.check_rate_limit("client", now=now) is True
        assert gen.check_rate_limit("client", now=now + 1) is True
        assert gen.check_rate_limit("client", now=now + 2) is False  # limit hit
        # other keys unaffected
        assert gen.check_rate_limit("other", now=now + 2) is True
        # window slides
        assert gen.check_rate_limit("client", now=now + 61) is True

    def test_injectable_rate_limiter_hook(self) -> None:
        calls: list[str] = []

        def deny_all(key: str) -> bool:
            calls.append(key)
            return False

        gen = CAPTCHAGenerator(seed=42, rate_limiter=deny_all)
        assert gen.check_rate_limit("x") is False
        assert calls == ["x"]

    def test_export_artifact_wav_and_sidecar(self, tmp_path) -> None:
        import numpy as np

        from audiocaptcha_dsp.core.signal import Signal

        gen = CAPTCHAGenerator(seed=42)
        ch = gen.generate("utt_1", "bravo", "psychoacoustic.masked_noise",
                          {"margin_db": 6.0})
        sr = 16000
        sig = Signal(waveform=0.1 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr),
                     sample_rate=sr)
        audio = gen.export_artifact(ch, sig, tmp_path)
        assert audio.exists() and audio.suffix == ".wav"
        sidecar = tmp_path / f"{ch.challenge_id}.json"
        assert sidecar.exists()
        # Research sidecar must allow exact replay (transform + params present)
        import json as _json
        meta = _json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta["transform_name"] == "psychoacoustic.masked_noise"
        assert meta["transform_params"] == {"margin_db": 6.0}
        assert meta["correct_answer"] == "bravo"
