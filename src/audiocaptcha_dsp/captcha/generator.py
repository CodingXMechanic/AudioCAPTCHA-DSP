"""
Research-Only Audio CAPTCHA Generation Layer
=============================================
Generates audio CAPTCHA challenges for Human-vs-ASR research studies.
This is a research prototype, NOT a production security system.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from audiocaptcha_dsp.core.signal import Signal

logger = logging.getLogger(__name__)

@dataclass
class CAPTCHAChallenge:
    challenge_id: str
    prompt: str
    correct_answer: str
    answer_choices: list[str]
    transform_name: str
    transform_params: dict
    created_at: float
    expires_at: float
    replay_limit: int = 3
    utterance_id: str = ''
    condition_index: int = -1
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
        
    def verify_answer(self, response: str) -> bool:
        return response.strip().lower() == self.correct_answer.strip().lower()
        
    def to_dict(self) -> dict:
        return {
            'challenge_id': self.challenge_id,
            'prompt': self.prompt,
            'answer_choices': self.answer_choices,
            'transform_name': self.transform_name,
            'transform_params': self.transform_params,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'replay_limit': self.replay_limit,
            'utterance_id': self.utterance_id,
            'condition_index': self.condition_index
        }
        
    def to_research_dict(self) -> dict:
        d = self.to_dict()
        d['correct_answer'] = self.correct_answer
        return d

class CAPTCHAGenerator:
    WORD_LIST = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot',
                 'golf', 'hotel', 'india', 'juliet', 'kilo', 'lima',
                 'mike', 'november', 'oscar', 'papa', 'quebec', 'romeo',
                 'sierra', 'tango', 'uniform', 'victor', 'whiskey', 'xray',
                 'yankee', 'zulu']
    
    def __init__(self, seed: int = 42, expiry_seconds: float = 300.0, n_distractors: int = 3,
                 rate_limit_per_minute: int = 0,
                 rate_limiter: Callable[[str], bool] | None = None):
        """
        Parameters
        ----------
        rate_limit_per_minute :
            Built-in sliding-window limit per client key; 0 disables it.
        rate_limiter :
            Pluggable rate-limiting hook (WHAT-REMAINS §11).  Receives the
            client key and returns True when the request is allowed.  When
            provided it takes precedence over the built-in window.
        """
        self.rng = np.random.default_rng(seed)
        self.expiry_seconds = expiry_seconds
        self.n_distractors = n_distractors
        self.rate_limit_per_minute = int(rate_limit_per_minute)
        self.rate_limiter = rate_limiter
        self._rate_windows: dict[str, list[float]] = {}

    def check_rate_limit(self, client_key: str = "anonymous", now: float | None = None) -> bool:
        """Rate-limiting hook (Phase 11). Returns True when the request may proceed."""
        if self.rate_limiter is not None:
            return bool(self.rate_limiter(client_key))
        if self.rate_limit_per_minute <= 0:
            return True
        t = time.time() if now is None else now
        window = self._rate_windows.setdefault(client_key, [])
        cutoff = t - 60.0
        window[:] = [ts for ts in window if ts > cutoff]
        if len(window) >= self.rate_limit_per_minute:
            return False
        window.append(t)
        return True
    
    def generate(self, utterance_id: str, correct_word: str, transform_name: str,
                 transform_params: dict, condition_index: int = -1) -> CAPTCHAChallenge:
        """Generate a challenge for a single word CAPTCHA."""
        challenge_id = str(uuid.uuid4())
        created_at = time.time()
        expires_at = created_at + self.expiry_seconds
        
        choices = [correct_word]
        candidates = [w for w in self.WORD_LIST if w.lower() != correct_word.lower()]
        distractors = self.rng.choice(candidates, self.n_distractors, replace=False)
        choices.extend(distractors)
        self.rng.shuffle(choices)
        
        return CAPTCHAChallenge(
            challenge_id=challenge_id,
            prompt="Select the word you hear:",
            correct_answer=correct_word,
            answer_choices=list(choices),
            transform_name=transform_name,
            transform_params=transform_params,
            created_at=created_at,
            expires_at=expires_at,
            utterance_id=utterance_id,
            condition_index=condition_index
        )
    
    def generate_batch(self, n: int, words: list[str] | None = None) -> list[CAPTCHAChallenge]:
        """Generate n challenges for research batch evaluation."""
        if words is None:
            words = list(self.rng.choice(self.WORD_LIST, n))
            
        challenges = []
        for i in range(n):
            word = words[i % len(words)]
            challenges.append(
                self.generate(f"utterance_{i}", word, "none", {}, i)
            )
        return challenges
    
    def log_response(self, challenge: CAPTCHAChallenge, response: str, 
                     completion_time_s: float, replay_count: int = 0,
                     participant_id: str | None = None,
                     confidence: float | None = None,
                     difficulty: float | None = None,
                     naturalness: float | None = None,
                     device: str | None = None,
                     is_attention_check: bool = False) -> dict[str, Any]:
        """Log an anonymized research response (WHAT-REMAINS §10 data model).

        Only pseudonymous, study-necessary fields are recorded; no raw audio
        and no direct identifiers. Returns the log record.
        """
        is_correct = challenge.verify_answer(response)
        return {
            'challenge_id': challenge.challenge_id,
            'condition_index': challenge.condition_index,
            'transform_name': challenge.transform_name,
            'participant_id': participant_id,
            'is_correct': is_correct,
            'completion_time_s': completion_time_s,
            'replay_count': replay_count,
            'confidence': confidence,
            'difficulty': difficulty,
            'naturalness': naturalness,
            'device': device,
            'is_attention_check': is_attention_check,
            'timestamp': time.time(),
            'expired': challenge.is_expired()
        }
    
    def compute_batch_hsr(self, responses: list[dict[str, Any]]) -> float:
        """Compute Human Success Rate from batch of logged responses."""
        if not responses: return 0.0
        correct = sum(1 for r in responses if r.get('is_correct', False) and not r.get('expired', False))
        return float(correct) / len(responses)

    def export_artifact(self, challenge: CAPTCHAChallenge, signal: 'Signal',
                        out_dir: Path | str) -> Path:
        """Export a challenge's audio + research sidecar (WHAT-REMAINS §11).

        Writes ``<challenge_id>.wav`` and ``<challenge_id>.json`` (the
        research dict incl. transform parameters) so a challenge can be
        replayed exactly. Returns the audio path.
        """
        from audiocaptcha_dsp.io.artifacts import save_artifact, save_signal_wav

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        audio_path = save_signal_wav(signal, out / f"{challenge.challenge_id}.wav")
        save_artifact(challenge.to_research_dict(), out / f"{challenge.challenge_id}.json")
        return audio_path

@dataclass
class HumanStudyProtocol:
    """Research protocol configuration for human listening studies."""
    study_id: str
    n_practice_trials: int = 3
    n_main_trials: int = 20
    n_attention_checks: int = 2
    counterbalanced: bool = True
    seed: int = 42
    exclusion_accuracy_threshold: float = 0.5  # exclude participants below this
    
    def generate_trial_order(self, conditions: list[str], n_participants: int) -> list[list[str]]:
        """Generate counterbalanced trial order for n_participants."""
        rng = np.random.default_rng(self.seed)
        orders = []
        for _ in range(n_participants):
            order = list(conditions)
            if self.counterbalanced:
                rng.shuffle(order)
            if len(order) > self.n_main_trials:
                order = order[:self.n_main_trials]
            else:
                while len(order) < self.n_main_trials:
                    extra = list(conditions)
                    rng.shuffle(extra)
                    order.extend(extra)
                order = order[:self.n_main_trials]
            orders.append(order)
        return orders
    
    def check_attention(self, responses: list[dict]) -> bool:
        """Check if participant passed attention checks."""
        att_checks = [
            r for r in responses
            if r.get('is_attention_check', False) or r.get('condition_index', 0) < 0
        ]
        if not att_checks:
            return True
        correct = sum(
            1 for r in att_checks
            if r.get('is_correct', False) or r.get('correct', False)
        )
        return (correct / len(att_checks)) >= self.exclusion_accuracy_threshold
    
    def to_dict(self) -> dict:
        return {
            'study_id': self.study_id,
            'n_practice_trials': self.n_practice_trials,
            'n_main_trials': self.n_main_trials,
            'n_attention_checks': self.n_attention_checks,
            'counterbalanced': self.counterbalanced,
            'seed': self.seed,
            'exclusion_accuracy_threshold': self.exclusion_accuracy_threshold
        }

    @staticmethod
    def export_responses(responses: list[dict[str, Any]], path: Path | str) -> Path:
        """Export response records as JSONL (anonymized data export format).

        One JSON object per line — append-safe, stream-readable, and the
        format expected by the analysis helpers in ``evaluation.stats``.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for rec in responses:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return p
