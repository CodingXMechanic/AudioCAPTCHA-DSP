# Human Study Protocol

WHAT-REMAINS §10 — **Protocol ready; study not conducted.** No participant
data exists, and no output of this repository may present human numbers as
measured until this protocol has been executed and the `HSR_LABEL` retired
for that run.

---

## 1. Why the current HSR is a labelled proxy

`SecurityEvaluator` computes `hsr = clip(0.6 + 0.4·STOI_proxy, 0, 1)` when
`human_responses` is empty (`n_human_participants = 0`). That value exists
only to keep pipeline plumbing complete and is stamped
`illustrative (STOI-derived proxy; no human study conducted)` everywhere it
appears. **It is not evidence of human performance.**

## 2. Study design (mirrors the base paper, extended)

### Mode A — Transcription task (base-paper layout)

| Element | Value | Source |
|---|---|---|
| Conditions | subset of the ranked transforms: ~21 samples covering (i) top Pareto attacks, (ii) λ-sweep points, (iii) control, (iv) identity | base paper: 21 samples |
| Listeners | ≥ 22, screened normal hearing (or target population e.g. visually-impaired users per paper 1) | base paper: 21 × 22 |
| Task | type what you hear; free repeats allowed and logged | base paper |
| Primary measures | transcription WER (human), task success, completion time | base paper reported 12.59 % vs 12.61 % WER, t-test 1 % |
| Randomization | sample order randomized per listener; condition blinded | WHAT-REMAINS §7 |
| Statistics | paired tests (same stimuli across conditions), BH correction, effect sizes + CIs | `evaluation/stats.py` |

### Mode B — CAPTCHA task (`CAPTCHAGenerator`)

- Present challenges generated from chosen conditions; measure solve rate,
  time, and bot-solve rate under the same ASR engines.
- Supports WHAT-REMAINS §7 conditions: original, identity, degradation,
  psychoacoustic, adversarial, novel, composite, defense-post-processed.

### Mode C — MUSHRA-style intelligibility rating

- 9 samples per set (base paper), hidden reference + anchor included;
  rating scale per `HumanStudyProtocol` in `captcha/generator.py`.

## 3. Ethics & logistics — currently blocking

- Participant recruitment requires ethics/organizational approval
  (**not obtained**).
- Compensation, consent, hearing screening: defined in the protocol code
  skeleton; execution **Planned**.
- Until then: every human-facing number carries `HSR_LABEL`.

## 4. What the code already provides

```python
from audiocaptcha_dsp.captcha.generator import CAPTCHAGenerator, HumanStudyProtocol
from audiocaptcha_dsp.evaluation.hag_metrics import SecurityEvaluator

# --- data model (§10): anonymized, study-necessary fields only --------------
gen = CAPTCHAGenerator(seed=42, rate_limit_per_minute=30)   # §11 rate-limit hook
challenge = gen.generate(utt_id, "bravo", "psychoacoustic.masked_noise", {"margin_db": 6.0})
record = gen.log_response(
    challenge, response, completion_time_s=12.4, replay_count=1,
    participant_id="P-007",        # pseudonymous, no direct identifiers
    confidence=0.9, difficulty=2.0, naturalness=4.0, device="lab_headphones",
)
HumanStudyProtocol.export_responses([record], "study/responses.jsonl")   # JSONL export
gen.export_artifact(challenge, signal, "study/artifacts")                # wav + sidecar json

# --- HSR becomes measured once real responses exist -------------------------
evaluator = SecurityEvaluator()
evaluation = evaluator.evaluate(
    transform_name=..., condition_params=...,
    dsp_metrics=[...],                    # per-utterance compute_metrics dicts
    asr_results={"whisper_tiny": [...]},  # per-utterance WERs
    human_responses=[...],                # ← real listener data goes here
    original_wer=...,
)
# with human_responses populated: n_human_participants > 0,
# hsr becomes measured, label requirement lifted for that run
```

Protocol scaffolding: `HumanStudyProtocol.generate_trial_order` (randomized,
counterbalanced), `check_attention` (attention-check screening),
`exclusion_accuracy_threshold` (exclusion rule), `n_practice_trials`
(practice trials), `to_dict` (pilot-study configuration).

## 5. Exit criteria for retiring the illustrative label

1. ≥ 22 listeners per the Mode-A template (or a justified deviation).
2. Raw responses stored with the run (`human_responses` file + manifest
   entry with participant count).
3. Agreement check: STOI-proxy vs measured HSR (correlation + Bland-Altman)
   — this validation *is* a publishable result in itself (validates or
   refutes the proxy for future runs).
4. Update `HSR_LABEL` handling and re-run `make_tables`/`make_figures` so
   captions switch to measured wording.
