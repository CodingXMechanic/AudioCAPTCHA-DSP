# Novelties — What This Project Adds Beyond All 23 Surveyed Papers

Mapped to the contribution list of `WHAT-REMAINS.txt` §16. Each novelty
states *why it is not present in the surveyed literature* and *where the
artifact lives*. No claim here is made without a code or result pointer.

---

## N1. A unified benchmark over a broad DSP transform space

**Novel because:** papers 1–4 evaluate CAPTCHA designs; 5–10 evaluate ASR
robustness/attacks; 11–13 one psychoacoustic attack family; none compares
*human intelligibility* and *ASR failure* across a diverse transform
space under one protocol.

**Ours:** 119 registered transforms in 8 families (A–H), one selection
protocol, one metric stack, one ranking.
**Artifact:** `transforms/registry.py`; `results/comparative/main/summary.csv`.

## N2. Dual-axis ranking with an explicit Pareto frontier

**Novel because:** surveyed work ranks attacks by WER alone (2, 6, 7, 11)
or usability by MOS alone (1, 4, 18); the *joint* human-vs-ASR plane with
Pareto flags per condition is not provided by any of them.

**Ours:** `human_rank` (mean STOI), `attack_rank` (cross-family ΔWER),
`pareto` (nondominated on both axes) computed for every condition.
**Artifact:** `experiments/comparative.py::_is_nondominated`;
`fig1_pareto_human_vs_asr.*`, `ranking.md`.

## N3. Cross-family transferability of psychoacoustic attacks

**Novel because:** paper 6 shows transfer of *gradient* attacks within
Whisper models; paper 7 attacks SSL feature extractors; the base paper (11)
evaluates only its white-box Kaldi DNN. Whether *threshold-shaped*
(imperceptible) noise transfers across **architecturally independent**
families (attention encoder-decoder vs. Kaldi nnet3) is unexamined.

**Ours:** every condition is transcribed by Whisper tiny **and** Vosk
(Kaldi lineage); the headline attack metric is the **mean ΔWER across both
families**, and transfer scatter is a first-class figure.
**Artifact:** `fig3_cross_family_transfer.*`, `cross_delta_wer` column.

## N4. Matched-power unconstrained control ("None" done rigorously)

**Novel because:** the base paper's "None" column (attacks without hearing
thresholds) never states the unconstrained perturbation's power level, so
shaping benefits and power benefits are conflated; papers 2/6 use
unweighted noise at unrelated strengths.

**Ours:** `apply_no_threshold_control` builds white noise whose total power
*equals the λ = 0 at-threshold injection* — any difference between the
control and λ=0 rows is attributable purely to psychoacoustic structure.
**Artifact:** `experiments/comparative.py`; verified by
`tests/test_experiments/test_comparative.py::TestNoThresholdControl`.

## N5. Uniform λ-sweep across four psychoacoustic transforms

**Novel because:** the paper sweeps λ only for its own hiding method, on
one attack. Applying the identical λ grid (0…50 dB) to four different
budget-constrained transforms makes margin sensitivity *comparable across
methods*.
**Artifact:** `SWEEP_TARGETS`/`SWEEP_MARGINS_DB`; `fig2_lambda_sweep_mirror.*`,
`lambda_sweep.*` tables.

## N6. Formal Human-ASR Gap (HAG) metric stack

**Novel because:** no surveyed paper defines a per-condition gap metric;
they report WER and intelligibility in separate sections.

**Ours:** `HSR` (human success rate, proxy-labelled), `ASR-SR`, `HAG`,
`CSS`, defense-aware `DSR`/`CMFR`, computed per condition with threshold
metadata.
**Artifact:** `evaluation/hag_metrics.py` (`SecurityEvaluator`).

## N7. Dataset-fidelity + provenance standard for audio-CAPTCHA research

**Novel because:** surveyed experiments cite corpora but publish neither
selection identities nor content hashes; re-running "the same 70 WSJ
utterances" was impossible even for the base paper's own readers.

**Ours:** protocol-exact subset selection (A/B/C with paper counts,
selection mode, seed), `dataset_manifest.json` with full utterance lists
and `corpus_sha256`, run manifest with git commit + versions.
**Artifact:** `docs/BASE_PAPER_DATASET.md`, `write_dataset_manifest`,
`write_run_manifest`.

## N8. License-aware dataset layer (exact WSJ path + labelled stand-in)

**Novel because:** no surveyed artifact ships a first-class WSJ/Kaldi
adapter *with* an honest substitute policy.
**Ours:** `WSJAdapter` (Kaldi + raw LDC layouts, standard test-set
preference), `scripts/prepare_wsj.py` (`--check/--convert/--build-kaldi/
--subsets`), and headline runs on LibriSpeech labelled
`"stand-in for WSJ: LDC-licensed"` in every manifest row.
**Artifact:** `evaluation/dataset.py`, `scripts/prepare_wsj.py`.

## N9. Phoneme-granular attack + phone-rate-faithful subset protocol

**Novel because:** papers 10/21 argue phonetic granularity matters; nobody
in the survey couples it with the CAPTCHA subset protocol (the base paper
filters ≤6 phones/s but doesn't expose a phoneme-guided attack or phone
estimator publicly).

**Ours:** `adversarial.phoneme_guided` transform + `estimate_phoneme_count`
/ `phone_rate` helpers wired into subset selection.
**Artifact:** `evaluation/dataset.py`, `transforms/adversarial/search.py`.

## N10. Reproducible, resumable artifact standard

**Novel because:** none of the 23 papers ships a benchmark that survives
interruption and proves what data produced each row.

**Ours:** append-only `rows.csv` with (condition, utterance) keys,
automatic resume, per-run manifests, deterministic seeds propagated into
transforms that accept them.
**Artifact:** `run_benchmark` resume path;
`tests/test_experiments/test_comparative.py`.

---

## N11. CAPTCHA shelf-life forecasting from gap-vs-capacity scaling

**Novel because:** no surveyed paper — and no audio-CAPTCHA work we know of —
treats the human–ASR gap as a *function of attacker model capacity* and
forecasts when a policy stops working. Attacks and defenses are evaluated at
one capacity; security is reported as a snapshot, never as a lifetime.

**Ours:** a controlled **capacity ladder** (Whisper tiny 39M → base 74M →
small 244M; one architecture, three sizes — architecture held fixed so the
scaling axis is not confounded) measures the attacked-ASR WER and the
human–ASR gap at each rung, fits a neural-scaling-style law

    WER(C) = a · C^(−b)      (log-log OLS; paired bootstrap over evaluation
                              pairs for 95% CIs on a, b, and the break point),

and solves it for the **break capacity** C\* — the model size at which
expected WER on attacked audio falls to the security threshold
(W_break = 0.3, the same threshold as `SecurityEvaluator.asr_sr`). C\* is
translated into **months-to-break under explicit capacity-doubling
scenarios** (6/12/24 months per doubling), giving each psychoacoustic policy
a *shelf life* instead of a boolean.

**Honesty contract:** fits use only 3 ladder points (df = 1); doubling rates
are stated scenarios, not measured trends; the human axis is the
STOI-derived illustrative proxy (no human study); other ASR families
(Vosk) are reported only as out-of-family markers, never mixed into the
scaling axis.

**Artifact:** `experiments/shelf_life.py`, `scripts/run_shelf_life.py`,
`results/shelf_life/` (`rows.csv`, `shelf_life.{json,md}`, manifests),
`fig16_shelf_life_forecast`,
`tests/test_experiments/test_shelf_life.py` (17 tests).

---

### Relationship to WHAT-REMAINS §16

| §16 contribution | Covered by |
|---|---|
| 1. Unified benchmark | N1 |
| 2. Transform taxonomy A–H | `transforms/registry.py` (families in every result row) |
| 3. Formal HAG metric + protocol | N6 |
| 4. Cross-model & defense-aware analysis | N3, defenses in `asr/defense.py` |
| 5. Multi-objective ranking / Pareto method | N2 |
| 6. Psychoacoustically constrained family showing separation | N4, N5 |
| 7. Phoneme-/representation-level explanation of ASR degradation | N9 |
| 8. Reproducible artifact & manifest standard | N7, N8, N10 |
