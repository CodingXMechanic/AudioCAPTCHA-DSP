# Research Questions

Primary line of inquiry for AudioCAPTCHA-DSP, motivated by the base paper
(Schönherr et al., *Adversarial Attacks Against ASR Systems via Psychoacoustic
Hiding*, arXiv:1808.05665) and the 23-paper survey in `WHAT-REMAINS.txt`.

---

## RQ1 — The central question

**Which DSP transforms maximize ASR word-error degradation while preserving
human speech intelligibility, and how do they rank against each other?**

- *Why it matters:* the base paper evaluates ~3 attack variants; the
  literature evaluates isolated transforms. Nobody compares the human axis
  and the machine axis over a broad transform space under one protocol.
- *Operationalization:* every condition in
  `scripts/run_comparative_benchmark.py` receives a `human_rank`
  (mean STOI proxy, descending), an `attack_rank` (cross-family ΔWER,
  descending) and a Pareto flag on the (STOI, cross-ΔWER) plane.
- *Status:* **Implemented** — `experiments/comparative.py`, results in
  `results/comparative/main/{summary.csv,ranking.md}`.

## RQ2 — Psychoacoustic constraint vs. unconstrained perturbation

**Does constraining perturbations to the psychoacoustic masking threshold
improve the human-vs-ASR gap compared with an unconstrained perturbation of
equal power?**

- *Why it matters:* the base paper's "None" column (no hearing threshold)
  never states the unconstrained perturbation's power, so *shaping* and
  *power* benefits are conflated.
- *Operationalization:* `apply_no_threshold_control` builds white noise
  **power-matched to the λ = 0 at-threshold injection**; the control row and
  the λ = 0 row differ only in psychoacoustic structure.
- *Status:* **Implemented** — control condition in every benchmark run;
  power matching pinned by
  `tests/test_experiments/test_comparative.py::TestNoThresholdControl`.

## RQ3 — Cross-family transfer of imperceptible attacks

**Do psychoacoustic (threshold-shaped) attacks transfer across architecturally
independent ASR families?**

- *Why it matters:* the base paper is white-box (Kaldi DNN-HMM only);
  paper 6 transfers *gradient* attacks within Whisper; neither examines
  imperceptible noise across architectures.
- *Operationalization:* every utterance is transcribed by **Whisper tiny**
  (attention encoder-decoder) and **Vosk small-en** (Kaldi nnet3 — the base
  paper's toolkit lineage); the headline attack metric is the mean ΔWER over
  both families; transfer is plotted against the y = x reference.
- *Status:* **Implemented** — `asr/whisper_adapter.py`, `asr/vosk_adapter.py`;
  `fig3_cross_family_transfer.*`.

## RQ4 — Margin sensitivity (λ-sweep)

**How does the hearing-threshold margin λ trade off human intelligibility
against ASR attack strength, and is the trade-off consistent *across*
different budget-constrained transforms?**

- *Operationalization:* λ ∈ {0, 5, 10, 20, 30, 40, 50} dB applied uniformly
  to four psychoacoustic transforms (`masked_noise`, `bark_perturbation`,
  `temporal_masking`, `signal_threshold`), plus the power-matched control,
  mirroring the base paper's grid.
- *Status:* **Implemented** — `SWEEP_TARGETS`/`SWEEP_MARGINS_DB`;
  `fig2_lambda_sweep_mirror.*`, `tables/lambda_sweep.*`.

## RQ5 — Family-level structure

**Which taxonomy families (A–H) dominate on each axis — do families have
distinct "signatures" (e.g. family F transparent-but-weak, family G
strong-but-transparent)?**

- *Operationalization:* per-family mean cross-ΔWER ± 95 % CI and mean STOI;
  family column present in every result row.
- *Status:* **Implemented** — `fig4_family_summary.*`, `family_summary.*`.

## RQ6 — Defense-awareness (secondary)

**Do common preprocessing defenses (denoising, resampling, codec,
replay) restore ASR performance against these transforms, and at what human
cost?**

- *Operationalization:* six built-in defenses (`asr/defense.py`) with
  defense-aware HAG fields `DSR`/`CMFR`.
- *Status:* **Implemented (library + metrics)**; the headline benchmark runs
  with defenses **off** for cost reasons — defense sweeps are run on demand
  (see `LIMITATIONS.md`).

---

## RQ7 — Shelf life of the gap (novelty N11)

**As attacker ASR capacity grows, how fast does the human–ASR gap decay, and
what is the predicted *shelf life* of a psychoacoustic CAPTCHA policy?**

- *Operationalization:* evaluate the same psychoacoustic policy set
  (λ ∈ {0, 10, 20, 40} dB + matched-power control) on a controlled capacity
  ladder (Whisper tiny → base → small; one architecture, three sizes), fit
  attacked-WER = a·C^(−b) with paired-bootstrap CIs, solve for the break
  capacity C\* at WER = 0.3, and translate C\* into months-to-break under
  explicit capacity-doubling scenarios.
- *Status:* **Implemented (novelty N11)** —
  `experiments/shelf_life.py`, `scripts/run_shelf_life.py`,
  `results/shelf_life/shelf_life.{json,md}`, `fig16_shelf_life_forecast`.
- *Caveats:* 3-point ladder (df = 1); doubling rates are scenarios, not
  measured trends; human axis = illustrative STOI proxy.

---

## Questions we deliberately do *not* claim to answer

| Question | Why not |
|---|---|
| True human task success (HSR) | No human participants approved; HSR reported only as an explicitly labelled **illustrative** STOI-derived proxy (`HSR_LABEL`) |
| Bit-exact base-paper reproduction | WSJ is LDC-licensed; headline runs use a labelled LibriSpeech stand-in (protocol-identical, see `docs/BASE_PAPER_DATASET.md`) |
| Representation-space (SSL) attacks | `transformers` unavailable in this environment; documented as not-supported rather than half-implemented |
| MOS-based quality claims | MOS predictors (papers 17/18) not shippable offline; PESQ licence-restricted; MBSD + STOI used instead, declared |
