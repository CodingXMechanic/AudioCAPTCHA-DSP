# Improvements Over Each Surveyed Paper (1–23)

Per-paper view: what each work contributes, what we **adopt**, and what we
**improve on or do differently** — with artifact pointers. The base paper
(Paper 11) is treated in depth in `GAP_ANALYSIS.md`; the entry here is a
summary. *(The survey in `WHAT-REMAINS.txt` skips number 14; numbering is
preserved as given.)*

---

### Section A — Audio CAPTCHA security & usability

**Paper 1 — Fanelle et al., "Blind and Human", SOUPS 2020**
- Gives: real-user CAPTCHA study (67 visually impaired subjects, 51 % vs
  50 % success), public protocol.
- We adopt: their protocol shape (balanced conditions, success-rate
  reporting) as the target structure for our future human study.
- We improve: their evaluation is usability-only; we couple human success
  with ASR failure on *the same stimuli* and rank the pair (N2, N6).
  Until participants are approved we refuse to fabricate their central
  number (HSR is labelled illustrative).

**Paper 2 — aaeCAPTCHA, 2022**
- Gives: targeted adversarial perturbations defeating ASR while keeping
  human intelligibility; Whisper-class evaluation.
- We improve: their perturbations are unbounded (no hearing-threshold
  budget) — we constrain *every* candidate with psychoacoustic budgets and
  quantify imperceptibility (STOI/MBSD/SNR) per condition; we also test
  **independent ASR families** — two in the full matrix, four engines in
  the transfer-validation run (N3) rather than one model class.

**Paper 3 — "Constructing Secure Audio CAPTCHAs…", CCS 2016**
- Gives: foundational human-vs-machine gap design principle.
- We improve: the principle is formalized into measurable metrics
  (`HAG`, `HSR`, `ASR-SR`) instead of a binary design argument (N6).

**Paper 4 — "Dazed & Confused", 2023**
- Gives: large-scale reCAPTCHA usability methodology.
- We improve: complementary — we measure the *signal-level* precondition
  (intelligibility vs. ASR error) that such field studies assume; our
  manifest/seed standard makes our numbers re-runnable (N10).

### Section B — ASR robustness & Whisper evaluation

**Paper 5 — Radford et al., Whisper, ICML 2023**
- Adopt: Whisper as an evaluation engine (tiny, deterministic, offline).
- We improve: they measure robustness across noise corpora; we measure
  *attack* effectiveness per transform with baselines, ΔWER, and transfer
  to an unrelated architecture.

**Paper 6 — Multi-objective adversarial attacks on Whisper, 2026**
- Adopt: dual-objective thinking (error + efficiency).
- We improve: they optimize against one model family with gradient access;
  we evaluate black-box transfer of *non-gradient* DSP transforms across
  families and add the human axis + matched-power control (N3, N4).

**Paper 7 — Feature-vocoder adversarial attacks, 2025**
- Adopt: the framing that representation-space attacks matter.
- We improve: our SSL-space attacks are deliberately **not adopted** in this
  environment (scope decision — original novelty N11 prioritized; honest
  future-work note) while waveform-space psychoacoustic transfers
  are quantified end-to-end; we do not claim representation-space results.

**Paper 8 — Certified robustness for ASR, 2025**
- Adopt: confidence-calibration discipline — intervals on WER-derived
  quantities (t-based 95 % CIs in family summaries; per-condition n and
  error counts published).
- We improve: certification needs white-box training access we don't have;
  we instead publish *provenance* (manifests + hashes) so external
  verification is possible.

**Paper 9 — Deep speech denoising under adversarial noise, 2025**
- Adopt: the finding that enhancement can fail under adversarial inputs →
  denoising, resampling, codec and replay are implemented as *evaluated
  defenses* (`asr/defense.py`) with defense-aware HAG fields.
- We improve: defenses are scored together with human intelligibility,
  exposing defense costs a security evaluation alone would miss.

**Paper 10 — WavAugment phoneme adversarial training, 2023**
- Adopt: phoneme-level vulnerability as the explanatory mechanism.
- We improve: we expose a *phoneme-guided attack transform* and a
  phone-rate-faithful subset constraint in the benchmark itself (N9).

### Section C — Psychoacoustic foundations

**Paper 11 — Schönherr et al. (BASE PAPER)**
- Adopt: WSJ/Kaldi basis, subset protocol (70/10, 72+70, 150+72, ≤6
  phones/s), λ ∈ {0…50} grid, "None" column concept, MUSHRA/transcription
  condition layouts.
- Improve: see `GAP_ANALYSIS.md` G1–G12 — multi-ASR cross-family transfer,
  matched-power control, 4-transform λ-sweep, 119-transform space, formal
  HAG, manifests, defense evaluation, honest HSR labelling.

**Paper 12 — Generalized auditory model, 2021**
- Adopt: critical-band / masking-threshold methodology behind our Bark/ERB
  budget models (`psychoacoustics/masking_models.py`).
- We improve: thresholds are turned into *per-bin perturbation budgets*
  with tests pinning `budget = 10^(M/10)`, then validated by implied-SNR
  tracking of λ.

**Paper 13 — Modified Bark Spectral Distortion, ~2000**
- Adopt: MBSD as a first-class perceptual metric (`mbsd` column in every
  row, acceptability threshold in `SecurityEvaluator`).
- We improve: MBSD is aggregated with STOI/SNR/WER into one per-condition
  record instead of standing alone as a quality score.

### Section D — Speech quality metrics

**Paper 15 — Objective vs subjective correlation study, 2023**
- Adopt: metric-choice discipline — STOI as the intelligibility axis, with
  confidence-interval practice on aggregates.
- We improve: we never present proxy metrics as listener results; the
  human axis carries an explicit label in every table, figure and JSON.

**Paper 16 — PESQ (ITU-T P.862)**
- Not adopted: ITU reference-code licensing.
- We improve: substituting MBSD + STOI keeps the pipeline fully open;
  the substitution is declared rather than silently swapping metrics.

**Paper 17 — MOSA-Net, 2021 / Paper 18 — SALF-MOS, 2025**
- Not adopted: pretrained weights not available offline in this
  environment.
- We improve: none claimed — listed as future integration once weights can
  be vendored; MOS would slot into the human axis beside STOI.

### Section E — SSL representations

**Paper 19 — wav2vec 2.0 / Paper 20 — HuBERT / Paper 21 — More Phonetic
than Semantic**
- Adopt: the phonetic-dominance explanation motivating our phoneme-granular
  transforms and the choice of a *Kaldi-lineage* second ASR (phonetic
  bias) to complement Whisper — plus wav2vec 2.0 itself as a third
  evaluation family in the transfer-validation run, so the transfer claim
  on the paper's top-ranked conditions spans attention, Kaldi-hybrid and
  SSL architectures.
- We improve: the SSL family is carried by a *real* engine
  (`wav2vec2-base`, greedy CTC decoding, cached weights, hard error on a
  missing dependency) rather than a half-integrated stub; the heuristic
  `IndependentASREngine` stays excluded from headline results and labelled
  as such.

### Section F — Deepfake detection

**Paper 22 — Audio Deepfake Detection survey / Paper 23 — ADD 2022**
- Adopt: awareness that transforms leave authenticity markers — our
  transforms keep full provenance (family, parameters, seed), which is the
  inverse of detection and useful for it.
- We improve: out of scope; noted as a downstream study (our condition
  metadata is exactly what a detector experiment needs as labels).

---

## Cross-cutting improvements (vs. all 23)

1. **No paper reports both axes per condition** — we do, with ranks and
   Pareto flags (N2).
2. **No paper ships reproducibility manifests** — we do (N7, N10).
3. **No paper uses a matched-power control for threshold attacks** — we do
   (N4).
4. **No paper evaluates cross-architecture transfer of imperceptible
   noise** — we do (N3).
5. **No paper labels its human-proxy honestly as proxy** — ours is labelled
   in code (`HSR_LABEL`), CSV, Markdown and JSON.
