# Limitations

Honest inventory of what this work does **not** establish. Every limitation
below is also reflected in the generated outputs' labels.

---

## 1. Data limitations

| Limitation | Impact | Status |
|---|---|---|
| **WSJ corpus absent** (LDC licence) | Headline runs use LibriSpeech test-clean — protocol-identical (70 utts / 10 speakers, same selection code) but acoustically *not* WSJ. Cross-corpus effects on absolute WER cannot be ruled out. | Stand-in **labelled** in all manifests/captions; exact re-run command ready (`--dataset wsj`) |
| Paper's subset identities undisclosed | "70 utterances from one WSJ test set" cannot be bit-matched; our selection is deterministic but necessarily one valid realization | Protocol parity + manifest hashing documented (`docs/BASE_PAPER_DATASET.md`) |
| Music corpus unnamed in the paper | Subsets B/C music side cannot be exact; music omitted unless `--music-dir` given | Omitted-and-**labelled** |
| Phone estimator approximate | ≤ 6 phones/s filter uses ≈0.96 phonemes/letter, not the paper's phonemizer | marked `estimated` |

## 2. Human-axis limitations

- **No human study has been conducted.** `hsr` is an STOI-derived
  **illustrative** proxy; it is stamped as such in every artifact. Real
  human task success, accessibility findings (papers 1/4) and MUSHRA scores
  remain unmeasured.
- STOI is a *proxy for intelligibility*, not for CAPTCHA usability;
  perceived effort, timing and hearing-impaired populations are out of
  proxy's reach.

## 3. ASR-axis limitations

- **Two engines, both small models** (Whisper tiny, Vosk small). Larger
  Whisper variants may be more robust; results are engine-specific. No claim
  about unseen future architectures.
- **`IndependentASREngine` is heuristic/metadata-driven** — usable as a test
  double only; excluded from every ΔWER claim (tagged `heuristic_proxy`).
- **SSL family (papers 19–21) not evaluated**: `transformers` unavailable in
  this environment → *Not supported*.
- Exact paper ASR (Kaldi WSJ DNN-HMM) requires WSJ training; Vosk is a
  lineage stand-in, not the same model.

## 4. Metric limitations

- PESQ (paper 16) excluded for licence reasons → quality axis = MBSD + STOI
  (declared substitution).
- MOS predictors (papers 17/18) not shippable offline → no MOS claims.
- MBSD/STOI thresholds in `css` are defaults from the evaluator, not
  psychophysically validated *for this stimulus set*.
- Identity rows have `snr_db = inf` (stored empty) — SNR means over mixed
  sets exclude non-finite values.

## 5. Experimental-design limitations

- **Headline run evaluates transforms without defenses** (defense sweep is
  6× ASR cost; defense *metrics* are implemented and tested).
- **No per-condition compute-cost column** (only aggregate throughput is
  logged) → the "lowest computational cost" ranking view (§8) is
  *Planned, not implemented*.
- Fixed condition ordering (registry order) rather than randomized
  interleaving; all conditions share identical utterances, so drift is
  common-mode, but ordering effects are not statistically removed.
- λ-sweep covers 4 psychoacoustic transforms (of 9 in family F) to bound
  runtime; the other 5 are evaluated at defaults only.
- Whisper+vok throughput on 4 CPUs → full runs take hours; conclusions wait
  for the run, not the smoke tests (n ≤ 2 smoke outputs are non-citable).
- Composite/chain conditions are supported by the runner but not part of the
  headline grid.

## 6. Security-claim limitations

- Security is only ever claimed **relative to the evaluated engines,
  defenses, corpus and manifests** (see `THREAT_MODEL.md` §8).
- No evaluation against human-in-the-loop solvers (CAPTCHA mode) — the
  threat model's K3 (human attackers/bots-with-humans) is untested.
- Attack transfer to physical recordings (replay) is simulated
  (`channel.*`, `ReplaySimulationDefense`), not measured in situ.

## 7. Statistical limitations

- Family-level CIs use t-based 95 % intervals over *conditions* (not over
  listeners — there are none).
- Hundreds of pairwise comparisons are possible from `rows.csv`; only
  descriptive stats ship by default. Users must apply BH/Bonferroni
  themselves (helpers provided) — raw p-values without correction must not
  be quoted.
- Illustrative HSR is never a subject of inference.
- **Shelf-life forecasts (novelty N11)** rest on a **3-point capacity ladder**
  (df = 1): bootstrap CIs quantify utterance-sampling variation, *not*
  uncertainty over the population of possible models. Capacity-doubling
  rates (6/12/24 months) are stated scenarios, not measured trends, and any
  value read beyond the largest evaluated rung (244 M) is an extrapolation
  — reported with the fit band, never as a point prediction.

## 8. Integrity rules tied to these limitations

1. No human number without the illustrative label.
2. No WSJ-exactness claim from stand-in runs.
3. No ΔWER claim from the heuristic engine.
4. No PESQ/MOS numbers.
5. Smoke runs (n ≤ 2) are pipeline checks, never conclusions.
