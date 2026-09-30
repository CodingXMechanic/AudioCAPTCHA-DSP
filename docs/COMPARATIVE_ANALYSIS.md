# Comparative Analysis — Human Intelligibility vs ASR Effectiveness

> **Status:** final analysis of the headline benchmark run `results/comparative/main`
> (147 conditions × 70 utterances × 2 ASR engines = **10,290 tasks, 0 error rows**).
>
> **Provenance:** every number below comes from `results/comparative/main/{rows,summary}.csv`
> and `ranking.md`, each traceable to `dataset_manifest.json` + `run_manifest.json`.
> Regenerate with the commands in §10.
>
> **Honesty labels (see `docs/LIMITATIONS.md`):**
> - **HSR/HAG use an illustrative STOI-derived proxy — no human study was conducted.**
> - The corpus is **LibriSpeech test-clean, a labeled stand-in for the base paper's WSJ**
>   (LDC-licensed; adapter ready in `scripts/prepare_wsj.py`).
> - ASR engines are *real recognizers* but small open models (Whisper tiny 39M;
>   Vosk small-en 0.15 — Kaldi nnet3, the base-paper toolkit lineage).

---

## 1. Protocol summary

| Item | Value |
|---|---|
| Conditions | 147 (118 transform keys incl. 6 novel, 27 λ-sweep rows, control, original) |
| Utterances | 70 (subset-A protocol: 10 speakers × 7 utts, speaker list seed=42) |
| Engines | `whisper_tiny` (attention encoder–decoder), `vosk_small_en` (Kaldi nnet3) |
| Baseline WER (subset A) | whisper_tiny = **0.079**, vosk_small_en = **0.118** |
| Headline attack metric | `cross_delta_wer` = mean ΔWER over both engines vs `original` |
| Human metric | mean STOI proxy ∈ [0, 1]; HSR = illustrative thresholded proxy |
| Completion | 147/147 conditions with `n_ok=70`, **0 error rows** |

---

## 2. Headline results

### 2.1 Strongest attacks (top 10 by cross-family ΔWER)

| # | condition | fam | STOI | ΔWER whisper | ΔWER vosk | cross-ΔWER | HSR (ill.) | Pareto |
|---:|---|:-:|---:|---:|---:|---:|---:|:-:|
| 1 | `spectral.minimum_phase` | D | 0.489 | 2.084 | 0.882 | **1.483** | 0.796 | ★ |
| 2 | `novel.captcha_optimal` | G | 0.829 | 1.091 | 0.848 | **0.969** | 0.932 | ★ |
| 3 | `noise.clicks` | E | 0.535 | 1.027 | 0.880 | 0.954 | 0.814 |  |
| 4 | `spectral.phase_randomization` | D | 0.501 | 0.921 | 0.882 | 0.902 | 0.800 |  |
| 5 | `novel.multi_domain` | G | 0.877 | 0.721 | 0.586 | 0.653 | 0.951 | ★ |
| 6 | `control.no_hearing_threshold` | F | 0.675 | 0.465 | 0.530 | 0.498 | 0.870 |  |
| 7 | `novel.defense_robust` | G | 0.713 | 0.387 | 0.386 | 0.386 | 0.885 |  |
| 8 | `channel.codec_simulation` | H | 0.899 | 0.354 | 0.416 | 0.385 | 0.960 | ★ |
| 9 | `psychoacoustic.masked_noise#margin=0` | F | 0.882 | 0.324 | 0.400 | 0.362 | 0.953 |  |
| 10 | `channel.packet_jitter` | H | 0.797 | 0.245 | 0.411 | 0.328 | 0.919 |  |

- **17 conditions** exceed cross-ΔWER 0.10; **25** exceed 0.05.
- ΔWER > 1 (as for `spectral.minimum_phase` on Whisper) is possible: WER counts
  insertions, so heavily corrupted transcripts can exceed the reference length.

### 2.2 Human–ASR Gap view (top 10 by HAG = HSR − ASR success rate)

| rank | condition | fam | HAG | HSR (ill.) | ASR-SR |
|---:|---|:-:|---:|---:|---:|
| 1 | `novel.captcha_optimal` | G | **0.924** | 0.932 | 0.007 |
| 2 | `novel.multi_domain` | G | 0.837 | 0.951 | 0.114 |
| 3 | `noise.clicks` | E | 0.814 | 0.814 | 0.000 |
| 4 | `spectral.phase_randomization` | D | 0.800 | 0.800 | 0.000 |
| 5 | `spectral.minimum_phase` | D | 0.796 | 0.796 | 0.000 |
| 6 | `control.no_hearing_threshold` | F | 0.699 | 0.870 | 0.171 |
| 7 | `psychoacoustic.masked_noise#margin=0` | F | 0.638 | 0.953 | 0.314 |
| 8 | `channel.codec_simulation` | H | 0.638 | 0.960 | 0.321 |
| 9 | `channel.packet_jitter` | H | 0.569 | 0.919 | 0.350 |
| 10 | `novel.defense_robust` | G | 0.542 | 0.885 | 0.343 |

**10 conditions** reach HAG ≥ 0.5 and **13** reach HAG ≥ 0.3. The designed
objective works as intended: `novel.captcha_optimal` combines near-top attack
strength (0.969) with high human preservation (HSR 0.932) and takes **rank 1
on the actual CAPTCHA objective (HAG)** — not merely on raw WER.

### 2.3 Statistical significance (paired Wilcoxon + BH + bootstrap CI)

Per-utterance pooled (both engines) WER of each condition vs `original`,
n = 70 paired utterances; percentile bootstrap CI (B = 2000, seed 42);
Benjamini–Hochberg across the 9 headline tests (α = 0.05). Functions used:
`audiocaptcha_dsp.evaluation.stats.{wilcoxon_signed_rank, bootstrap_ci,
benjamini_hochberg_correction}`.

| condition | mean ΔWER | 95% CI | W | raw p | p (BH) | significant |
|---|---:|---:|---:|---:|---:|:-:|
| `spectral.minimum_phase` | +1.483 | [+1.036, +2.218] | 0.0 | 3.5e-13 | 4.0e-13 | yes |
| `novel.captcha_optimal` | +0.969 | [+0.886, +1.073] | 0.0 | 3.5e-13 | 4.0e-13 | yes |
| `noise.clicks` | +0.954 | [+0.882, +1.071] | 0.0 | 3.4e-13 | 4.0e-13 | yes |
| `spectral.phase_randomization` | +0.902 | [+0.877, +0.925] | 0.0 | 3.4e-13 | 4.0e-13 | yes |
| `novel.multi_domain` | +0.653 | [+0.539, +0.801] | 0.0 | 3.6e-13 | 4.0e-13 | yes |
| `control.no_hearing_threshold` | +0.498 | [+0.448, +0.554] | 0.0 | 3.6e-13 | 4.0e-13 | yes |
| `channel.codec_simulation` | +0.385 | [+0.332, +0.443] | 0.0 | 3.6e-13 | 4.0e-13 | yes |
| `psychoacoustic.masked_noise#margin=0` | +0.362 | [+0.322, +0.402] | 0.0 | 3.6e-13 | 4.0e-13 | yes |
| `temporal.rhythm` | +0.202 | [+0.147, +0.291] | 0.0 | 1.6e-11 | 1.6e-11 | yes |

All headline conditions survive multiplicity correction with W = 0 (every
paired difference positive). Full-condition inference is reproduced in
`fig14` (effect sizes ± 95% CI) and `fig15` (bootstrap rank stability,
300 resamples of utterances).

---

## 3. Family-level analysis (families A–H)

| family | n | mean STOI | mean cross-ΔWER | best attack condition |
|:-:|---:|---:|---:|:--|
| A baseline/level | 10 | 0.927 | 0.003 | `baseline.bit_depth_converter` |
| B temporal | 16 | 0.729 | 0.050 | `temporal.rhythm` |
| C multirate | 11 | 0.854 | 0.030 | `multirate.aliasing` |
| D spectral | 24 | 0.940 | 0.116 | `spectral.minimum_phase` |
| E additive/interference | 20 | 0.926 | 0.077 | `noise.clicks` |
| F psychoacoustic | 37 | 0.983 | 0.036 | `control.no_hearing_threshold` |
| G adversarial/optimized | 10 | 0.921 | **0.217** | `novel.captcha_optimal` |
| H channel/codec | 18 | 0.952 | 0.049 | `channel.codec_simulation` |

Reading (`fig4`, `fig5`):

- **G is the strongest family** (mean cross-ΔWER 0.217) while retaining
  mean STOI 0.921 — optimization finds human-sparing attacks, consistent
  with the HAG table in §2.2 (G takes ranks 1, 2, 10).
- **F (psychoacoustic) is the stealthiest effective family**: highest mean
  STOI (0.983) with non-trivial attack power at λ = 0 (0.362); its mean is
  diluted by many high-λ rows that are intentionally near-inert.
- **A is the sanity check**: level/normalization transforms average 0.003 —
  as expected, they must not attack anything (a negative control that works).
- **B hurts humans most** (mean STOI 0.729, the lowest): temporal
  manipulation degrades intelligibility more than it defeats ASR.

---

## 4. λ-sweep mirror of the base paper

The base paper (Schönherr et al. 2018) sweeps the hearing-threshold margin λ;
our sweep conditions reproduce the grid λ ∈ {0…50} dB with a power-matched
unconstrained control (`None`).

`psychoacoustic.masked_noise` (representative; full grid in `ranking.md` §λ):

| λ (dB) | STOI | cross-ΔWER | SNR dB |
|---:|---:|---:|---:|
| 0 | 0.882 | 0.362 | 0.3 |
| 5 | 0.951 | 0.105 | 5.3 |
| 10 | 0.982 | 0.045 | 10.3 |
| 30 | 1.000 | 0.008 | 30.3 |
| 50 | 1.000 | 0.002 | 50.3 |
| None (control) | 0.675 | 0.498 | 0.3 |

Findings (see `fig2`, `fig6`, `tables/lambda_sweep.csv`):

1. **Monotonic trade-off**: a tighter auditory masking constraint monotonically
   lowers attack strength and raises human intelligibility — the same
   qualitative trend the base paper reports. **λ conventions are mirrored,
   however:** our λ is a margin *below* the threshold (larger λ = tighter,
   noise further below signal: SNR ≈ λ + 0.3 dB), while their λ is an allowed
   *excess above* the threshold (larger λ = looser: their WER-to-target drops
   138 % → 6.96 % from λ=0 to λ=50, Table I). Their WER is measured against
   the **target** text (targeted insertion), ours against the **reference**
   (untargeted degradation) — so trends are compared in constraint-tightness
   terms; numeric λ values must never be compared across the two papers.
2. **λ = 0 is the operating point**: unconstrained-in-mask cross-ΔWER 0.362
   at STOI 0.882; the *unconstrained* control (0.498 at STOI 0.675) confirms
   that the masking constraint — not power — is what buys human preservation.
3. **Claim scope** (gating rule, `PAPER_RESULTS_GUIDE` §4): we claim
   *qualitative trend reproduction*, not numeric equality — different corpus,
   engines, and reference WER.

---

## 5. Cross-family transfer

`fig3` plots per-condition ΔWER(Whisper) vs ΔWER(Vosk) over all 147
conditions. Reproduce the correlation in one line:

```python
import pandas as pd; from scipy.stats import spearmanr
s = pd.read_csv("results/comparative/main/summary.csv")
spearmanr(s.wer_whisper_tiny_delta, s.wer_vosk_small_en_delta)
# SignificanceResult(statistic=0.782, pvalue=1.38e-31)  # n = 147
```

- **Spearman ρ = 0.782 (p = 1.4e-31)** — strong agreement across *independent* ASR
  families (attention encoder–decoder vs Kaldi nnet3).
- Conditions hurting both engines (ΔWER > 0.005 on each): **72**;
  Whisper-only: **25**; Vosk-only: **6**; neither: **44**.
- Interpretation: most effective attacks **transfer** rather than overfit
  one recognizer; the 25 Whisper-only conditions are the transferability
  failure cases worth flagging (they would not constitute a robust CAPTCHA
  defense against an attacker who switches engines).

---

### 5.1 Four-engine transfer validation (top-K subset)

The eight strongest conditions by cross-ΔWER, the matched-power control,
a benign condition and `original` (11 × 70 = 770 rows, 0 errors) were
re-evaluated with two extra engines — Whisper small (capacity) and
wav2vec2-base (self-supervised). Per-engine ΔWER
(`results/comparative/engine_validation/summary.csv`; CSV/LaTeX in
`results/tables/validation/`):

| condition | tiny | small | Vosk | wav2vec2 | cross (4-eng) | STOI |
|---|---|---|---|---|---|---|
| spectral.minimum_phase | 2.084 | 0.966 | 0.881 | 0.950 | **1.119** | 0.489 |
| noise.clicks | 1.027 | 1.026 | 0.879 | 0.954 | 0.953 | 0.535 |
| spectral.phase_randomization | 0.921 | 0.958 | 0.881 | 0.954 | 0.925 | 0.501 |
| novel.captcha_optimal | 1.091 | 0.812 | 0.844 | 0.941 | 0.912 | 0.829 |
| novel.multi_domain | 0.721 | 0.202 | 0.589 | 0.894 | 0.648 | 0.877 |
| control.no_hearing_threshold | 0.465 | 0.161 | 0.527 | 0.856 | 0.565 | 0.675 |
| novel.defense_robust | 0.387 | 0.099 | 0.384 | 0.640 | 0.422 | 0.713 |
| channel.codec_simulation | 0.354 | 0.101 | 0.397 | 0.169 | 0.264 | 0.899 |
| channel.packet_jitter | 0.245 | 0.070 | 0.411 | 0.106 | 0.225 | 0.797 |
| original | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| baseline.loudness_normalize | 0.002 | −0.001 | −0.003 | 0.000 | −0.001 | 1.000 |

Readings:

- The attack ordering is stable across engine sets: Spearman ρ = 0.964
  (2-engine vs 4-engine ranking, 10 conditions, p = 7.3e-6); the #1
  condition is unchanged.
- Scale point: Whisper small is harder to move than tiny (8 of 9 attack
  conditions), but no engine escapes the top attacks — every engine
  loses ≥ 0.81 absolute WER on each of the top four conditions.
- SSL profile: wav2vec2 degrades like the others on spectral/noise
  attacks but is markedly steadier under channel distortions
  (codec 0.169, jitter 0.106).
- Integrity vs the headline: Whisper tiny is bit-exact (0/770) and the
  audio is bit-identical (STOI/SNR to 1e-16); Vosk differs on 106/770
  hypotheses across run configurations (≤ 1.8 pp; 4/11 conditions
  bit-identical; same-config A/B 0/15), which changes no reading above.

## 6. Designing CAPTCHAs from the frontier

`fig1` (Pareto) shows **13 frontier conditions**:

| regime | representative | STOI | cross-ΔWER | note |
|---|---|---:|---:|---|
| maximum strength | `spectral.minimum_phase` | 0.489 | 1.483 | but human proxy < 0.5 — unusable CAPTCHA |
| **balanced (best HAG)** | `novel.captcha_optimal` | 0.829 | 0.969 | HAG 0.924 — front of both axes |
| stealthy-moderate | `channel.codec_simulation` | 0.899 | 0.385 | survives-as-audio, strong vs ASR |
| near-transparent | `noise.echo` (0.137), `spectral.hole` (0.078) | ≥0.98 | ≤0.14 | gentle layer for usability-first designs |
| psychoacoustic frontier | `bark_perturbation#0/5`, `erb_perturbation`, `signal_threshold*` | ≈1.00 | 0.01–0.02 | valid only where ASR baseline is already fragile |

Practical reading: CAPTCHA designers should pick **on the frontier**, not at
the top of the attack axis — conditions above HSR ≈ 0.8 with cross-ΔWER ≥ 0.3
(`novel.captcha_optimal`, `channel.codec_simulation`,
`psychoacoustic.masked_noise#0`, `channel.packet_jitter`) are the defensible
set under the illustrative human proxy.

---

## 7. Variability and stability

- **Speakers** (`fig9`): per-speaker ΔWER spreads within families; conclusions
  are drawn on 10-speaker means, not single speakers.
- **Bootstrap rank stability** (`fig15`, 300 paired resamples of utterances):
  the top of the attack ranking is stable — `spectral.minimum_phase`,
  `novel.captcha_optimal`, `noise.clicks` keep ranks 1–3 in nearly all
  resamples; mid-table ranks interleave (expected at n = 70).
- **Metric correlation** (`fig13`): STOI, HSR, ASR-SR, HAG and ΔWER behave as
  designed (ΔWER ↑ ⇒ ASR-SR ↓; STOI ≈ HSR by construction — flagged as a
  proxy redundancy, not an independent confirmation).

---

## 8. Novelty N11 — shelf-life forecasting

The capacity-ladder experiment (Whisper tiny → base → small; attacked-WER
scaling fit a·C^(−b); break-capacity and months-to-break forecasts) ran in
full: **1,260/1,260 tasks, 0 errors** → `results/shelf_life/shelf_life.{json,md}`,
visualized in `fig16`; method in `docs/STATISTICAL_ANALYSIS.md` §9.

| policy | b [95% CI] | R² | C\* (M) [95% CI] | gap decay |
|---|---|---:|---|---|
| control | 0.53 [0.43, 0.63] | 0.987 | 112 [81–159] | C^(−0.53) |
| masked_noise λ=0 | 0.64 [0.50, 0.83] | 0.999 | 60 [45–81] | C^(−0.64) |
| masked_noise λ=10 | 0.55 [0.38, 0.75] | 0.965 | 8 [2–17] | C^(−0.55) |
| masked_noise λ=20 | 0.50 [0.32, 0.70] | 0.996 | 4 [1–11] | C^(−0.50) |
| masked_noise λ=40 | 0.41 [0.22, 0.61] | 1.000 | 2 [0–8] | C^(−0.41) |

Ladder means over the five policies (B = 1000 paired bootstrap for C\*):

- gap = HSR − ASR-SR falls **0.260 → 0.157 → 0.034** from 39M → 74M → 244M;
- all C\* ≤ 112M < 244M ⇒ months-to-break = 0 under every doubling
  scenario — under the illustrative W_BREAK = 0.3 threshold the evaluated
  policies' shelf-life has **already expired at current open-model sizes**;
- stronger policies retain ~25–50× more break capacity than gentle ones
  (monotone in λ/attack strength — an internal-consistency check).

---

## 9. What this analysis does *not* claim

1. **No human subjects**: HSR/HAG inherit STOI's assumptions; they are an
   *illustrative proxy* wherever they appear (figures, tables, ranking).
2. **No WSJ equality**: LibriSpeech stand-in; WSJ adapter documented but
   blocked by the LDC license (`docs/DATASETS.md`).
3. **No CER / phoneme-level / defense-recovery results** — documented gaps
   in `PAPER_RESULTS_GUIDE.md` (coverage matrix) and `docs/LIMITATIONS.md`.
4. **Small models**: conclusions are about *these* engines; `fig8` shows the
   transform × model interaction is real (same condition ≠ same ΔWER per
   engine), so engine-specific attacks exist (§5, 25 Whisper-only cases).
5. **No security proof**: attack metrics ≠ CAPTCHA answer-extraction failure
   ≠ deployment security — see `THREAT_MODEL.md` §8 (non-claims).

---

## 10. Artifact index & regeneration

| Artifact | Contents |
|---|---|
| `results/comparative/main/rows.csv` | 10,290 per-task rows (21 fields, resumable) |
| `results/comparative/main/summary.csv` | 147 conditions × 29 aggregated fields |
| `results/comparative/main/ranking.md` | full ranking + HAG view + λ grids + families |
| `results/comparative/main/{dataset,run}_manifest.json` | provenance (corpus, engines, seed, args) |
| `results/tables/*.csv,*.tex` | ranking_full, top10_attack, top10_human, lambda_sweep, family_summary (LaTeX) |
| `results/figures/fig{1..16}_*.{png,svg,pdf}` | 16 figures with metadata footers |
| `results/shelf_life/` | N11 ladder run + forecasts |

```powershell
# full benchmark (resumable — same command continues an interrupted run)
python scripts/run_comparative_benchmark.py --transforms all --workers 4 --out results/comparative/main
# N11 shelf-life
python scripts/run_shelf_life.py --out results/shelf_life --workers 2
# artifacts
python scripts/make_figures.py --run results/comparative/main --out results/figures --shelf-run results/shelf_life
python scripts/make_tables.py  --run results/comparative/main --out results/tables
```
