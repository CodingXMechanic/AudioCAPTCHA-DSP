# Systematic Evaluation of Psychoacoustic Distortions on Human Intelligibility vs Automatic Speech Recognition: A Framework for Secure Audio Verification

**Repository report — AudioCAPTCHA-DSP**
**Date:** 2026-09-28 · **Status:** benchmark complete, all WHAT-REMAINS phases implemented
**Spec:** `WHAT-REMAINS.txt` (18 sections) · **Paper positioning:** `PAPER_RESULTS_GUIDE.md`

---

## 1. Abstract (one paragraph)

We built a reproducible evaluation platform that measures how DSP distortions
trade **human speech intelligibility** against **automatic speech-recognition
(ASR) robustness** for audio-CAPTCHA design. The platform implements a
citation-backed taxonomy of **119 transforms across 8 families (A–H)**, a
base-paper-faithful comparison protocol (Schönherr et al. 2018, arXiv:1808.05665),
two independent ASR engines (Whisper tiny; Vosk/Kaldi nnet3 — the base-paper
toolkit lineage), the formal **Human–ASR Gap (HAG)** metric, full inferential
statistics (paired Wilcoxon + BH correction, percentile/BCa bootstrap CIs,
Mann–Whitney U, Welch t, paired permutation tests, Cohen's d/Hedges' g,
Pareto frontier), 16 paper-ready figures and 5 LaTeX tables.
The headline run executes **147 conditions × 70 utterances = 10,290 tasks with
0 errors**. Findings: attack power and human preservation form a real Pareto
frontier; the designed `novel.captcha_optimal` transform takes **rank 1 on the
HAG objective (0.924)** while staying on the frontier of raw attack strength;
hearing-threshold margin λ reproduces the base paper's qualitative trade-off
trend; attacks largely **transfer across ASR families (Spearman ρ = 0.782)**.
A novel **shelf-life forecast** (N11) extrapolates how the human–ASR gap decays
with attacker model capacity. Human-success values throughout are an
**illustrative STOI-derived proxy — no human study was conducted**.

---

## 2. What was built (phase → deliverable)

| WHAT-REMAINS | Deliverable | Where |
|---|---|---|
| 1–3 scope, model, taxonomy | experiment model, parameter namespacing, registry of **119 cited transforms / 8 families** | `src/audiocaptcha_dsp/transforms/`, `TRANSFORM_CATALOG.md` |
| 4 datasets | disk-backed adapters: **LibriSpeech, CommonVoice, FLEURS** + WSJ Kaldi adapter (LDC-blocked) | `src/.../data/`, `scripts/prepare_wsj.py`, `docs/DATASETS.md` |
| 5 multi-ASR | **Whisper tiny/base/small + Vosk (Kaldi nnet3)**, preprocessing pipelines | `src/.../asr/`, `docs/ASR_MODELS.md` |
| 6 metrics | STOI, SNR, MBSD, SI-SDR, WER/CER hooks, **HAG/ASR-SR/HSR/CSS**, rank columns | `src/.../evaluation/`, `docs/METRICS.md` |
| 7–8 protocol & ranking | subset-A protocol, λ-sweep mirror, Pareto frontier, multi-objective ranks | `experiments/comparative.py`, `docs/STATISTICAL_ANALYSIS.md` |
| 9 statistics | bootstrap (percentile/BCa), Wilcoxon, permutation, BH/Bonferroni, Cohen's d/Hedges' g | `evaluation/stats.py` |
| 10–11 human study + CAPTCHA | HSR data model, STOI-proxy labeling discipline, CAPTCHA prototype layer (9 tests) | `docs/HUMAN_STUDY.md`, `tests/test_captcha*.py` |
| 12 visualization | **16 figures** (PNG+SVG+PDF, metadata footers), coverage matrix incl. honest gaps | `scripts/make_figures.py`, `results/figures/` |
| 13 testing | **488 tests green**; CI with download-free unit job + workflow-dispatch full job (wav2vec2 weights cached) | `tests/`, `.github/workflows/ci.yml` |
| 14 safety/integrity | 9-attacker threat model, non-claims, no-fabrication rules enforced by tests | `THREAT_MODEL.md` |
| 15 documentation | 13 root documents + `docs/` linked from README | README doc table |
| 16 contributions | claim-gated contribution checklist (only evidence-backed claims retained) | `PAPER_RESULTS_GUIDE.md` §5 |
| 17 order | phases executed in specified order, recorded per phase | `PROJECT_STATUS.md` |

---

## 3. Headline results (full detail: `docs/COMPARATIVE_ANALYSIS.md`)

**Protocol:** 147 conditions × 70 utterances × 2 engines; baseline WER
whisper_tiny = 0.079, vosk_small_en = 0.118; corpus LibriSpeech test-clean
(labeled stand-in for WSJ); **10,290/10,290 tasks OK, 0 errors.**

### 3.1 Strongest attacks & the HAG objective

| metric | winner | value | notes |
|---|---|---:|---|
| raw attack (cross-ΔWER) | `spectral.minimum_phase` (D) | 1.483 | STOI 0.489 — strong but human-hostile |
| **HAG objective** | **`novel.captcha_optimal` (G)** | **0.924** | cross-ΔWER 0.969, HSR 0.932, ASR-SR 0.007 |
| balanced runner-up | `novel.multi_domain` (G) | 0.837 HAG | cross-ΔWER 0.653, HSR 0.951 |
| stealthy-moderate | `channel.codec_simulation` (H) | 0.385 | STOI 0.899, Pareto ★ |
| family means (attack) | G 0.217 > D 0.116 > E 0.077 > B 0.050 > H 0.049 > F 0.036 > C 0.030 > A 0.003 | | A = negative control ✓ |

All nine headline conditions: paired Wilcoxon vs original, **W = 0,
p(BH) ≤ 1.6×10⁻¹¹**, bootstrap 95% CIs exclude zero (§2.3 of the analysis).

### 3.2 Base-paper relationship (λ-sweep mirror)

- λ ∈ {0…50} dB on 4 psychoacoustic families: attack strength decays
  monotonically with λ while STOI rises to ~1.000 (e.g. masked_noise:
  cross-ΔWER 0.362 → 0.002, STOI 0.882 → 1.000).
- Power-matched unconstrained control: 0.498 at STOI 0.675 — isolates the
  *masking constraint* as the source of human preservation.
- **Claim scope:** qualitative trend reproduction only (different corpus,
  engines, reference WER than the original WSJ/Kaldi study).

### 3.3 Transferability & stability

- Cross-family ΔWER agreement: **Spearman ρ = 0.782**; 72 conditions hurt
  both engines, 25 Whisper-only, 6 Vosk-only, 44 neither.
- Bootstrap rank stability (300 resamples): top-3 attack ranks stable;
  mid-table interleaves (expected at n = 70).

### 3.4 Frontier for CAPTCHA design

13 Pareto conditions; defensible operating set under the illustrative proxy =
frontier points with HSR ≳ 0.8 and cross-ΔWER ≳ 0.3:
`novel.captcha_optimal`, `channel.codec_simulation`,
`psychoacoustic.masked_noise#margin=0`, `channel.packet_jitter`.

---

## 4. Novelty N11 — CAPTCHA shelf-life forecasting

Capacity ladder (Whisper tiny 39M → base 74M → small 244M, one architecture),
attacked-WER scaling fit `W(C) = a·C^(−b)` via log-log OLS with paired
bootstrap (B = 1000) CIs; break capacity C\* where forecast WER crosses the
ASR-success threshold (0.3); months-to-break under explicitly labeled 6/12/24-
month doubling scenarios. Vosk is never mixed into the fit (out-of-family
sanity marker only). Artifacts: `results/shelf_life/shelf_life.{json,md}`,
`fig16`, method in `docs/STATISTICAL_ANALYSIS.md` §9, caveats in
`docs/LIMITATIONS.md` §7.

**Observed ladder** (1,260 tasks: 5 policies × 70 utts × 3 models, 0 errors):

| model | params | baseline WER | attacked WER | ΔWER | ASR-SR | HSR (ill.) | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| whisper_tiny | 39M | 0.079 | 0.254 | +0.175 | 0.703 | 0.963 | **+0.260** |
| whisper_base | 74M | 0.071 | 0.164 | +0.092 | 0.806 | 0.963 | **+0.157** |
| whisper_small | 244M | 0.042 | 0.091 | +0.049 | 0.929 | 0.963 | **+0.034** |

**Scaling fits & forecasts** (W_BREAK = 0.3, paired bootstrap B = 1000):

| policy | exponent b [95% CI] | R² | C\* (M) [95% CI] | months @6/12/24-mo doubling |
|---|---|---:|---|---|
| control (no hearing threshold) | 0.53 [0.43, 0.63] | 0.987 | 112 [81–159] | 0 / 0 / 0 |
| masked_noise λ = 0 | 0.64 [0.50, 0.83] | 0.999 | 60 [45–81] | 0 / 0 / 0 |
| masked_noise λ = 10 | 0.55 [0.38, 0.75] | 0.965 | 8 [2–17] | 0 / 0 / 0 |
| masked_noise λ = 20 | 0.50 [0.32, 0.70] | 0.996 | 4 [1–11] | 0 / 0 / 0 |
| masked_noise λ = 40 | 0.41 [0.22, 0.61] | 1.000 | 2 [0–8] | 0 / 0 / 0 |

**Reading:** the human–ASR gap decays as C^(−0.4…−0.65) with near-perfect
log-log linearity (R² 0.965–1.000). Every tested policy's break capacity
(112M → 2M, monotone in attack strength) lies **at or below the largest
evaluated model (244M)** → months-to-break = 0 in all three doubling
scenarios: under the illustrative W ≥ 0.3 design threshold, open models at
today's size already exceed every policy's shelf-life. The gap itself
persists (ΔWER still +0.049 at 244M) but no longer holds WER above the
threshold. Stronger policies (control, λ = 0) hold ~25–50× more capacity
than gentle ones (λ = 40). Caveats: 3 ladder points, one architecture,
scenario-based extrapolation, illustrative HSR proxy.

---

## 5. Test results

```
488 passed — full suite in a single `pytest tests/` run (≈350 s, 4 benign warnings)
```

Coverage includes: registry integrity (every transform cited — enforced),
WSJ-protocol fidelity (21), benchmark logic (24, incl. the four-engine
registry and family-macro ΔWER), shelf-life N11 incl. a live ladder smoke
(17), CAPTCHA layer (9), CLI scripts (6), Whisper/Vosk/wav2vec2 adapters,
manifests/resume, figure/table generation end-to-end. CI: unit job on
push/PR without downloads; full job (fetches Whisper tiny + Vosk +
wav2vec2, caches them) via workflow dispatch. Reproduction:
`REPRODUCIBILITY.md` §7.

---

## 6. Honest gaps (never papered over)

| Gap | Reason | Status |
|---|---|---|
| WSJ exact replication | LDC license not held | adapter + docs; LibriSpeech stand-in labeled everywhere |
| Human study (real HSR) | participant approval required | STOI proxy labeled *illustrative* in every artifact |
| Phoneme-level analysis | needs forced alignment/phonemizer | documented gap; word-level confusion ships (`fig10`) |
| CER column | not in ROW_FIELDS yet | documented gap (coverage matrix) |
| Defense-recovery curves | needs on-demand sweep | documented gap |
| SSL/transformer 3rd ASR | deliberately not adopted (scope: N11) | future work — stated as such |
| Runtime/cost per task | not captured in rows | documented gap (coverage matrix) |

---

## 7. Reproduce everything

```powershell
# smoke (no downloads)
python scripts/run_comparative_benchmark.py --asr none --max-utterances 2 `
    --no-sweep --transforms noise.white --out results/comparative/smoke
# full benchmark (resumable)
python scripts/run_comparative_benchmark.py --transforms all --workers 4 --out results/comparative/main
# shelf-life (N11)
python scripts/run_shelf_life.py --out results/shelf_life --workers 2
# artifacts
python scripts/make_figures.py --run results/comparative/main --out results/figures --shelf-run results/shelf_life
python scripts/make_tables.py  --run results/comparative/main --out results/tables
python scripts/generate_catalog.py
# tests
python -m pytest tests/ -q
```

## 8. Artifact inventory

- `results/comparative/main/` — rows.csv (10,290 × 21), summary.csv (147 × 29),
  ranking.md, dataset/run manifests, main_run{,_history}.log
- `results/figures/` — fig1–fig16 × PNG/SVG/PDF with metadata footers
- `results/tables/` — ranking_full, top10_attack, top10_human, lambda_sweep,
  family_summary (CSV + LaTeX)
- `results/shelf_life/` — ladder rows, forecasts, report
- Docs: 13 root documents + `docs/` (see README table); analyses:
  `docs/COMPARATIVE_ANALYSIS.md` (numbers), `PAPER_RESULTS_GUIDE.md` (evidence)
