# Study on the Correlation between Objective Evaluations and Subjective Speech Quality and Intelligibility

## Full reference
- **Title:** Study on the Correlation between Objective Evaluations and Subjective Speech Quality and Intelligibility
- **Authors (as printed):** Hsin-Tien Chiang¹, Kuo-Hsuan Hung², Szu-Wei Fu³, Heng-Cheng Kuo¹,
  Ming-Hsueh Tsai⁴, Yu Tsao¹
- **Affiliations:** ¹Academia Sinica, ²National Taiwan University, ³NVIDIA, ⁴National Academy for Educational Research
- **Printed identifiers:** copyright line **"979-8-3503-0689-7/23/$31.00 ©2023 IEEE"**;
  **arXiv:2307.04517v2 [eess.AS] 10 Oct 2023**
- **Venue/year:** IEEE-sponsored conference paper, **2023** — the conference name itself is **not printed** in the
  extracted text (only the IEEE copyright/ISBN line and the arXiv ID).
- **Dataset links printed:** TMHINT-QI http://gofile.me/6PGhz/4U6GWaOtY ; description
  https://github.com/yuwchen/InQSS

## One-line contribution
Measures how well twelve off-the-shelf objective metrics (PESQ, P.835, DNSMOS, MOSA-Net, NCM, STOI, ESTOI, WER…)
match **human** quality/intelligibility ratings on the Mandarin **TMHINT-QI** corpus (none correlates above 0.8),
then trains a small **DL combiner** of those metrics that predicts both subjective scores — even with very little
training data — and shows subjective *quality* ratings can substantially improve subjective *intelligibility*
prediction.

## Problem & motivation
- Subjective listening tests are the gold standard but are slow and expensive; objective surrogates must be shown to
  actually track human perception.
- **PESQ and POLQA correlate suboptimally with subjective tests**; **STOI** is reported to be suboptimal for
  Wiener-filtering and deep-learning-based enhancement; intrusive metrics also need a clean reference.
- Non-intrusive options (ITU-T P.563, ANIQUE+, SRMR) and DL predictors (DNSMOS, NISQA) exist, but DL models trained
  on *objective* labels inherit those labels' misalignment with humans.
- Prior work combined quality metrics (composite measures) or used ASR + quality metrics for intelligibility, but
  **few studies examine quality *and* intelligibility together and interpret how each objective measure reflects
  human perception**.

## Method (precise but simple)
1. Compute **eight named objective measures** on TMHINT-QI:
   *quality:* **PESQ** (−0.5…4.5), **ITU-T P.835** (SIG/BAK/OVRL), **DNSMOS P.835** (SIG/BAK/OVRL), **MOSA-Net**;
   *intelligibility:* **NCM** (0…1), **STOI** (0…1), **ESTOI** (0…1), **WER** (Google ASR).
   The DL input vector is stated as **twelve objective measures** (P.835/DNSMOS sub-scores counted separately);
   third-party pre-trained APIs were used for WER, DNSMOS and MOSA-Net.
2. **Correlation analysis:** Pearson correlation (PCC) of each objective measure against averaged subjective
   quality and intelligibility (Fig. 1), plus PESQ↔STOI↔WER scatter plots (Fig. 2).
3. **Proposed DL model (Fig. 3):** all measures min–max normalized to [0,1] → **six dense layers**, GELU after each
   except the last, **sigmoid** output → split into two heads (**quality** and **intelligibility**) → de-normalized.
   Loss/training details beyond the architecture are not printed.
4. Baselines: **linear regression (LR)** predicting each target separately; **InQSS** (SSL + scattering transform,
   Q+I multi-task); **MOS-SSL** (fine-tuned wav2vec2.0, single-task, retrained here on TMHINT-QI).
5. Evaluation: **MSE, PCC, SRCC**; two split protocols — Table 1 follows InQSS (random 90/10 train/val split),
   Table 2 enforces **no speaker overlap** (generalization).
6. **Interpretation probe (Fig. 5):** feed samples from a multivariate normal distribution through the DL model,
   bin each objective measure into **200 equal parts**, average, repeat **1,000 times** (mean ± SD curves).
7. **Ablation (Table 4):** add **subjective quality** as an extra input when predicting subjective intelligibility.

## Key quantitative results
**Correlation findings (Section 3.2, Fig. 1/2):**
- Subjective quality ↔ subjective intelligibility: **≈ 0.68** ("moderately correlated").
- **No objective measure correlates above 0.8** with either subjective score.
- All objective measures **except WER** correlate more with subjective **quality** than with subjective
  intelligibility; objective *quality* measures vs subjective *intelligibility* are **below 0.24**.
- Strongest absolute correlate of subjective **quality** = **NCM**, then **ESTOI**, then **STOI**.
- Highest absolute correlate of subjective **intelligibility** = **WER**, then subjective quality, then NCM.
- STOI↔WER correlation is higher than PESQ↔WER (consistent with prior work; supports optimizing enhancement on STOI
  to improve WER).

**Table 1 — random split (InQSS configuration), PCC / SRCC:**

| System | Quality PCC/SRCC | Intelligibility PCC/SRCC |
|---|---|---|
| InQSS (Q+I) | 0.804 / 0.759 | 0.791 / 0.730 |
| MOS-SSL (Q) | 0.805 / 0.761 | – |
| MOS-SSL (I) | – | 0.774 / 0.67 |
| LR (Q) | 0.797 / 0.751 | – |
| LR (I) | – | 0.739 / 0.676 |
| **DL (Q+I)** | **0.806 / 0.763** | **0.797 / 0.730** |

**Table 2 — no speaker overlap (generalization), PCC / SRCC:**
LR (Q) 0.799 / 0.733 · LR (I) 0.733 / 0.728 · **DL (Q+I) 0.794 / 0.741 (quality), 0.766 / 0.733 (intelligibility)**.

**Table 3 — PCC at reduced training sizes (PC = % decrease vs 12,000 utterances):**

| Model | Data% | Quality PCC / PC | Intell. PCC / PC |
|---|---|---|---|
| InQSS | 1.66% | 0.236 / 70.35 | 0.262 / 66.54 |
| InQSS | 5% | 0.501 / 37.06 | 0.521 / 33.46 |
| InQSS | 25% | 0.771 / 3.14 | 0.723 / 7.66 |
| InQSS | 100% | 0.796 / – | 0.783 / – |
| MOS-SSL | 1.66% | 0.578 / 27.57 | 0.080 / 89.58 |
| MOS-SSL | 5% | 0.675 / 15.41 | 0.407 / 47.01 |
| MOS-SSL | 25% | 0.767 / 3.88 | 0.714 / 7.03 |
| MOS-SSL | 100% | 0.798 / – | 0.768 / – |
| **DL** | 1.66% | 0.752 / 6.47 | 0.688 / 12.91 |
| **DL** | 5% | 0.777 / 3.36 | 0.754 / 4.56 |
| **DL** | 25% | 0.796 / 1.00 | 0.786 / 0.51 |
| **DL** | 100% | 0.804 / – | 0.790 / – |
- At **25 %** of data: InQSS/MOS-SSL degrade ≤ 3 % (quality) and **8 %** (intelligibility); DL degrades **1 %** and
  **5 %**. At **5 %** of data DL drops only **3.4 % (quality)** / **4.6 % (intelligibility)**. PCC gains slow once
  training data exceeds **1,000** utterances (Fig. 4).

**Table 4 — subjective-intelligibility prediction with vs without subjective quality:**
- objective measures only: **MSE 1.771, PCC 0.793, SRCC 0.726**
- objective + subjective quality: **MSE 1.234, PCC 0.870, SRCC 0.756**
- (prose says "PCC value increased from **0.792** to **0.870**" — the 0.792/0.793 mismatch is as printed.)

**Fig. 5 interpretation:** relationships of objective measures to subjective **quality** are roughly linear;
to subjective **intelligibility** they **saturate** (slope flattens). When **DNSMOS-BAK ≈ 2.0**, subjective
intelligibility drops from **9.2 to 8.8** (0–10 scale) and quality from **3.4 to 3.2** (1–5) — attributed to speech
distortion incurred by background-noise suppression.

## Datasets / corpora used
- **TMHINT-QI** (Mandarin Chinese, noisy + enhanced). Noisy: TMHINT clean speech corrupted with **4 noises
  (babble, street, pink, white)** at **4 SNRs (−2, 0, 2, 5 dB)**; then enhanced by **MMSE, KLT, DDAE, FCN,
  transformer**.
- **24,408** collected rating samples → after averaging per utterance and the prescribed split:
  **12,937 training** and **1,978 test** unique utterances with their subjective quality and intelligibility scores.

## Models / systems evaluated
- **Proposed DL combiner** (6 dense layers, GELU, sigmoid, 2 outputs) over 12 objective measures.
- **Linear regression** baseline (per-target).
- **InQSS** (self-supervised + scattering transform, multi-task Q+I).
- **MOS-SSL** (fine-tuned wav2vec2.0 MOS predictor, retrained here).
- Objective measures: **PESQ, ITU-T P.835, DNSMOS P.835, MOSA-Net (quality); NCM, STOI, ESTOI, WER via Google
  ASR (intelligibility)** — i.e. both intrusive and non-intrusive families.

## Human-study details
- **226 individuals, aged 20–50**, participated in the TMHINT-QI listening test (source dataset; follows ref. [21]/InQSS).
- **Quality:** 5-point scale, **1–5** (higher = better). **Intelligibility:** number of correctly recognized words
  in a **ten-word sentence**, score **0–10** (higher = better).
- Per-utterance subjective scores were **averaged** across listeners to obtain ground truth.
- Total **24,408** ratings collected; train/test = 12,937 / 1,978 unique utterances.
- No further protocol detail (environment, screening, device) is printed here — deferred to reference [21].

## Limitations acknowledged by authors
- **No single objective measure reaches a strong correlation (> 0.8)** with human ratings — the core limitation the
  DL combination is designed to work around; individual measures "cannot fully capture subjective quality and
  intelligibility".
- Findings are established on **one Chinese corpus (TMHINT-QI)** with specific noises/SNRs and five enhancement
  systems; cross-language/cross-domain generality is not demonstrated.
- Figure-5 interpretation is explicitly **"limited to several objective measures because of space limitations."**
- The DL model is a black box: interpretation relies on synthetic multivariate-normal inputs rather than the real
  data distribution.
- Interpreted failure mode: **noise suppression trades background cleanliness for speech distortion**, depressing
  both quality and intelligibility (the DNSMOS-BAK ≈ 2.0 knee) — i.e. metrics computed on different axes move
  inconsistently with human judgement.

## Relevance to our project
This is the closest methodological sibling of our Human–ASR Gap work: it is an explicit, quantitative decomposition
of the *disagreement between objective metrics and human listeners*, and it contains the exact ingredient our HAG
metric needs — **ASR word error rate treated as one of the objective "intelligibility" measures**, which here turns
out to be the **single best correlate of human intelligibility**, while being the worst correlate of human quality,
and **STOI↔WER correlation exceeding PESQ↔WER**. That asymmetry is the empirical seed of a formal HAG: for our
CAPTCHA distortions we can compute the same panel (PESQ/MBSD/STOI/ESTOI/NCM + WER) and ask which axis the
distortion moves — if a psychoacoustic distortion pushes quality metrics down while WER and human intelligibility
stay flat (or the reverse, as with the "peak-clipped speech keeps intelligibility but loses quality" logic), the
size and sign of the divergence is a defensible HAG definition rather than a single brittle number. The
**0.68 quality–intelligibility correlation** and the **sub-0.24 quality-metric↔human-intelligibility correlations**
also justify reporting both axes for every CAPTCHA variant. Finally, the practical recipe — a **six-layer dense
combiner that matches or beats InQSS/MOS-SSL while needing only 5 % of the labelled data (PCC 0.777/0.754 at 5 %)**
— is a low-cost way to calibrate objective metrics to a small pool of *our* human CAPTCHA solvers, and the
DNSMOS-BAK ≈ 2.0 knee (intelligibility 9.2→8.8, quality 3.4→3.2) is a concrete threshold to watch when our
distortions start suppressing "noise" that listeners actually use as a cue.

## Keywords
objective vs subjective speech assessment, correlation analysis, PESQ, STOI, ESTOI, NCM, DNSMOS P.835, ITU-T
P.835, MOSA-Net, word error rate, TMHINT-QI, multi-task deep learning, MOS prediction, subjective intelligibility,
quality–intelligibility gap
