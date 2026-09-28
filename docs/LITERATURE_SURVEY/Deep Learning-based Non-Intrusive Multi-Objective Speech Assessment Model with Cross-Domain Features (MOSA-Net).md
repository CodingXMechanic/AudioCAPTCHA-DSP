# Deep Learning-based Non-Intrusive Multi-Objective Speech Assessment Model with Cross-Domain Features (MOSA-Net)

## Full reference
- **Title:** Deep Learning-based Non-Intrusive Multi-Objective Speech Assessment Model with Cross-Domain Features
- **Authors (as printed):** Ryandhimas E. Zezario (Student Member, IEEE), Szu-Wei Fu, Fei Chen (Senior Member, IEEE),
  Chiou-Shann Fuh, Hsin-Min Wang (Senior Member, IEEE), Yu Tsao (Senior Member, IEEE)
- **Affiliations:** Dept. of CSIE, National Taiwan University & Research Center for Information Technology
  Innovation, Academia Sinica (Taipei); Microsoft (Vancouver, Canada); Dept. of EEE, Southern University of Science
  and Technology of China (Shenzhen); Institute of Information Science, Academia Sinica. Corresponding: yu.tsao@sinica.edu.tw
- **arXiv ID (printed on p. 1):** **arXiv:2111.02363v5 [eess.AS] 19 Dec 2024**
- **Venue / year:** No journal or conference name is printed in this PDF (IEEE-format preprint).
  Cross-reference printed in two other assigned PDFs (SALF-MOS ref. [9]; "Study on the Correlation…" ref. [23]):
  *"IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 54–70, 2022."*
- **Code link printed:** https://github.com/dhimasryan/MOSA-Net-Cross-Domain

## One-line contribution
A **non-intrusive, multi-task** speech-assessment network (**CRNN + multiplicative attention**) that fuses
**cross-domain inputs** (power-spectrogram / complex / learnable-filter-bank + SSL embeddings from wav2vec 2.0 and
HuBERT) to predict **PESQ, STOI and SDI simultaneously**, and doubles as a pre-trained model adaptable to
**human MOS/intelligibility** prediction and as a guide for speech enhancement (QIA-SE).

## Problem & motivation
- Subjective tests (MOS, word-recognition scores) are the ground truth but are expensive, slow and hard to extend to
  new domains; objective surrogates are needed.
- Existing DL assessors use **one acoustic feature type** and **one training objective**; objective metric labels
  (PESQ/STOI/SDI) and human subjective ratings are themselves only *moderately* correlated.
- Intrusive metrics (PESQ, POLQA) need a clean reference; non-intrusive ones are more deployable but weaker.
- Objective-intelligibility families reviewed: AI/SII/ESII/CSII (subband-SNR) and STI/NCM/STOI/eSTOI/NSIM/
  wSTMI (modulation-depth), plus non-intrusive ModA, SRMR, non-intrusive STOI.

## Method (precise but simple)
1. Waveform X feeds **two branches**: (a) STFT → **PS (power-spectrogram)** and **learnable filter banks (LFB,
   SincNet)**; (b) **SSL model** (wav2vec 2.0 or HuBERT) → latent representation, reduced by a linear layer.
   Features are concatenated along the **time dimension** (total frames = frames of PS + LFB + SSL).
2. Shared trunk: **12 convolutional layers** (channels {16, 32, 64, 128}, strides {1, 1, 3}) → **1-layer BLSTM
   (128 nodes)** → fully-connected (128) → per-metric **multiplicative attention** → per-metric FC (1 neuron) →
   **global average** over frames → utterance score. (CNN-BLSTM without attention is called "CRNN"; with attention,
   "CRNN+AT".)
3. Loss (Eq. 1), weights all set to 1:
   `L_All = γ₁L_PESQ + γ₂L_STOI + γ₃L_SDI`, each `L = (1/N)Σₙ [ (targetₙ − target̂ₙ)² + α·(1/L(Uₙ))Σₗ
   (targetₙ − target̂ₙₗ)² ]` — i.e. **utterance-level MSE + frame-level MSE**; frame-level ground truth = the
   utterance ground truth; L(Uₙ)= frames of STFT + LFB + SSL.
4. **QIA-SE:** MOSA-Net (frozen) produces latent code A = FC-after-attention output; A is concatenated into a
   middle layer of a 12-conv-layer CNN enhancer: `Hₖ₊₁ = F_θ^([Hₖᵀ, Aᵀ]ᵀ)`, trained by MSE to clean X;
   ISTFT with the noisy phase rebuilds the waveform.
5. Evaluation metrics: **MSE, LCC, SRCC**; statistical significance by t-test on 20 matched pairs (5 utterances each),
   **p < 0.05**.

## Key quantitative results
**Abstract headline deltas (as printed):**
- PESQ LCC: **+0.026 (0.990 vs 0.964, seen noise)** and **+0.012 (0.969 vs 0.957, unseen)** vs **Quality-Net**.
- STOI LCC: **+0.021 (0.985 vs 0.964, seen)** and **+0.047 (0.836 vs 0.789, unseen)** vs **STOI-Net** (CRNN-based).
- MOS LCC: **+0.018 (0.805 vs 0.787)** vs **MOS-SSL** after adaptation with limited data.
- QIA-SE PESQ: **+0.301 (2.953 vs 2.652, seen)** and **+0.18 (2.658 vs 2.478, unseen)** over the CNN baseline SE.

**Table I — architecture study (PS features, single metric), LCC / SRCC / MSE:**
- PESQ, seen: BLSTM 0.964/0.945/0.074 · CNN 0.975/0.959/0.055 · CRNN 0.981/0.965/0.042 · **CRNN+AT 0.982/0.967/0.040**
- PESQ, unseen: BLSTM 0.957/0.932/0.075 · CNN 0.947/0.931/0.117 · CRNN 0.966/0.949/0.078 · CRNN+AT 0.965/0.954/0.092
- STOI, seen: BLSTM 0.923/0.929/0.005 · CNN 0.936/0.939/0.004 · CRNN 0.964/0.962/0.002 · **CRNN+AT 0.970/0.968/0.001**
- STOI, unseen: BLSTM 0.764/0.784/0.029 · CNN 0.698/0.694/0.012 · CRNN 0.789/0.797/0.016 · **CRNN+AT 0.827/0.815/0.015**

**Table II — vs baselines (LCC/SRCC/MSE):** PESQ: Quality-Net 0.964/0.945/0.074 (seen), 0.957/0.932/0.075* (unseen)
vs MOSA-Net 0.982*/0.967*/0.040* (seen), 0.965*/0.954/0.092 (unseen). STOI: STOI-Net 0.964/0.962/0.002 (seen),
0.789/0.797/0.016 (unseen) vs MOSA-Net 0.970*/0.968*/0.001* (seen), 0.827*/0.815*/0.015* (unseen).
All differences significant at **p < 0.05**, except the PESQ MSE under unseen noise (MOSA-Net worse).

**Table III — single vs multi-task (LCC/SRCC/MSE), seen:**
- PESQ: Q 0.982/0.965/0.043 → Q+I 0.987/0.974/0.028 → Q+D 0.986/0.975/0.028 → **Q+I+D 0.988*/0.977*/0.026***
- STOI: I 0.970/0.968/0.001 → Q+I 0.971/0.968/0.002 → I+D 0.973/0.968/0.001 → **Q+I+D 0.977*/0.974*/0.001**
- SDI: D 0.883/0.904/0.045 → Q+D 0.939/0.947/0.024 → **I+D 0.952*/0.955*/0.019*** → Q+I+D 0.947/0.954/0.022
- **Unseen noise: neither double- nor triple-task improved over single-task** (e.g. STOI unseen I 0.827 vs Q+I 0.802,
  I+D 0.785, Q+I+D 0.790).

**Table IV — vs AMSA (multi-task):** PESQ seen AMSA 0.985/0.973/0.031 vs MOSA-Net 0.988/0.977/0.026;
STOI seen 0.975/0.973/0.001 vs 0.977/0.974/0.001; SDI seen 0.929/0.942/0.029 vs 0.947/0.954/0.022
(MOSA-Net better on all LCC/SRCC/MSE in both seen and unseen).

**Table V — single features (PESQ LCC seen):** PS 0.988, Complex 0.985, LFB 0.981, SSL(W2V) 0.984, SSL(Hub) 0.981;
STOI seen best = SSL(Hub) 0.980; STOI unseen best = SSL(Hub) 0.807.

**Table VI — cross-domain features (best rows):** PESQ seen **PS+SSL(Hub) 0.991/0.981/0.020**; unseen
**PS+LFB+SSL(Hub) 0.969/0.957/0.070**. STOI seen **PS+SSL(Hub) 0.989/0.985/0.001**; unseen
**PS+LFB+SSL(Hub) 0.836/0.839/0.017**. SDI seen **Complex+SSL(Hub) 0.971/0.973/0.012**; unseen
**Complex+LFB+SSL(Hub) 0.895/0.899/0.033**. All cross-domain vs single-domain gains significant at p < 0.05.

**Table VII — WSJ-trained models tested on TIMIT (LCC/SRCC/MSE):**
- PESQ: AMSA 0.728/0.673/0.765 · MOSA-Net 0.754/0.710/0.654 · **MOSA-Net(Cross-Domain) 0.960/0.948/0.111**
- STOI: AMSA 0.701/0.584/0.015 · MOSA-Net 0.746/0.608/0.011 · **Cross-Domain 0.920/0.936/0.004**
- SDI: AMSA 0.873/0.678/0.042 · MOSA-Net 0.859/0.643/0.047 · **Cross-Domain 0.895/0.901/0.051**
  (8 of 9 cross-domain-vs-AMSA improvements significant, p < 0.05.)

**Table VIII — human listening-test prediction (LCC/SRCC/MSE), TMHINT-QI:**
- MOS: CSIG 0.555/0.453 · CBAK 0.545/0.343 · COVL 0.556/0.450 · MOSNet 0.724/0.656/0.489 ·
  MOS-SSL 0.787/0.746/0.440 · MOSA-Net(WSJ) 0.535/0.371/1.636 · MOSA-Net(Scratch) 0.777/0.724/0.411 ·
  MOSA-Net(Adapt) 0.795/0.742/0.389 · Scratch_FT−SSL 0.804/0.758/0.360 · **Adapt_FT−SSL 0.805/0.763/0.356**
- Intelligibility: ESTOI 0.461/0.465/0.162 · MOSNet 0.658/0.607/0.027 · MOS-SSL 0.760/0.655/0.024 ·
  MOSA-Net(WSJ) 0.385/0.378/0.056 · Scratch 0.740/0.698/0.023 · Adapt 0.756/0.702/0.021 ·
  Scratch_FT−SSL 0.796/0.712/0.018 · **Adapt_FT−SSL 0.807/0.730/0.017**
- **Fig. 8 (objective vs human, all test utterances):** subjective **intelligibility vs STOI: LCC 0.482, SRCC 0.461**;
  subjective **MOS vs PESQ: LCC 0.574, SRCC 0.377** → only *moderate* alignment.

**Table IX — speech enhancement (avg PESQ / STOI / CSIG / SSNRI):**
- Seen: noisy 2.211/0.830/1.790/– ; CNN 2.652/0.850/2.091/4.135 ; SSEMS 2.675/0.851/2.089/4.541 ;
  ZMOS 2.678/0.851/2.060/3.640 ; **QIA-SE 2.953/0.868/2.169/5.516**
- Unseen: noisy 2.118/0.799/1.989/– ; CNN 2.478/0.820/2.238/3.178 ; SSEMS 2.484/0.820/2.227/3.506 ;
  ZMOS 2.507/0.821/2.207/2.683 ; **QIA-SE 2.658/0.828/2.312/5.731**

## Datasets / corpora used
- **WSJ** (Wall Street Journal): 37,416 training utterances, 330 test utterances, 16 kHz.
- **PN 100 nonspeech sounds** noise corpus (100 noise types) at **31 SNR levels, −10 to 20 dB, 1 dB step** →
  training pairs; a BLSTM SE model (2 bi-layers × 300 units) produced the enhanced utterances.
- MOSA-Net training set: 1,500 clean + 15,000 noisy + 15,000 enhanced utterances.
- **Seen test:** 300 clean + 2,350 noisy + 2,350 enhanced. **Unseen test:** 300 WSJ-test utterances × 4 unseen
  noises (car, pink, street, babble) × 6 SNRs (−10, −5, 0, 5, 10, 15 dB) = 7,200 noisy; 2,350 noisy + 2,350
  enhanced + 300 clean selected; unseen speakers excluded from training.
- **TIMIT** (cross-dataset test): 750 noisy + 750 enhanced + 500 clean; noises at 8 SNRs (−10…25 dB, step 5).
- **TMHINT-QI** (Mandarin, subjective): clean/noisy/enhanced from 5 SE methods (KLT, MMSE, FCN, DDAE, Transformer);
  1,900 test utterances (rated by multiple subjects), 15,000 training utterances (single subject each).
- **TMHINT** (SE task): 1,200 train utterances (3 male + 3 female × 200), 36,000 noisy (100 noises, 31 SNRs);
  test 120 utterances (1 male + 1 female) → 120 seen + 120 unseen noisy at 6 SNRs.

## Models / systems evaluated
- Proposed: **MOSA-Net** in variants CNN, CRNN, CRNN+AT, single- vs multi-task (Q/I/D), five single feature types
  (PS, Complex, LFB, SSL(W2V), SSL(Hub)) and combinations; **MOSA-Net(WSJ / Scratch / Adapt / …FT−SSL)** for MOS.
- Baselines: **Quality-Net** (BLSTM, PESQ), **STOI-Net** (CRNN, STOI), **AMSA** (attention-enhanced multi-task),
  **MOSNet**, **MOS-SSL**, **CSIG/CBAK/COVL**, **ESTOI**, **DNSMOS** (referenced).
- SE systems: **CNN** baseline, **SSEMS**, **ZMOS**, **QIA-SE**; SE trained with BLSTM (2×300).
- Underlying SSL encoders: **wav2vec 2.0**, **HuBERT**; learnable filterbank: **SincNet**.
- Note: WSJ-trained MOSA-Net(WSJ) is also cited by the authors' follow-ups for **ASR WER prediction** (MTI-Net)
  and hearing-loss intelligibility (MBI-Net).

## Human-study details
- **TMHINT-QI listening test: 226 subjects** participated (source dataset, not run by these authors).
- Protocol (as printed): quality rated **1–5**; intelligibility = proportion of recognizable characters, **0–1**
  (each utterance = **10 Chinese characters**, so per-utterance human scores fall on 0.1 steps).
- Most utterances rated by **one** subject, some by more than one; the authors use **single-subject** ratings for
  training (15,000) and **averaged** scores over 1,900 multi-rated utterances for testing.
- Informed consent approved by the **Academia Sinica Institutional Review Board**.
- Correlation of the human labels with the objective metrics: **STOI↔intelligibility LCC 0.482 / SRCC 0.461**;
  **PESQ↔MOS LCC 0.574 / SRCC 0.377**.

## Limitations acknowledged by authors
- **Multi-task gains do not transfer to unseen noise** (Table III): "neither double-task nor triple-task criteria led
  to performance gains" under unseen conditions; PESQ MSE under unseen noise is worse than Quality-Net's (0.092 vs
  0.075).
- **Objective labels ≠ human labels:** MOSA-Net(WSJ) predicts subjective MOS/intelligibility poorly
  (MOS LCC 0.535, intelligibility LCC 0.385) "due to data mismatch and the gap between the PESQ/STOI metrics and the
  subjective quality/intelligibility scores".
- Non-intrusive metrics "generally have lower assessment capabilities" than intrusive ones (field-level caveat).
- Future work stated: apply the architecture to other metrics, **automatically optimize loss scaling factors**, and
  "improving the robustness of MOSA-Net in real-world scenarios" (i.e. robustness is an acknowledged open issue).
- Subjective ground truth is expensive; adapting with limited data is a workaround, not a full solution.

## Relevance to our project
MOSA-Net supplies the missing middle rung of our evaluation ladder: we already plan **STOI** (human-intelligibility
proxy) and **ASR WER** (machine intelligibility), and this paper quantifies precisely how far both can drift from
actual humans — **STOI↔human intelligibility only LCC 0.482/SRCC 0.461 and PESQ↔MOS only LCC 0.574/SRCC 0.377 on
TMHINT-QI**, i.e. the exact caveat any Human–ASR Gap number must carry. It also gives us a ready-made, open-source
**non-intrusive multi-metric assessor** (PESQ + STOI + SDI heads, GitHub link printed) that can score CAPTCHA audio
**without a clean reference** — useful when our psychoacoustic distortions are applied to an already-degraded
CAPTCHA recording where the "reference" is itself synthetic. Two methodological lessons transfer directly: (1)
**cross-domain features (spectral + learnable filterbank + SSL) are what make metrics generalize** to unseen noise
and unseen corpora (TIMIT LCC 0.960 vs 0.728 for a single-domain model), so our HAG study should prefer SSL-backed
or cross-domain metrics over any single-feature metric when stimuli move from clean TTS to noisy/distorted
CAPTCHA conditions; and (2) **objective-score training can be adapted to predict human ratings with little data**
(0.805 vs 0.787 MOS LCC), which is a practical recipe for calibrating a metric to *our* CAPTCHA listener pool
instead of assuming ITU-standard PESQ/STOI behaviour. The paper's finding that better assessor → better downstream
task (QIA-SE) also argues for validating our metric suite before trusting it to rank CAPTCHA distortion recipes.

## Keywords
MOSA-Net, non-intrusive speech assessment, multi-task learning, cross-domain features, PESQ prediction, STOI
prediction, speech distortion index, CRNN, BLSTM, multiplicative attention, self-supervised learning, wav2vec 2.0,
HuBERT, MOS prediction, speech enhancement, human-vs-objective correlation
