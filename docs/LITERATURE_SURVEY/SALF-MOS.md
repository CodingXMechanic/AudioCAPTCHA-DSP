# SALF-MOS: Speaker Agnostic Latent Features Downsampled for MOS Prediction

## Full reference
- **Title:** SALF-MOS: Speaker Agnostic Latent Features Downsampled for MOS Prediction
- **Authors (as printed):** Saurabh Agrawal, Raj Gohil, Gopal Kumar Agrawal, Vikram C M, Kushal Verma
- **Affiliation:** Samsung R&D Institute Bangalore, India (emails sauagr17@gmail.com, raj1996gohil@gmail.com,
  {gopal.kumar, vikram.cm, kushal.verma}@samsung.com)
- **arXiv ID (printed):** **arXiv:2506.02082v1 [cs.SD] 2 Jun 2025**; DOI (PDF metadata):
  https://doi.org/10.48550/arXiv.2506.02082
- **Venue (printed as footnote 1):** "**The paper is accepted at SPCOM 2024**"

## One-line contribution
A **1,574-parameter**, U-Net-style, end-to-end MOS regressor that stacks and downsamples frozen **wav2vec**
latents (no SSL fine-tuning, no listener/domain IDs) and reports state-of-the-art MSE/LCC/SRCC/KTAU on
BVCC, VCC2018, SOMOS and TMHINT-QI.

## Problem & motivation
- MOS is the most reliable way to evaluate TTS/voice-conversion output but is costly, slow, environment/hardware
  sensitive and needs "around 20" knowledgeable listeners per audio to overcome score bias; hard for low-resource
  languages.
- Objective metrics (PESQ, POLQA, STOI) avoid listening tests but are declared **"not feasible in selecting the
  best model"** — they have "limited correlation with speech quality assessment for TTS" and are scoped to
  particular degradation types (background noise, transmission, voice conversion).
- Prior MOS predictors depend on **fine-tuned/pre-trained SSL models, listener IDs or domain IDs**, or ensembles
  of seven SSL models → large, hard to generalise, complex to train (MOSNet, MBNet, LDNet, DDOS, UTMOS,
  FUSION-SSL, MOSPC, NORESQA-MOS are the compared lineage).

## Method (precise but simple)
1. **Feature generation:** frozen **wav2vec** encodings of the audio (also tried MFCC, LFCC, x-vector);
   all audio downsampled to **16 kHz**.
2. **Architecture (U-Net inspired, "depth four"):** four **Double Convolutions** interleaved with **three
   downsampling layers**. Each Double Convolution = two blocks of **1-D conv (kernel 3, stride 1, padding 1) →
   BatchNorm1D → ReLU**.
3. **Latent Feature Extraction (LFE):** after each double convolution the features also go to a linear layer.
4. **Downsampling:** kernel size 2, stride 2 → dimension halved, then another double convolution.
5. **Latent Feature Stacking:** the LFE outputs from all levels are stacked to form the feature-mapping layer, fed
   to a **final linear layer** that learns the MOS mapping (scale of 5).
6. Design claims: no SSL fine-tuning/pre-training, no multi-loss (unlike MOSNet/MBNet), no SSL-model ensembling,
   no listener/domain ID.
7. Metrics: **MSE** (eq. 1), **LCC / concordance correlation** (eq. 2), **SRCC** (eq. 3), **KTAU** (eq. 4).

## Key quantitative results
**Table I — benchmark (MSE / LCC / SRCC / KTAU), test split:**

| Model | #Params | BVCC | VCC2018 | SOMOS | TMHINTQI |
|---|---|---|---|---|---|
| **SALF-MOS** | **1574** | **0.144 / 0.948 / 0.946 / 0.819** | **0.336 / 0.825 / 0.829 / 0.678** | **0.15 / 0.773 / 0.771 / 0.583** | **0.383 / 0.795 / 0.747 / 0.576** |
| NORESQA-MOS | 92M | 0.17 / 0.89 / 0.87 / – | – | – | – |
| UTMOS | – | 0.219 / 0.8822 / 0.88 / 0.707 | – | – | – |
| DDOS | – | 0.201 / 0.877 / 0.875 / 0.701 | – | – | – |
| MOSNet | – | 0.816 / 0.294 / 0.263 / – | 0.538 / 0.643 / 0.589 / – | 0.24 / 0.515 / 0.498 / 0.346 | – |
| LDNet | 0.96M | 0.338 / 0.774 / 0.773 / 0.582 | 0.441 / 0.664 / 0.626 / 0.465 | 0.223 / 0.584 / 0.568 / 0.401 | – |
| MBNet | 1.38M | 0.433 / 0.727 / 0.753 / 0564 | 0426 / 0.68 / 0.647 / – | – | – |
| Fusion-SSL | – | 0.156 / 0.902 / 0.901 / 0.735 | 0.359 / 0.74 / 0.711 / 0.542 | – | – |
| MOSPC | – | 0.148 / 0.906 / 0.906 / 0.742 | 0.352 / 0.748 / 0.721 / 0.551 | – | – |
| LE-SSL-MOS | – | – | – | – | 0.951 / 0.537 / 0.517 / 0.388 |
("0564", "0426" printed without decimal points in the PDF.)

**Table II — datasets:** BVCC 7,106 samples / 8.02 h · VCC2018 20,871 / 19.09 h · SOMOS 20,100 / 26.17 h ·
TMHINTQI 14,915 / 13.07 h.

**Table III — feature ablation on BVCC (MSE / LCC / SRCC / KTAU):**
MFCC 0.56 / 0.54 / 0.55 / 0.49 · LFCC 0.43 / 0.683 / 0.68 / 0.62 · X-Vector 0.48 / 0.47 / 0.48 / 0.43 ·
**wav2vec 0.144 / 0.948 / 0.946 / 0.81**.

**Training setup:** learning rate **1e-4**, **L1 loss**, **SGD** optimizer, batch size **4 per epoch**,
**early stopping at 20 epochs**, split **8 : 1 : 1** (train/val/test), results reported on **test** data,
single **A10 GPU, 26 GB**, Ubuntu 20.04, 4 CPU cores × 16 GB RAM @ 3.00 GHz.
**Depth ablation (Fig. 5, BVCC):** deeper models under-fit; **optimal depth = 4** (4 double-convolutions +
3 downsampling layers).

## Datasets / corpora used
**BVCC** (VoiceBiosynthesis/Voices MOS corpus), **VCC2018** (Voice Conversion Challenge 2018), **SOMOS** (Samsung
Open MOS Dataset), **TMHINTQI** (Mandarin quality+intelligibility, same corpus used by the MOSA-Net and
"Study on the Correlation…" papers). All audio resampled to **16 kHz**. Distribution note printed: BVCC is
normally distributed around MOS ≈ 3, while TMHINTQI spans all MOS scores (attributed to LE-SSL-MOS's poor
TMHINTQI result, LCC 0.537).

## Models / systems evaluated
- **SALF-MOS** (proposed), three input choices ablated: **MFCC, LFCC, x-vector, wav2vec**.
- Compared MOS predictors: **NORESQA-MOS, UTMOS, DDOS, MOSNet, LDNet, MBNet, Fusion-SSL, MOSPC, LE-SSL-MOS**
  (numbers taken from their papers; SALF-MOS runs its own).
- Objective quality/intelligibility metrics discussed as alternatives that are deemed unsuitable for TTS model
  selection: **PESQ, POLQA, STOI**.

## Human-study details
- No listening test conducted by the authors; MOS labels come from the four public datasets (BVCC, VCC2018,
  SOMOS, TMHINTQI), whose collection protocols/subject counts are **not printed here**.
- Design-relevant statement from the introduction: *"Generally, for each audio, a good number of listeners are
  required (**around 20** to overcome score bias) who have the domain knowledge of the language"*, and listeners
  must rate on a **5-point scale**; listening environment and hardware must be controlled.

## Limitations acknowledged by authors
- No dedicated limitations section; the ablation states that **MFCC, LFCC and x-vector features "were not able to
  generalize well on the data sets used"** (only wav2vec reaches SOTA), and that increasing model depth causes
  **underfitting** → poor MSE (optimal depth fixed at 4).
- Implicit limits: comparisons on BVCC/VCC2018/SOMOS rely on **numbers reported by other papers** rather than
  re-training (no significance tests, no error bars printed); SOMOS/TMHINTQI coverage is partial for baselines;
  only one training seed/split (8:1:1) is reported; results are on a single domain per dataset and no unseen-noise
  or unseen-domain test is reported.

## Relevance to our project
SALF-MOS matters to AudioCAPTCHA-DSP mainly as the **cheap, reference-free "human-quality" estimator** that could
close the third leg of our triangle (human intelligibility proxy ≈ STOI, machine intelligibility = ASR WER, human
*perceived quality* ≈ MOS). Two printed results are directly actionable: (1) the paper's own framing that
**PESQ/POLQA/STOI are "not feasible" for selecting among synthesized/processed speech systems** — which is exactly
our situation when we rank CAPTCHA distortion recipes applied to TTS prompts, so a MOS predictor should accompany
STOI/MBSD/PESQ rather than be replaced by them; and (2) its **TMHINT-QI column (LCC 0.795, SRCC 0.747, MSE 0.383)**
and its note that LE-SSL-MOS collapses there (LCC 0.537) because TMHINT-QI's MOS distribution is spread rather than
clustered around 3 — a warning that MOS predictors trained on BVCC-like data behave differently on the noisy,
distortion-rich corpora our CAPTCHAs resemble. Because the model is only **1,574 parameters** and needs **no SSL
fine-tuning**, it is also a realistic in-browser/on-device assessor for A/B testing CAPTCHA variants, and its
speaker-agnostic claim (no speaker/listener/domain ID) is the property we need if HAG measurements must hold across
TTS voices, noise beds and distortion types. Its feature ablation (wav2vec LCC 0.948 vs LFCC 0.683 vs MFCC 0.54 vs
x-vector 0.47) is a compact argument for using SSL features — already used elsewhere in our pipeline (wav2vec 2.0,
HuBERT) — when any learned assessor is part of the evaluation loop.

## Keywords
MOS prediction, Mean Opinion Score, SALF-MOS, U-Net, wav2vec, self-supervised learning, TTS evaluation, voice
conversion, speaker agnostic, latent feature downsampling, model compression, LCC, SRCC, Kendall tau, non-intrusive
quality assessment
