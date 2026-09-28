# aaeCAPTCHA: The Design and Implementation of Audio Adversarial CAPTCHA

## Full reference
- Title: aaeCAPTCHA: The Design and Implementation of Audio Adversarial CAPTCHA
- Authors (as printed): Md Imran Hossen (University of Louisiana at Lafayette), Xiali Hei (University of Louisiana at Lafayette)
- Venue as printed: arXiv preprint **arXiv:2203.02735v1 [cs.CR], 5 Mar 2022**. No conference/journal venue or DOI printed on the PDF.
- Funding (as printed): US NSF grants OIA-1946231 and CNS-2117785.

## One-line contribution
Turns the vulnerability of ASR models into a defense: PGD-generated *untargeted* audio adversarial examples are served as CAPTCHA challenges, giving 100% adversarial success against the target DeepSpeech and 0–4.4% attack success for five other open-source ASRs (56–80% WER) at a ~11-point human success-rate cost (85.13% → 74.24%).

## Problem & motivation
- Audio CAPTCHAs are the accessible alternative for visually impaired users, but prior work shows they are highly breakable by ASR/STT: unCaptcha broke reCAPTCHA audio with **85.15%** accuracy; Solanki et al. broke every audio CAPTCHA scheme they analyzed.
- DNN-ASRs are vulnerable to adversarial examples; psychoacoustic-hiding attacks (Schönherr et al. [68], Qin et al. [63]) make audio adversarial examples imperceptible, and countermeasures (quantization, smoothing, filtering, compression) are often broken in adaptive settings.
- Prior art covered text/image *adversarial* CAPTCHAs [58],[71] and a suggestion to use audio adversarial examples for CAPTCHAs [1], but no comprehensive security + usability evaluation of an audio adversarial CAPTCHA existed.
- Design idea: ask humans to transcribe audio with an adversarial perturbation — humans can still understand it (imperceptibility is explicitly *not* a goal; intelligibility is), while ASR systems mistranscribe it.

## Method (precise but simple language)
- Untargeted objective: find δ with `f(x+δ) ≠ f(x)` s.t. `‖δ‖∞ ≤ ε` and `x+δ ∈ [−M, M]` (M = 2^15), minimizing `ℓ(x,δ,y) = −c1·ℓ_net(f(x+δ),y) + c2·‖δ‖2` with ℓ_net = CTC loss of DeepSpeech (Eq. 1–3).
- Solver: PGD — `δ0 = 0; δ_{t+1} = clip_ε(δ_t − α·sign(∇_δ ℓ))` (Eq. 4); a single-step FGSM variant is also evaluated (Eq. 5). Global settings: c1 = 1, c2 = 0.
- Hyperparameter search: ε ∈ {250, 300, 350, 400} (step 50), steps ∈ [20,100] step 10, α ∈ [20,100] step 10 → 81 (steps, α) combos per ε; 40,500 adversarial audios per ε, **162,000 total**.
- Chosen "optimal" CAPTCHA hyperparameters: **ε = 350, steps = 50, α = 40** (trade-off: ‖δ‖1 below 230 was found vulnerable to GCP/Wit.ai STT, which could still transcribe >20% correctly).
- Metrics: WER = (S+D+I)/NW (Eq. 6); adversarial success rate Aa = N_f/N_a (non-zero WER) (Eq. 7); **SRoA = 1 − Aa** = share of samples still *correctly* transcribed (Eq. 8, attacker's success rate); SNR(dB) = 10·log10(P_x/P_δ) (Eq. 9). Also ‖δ‖1 as a distortion proxy.
- Requirements for the generator: adversarial, transferable, robust, human-intelligible (not imperceptible), efficient.

## Key quantitative results
**FGSM on DeepSpeech (Table 1, 500 LibriSpeech dev-clean samples):**

| ε | 250 | 300 | 350 | 400 |
|---|---|---|---|---|
| WER (%) | 55.74 | 59.77 | 63.46 | 67.58 |
| Aa (%) | 94.80 | 96.00 | 97.20 | 98.40 |
| SNR (dB) | 16.95 | 15.37 | 14.03 | 12.87 |

(ε < 200 usually gives too little transcription error; FGSM never reaches 100% Aa and degrades SNR → deemed impractical.)

**PGD (Table 3, means over 162,000 adversarial audios):** Aa = **100.00% (std 0.00) for every ε**; mean WER 111.82 / 112.26 / 112.53 / 112.18 for ε = 250/300/350/400; SNR 17.53 / 16.16 / 15.04 / 14.10 dB; ‖δ‖1 227.40 / 263.51 / 296.80 / 327.78. (WER >100% because insertions dominate.)
**Selected optimal config (Table 4): ε=350, steps=50, α=40 → WER 111.88%, Aa 100.00%, SNR 15.89 dB, ‖δ‖1 256.61.** Generation cost: 500 CAPTCHAs in **3.67 minutes** (batch size 50).

**Transferability, Table 5 (500 adversarial vs normal audios; WER % and SRoA %):**

| Model | DeepSpeech | DS2 | Jasper | W2L+ | Lingvo | Kaldi |
|---|---|---|---|---|---|---|
| WER normal / adv | 6.63 / 111.88 | 5.65 / 78.65 | 2.49 / 56.32 | 4.83 / 76.40 | 2.98 / 74.64 | 8.32 / 80.45 |
| SRoA normal / adv | 66.00 / **0.00** | 68.60 / 0.40 | 84.40 / **4.40** | 71.40 / 0.40 | 82.20 / 0.40 | 59.20 / 0.40 |

**Commercial STT (Table 6):**

| Model | GCP STT | IBM Watson | Wit.ai |
|---|---|---|---|
| WER normal / adv | 8.37 / 34.75 | 7.34 / 58.74 | 5.52 / 36.57 |
| SRoA normal / adv | 61.60 / **17.60** | 63.80 / **5.60** | 68.00 / **12.20** |

**Adaptive attacks (SRoA stays ≈0% everywhere):** quantization q = 128/256/512/1024 → WER 109.51/108.87/107.46/87.69, SRoA 0.00 for all q; average smoothing k=3…17 → WER 107.80→100.53, SRoA 0; median smoothing best at k=9 → WER 84.95, SRoA 0; down-sampling 5.6/6.4/7.2/8 kHz → WER 88.51/93.20/98.48/102.71, SRoA 0; low-pass 1.5 kHz → WER 85.88, SRoA 0 (band-pass 2 kHz gives the only non-zero: 0.20%); compression MP3/OPUS/AAC/SPEEX → WER 99.08/105.79/107.08/104.12, SRoA 0.
- Strongest preprocessing attack = quantization q=1024: raises Jasper SRoA 4.40→**16.40%**, GCP 17.60→23.40%, Wit.ai 12.20→17.20% (Table 13); countered by BPDA-in-the-loop generation, which drops Jasper to 6% and WER stays 94.85 (mean SNR 15.58 dB, ‖δ‖1 271.20 vs 15.89/256.61).
- Adversarial training (Table 15, SRoA % under FGSM-AT / PGD-AT): Jasper 23.60 / 34.80, Kaldi 12.80 / 21.00, DS2 6.80 / 14.60, Lingvo 7.80 / 16.60, DeepSpeech 0.20 / 13.20 → AT is the most effective attack; countermeasure = regenerate CAPTCHAs against the adversarially trained model → SRoA 0% for both FGSM- and PGD-AT DeepSpeech, and <7% for all models except Jasper (14.80 / 13.60%) (Table 16).
- Usability, Table 17 (100 MTurk users, WER = 0 counts as solved):

| Condition | Normal | ε=250 | ε=300 | ε=350 | ε=350 (BPDA) | ε=400 |
|---|---|---|---|---|---|---|
| N | 195 | 197 | 194 | 198 | 197 | 196 |
| Success rate (%) | **85.13** | 76.65 | 72.68 | **74.24** | 72.08 | **70.41** |
| Avg time (s) | 41.83 | 43.76 | 44.83 | 48.86 | 45.11 | 59.69 |
| Median time (s) | 28.50 | 27.50 | 29.00 | 28.00 | 31.00 | 48.00 |

- Human success drops monotonically with ε; ε=350 costs ~11 percentage points vs normal audio (85.13 → 74.24); ε=400 is worst (70.41%) with avg 59.69 s.
- By transcription length (Table 18, normal vs ε=350): 6 words 91.89% vs 78.12%; 12 words 75.00% vs 58.33% — shorter clips are easier.
- Sample transcriptions (Table 20): ground truth "yes he's mercurial in all his movements" → e.g. Kaldi "…for the erse who using off claudia yer ole kit arms he lay four hissed", IBM Watson "…yes i can only" (STT results collected January 2022).

## Datasets / corpora used
- **LibriSpeech** (~1,000 h of 16 kHz English audiobook speech); all experiments on **500 randomly selected dev-clean samples** with transcription lengths 6–12 words (mean 8.98 words).
- Adversarial training (attack side): LibriSpeech train-clean-100 (clips < 16 s), dev-other, dev-clean (excluding the 500 evaluation audios).

## Models / systems evaluated
- Target/attack model: **DeepSpeech v0.4.1** (Mozilla), pretrained on LibriSpeech; differentiable MFCC (TensorFlow 1.12, Python 3.6).
- Other open-source ASRs: DeepSpeech 2, Jasper, Wave2Letter+ (NVIDIA OpenSeq2Seq), Lingvo (from Qin et al.), Kaldi (Librispeech ASR Chain 1d, DNN-HMM).
- Commercial STT: Google Cloud STT, IBM Watson STT, Wit.ai.
- Adaptive attacks: quantization, average/median smoothing, down-sampling, low-pass/band-pass filtering, MP3/OPUS/AAC/SPEEX compression, BPDA, adversarial training (FGSM-AT and PGD-AT, 32 epochs, batch 64).

## Human-study details (if any)
- 100 users recruited from Amazon Mechanical Turk, US-based, HIT approval ≥ 85%; IRB-approved (Exempt).
- Each participant saw **12 CAPTCHAs**: 2 normal, 2 each for ε = 250/300/350/400, and 2 BPDA (ε=350) → 1200 submissions; 23 discarded for abnormally long times (e.g. >350 s for a 2-s clip) → **1177 analyzed**.
- Task: transcribe the audio; solved iff WER vs ground truth = 0.
- Headline: normal 85.13% success / 41.83 s avg; ε=350 74.24% / 48.86 s; ε=400 70.41% / 59.69 s.
- No PII collected; because of this, demographic breakdowns (age/gender) could not be reported (footnote 5).

## Limitations acknowledged by authors
- **No standard exists for "human-intelligible" perturbation budgets** — budgets were set from experience/preliminary evaluation; dedicated research on the security–usability trade-off is needed.
- The adversarial-training attack (§8.3.4) represents a *theoretical bound* under a fully informed adversary; periodic retraining of the target ASR + regeneration may be unnecessary in realistic settings, and practical AT attacks/defenses need more study.
- Perceptibility of perturbations is not minimized; the authors explicitly suggest incorporating **psychoacoustic hiding** (citing Schönherr et al. [68] and Abdullah et al. [2]) to make challenges more human-intelligible and less audibly distorted.
- Only the acoustic model was attacked; attacking the shared front-end pipeline (feature extraction) could yield cheaper, less distorted, more transferable samples [1].
- aaeCAPTCHA is positioned as an *enhancement* of existing audio CAPTCHAs, not a replacement.

## Relevance to our project
aaeCAPTCHA is the closest existing construction to our Human–ASR Gap framing: it explicitly optimizes one side of the gap (Aa/SRoA against ASR) while measuring the other side (human success rate and completion time), and reports the pair — 74.24% human / 0.00% machine at ε=350 — as a single design point, i.e. an implicit HAG of ~74 points. For our framework it contributes: (i) a well-defined metric suite for the ASR side (WER with insertions allowed to exceed 100%, Aa, SRoA = 1−Aa, SNR, ‖δ‖1) that complements STOI-style human intelligibility proxies; (ii) a ready-made catalogue of DSP-domain *adaptive* preprocessing attacks — quantization, smoothing, down-sampling, filtering, lossy compression — that mirror the distortion operators we apply, plus evidence that MP3/compression alone barely moves the gap (WER 99.08%, SRoA 0%); and (iii) a concrete invitation to combine our psychoacoustic-distortion track with adversarial perturbations, since the authors state they do not minimize perceptibility and point to psychoacoustic hiding as future work. Its usability protocol (100 MTurk users × 12 items, WER=0 success criterion, discarded-timing outliers) is a directly reusable template for our human arm, and its key limitation — no validated measure of "human-intelligible" distortion — is precisely the gap our STOI-based proxy is meant to fill.

## Keywords
audio CAPTCHA, adversarial examples, PGD, FGSM, DeepSpeech, transferability, word error rate, adversarial success rate, signal-to-noise ratio, audio preprocessing defenses, quantization, MP3 compression, adversarial training, usability study, LibriSpeech, human-vs-ASR gap
