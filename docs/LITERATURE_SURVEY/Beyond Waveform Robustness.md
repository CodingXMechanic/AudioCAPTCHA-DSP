# Beyond Waveform Robustness: Robust Feature-Vocoder Adversarial Attacks on Automatic Speech Recognition

## Full reference
Yifan Liao, Zongmin Zhang, Zhen Sun, Yuhui Sun, Xinhu Zheng, Xinlei He. *Beyond Waveform Robustness: Robust Feature-Vocoder Adversarial Attacks on Automatic Speech Recognition.* arXiv:2606.05678v1 [cs.SD], 4 Jun 2026. Affiliations: 1The Hong Kong University of Science and Technology (Guangzhou); 2Wuhan University. Corresponding author: Xinlei He (xinlei.he@whu.edu.cn). No venue printed; no DOI printed.

## One-line contribution
A surrogate-based black-box attack that moves the adversarial search from the waveform into frozen SSL (WavLM-Large) feature space and reconstructs audio with a frozen HiFi-GAN vocoder, transferring to black-box ASRs at +26.6 WER over the SOTA baseline and staying effective against adversarial-training defenses at +36.2 WER over SOTA.

## Problem & motivation
- Waveform-level attacks have two structural weaknesses: (1) poor black-box transferability, because sample-wise perturbations overfit surrogate-specific gradients; (2) they are increasingly neutralized by defenses whose assumptions are "additive waveform noise" (input preprocessing, ℓp-bounded adversarial training).
- Existing robustness evaluations therefore *overestimate* ASR security: they never test perturbations that are not explicit additive waveform noise.
- Prior psychoacoustic work (Carlini & Wagner 2018; Qin et al. 2019 — imperceptible/robust/targeted; Schönherr-style psychoacoustic hiding) is cited as the lineage of imperceptible speech attacks; universal acoustic prefixes (Muting Whisper) and SlothSpeech as the practical-efficiency line.
- Insight borrowed from voice conversion: high-quality speech lives on a manifold jointly shaped by SSL representations and neural vocoders (WavLM, kNN-VC, ACE-VC, HiFi-GAN) — so adversarial search can be done as *content-preserving generation on that manifold*.

## Method (precise but simple language, 5-12 lines; key equations if present)
1. Threat model: white-box surrogate f_s only (target f_t never used); transfer is measured as Err(f_t(x_adv), y) with WER (English) or CER (Chinese).
2. Frozen SSL encoder E (WavLM-Large) gives q = E(x) ∈ R^{T×D}; a learnable feature perturbation δ gives z(δ) = q + δ (Eq. 4–5).
3. Budget in a *normalized feature-space radius*: U_ρ(x) = {δ : ‖δ‖_F / (‖q‖_F + ε) ≤ ρ} (Eq. 6), default ρ = 0.1.
4. Adversarial waveform via frozen vocoder V (HiFi-GAN): x_adv(δ) = V(E(x)+δ); feasible set M_FV(x;ρ) (Eq. 7–8).
5. Attack loss: maximize NLL_text(y | x_adv; f_s) over lexical tokens only (Whisper prompt/language/task/special tokens excluded) (Eq. 9–10).
6. Clean-referenced perceptual regularizer (Eq. 11): L_perc = TV(z)/sg(TV(q)) + α·HF(x_adv)/sg(HF(x)), where TV(z) = (1/(T−1))Σ‖z_t − z_{t−1}‖² penalizes temporal jitter (Eq. 12) and HF(x) = Σ_{f>f_c}|S_f|² / Σ_f |S_f|² penalizes high-frequency energy with f_c = 6000 Hz (Eq. 13); both terms normalized against the *clean* reference (stop-gradient).
7. Final objective: δ* = argmin_{δ∈U_ρ} [L_attack + λ_perc L_perc], λ_perc = 1, 50 optimization steps; gradients flow through surrogate ASR + frozen vocoder, updating only δ (Eq. 14–15).

## Key quantitative results
Setup: surrogate = raw Whisper-small (English, LibriSpeech) or AISHELL-1-fine-tuned Whisper-small (Chinese); 50 steps; ρ = 0.1, λ_perc = 1; attack eval on a held-out 999-utterance subset of test-clean; hyperparameters of *all* attacks selected on a validation split with the constraint DNSMOS/NISQA/UTMOS > 2.5 (Table 2; e.g., PGD ε = 0.02, α = 2ε/50; Muting Whisper prefix 0.64 s, scale 0.02; Sloth ℓ2, lr 10⁻¹, factor 0.1); single NVIDIA L20 (46 GB), ~10 s/utterance, ≈2.8 GPU-hours for test-clean (≈4 GPU-hours total).

Table 1 — WER (%) English / CER (%) Chinese on Whisper-small, columns: Raw | CE-AT | DT-AT | DMW | TI | LPF | WaveGuard | AudioPure | PVP Vote:
- LibriSpeech Clean: 4.75 | 5.72 | 5.70 | 5.86 | 5.81 | 6.45 | 6.76 | 6.17 | 4.80.
- LibriSpeech PGD: 61.06 | 8.69 | 8.13 | 9.61 | 9.80 | 16.28 | 15.47 | 8.38 | 45.62.
- LibriSpeech MI-FGSM: 77.65 | 16.20 | 16.98 | 16.86 | 16.85 | 39.24 | 32.25 | 27.33 | 76.11.
- LibriSpeech VMI-FGSM: 78.84 | 30.29 | 30.91 | 34.48 | 39.14 | 42.04 | 55.36 | 42.38 | 78.34.
- LibriSpeech Muting Whisper: **99.52** | 6.11 | 6.09 | 5.94 | 6.04 | 7.52 | 7.62 | 6.80 | 79.52 (collapses to near-clean under AT defenses).
- LibriSpeech Sloth: 39.50 | 9.38 | 8.71 | 9.26 | 9.38 | 14.88 | 14.98 | 8.70 | 28.47.
- LibriSpeech **Feature Attack (Ours): 75.43 | 71.26 | 67.23 | 70.61 | 70.63 | 68.39 | 57.03 | 70.86 | 78.60**.
- AISHELL-1 Clean: 6.29 | 5.51 | 5.87 | 5.66 | 5.22 | 10.03 | 7.74 | 27.67 | 5.75.
- AISHELL-1 PGD: 69.01 | 15.08 | 14.86 | 15.19 | 17.39 | 26.30 | 12.67 | 26.71 | 64.27.
- AISHELL-1 MI-FGSM: 79.87 | 23.29 | 23.72 | 28.33 | 23.25 | 51.71 | 59.45 | 33.49 | 74.18.
- AISHELL-1 VMI-FGSM: 80.36 | 31.26 | 31.90 | 35.58 | 40.39 | 43.38 | 57.13 | 43.73 | 79.82.
- AISHELL-1 Muting Whisper: 94.47 | 5.93 | 6.05 | 5.87 | 5.41 | 6.74 | 6.89 | 5.96 | 75.24.
- AISHELL-1 Sloth: 53.11 | 14.33 | 11.26 | 8.43 | 9.82 | 36.04 | 31.92 | 20.24 | 54.29.
- AISHELL-1 **Feature Attack (Ours): 72.25 | 66.57 | 66.10 | 66.07 | 65.49 | 72.10 | 70.34 | 69.92 | 75.31**.

Headline claims (abstract/intro): **+26.6 WER** average over the strongest baseline on black-box transfer; **+36.2 WER** average over SOTA on adversarial-training defenses (English); **+31.3 CER** on the Chinese dataset.

Transfer (Figs. 3–4, values stated in text): Whisper-tiny 64.90% WER under DT-AT, 54.75% under WaveGuard; Whisper-large 52.96% WER raw (surrogate is Whisper-small); HuBERT-CTC 39.80% WER raw and >30% WER under all defenses; wav2vec2-CTC 42.12% WER raw, 48.24% under LPF.

Imperceptibility (objective): clean reference 3.07 DNSMOS / 3.71 NISQA / 3.76 UTMOS; default config (λ_perc = 1, ρ = 0.1) gives **75.43% WER at 2.95 DNSMOS, 3.53 NISQA, 3.59 UTMOS**. Table 4 MOS (DNSMOS/NISQA/UTMOS): Clean 3.075/3.714/3.755; PGD 2.761/3.094/3.314; MI-FGSM 2.804/2.523/2.518; VMI-FGSM 2.763/2.517/2.528; Muting Whisper 3.011/3.640/3.729; SlothSpeech 2.895/3.152/3.458; Ours 2.954/3.526/3.590.

Human study (see below): 86% of paired clean/adversarial samples judged indistinguishable; 92% of adversarial samples not flagged as significantly distorted in the no-reference condition; manual human transcription of the adversarial audio = **5.47% WER** (vs 75.43% ASR WER).

Other numbers: representative sample clean-vs-adversarial mel-spectrogram correlation 0.9733; loss ablation — removing the ASR attack loss drops WER to 6.6% (near clean); over-the-air: clean 7.45% WER vs adversarial **78.23% WER** (Table 3); ρ sweep {0.05, 0.1, 0.5, 1.0} and λ_perc sweep {0.1, 0.5, 1, 2} in Figs. 5–6 (values plotted, not tabulated).

## Datasets / corpora used
- LibriSpeech train-clean-100 (~100 h English read speech) for preparation/fine-tuning; attack evaluation on a held-out 999-utterance subset of test-clean.
- AISHELL-1 (~170 h Mandarin) with a held-out AISHELL-1 test set never used for surrogate fine-tuning or defense training.
- Physical experiment: 100 English utterances from three human speakers, played over a loudspeaker and re-recorded by smartphones indoors.

## Models / systems evaluated
- Feature extractor: frozen WavLM-Large (SSL encoder); vocoder: frozen HiFi-GAN.
- Surrogate: raw Whisper-small (English); Whisper-small fine-tuned on AISHELL-1 development set (Chinese).
- Black-box targets: Whisper-tiny/base/medium/large (transfer family), HuBERT-CTC, wav2vec2-CTC.
- Defenses: adversarial training — CE-AT, DT-AT (Decoding Trajectory), DMW (Dynamic Margin Weighting), TI (Translation-Invariant); input preprocessing — LPF, WaveGuard, AudioPure, PVP Vote.
- Attack baselines: PGD, MI-FGSM, VMI-FGSM, Muting Whisper, SlothSpeech.
- Perceptual metrics: DNSMOS, NISQA, UTMOS.

## Human-study details (if any: n participants, protocol, key numbers)
- n = 10 volunteers; each participant evaluates two independent sets of 100 audio samples.
- Set 1 (paired comparison): participants hear paired clean and adversarial clips and judge whether they perceive any difference → **86% of paired samples judged indistinguishable**.
- Set 2 (single/unpaired): participants hear adversarial clips alone and flag noticeable distortion/suspicious artifacts → **92% of adversarial samples not perceived as having significant distortion**.
- Additional task: participants manually transcribe the adversarial audio → **human transcription WER 5.47%**, versus 75.43% ASR WER for the same material.
- Separately, the over-the-air experiment used three human speakers (100 utterances) as *stimulus material*, not as listeners.

## Limitations acknowledged by authors
- Evaluation covers a finite set of ASR systems and defenses; generalization to commercial ASR services, streaming ASR and larger multilingual foundation models is untested.
- The pipeline depends on one specific SSL encoder (WavLM-Large) and one vocoder (HiFi-GAN); other choices may change attack strength and perceptual quality.
- Imperceptibility "remains difficult to fully characterize with automatic metrics alone"; larger human studies in more diverse listening environments are needed (the current study has only 10 volunteers).
- The physical-world experiment is preliminary: limited speakers, devices and acoustic conditions; more extensive OTA evaluation (rooms, reverberation, background noise, device pipelines) is left to future work.
- Exact norm matching across attack spaces (additive waveform noise vs universal prefixes vs feature-vocoder) is "not meaningful", so baseline comparison relies on a validation-based selection protocol instead of matched budgets.

## Relevance to our project (one specific paragraph)
This is the single most directly usable paper for AudioCAPTCHA-DSP, because it *is* the human-vs-ASR gap quantified: the same adversarial audio scores 75.43% WER for Whisper while a 10-person listening study gives only 5.47% human WER, 86% pairwise indistinguishability and 92% no-perceived-distortion — i.e., a Human-ASR Gap of roughly 70 points, with objective perceptual scores (DNSMOS 2.95 vs 3.07 clean) confirming near-clean quality. It gives us a validation template for our own HAG metric: report ASR WER, human (or STOI-proxy) intelligibility, and a perceptual-quality triple (DNSMOS/NISQA/UTMOS) at the *same* distortion severity, plus a paired-difference and a no-reference listening protocol with n participants. It also expands our distortion vocabulary: instead of (or alongside) masking-based waveform shaping, a CAPTCHA can perturb the SSL/vocoder manifold (their normalized feature budget ρ and clean-referenced temporal-jitter + high-frequency regularizers are a ready-made recipe for keeping speech natural-sounding to humans while breaking machine recognizers), and it shows such distortions survive both adversarial training and preprocessing defenses — meaning a CAPTCHA built on feature-space distortion should be robust to ASRs that deploy WaveGuard/AudioPure-style defenses. Finally, their over-the-air result (7.45% clean → 78.23% adversarial WER after loudspeaker playback) tells us our distortion chain must be evaluated through a physical playback channel, exactly as a real audio CAPTCHA would be.

## Keywords
adversarial attack, black-box transferability, SSL representations, WavLM, neural vocoder, feature-space perturbation, automatic speech recognition, Whisper, adversarial training, input preprocessing defense, imperceptibility, DNSMOS, NISQA, UTMOS, human study
