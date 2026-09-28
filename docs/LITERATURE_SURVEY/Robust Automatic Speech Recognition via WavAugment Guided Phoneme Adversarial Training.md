# Robust Automatic Speech Recognition via WavAugment Guided Phoneme Adversarial Training

## Full reference
Gege Qi, Yuefeng Chen, Xiaofeng Mao, Xiaojun Jia, Ranjie Duan, Rong Zhang, Hui Xue. *Robust Automatic Speech Recognition via WavAugment Guided Phoneme Adversarial Training.* arXiv:2307.12498v1 [cs.SD], 24 Jul 2023. Affiliations: 1Alibaba Group, China; 2Chinese Academy of Sciences, China. No venue printed on the PDF (two-column IEEE/Interspeech-style layout with "Index Terms"). No DOI printed.

## One-line contribution
WAPAT performs adversarial training in the *phoneme-unit space* of a SpeechLM recognizer, with the adversary's gradients guided by WavAugment time-domain augmentations (via a KL-divergence term), reaching the best cross-domain WER on the ESB benchmark (state-of-the-art ESB score 32.58).

## Problem & motivation
- "Practically robust" ASR must keep clean accuracy while surviving (a) small volume perturbations (noise, reverb, background speakers) and (b) large domain/speaking-style shifts; robustness to one perturbation type does not imply robustness to unseen domains.
- Existing data augmentation is hand-designed per domain and does not generalize; speech-enhancement front ends add compute and "do not really improve the robustness of ASR itself".
- Adversarial training (AT) is known to trade off clean accuracy vs robustness, but work in NLP/vision shows that aligning adversarial and clean distributions during AT can help both; nobody had done AT on the phoneme-unit space for ASR.
- Adversarial examples are unstable (near the decision boundary), so plain AT learns brittle features; the authors want adversaries that are both *stable* and *diverse*.

## Method (precise but simple language, 5-12 lines; key equations if present)
1. Baseline AT (Eq. 1): min_θ E[max_{‖δ‖_p≤ε} L_ctc(x+δ, y, θ)], inner maximization by PGD — but here applied to a speech tokenizer output instead of the waveform.
2. Phoneme Adversarial Training (PAT, Eq. 2): min_θ E[max_{‖δ‖∞≤ε} L_ctc(T(x)+δ, y, θ)], where T is a frozen transformer-based phoneme-unit tokenizer of the SpeechLM framework; the perturbation lives in phoneme space, so adversarial speech keeps realistic semantics.
3. WAPAT guidance (Eq. 3): add L_wag(z+η, z_a+η_a, θ) = −D_KL[ p(z+η,θ) ‖ p(z_a+η_a,θ) ], where z = T(x), z_a = T(D_A(x)) are phoneme representations of the clean and augmented audio, and η, η_a are their gradients. Maximizing this aligns the two adversaries' predictions → more stable adversaries.
4. Diversity: one WavAugment transformation is sampled per batch (pitch, additive noise, band-reject, time mask, reverb), so gradient directions vary across batches.
5. Algorithm 1: z ← T(x); z ← U(B^∞_ε(z)); z_a ← T(D_A(x)); δ ← ∇_z[L_ctc + L_wag]; ẑ ← proj_{B^∞_ε}(z+δ); update θ on L_ctc(ẑ, y, θ). Tokenizer T stays fixed (straight-through gradient).

## Key quantitative results
Setup: SpeechLM-P (Speech Transformer + Shared Transformer + CTC head) pre-trained on LibriSpeech-960h audio + LibriSpeechLM text, fine-tuned on LibriSpeech-100h; 30K steps, batch 800 s, Adam, max LR 1e-5 (tri-stage schedule [0.1, 0.4, 0.5]); ℓ∞ perturbation ε = 0.01; 4× NVIDIA Tesla A100; 16-bit WAV @16 kHz. Metric = WER (%); "ESB score" = macro-average over the ESB datasets excluding LibriSpeech (as defined in the paper).

Table 1 — WER per dataset and ESB score (best in bold in the original):
| Method | LS test-clean | LS test-other | CHiME-4 | Common Voice | VoxPopuli | TED-LIUM | GigaSpeech | SPGISpeech | Earnings-22 | AMI | ESB score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SpecAugment | 3.32 | 7.34 | 45.49 | 38.46 | 36.47 | 19.03 | 24.57 | 20.10 | 51.10 | 45.02 | 36.19 |
| WavAugment | 3.34 | 7.35 | 35.65 | 38.16 | 36.64 | 18.12 | 25.53 | 18.99 | 52.79 | 46.10 | 34.18 |
| AdvEx (waveform AT) | 3.36 | 7.36 | 46.10 | 38.35 | 36.74 | 18.18 | 24.49 | 19.36 | 52.01 | 44.79 | 36.24 |
| DEMUCS (enhancement front end) | 3.33 | 7.29 | 33.57 | 43.63 | 36.71 | 18.31 | 24.39 | 26.24 | 56.76 | 44.63 | 35.32 |
| **WAPAT** | **3.32** | **7.28** | **32.68** | **36.43** | **36.38** | **18.12** | **24.25** | **18.40** | **49.78** | **44.53** | **32.58** |

Headline claims as printed: abstract — "SpeechLM-WAPAT outperforms the original model by 6.28% WER reduction on ESB, achieving the new state-of-the-art"; body — vs AdvEx, "WAPAT achieves 10.01% improvement in ESB score"; WAPAT beats WavAugment (34.18) and SpecAugment (36.19) by a large margin while matching or beating clean WER on LibriSpeech.

Table 2 ablation (LibriSpeech test-clean / test-other / ESB score):
- (a) NO-AT 3.34 / 7.38 / 36.47; w/ PHONEME AT (plain PAT) 3.32 / 7.34 / 35.18; w/ WAVAUGMENT PAT (= full WAPAT) 3.32 / 7.28 / 32.58. The ESB score drops ≈7.4% from PAT to WAPAT.
- (b) ε = 0.005: 3.32 / 7.35 / 34.42; ε = 0.01: 3.32 / 7.28 / 32.58 (best); ε = 0.015: 3.32 / 7.31 / 33.24 (too-large ε hurts generalization while clean WER barely moves).
- Figure 2 (WER-reduction per augmentation vs baseline): individual WavAugment transforms oscillate (time mask even *increases* WER on TED-LIUM), while WAPAT improves consistently across all transforms; exact bar values are not printed as numbers in the text.

## Datasets / corpora used
- ESB benchmark (8 datasets): LibriSpeech, Common Voice, VoxPopuli, TED-LIUM, GigaSpeech, SPGISpeech, Earnings-22, AMI — covering narrated, oratory and spontaneous speaking styles.
- Optional CHiME-4 (narrated) added for generalization testing.
- Fine-tuning corpus: LibriSpeech-100h; pre-training corpora: LibriSpeech-960h audio and LibriSpeechLM text.
- Augmentation sources: MUSAN (additive noise), gpuRIR-simulated room impulse responses (reverb).
- Standard splits; transcripts unified to the "normalized" format.

## Models / systems evaluated
- SpeechLM-P (Speech Transformer + Shared Transformer + CTC head, from Microsoft SpeechT5/SpeechLM), referred to as "SpeechLM" / "SpeechLM-P-Base".
- Baselines/competitors: SpecAugment, WavAugment, AdvEx (waveform-space adversarial training), DEMUCS (real-time waveform speech enhancement front end).
- Not evaluated: Whisper, Vosk, or any non-CTC/decoder-based ASR.

## Human-study details (if any: n participants, protocol, key numbers)
None — no human listeners, no intelligibility or perceptual test of any kind. Robustness is measured exclusively by machine WER on noisy/cross-domain corpora.

## Limitations acknowledged by authors
- "WAPAT still costs increased training time, this limitation also holds for any adversarial training" — left as a future optimization direction (the paper's only explicit limitation).
- Hard augmentations can improve robustness on some datasets and fail on others (generalizing audio augmentation across domains is inherently hard).
- Sensitivity to ε: too large a perturbation budget (0.015) damages generalization even though clean WER is insensitive to ε.
- Evaluation confined to one model family (SpeechLM) and one benchmark (ESB).

## Relevance to our project (one specific paragraph)
WAPAT is our reference point for the *defense* side of the Human-ASR Gap: it is the strongest published recipe (ESB score 32.58) for making an ASR robust to exactly the class of distortions we plan to apply — additive noise, band-reject filtering, time masking, pitch shifts and reverb (the same WavAugment menu our DSP/psychoacoustic chain will draw from). Two implications for AudioCAPTCHA-DSP: first, if a CAPTCHA's anti-ASR distortion is chosen from "ordinary" corruptions, a WAPAT-style trained recognizer may absorb it, so our distortion budget should be pushed into psychoacoustic/perceptual dimensions (masking-based shaping, spectral fine structure) rather than plain noise — where human intelligibility stays high (STOI ≈ high) but ASR training distributions do not cover the artifact; second, their ablation structure (clean vs single-transform vs AT variants, plus an ε sweep) is a good template for reporting *our* HAG curves: for each distortion severity we should report both the human-side proxy and the ASR-side WER, and note that WAPAT-type training would deliberately shrink that gap. The paper also reminds us that ESB-style cross-domain evaluation matters — a CAPTCHA recognizer and a general ASR see very different audio, so we must fix which "machine side" of the gap we measure.

## Keywords
adversarial training, robust ASR, cross-domain generalization, phoneme space, data augmentation, WavAugment, SpeechLM, ESB benchmark, word error rate, KL divergence, perturbation budget, speech corruption robustness, CTC, domain shift, state-of-the-art
