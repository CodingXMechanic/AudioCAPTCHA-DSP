# MORE: Multi-Objective Adversarial Attacks on Speech Recognition

## Full reference
Xiaoxue Gao, Zexin Li, Yiming Chen, Nancy F. Chen. *MORE: Multi-Objective Adversarial Attacks on Speech Recognition.* Manuscript dated January 15, 2026; arXiv:2601.01852v2 [eess.AS], 14 Jan 2026. Affiliations: 1A*STAR (Agency for Science, Technology and Research, Singapore), 2University of California, Riverside, USA, 3National University of Singapore, Singapore. No venue printed on the PDF (the Ethics Statement references the ICLR Code of Ethics, indicating an ICLR-style submission). No DOI printed.

## One-line contribution
First unified white-box attack (MORE) that jointly degrades ASR *accuracy* (WER) and *inference efficiency* (output length/FLOPs) through a hierarchical two-stage "repulsion–anchoring" optimization plus a Repetitive Encouragement Doubling Objective (REDO).

## Problem & motivation
- Prior adversarial work on ASR (Whisper included) measures only accuracy degradation; robustness of *inference efficiency* is "largely unexplored".
- The only prior efficiency attack, SlothSpeech (Interspeech 2023), does not consider accuracy and does not systematically structure the output pattern.
- Accuracy gradients are broad (spread over many token positions) while efficiency gradients are narrow and concentrated on a single EOS token, so naively combining them in one step makes one objective dominate → unstable multi-objective optimization.
- Motivating use cases given: distorting transcription of harmful/private speech, and denial-of-service-style degradation of real-time ASR.

## Method (precise but simple language, 5-12 lines; key equations if present)
1. Threat model: white-box, full model access; find perturbation δ with SNR and ℓ∞ constraints: SNR_dB = 20·log10(‖X‖₂/‖δ‖₂), Δ = {δ : ‖δ‖∞ ≤ ε} with ε = ‖X‖∞/SNR (Eq. 2–3). The ℓ∞ cap is said to align with psychoacoustic masking principles.
2. Dual objective (Eq. 1): maximize (WER(f(X+δ), Y), |f(X+δ)|) — i.e., wrong *and* long output.
3. Hierarchical two-stage reformulation (Eq. 4): Stage 1 "repulsion" first maximizes a differentiable WER proxy, Stage 2 "anchoring" maximizes sequence length while holding the achieved error.
4. Repulsion loss: L_acc = −CE(f(X+δ), Y) (Eq. 5), optimized with signed-gradient (PGD-style) steps clipped to [−ε, ε].
5. Anchoring loss L_eff = L_REDO + L_EOS (Eq. 6–9):
   - EOS suppression L_EOS = P^EOS_L − P^z_L, where z is the second-most-likely token at the final position (Eq. 6–7).
   - REDO: every D steps the target sequence is replaced by a doubled copy of itself, Ȳ_i = ŷ_⌊i/D⌋[1:L−1] ∥ ŷ_⌊i/D⌋[1:L−1] (Eq. 8), and L_REDO = CE(f(X+δ), Ȳ_i) (Eq. 9).
6. "Asymmetric interleaving": the doubled target is held fixed for D steps and only extended when s mod D = 0 (curriculum-style); Algorithm 1 gives the full loop (K_a accuracy steps, then K−K_a efficiency steps).
7. Appendices give attack-time and victim-time complexity analysis (decoder self-attention scales quadratically in output length ⇒ Θ(4^{M*}) FLOPs growth before a length cap).

## Key quantitative results
Setup for all of the following unless stated: first 500 utterances of LibriSpeech test-clean and of LJ-Speech, 16 kHz; white-box attacks on Whisper-tiny/base/small/medium/large (HuggingFace); SNR 35 dB (ε = 0.002) or 30 dB (ε = 0.0035); WER truncated to reference length and capped at 100%; "length" = average predicted token count. Hardware: 1× NVIDIA H100. Hyperparameters: doubling period D (called "I") = 10, accuracy steps K_a = 50.

Table 1 (SNR 35 dB), selected rows — (WER, length):
- LibriSpeech / clean: 6.66/21.84 (tiny) → 3.01/21.84 (large).
- LibriSpeech / Whisper-tiny: PGD 93.17/35.02; SlothSpeech 46.80/119.38; SAGO 93.19/31.93; VMI-FGSM 87.91/32.39; MI-FGSM 93.32/34.38; **MORE 91.01/296.28**.
- LibriSpeech / Whisper-base: SlothSpeech 54.63/156.07; PGD 88.73/31.65; **MORE 88.42/300.13**.
- LibriSpeech / Whisper-small: SlothSpeech 38.25/110.40; **MORE 74.28/213.94**.
- LibriSpeech / Whisper-medium: SlothSpeech 31.09/81.75; **MORE 64.04/234.25**.
- LibriSpeech / Whisper-large: PGD 33.33/21.80; SlothSpeech 34.21/79.78; SAGO 30.26/21.12; **MORE 53.72/301.47** (≈10× longer than accuracy-only baselines, ≈3.8× SlothSpeech).
- LJ-Speech / clean: 5.34/18.55 (tiny) → 3.36/18.55 (large).
- LJ-Speech / Whisper-tiny: **MORE 90.85/296.66**; SlothSpeech 47.10/116.05.
- LJ-Speech / Whisper-small: SlothSpeech 32.52/65.85; **MORE 74.33/208.51**.
- LJ-Speech / Whisper-large: **MORE 43.13/231.52**; SlothSpeech 21.14/35.33.

Table 2 (SNR 30 dB), selected rows — (WER, length):
- LibriSpeech / Whisper-tiny: **MORE 94.73/300.79**; PGD 96.22/35.30; SlothSpeech 60.06/123.93.
- LibriSpeech / Whisper-large: **MORE 60.90/277.65**; SlothSpeech 48.34/78.65; PGD 47.93/21.92.
- LJ-Speech / Whisper-tiny: **MORE 94.80/326.62**; SlothSpeech 61.62/123.58.
- LJ-Speech / Whisper-large: **MORE 54.13/229.08**; SlothSpeech 35.74/37.50.
- General finding: 30 dB (noisier) gives higher WER and longer transcripts than 35 dB for every attack.

Table 3 ablation (LJ-Speech, WER/length; full = 90.85/296.66 at 35 dB, 94.80/326.62 at 30 dB):
- −L_acc: 27.58/293.03 (35 dB), 34.33/296.91 (30 dB) → accuracy collapses, efficiency survives.
- −L_eff (all efficiency losses): 93.63/30.60, 96.56/31.79 → efficiency collapses completely.
- −L_EOS: 93.92/233.84, 96.68/269.36; −L_REDO: 92.42/120.67, 95.72/146.54 (REDO is the main efficiency driver).
- −L_REDO −L_acc: 47.10/116.05, 61.62/123.58; −L_EOS −L_acc: 6.97/270.21, 7.96/307.38.

Table 4 case study (LibriSpeech): clean transcript length 12 / WER 0.00; PGD 18/100.00; SlothSpeech 42/72.73; SAGO 17/90.91; VMI-FGSM 14/100.00; MI-FGSM 18/100.00; **MORE 334/100.00** with "and her voice" repeated >100 times. Appendix samples reach lengths 303, 369, 363, 387 (all WER 100.00).

Table 7 inference FLOPs (per example, LibriSpeech; baseline ≈22 tokens): Tiny 1.7 G → 23.1 G (13.5×); Base 3.3 → 44.4 G (13.6×); Small 10.7 → 104.4 G (9.7×); Medium 33.8 → 359.9 G (10.6×); Large 68.2 → 933.1 G (13.7×). Text summarizes "≈9–14×" increase; Fig. 2 shows Whisper-large inference time rising roughly linearly with output tokens (up to ≈8 s near 400 tokens).

## Datasets / corpora used
- LibriSpeech (test-clean subset), first 500 utterances.
- LJ-Speech, first 500 utterances.
- Both resampled to 16,000 Hz; sourced from HuggingFace.

## Models / systems evaluated
- Victims: Whisper-tiny, Whisper-base, Whisper-small, Whisper-medium, Whisper-large (HuggingFace), white-box, greedy-style decoding.
- Attacks compared: PGD, MI-FGSM, VMI-FGSM, SAGO (speech-aware gradient optimization), SlothSpeech, plus the proposed MORE.
- Related work cited but not evaluated: DolphinAttack, hidden voice commands, MFCC-domain attacks, universal attacks (Muting Whisper), C&W attacks.

## Human-study details (if any: n participants, protocol, key numbers)
None. Ethics Statement: "This work does not involve human-subject studies, user experiments, or the collection of new personally identifiable information." Imperceptibility is argued only via SNR values (35 dB / 30 dB, described as "within the range generally considered inaudible to humans [13]") and the ℓ∞ cap.

## Limitations acknowledged by authors
- Code is not included in the submission ("proprietary requirements"); a public repository is promised only upon acceptance.
- Defenses against MORE are left to future work; suggested mitigations are decoding-time repetition/loop detectors, input band-limiting, and adversarial training focused on EOS/repetition pathologies.
- Ethics Statement explicitly lists misuse risks: impairing captioning/accessibility, disrupting safety-critical applications (clinical dictation, emergency transcription, navigation), and inflating compute cost of shared services.
- The FLOPs analysis is an *estimate* (2·N params per token), not a measured profiling, and inference-time scaling is capped by the implementation's maximum decoding length.
- Only accuracy (WER) and length are optimized/reported; no perceptual or human-intelligibility metric is measured.

## Relevance to our project (one specific paragraph)
MORE matters to AudioCAPTCHA-DSP mainly as an *ASR-side stress test and as a caution about the Human-ASR Gap*: it shows that a single imperceptible-looking perturbation (SNR 35/30 dB — the same ballpark we use for "inaudible" psychoacoustic distortion) can drive Whisper WER from ~3–7% clean to >90% while also making the machine output pathologically long (≈300 tokens, ≈9–14× FLOPs). For CAPTCHA design this is a third axis beyond intelligibility loss and mis-transcription: an attacker-distorted prompt could be made to *hang or bloat* an ASR pipeline, whereas a human listener would still hear ordinary speech — widening the HAG in a direction nobody measures with STOI. It also gives us useful methodological lessons: (i) report both a distortion budget in dB SNR *and* an ℓ∞/peak cap, (ii) always include an efficiency/output-length column when evaluating ASR on distorted audio, and (iii) their ablation-style decomposition (remove one loss at a time) is a template for isolating which DSP stage of our distortion chain causes which part of the human-vs-ASR divergence. Unlike our psychoacoustic pipeline, MORE never checks human intelligibility — a gap our framework is designed to fill.

## Keywords
adversarial attacks, automatic speech recognition, Whisper, multi-objective attack, denial of service, efficiency robustness, word error rate, repetition loop, EOS suppression, psychoacoustic masking, imperceptibility, SNR budget, REDO, transcript length, FLOPs
