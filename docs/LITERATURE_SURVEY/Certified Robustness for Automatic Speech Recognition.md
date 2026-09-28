# What Was That Again? Certified Robustness for Automatic Speech Recognition

## Full reference
Andrew C. Cullen, Neil G. Marchant, Jiani Xie, Paul Montague, Benjamin I. P. Rubinstein. *What Was That Again? Certified Robustness for Automatic Speech Recognition.* Preprint; arXiv:2606.27698v2 [cs.LG], 30 Jun 2026. Affiliations: University of Melbourne (Cullen, Marchant, Xie, Rubinstein); DST Group, Adelaide (Montague). Acknowledgments: supported by the Australian Defence Science and Technology (DST) Group via the ASCA program. No venue printed; no DOI printed.

## One-line contribution
A dual-gate "Two-Sided Atomic Audit + Rank-Based Tournament" built on E-values/Ville's inequality that gives anytime-valid, sentence-level certified transcriptions for ASR — cutting WER by up to ~55% relative while keeping certification recall 40.5–90.3% where ROVER/Cohen baselines collapse to <1% in high noise.

## Problem & motivation
- ASR systems are sensitive to adversarial *and* benign perturbations, but deployed systems cannot be audited with WER because there is no ground-truth transcription at runtime.
- Randomized smoothing (RS) for sequences fails because, under strong noise, the probability of any single sentence collapses — "majority class" selection over a combinatorial output space does not work (Olivier 2023).
- Sequence certification via multiple sequence alignment (ROVER/confusion networks) is fragile at low SNR: alignment mismatches create new slots, the Bonferroni budget α/L shrinks, and certifications become vacuous ("18 ground-truth words expanded to over thousands of slots").
- Clopper–Pearson RS needs tens of thousands of samples and suffers a *peeking* problem (no optional stopping); E-values + Ville's inequality are anytime-valid and allow optimal stopping with zero penalty.
- Motivation from psychoacoustics: small waveform changes ≠ small perceived-sound changes, and ℓp distances correlate poorly with human judgment of audibility (citing Schönherr et al. 2019, Qin et al. 2019).

## Method (precise but simple language, 5-12 lines; key equations if present)
1. Smoothed predictor F(x) = G(f(x+ε)), ε ~ N(0, σ²I), with error budget α = α_atomic + α_tourn; goal: P[∃δ, ‖δ‖₂<R, F(x+δ)≠Ŷ] ≤ α (Eq. 7).
2. Stage S1 Discovery: draw N1 noisy samples, take the union of observed tokens as candidate vocabulary V (samples then discarded for independence).
3. Stage S2 Atomic audit: for each w ∈ V maintain two log-space wealth martingales, ln E_pos,T = Σ ln(1+λ(W_{w,t}−0.5)) and ln E_neg,T = Σ ln(1+λ(0.5−W_{w,t})) (Eq. 8), testing H_pos: p_w ≤ 0.5 and H_neg: p_w ≥ 0.5. Ville: P(∃T: E_T ≥ 1/α) ≤ α.
4. Confidence sequences (Eq. 9) invert the likelihood-ratio martingale into anytime-valid probability bounds; these map to radii r_w = σΦ⁻¹(p̲_w,T) (certified) or σΦ⁻¹(1−p̄_w,T) (excluded) (Eq. 10); R_atomic = min_w r_w.
5. Stage S3: filter N3 samples down to V_cert tokens, keep the top-K most frequent unique cleaned sequences as candidate pool C.
6. Stage S4 Tournament: K parallel competitive E-values E_{i,t} ← E_{i,t−1}(1+λ(I[i = argmin_j WER(C_j, Y_t)] − 1/K)) (Eq. 13); stop when average wealth Ē ≥ 1/α_tourn (Lemma 1, "structural multiplicity subsidy", O(1) multiplicity scaling).
7. Structural radius R_tour = (σ/2)(Φ⁻¹(p_winner) − Φ⁻¹(p_runner-up)) (Eq. 14); final R = min(R_atomic, R_tourn) (Theorem 1). No sequence alignment/ROVER is used at all.

## Key quantitative results
Setup: AWGN at SNR ∈ {10, 5, 0, −5} dB, σ² scaled to target SNR, 16 kHz mono; zero-shot (inference-only) evaluation; 100 random sentences per permutation → 3,200 certification trials; H100 80 GB (16-bit for Whisper-Large-v3); total 8 GPU-days. Hyperparameters (Table 3): discovery N_S1 = 50, certification N_S2 = 1000, candidates K = 5, tournament max budget N_S4 = 250, α = 0.01, λ_inst = 0.50, λ_tourney = 0.20, threshold τ = 0.5. Recall = "99% confidence certification success"; ρ = Spearman(method confidence, WER). Models: H = HuBERT(-Large), W.L = Whisper-Large(-v3), W2 = wav2vec2-Large, W.S = Whisper-Small.

Table 1 (WER: Raw / Cohen / ROVER / Ours — Recall %: Cohen / ROVER / Ours — ρ: Cohen / ROVER / Ours):
| Model | SNR | Raw | Cohen | ROVER | Ours | Rec.C | Rec.R | Rec.O | ρC | ρR | ρO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H | 10.0 | 0.161 | 0.162 | 17.927 | 0.145 | 51.3 | 48.0 | 60.1 | −0.747 | −0.845 | −0.774 |
| H | 5.0 | 0.231 | 0.217 | 41.904 | 0.205 | 37.2 | 21.0 | 58.7 | −0.756 | −0.626 | −0.731 |
| H | 0.0 | 0.476 | 0.436 | 258.566 | 0.417 | 7.7 | 2.3 | 80.9 | −0.700 | −0.258 | −0.270 |
| H | −5.0 | 0.940 | 0.925 | 249.886 | 0.895 | 0.7 | 0.0 | 40.5 | −0.298 | – | −0.198 |
| W.L | 10.0 | 0.085 | 0.087 | 0.970 | 0.087 | 11.0 | 44.3 | 38.2 | −0.392 | −0.844 | −0.811 |
| W.L | 5.0 | 0.072 | 0.052 | 1.632 | 0.051 | 8.2 | 32.5 | 90.3 | −0.348 | −0.739 | −0.574 |
| W.L | 0.0 | 0.081 | 0.064 | 4.317 | 0.064 | 3.5 | 15.8 | 84.0 | −0.313 | −0.524 | −0.315 |
| W.L | −5.0 | 0.273 | 0.125 | 52.577 | 0.126 | 0.5 | 2.1 | 74.0 | −0.154 | −0.242 | −0.123 |
| W2 | 10.0 | 0.265 | 0.272 | 51.548 | 0.229 | 43.3 | 29.0 | 72.7 | −0.710 | −0.708 | −0.697 |
| W2 | 5.0 | 0.378 | 0.348 | 155.165 | 0.338 | 20.7 | 10.3 | 71.3 | −0.752 | −0.509 | −0.476 |
| W2 | 0.0 | 0.749 | 0.710 | 296.782 | 0.708 | 2.0 | 0.3 | 70.3 | −0.617 | −0.099 | 0.393 |
| W2 | −5.0 | 0.948 | 0.929 | 110.595 | 0.930 | 0.0 | 0.0 | 45.2 | −0.221 | – | −0.008 |
| W.S | 10.0 | 0.071 | 0.058 | 1.532 | 0.058 | 0.0 | 37.5 | 56.7 | 0.009 | −0.711 | −0.600 |
| W.S | 5.0 | 0.088 | 0.061 | 3.917 | 0.061 | 0.0 | 26.2 | 87.3 | −0.096 | −0.613 | −0.608 |
| W.S | 0.0 | 0.213 | 0.110 | 28.143 | 0.127 | 0.0 | 6.7 | 76.3 | – | −0.372 | −0.456 |
| W.S | −5.0 | 0.595 | 0.669 | 258.181 | 0.443 | 0.0 | 0.0 | 44.2 | – | – | −0.320 |

Other headline numbers as printed:
- Abstract: "up to a 55% relative reduction in Word Error Rate"; intro: certification recall "40.5%–90.3%" vs baselines that collapse to "<1%".
- §4: Olivier/ROVER recall collapses to 0.0–2.1% in extreme noise while the method holds 40.5–74.0%; Whisper-Large-v3 raw 0.273 → 0.126 WER (stated as "54% relative improvement"); "our tournament approach yields a 55.1% reduction in the WER".
- Appendix C: absolute WER reduction "up to 10.6% at SNR −5 dB on LibriSpeech, and 8.1% on Common Voice".
- Note (internal inconsistency in the paper): §4 prose quotes "ROVER's −0.825 for Whisper-Large at 10 dB" and "Recall (73%) vs ROVER (59%) … ρ = −0.310 at 10 dB", which do not match the corresponding Table 1 row (ρ_ROVER = −0.844, Recall_Ours = 38.2, Recall_ROVER = 44.3, ρ_Ours = −0.811).

Table 2 — linguistic fragility (Raw accuracy → Certified accuracy, with accepted/rejected/ambiguous proportions): CCONJ 0.97→0.994 (0.835/0.141/0.0236); DET 0.96→0.986 (0.444/0.5/0.0556); SCONJ 0.919→0.986; ADP 0.932→0.982; ADV 0.903→0.971; PRON 0.931→0.967; VERB 0.832→0.96 (0.554/0.412/0.0341); ADJ 0.839→0.947; NOUN 0.765→0.922 (0.455/0.508/0.0365, stated as "92.2% accuracy for nouns, a relative improvement of 20.5%"); INTJ 0.889→0.903; AUX 0.841→0.899; PART 0.784→0.795; PROPN 0.539→0.662 (0.296/0.67/0.0338); NUM 0.561→0.639 (0.603/0.347/0.0494).
Appendix B certification *recall* by POS: PRON 21.7%, CCONJ 16.7%, ADP 13.1% vs NOUN 2.8%, VERB 3.8%, PROPN 1.2%.

Table 5 — Real-Time Factor (Naive Cohen / ROVER / Ours-EE): HuBERT-large 0.05x/0.07x/0.06x; wav2vec2-large 0.05x/0.06x/0.05x; Whisper-large-v3 2.20x/2.12x/2.37x; Whisper-small 0.90x/0.83x/0.96x. Text: CTC architectures 17–20% faster than ROVER; anytime stopping gives a 28% compute saving vs fixed budget for Whisper-Large.
Figure 1: certified radius correlates with empirical mean WER on both LibriSpeech and Common Voice (left/right panels; axis ranges 0.06–0.20 radius, 0.0–1.0 WER).

## Datasets / corpora used
- LibriSpeech test-clean and test-other (CC-BY 4.0).
- Common Voice 17.0, English test split (MPL 2.0).
- Noise: synthetic Additive White Gaussian Noise at four SNR levels; audio standardized to single-channel 16 kHz.

## Models / systems evaluated
- Whisper Large-v3 and Whisper Small (transformer encoder–decoder).
- HuBERT-Large fine-tuned for CTC ASR (LS960-ft).
- wav2vec 2.0 Large (960h) CTC.
- Baselines: naive Cohen randomized smoothing; ROVER / Olivier & Raj (2021) sequential randomized smoothing.
- Post-processing tooling: spaCy en_core_web_sm for POS tagging.

## Human-study details (if any: n participants, protocol, key numbers)
None. No human listeners or intelligibility tests; all evaluation is machine-side (WER, certification recall, correlation, RTF). Human perception is only invoked in the Related Work discussion (ℓp distances vs perceived audibility).

## Limitations acknowledged by authors
- Atomic gate uses a fixed 1/α_atomic threshold per token → only a *local* guarantee; Family-Wise Error Rate is not controlled (true confidence scales as |V|·α_atomic), so an adversary might flip a word at a smaller radius than reported; certifications should be read as "a diagnostic marker, rather than a measure of true robustness".
- e-BH is discussed but excluded from the implementation (data-dependent threshold is hard to map to stable radii; sort-and-sum causes a GPU synchronization bottleneck).
- Discovery-phase limitation: any word absent from the first N1 samples can never be certified later.
- Threat model restricted to ℓ2 perturbations, acknowledged in the Impact Statement as "a restriction relative to the overall threat landscape".
- Impact Statement also notes a privacy upside to non-robustness and warns that the community's chosen threat models bias the perceived risk landscape.

## Relevance to our project (one specific paragraph)
This paper supplies AudioCAPTCHA-DSP with a *deployment-time evaluation methodology*: WER needs a ground truth we will not have when a CAPTCHA is served, whereas a certified radius/confidence score computed only from the model's own noisy decodes correlates with WER (Fig. 1) and could serve as a machine-side trust score alongside our offline human-intelligibility proxy. That matters for the Human-ASR Gap because it lets us ask a new question — not only "does our psychoacoustic distortion raise human-vs-ASR disagreement?" but "can the ASR *detect* that its own output is unstable under perturbation, and does that self-reported radius shrink as we increase masking-based distortion?" Their finding that robustness is unevenly distributed by word class (nouns 2.8% certification recall vs pronouns 21.7%) is directly relevant to CAPTCHA token design: content words — exactly what a CAPTCHA asks a human to type — are the most fragile tokens for the machine, so digit/letter/word choices in audio CAPTCHAs should be evaluated per word class rather than with a single aggregate WER. Finally, the paper's explicit caution that ℓp/SNR budgets correlate poorly with *audibility* (citing Schönherr et al.) supports our core design decision to constrain distortions with psychoacoustic/STOI-style perceptual criteria instead of raw norm budgets.

## Keywords
certified robustness, randomized smoothing, E-values, Ville's inequality, anytime-valid inference, sequence certification, automatic speech recognition, Whisper, word error rate, robustness certificate, adversarial perturbation, confidence sequence, part-of-speech fragility, trust score, SNR
