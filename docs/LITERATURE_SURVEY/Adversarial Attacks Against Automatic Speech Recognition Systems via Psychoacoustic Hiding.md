# Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding

## Full reference
- **Title:** Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding
- **Authors (as printed):** Lea Schönherr, Katharina Kohls, Steffen Zeiler, Thorsten Holz, and Dorothea Kolossa
- **Affiliation:** Horst Görtz Institute for IT Security, Ruhr-Universität Bochum, Germany
- **Venue / year as printed on the PDF:** arXiv preprint — `arXiv:1808.05665v2 [cs.CR] 30 Oct 2018`. No conference/journal venue, DOI, or page numbers are printed on the document (arXiv version).
- Demo site printed in the paper: http://adversarial-asr.selfip.org

## One-line contribution
A *targeted* adversarial attack on DNN-HMM ASR (Kaldi) that hides an attacker-chosen transcription inside an arbitrary audio file by combining backpropagation into the raw waveform with an MP3-style psychoacoustic hearing-threshold model and forced alignment, so the injected command stays below human audibility while the machine still transcribes it.

## Problem & motivation
- DNN-based ASR now performs "on par with human listening" and is deployed in smartphones, Amazon Echo/Alexa, smart-home setups; the huge parameter count of DNNs gives an adversary "much space to explore blind spots."
- Prior audio attacks were all flawed in different ways: Vaidya et al. (Cocaine Noodles) — high, easily perceptible distortion; Carlini et al. (hidden voice commands) — HMM-only ASR, output sounds like noise and makes listeners suspicious; DolphinAttack — ultrasound, costly, microphone-specific; Carlini & Wagner (CTC loss) — no human-perception constraint, grid search takes hours; CommanderSong — music only, image-domain noise minimization, no human-perception model.
- Gap addressed: no prior attack combined a *targeted* attack with the requirement that added noise be barely perceptible; none used psychoacoustics to hide a target transcription in another signal.
- Attacker model: **white-box**, targeted only (target transcription predefined), ASR configured for best recognition rate and unchanged over time, **perfect transmission channel** (audio file fed directly to the recognizer — no codecs, compression, hardware, or acoustic channel modelled).

## Method (precise but simple language, key equations)
1. Three-part pipeline: (1) **forced alignment** (Kaldi) between the original audio and the target text to find the best-fitting target pseudo-posterior sequence — if no alignment exists, the audio is divided equally among the HMM states; (2) **backpropagation** that updates the *input* (not the weights) with cross-entropy loss `L(yi,y') = −Σ yi·log(y')`, learning rate α = 0.05, 500 iterations by default; (3) **hearing thresholds** applied inside the backpropagation.
2. Preprocessing (framing + window, DFT, magnitude, log) is folded into the network — a joint "pre-sub-DNN + DNN" — so gradients reach the raw waveform: `∇x = ∂L/∂F(χ) · ∂F/∂FP(x) · ∂FP(x)/∂x`, with explicit derivatives `∂xw/∂x = w(n)`, `∂X/∂xw = e^{−i2πkn/N}`, `∇X = (2·Re(X), 2·Im(X))`, `∂χ/∂|X|² = 1/|X|²`.
3. Psychoacoustic model (MP3 / ISO 11172-3, thresholds from Zwicker & Fastl): 1024-sample buffers of two 576-sample granules, FFT → 32 bands → MPEG scale-factor bands measured in bark; threshold matrices normalized so the largest time-frequency bin energy is 95 dB.
4. Constraint: `D(t,f) ≤ H(t,k) ∀t,k` with `D(t,k) = 20·log10( |S(t,k)−M(t,k)| / max_{t,k}|S| )`; slack `Φ = H − D` (Eq. 3), `Φ* = Φ + λ` (Eq. 4, λ = allowed excess over threshold in dB), negatives clipped to 0, normalized to `Φ̂ ∈ [0,1]`; the DFT-domain gradient is scaled `∇X*(t,k) = ∇X(t,k)·Φ̂(t,k)·Ĥ(t,k)`.
5. Metrics: **WER = (D+I+S)/N** (can exceed 100 %); **perceptibility φ** = mean of all positive Φ(t,k) over T·N bins (dB) — chosen because SNR "does not represent the subjective, perceptible noise"; SNR only used for the CommanderSong comparison.

## Key quantitative results
- **Success rate: up to 98 %** of cases, **< 2 minutes** on a 6-core (12-thread) Intel Core i7-4960X for a **10-second** file at 500 backpropagation steps (abstract + §IV-D3).
- **Ablation (70 WSJ utterances, 10 speakers, 500 iterations, learning rate 0.05):**
  - baseline, no thresholds + no forced alignment: **WER 1.43 %**, **φ = 11.62 dB** (clearly perceptible)
  - + hearing thresholds (λ = 20): **WER 64.29 %**, **φ = 7.04 dB**
  - + forced alignment (λ = 20): **WER 36.43 %**, **φ = 5.49 dB** ("win-win": higher success *and* less noise)
- **Table I — WER % vs λ (columns: None, 50, 40, 30, 20, 10, 0 dB), test set 72 speech + 70 music samples:**
  - Speech, 500 iter: 2.14 | 6.96 | 11.07 | 16.43 | **36.43** | 92.69 | 138.21
  - Speech, 1000 iter: 1.79 | 3.93 | 5.00 | 7.50 | 22.32 | 76.96 | 128.93
  - Music, 500 iter: 1.04 | 8.16 | 13.89 | 22.74 | 31.77 | 60.07 | 77.08
  - Music, 1000 iter: 1.22 | 10.07 | 9.55 | 15.10 | 31.60 | 56.42 | 77.60
- **Table II — perceptibility φ (dB), same settings:**
  - Speech, 500: 10.11 | 6.67 | 6.53 | 5.88 | **5.49** | 4.70 | 3.05
  - Speech, 1000: 10.80 | 7.42 | 7.54 | 6.85 | 6.46 | 5.72 | 3.61
  - Music, 500: 4.92 | 3.92 | 3.56 | 3.53 | 3.39 | 2.98 | 2.02
  - Music, 1000: 5.03 | 3.91 | 3.68 | 3.40 | 3.49 | 3.20 | 2.30
  (Music always has lower φ than speech → "much easier to conceal adversarial examples in music.")
- **Table III — SNR (higher = less noise), successful adversarial samples:** None 15.88 | λ=40 17.93 | λ=20 **21.76** | λ=0 19.38 | CommanderSong 15.32. All of this paper's settings beat CommanderSong.
- **Phone rate (Fig. 7, 500 iter, λ=20, 200 adversarial examples per point):** WER rises with phone rate; **minimum at 4 phones/s**, which does not improve at lower rates → recommended setting.
- **Success vs iterations (Fig. 8):** speech files drawn from 150 samples, music from 72 samples, targets random from 120 predefined texts, ≤ 6 phones/s, 100 runs per λ, success = WER 0 %, cap 5000 iterations. Failures after 5000 iterations: only **2/100 for speech (λ=40)** up to **9/100 for music (λ=0)** → up to 98 % success. Authors recommend ≤ 500 iterations (more iterations add noise); increasing λ raises success more cheaply than λ=0 with many iterations.
- WER can exceed 100 % (insertions of the original text) — explicitly noted as normal under unfavourable conditions.

## Datasets / corpora used
- **Wall Street Journal (WSJ)**, default Kaldi WSJ training recipe: > **80 hours** of mostly clean read speech, dictionary of > **100,000 words**; only the preprocessing step was modified (to be integrated into the DNN).
- Evaluation subsets: 70 utterances from 10 speakers (one WSJ test set) for the ablations; 72 speech samples and 70 music samples for Tables I/II; 150 speech + 72 music samples and 120 predefined target texts for the success-rate experiments; bird-twittering recordings used as an additional MUSHRA condition.
- Target transcriptions are short command-like sentences (Appendix B), e.g. "DEACTIVATE SECURITY CAMERA AND UNLOCK FRONT DOOR", "THE SOUND OF SILENCE", "WINTER IS COMING".

## Models / systems evaluated
- **Kaldi** DNN-HMM ASR (default WSJ recipe; preprocessing → DNN pseudo-posteriors → FST graph decoding) — chosen because it is widely used in research and products; partial reverse engineering of Amazon Echo firmware indicated that the device also uses Kaldi internally.
- Attack implementation variants: baseline (no thresholds, no forced alignment), + hearing thresholds, + forced alignment; λ ∈ {0, 10, 20, 30, 40, 50} dB; 500 vs 1000 backpropagation iterations.
- Comparison target: **CommanderSong** (Yuan et al.) via reported SNR.

## Human-study details (if any: n participants, protocol, key numbers)
**Two-part user study, both conducted in a soundproof chamber with headphones.**

1. **Transcription test (§V-A)**
   - n = **22 listeners** (internal study at the university; **none were native English speakers**, all had sufficient English skills).
   - Protocol: each listener transcribed **21 audio samples**, identical utterances for everyone with randomly chosen conditions: **9 original utterances, 3 adversarial examples (λ = 0, 20, 40), and 3 difference signals** (original − adversarial, one per λ). All adversarial examples were valid (target hidden successfully) and required ≤ 500 iterations. Free typing into a blank text field, no auto-completion/spell-checking, unlimited repeats; manual correction of typos/proper nouns/numbers in post-processing (worked example in Appendix A).
   - Results: overall average **WER = 12.52 %**; **original utterances 12.59 %** vs **adversarial utterances 12.61 %** — two-sided t-test at **1 % significance level** found **no difference** in mean/variance. WER of listener transcripts against the **hidden target text was "far above 100 %"** → target incomprehensible; the only correct words were short frequent words shared with the original text (is, in, the). For the **difference signals, no listener recognized any speech at all** (no text transcribed).
2. **MUSHRA test (§V-B, ITU-R-style, webMUSHRA implementation)**
   - n = **30 test listeners, 3 discarded** under the MUSHRA exclusion rules → 27 analysed.
   - 9 stimuli sets: 3 speech, 3 music, 3 twittering-bird recordings; within a set the target text was constant; conditions = λ = 0, λ = 20, λ = 40, and no hearing thresholds; hidden reference = original audio, anchor = adversarial example made **without** hearing thresholds; all examples valid in ≤ 500 iterations.
   - Rating scale 0–100 (0–20 Bad, 21–40 Poor, 41–60 Fair, 61–80 Good, 81–100 Excellent). Exclusion rule: listeners rating the hidden reference < 90 points more than 15 % of the time, or the anchor > 90 points more than 15 % of the time, are dropped.
   - Results: reference usually rated **100 MUSHRA points**; anchor always lowest. One-sided t-tests at **1 % significance**: the no-threshold anchor is rated **significantly lower** than every hearing-threshold condition in **all cases**. No clear preference among λ values; only λ = 0 vs λ = 40 differ significantly (λ = 0 slightly higher — fewer iterations needed, each iteration adds noise). In one birds test, threshold-based adversarial examples were frequently rated **> 80 points**, i.e. barely distinguishable from the original.

## Limitations acknowledged by authors
- **White-box only**, targeted attacks only, **perfect channel**: audio is fed directly into the recognizer — no codec, compression, hardware, speaker/microphone, or acoustic transfer function is modelled (§VII-C "Future Work").
- Real-world (over-the-air) feasibility is unproven; needs the acoustic transfer function and environmental noise; commercial systems (Alexa) expose no architecture — would require model stealing; firmware reverse engineering only "indicates" Kaldi parts.
- Distillation is argued *not* to be an appropriate countermeasure for ASR (decoding also depends on temporal alignment); MP3 re-encoding as a defence was tested informally: the original transcription was not recovered, but the target was also distorted — the authors only *assume* training on MP3-encoded audio would push the vulnerability into the perceptible region rather than remove it.
- Parameter trade-offs: more iterations raise success *and* noise (recommend ≤ 500); λ influences WER strongly; phone rate must be tuned; the original audio choice matters a lot — music/birds are recommended because speech must be obfuscated and needs larger perturbations.
- Universal adversarial perturbations for ASR remain an open question; whether the attack works on commercial/black-box ASR and in the real world is future work.
- User-study caveats printed in the paper: listeners were **not native speakers**; WSJ texts are hard (baseline human WER 12.52 %); in the MUSHRA test listeners compared adversarial against the original directly, which would not happen in a real attack (which would tend to *underestimate* imperceptibility).

## Relevance to our project (one specific paragraph)
This is the base paper for AudioCAPTCHA-DSP and supplies both the attack-side threat model and much of the evaluation methodology for the Human–ASR Gap (HAG). Its central empirical fact — ASR transcribes the hidden command with high reliability while **human listeners are completely unaffected** (original 12.59 % vs adversarial 12.61 % human WER, target-text WER > 100 %, zero speech perceived in difference signals) — is exactly the extreme HAG we try to quantify and, conversely, to *avoid* when designing audio CAPTCHAs: the CAPTCHA wants distortions that hurt ASR while keeping human intelligibility intact, whereas psychoacoustic hiding wants perturbations invisible to humans yet decisive for the ASR. Three transferable elements for us: (a) the **hearing-threshold matrix / λ-slack construction** (Φ = H − D, Φ* = Φ + λ, gradients scaled by Φ̂·Ĥ) gives a principled, perceptually-calibrated *distortion budget* for our DSP transforms — instead of arbitrary SNR, we can bound perturbations by audibility and report the resulting φ-like excess-over-threshold measure alongside STOI; (b) their **parameter sweeps and metrics** (WER can exceed 100 %, φ vs λ tables, optimum 4 phones/s, success-rate-vs-iterations curves) are a template for how we should sweep distortion strength and report both human intelligibility (STOI proxy) and ASR WER on the same stimuli, with statistical tests (two-sided t-test at 1 %) on paired human vs machine scores; and (c) the **two-part listening-test protocol** (free-transcription test for *what* is heard, plus MUSHRA for *quality*, soundproof chamber, headphones, n = 22/30, pre-registered exclusion rules) is the template our HUMAN_STUDY_PROTOCOL.md should mirror when validating that an STOI-derived proxy really tracks human intelligibility under our distortions. Their explicit caution that SNR does not reflect perceived noise is the direct justification for our choice of an intelligibility/psychoacoustic metric over plain SNR when defining the HAG.

## Keywords
adversarial examples, audio adversarial attacks, psychoacoustic hiding, hearing threshold, masking, MP3 psychoacoustic model, targeted attack, Kaldi DNN-HMM, forced alignment, backpropagation to raw audio, word error rate, perceptibility metric φ, MUSHRA listening test, human listening study, human–machine speech recognition gap, voice assistant security
