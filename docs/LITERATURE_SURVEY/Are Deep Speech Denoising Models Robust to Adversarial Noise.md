# Are Deep Speech Denoising Models Robust to Adversarial Noise?

## Full reference
- Title: *Are Deep Speech Denoising Models Robust to Adversarial Noise?*
- Authors (as printed): Will Schwarzer*1, Neel Chaudhari2, Philip S. Thomas1, Andrea Fanelli2, Xiaoyu Liu†2 (* correspondence to wschwarzer@umass.edu; † affiliation reflects status at time of work)
- Affiliations: 1University of Massachusetts; 2Dolby Laboratories
- Venue/year (as printed): arXiv preprint — arXiv:2503.11627v2 [cs.SD], 11 Mar 2026 (22 pages; no conference/journal venue printed). Samples/code linked: sites.google.com/view/adv-dns.

## One-line contribution
Shows that four recent open-source deep noise suppression (DNS) models can each be driven to output unintelligible gibberish by psychoacoustically hidden (MP3-style masking-threshold-constrained) PGD perturbations — including in near-clean 70 dB SNR and simulated over-the-air settings — confirmed by a transcription study of audio experts and an ABX imperceptibility study.

## Problem & motivation
Deep noise suppression models are deployed in videoconferencing, ASR front ends, and (prospectively) hearing aids, emergency-responder and air-traffic-control channels; many are open source with public weights, giving attackers full gradient access, and safety-critical channels carry stereotyped utterances. Prior attacks on speech enhancement were perceptible (∞-norm bounded), limited to high-noise settings with no reverb, and of unclear over-the-air viability (Dong et al.). Because DNS models are *designed* to remove additive noise, one might expect adversarial perturbations to be suppressed — the paper tests this expectation systematically with strict psychoacoustic masking (the Schönherr/Qin line of work), covering near-clean to noisy/reverberant conditions, targeted vs untargeted goals, transfer, universal perturbations, defenses, and over-the-air propagation.

## Method (precise but simple language, 5-12 lines; key equations if present)
1. Threat model: input x = r*(y+b) (clean speech y, background b, RIR r); untargeted attack maximizes L(f(x+δ), y) over δ in the perceptible-feasible set D(x); targeted attack minimizes L(f(x+δ), y′).
2. Loss/optimization objective: STOI — untargeted L = −STOI(f(x+δ), y); targeted L = STOI(f(x+δ), y′) − STOI(f(x+δ), y). MSE, PESQ (no academic license), DNSMOS/NISQA/Whisper-WER were considered but rejected as optimization objectives (MSE is phase-dependent; DNN metrics get attacked instead).
3. Perceptibility: MP3-like psychoacoustic model (Lin & Abdulla; as used by Schönherr et al. and Qin et al.) computing masking thresholds θ(τ,ω) on the PSD, plus temporal pre-/post-masking (post: exponential decay 0.02 ms⁻¹, cut at 100 ms; pre: 0.16 ms⁻¹, cut at 20 ms); thresholds shifted down by 12 dB (default), with −8 dB (looser) and −16 dB (tighter) variants.
4. Optimization: projected gradient descent δ_{t+1} = Π_{D(x)}(δ_t + α ∂L/∂δ_t), where projection clips STFT magnitude so PSD(δ)_{τ,ω} ≤ θ_{τ,ω} while preserving phase (scaling by min(1, 10^((θ−PSD)/20))). Adam, lr 0.01, gradient clip 2-norm 10, lr ×0.99 after 10 stalled iterations; STFT: Hann, 512 FFT, window 512, hop 256; δ parameterized in time domain (STFT domain for OTA).
5. Over-the-air: the RIR is also applied to δ, so projection has no closed form — solved by Wiener deconvolution (ε = 10⁻⁴), gradient-descent projection on g(δ) = Σ max(PSD(r*δ) − (θ − d), 0) with d = 1 dB, or a combination; OTA constraints end up ≈ 6 dB looser.
6. Budgets standardized by wall-clock (~1 hour on an Nvidia L40S): 20,000 iterations (Demucs, FSN+), 10,000 (MP-SENet), 5,000 (FRCRN); 20 shared seeds (utterances) per (setting, model) combination.
7. Also tested: white-noise (Gaussian) defense at SNRs 0–90 dB, universal perturbations (D_U = intersection of per-utterance thresholds), cross-architecture and leave-one-out Demucs-checkpoint transfer, and an ablation of five constraint strategies (ℓ∞/ℓ2 in STFT, frequency-only masking, Qin et al. thresholds, full method).

## Key quantitative results
- Core result (Fig. 1, ΔSTOI = STOI(clean,output) − STOI(clean,input)): all four models flip from positive baseline ΔSTOI to negative after attack across all SNR (−10…70 dB) and reverb settings; 20 seeds, error bars ±SE. Unattacked average ΔSTOI ≈ +0.044.
- Table 1 (attacked ΔSTOI, architecture ablation): Demucs — time domain, 33.5 M params, ΔSTOI −1.08; FRCRN — TF, 10.3 M, −0.99; FSN+ — TF, 8.7 M, −0.49; MP-SENet (MPSE) — TF, 2.3 M, −1.25. Model size does not predict robustness.
- Table 3 (fixed 5,000 iterations for every model, SNR = 30 dB, reverb, 20 seeds, mean ± sd of ΔSTOI): FSN+ +0.035±0.022 → −0.338±0.245; FRCRN +0.057±0.024 → −1.034±0.159; Demucs +0.031±0.041 → −1.082±0.183; MP-SENet +0.036±0.040 → −1.258±0.102 (robustness ranking unchanged vs. equal-time budget).
- Table 2 (transfer matrix, ΔSTOI): diagonal white-box −1.08 / −0.49 / −0.99 / −1.25; all off-diagonals 0.03–0.08 — psychoacoustically hidden attacks do not transfer across architectures (nor across Demucs checkpoints dns48/dns64/master64 in leave-one-out, Fig. 3/8: only minor degradation).
- FSN+ "protection": STOI-loss gradients w.r.t. the adversarial waveform grow to norm ≥ 10³⁰, causing numerical instability (pseudo-robustness / obfuscated gradients).
- Targeted attacks: objective metrics can report success while listening fails — outputs with STOI > 0.5 to the target yield "at most a faint robotic hint of the target speech"; cross-speaker and MaskGCT voice-cloned targets were empirically ineffective.
- UAPs: only slight degradation while remaining imperceptible (no effective universal perturbation).
- Ablation (30 dB SNR + reverb, Demucs master64, 5,000 iters): full method (−12 dB + temporal masking) achieves ΔSTOI ≈ −1.1; Qin et al. thresholds with no offset (method 4) achieve the strongest attack ΔSTOI ≈ −1.4 but are most perceptible; frequency-only masking needs an offset of −8.4 dB to match, so temporal masking contributes a 3.6 dB extra perturbation budget; ℓ∞ (ε = 0.04) and ℓ2 (ε = 18) constraints reach the same ΔSTOI only with louder, audible perturbations.
- Real recorded RIRs (OpenSLR28, Fig. 14): STOI still drops by > 0.4 from input at background SNRs of 30 dB or noisier.
- Defense: Gaussian "white noise" at 0–90 dB SNR raises STOI but does not restore clean performance, and only helps at SNRs that also degrade normal operation.
- Human transcription study: 95% upper bounds on mean WAcc differences (intersection–union test) — attacked output vs. attacked input U1 = −0.464; vs. clean output U2 = −0.458 (both < 0 → rejection).
- Human ABX: mean accuracy 59% vs. 50% chance; one-sided 95% pigeonhole-bootstrap lower bound 0.478 → not significantly above chance (imperceptibility supported, conservatively).

## Datasets / corpora used
All audio from the main track of the ICASSP 2022 DNS Challenge dataset: 10-second clean speech randomly drawn from English read speech (LibriVox.org, per Dubey et al.) and the VCTK Corpus; noise also from the DNS challenge set (5 noise sources per condition, repeated up to 10 s); RIRs simulated plus real recorded RIRs from OpenSLR26/28. Audio 16-bit, 16 kHz, single channel; speech filtered to ≥ 15 words by Whisper; clips truncated to 5 s for MP-SENet (VRAM bug) and to 5 s in the human study (≥10 words, restricted to the 2,000 most common English words).

## Models / systems evaluated
- Attacked DNS models (public checkpoints): Demucs/Denoiser (master64, time domain, 33.5 M, also dereverberates), Full-SubNet+ (FSN+, TF, 8.7 M / 8.67 M), FRCRN (TF, 10.3 M), MP-SENet (TF, 2.26 M).
- Metrics: STOI (loss), ViSQOL, NISQA, DNSMOS, Whisper word accuracy (1 − min(WER,1)); Whisper also supplies ground truth on clean speech.
- Voice cloning for targeted attacks: MaskGCT.
- Defense: additive Gaussian white noise at controlled SNRs.

## Human-study details (if any: n participants, protocol, key numbers)
- Participants: 15 adult audio/multimedia researchers from an industrial research lab; voluntary, during paid hours, no extra compensation; institution's minimal-risk policy → no formal IRB review; informed consent, anonymized responses only.
- Protocol: fixed order — (1) transcription, then (2) ABX; online platform on participants' own devices; designed < 25 min; no returning from ABX to transcription; headphone use recommended (not enforced); blank transcription box allowed if no intelligible speech.
- Stimuli: 5 s clips; conditions 30 dB and 50 dB SNR with reverb; models Demucs, FRCRN, MP-SENet (FSN+ excluded to focus budget); attacks used the tighter masking constraint (−16 dB offset in the study; attention checks used +16 dB so they were clearly audible).
- Tasks: transcription — 18 clips per participant (6 Attacked Input, 6 Clean Output, 6 Attacked Output; identical across participants; second of 20 seeds omitted as near-unintelligible whispering); ABX — 12 pairs + 2 attention checks.
- Statistics/numbers: word accuracy (WAcc = 1 − WER); 95% two-way ("pigeonhole") bootstrap CIs; IUT upper bounds −0.464 (attacked output vs attacked input) and −0.458 (vs clean output); ABX mean 59%, lower 95% bound 0.478 (not significant vs 50%).

## Limitations acknowledged by authors
- Strongest attacks are white-box: naive cross-architecture transfer fails under strict masking (gradients are nevertheless always available for open-source models); pure black-box attacks need further study.
- FSN+ is shielded by exploding gradients (pseudo-defense, known to be easily circumvented per Athalye et al.); it may actually be more vulnerable to black-box attacks.
- Only fully differentiable DNS models were attacked; token-based DNS models (e.g., SELM) need new techniques.
- Attacks are offline and per-utterance; streaming would require universal perturbations (ineffective here), lookahead, or advance knowledge.
- OTA experiments model linear RIR propagation only — real playback chains add nonlinearity, AGC, and codec compression that may attenuate the perturbation; OTA also needed a looser (≈ 6 dB) constraint causing slightly audible crackling.
- STOI works as a minimization objective but breaks as a maximization metric (targeted attacks look successful objectively, not subjectively); PESQ could not be used (license).
- White-noise defense was evaluated only against a non-adaptive attacker; adaptive attacks likely bypass it; hyperparameters chosen by the authors' own subjective listening; one OTA run failed to converge (3 h on L40S) and was set to the initial value.

## Relevance to our project (one specific paragraph)
This is the closest methodological cousin of AudioCAPTCHA-DSP: it operationalizes exactly the human-vs-machine asymmetry we measure, using the same Schönherr-style psychoacoustic hiding (masking thresholds, pre/post masking, tunable dB offsets) but optimizing STOI — an intelligibility proxy for humans — against a speech-processing DNN, and then *validating the proxy against listeners* (n = 15 experts, transcription + ABX). Three transfers to our work: (1) it demonstrates that a distortion can be simultaneously (a) imperceptible to humans (ABX ≈ chance, 59%) and (b) catastrophic for a machine pipeline (ΔSTOI output −1.0 to −1.3), which is the HAG logic in another domain — for us, Whisper/Vosk failing while STOI-derived human intelligibility stays high; (2) its ablation quantifies design knobs we also expose in our distortion suite (masking offset −8/−12/−16 dB, temporal pre/post masking worth 3.6 dB of budget, ℓp vs psychoacoustic constraints), giving precedent for a power-vs-imperceptibility trade-off curve in CAPTCHA design; (3) its metric discipline (five metrics, disagreement between STOI-as-objective and STOI-as-metric, bootstrap CIs, negative results on transfer/UAPs) is a template for how carefully our HAG metric must be defined and uncertainty-reported, and its defense analysis (Gaussian noise as a weak baseline, RIR convolution acting as smoothing) informs what an attacker or a CAPTCHA designer could do to close the gap.

## Keywords
deep noise suppression, adversarial perturbation, psychoacoustic hiding, auditory masking thresholds, projected gradient descent, STOI, speech enhancement robustness, over-the-air attack, universal adversarial perturbation, imperceptibility study, transcription study, ABX, Whisper word accuracy, adversarial defenses, speech intelligibility metrics
