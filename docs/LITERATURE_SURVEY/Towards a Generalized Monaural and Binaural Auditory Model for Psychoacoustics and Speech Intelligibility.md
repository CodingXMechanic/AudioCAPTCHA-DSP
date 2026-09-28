# Towards a Generalized Monaural and Binaural Auditory Model for Psychoacoustics and Speech Intelligibility

## Full reference
- Title: *Towards a generalized monaural and binaural auditory model for psychoacoustics and speech intelligibility*
- Authors (as printed): Thomas Biberger, Stephan D. Ewert
- Affiliation: Medizinische Physik and Cluster of Excellence Hearing4all, Universität Oldenburg, 26111 Oldenburg, Germany (corresponding: thomas.biberger@uni-oldenburg.de)
- Running title (as printed): "Modeling masking and speech intelligibility"
- Venue/year (as printed): manuscript submitted to *Acta Acustica* — cover page states "This work has been submitted to Acta Acustica for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible." No arXiv ID or DOI for the paper itself is printed (56 pages; references carry their own DOIs).
- Funding printed: DFG – 352015383 – SFB1330 A2 and DFG – 390895286 – EXC 2177/1.

## One-line contribution
Extends the monaural Generalized power spectrum Model (GPSM/mr-GPSM) with a fixed, non-adaptive 5-channel binaural stage — the binaural matrix feature decoder (BMFD: BEL, BER, BIL, BIC, BIR) — that reproduces both monaural and binaural psychoacoustic benchmarks and a large share of spatial release from masking in speech intelligibility without any adaptive equalization–cancellation machinery.

## Problem & motivation
Auditory models have traditionally treated monaural phenomena (spectral/temporal masking) and binaural phenomena (BMLD, spatial release from masking, SRM) separately, even though real-world perception and practical applications (hearing aids, instrumental SI/quality prediction) mix both. Existing combination strategies require adaptation: the equalization–cancellation (EC) model needs signal-adaptive delays/gains and a channel selection among 3 outputs, and the Breebaart delay–gain matrix model needs a signal-adaptive template to pick matrix elements. Physiological work questions delay lines in mammals and suggests simpler hemispheric processing with fixed phase delays and contralateral excitation/inhibition; simple L+R summation alone has been shown to explain much of the SRM for symmetric interferers. The authors ask whether a *non-adaptive*, few-parameter binaural stage suffices, and whether one and the same model can be evaluated on monaural psychoacoustics, binaural psychoacoustics, and speech intelligibility (unified modelling), which also aids efficiency in real-time/hearing-device applications.

## Method (precise but simple language, 5-12 lines; key equations if present)
1. Monaural front end (as in mr-GPSM): outer/middle-ear weighting by the hearing threshold in quiet; 4th-order Gammatone filterbank, ERBN bandwidths, 1/3-octave spacing 63–12,500 Hz (psychoacoustics) or 63–8,000 Hz (SI); half-wave rectification; neural adaptation by a 1st-order 500 Hz low-pass (2 ms time constant).
2. Before the binaural stage, independent amplitude jitter σε = 0.25 and delay jitter σδ = 105 µs are applied per auditory channel.
3. BMFD (5 fixed channels) computes, per auditory channel p: BIL(p,t) = L(p,t) − α·R(p,t−τ(p)); BIC(p,t) = √(L(p,t)·R(p,t)); BIR(p,t) = R(p,t) − α·L(p,t−τ(p)), plus unchanged L and R (better-ear channels BEL, BER).
4. Fixed parameters: frequency-dependent delay τ = π/4 phase shift, amplification α = 3 (values 3–5 gave similar performance).
5. Per BMFD channel: 150 Hz low-pass, then two pathways — power SNR: P_DC,j(p) = [mean E_j(p)]², windows 45 ms (63 Hz) to 8 ms (highest cf., values ×2.5 per Rhebergen & Versfeld), SNR_DC,j(p) = 10·log10(P_targ+mask/P_mask); envelope-power pathway: modulation filterbank 2–256 Hz (Q = 1, 3rd-order 1 Hz LPF), AC-coupled envelope power with −27 dB lower limit, logarithmic weighting below 35 dB.
6. Decision stage combines the 5 channels by taking the per-frame maximum, then SNR = max[ β·(ΣΣ SNR_env²)^(1/2), γ·(Σ SNR_DC²)^(1/2) ] with β = 0.21, γ = 0.45; detection criterion SNR > −6 dB ⇔ d′ = (2·SNR)^(1/2) ≈ (0.5)^(1/2).
7. For SI, an optional band-importance function (ANSI S3.5 Table 3) multiplies the power SNRs; the overall SNR → d′ → percent correct (Jørgensen et al. 2013 eq. 6).
8. Evaluation: three configurations — BMFD (all 5 channels), BIL,C,R (3 binaural-interaction channels), BIL,R (only the two difference channels) — against 13 psychoacoustic experiments and 6 SI conditions; SI calibration (k, q, m, σs) is fit once on the co-located SSN condition and then held fixed.

## Key quantitative results
(Table 1: RMSE / R² between data and predictions; monaural columns BMFD | BIL,R | mr-GPSM; binaural columns BMFD | BIL,C,R | BIL,R)

Monaural experiments:
| Experiment | BMFD | BIL,R | mr-GPSM |
|---|---|---|---|
| 1 Hearing threshold | 3.3 dB / 0.99 | 3.3 / 0.99 | 1.7 dB / 0.99 |
| 2 Intensity JNDs | 0.2 dB / 0.66 | 0.2 / 0.64 | 0.3 dB / 0.57 |
| 3 Tone in noise | 1.3 dB / 0.99 | 1.3 / 0.99 | 2.1 dB / 0.99 |
| 4 Spectral masking | 9.5 dB / 0.82 | 9.5 / 0.8 | 7.9 dB / 0.9 |
| 5 AM detection | 4.0 dB / 0.71 | 4 / 0.78 | 4.5 dB / 0.68 |
| 6 AM discrimination | 2.4 dB / 0.94 | 2.4 / 0.92 | 1.6 dB / 0.94 |
| 7 AM masking | 4.6 dB / 0.77 | 4.7 / 0.79 | 6.2 dB / 0.73 |

Binaural experiments:
| Experiment | BMFD | BIL,C,R | BIL,R |
|---|---|---|---|
| 1 ITD discrimination | 0.019 ms / 0.89 | 0.019 / 0.90 | 0.019 ms / 0.93 |
| 2 IID discrimination | 0.5 dB / 0.002 | 0.5 / 0.0014 | 0.5 dB / 0.005 |
| 3 Wideband freq./interaural phase | 9.1 dB / 0.86 | 8.5 dB / 0.85 | 6.7 / 0.88 |
| 4 N0Sπ vs signal duration | 2.9 dB / 0.92 | 3.0 / 0.92 | 3.2 dB / 0.90 |
| 5 Temporal phase transition | 2.6 dB / 0.80 | 2.7 dB / 0.80 | 2.7 dB / 0.81 |
| 6 Time-intensity trading | 0.5 / 0.38 | 0.6 / 0.58 | 0.6 / 0.61 |

Other printed numbers:
- Benchmark size: 13 psychoacoustic experiments + 6 conditions of one speech intelligibility experiment.
- Measured ITD thresholds: smallest ≈ 0.012 ms at 1 kHz (constant IPD ≈ 0.05 rad ≈ 3°); model predicts constant IPD ≈ 0.07–0.08 rad (≈ 4°–5°) and lowest ITD threshold ≈ 0.023 µs at ≈ 700 Hz (as printed).
- IID thresholds: Mills average ≈ 0.8 dB (max ≈ 1 dB at 1 kHz); Grantham ≈ 1.3 dB higher; model predicts ≈ 2 dB at 62.5 Hz falling to ≈ 1.1 dB at 2 kHz; BIL,R/BIL,C,R ≈ 0.2 dB higher than BMFD between 62.5 Hz and 2 kHz.
- BMLD threshold differences NπSm−N0Sm and NπS0−N0Sπ up to ≈ 9.5 dB below 500 Hz; model deviation up to 10 dB at 250 Hz (BIC over-predicts NπSm/NπS0).
- N0Sπ duration slopes: data ≈ 4.5 dB per doubling (500 Hz, short durations), ≈ 1.5 dB (longer), ≈ 3 dB (4 kHz); model ≈ 3 dB per doubling.
- SI data (Ewert et al. 2017, headphone OLSA): measured SRM 4.3–13.5 dB (SSN smallest ≈ 4.3 dB; ISTS 10.1 dB; single talker ST 13.5 dB); co-located ST SRT ≈ 5.5 dB higher than SSN.
- Model-vs-data SI errors: BMFD SAM SRT ≈ 3 dB too high; co-located ISTS/ST SRTs off by up to 13 dB; SRM under-predicted ≈ 2 dB (SAM), ≈ 3 dB (BB), up to 5 dB (ISTS, ST).
- Subtractive vs additive binaural interaction: SRM RMSE 3.3 dB (current subtractive) vs 5.5 dB (additive variant).
- Table 2 calibration: BMFD k = 0.6, q = 0.5, m = 50, σs = 0.6; BIC and BIC^AC k = 0.72 (q, m, σs identical). SI predictions averaged over 5 repeated simulations × 20 OLSA sentences.
- Temporal phase-transition experiment quoted in text as "RMSE ≈ 2.7 dB and R² ≈ 0.8".
- Comparison parameters: Breebaart model internal delays up to 5 ms (π phase shift at 100 Hz) and interaural gain difference up to 10 dB; binaural sluggishness time constants up to ≈ 200 ms (not modelled).

## Datasets / corpora used
No new recordings; the model is evaluated on previously published psychoacoustic data (baseline database from the literature): monaural — intensity discrimination/hearing thresholds (Houtsma et al.; ISO 389-7), spectral masking patterns (Moore et al.), tone-in-noise with durations (Jepsen et al.), AM-depth discrimination (Ewert & Dau), TMTF/AM detection (Dau et al.; Viemeister), AM masking (Biberger & Ewert); binaural — ITD (Klumpp & Eady; Zwislocki & Feldman), IID (Mills; Grantham), wideband N0Sm/N0Sπ/NπSm/NπS0 (Hirsh; van de Par & Kohlrausch; Hirsh & Burgeat; Kohlrausch), N0Sπ vs duration (Yost; Wilson & Fowler/Fugleberg; Bernstein & Trahiotis), temporal phase transition (Kollmeier & Gilky), time–intensity trading (Hafter & Carrier). Speech intelligibility: headphone dichotic data of Ewert et al. (2017) using the German Oldenburger Satztest (OLSA) sentences with maskers SSN, SAM, BB, AFS, ISTS and single talker (ST), co-located (0°) vs separated (±60°), masker 65 dB SPL each (68 dB SPL total).

## Models / systems evaluated
- Proposed: GPSM with BMFD extension (three read-out versions: BMFD = 5 channels; BIL,C,R; BIL,R), plus a BIC-only and BIC^AC (envelope-power-only) analysis for SI.
- Baseline/monaural reference: mr-GPSM (Biberger & Ewert 2017).
- Conceptual comparisons (discussed, not re-run as implementations): EC model (Durlach), Breebaart et al. contralateral-inhibition matrix model, BSIM, b-sEPSM, ESII; a variant replacing subtractive BIL/BIR processing with additive processing (SRM RMSE 5.5 dB).

## Human-study details (if any: n participants, protocol, key numbers)
No new human experiments were run — the paper compares model predictions against previously published listener data (this is the authors' stated design). Numbers printed about the underlying studies: Kollmeier & Gilky temporal-phase-transition data from 4 subjects; Hafter & Carrier time–intensity trading shown for subjects S1 and S4 (the two subjects differing most); SI data from Ewert et al. (2017) OLSA headphone measurements (number of listeners not restated in this paper — not stated/illegible here). SI predictions are averaged over 5 model repeats × 20 sentences and calibrated on the co-located SSN condition only.

## Limitations acknowledged by authors
- Informational masking is not covered: intrusive SI models have a-priori knowledge of target and masker, so co-located speech-like maskers (ISTS, ST) are predicted far too well (SRT errors up to 13 dB) while human thresholds are high and variable.
- Time–intensity trading (Hafter & Carrier) is poorly captured (BMFD R² = 0.38; BMFD fails to predict the ILD-dependence on ITD); binaural psychoacoustics was "well covered except for larger discrepancies for time-intensity trading".
- The BIC channel overestimates human performance for NπSm and NπS0 at 250/500 Hz (deviation up to 10 dB at 250 Hz).
- Binaural sluggishness is not modelled (monaural and binaural channels share the same time constants, so transition slopes in Kollmeier & Gilky data are not matched); adding a task-dependent binaural temporal window (up to ≈ 200 ms) could improve predictions.
- Physiology is strongly simplified: subtraction of half-wave-rectified signals only partially resembles hemispheric net neural activation; no PSP dynamics, no distinct MSO/LSO low- vs high-frequency processing; an additive (excitatory) variant yields worse SRM (5.5 vs 3.3 dB RMSE).
- A "specialist" per-experiment model may outperform this general model; power-SNR-only predictions overestimate SRM and under-predict SRTs in fluctuating maskers (a forward-masking function or SNR limit is suggested).
- Calibration caveat: SI parameters were fit on the SSN condition; one parameter set is used for all experiments.

## Relevance to our project (one specific paragraph)
This paper is our anchor for *model-based* human intelligibility estimation beyond simple STOI-style correlation metrics: the GPSM/BMFD is an explicit perceptual front end (gammatone filtering, adaptation, envelope-power and power SNRs, d′ decision) validated on 13 psychoacoustic masking benchmarks plus spatial release from masking, i.e., exactly the machinery that determines whether a human can still decode a distorted CAPTCHA utterance while an ASR system cannot. For AudioCAPTCHA-DSP it matters in three concrete ways: (a) its masking/threshold apparatus (ERB bands, modulation filterbank, −6 dB detection criterion, β/γ combination) is a principled alternative or complement to STOI for the human side of our Human–ASR Gap metric, and it exposes which cues (AM coherence, better-ear selection, energetic masking) survive a given DSP distortion; (b) its finding that a few fixed channels plus spectro-temporal frame selection explain most human performance argues that CAPTCHA distortions should be evaluated per time-frequency frame rather than in global norms, matching the psychoacoustic-hiding logic of Schönherr et al.; and (c) its documented failure modes (informational masking with speech maskers, binaural sluggishness) delimit where any intrusive intelligibility proxy — including ours — is expected to deviate from real listeners, which we should report as uncertainty in the HAG rather than as model error.

## Keywords
auditory modeling, GPSM, binaural masking level difference, spatial release from masking, envelope power spectrum model, speech intelligibility prediction, psychoacoustic masking, binaural interaction, equalization-cancellation, Gammatone filterbank, speech reception threshold, auditory masking thresholds, hearing aids, signal detectability d′, monaural-binaural model
