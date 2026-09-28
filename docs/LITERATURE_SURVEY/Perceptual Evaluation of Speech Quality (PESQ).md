# Objective Speech Quality Metric (ViSQOL) — ⚠ file named "Perceptual Evaluation of Speech Quality (PESQ).pdf" does NOT contain the PESQ paper

> **FILE-CONTENT WARNING (read first).** The PDF stored as `Perceptual Evaluation of Speech Quality (PESQ).pdf`
> is **not** the PESQ paper (ITU-T P.862 / Rix et al.). It is a **20-page scanned US patent** —
> US 9,524,733 B2, "Objective Speech Quality Metric", Google Inc. — describing the **ViSQOL** metric.
> The PDF has no text layer (Producer: USPTO, Title metadata = "9524733"); it had to be page-rendered at
> 2560×3300 px and OCR'd, so some symbols (percent signs, subscripts, equation operands) were lost or garbled.
> PESQ itself appears only as *background/prior art* inside this document. Anything marked "(OCR)" below is
> literally what the recognizer produced and may be imperfect.

## Full reference
- **Title (as printed):** OBJECTIVE SPEECH QUALITY METRIC
- **Document type:** United States Patent, **Patent No. US 9,524,733 B2**, **Date of Patent: Dec. 20, 2016**
- **Inventors:** Jan Skoglund (Mountain View, CA); Andrew J. Hines (Dublin, IE); "Noami A. Harte" (OCR; Dublin, IE); Anil Kokaram (Mountain View, CA)
- **Applicant / Assignee:** Google Inc., Mountain View, CA (US)
- **Appl. No.:** 13/891,978 · **Filed:** May 10, 2013 · **Prior Publication Data:** US 2015/0199959 A1, Jul. 16, 2015 · Provisional application No. 61/645,433, filed May 10, 2012
- **Classification (OCR):** Int. Cl. "GIOL 25/60" (almost certainly G10L 25/60); U.S. Cl. / CPC 704/239, 324/76.38, G01L 19/0204, 704/203, H04R 25/70, 381/60 (OCR garbled)
- **Examiner:** Barbara Reinier · **Attorney, Agent, or Firm:** "— Brake Hughes LLP" (OCR, garbled)
- **Extent:** 26 Claims, 9 Drawing Sheets (20 PDF pages)
- **Venue/year:** USPTO-granted patent, 2016 (no arXiv ID / DOI printed)
- No standard bibliographic venue (journal/conference) is printed anywhere in the document.

## One-line contribution
A signal-based **full-reference** speech-quality metric, **ViSQOL**, that compares reference/test spectrograms with
patch-wise **Neurogram Similarity Index Measure (NSIM)** plus temporal patch warping, and additionally outputs an
estimate of **clock drift / time-warp factor** — designed to fix the known failure of PESQ/POLQA on
VoIP clock-drift and jitter degradations.

## Problem & motivation
- PESQ (ITU-T P.862) and its successor POLQA (P.863) are ITU full-reference measures that predict speech quality by
  comparing reference and received signals, but **clock drift** in VoIP systems causes PESQ/POLQA quality
  estimates to **drop even though listeners perceive no quality change**.
- QoS metrics can predict delay/drift but are limited in predicting **end-user perceptual quality of experience**.
- The ITU standard itself acknowledges PESQ gives inaccurate predictions for listening levels, loudness loss,
  effects of delay in conversational tests, talker echo and side tones.
- NSIM was originally built to predict **speech intelligibility** from auditory-nerve ("neurogram") outputs; the
  document argues peak-clipped speech keeps intelligibility but loses "aesthetic" quality, so NSIM-like similarity
  may capture quality beyond intelligibility — and applies it to plain spectrograms instead of neurograms.

## Method (precise but simple)
1. Input: a **short speech reference signal (3–15 s)** and its degraded **test signal**.
2. Build STFT spectrograms with **30 frequency bands logarithmically spaced between 250 and 8,000 Hz**;
   **512-sample window, 50 % overlap (OCR: "5000 overlap"), Hamming** at 16 kHz, **256-sample window** at 8 kHz.
3. **Patch selection:** from the reference spectrogram pick **three patches, each 30 frames × 30 frequency bands**
   (23 bands, 250–3.4 kHz, for narrowband assessment), one per selected band — **bands 2, 6, 10 ≈ 250, 450, 750 Hz** —
   chosen as the **maximum-intensity frame** in each band so patches contain speech, not silence.
4. **Patch alignment:** slide across the test spectrogram and compute a **relative mean squared error (RMSE)**
   between reference patch and test patch frame by frame → maximum-correlation frame index.
5. **Temporal warping:** because NSIM is more time-warp sensitive than human listeners, also build warped versions of
   each reference patch **10 % to 50 % longer and shorter (OCR: "100 to 500")** via **cubic 2-D interpolation**.
6. **Similarity:** NSIM between test patch and each (un)warped reference patch; take the maximum per patch; return the
   **mean NSIM over the three test patches** as the signal-similarity estimate. NSIM output is bounded **0 (no
   similarity) to 1 (identical)**.
7. **Outputs:** predicted speech quality on a **0 to 1 scale**, plus the list of **warp factors** used, which reveals
   whether (and by how much) the test signal was time-warped.
8. **Warp prediction:** a **Laplacian function, eq. (1)**, is fitted to mean NSIM per resample factor and inverted to
   give eq. (2), a predictor of warp factor from NSIM; the sign (faster/slower) is disambiguated from the ratio of
   patches smaller vs. larger than the original size. *(Equation operands are partly illegible in OCR — recovered
   fragments: "− c)) + g, 0.06 y 2 0.89".)*

## Key quantitative results
- **PESQ scale (as described):** scored **−0.5 to 4.5**, speech results usually **1 to 4.5**; a transfer function
  mapping PESQ → MOS-LQO exists (ITU-T P.862.1 cited).
- **Example 1 — clock-drift simulation:** ten sentences, 8 kHz reference, **resampling factors 0.85 → 1.15**
  (14 warp factors per sentence). PESQ predicted-quality drop occurs **"between 300 and 400"** resampling while the
  NSIM drop occurs **"between 500 and 1000"**, matching the listeners (OCR lost the %/decimal markers — presumably
  3–4 % vs 5–10 %). Audibility statements as OCR'd: differences **not audible at "20 0"** resampling or less; not a
  dramatic degradation until **"500 to 1000"**; small delay **"I to 4, or 50 0"** unlikely to be noticed.
- **PESQ vs ViSQOL dispersion:** "The standard deviation for PESQ is significantly larger than for ViSQOL, which is
  more consistent for the same time warp."
- **Warp-factor prediction:** model is "**very accurate at predicting warps of 1000**" (OCR; ≈ 10 % around the
  reference rate) for clean data; magnitude at **"1500"** (≈ 15 %) still predicted well, **but the direction is
  ambiguous: a 1.15 resample factor is predicted as 0.85** (Test A and Test B speakers).
- **Example 2 — clock drift + jitter (human listening):** eight concatenated IEEE sentences, **ten jitter
  conditions**; **mean MOS = 3.6, SD = 0.23**; **mean PESQ-LQO = 3.33 (σ = 0.38)**. With jitter and no time warping,
  **PESQ-LQO was within 0.3 of MOS**, but PESQ-LQO "**drops significantly for warps greater than 100**" (OCR, ≈ 10 %).
  For ViSQOL under jitter: **maximum NSIM (unwarped) just over 0.6**, falling to **≈ 0.4** at the largest warp.
- **Test sets used for the Laplace fit / Figs. 4–7:** Test A = "IEEE Speaker" (model fit), Test B = "TIMIT Speaker",
  Test C = "Jitter Warp Speaker"; each = **single speaker × ten reference sentences × fourteen warp factors**.
- **ITU / standard references printed in the patent:** ITU-T P.862 (Feb. 2001), P.862.1 (Nov. 2003),
  P.863 / POLQA (Jan. 2011), IEEE recommended practice for speech quality measurements
  (IEEE Trans. Audio Electroacoust., vol. AU-17, no. 3, pp. 225–246, Sep. 1969).
- No correlation-coefficient table (PCC/SRCC) is printed — this is a patent, not an experimental paper.

## Datasets / corpora used
- "Ten sentences from a speech corpus" (unspecified) at **8 kHz** for the clock-drift experiment.
- **IEEE sentences** (eight, concatenated) for the jitter listening test.
- **TIMIT speaker** referenced as one of the three example-speaker tests; **IEEE speaker** used for the Laplace fit.
- No dataset names, counts of speakers, or licence details are given beyond the above.

## Models / systems evaluated
- **ViSQOL** (proposed) — NSIM-based full-reference metric.
- **PESQ / PESQ-LQO** (ITU-T P.862 / P.862.1) — baseline comparator.
- **POLQA (ITU-T P.863)** — discussed as successor to PESQ, not experimentally compared in the examples.
- **NSIM** — adapted from intelligibility prediction to quality prediction.
- Underlying implementation target: generic computing device (Fig. 9 block diagram, DSP/microprocessor).

## Human-study details
- **Example 2 only:** eight concatenated IEEE sentences presented to listeners comparing a reference sample against
  samples under **ten jitter conditions**. **Number of listeners is NOT printed.**
- Result: **mean MOS 3.6, SD 0.23** across the ten conditions (5-point scale implied by MOS but not stated).
- No listening environment, equipment, or screening details are printed.

## Limitations acknowledged by authors
- PESQ/POLQA are explicitly described as having known weaknesses (clock drift, listening level, loudness loss,
  conversational delay, talker echo, sidetones) — ViSQOL is positioned as complementing, not replacing, them;
  one embodiment suggests using ViSQOL **in combination with PESQ to flag poor quality estimates caused by time warping**.
- The described embodiment is **narrowband**; wideband adaptation requires adjusting spectrogram parameters.
- The **NSIM → MOS transfer function is not yet developed** ("a transfer function *may be developed*").
- The warp-factor predictor is **symmetric** and cannot by itself tell faster from slower sampling; at extremes
  (1.15 predicted as 0.85) it fails on direction.
- Experiments focus on **constant** time warping; varying warps are only claimed to be "could also be handled".
- NSIM is stated to be **more sensitive to time warping than a human listener** (hence the warping stage).

## Relevance to our project
This file is the wrong document for a PESQ citation, but it is still highly usable for AudioCAPTCHA-DSP: it documents,
first-hand, **exactly where the ITU full-reference quality metrics we plan to run (PESQ/POLQA) break down** — small
temporal warps, resampling/clock drift and jitter — which is precisely the family of DSP distortions we can apply to
CAPTCHA speech. If our distortions include time-warping/resampling (a cheap, psychoacoustically subtle attack),
PESQ-derived scores may over-penalise a condition humans rate as transparent (their example: MOS 3.6 vs PESQ-LQO 3.33,
with PESQ "dropping significantly" once warp exceeds a few percent), so any Human–ASR Gap (HAG) number built on
PESQ would be biased; ViSQOL/NSIM-style patch-similarity, or a warp-robust metric, is the sanity check. The document
is also a ready-made argument for our evaluation methodology: it separates *intelligibility* from *aesthetic quality*
(the peak-clipping example), echoes the STOI-vs-quality split we already use, and shows an objective metric that
reports **0–1 similarity plus an explicit estimate of the distortion magnitude (warp factor)** — a pattern we could
copy for reporting "how much psychoacoustic distortion was actually applied" alongside STOI/MBSD/PESQ and ASR WER.
Finally, it is a concrete, citable instance of Google patenting a perceptually-grounded full-reference metric, i.e.
prior art for the metric family our framework will compare.

## Keywords
ViSQOL, PESQ limitations, POLQA, NSIM, neurogram similarity, full-reference metric, clock drift, time warping,
jitter, VoIP quality of experience, spectrogram patch alignment, RMSE alignment, speech quality vs intelligibility,
MOS prediction, US patent 9,524,733
