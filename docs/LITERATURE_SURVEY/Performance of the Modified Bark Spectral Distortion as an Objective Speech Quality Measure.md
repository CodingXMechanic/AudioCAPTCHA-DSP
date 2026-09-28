# Performance of the Modified Bark Spectral Distortion as an Objective Speech Quality Measure

## Full reference
- **Title:** Performance of the Modified Bark Spectral Distortion as an Objective Speech Quality Measure
- **Authors (as printed):** Wonho Yang, Majid Benbouchta, Robert Yantorno
- **Affiliation:** Speech Processing Lab, Electrical & Computer Engineering, Temple University, Philadelphia, PA 19122-6077 (emails: wonho@astro.temple.edu, mbenbouc@nimbus.temple.edu, ryantorno@nimbus.temple.edu)
- **Venue / year:** **Not printed in the paper body.** PDF metadata (Acrobat Distiller 3.01, "Electronic Manuscript"): Title = same as above; **Subject = "SP16 : Topics in Speech Coding II"**; Author = "Wonho Yang, Majid Benbouchta, Robert Yantorno"; CreationDate **1997-11-03**; ModDate **1998-04-08** → i.e. a 1997/1998 IEEE-style conference manuscript (session SP16, speech coding). No DOI/arXiv ID is printed.
- Predecessor work cited: W. Yang, M. Dixon, R. Yantorno, "A modified bark spectral distortion measure which uses noise masking threshold," IEEE Speech Coding Workshop, pp. 55–56, Pocono Manor, 1997.

## One-line contribution
Reworks Bark Spectral Distortion (BSD) into **MBSD**, which replaces BSD's empirical power threshold with a
**psychoacoustic noise-masking threshold** and its squared-Euclidean distance with an **average loudness difference**,
then reports which metric, frame size, speech class and spectral region make MBSD correlate best with MOS.

## Problem & motivation
- Subjective tests are expensive/time-consuming → need objective measures that correlate well with MOS.
- BSD is a strong perceptual candidate (assumes quality ≈ loudness: critical-band analysis + equal-loudness
  pre-emphasis + intensity–loudness power law) but uses an **empirically derived distortion threshold**.
- Inspired by transform coding of audio, **distortion below the noise masking threshold is inaudible** and should be
  excluded — motivated by coding-gain results showing spectral samples below the threshold need not be transmitted.
- Open questions the paper addresses: which distortion metric is "right" for the loudness domain, whether the
  correlation is better against **MOS** or **MOS difference**, sensitivity to frame size / speech class /
  spectral region.

## Method (precise but simple)
1. Both signals x(n) (input/"original" = 64 kbps PCM in the experiments) and y(n) (coded) go through
   **loudness calculation**: critical-band analysis → equal-loudness pre-emphasis → intensity-loudness power law
   (identical to BSD).
2. A **noise masking threshold** is estimated by critical-band analysis, spreading-function application and the
   absolute threshold of hearing (tone-masking-noise and noise-masking-tone both considered).
3. Per critical band i, define a binary **indicator of perceptible distortion M(i)**: M(i)=1 if the loudness
   difference exceeds the loudness of the noise-masking threshold, else 0.
4. **BSD (eq. 1):** BSD = (1/N) Σ_j Σ_i [ L_x(j,i) − L_y(j,i) ]² — average *squared Euclidean* distance of
   estimated loudness (N frames, K critical bands).
5. **MBSD (eq. 2):** MBSD = (1/N) Σ_j Σ_i M(i) · [ L_x(j,i) − L_y(j,i) ] — average *(non-squared) loudness
   difference*, only over perceptible (above-threshold) bands.
6. Five candidate metrics were tested by varying the norm n (1,2), normalisation (by average loudness of the
   original, by total loudness squared) and the metric itself (see Table 1).
7. Because objective measures are *comparison* measures while MOS is absolute, the authors estimate a
   **second-order regression** and report correlation against **MOS difference**.

## Key quantitative results
**Table 1 — correlation with MOS (row I) vs correlation with MOS difference (row II), by metric:**

| Metric | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| I (vs MOS) | 0.940 | 0.931 | 0.917 | 0.911 | 0.898 |
| II (vs MOS difference) | **0.956** | 0.946 | 0.938 | 0.931 | 0.906 |
- Metric 1 = eq. 2 (MBSD) with n = 1 (average difference of loudnesses); Metric 2 = eq. 2 with n = 2;
  Metric 3 = eq. 2 ÷ average loudness of original, n = 2; Metric 4 = eq. 2 ÷ total loudness² of original, n = 2;
  **Metric 5 = eq. 1 (BSD)**.
- "Depending upon the metric, the correlation coefficient could vary by **0.01 to 0.05**."
- ⚠ Internal inconsistency as printed: the prose says "the 5th metric in Table 1 (Metric I) showed the **highest**
  correlation coefficient", yet Metric 5 (BSD) is the **lowest** (0.898 / 0.906). Table-1 values are reported here as printed.

**Table 2 — BSD vs MBSD correlation coefficients:**

| | MIXED | FEMALE | MALE |
|---|---|---|---|
| BSD | 0.898 | 0.938 | 0.854 |
| **MBSD** | **0.956** | **0.969** | **0.945** |
(Used here had **no** empirically determined power threshold; MBSD more consistent across male/female speech.)

**Table 3 — MBSD correlation by frame size (samples) × speech class:**

| Class | 40 | 80 | 160 | 240 | 320 | 400 |
|---|---|---|---|---|---|---|
| VOICED | 0.956 | 0.956 | 0.955 | 0.954 | 0.954 | 0.953 |
| UNVOICED | 0.604 | 0.657 | 0.691 | 0.718 | 0.736 | 0.745 |
| TRANSITIONAL | 0.627 | 0.731 | 0.794 | 0.816 | 0.709 | 0.674 |
| NON-SILENT | 0.943 | 0.955 | **0.957** | 0.956 | 0.955 | 0.954 |
- Best setting recommended: **non-silent regions with 160-sample frames**; performance "not very sensitive" to
  frame size between 40 and 400 samples.

**Table 4 — correlation by spectral region (critical bands / frequency bandwidth in Hz / correlation coefficient):**
- **Low frequency:** bands 1–8, 100–1080 Hz → **0.554**
- **Mid frequency:** bands 9–13, 1081–2320 Hz → **0.926**
- **High frequency:** bands 14–18, 2321–4400 Hz → **0.953**
- **All bands:** 1–18, 100–4400 Hz → **0.956**
- Interpretation: **high-frequency region drives perceived speech *quality***, whereas the low-frequency region is
  important for *intelligibility*.

Experimental conditions for Tables 1–3 (exps. 1–2): frames of **160 samples**, **Hanning window**, **voiced frames
only**, dataset containing **MNRU distortions plus various speech coders**, **64 kbps PCM treated as original**,
second-order regression, MOS *difference* as target.

## Datasets / corpora used
- A speech data set containing **MNRU distortions** and **various different types of speech coders** (source of the
  MOS labels; supplied by Peter Kroon of Lucent Technologies, acknowledged — "original and coded speech and
  associated MOS scores"). No utterance/speaker counts are printed.
- 64 kbps PCM serving as the "original" reference condition.

## Models / systems evaluated
- **MBSD** (proposed; eq. 2 with the selected metric) vs **conventional BSD** (eq. 1, without the empirical power
  threshold in this comparison).
- Five distortion-metric variants (Table 1) in the loudness domain.
- Comparators referenced but not re-run: PSQM (Beerends & Stemerdink 1994), other perceptual quality measures.

## Human-study details
- No listening test was run by the authors themselves; they reuse **previously obtained MOS scores** (supplied with
  the coded speech by Peter Kroon, Lucent Technologies). **Number of listeners, protocol and scale are not printed.**
- All correlation figures are therefore "objective metric vs existing MOS / MOS difference" numbers.

## Limitations acknowledged by authors
- Results are called **"preliminary simulation results"** (twice) and "initial attempt to search for a proper metric".
- **"Currently, the validation of this metric is being examined with different speech databases"** — i.e. the chosen
  metric is not yet validated cross-corpus.
- The choice of distortion metric (squared vs absolute loudness difference) had "never been determined" for BSD; the
  metric search is bounded to "variation of the first and the second norms".
- Only voiced frames are processed in the first two experiments (unvoiced/LPC-degradation argument from prior work);
  unvoiced and transitional classes correlate much worse (0.604–0.816).
- The paper itself notes the unresolved MOS vs MOS-difference evaluation question.

## Relevance to our project
MBSD is the most explicitly **psychoacoustic** objective metric in our reference set: it is BSD plus a genuine
**noise-masking threshold**, i.e. it scores only distortion a human ear can hear — exactly the criterion behind
Schönherr-style psychoacoustic hiding, where an attacker hides adversarial energy under masking thresholds. For
AudioCAPTCHA-DSP this gives us two concrete tools: (a) an objective *quality* metric (MBSD ≈ 0.956 correlation with
MOS difference, 0.969 female / 0.945 male) to pair with **STOI** as the *intelligibility* proxy, and (b) a design
warning from Table 4: **high frequencies (2321–4400 Hz) carry perceived quality (r = 0.953) while low frequencies
carry intelligibility (r = 0.554)** — so CAPTCHA-distortion energy placed in high bands may look "clean" to
listeners yet still be transcribed by ASR, and vice versa. That frequency-dependent quality/intelligibility split is
a natural mechanistic explanation for a Human–ASR Gap and suggests reporting MBSD alongside STOI/WER when we
attribute HAG to a particular band. The paper's frame-size insensitivity (0.943–0.957 over 40–400 samples) also
means MBSD can be computed cheaply on short CAPTCHA utterances.

## Keywords
Modified Bark Spectral Distortion, MBSD, BSD, noise masking threshold, psychoacoustics, loudness, critical bands,
objective speech quality, correlation with MOS, MOS difference, frame size, speech classes, spectral regions,
high-frequency quality, perceptual distortion measure
