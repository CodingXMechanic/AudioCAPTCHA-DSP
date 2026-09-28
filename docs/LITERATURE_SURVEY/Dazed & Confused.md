# Dazed & Confused: A Large-Scale Real-World User Study of reCAPTCHAv2

**Classification note (assigned as "topic uncertain"):** This is **not** an audio-CAPTCHA or speech/DSP paper. It is a large-scale *usability / user-study + cost-and-security analysis* paper on the visual/behavioral **reCAPTCHAv2** system (checkbox + image challenges), with a perfunctory literature-summary of the audio challenge (unCaptcha) inside its security-analysis section. It contains no psychoacoustics, no speech corpus experiments, and no ASR evaluation of its own.

## Full reference
- Title: Dazed & Confused: A Large-Scale Real-World User Study of reCAPTCHAv2
- Authors (as printed): Andrew Searles (searlesa@uci.edu), Renascence Tarafder Prapty (rprapty@uci.edu), Gene Tsudik (gene.tsudik@uci.edu) — all UC Irvine
- Venue as printed: arXiv preprint **arXiv:2311.10911v2 [cs.CR], 21 Nov 2023**. No conference/journal venue or DOI printed on the PDF.

## One-line contribution
A 13-month, real-world, unwitting-participant study of reCAPTCHAv2 (>3,600 users, 9,141 valid solves) quantifying solving-time, SUS usability and preference differences between checkbox and image challenges, plus a cost analysis (819 million human hours ≈ $6.1B) and a security analysis leading to the recommendation that reCAPTCHA v2 be deprecated.

## Problem & motivation
- CAPTCHAs have been used since ~2003 and an arms race has run ever since; bots reached >99% accuracy on distorted text by 2014, and both checkbox and image stages of reCAPTCHAv2 were defeated by 2016 — yet reCAPTCHA is deployed on 13+ million websites (2023).
- Prior user studies have methodological weaknesses: MTurk data-quality issues, biased (informed) participants, mocked-up CAPTCHA types, and only two prior studies (2019/2023) of reCAPTCHAv2 — one with only 40 participants and unclear methodology, no usability.
- Missing: multiple solving attempts per person, service/context effects, education level and major effects, and unbiased/unaware participants on a real service.

## Method (precise but simple language)
- Insert reCAPTCHAv2 (Google "easy" difficulty setting) into a live UC Irvine SICS (School of Information & Computer Sciences) account-creation and password-recovery workflow; participants must be on campus VPN with a university account → data are real humans, mostly students, unaware of the study (IRB-approved with deception/incomplete-disclosure and consent-waiver filings).
- Run continuously for ~13 months (2022–2023), timed via JavaScript `Date` (ms precision): solving time = captcha render → successful validation response from Google (includes image challenges and failed attempts).
- A JavaScript directory crawler enriches each email with major and education level (freshman…graduate); no PII used in analysis.
- Data cleaning: 9,169 raw records → remove 28 with solving time > 60 s → **9,141 valid records**, of which 8,915 map to **3,625 unique participants** (3,573 unique students / 8,631 challenges used for education & major analyses).
- Checkbox-vs-image split: because behavioral accuracy is 80% (Google dashboard), the fastest 80% of solves are treated as checkbox-only; split point ≈5 s.
- Statistics (scipy): Shapiro–Wilk (non-normal, p<0.001), skewtest (right-skewed, p<0.001), tailedness test (heavy-tailed, p<0.001), Brown–Forsythe (unequal variances), Kruskal–Wallis with Holm–Bonferroni for equality of means across modes, services, attempts, majors, education levels.
- Post-study survey: 800 randomly selected participants emailed, **108 completed**, incentive $5 Amazon gift card; System Usability Scale (SUS) for checkbox and image, custom 5-point annoyance→pleasant scale, open-ended feedback (word clouds).
- Cost analysis: extrapolates 100 million reCAPTCHAs/day, mean solve times (v1 9.8 s, v2 3.53 s), US federal minimum wage $7.5/h, network overhead measured with Chrome DevTools/pingdom/webpagetest (.har files), energy 0.06 kWh/GB, EPA/EIA 1 kWh = 1–2.4 lb CO2.
- Security analysis: literature-based review of clickjacking, cookie farming, image-solving and audio-solving attacks on reCAPTCHAv2 and v3.

## Key quantitative results
**Solving time (Table 3, seconds):**

| Mode | Count | Mean | Median | Std | Max | Min |
|---|---|---|---|---|---|---|
| behavior (checkbox) | 7,334 | 1.85 | 1.67 | 0.71 | 4.99 | 0.51 |
| image | 1,807 | 10.3 | 8.20 | 6.54 | 59.8 | 4.99 |
| total | 9,141 | 3.53 | 1.83 | 4.50 | 59.8 | 0.51 |

- Image solving is a **557% increase** over checkbox.
- Google dashboard over the whole study (Table 2): 7,629 no-CAPTCHA (checkbox) passes, 1,890 passed images, 143 failed images, 9,538 total sessions, 19 failed sessions → **image accuracy 92.96%**, **behavior accuracy 79.98%**.
- Service context (Tables 4/6): checkbox mean 1.67 s (password reset, n=2,654) vs 1.96 s (account creation, n=4,680) → account creation **17% slower**, Kruskal–Wallis **p = 1.1e−115**; total times 2.63 s vs 3.97 s (**p = 6.7e−162**); no significant service effect on image times.
- Attempts (Table 7): 1st checkbox attempt 2.02 s (n=2,888) vs 10th 1.56 s (n=112) — contributions state the first attempt is **35% slower than the 10th**; statistically significant improvement from attempt 1 onward (p<0.001); average 3.52 attempts/checkbox user, 1.73/image user; max 37 (checkbox) / 20 (image) attempts. Image attempts show no significant differences.
- Education level (Table 11, total time): freshman 5.15 s, sophomore 4.33 s, junior 3.09 s, senior 2.85 s, graduate 3.82 s → freshmen are **80% slower than seniors**, all pairwise differences significant (Figure 12).
- Majors (Table 14): 22 of 62 majors shown (others had <20 sessions); Computer Science total mean 3.19 s vs Informatics (IN4MATX) 4.14 s; only 8 majors differ significantly; STEM faster than non-technical (minor trends with statistical significance).
- Usability (Table 15, SUS): **checkbox-only 78.51**, **checkbox in checkbox+image 76.21** ("Good" on the adjective scale, 71.4 = Good), **image 58.90** ("OK", 50.9 = OK); summary quotes averages of 77 and 59, max observed 90.
- Preference (Figures 14/15): 61.9% (checkbox-only) and 54.6% (checkbox in combo) find checkbox pleasant/very pleasant vs **30.3%** for image; **40% found image annoying/very annoying** (12.1% + 28.8%) vs the contributions' "<10%" for checkbox (figure shows 1.5% + 9.1% = 10.6%). Converted 5-point ratings: checkbox 3.62 (only) / 3.68 (combo), image 2.84. Word clouds: "easy"/"simple" for checkbox, "annoying" for image.
- Cost analysis (Section 6.1, stated as generous lower bounds): **at least 512 billion reCAPTCHA sessions**, 2.95 trillion seconds = **819 million hours** of human time = **at least $6.1 billion USD** in free wages (v1: 183 billion sessions, 497 million hours, $3.7B; v2: 329 billion sessions, 322 million hours, $2.4B); **134 Petabytes** of bandwidth → **~7.5 million kWh** → **7.5 million pounds of CO2** (range 7.5–18 million); of 329 billion v2 sessions, **65.8 billion image challenges** and (as printed) "263.2 million would have been checkbox challenges" → **250 billion labeled-data challenges**, worth **$8.75–32.3 billion per sale** at Google's $35–129/1,000-items rate; tracking-cookie lifetime value put at **$888 billion** ($2.7 per cookie × 329 billion).
- Network overhead (Appendix B, Table 26, KB): first page load 408.5, subsequent 29.319, checkbox click 24.43–41.77, image load 64.03–96.72, correct solution verification 0.6, wrong + new image 41.58, expiration 29.
- Human vs bot (Table 17): checkbox human 1.85 s / 80% (and 3.1–4.9 s / 85% from ref [63]) vs bot **1.4 s / 100%** [66]; image human 10.4 s / 93% (16–26 s / 81% in [63]) vs bot **17.5 s / 85%** [49].
- Security survey numbers quoted: cookie farming yields 63,000 valid cookies/day/IP, checkbox successes 9 days after cookie creation, **52,000–59,000 checkbox solves/day/IP at 100% accuracy, 1.4 s average**; Sivakorn's own image solver 70.8% @ 19.2 s; Hossen et al. 85% @ 17.5 s; **audio**: unCaptcha solves reCAPTCHA audio challenges with **85.15% accuracy in 5.42 s** using Google's own speech recognition; reCAPTCHA v3 broken by RL attack with .9+ scores, **97% accuracy from only 2,000 training points**.
- Headline conclusion: "reCAPTCHA v2 and similar reCAPTCHA technology should be deprecated."

## Datasets / corpora used
- Live production logs of the UC Irvine SICS account-creation / password-recovery service: 9,169 raw → 9,141 valid reCAPTCHAv2 solving-time records with timestamps, service type, student ID (via directory crawler), major and education level.
- Google reCAPTCHA admin-dashboard CSV exports (daily counts, scores, response times) over the study period.
- Post-study Google-Forms survey (108 responses) with SUS answers and open-ended feedback.
- Black-box network captures (.har) of reCAPTCHA page loads for the cost analysis.

## Models / systems evaluated
- **reCAPTCHAv2** (Google) in "easy" mode: behavioral checkbox stage + image-labeling fallback (image bounding-box/label tasks); audio challenge discussed only via cited work.
- No models are trained/evaluated by the authors; the security section summarizes third-party attacks: Sivakorn et al. 2016 (cookie farming), Hossen et al. 2020 (object-detection image solver), Bock et al. 2017 unCaptcha (audio/STT), Akrout et al. 2019 (RL against v3), Homakov 2014 (clickjacking).
- Tools: Chrome DevTools, pingdom.com, webpagetest.org, scipy stats, Jitbit mouse macro + Playwright headless Chrome for a brief automation-detection check (macro not flagged; Playwright headless always served an image challenge on first request).

## Human-study details (if any)
- **>3,600 distinct users**: 3,625 unique participants (8,915 of 9,141 valid records), plus 52 unique non-students (231 submissions) excluded from education/major analyses; 13 months, unaware participants on a real service (no recruitment, no consent beforehand, IRB deception approval; all debriefed by email afterwards).
- Demographics of the main experiment are unknown; the 108 survey respondents (62 male / 44 female / 2 non-binary; 87.04% under 25; 53.7% undergraduate, 46.3% graduate; 82.4% US residents) are argued to resemble the campus population (total ≈36,000 students: 54% female, 44.6% male; 82% aged 18–24).
- Survey protocol: 800 invited, 108 completed, $5 gift card for ~5 minutes; SUS (10 statements) for checkbox and image plus preference scale and word-cloud feedback.
- Key numbers: mean solve 1.85 s (checkbox) vs 10.3 s (image); SUS 78.51/76.21 vs 58.90; 40% find images annoying; first attempt 2.02 s vs 10th 1.56 s.

## Limitations acknowledged by authors
- Population is narrow (one university school, mostly 18–25-year-old students) — argued to be an "optimistic" (tech-savvy, least CAPTCHA-allergic) segment, not globally representative; a multi-university study would be more valuable but impractical.
- Google denied the request for large-scale reCAPTCHA data; building their own service was deemed infeasible.
- Checkbox/image splitting is *inferred* from the 80% behavioral-accuracy figure rather than directly logged.
- Network proximity of on-campus/VPN participants may make times faster than global real-world values; exact number of remote participants unknown.
- 9,538 dashboard sessions vs 9,169 supplied records (369 incomplete/errored sessions); 28 records with >60 s times dropped as high-variance.
- Cost figures are self-described estimates ("generous lower bounds"), and the labeled-data split sentence ("65.8 billion image / 263.2 million checkbox") is internally inconsistent as printed.
- The custom preference scale may bias toward the word "annoying" (authors argue against this using the word clouds).

## Relevance to our project
This paper is our *methodological* reference for the human side of a Human–ASR Gap study rather than a source of audio results: it demonstrates how to obtain unbiased human performance data at scale (unwitting participants, live service, ms-precision timing, deception-approved IRB) and how to combine objective measures (solve time, success rate) with validated subjective instruments (SUS with Bangor adjective scaling, preference scales, word clouds) — a reporting standard our human-intelligibility arm should emulate alongside STOI-derived proxies. Two of its findings transfer directly to audio CAPTCHA design and evaluation: (1) *context and repetition materially change human numbers* (password recovery 17% faster than account creation, p = 1.1e−115; first attempt 35% slower than the tenth), so our human-vs-ASR comparison must control for task context, practice effects and attempt order, ideally with mixed-effects models; and (2) its human-vs-bot table (checkbox 1.85 s/80% humans vs 1.4 s/100% bots; image 10.4 s/93% vs 17.5 s/85%) is a ready-made worked example of a *negative* gap — machines already outperform humans — which is the failure mode audio CAPTCHA designers (and our psychoacoustic-distortion study) try to reverse, and it also records the one audio datapoint in the paper: unCaptcha's 85.15% success in 5.42 s against reCAPTCHA's audio challenge, a useful historical ASR-side baseline for our Whisper/Vosk measurements.

## Keywords
reCAPTCHAv2, CAPTCHA usability, user study, System Usability Scale, solving time, unwitting participants, field experiment, checkbox challenge, image challenge, cost analysis, environmental impact, security analysis, audio challenge (unCaptcha), human-vs-bot performance, arXiv 2311.10911
