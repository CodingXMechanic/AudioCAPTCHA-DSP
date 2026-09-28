# Blind and Human: Exploring More Usable Audio CAPTCHA Designs

## Full reference
- Title: Blind and Human: Exploring More Usable Audio CAPTCHA Designs
- Authors (as printed): Valerie Fanelle, Aditi Shah, Sepideh Karimi, Bharath Subramanian, Sauvik Das — all Georgia Institute of Technology
- Venue: Sixteenth Symposium on Usable Privacy and Security (SOUPS 2020), August 10–11, 2020, Virtual Conference; USENIX Association; ISBN 978-1-939133-16-8; pp. 111–125.
- URL printed: https://www.usenix.org/conference/soups2020/presentation/fanelle . No arXiv ID/DOI printed.

## One-line contribution
Designs and evaluates four novel (mostly rule-based) audio CAPTCHAs — Math, Character, Pauses, Categories — showing in a three-session within-subjects study with 67 people with visual impairments that all four are significantly more accurate and faster than a standard alphanumeric control, while exposing a usability-vs-security trade-off against random-guessing and NLP adversaries.

## Problem & motivation
- 285 million people worldwide have visual impairments (39 million totally blind, 246 million low vision) and rely on audio CAPTCHAs.
- Usability gap is large: visual CAPTCHAs average **9.8 s / 93% success**, audio CAPTCHAs **51 s / 50% success**; a 2017 WebAIM survey of 1792 PVIs found **90%** rank audio CAPTCHAs somewhat/very difficult (2nd most problematic daily web issue after Adobe Flash).
- Root cause: audio CAPTCHAs are translations of visual designs, imposing impractical attention/memory load and creating screen-reader audio interference (typed letters read aloud while listening for the next character).
- Prior taxonomy: content-based (speech→text) vs rule-based (interpret/aggregate) challenges; rule-based designs reduce short-term-memory burden. Prior findings motivate the design: Meutzner et al. reported human success 2.8–3.9× machine success → focus on human cognition; Lazar et al. achieved ≥90% accuracy on sound-category identification but with only 20 co-located participants and no baseline.

## Method (precise but simple language)
- Four prototypes (Table 1): **Math** (rule-based; running sum of spoken single-digit add/subtract operations, e.g. "7+4-2-1" → 8), **Character** (rule-based; count occurrences of one character in an alphanumeric string, e.g. "6R169Y6" → 3), **Pauses** (content-based control variant; transcribe characters with 2-s gaps to avoid screen-reader interference), **Categories** (rule-based; count sounds belonging to a category, e.g. birds among robin/train/motor/owl/rooster → 3); plus a **Control** emulating the industry-standard alphanumeric CAPTCHA.
- Challenges perturbed per prior advice: speed (very slow → very fast), pitch (male/female voices), background-noise type (cafés, planes, wind), same decibel level across clips; 16–18 s long, 1-s lead-in pause (control has none), 1.25 s gaps judged optimal.
- Created with Audacity + text-to-speech clips plus open-source sound effects (bigsoundbank, Zapsplat); test-bed in jQuery/HTML5 + PHP on Heroku, screen-reader accessible.
- Design: controlled, randomized, within-subjects, **3 sessions one week apart**; per session 3 challenges × 4 prototypes + 1 control = 13 challenges, random order → 39 distinct challenges total; pilot in early 2019, main study summer 2019; IRB-approved.
- Analysis: random-intercepts logistic regression (accuracy) and linear regression (time) with lme4, participant + challenge random intercepts, covariates session number and age, pairwise `multcomp` comparisons with Bonferroni correction (Table 2).

## Key quantitative results
**Sample:** 225 outreach responses → 150 scheduled → **67 PVIs interviewed** (38 USA, 22 India, 2 Italy, 2 Germany, 2 Czech Republic, 1 South Africa); **33 completed all three sessions** (abstract; §5.1 says 34 continued to the remaining sessions). Mean age **33.1 (σ = 15.3)**.
**Data cleaning:** 2,259 attempts → 11 corrupted + 1 outlier (93 min) dropped → **2,247 attempts from 67 PVIs**.

**Task performance (RQ1):**
- Accuracy: **Math 89.2%, Character 86.9%, Pauses 76.2%, Categories 70.3%, Control 42.9%** (Table 5 rounds to 89 / 87 / 76 / 70 / 43%).
- Mean completion time: **Categories 31.1 s, Math 31.7 s, Character 32.7 s, Pauses 35.4 s, Control 53.6 s**.
- Mixed-effects accuracy coefficients vs control (Table 2): Math **+2.78** (p≤0.001), Character **+2.50** (p≤0.001), Pauses **+1.77** (p≤0.01), Categories **+1.53** (p≤0.05), session number **+0.35** (p≤0.05, learning effect); Categories vs Math −1.24 (p≤0.05) is the only significant pairwise design difference. (Body text quotes +1.85 for Pauses and +1.50 for Categories — small text/table discrepancy as printed.) Random-intercept variances: participant 0.50, challenge 0.61.
- Time coefficients (all p≤0.001): Math −0.73, Character −0.70, Pauses −0.61, Categories −0.76, session number −0.17; no significant time differences among the four prototypes.
- Extreme cases: best participant 100% correct, worst 46%; best challenge 97.5%, worst 17.2% (a control challenge).

**Security (RQ2):**
- Random-guessing adversary: control/Pauses search space 32^6 (impractical); Character & Categories outputs 0–10 → **1/11 ≈ 9%** success; Math has 100,016 inputs but 72 possible outputs → **1/72 ≈ 1%**, and always guessing "5" (normal distribution centered at 5) raises it to **≈3%**.
- NLP adversary (Google off-the-shelf speech recognition; a clip counts as broken if all entities parse): **Control 0/3 (0%), Math 2/9 (22%), Character 1/9 (11%), Pauses 6/9 (67%)**.
- Categories can't be parsed by off-the-shelf services, so the authors trained a TensorFlow parser on Google AudioSet (632 event classes, >2 million 10-s clips): over 48 sub-clips, 13 true positives, 16 true negatives, 8 false positives, 11 false negatives, average error **2.1% per clip** → **none of the Categories clips were fully parsed**.
- Summary: Control & Pauses best vs random guessing; Math, Character & Categories best vs NLP; all designs (incl. control) breakable by motivated adversaries → only suitable for low-risk contexts.

**Usability (RQ3):**
- Ratings on 1–5 Likert; 1,916 challenges rated "5" for usability, 1,865 for satisfaction → binarized and modeled with random-intercepts logistic regressions (Table 3).
- Preference over control: **Pauses 73%, Character 67%, Categories 61%, Math 52%**.
- Satisfaction scores: Character **4.93**, Pauses **4.85** (highest); heuristic Table 4 — session-1 accuracy (learnability): Control 32%, Math 83%, Character 89%, Pauses 72%, Categories 56%; avg time to answer correctly: 39.0 / 30.2 / 30.1 / 34.6 / 29.6 s; avg satisfaction: Control n/a, Math 4.6, Character 4.8, Pauses 4.8, Categories 4.5.
- Math accuracy improves 84.8% (session 1) → 92.8% (session 3).

## Datasets / corpora used
- Self-built CAPTCHA corpus: 39 distinct challenges built from TTS clips (0-9, a-z, "add/subtract/plus/minus") + open-source sound effects and background noises (Big Sound Bank, Zapsplat), created in Audacity.
- Google **AudioSet** (632 audio event classes, >2 million 10-s clips) for training the Categories-prototype sound classifier.
- Collected response data: 2,247 CAPTCHA attempts with accuracy, completion time, replay counts, and per-challenge 1–5 ratings.

## Models / systems evaluated
- Adversary 1: random guessing (analytic search spaces).
- Adversary 2: Google's off-the-shelf speech recognition (for Control, Math, Character, Pauses).
- Adversary 3 (for Categories): custom TensorFlow deep-learning audio-event classifier trained on AudioSet.
- Statistical models: mixed-effects (random-intercepts) logistic and linear regressions in R (lme4, multcomp, Bonferroni).

## Human-study details (if any)
- **67 blind/visually impaired participants** (all but one used screen readers; one used screen magnification; none used braille displays), recruited via American Foundation for the Blind, National Federation of the Blind, Braille Works, American Printing House for the Blind, Blind Graduate's Forum of India, VisionAid, and social/mailing lists; all verified proficient in English via Zoom screen-share in session 1.
- Compensation: **10 USD per completed session (30 USD total)** via regional Amazon gift certificates; 3 sessions one week apart; Zoom onboarding for session 1 (screen sharing to confirm screen-reader use).
- 13 challenges/session × 3 sessions, randomized; 2,247 usable attempts after dropping 12 (0.5%).
- Key numbers: 89.2% vs 42.9% accuracy (Math vs control); 31.1–35.4 s vs 53.6 s; all four designs significantly better (p ≤ 0.05 or better) on both accuracy and time; 73% of participants preferred even the Pauses variant over the control.

## Limitations acknowledged by authors
- **Participant retention:** 34 of the initial 67 did not complete the final two sessions; some procrastinated beyond the one-week spacing.
- **One-second-pause asymmetry:** prototypes begin with a 1-s pause, the control does not (to mimic real-world designs) — this may have inflated prototype-vs-control performance.
- **Ecological validity:** the test-bed was uniquely accessible; embedding the designs in otherwise inaccessible sites may change absolute accuracy/speed (relative results should hold); a field study is future work.
- **Intersectional accessibility:** designs assume no hearing impairment, assume mental-arithmetic ability (Math), and category membership can be culturally specific (is a rooster a bird?); no single design was a clear winner.
- Security is explicitly weak against motivated adversaries; not for security-critical applications.

## Relevance to our project
This paper supplies the *human* half of our evaluation framework in its most realistic form: a large PVI cohort with completion-time and accuracy ground truth, plus an explicit human-vs-machine comparison that is deliberately machine-side cheap (off-the-shelf Google STT and a single AudioSet classifier) rather than a controlled ASR benchmark. For our project it matters in three concrete ways: (1) it shows distortion is a *usability parameter*, not just a security one — pitch/speed/noise perturbations were applied to every clip yet Math/Character still reached 89%/87% accuracy, i.e. well-designed distortion preserved human intelligibility while (per their NLP tests) breaking naive speech parsing on Pauses (67% broken) — exactly the human-vs-ASR gap direction we quantify with STOI + Whisper/Vosk WER; (2) it hands us ready-made operational metrics for the human side (accuracy, completion time, replay count, Likert/SUS-style ratings) and a mixed-effects modeling recipe with participant- and challenge-level random intercepts, which is the right statistics for repeated-measures intelligibility experiments; and (3) its security analysis is a caution about *task type* rather than signal distortion — rule-based aggregation (counting) changes the answer space to 0–10 (9% random-guess success) rather than the 32^6 string space, reminding us that our HAG metric must be defined on a fixed task so that human and ASR numbers remain comparable across distortion conditions.

## Keywords
audio CAPTCHA, accessibility, people with visual impairments, screen readers, rule-based CAPTCHA, usability, completion time, accuracy, mixed-effects regression, NLP adversary, speech-to-text attack, AudioSet, sound-category recognition, SOUPS 2020, human-vs-machine gap
