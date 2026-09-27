# Base-Paper Dataset — Fidelity Contract

**Base paper:** Schönherr, Kohls, Zeiler, Holz, Kolossa, "Adversarial
Attacks Against Automatic Speech Recognition Systems via Psychoacoustic
Hiding," arXiv:1808.05665v2, 2018.

*(Citation and every fact in §1 re-verified against the arXiv v2 HTML on
2026-09-28. An earlier header in this file misnamed the title, authors,
and venue — corrected. Matches Paper 11 in `WHAT-REMAINS.txt`.)*

This document records *exactly* what dataset and subsets the base paper used,
what of that we reproduce, what cannot be reproduced bit-exactly (and why),
and how every benchmark run in this repository proves which dataset it used.
It is the reference for all "improvement over the base paper" claims in our
reports.

---

## 1. What the base paper used (verbatim facts)

| Fact | Paper location | Value |
|---|---|---|
| Corpus | §V-A setup | **Wall Street Journal (WSJ)**, "the default settings of the Wall Street Journal (WSJ) training recipe of the Kaldi toolkit" |
| Corpus properties | §V-A | phone-based read speech, > 80 hours training data, > 100,000-word dictionary |
| ASR system | §V-A | **single** Kaldi DNN-HMM, white-box, trained from the WSJ recipe defaults |
| Subset A | §IV-A | "a subset of **70 utterances for 10 different speakers** from one of the WSJ test sets" |
| Subset B | §IV-B, Table I | test set of speech = **72 samples**, test set of music = **70 samples** ("the test set of speech was the same as for the previous evaluations") |
| Subset C | §IV-D3 | "randomly chose speech files from **150 samples** and music files from **72 samples**"; only **audio–text pairs with a phone rate of ≤ 6 phones per second**; experiment repeated **100× per λ** for speech and music; target chosen from **120 predefined texts** |
| Phone-rate optimum | §IV-D2 | minimum WER at **4 phones/s**, hence the ≤ 6 phones/s constraint |
| λ (margin) sweep | Tables I–II | λ ∈ {0, 10, 20, 30, 40, 50} dB (+ "None" = no hearing thresholds), iterations 500/1000; §IV-D3 uses λ ∈ {0, 20, 40}, success = 0 % WER vs target, max 5000 iterations |
| Transcription test | §V-A1–A2 | **21 samples per listener** = 9 original + 9 adversarial (3 each at λ=0/20/40) + 3 difference signals; **22 listeners**; WER original 12.59 % vs adversarial 12.61 %; two-sided t-test at 1 % (no difference) |
| MUSHRA test | §V-B | **9 samples** (3 speech, 3 music, 3 bird twittering) × {λ=0/20/40, no-threshold anchor}; one-sided t-tests at 1 %; 30 listeners collected, 3 excluded → 27 analysed |
| Attack success + cost | Abstract, §VII | **up to 98 %** of targeted cases; **< 2 min** generation per 10 s audio file (6-core Intel Core i7-4960X) |
| WER over λ (speech, 500 iter) | Table I | None 2.14 %, 50 dB 6.96, 40 dB 11.07, 30 dB 16.43, 20 dB 36.43, 10 dB 92.69, 0 dB 138.21 — WER vs **target** text; their λ = allowed *excess above* threshold (mirrored vs our margin convention) |
| Perceptibility ϕ (speech, 500 iter) | Table II | None 10.11 dB, 50 dB 6.67, 40 dB 6.53, 30 dB 5.88, 20 dB 5.49, 10 dB 4.70, 0 dB 3.05 |
| Attack ablation | §IV-C | baseline WER 1.43 % (ϕ 11.62 dB) → +hearing thresholds 64.29 % (ϕ 7.04 dB) → +forced alignment at λ=20: 36.43 % (ϕ 5.49 dB) |

Notes:

- The paper writes digits doubled in the HTML rendering of some numbers
  (e.g. "7070 utterances", "1010 speakers"); these are the same single
  numbers as in the table above (70, 10, 120, 150, 72, 100, 80, 100,000).
- **Music is not part of WSJ.** The paper does not name its music corpus;
  subsets B and C therefore require a separately supplied music directory.

## 2. What this repository reproduces — exactly

| Item | Implementation | Status |
|---|---|---|
| Corpus | `WSJAdapter` (`evaluation/dataset.py`) loads **the same WSJ corpus**, preferring the Kaldi recipe's own test-set splits `test_dev93` → `test_eval92` | exact when WSJ is available (LDC license) |
| Layout | Kaldi recipe layout (`wav.scp`/`text`/`spk2utt`) *and* raw LDC tree (`*.wv1/*.wv2` + transcripts); `scripts/prepare_wsj.py` converts Sphere → wav and builds the Kaldi indexes | exact |
| Subset A protocol | `load_base_paper_subset("A")` / `select_paper_subset(total=70, n_speakers=10)`: 70 utterances, 10 speakers, deterministic round-robin | protocol-exact (IDs disclosed only in our manifest) |
| Subset B protocol | 72 speech (+70 music from user-supplied `music_dir`) | protocol-exact |
| Subset C protocol | 150 speech, seeded random draw; ≤6 phones/s audio–text pair constraint via `estimate_phoneme_count`/`phone_rate`; benchmark repeats mirror "100× per λ" as replicates | protocol-exact (phone counts *estimated*, see §3) |
| λ sweep | `configs/experiments/exp_07_psychoacoustic_lambda_sweep.yaml` mirrors λ ∈ {0,…,50} dB + no-threshold control | exact grid |
| Metrics | WER (same definition), perceptibility ϕ-style proxy, STOI intelligibility, MUSHRA-style condition layout | comparable (see §3 for HSR labelling) |

## 3. What cannot be bit-exact — and how it is disclosed

| Item | Paper | Ours | Rationale |
|---|---|---|---|
| Which WSJ test set | "one of the WSJ test sets" (unspecified) | prefers `test_dev93`, supports `test_eval92`; the actual choice is recorded in every run's `dataset_manifest.json` | paper does not disclose it |
| The 70 utterance / 10 speaker IDs | not disclosed | deterministic, seeded protocol selection; full ID list written to the manifest → any reviewer can re-run the identical subset | protocol-level reproduction is the strongest possible fidelity |
| The 120 target texts | not disclosed | benchmark target texts recorded in the manifest | cannot invent the paper's private list |
| Phone counts | "phone rate … based on the previous phone rate evaluation" | `estimate_phoneme_count()` (≈0.96 phonemes/letter, ≈1.5/spoken digit); labelled `est_*` in metadata | exact counts need CMUdict; the estimator is documented and conservative |
| Music corpus | not named by the paper | user-supplied `music_dir`, recorded in the manifest | WSJ contains no music |
| ASR | single white-box Kaldi DNN-HMM | **Whisper (tiny) + independent second ASR family** (multi-ASR, black-box view) | deliberate improvement over the base paper, documented in GAP_ANALYSIS |
| Human study | 21×22 listeners, in-lab | STOI-based proxy **labelled "illustrative — no human study"**; MUSHRA-style condition layout provided for a real study | no participant approval; never fabricate HSR |
| Corpus used for headline runs (pre-license) | WSJ | **LibriSpeech test-clean stand-in** with *identical subset geometry* (10 speakers × 7 utterances = 70 via `select_paper_subset`), every row labelled `corpus=librispeech_test-clean (stand-in)`, `protocol=base-paper-A` | WSJ is LDC-licensed and not redistributable |

## 4. Provenance rule (enforced by the benchmark)

Every benchmark run writes `results/<run>/dataset_manifest.json` containing:

```json
{
  "corpus": "librispeech_test-clean (stand-in for WSJ: LDC-licensed)",
  "base_paper": "arXiv:1808.05665",
  "protocol": "base-paper-A (70 utterances / 10 speakers / 7 each)",
  "adapter": "LibriSpeechAdapter",
  "root": "data/raw/LibriSpeech/test-clean",
  "selection": {"mode": "balanced", "total": 70, "n_speakers": 10, "seed": 42},
  "speakers": ["…"],
  "utterances": [{"id": "…", "speaker": "…", "duration": 1.62, "transcript": "…"}],
  "corpus_sha256": "<hash over sorted utterance ids + transcripts>",
  "phone_constraint": "estimated ≤6 phones/s where target texts apply",
  "created": "ISO-8601"
}
```

Any result table/figure produced without a manifest entry is not a result.
Conclusions in `docs/COMPARATIVE_ANALYSIS` cite the manifest alongside the
paper tables, so "better/worse than the base paper" statements are traceable
to a specific, re-runnable dataset selection.

## 5. Re-running on the exact WSJ corpus

```bash
# after obtaining LDC93S6B/LDC94S37A (or pointing at a Kaldi wsj recipe dir)
python scripts/prepare_wsj.py --check   /path/to/wsj --subset test_dev93
python scripts/prepare_wsj.py --convert /path/to/wsj --out /path/to/wsj_wav
python scripts/prepare_wsj.py --build-kaldi /path/to/wsj_wav --out data/raw/wsj/test_dev93
python scripts/prepare_wsj.py --subsets data/raw/wsj
```

The adapter then reports `corpus=wsj`, subset A/B/C selections are computed
with the same protocol, and manifest hashes let reviewers diff our run
against the stand-in run.
