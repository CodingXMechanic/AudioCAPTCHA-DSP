# Human-Proxy Validation — checking our human score against real listeners

**Status:** analysis note supporting the artifact (not a paper draft).
Everything below regenerates from `scripts/validate_human_proxy.py`; the
per-file evidence lives in `results/validation/` (local — `results/` is
gitignored, per repo convention numbers are recorded here).

---

## 1. Purpose and scope

The paper title puts **human intelligibility** on an axis against ASR. Our
human axis is `hsr = clip(0.6 + 0.4 * stoi_proxy, 0, 1)`, a *proxy*: every
output it appears in is labelled
`illustrative (STOI-derived proxy; no human study conducted)` (`HSR_LABEL`).

This note documents two checks of that proxy against **actual human
judgments**:

- **Layer 1** — qualitative agreement with Paper 11's (Schönherr et al.)
  real listener study, the only published human study in our reference set.
- **Layer 2** — quantitative correlation of our exact proxy metric with
  per-utterance *word-intelligibility scores from real listeners* on the
  public TMHINT-QI corpus.

**What this does NOT do (non-claims):**

1. It produces **no human scores for our own conditions** — the
   `HSR_LABEL` stays exactly as it is.
2. It makes **no numeric comparison** against Paper 11's listener results
   (different corpus, language, task, listeners).
3. Layer 2's correlations are **never** presented as results on our
   corpus; they validate the *metric*, with stated caveats (Mandarin,
   ceiling, ~1.6 raters/file).
4. Nothing here is "better than the base paper" — their axis is real
   listeners, ours is a checked proxy.

---

## 2. Layer 1 — agreement with Paper 11's actual listener outcomes

Source: arXiv:1808.05665v2 verified verbatim (§V-A, §V-B, Table I) and
`results/comparative/main/summary.csv` (our λ-mirror run, seed 42).

| Question | Paper 11 — real listeners | Ours — STOI proxy, mirrored conditions |
|---|---|---|
| Does threshold-shaped hiding preserve the original speech? | Transcription WER: original **12.59 %** vs adversarial **12.61 %**; 22 listeners x 21 stimuli; two-sided t-test at 1 %: **no difference** (§V-A2) | `psychoacoustic.masked_noise` sweep: STOI **0.882** (margin 0) -> **1.000** (margin 50); HSR 0.953 -> 1.000 |
| Is the *no-threshold* (unconstrained) version perceptibly worse? | MUSHRA anchor (adversarial without hearing thresholds) rated **lowest in all nine cases**, significantly below every thresholded condition (one-sided t at 1 %; 30 listeners collected, 3 excluded -> 27 analysed; §V-B2) | power-matched control `control.no_hearing_threshold`: STOI **0.675** vs **0.882** at equal perturbation power (lambda 0) |
| Is the hidden target audible to humans? | "impossible to comprehend the target transcription"; WER vs target reaches 138.21 % (Table I, speech, 500 iters) | out of scope (our benchmark is untargeted cross-dWER; not re-tested) |

Our sweep rows (same table):

| condition | STOI | HSR | cross-dWER |
|---|---|---|---|
| `control.no_hearing_threshold` (power-matched) | 0.6751 | 0.8700 | 0.4976 |
| `masked_noise#margin=0` | 0.8819 | 0.9527 | 0.3618 |
| `masked_noise#margin=5` | 0.9509 | 0.9803 | 0.1047 |
| `masked_noise#margin=10` | 0.9818 | 0.9927 | 0.0451 |
| `masked_noise#margin=30` | 0.9998 | 0.9999 | 0.0085 |
| `masked_noise#margin=50` | 1.0000 | 1.0000 | 0.0018 |

**Agreement:** both studies find the same *structure* — threshold-shaped
distortions read as transparent on the human axis, while the unconstrained
control is measurably worse (their listeners: anchor significantly lower;
our proxy: 0.675 vs 0.882 at matched power). This is a directional
cross-check only. Their listener responses were never published and their
demo site (`adversarial-asr.selfip.org`) is offline, so per-utterance
re-scoring of their data is impossible — which is exactly why Layer 2
exists.

---

## 3. Layer 2 — quantitative check against real per-utterance listener scores

### 3.1 Corpus (TMHINT-QI)

Public release accompanying Chen & Tsao, *InQSS* (INTERSPEECH 2022):

- Mandarin HINT sentences (10 words each), degraded by 6 families
  (`Noisy`, `MMSE`, `FCN`, `DDAE`, `KLT`, `Trans`) x SNR {-2, 0, 2, 5} dB
  x noises {white, pink, babble, street}.
- Web-based listening test, **226 listeners** (110 with the provided
  headphone, 116 with their own).
- `raw_data.csv`: **24,408 rating rows** — subject index, method, SNR,
  noise, **quality 1-5**, **word-intelligibility 0-10**; 14,915 wav files
  (test 1,978 / train 12,937, 16 kHz mono).
- Rating density: ~1.6 listeners per file.

### 3.2 Method

- Rows with `method = None` (pretest) and `method = clean` (reference
  rated alone) excluded: 24,408 -> 22,148 rows; `STOI(clean, clean) = 1`
  would be tautological.
- Per file: human score = mean over raters; objective =
  `compute_stoi_proxy(clean_ref, degraded)` at 16 kHz — **the exact metric
  inside HSR** — plus reference STOI (`pystoi`, extended=False).
- Clean-reference pairing verified: 0 file/utterance mismatches; refs
  resolved in the file's own folder (test: 1,455) or from the paired
  folder's same-utterance clean file (315); 16 test files had no
  resolvable reference and were skipped (1,770 of 1,786 rated test
  files scored).
- Pairing-provenance control: mean proxy STOI is identical for
  own-folder vs cross-folder references (0.771 vs 0.774) — no pairing
  artefact.

### 3.3 Results

Pearson r / Spearman rho vs **mean human intelligibility**:

| split | metric | n | Pearson r (p) | Spearman rho (p) |
|---|---|---|---|---|
| **test** | our `stoi_proxy` (= HSR) | 1,770 | **0.336** (7.8e-48) | **0.329** (6.2e-46) |
| test | reference STOI (pystoi) | 1,770 | 0.473 (2.3e-99) | 0.388 (1.1e-64) |
| train | our `stoi_proxy` (= HSR) | 12,010 | 0.292 (5.3e-235) | 0.281 (3.1e-217) |
| train | reference STOI | 12,010 | 0.408 (p < 1e-300) | 0.348 (p < 1e-300) |
| **all** | our `stoi_proxy` (= HSR) | 13,780 | **0.297** (3.5e-278) | **0.287** (3.0e-259) |
| all | reference STOI | 13,780 | 0.415 (p < 1e-300) | 0.351 (p < 1e-300) |

Secondary (test split): proxy vs human **quality** r = 0.364 (p = 1.0e-56);
proxy vs reference STOI r = 0.349 (p = 8.9e-52).

`hsr` rows equal `stoi_proxy` rows exactly (affine map, clipping not
reachable for stoi in [0, 1]) — i.e. the table *is* the HSR validation.

### 3.4 Interpretation and caveats

- **The proxy tracks real listener judgments significantly** at every
  split (worst case p ~ 1e-235) and in the expected direction: clips
  listeners understood worse score lower on our proxy. This is the
  "human score was actually checked against humans" evidence behind the
  title.
- Magnitude is **moderate, not strong** — and the ceiling is
  expected here: on this corpus *no objective metric exceeds 0.8*
  correlation with human scores, and STOI is documented as suboptimal
  for intelligibility (WER and NCM rank higher) — see the corpus's own
  correlation study in `data/references/papers/notes/`. Reference STOI
  itself reaches only r = 0.41-0.47 under identical conditions; our
  lightweight proxy (STFT envelope correlation, no 1/3-octave banding or
  temporal context) sits just below the standard metric, which its
  "proxy" naming already says (proxy-vs-reference r = 0.35-0.38).
- Attenuators: heavy ceiling (test per-file mean intelligibility
  median 9.5/10, mean 8.45), ~1.6 raters/file (single-listener noise on
  the labels), remote uncontrolled listening, Mandarin content vs our
  English stand-in corpus.
- **Allowed claim:** "the HSR proxy correlates significantly with real
  listener word-intelligibility scores on a public human-rated corpus
  (test r = 0.336, n = 1,770, p < 1e-47; full r = 0.297, n = 13,780), so
  proxy trends are human-grounded in direction; HSR remains labelled
  illustrative for our own conditions."
- **Forbidden:** transferring these r values to our corpus; calling HSR a
  "human score" without the caveats; any numeric ranking against Paper
  11's listeners.

---

## 4. Reproduction

```powershell
# dataset: TMHINT-QI official release (Google Drive, linked from
# github.com/yuwchen/InQSS) -> data/references/tmhintqi/TMHINTQI/
pip install pystoi                               # optional reference metric
python scripts/validate_human_proxy.py --split test --jobs 4   # ~2 min
python scripts/validate_human_proxy.py --split all  --jobs 4   # ~9 min
# -> results/validation/human_proxy_per_file.csv
# -> results/validation/human_proxy_summary.csv
```

## 5. References (for the later citation pass)

1. Schönherr, Kohls, Zeiler, Holz, Kolossa — arXiv:1808.05665v2 (2018),
   §V-A transcription test, §V-B MUSHRA test, Table I.
2. Chen & Tsao — *InQSS*, INTERSPEECH 2022 (+ TMHINT-QI release;
   github.com/yuwchen/InQSS).
3. The TMHINT-QI objective-vs-subjective correlation study (survey note:
   `data/references/papers/notes/Study on the Correlation ...`).
4. Taal et al. (2011) — STOI (basis of `compute_stoi_proxy`).
