# Gap Analysis — Base Paper & Literature Survey

**Base paper:** Schönherr, Kohls, Zeiler, Holz, Kolossa — *Adversarial Attacks
Against Automatic Speech Recognition Systems via Psychoacoustic Hiding*
(arXiv:1808.05665v2, 2018; the WHAT-REMAINS survey lists the venue as
IEEE/ACM TASLP — verify against the published version when the PDF is on
file; cited as Paper 11 in `WHAT-REMAINS.txt`, the recommended base).

This document lists every gap we identified in the base paper and in the
23-paper literature survey, and maps each gap to the concrete artifact in
this repository that addresses it. Every "evidence" pointer is a file or a
generated result — claims are not made without one.

---

## 1. Gaps in the base paper (Schönherr et al. 2018)

| # | Gap in the base paper | What they did | What we do | Evidence |
|---|---|---|---|---|
| G1 | **Single, white-box ASR** | One Kaldi DNN-HMM from the WSJ recipe; attacks optimized against it only | Two *real, independent* ASR families in the full matrix: **Whisper tiny** (attention encoder-decoder) and **Vosk small-en** (Kaldi nnet3 — same lineage as their model), extended to a **four-engine transfer validation** (+ **Whisper small**, **wav2vec2-base** SSL) on the top-8 attacks. Headline metric is **cross-family ΔWER**, i.e. black-box transfer | `asr/whisper_adapter.py`, `asr/vosk_adapter.py`, `asr/wav2vec2_adapter.py`, `experiments/comparative.py`; per-engine columns in `results/comparative/main/summary.csv` |
| G2 | **Undisclosed subset identity** | "70 utterances for 10 different speakers from one of the WSJ test sets" — no test set named, no utterance IDs, no target texts published | Protocol-exact reproduction (counts, selection procedure, ≤6 phones/s constraint) + **dataset manifest with corpus_sha256 and full ID lists**, so any reviewer re-runs the identical selection | `docs/BASE_PAPER_DATASET.md`, `evaluation/dataset.py` (`BASE_PAPER_SUBSETS`, `select_paper_subset`), `results/*/dataset_manifest.json` |
| G3 | **Human study not reproducible** | 21 samples × 22 listeners transcription test; MUSHRA with 9 samples | Same *condition layout* supported (MUSHRA-style conditions, 9-sample sets) but HSR reported as an explicitly **labelled illustrative STOI-derived proxy** — never a fabricated human result | `evaluation/hag_metrics.py` (`SecurityEvaluator`, `n_human_participants=0` path), `HSR_LABEL` in `experiments/comparative.py` |
| G4 | **λ-sweep only for their own attack** | λ ∈ {0…50} dB + "None" for their psychoacoustic hiding only | λ-grid **sweep across four psychoacoustic transforms** (masked_noise, bark_perturbation, temporal_masking, signal_threshold) *plus* a rigorously **power-matched "None" control** (same perturbation energy as λ=0, no masking structure) — the paper's "None" power level was not specified | `build_conditions`/`apply_no_threshold_control` in `experiments/comparative.py`; `fig2_lambda_sweep_mirror.*`, `lambda_sweep.*` |
| G5 | **One attack family vs. a whole space** | ~3 psychoacoustic/gradient attack variants | **119 registered transforms in 8 families (A–H)**, all ranked on the same two axes | `transforms/registry.py`, `results/*/summary.csv` |
| G6 | **No defense/preprocessing evaluation** | Attacks evaluated on raw audio only | 6 built-in ASR defenses (`LoudnessNorm`, `Resampling`, `SpectralDenoising`, `CodecSimulation`, `ReplaySimulation`, `Identity`) + channel-family transforms as defenders; defense-aware HAG fields (`dsr`, `cmfr`) | `asr/defense.py`, `evaluation/hag_metrics.py` |
| G7 | **No formal human-vs-ASR gap metric** | Reports WER and listener WER separately | Formal metric stack: **HSR, ASR-SR, HAG, CSS, DSR/CMFR**, plus Pareto flags, all computed per condition | `evaluation/hag_metrics.py` (`SecurityEvaluation`), `summary.csv` |
| G8 | **No cross-model transfer analysis** | Implicitly tied to their Kaldi model | Explicit **cross-family transfer figure + correlation** (Whisper ΔWER vs Vosk ΔWER, y=x reference) | `fig3_cross_family_transfer.*` |
| G9 | **No computational-cost axis** | Iteration counts mentioned only informally | Per-task timing captured in run progress; cost axis listed in the ranking spec (WHAT-REMAINS §8 "Lowest computational cost") as a reported dimension | `run_benchmark` progress log (`main_run.log`) |
| G10 | **No public reproducibility artifact** | No code release at publication | Full artifact standard: seeds, git commit, package versions, resumable rows, manifests | `results/*/run_manifest.json` |
| G11 | **Music corpus unnamed** | Subsets B/C use 70/72 music files from an undisclosed source | `music_dir` parameter with manifest disclosure; music omitted and *labelled omitted* when unavailable | `load_base_paper_subset`, `dataset_manifest.json` (`"music": "omitted …"`) |
| G12 | **Phone-rate constraint approximate** | "phone rate … based on the previous phone rate evaluation" (unspecified method) | `estimate_phoneme_count()` (≈0.96 phonemes/letter, 1.5/spoken digit) documented and labelled `estimated` | `evaluation/dataset.py`, `BASE_PAPER_DATASET.md` §3 |

## 2. What the literature survey (papers 1–23) does *not* provide

The survey spans CAPTCHA usability (1, 3, 4), adversarial ASR (2, 6, 7),
certification/statistics (8), defenses (9), phoneme-level robustness (10),
psychoacoustics (11–13), quality metrics (15–18), SSL representations
(19–21), and deepfake detection (22, 23). The common gap: **no work
combines them into a single, reproducible human-vs-ASR evaluation over a
diverse DSP transform space.** Specifically absent across all 23:

1. A **unified transform benchmark** ranking human intelligibility against
   ASR attack strength (this is WHAT-REMAINS contribution #1).
2. A **formal Human-ASR Gap (HAG) metric** with a defined protocol
   (contribution #3) — papers report WER and MOS/STOI separately.
3. **Cross-family transferability** quantified for *psychoacoustic*
   (imperceptible) attacks — paper 6 transfers gradient attacks on Whisper,
   paper 7 works in SSL feature space; neither tests whether
   threshold-shaped noise transfers across ASR architectures.
4. A **matched-power unconstrained control** for hearing-threshold attacks
   — the standard comparison (papers 2, 6, 11) uses unweighted noise or no
   control, which conflates *shaping* with *power*.
5. **Dataset provenance infrastructure** (manifest + content hash) for
   audio-CAPTCHA robustness results.

## 3. What we adopt from the survey (and where)

| Paper(s) | Adopted element | Where |
|---|---|---|
| 5 (Whisper) | Whisper as evaluation engine (tiny, deterministic decoding, offline weights) | `asr/whisper_adapter.py` |
| 6 (multi-objective attacks) | dual-objective view: WER degradation **+** perceptual cost | `comparative.py` ranks + Pareto |
| 8, 15 (confidence methods) | confidence intervals on aggregated metrics; t-based 95 % CIs in family summary | `make_figures.fig_family_summary`, stats phase |
| 9 (denoising vs attacks) | denoising/replay defenses as evaluated conditions | `asr/defense.py` |
| 10, 21 (phonetic granularity) | phoneme-guided attack + phone-rate subset constraint | `transforms/adversarial/search.py`, `estimate_phoneme_count` |
| **11 (base paper)** | **dataset/subset protocol, λ-grid, "None" column concept, WSJ/Kaldi basis** | `dataset.py`, `build_conditions`, `docs/BASE_PAPER_DATASET.md` |
| 12, 13 (Bark/MBSD) | Bark-scale masking + **MBSD** as perceptual metric (`mbsd` column everywhere) | `psychoacoustics/`, `evaluation/metrics.py` |
| 15 (STOI correlation) | STOI proxy as the human axis (chosen over PESQ per metric-validation evidence) | `compute_stoi_proxy` |
| 19 (wav2vec 2.0) | self-supervised family as an **evaluation engine in the transfer-validation run** (`wav2vec2-base`, greedy CTC, offline-cached weights) | `asr/wav2vec2_adapter.py` |
| 20, 21 (HuBERT / phonetic-vs-semantic) | phonetic-dominance rationale for the Kaldi-lineage engine and phoneme-granular transforms (larger SSL models *not* evaluated — stated) | `docs/IMPROVEMENTS_VS_LITERATURE.md` §E |

**Not adopted (declared):** PESQ (paper 16) — ITU code licence, replaced by
MBSD+STOI; MOS predictors (17, 18) — model weights not shippable offline;
SSL *feature-space* attacks (7) — deliberately **not adopted** (scope
decision: original novelty N11, shelf-life forecasting, prioritized
instead), recorded as future work (the SSL *representation* papers 19–21
are adopted on the evaluation side instead: wav2vec2-base as an engine in
the four-engine transfer-validation run); deepfake detection (22, 23) —
inverse problem, out of scope.

## 4. Residual gaps (honest, open)

- **Real human study** — participants require approval; HSR stays
  illustrative until then (protocol/layout ready, WHAT-REMAINS Phase 10).
- **Exact WSJ files** — LDC-licensed; adapter + `scripts/prepare_wsj.py`
  make runs bit-exact once licensed; headline runs use the labelled
  LibriSpeech stand-in.
- **The paper's 120 target texts** — private; our target texts are recorded
  in manifests instead.
- **Full four-engine matrix** — Vosk adds Kaldi-lineage evidence in the
  headline; Whisper small + wav2vec2-base add capacity and self-supervised
  transfer evidence on the top-K subset only (the 147-condition × 4-engine
  matrix was not run — cost declared); larger models (Whisper
  medium/large, wav2vec2-large, HuBERT/WavLM) remain future work, stated
  rather than half-implemented.
