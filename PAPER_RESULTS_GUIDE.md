# Paper Results Guide

How to turn this repository's artifacts into a defensible research paper —
which artifact supports which claim, how to read every column, and the
wording rules that keep claims honest. WHAT-REMAINS §15.

---

## 1. Evidence → artifact map

| Claim (from `docs/NOVELTIES.md`) | Evidence artifact | Paper element |
|---|---|---|
| Unified benchmark over 119 transforms / 8 families | `results/comparative/main/summary.csv` + `dataset_manifest.json` | §Experiments, main table |
| Dual-axis ranking + Pareto | `tables/ranking_full.*`, `fig1_pareto_human_vs_asr` | Main result figure |
| Cross-family transfer of psychoacoustic attacks | `fig3_cross_family_transfer`, `cross_delta_wer` column | Transfer analysis |
| Matched-power "None" control | `fig2_lambda_sweep_mirror`, control row in `lambda_sweep.*` | λ-sweep analysis |
| λ-sweep across 4 transforms | `tables/lambda_sweep.*`, `fig2_*` | Base-paper comparison |
| Formal HAG metric stack | `hsr/asr_sr/hag/css` columns; `evaluation/hag_metrics.py` | Metrics section |
| Family-level signatures | `tables/family_summary.*`, `fig4_family_summary` | Analysis |
| Reproducibility standard | `run_manifest.json`, `dataset_manifest.json`, `rows.csv` | Reproducibility statement |
| Four-engine transfer validation (top-K) | `results/comparative/engine_validation/summary.csv`, `tables/validation/top10_attack.*` | Robustness / appendix table |
| Gaps vs base paper | `docs/GAP_ANALYSIS.md` (G1–G12) | Related work / contributions |
| **Shelf-life forecasting (novelty N11)** | `results/shelf_life/shelf_life.{json,md}`, `fig16_shelf_life_forecast`, `rows.csv` (capacity ladder) | New-method + forecasting result |

**Gating rule:** a claim may enter the paper only when its evidence file
exists for the *final* run (not `smoke*` outputs). `summary.json.created`
and `run_manifest.json.git_commit` identify the run being cited.

## 2. Reading the outputs

### `rows.csv` (long format — ground truth)
One row per (condition, utterance): `condition_id, kind, transform_key,
family, params_json, margin_db, utt_id, spk, dur_s, status, error,
snr_db, stoi_proxy, mbsd, si_sdr_db, rms_ratio, ref_text,
wer_whisper_tiny, wer_vosk_small_en, hyp_*, hyp_vosk_small_en`.
`status=error` rows explain exclusions; `n_err` per condition is always
published — never silently drop rows.

### `summary.csv` (one row per condition)
| Column | Meaning |
|---|---|
| `n_ok / n_err` | sample counts incl. failures |
| `stoi_mean/std` | human axis (mean over utterances) |
| `snr_mean, mbsd_mean, si_sdr_mean` | fidelity/quality axis (finite values only) |
| `wer_<engine>_mean` | absolute WER per engine |
| `wer_<engine>_delta` | ΔWER vs that engine's `original` baseline |
| `cross_delta_wer` | **attack axis** = mean ΔWER over the run's independent engines (family-macro when one family has >1 engine) |
| `hsr, asr_sr, hag, css` | SecurityEvaluator outputs (HSR = illustrative proxy) |
| `human_rank` / `attack_rank` / `gap_rank` / `quality_rank` | WHAT-REMAINS §8 views |
| `pareto` | nondominated on (STOI, cross-ΔWER) |
| `hsr_label` | the mandatory label text |

### Figures
| File | Paper use |
|---|---|
| `fig1_pareto_human_vs_asr` | headline: human-vs-ASR plane + front |
| `fig2_lambda_sweep_mirror` | base-paper comparison (ΔWER + STOI vs λ, control line) |
| `fig3_cross_family_transfer` | black-box transfer (y = x = perfect transfer) |
| `fig4_family_summary` | family structure, mean ± 95 % CI |
| `fig5_taxonomy_overview` | taxonomy overview (registry + run coverage) |
| `fig6_strength_and_gap_curves` | WER / human success / HAG vs strength |
| `fig7_quality_vs_degradation` | MBSD (quality) vs ΔWER |
| `fig8_parameter_model_heatmaps` | λ × transform and transform × engine heatmaps |
| `fig9_speaker_variability` | speaker × family ΔWER variability |
| `fig10_word_confusion` | word-level confusion matrix |
| `fig11_spectrograms` | original vs transformed spectrograms |
| `fig12_masking_thresholds` | ATH + masking threshold + λ margins |
| `fig13_metric_correlation` | metric correlation matrix |
| `fig14_forest_effect_sizes` | forest plot: effect sizes ± 95 % CI |
| `fig15_rank_stability` | ranking stability under bootstrap |
| `fig16_shelf_life_forecast` | novelty N11: gap-vs-capacity scaling |

PDF, PNG (300 dpi) and SVG are written for every figure.

### WHAT-REMAINS §12 coverage matrix

| §12 item | Status |
|---|---|
| Transform taxonomy overview | `fig5` |
| WER versus transformation strength | `fig6a` (+ `fig2` per-target) |
| CER versus transformation strength | **not captured** — WER only (CER absent from `ROW_FIELDS`); documented gap |
| Human success versus transformation strength | `fig6b` |
| Human-ASR gap curves | `fig6c` |
| Perceptual quality versus ASR degradation | `fig7` |
| Pareto frontiers | `fig1` |
| Heatmaps across transform parameters and ASR models | `fig8` |
| Speaker-level variability plots | `fig9` |
| Phoneme-level vulnerability plots | **not supported** — needs forced alignment/phonemizer; documented gap (`LIMITATIONS.md`) |
| Confusion matrices | `fig10` (word level; phoneme level blocked as above) |
| Original/transformed spectrograms | `fig11` |
| Bark-scale masking plots | `fig12` |
| Masking thresholds and perturbation margins | `fig12` |
| Metric correlation matrices | `fig13` |
| Defense recovery plots | **on demand** — headline run has defenses off (`RQ6`); documented gap |
| Cross-model transfer plots | `fig3` |
| Runtime and cost comparisons | **not captured** — per-task timing absent from rows; documented gap |
| Forest plots for effect sizes | `fig14` |
| Confidence interval plots | `fig4`, `fig14` (CIs in `fig16` band) |
| Ranking stability under bootstrap resampling | `fig15` |

### Tables
`tables/*.csv` for further analysis, `tables/*.tex` (booktabs,
`table*`) drop straight into LaTeX; captions already contain the protocol
sentence and the HSR-label sentence (`HSR_NOTE`) — **keep them**.

## 3. Wording rules (scientific integrity)

1. **Human numbers:** always "illustrative (STOI-derived proxy; no human
   study conducted)" on first mention and in every caption containing HSR.
   Never "listeners", "heard", "participants" for proxy values.
2. **Corpus:** "protocol-exact stand-in (LibriSpeech test-clean) for the
   base paper's WSJ corpus (LDC-licensed)" — never "on WSJ".
3. **Engines:** name both families and their lineage; never generalize to
   "ASRs" beyond the two evaluated.
4. **Heuristic engine:** if mentioned at all, "metadata-driven heuristic
   proxy, excluded from quantitative claims".
5. **Base-paper comparison:** qualitative reproduction claims ("our λ
   curves reproduce the paper's trend") are allowed; numeric equality claims
   require an actual WSJ re-run (see `docs/BASE_PAPER_DATASET.md`).
6. **Statistics:** report n + effect size + CI with any p-value; apply
   BH correction for multiple comparisons (`STATISTICAL_ANALYSIS.md` §5).
7. **Smoke outputs** are pipeline checks — never citable numbers.

## 4. Suggested paper skeleton

**Working title:** *Systematic Evaluation of Psychoacoustic Distortions on
Human Intelligibility vs Automatic Speech Recognition: A Framework for Secure
Audio Verification.*

1. **Intro** — human–machine gap as CAPTCHA security premise (papers 1/3).
2. **Related work** — survey A–F sections (`IMPROVEMENTS_VS_LITERATURE.md`).
3. **Threat model** — `THREAT_MODEL.md` (K0/K1/K2 cells).
4. **Method** — transform taxonomy (A–H, `TRANSFORM_CATALOG.md`), dataset
   protocol + manifests, metric stack (HSR/HAG/CSS), ranking definitions.
5. **Experiments** — engines (`ASR_MODELS.md` incl. the capacity ladder),
   conditions, statistics.
6. **Results** — fig1 → table (ranking) → fig3 (transfer) → fig2 (λ-sweep +
   matched-power control) → fig4 (families) → fig16 (shelf-life forecast,
   novelty N11).
7. **Limitations** — paste-adapted from `LIMITATIONS.md`.
8. **Reproducibility** — from `REPRODUCIBILITY.md` §1–4.

## 5. Contribution checklist (WHAT-REMAINS §16)

Retain a claim only with its evidence column:

| Contribution claim | Status |
|---|---|
| 1. Unified benchmark (human vs ASR over broad transform space) | evidence: main run artifacts |
| 2. Comprehensive taxonomy + scientific organization | evidence: `TRANSFORM_CATALOG.md` (119/8) |
| 3. Formal HAG metric + protocol | evidence: `hag_metrics.py` tests + summary columns |
| 4. Cross-model & defense-aware analysis | evidence: `fig3`; defense metrics implemented (defense sweep optional) |
| 5. Multi-objective ranking/Pareto | evidence: rank columns + `fig1` |
| 6. Psychoacoustically constrained family + separation vs control | evidence: λ-sweep + control rows |
| 7. Phonetic/representation-level explanation | partial: phoneme-guided transform + phone-rate protocol; representation axis via the 4-engine transfer validation (SSL *attack* feature space, paper 7, still out of scope) |
| 8. Reproducible artifact & manifest standard | evidence: manifests + resume + full test suite |
| 9. **Shelf-life forecasting (novelty N11, beyond §16)** | evidence: `results/shelf_life/shelf_life.{json,md}` + `fig16` + `test_shelf_life.py` |

## 6. One-command regeneration

```powershell
python scripts/run_comparative_benchmark.py --transforms all --workers 4 --out results/comparative/main
python scripts/run_shelf_life.py --out results/shelf_life --workers 2
python scripts/make_figures.py --run results/comparative/main --out results/figures --shelf-run results/shelf_life
python scripts/make_tables.py  --run results/comparative/main --out results/tables
python scripts/generate_catalog.py
python -m pytest tests/ -q
```
