# Project Status

Status of every WHAT-REMAINS.txt phase. Labels follow §15:
**Implemented / Partially implemented / Experimental / Optional / Planned /
Not validated / Not supported.**

*Baseline for regression counting: 488 tests green (`pytest tests/`).*

---

## Phase-by-phase

| # | Phase | Status | Evidence |
|---|---|---|---|
| 1 | Audit the existing repository | **Implemented** | Audit performed; prior gaps documented in `docs/GAP_ANALYSIS.md` |
| 2 | Scientifically precise experiment model | **Implemented** | `core/signal.py`, `core/types.py`, `experiments/runner.py` (see `METHODOLOGY.md` §1) |
| 3 | Comprehensive DSP taxonomy (A–H) | **Implemented** | `transforms/registry.py` — **119 transforms**: A 11, B 16, C 11, D 24, E 20, F 9, G 10, H 18; `TRANSFORM_CATALOG.md` (all cited) |
| 4 | Real dataset layer | **Partially implemented (by licence)** | WSJ adapters + `scripts/prepare_wsj.py` implemented; **WSJ files not present (LDC)**; LibriSpeech stand-in labelled in manifests; `docs/BASE_PAPER_DATASET.md` |
| 5 | Multi-ASR evaluation system | **Implemented** | Two **real independent families** in the headline matrix (Whisper tiny; Vosk/Kaldi lineage) plus a **four-engine transfer validation** (+ Whisper small, wav2vec2-base) on the top-K conditions — run complete: 770 rows, 0 errors → `results/tables/validation/`; `IndependentASREngine` is **heuristic and excluded from headline results**; defenses in `asr/defense.py` |
| 6 | Human-vs-ASR research metrics | **Implemented (human = proxy)** | `evaluation/metrics.py` (13 metrics), `evaluation/hag_metrics.py` (HSR/ASR-SR/HAG/CSS/DSR/CMFR/PRR/PAR/TRS); HSR labelled *illustrative* |
| 7 | Fair comparison protocol | **Implemented** | `METHODOLOGY.md` §3; controls + matched-power control in every run |
| 8 | Multi-objective transform ranking | **Implemented** | `human_rank`, `attack_rank`, `gap_rank`, `quality_rank`, `pareto` in `summary.csv`; `ranking.md` |
| 9 | Statistical analysis | **Implemented** | `evaluation/stats.py` (bootstrap/BCa, effect sizes, parametric + nonparametric tests, multiple-comparison corrections, Pareto frontier); CIs in `fig4` |
| 10 | Human evaluation support | **Protocol ready; not conducted** | `captcha/generator.py::HumanStudyProtocol` (trial orders, counterbalancing, attention checks, exclusion, JSONL export) + §10 response data model in `log_response`; no participants ⇒ proxy-only HSR; `HUMAN_STUDY_PROTOCOL.md` |
| 11 | CAPTCHA prototype layer | **Implemented** | `captcha/generator.py` (`CAPTCHAGenerator`, `CAPTCHAChallenge`, both modes; challenge IDs, expiry, verify, replay limit, rate-limit hooks, wav+sidecar artifact export, seeded reproducibility) |
| 12 | Visualization & paper-ready reporting | **Implemented** | `scripts/make_figures.py` — **16 figures** PNG+SVG+PDF (figs 1–4 + WHAT-REMAINS §12 extended set figs 5–16, incl. heatmap/forest/confusion/spectrogram/masking/rank-stability/shelf-life; §12 coverage matrix in `PAPER_RESULTS_GUIDE.md`), `scripts/make_tables.py` (CSV+LaTeX); outputs in `results/figures`, `results/tables` |
| 13 | Testing & quality requirements | **Implemented** | **488 tests green**; incl. 21 WSJ-protocol, 24 comparative-benchmark (4-engine registry + family-macro ΔWER), 17 shelf-life (N11), 9 CAPTCHA, **6 CLI** (script entry points), Whisper/Vosk/wav2vec2 adapters, registry-integrity tests |
| 14 | Research safety & scientific integrity | **Implemented** | `HSR_LABEL` enforced in code, CSV, Markdown, JSON; heuristic engine tagged `heuristic_proxy`; stand-in corpus always labelled; no fabricated results |
| 15 | Update the documentation | **Implemented** | `README.md` + this file + 13 docs listed below; per-transform citations in `TRANSFORM_CATALOG.md` |
| 16 | Final paper contributions | **Implemented (claims drafted, evidence-gated)** | `docs/NOVELTIES.md` N1–N11 mapped to §16 (N11 = shelf-life forecasting, original contribution); only claims with artifacts retained; quantitative claims gated on `results/comparative/main` + `results/shelf_life` |
| 17 | Implementation order | **Implemented** | Order followed: audit → model → taxonomy → data → ASR → metrics → ranking → benchmark → figures → docs |

## Benchmark run (central artifact)

| Item | Status |
|---|---|
| `scripts/run_comparative_benchmark.py` | **Implemented** — manifest provenance, λ-sweep mirror, resumable `rows.csv`, 4-worker execution |
| Headline run `results/comparative/main` (119 transforms + sweep + control, 70 utts, Whisper+Vosk) | **Complete** — 10,290/10,290 tasks OK, 0 errors; 147 conditions in `summary.csv`, `ranking.md`, manifests; log `main_run.log` (history in `main_run_history.log`); re-run the same command to resume/refresh |
| Novelty N11 pass `scripts/run_shelf_life.py` → `results/shelf_life` | **Complete** — 1,260/1,260 tasks (5 policies × 70 utts × 3 models), 0 errors; `shelf_life.{json,md}` + manifests; fits b=0.41–0.64 (R² 0.965–1.000), C\* = 112M→2M, B=1000 CIs; log `run.log` |
| Smoke runs (`smoke`, `smoke_asr`) | **Completed** — used for pipeline validation only, n ≤ 2; not citable for conclusions |

## Document set (WHAT-REMAINS §15)

| Document | Status |
|---|---|
| `README.md` | Implemented (rewritten) |
| `PROJECT_STATUS.md` | this file |
| `RESEARCH_QUESTION.md` | Implemented |
| `METHODOLOGY.md` | Implemented |
| `TRANSFORM_CATALOG.md` | Implemented (generated from registry; regenerate: `python scripts/generate_catalog.py`) |
| `DATASETS.md` | Implemented |
| `ASR_MODELS.md` | Implemented |
| `METRICS.md` | Implemented |
| `STATISTICAL_ANALYSIS.md` | Implemented |
| `HUMAN_STUDY_PROTOCOL.md` | Protocol ready; study **not validated** (no participants) |
| `THREAT_MODEL.md` | Implemented |
| `REPRODUCIBILITY.md` | Implemented |
| `LIMITATIONS.md` | Implemented |
| `PAPER_RESULTS_GUIDE.md` | Implemented |
| `REPORT.md` | Implemented |
| `docs/BASE_PAPER_DATASET.md`, `docs/GAP_ANALYSIS.md`, `docs/NOVELTIES.md`, `docs/IMPROVEMENTS_VS_LITERATURE.md`, `docs/COMPARATIVE_ANALYSIS.md` | Implemented (comparative analysis finalized with the run's results) |

## Remaining work (honest)

1. **Real human study** — recruitment/approval required; everything else is
   wired (`HUMAN_STUDY_PROTOCOL.md`).
2. **Exact WSJ corpus** — requires LDC licence; then one re-run of the same
   command with `--dataset wsj` gives bit-faithful base-paper numbers.
3. **Full four-engine matrix** — `wav2vec2-base` and Whisper small are
   adopted as evaluation engines in the targeted transfer-validation run
   (top-K conditions), not in the full matrix; larger/other SSL models
   (HuBERT, WavLM, wav2vec2-large) are **not evaluated** and are stated as
   such rather than half-implemented.
4. **Per-condition compute-cost column** — *Planned*; only aggregate
   throughput logged today.
5. **Defense sweep in the headline run** — *Optional/on-demand* (6× ASR
   cost); defense metrics validated in tests.
