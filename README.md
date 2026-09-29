# AudioCAPTCHA-DSP

**Paper:** *Systematic Evaluation of Psychoacoustic Distortions on Human
Intelligibility vs Automatic Speech Recognition: A Framework for Secure Audio
Verification.*

Research framework for experimentally studying differences between human
intelligibility and automatic speech recognition robustness under controlled
psychoacoustic transformations — built to extend Schönherr et al. 2018
(*Adversarial Attacks Against ASR Systems via Psychoacoustic Hiding*,
arXiv:1808.05665) on its own dataset/protocol.

## Status

Research-complete pipeline:

- **119 registered transforms** across taxonomy families A-H
  (`transforms/registry.py`).
- **Novelty N11 — CAPTCHA shelf-life forecasting**: the human–ASR gap is
  measured along a controlled capacity ladder (Whisper tiny/base/small),
  fit with a scaling law + paired bootstrap, and translated into a break
  capacity and months-to-break scenarios
  (`docs/NOVELTIES.md` N11, `scripts/run_shelf_life.py`, `fig16`).
- **Base-paper dataset protocol**: WSJ subsets A/B/C reproduced exactly
  (70 utterances/10 speakers; 72 speech + 70 music; 150 speech + 72 music,
  ≤6 phones/s) with Kaldi + raw-LDC adapters and full run manifests —
  see [`docs/BASE_PAPER_DATASET.md`](docs/BASE_PAPER_DATASET.md).
- **Multi-ASR evaluation** with two *real, independent* families in the
  full matrix — Whisper tiny (attention encoder-decoder) and Vosk small-en
  (Kaldi nnet3 — the base paper's toolkit lineage) — plus a **four-engine
  transfer validation** (adding Whisper small and self-supervised
  wav2vec 2.0) on the top-ranked attack conditions.
- **Human-vs-ASR metrics**: STOI proxy, MBSD, SNR/SI-SDR, WER per engine,
  cross-family ΔWER (family-macro when one family has two sizes), and the
  HSR/HAG stack — human numbers are always labelled *illustrative* (no
  human study has been conducted).
- **488 tests** green (`pytest tests/`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # add ".[dev,ssl]" for the wav2vec2 engine
```

Optional data for the headline benchmark:

```powershell
# LibriSpeech test-clean (public stand-in for WSJ) → data/raw/LibriSpeech/test-clean
# Vosk model (Kaldi-lineage ASR family):
Invoke-WebRequest https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip `
  -OutFile vosk.zip; Expand-Archive vosk.zip data/raw/vosk
# Whisper tiny/small weights auto-download to ~/.cache/whisper on first use.
# wav2vec2 weights auto-download to ~/.cache/huggingface (needs .[ssl] extra).
# Exact WSJ (LDC): python scripts/prepare_wsj.py --help
```

## Run Tests

```bash
make test
# or: python -m pytest tests/
```

## Run the Comparative Benchmark

```bash
# Headline run: base-paper subset-A protocol (70 utterances / 10 speakers),
# curated transforms + λ-sweep mirror, Whisper + Vosk, resumable:
python scripts/run_comparative_benchmark.py --out results/comparative/main

# Full catalog (all 119 transforms) - re-run to resume after interruption:
python scripts/run_comparative_benchmark.py --transforms all --out results/comparative/main

# Fast DSP-only smoke test:
python scripts/run_comparative_benchmark.py --asr none --max-utterances 2 \
    --no-sweep --transforms noise.white --out results/comparative/smoke
```

Outputs per run: `dataset_manifest.json` (protocol + corpus hash),
`run_manifest.json` (config/versions/labels), `rows.csv` (append-only,
resumable), `summary.csv/json`, `ranking.md`.

## Figures & Tables

```bash
python scripts/make_figures.py --run results/comparative/main --out results/figures
python scripts/make_tables.py  --run results/comparative/main --out results/tables
```

Four paper figures (Pareto human-vs-ASR, λ-sweep mirror, cross-family
transfer, family summary) as PNG+SVG; tables as CSV + LaTeX (booktabs).

## Documentation

| Document | Contents |
|---|---|
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | WHAT-REMAINS phase-by-phase status (1–17) with evidence |
| [`RESEARCH_QUESTION.md`](RESEARCH_QUESTION.md) | RQ1–RQ6, operationalized + non-claims |
| [`METHODOLOGY.md`](METHODOLOGY.md) | Experiment model, fair-comparison protocol, axes, statistics |
| [`TRANSFORM_CATALOG.md`](TRANSFORM_CATALOG.md) | All 119 transforms w/ scientific basis (generated) |
| [`DATASETS.md`](DATASETS.md) | Corpus adapters, subset protocol, manifests |
| [`ASR_MODELS.md`](ASR_MODELS.md) | Two real ASR families (headline matrix) + 4-engine transfer validation; heuristic engine policy |
| [`METRICS.md`](METRICS.md) | Fidelity/quality/ASR/human/security metric inventory |
| [`STATISTICAL_ANALYSIS.md`](STATISTICAL_ANALYSIS.md) | CIs, effect sizes, tests, corrections |
| [`HUMAN_STUDY_PROTOCOL.md`](HUMAN_STUDY_PROTOCOL.md) | Listener-study design; label-retirement criteria |
| [`THREAT_MODEL.md`](THREAT_MODEL.md) | Attacker/defender models mapped to taxonomy |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Exact commands, manifests, verification steps |
| [`LIMITATIONS.md`](LIMITATIONS.md) | Honest inventory incl. integrity rules |
| [`PAPER_RESULTS_GUIDE.md`](PAPER_RESULTS_GUIDE.md) | Evidence→claim map, wording rules, paper skeleton |
| [`docs/BASE_PAPER_DATASET.md`](docs/BASE_PAPER_DATASET.md) | Dataset fidelity contract: exact vs. approximated vs. stand-in |
| [`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md) | Gaps in the base paper (G1–G12) and the 23-paper survey |
| [`docs/NOVELTIES.md`](docs/NOVELTIES.md) | 10 novel contributions mapped to WHAT-REMAINS §16 |
| [`docs/IMPROVEMENTS_VS_LITERATURE.md`](docs/IMPROVEMENTS_VS_LITERATURE.md) | Per-paper (1–23) adopt/improve comparison |
| [`docs/COMPARATIVE_ANALYSIS.md`](docs/COMPARATIVE_ANALYSIS.md) | Results of the comparative benchmark |
| [`REPORT.md`](REPORT.md) | Project report |

## Run CLI

```bash
audiocaptcha --help
```

## License

Research Use Only. Not for production deployment.
