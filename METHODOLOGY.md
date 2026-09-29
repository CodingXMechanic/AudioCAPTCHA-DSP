# Methodology

End-to-end description of the experimental method. Status labels follow the
WHAT-REMAINS §15 convention: *Implemented / Partially implemented /
Experimental / Optional / Planned / Not validated / Not supported.*

---

## 1. Experiment model — *Implemented*

Everything reduces to a single typed pipeline:

```
Signal(waveform, sample_rate, metadata)
   └─ transform: Signal → Signal            (119 registered transforms, families A–H)
        └─ metrics: (original, processed) → dict[str, float]
             └─ ASR engines: Signal → TranscriptionResult → WER
                  └─ SecurityEvaluator: (dsp_metrics, asr_results, …) → SecurityEvaluation
```

- `core/signal.py` — immutable-by-convention signal container.
- `experiments/runner.py::resolve_transform(type, params)` — resolves against
  the **central registry** (`transforms/registry.py`); every result row
  records `transform_key` + JSON parameters for exact replay.
- Determinism: transforms that accept `seed` receive the run seed (42
  default) via `build_conditions`; seeds, versions and git commit are
  written to `run_manifest.json`.

## 2. Dataset selection — *Implemented (protocol), Partially (corpus)*

The sample selection **reproduces the base-paper subset protocol exactly**
(`docs/BASE_PAPER_DATASET.md`):

| Subset | Paper spec | Implementation |
|---|---|---|
| A | 70 utterances, 10 speakers, one WSJ test set | `select_paper_subset(total=70, n_speakers=10, mode='balanced')` → 7 utterances/speaker |
| B | 72 speech + 70 music | spec honored; music requires `--music-dir` |
| C | 150 speech + 72 music, random, ≤6 phones/s | spec + `phone_rate ≤ 6.0` filter |

- Corpus: WSJ adapter (`evaluation/dataset.py::WSJAdapter`, Kaldi + raw LDC
  layouts) when licensed; otherwise LibriSpeech test-clean labelled
  `stand-in for WSJ: LDC-licensed`.
- Every run writes **`dataset_manifest.json`**: corpus, protocol, selection
  mode/seed, per-utterance IDs/transcripts/source paths, `corpus_sha256`.
  *A result without a manifest entry is not a result.*

## 3. Fair comparison protocol — *Implemented*

Mirrors WHAT-REMAINS §7 / base-paper §"fair comparison":

1. **Original (control)** — baseline WER per engine, identity condition.
2. **Identity / baseline transforms** — family A (gain, limiter, bit-depth…) run as ordinary conditions.
3. **Every transform** — same utterances, same seed, same engine settings;
   only the transform differs.
4. **Matched-power unconstrained control** — `control.no_hearing_threshold`
   (RQ2).
5. **λ-sweep** — identical grid over four psychoacoustic transforms (RQ4).
6. **Condition order randomization** — conditions are executed in registry
   order; each condition sees the identical utterance set, so any ordering
   effect is common across conditions (no per-condition drift possible).
   *Randomized interleaving: Optional, not implemented — noted in `LIMITATIONS.md`.*

## 4. Evaluation axes — *Implemented*

| Axis | Metric(s) | Source |
|---|---|---|
| Human intelligibility | `stoi_proxy` (STOI-style correlation-based proxy), MBSD, SNR | `evaluation/metrics.py` |
| Human success | `hsr` — **labelled illustrative** (STOI-derived; no human study) | `evaluation/hag_metrics.py` |
| ASR robustness | WER per engine (jiwer), ΔWER vs per-engine baseline | `asr/`, `compute_wer` |
| Cross-model transfer | `cross_delta_wer` = mean ΔWER over the run's independent engines (family-macro when a family has >1 engine) | `experiments/comparative.py` |
| Security gap | `hag`, `asr_sr`, `css`, `dsr`, `cmfr`, `prr` | `SecurityEvaluator` |

Transcripts are normalized (`normalize_transcript`: lowercase, strip
punctuation, collapse whitespace) before jiwer on both reference and
hypothesis.

## 5. Ranking — *Implemented* (WHAT-REMAINS §8)

Per condition (original excluded):

- `human_rank` — mean STOI, descending.
- `attack_rank` — cross-family ΔWER, descending.
- `gap_rank` — HAG, descending ("highest human-ASR gap").
- `quality_rank` — mean MBSD, ascending (lower distortion = better quality).
- `pareto` — nondominated on (STOI, cross-ΔWER), both maximized.

Cost axis (§8 "lowest computational cost"): *Partially implemented* —
throughput is logged per run; per-condition timing is not stored in rows
(see `LIMITATIONS.md`).

## 6. Statistics — *Implemented*

`evaluation/stats.py`: bootstrap CIs (percentile + BCa), Cohen's *d*,
Hedges' *g*, Welch *t*, Mann–Whitney *U*, Wilcoxon signed-rank, paired
permutation test, Bonferroni and Benjamini–Hochberg corrections,
`summarize_condition`/`compare_conditions`, Pareto frontier. Family summary
figures carry t-based 95 % CIs.

## 7. Human evaluation support — *Protocol ready, not conducted*

`captcha/generator.py::HumanStudyProtocol` + `asr/defense.py` condition
layouts support both modes of WHAT-REMAINS §10:

- **A. Human transcription mode** — 21-sample/22-listener layout of the base
  paper is the template.
- **B. CAPTCHA mode** — `CAPTCHAGenerator` produces challenges.

No participants have been recruited: all human numbers in outputs are the
labelled proxy. See `HUMAN_STUDY_PROTOCOL.md`.

## 8. Reporting — *Implemented*

`scripts/make_figures.py` (4 paper figures, PNG+SVG at 300 dpi) and
`scripts/make_tables.py` (CSV + booktabs LaTeX). Every caption/table carries
the dataset protocol reference and the HSR illustrative-label sentence
(`HSR_NOTE`).

## 9. Reproducibility commands

```bash
# tests (488 green)
python -m pytest tests/ -q

# headline benchmark (resumable)
python scripts/run_comparative_benchmark.py --transforms all \
    --workers 4 --out results/comparative/main

# artifacts
python scripts/make_figures.py --run results/comparative/main --out results/figures
python scripts/make_tables.py  --run results/comparative/main --out results/tables
```
