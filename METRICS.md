# Metrics

Metric inventory across WHAT-REMAINS §6 groups A–E, with implementation
status and honest labelling. Source of truth:
`evaluation/metrics.py`, `evaluation/hag_metrics.py`.

---

## A. Audio fidelity / distortion metrics — *Implemented*

All computed by `compute_metrics(original, processed)` and stored per
(condition, utterance) row in `rows.csv`:

| Key | Definition | Notes |
|---|---|---|
| `snr_db` | 10·log₁₀(‖x‖² / ‖x−x̂‖²) | identity → `inf` (stored empty) |
| `si_sdr_db` | scale-invariant SDR | |
| `rmse`, `mse` | time-domain error | |
| `psnr_db` | peak-normalized SNR | |
| `spectral_convergence` | ‖X−X̂‖_F / ‖X‖_F | |
| `normalized_cross_correlation` | waveform NCC | |
| `log_spectral_distance` | mean log-magnitude spectral distance | |
| `rms_original`, `rms_processed`, `rms_ratio` | level bookkeeping | `rms_ratio` ≈ 1 for budgeted transforms |

## B. Speech quality / intelligibility metrics — *Implemented*

| Metric | Status | Notes |
|---|---|---|
| **MBSD** (Modified Bark Spectral Distortion, paper 13) | **Implemented** | Bark-scale weighted spectral distortion; `SecurityEvaluator` acceptability default 5.0; drives `quality_rank` |
| **STOI proxy** (`compute_stoi_proxy`) | **Implemented — proxy** | Short-time envelope-correlation intelligibility measure consistent with paper 15's correlation findings; **the human axis of every ranking** |
| PESQ (paper 16) | **Not supported** | ITU reference-code licence; substitution (MBSD+STOI) declared, never silent |
| MOS predictors (papers 17/18) | **Not supported here** | pretrained weights not shippable offline |

## C. ASR metrics — *Implemented*

| Metric | Definition |
|---|---|
| `wer` | jiwer word error rate on normalized text; per engine, per row |
| `cer` | `compute_cer` (available; benchmark uses WER for comparability with the base paper) |
| `detailed_wer` | substitutions/deletions/insertions breakdown |
| **ΔWER** | WER(condition) − WER(original) *per engine* |
| **`cross_delta_wer`** | mean ΔWER over the engines evaluated in the run — headline (Whisper tiny + Vosk, the two independent families); family-macro when one lineage has >1 engine (4-engine validation) — headline attack metric |

Baseline WERs are always printed with the run
(`[baseline WER] whisper_tiny=…, vosk_small_en=…` for the headline; the
validation run additionally prints `whisper_small=…, wav2vec2_base=…`) and
stored in `summary.json.baseline_wer`.

## D. Human metrics — *Protocol ready; values currently illustrative*

| Metric | Status | Definition |
|---|---|---|
| `hsr` — human success rate | **Illustrative (proxy)** | default `clip(0.6 + 0.4·STOI_proxy, 0, 1)`; becomes a real value only when `human_responses` are supplied (`n_human_participants > 0`) |
| MUSHRA-style scoring | Protocol ready | 9-sample condition sets (base paper layout) via `HumanStudyProtocol` |
| Transcription task success | Protocol ready | 21-sample/22-listener template; see `HUMAN_STUDY_PROTOCOL.md` |

**Labelling contract:** `HSR_LABEL = "illustrative (STOI-derived proxy; no
human study conducted)"` is written into every `summary.csv` row, every
`summary.json`, `ranking.md` header, figure captions (via `HSR_NOTE`) and
LaTeX captions. A human number must never appear unlabelled.

## E. Security & robustness metrics — *Implemented*

From `evaluation/hag_metrics.py::SecurityEvaluator.evaluate(...)`:

| Field | Meaning | Default threshold |
|---|---|---|
| `hsr` | human success rate (see D) | — |
| `asr_sr` | ASR success rate = share of utterances with WER ≤ threshold | `wer_threshold = 0.3` |
| **`hag`** | Human-ASR Gap = HSR − ASR-SR | ∈ [−1, 1]; higher = better separation |
| `css` | composite stealth score over STOI/MBSD/SNR | STOI ≥ 0.7, MBSD ≤ 5.0, SNR ≥ 15 dB |
| `trs` | transfer robustness score over per-model WERs | `wer_threshold = 0.5` |
| `par` | perceptual-attack rate: share stealthy **and** attacking | combines CSS + WER |
| `dsr` / `cmfr` / `prr` | defense success rate, defense-conditioned metric, perturbation resilience rate | defense-aware (`weight_dsr=0.6`) |
| `mean_*` | mean STOI/MBSD/SNR/WER/CER over the condition | |
| `is_pareto_efficient` | evaluator's own efficiency flag | complemented by the benchmark's (STOI, cross-ΔWER) Pareto flag |

`SecurityEvaluation.to_dict()` is what lands in `summary.csv` columns
(`hsr`, `asr_sr`, `hag`, `css`), plus `gap_rank` ordering by `hag`.

**Forecast metrics (novelty N11, `experiments/shelf_life.py`):**

| Field | Meaning | Notes |
|---|---|---|
| `a`, `b`, `r2` | power-law fit of attacked-WER vs capacity: `WER = a·C^(−b)` | log-log OLS over the 3-rung ladder; `b_ci` from paired bootstrap |
| `break_capacity_m_params` | **C\***: capacity at which predicted attacked WER reaches `w_break` (= 0.3, same threshold as `asr_sr`) | `None` when b ≤ 0 or C\* > 1e12 M; always with `break_ci` |
| `forecast_months` | months from the largest evaluated rung to C\* under 6/12/24-month doubling **scenarios** | assumptions, not measured trends |
| `gap` (per rung) | HSR − ASR-SR at that capacity (same definition as `hag`) | HSR = illustrative proxy |

## F. Aggregation & ranking outputs — *Implemented*

Per condition (aggregation in `experiments/comparative.py`):
`n_ok`, `n_err`, `stoi_mean/std`, `snr_mean`, `mbsd_mean`, `si_sdr_mean`,
per-engine `wer_*_mean` and `wer_*_delta`, `cross_delta_wer`,
`human_rank`, `attack_rank`, `gap_rank`, `quality_rank`, `pareto`,
`hsr_label`.

## G. What we deliberately do not report

- Any human number without the illustrative label.
- ΔWER from the heuristic engine (it fabricates WER from metadata).
- PESQ/MOS values (metric not available — declared, not substituted silently).
- Bit-exact WSJ claims from stand-in-corpus runs (manifests say stand-in).
