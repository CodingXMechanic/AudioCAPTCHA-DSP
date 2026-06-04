# AudioCAPTCHA-DSP — Rational Run & Composite Run Report

**Date:** 2026-06-04
**Phase:** Experiment Harness (Phase 3)
**Scope:** Rational run (single-transform), Composite run (multi-transform)

---

## 1. Implementation Report

### 1.1 What Was Built

Six new or rewritten modules implement a complete experiment lifecycle:

| Module | Purpose | Status |
|---|---|---|
| `core/types.py` | `ExperimentConfig`, `ExperimentResult`, `BenchmarkResult`, `TransformSpec` — frozen dataclasses with YAML parsing and cartproduct expansion | Done |
| `evaluation/dataset.py` | `AudioDataset.synthetic()` — deterministic synthetic signal generation (8–N tones, seeded RNG, no I/O dependency) | Done |
| `evaluation/metrics.py` | `compute_metrics()` — SNR, RMSE, MSE, PSNR, spectral convergence, normalized cross-correlation, RMS ratio | Done |
| `evaluation/stats.py` | `bootstrap_ci()`, `summarize_condition()` — percentile bootstrap confidence intervals, per-condition aggregation with mean/std/median/min/max/CI | Done |
| `io/artifacts.py` | `save_experiment_manifest()`, `save_signal_wav()` — JSON serialization with numpy-to-JSON coercion, manifest versioning | Done |
| `experiments/runner.py` | `ExperimentRunner` — condition grid expansion, transform chain construction, per-sample timing, metric computation, manifest persistence | Done |

Two new experiment configurations:

| Config | Transform | Conditions | Grid |
|---|---|---|---|
| `exp_05_rational_run.yaml` | `temporal.jitter` | 6 amplitudes x 4 frequencies = 24 | Single-transform parameter sweep |
| `exp_06_composite_run.yaml` | `temporal.jitter` + `spectral.masking` | 2x2x1x2 = 8 | Cross-product of per-transform parameter lists |

CLI extended with `audiocaptcha run <name>` which loads config, executes the full experiment, prints summary, and writes `results/<id>/manifest.json`.

### 1.2 Test Results

```
83 tests, 83 passed, 0 failed
```

Coverage spans: signal construction, pipeline identity/chaining, all 6 temporal transforms (resampler, jitter, OLA), all 3 spectral transforms (masking, notch, warping), composite chain, psychoacoustics stubs, WER/CER metrics.

### 1.3 Compile & Import Validation

- `python -m compileall src/audiocaptcha_dsp/` — clean, 0 errors
- All 28 public symbols across 12 `__init__.py` files import without error
- All 6 transforms produce valid `Signal` outputs with correct metadata

### 1.4 Experiment Execution Results

**Rational Run (exp_05):**
- 24 conditions x 8 samples = 192 transform executions
- Total wall time: 2.22 seconds
- Throughput: ~86 samples/second
- Manifest: `results/exp_05/manifest.json` (192 result records + 24 condition summaries)

**Composite Run (exp_06):**
- 8 conditions x 8 samples = 64 transform executions (each sample passes through jitter + masking)
- Total wall time: 1.02 seconds
- Throughput: ~63 samples/second
- Manifest: `results/exp_06/manifest.json` (64 result records + 8 condition summaries)

### 1.5 Reproducibility & Determinism

Both properties are structurally guaranteed:

- **Determinism:** `AudioDataset.synthetic(seed=N)` uses `np.random.default_rng(N)`. All transforms accept `seed` parameters. The `ExperimentConfig` freezes all parameters. Re-running with the same config produces bit-identical results.
- **Reproducibility:** The manifest JSON captures every result with full condition parameters, per-sample timing, and metric values. The experiment can be re-executed from the YAML config alone — no external state.

---

## 2. Methodology Summary

### 2.1 Signal Generation

Synthetic signals are exponentially-decaying sinusoids at base frequencies linearly spaced from 150 Hz to 2000 Hz, with additive Gaussian noise (sigma=0.005), at 16 kHz sample rate, 2-second duration. This provides a controlled, reproducible test set with spectral diversity across the speech-relevant range.

### 2.2 Transform Architecture

All transforms implement the `Transform` protocol (`Callable[[Signal], Signal]`) and are pure functions — they receive a `Signal` and return a new `Signal` without mutating the input. The `TransformChain` composes any number of transforms sequentially. The `ExperimentRunner` builds chains from `TransformSpec` + condition parameters.

### 2.3 Condition Grid

For single-transform experiments, the grid is the cartesian product of all parameter lists in the YAML `parameters` dict. For composite experiments, `_expand_composite_conditions` computes the cross-product of per-transform parameter lists, producing flat condition dicts that are then merged back into per-transform parameter sets at execution time.

### 2.4 Metrics

Seven DSP-level metrics are computed per sample:

| Metric | What it measures |
|---|---|
| SNR (dB) | Signal-to-noise ratio between original and processed |
| RMSE | Root-mean-square error |
| MSE | Mean-square error |
| PSNR (dB) | Peak signal-to-noise ratio |
| Spectral convergence | L2 norm of magnitude spectrum difference / L2 norm of original |
| Normalized cross-correlation | Pearson correlation of centered waveforms |
| RMS ratio | processed RMS / original RMS |

### 2.5 Statistical Summary

Per-condition summaries include: mean, std, median, min, max, and 95% bootstrap CI (percentile method, 1000 resamples) for each metric. The `ConditionSummary.to_dict()` produces JSON-serializable output.

### 2.6 Artifact Manifest

Each experiment run produces a versioned JSON manifest containing:
- Experiment metadata (id, timestamp, summary statistics)
- Per-condition summaries with full metric statistics
- Per-sample result records (condition params, sample id, all metrics, transform chain, wall time)

---

## 3. Limitations

### 3.1 Known Issues

1. **`snr_db` and `psnr_db` are `inf` for identity transforms.** When amplitude=0 (no jitter), the processed signal is identical to the input, producing infinite SNR/PSNR. The `summarize_condition` function correctly filters these via `isfinite`, so they are excluded from condition summaries. The raw result records still contain `inf`. This is mathematically correct but means those two metrics are only meaningful for non-identity conditions.

2. **`condition_index` leaks into metrics dict.** The runner stores `condition_index` in the metrics dict for internal tracking, but it is a non-string integer key that gets included in the per-result metrics. The `summarize_condition` function filters it by type-checking, so it does not affect statistics, but it adds noise to the raw result records.

3. **Composite parameter name collisions.** The `_merge_composite_specs` function matches condition parameters to transform parameters by bare key name. If two transforms in a composite chain share a parameter name (e.g., both have a `strength` parameter), the condition value will be applied to both. The current configs avoid this, but it is a structural limitation.

4. **No disk-based dataset loading.** `AudioDataset._load_from_disk()` is implemented but untested. The `synthetic()` factory is the only tested path. Real speech datasets (LibriSpeech etc.) require manual download and are explicitly out of scope.

5. **No ASR evaluation.** WER/CER metrics are implemented but require text transcripts and an ASR engine. The current pipeline computes DSP-level metrics only. ASR integration (Whisper) is explicitly excluded.

6. **Bootstrap CI uses simple percentile method.** The percentile bootstrap is adequate for confidence intervals but does not correct for bias or skewness. For publication-quality statistics, BCa (bias-corrected and accelerated) bootstrap would be preferable.

7. **Fixed synthetic signal set.** The 8-signal synthetic set covers a limited frequency range. Experiments requiring phonetic diversity, varying durations, or real speech characteristics are not supported without implementing disk loading.

### 3.2 Scope Boundaries (by design)

- No Whisper / ASR inference
- No CAPTCHA generation
- No ML model training or fine-tuning
- No internet downloads
- No paper drafting

---

## 4. Readiness Score

| Criterion | Score | Notes |
|---|---|---|
| **Correctness** | 9/10 | 83/83 tests pass. All transforms produce valid outputs. Known: `inf` SNR for identity, `condition_index` leak in metrics |
| **Reproducibility** | 10/10 | Fully deterministic from config YAML. Seeded RNG throughout. Manifest captures all inputs and outputs |
| **Type safety** | 9/10 | All public APIs use dataclasses with type hints. `Transform` protocol enforced. `frozen=True` on config/result types. Minor: `summarize_condition` accepts `list[dict[str, Any]]` |
| **Benchmarkability** | 9/10 | Per-sample timing, throughput measurement, per-condition statistics with bootstrap CI. Minor: no built-in benchmark comparison across runs |
| **Modularity** | 10/10 | Clean separation: transforms, dataset, metrics, stats, artifacts, runner, CLI. Each module independently testable |
| **Local execution** | 10/10 | No network calls. No external services. All computation is local numpy/scipy |
| **Documentation** | 6/10 | Code is well-structured with type hints but lacks docstrings on most functions. Report fills some gaps |
| **Extensibility** | 9/10 | New transforms: add class + register in `_TRANSFORM_REGISTRY`. New metrics: add to `compute_metrics()`. New experiment types: extend `ExperimentConfig.from_yaml()` |

**Overall readiness: 9/10**

The framework is ready for systematic DSP-level experiment campaigns. The core loop (config → generate → transform → measure → summarize → persist) is complete, tested, and deterministic. The primary gaps are ASR integration (by design) and the two minor metric-handling issues noted above.

---

## 5. File Inventory (changed/created in this phase)

### New files
- `configs/experiments/exp_05_rational_run.yaml`
- `configs/experiments/exp_06_composite_run.yaml`
- `results/exp_05/manifest.json` (generated)
- `results/exp_06/manifest.json` (generated)

### Modified files
- `src/audiocaptcha_dsp/core/types.py` — added `ExperimentConfig`, `ExperimentResult`, `BenchmarkResult`, `TransformSpec`
- `src/audiocaptcha_dsp/core/__init__.py` — updated exports
- `src/audiocaptcha_dsp/evaluation/dataset.py` — full implementation with synthetic generation
- `src/audiocaptcha_dsp/evaluation/metrics.py` — 7 DSP metrics + `compute_metrics()` aggregator
- `src/audiocaptcha_dsp/evaluation/stats.py` — `bootstrap_ci()`, `summarize_condition()`, `ConditionSummary`
- `src/audiocaptcha_dsp/evaluation/__init__.py` — updated exports
- `src/audiocaptcha_dsp/io/artifacts.py` — full implementation with manifest serialization
- `src/audiocaptcha_dsp/experiments/runner.py` — full `ExperimentRunner` implementation
- `src/audiocaptcha_dsp/experiments/__init__.py` — updated exports
- `src/audiocaptcha_dsp/cli.py` — `run` command executes experiments end-to-end
