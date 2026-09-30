# Reproducibility

How to obtain exactly what this repository produced, and how to prove it.

---

## 1. Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"        # incl. vosk>=0.3.45, openai-whisper, jiwer, pandas
pip install -e ".[dev,ssl]"    # + wav2vec2 adapter (optional `ssl` extra)
```

- Determinism: run seed default **42** (`--seed`), propagated into every
  transform whose `__init__` accepts `seed`; seeds recorded in each
  condition's `params_json`.
- Whisper: `temperature=0`, `fp16=False` (CPU) → deterministic decoding.
- Vosk: model pinned by directory name `vosk-model-small-en-us-0.15`.
  Re-runs under an identical configuration reproduce (A/B check: 0/15
  differing hypotheses); runs under a *different* worker/engine
  configuration flip a small share of near-tied lattices on noise-like
  conditions (106/770 across two full runs, ≤ 1.8 pp condition-level WER;
  attack ranking unchanged, ρ = 0.964) — see `LIMITATIONS.md` §3.
- wav2vec2: greedy CTC decoding (no sampling) → deterministic; model pinned
  by hub id `facebook/wav2vec2-base-960h`, cached under
  `~/.cache/huggingface`.

## 2. Exact commands

```powershell
# ---- tests (expect: 488 passed) ----
.venv\Scripts\python -m pytest tests/ -q --basetemp=$env:TEMP\audiocaptcha-dsp\pytest

# ---- data ----
# LibriSpeech test-clean → data/raw/LibriSpeech/test-clean   (present)
# Vosk model → data/raw/vosk/vosk-model-small-en-us-0.15      (present)
# Whisper tiny/small → ~/.cache/whisper/{tiny,small}.pt       (auto)
# wav2vec2 → ~/.cache/huggingface (auto; needs the `ssl` extra)
# WSJ (optional, licensed):
python scripts/prepare_wsj.py --check
python scripts/prepare_wsj.py --build-kaldi --wsj-root <path> --out data/raw/wsj

# ---- headline benchmark (resumable; identical command resumes) ----
python scripts/run_comparative_benchmark.py --transforms all --workers 4 `
    --out results/comparative/main
# ---- 4-engine transfer validation (top-K conditions) ----
# each worker holds all four models (~3 GB) -> 3 workers on a 16 GB machine
python scripts/run_comparative_benchmark.py --workers 3 `
    --transforms spectral.minimum_phase,novel.captcha_optimal,noise.clicks,`
spectral.phase_randomization,novel.multi_domain,novel.defense_robust,`
channel.codec_simulation,channel.packet_jitter,baseline.loudness_normalize `
    --no-sweep --asr whisper_tiny,vosk_small_en,whisper_small,wav2vec2_base `
    --out results/comparative/engine_validation
# smaller/clean-room variants:
python scripts/run_comparative_benchmark.py --dataset wsj --data-root data/raw/wsj `
    --subset A --transforms curated --out results/comparative/wsj_exact
python scripts/run_comparative_benchmark.py --asr none --max-utterances 2 `
    --no-sweep --transforms noise.white --out results/comparative/smoke

# ---- artifacts ----
python scripts/make_figures.py --run results/comparative/main --out results/figures
python scripts/make_tables.py  --run results/comparative/main --out results/tables
python scripts/make_tables.py  --run results/comparative/engine_validation --out results/tables/validation

# ---- human-proxy validation vs real listener scores (docs/HUMAN_PROXY_VALIDATION.md) ----
# data: TMHINT-QI release (InQSS) → data/references/tmhintqi/TMHINTQI/{raw_data.csv,test,train}
pip install pystoi                               # optional reference metric
python scripts/validate_human_proxy.py --split test --jobs 4
python scripts/validate_human_proxy.py --split all  --jobs 4   # → results/validation/
```

## 3. What each run proves (manifests)

| File | Proves |
|---|---|
| `<run>/dataset_manifest.json` | exactly which utterances/speakers/transcripts, selection mode + seed, `corpus_sha256`, stand-in vs exact corpus |
| `<run>/run_manifest.json` | CLI args, all condition definitions, ASR engine labels (`kind: real`/`heuristic_proxy`), HSR label, git commit, python/platform, package versions |
| `<run>/rows.csv` | append-only per-(condition, utterance) evidence: metrics, WERs, hypotheses, transcripts, error rows (`status`, `error`) |
| `<run>/summary.csv` / `summary.json` | aggregates, ranks, Pareto flags, baselines, labels |
| `<run>/ranking.md` | human-readable report with provenance header |

**Rule of the house:** a number without a manifest entry is not a result.

## 4. Resume / crash safety

- `rows.csv` keys are `(condition_id, utt_id)`; on restart completed tasks
  are skipped (`[tasks] X total, Y already done`).
- Rows are written immediately after each task; killing the process loses at
  most in-flight tasks.
- Aggregation is pure over `rows.csv` — re-running the same command after
  completion simply re-aggregates (and picks up code changes).

## 5. Verifying claims independently

1. Match `git_commit` + package versions in `run_manifest.json`.
2. Re-select the corpus and compare `corpus_sha256`.
3. Spot-check any row: re-apply
   `resolve_transform(transform_key, params)` to the `source_path` file and
   re-transcribe with the labelled engine; hypothesis text is stored.
4. `n_ok`/`n_err` per condition expose partial failures; error rows keep the
   exception message.

## 6. Known non-reproducible-without-permission items

| Item | Substitute |
|---|---|
| WSJ audio (LDC licence) | LibriSpeech stand-in, labelled everywhere; adapter gives bit-exact WSJ once licensed |
| Paper's 120 target texts (private) | manifest-recorded transcripts |
| Paper's music set (source unnamed) | `--music-dir` + manifest entry, or omitted-and-labelled |
| Human listener data | none exists for our corpus; see `HUMAN_STUDY_PROTOCOL.md` — the STOI-based proxy is checked against real listener scores on the public TMHINT-QI corpus (`docs/HUMAN_PROXY_VALIDATION.md`) |

## 7. Continuous integration (WHAT-REMAINS §13)

`.github/workflows/ci.yml` runs two tiers:

| Job | Trigger | What runs | Downloads |
|---|---|---|---|
| `unit` | push / pull request | `pytest tests/ -q` — tests needing Whisper weights, the Vosk model, wav2vec2 weights or LibriSpeech **self-skip** (`skipif` guards in `tests/test_asr.py`); everything else uses synthetic/tmp fixtures | **none** |
| `full` | manual (`workflow_dispatch`, `full=true`) | same suite after installing `.[dev,ssl]` and fetching Whisper tiny (~40 MB, cached) + Vosk small-en (~40 MB, cached) + wav2vec2-base (~360 MB, Hugging Face cache) | ASR weights only |

Smoke vs full locally:

```powershell
# lightweight (what CI's unit job executes; no model downloads):
python -m pytest tests/ -q                       # heavy tests skip if weights/corpora absent
# full local run (all weights + LibriSpeech present):
python -m pytest tests/ -q                       # expect every test to execute, 0 skipped ASR
```
