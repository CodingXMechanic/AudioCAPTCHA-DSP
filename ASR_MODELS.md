# ASR Models

Evaluation engines for the ASR axis. The core design goal versus the base
paper (single white-box Kaldi DNN-HMM) is **independent ASR families**, so
that "attack success" means *black-box transfer*, not overfitting to one
recognizer.

---

## 1. Engines used in headline results

The headline matrix (`results/comparative/main`: 147 conditions × 70
utterances) is evaluated on **two genuinely independent families**:

| Engine id | Model | Family / lineage | Type | Weights |
|---|---|---|---|---|
| `whisper_tiny` | openai-whisper **tiny** (39 M) | Attention encoder-decoder | **real** | `~/.cache/whisper/tiny.pt` (auto-download) |
| `vosk_small_en` | **vosk-model-small-en-us-0.15** (~40 MB) | **Kaldi nnet3** — same toolkit lineage as the base paper's WSJ-recipe DNN-HMM | **real** | `data/raw/vosk/vosk-model-small-en-us-0.15` |

These two are the defaults (`DEFAULT_ENGINES`), so the documented benchmark
command reproduces the headline table as-is.

### Transfer-validation engines (top-K subset)

The targeted run `results/comparative/engine_validation` extends the same
protocol to **four engines** for exactly the conditions the paper discusses
— the top-8 attacks by cross-ΔWER, the matched-power control, a benign
near-null condition (`baseline.loudness_normalize`) and `original`
(11 conditions × 70 utterances, no λ-sweep):

| Engine id | Model | Family / lineage | Weights |
|---|---|---|---|
| `whisper_small` | openai-whisper **small** (244 M) | Attention encoder-decoder (same family as tiny, larger scale point) | `~/.cache/whisper/small.pt` (auto-download) |
| `wav2vec2_base` | **facebook/wav2vec2-base-960h** (95 M) | **Self-supervised** (wav2vec 2.0) encoder + CTC head — the family of the SSL papers in the survey | `~/.cache/huggingface/hub/models--facebook--wav2vec2-base-960h` (optional `ssl` extra: `pip install -e ".[ssl]"`) |

Selected explicitly with
`--asr whisper_tiny,vosk_small_en,whisper_small,wav2vec2_base`. It answers
one question: *do the headline attacks transfer to a larger attention model
and to an SSL model whose pretraining never saw transcripts?* The full
147-condition × 4-engine matrix was **not** run (≈ 35 h on this hardware) —
declared as a limitation, never implied.

Result: `results/comparative/engine_validation/` (770 rows, 0 errors) with
tables in `results/tables/validation/`; the headline `#1` attack
(`spectral.minimum_phase`) stays `#1` on four engines (cross-ΔWER 1.119)
and the ranking agrees with the 2-engine matrix at ρ = 0.964.

All four engines:

- transcribe any `Signal` (mono conversion + resample to 16 kHz inside the
  adapter — 44.1 kHz inputs verified);
- run deterministically for Whisper (`temperature=0`, `fp16=False` on CPU)
  and wav2vec2 (greedy CTC); Vosk re-runs with an *identical*
  configuration reproduce (A/B check: 0/15), while runs under a different
  worker/engine configuration flip a small share of near-tied lattices on
  noise-like conditions (106/770 hypotheses, ≤ 1.8 pp condition-level
  WER — `LIMITATIONS.md` §3);
- are **loaded once per worker** and reused (`_worker_init` in
  `experiments/comparative.py`);
- fail loudly: `offline_fallback=False` for Whisper/Vosk and an explicit
  `RuntimeError` in the wav2vec2 adapter mean a missing weight aborts the
  run — never a silent heuristic fallback.

### Why Vosk specifically

The base paper's ASR was the default Kaldi WSJ recipe. Vosk's small English
model is Kaldi-lineage (nnet3 + lattice decoding), i.e. a *modern relative of
the paper's own recognizer*. Cross-family results therefore read as
"attention-based ↔ Kaldi-lineage" transfer in the headline, extended to
"↔ self-supervised" in the validation run — bracketing the paper's setup
from both sides.

### Why wav2vec2

The survey's self-supervised papers (wav2vec 2.0 / HuBERT lineage) were
cited but never evaluated — the one-sidedness `LIMITATIONS.md` flagged.
wav2vec2-base carries that family in the validation run, turning the
transfer question into a three-architecture comparison: a supervised
attention model, a Kaldi-lineage hybrid, and an SSL model. Whisper tiny
**and** small supply the within-family scale point (244 M vs 39 M) so
robustness gains from capacity are visible instead of confounded with
architecture.

### Headline metric

```
delta(e)        = WER(condition, e) − WER(original, e)      per engine e
cross_delta_wer = mean over the engines evaluated in the run
```

For the headline that is the plain mean over the two independent families —
the `attack_rank` ordering key. When a run measures one family at more than
one size (the four-engine validation), the rule is the **family
macro-average**: engines of a lineage are averaged first, then the family
means are averaged (`_macro_family_delta` / `asr_family_groups` in
`experiments/comparative.py`; `scripts/make_figures.py` mirrors it for
per-utterance deltas), so Whisper at two sizes cannot outweigh Kaldi or SSL.

---

## 2. Engines present but **not** used for conclusions

| Engine | What it is | How it is handled |
|---|---|---|
| `IndependentASREngine` (`asr/engine.py`) | **Heuristic, metadata-driven** — WER is a function of `signal.metadata`, audio content is never recognized | Tagged `asr_kind: heuristic_proxy` in manifests; **excluded from ΔWER/headline claims**; retained only as a deterministic plumbing test double |
| `MockASREngine` | Test double | Tests/CI only |
| Whisper `base` | Same family as tiny/small | Used only as the shelf-life capacity-ladder middle rung (`experiments/shelf_life.py`), not in the headline benchmark |
| HuBERT / WavLM / wav2vec2-large | Same SSL family at other scales | **Not evaluated** — documented residual in `LIMITATIONS.md` (`wav2vec2-base` carries the SSL family in the validation run) |
| Kaldi DNN-HMM (exact paper model) | The paper's white-box ASR | Requires WSJ training recipe + corpus; Vosk stands in as the Kaldi-lineage family (`scripts/prepare_wsj.py --build-kaldi` unlocks the exact path) |

## 3. Defenses (evaluated preprocessors, not ASR engines)

`asr/defense.py`: `IdentityDefense`, `LoudnessNormDefense`,
`ResamplingDefense`, `SpectralDenoisingDefense`, `CodecSimulationDefense`,
`ReplaySimulationDefense`, composed via `DefensePipeline`. Used by the
defense-aware HAG fields (`DSR`, `CMFR`) and WHAT-REMAINS §5 condition list;
off by default in the headline benchmark (6× ASR cost).

## 4. Normalization & WER rules

- `normalize_transcript`: lowercase → strip punctuation → collapse spaces —
  applied to **both** reference and hypothesis before jiwer.
- LibriSpeech/WSJ adapter transcripts arrive pre-normalized; engine outputs
  are normalized in the benchmark row writer.
- Baseline per engine = mean WER of the `original` condition; ΔWER is
  computed against the same engine's baseline (never across engines).

## 5. Adding an engine

```python
class MyAdapter(ASREngine):
    def load(self): ...
    def transcribe(self, signal: Signal) -> TranscriptionResult: ...
```

Register the adapter in `asr/__init__.py`, then add its id to `ASR_LABELS`
(family label + `group` for the macro-average) and to the `_make_engine`
factory in `comparative.py`; `ROW_FIELDS` (`wer_<id>` / `hyp_<id>` columns)
are derived from that single registry. `DEFAULT_ENGINES` (the two-family
headline) and any explicit `--asr` list (e.g. the four-engine validation
set) are both drawn from it. Required:
`kind: real` must be earned — the heuristic engine's exclusion rule exists
because metadata-driven WER must never look like recognition. Resume safety:
`rows.csv` produced with one engine set cannot be continued with another
(`_check_engine_set` refuses the mix rather than silently averaging across
two definitions of the headline metric).

## 6. Capacity ladder (shelf-life novelty N11)

The shelf-life pass (`experiments/shelf_life.py`) evaluates one architecture
at three sizes so the capacity axis is not confounded by architecture:

| Model | Size | Params (M) | Role |
|---|---|---:|---|
| Whisper tiny | `tiny` | 39 | ladder rung 1 |
| Whisper base | `base` | 74 | ladder rung 2 |
| Whisper small | `small` | 244 | ladder rung 3 + reference capacity C_ref |
| Vosk small-en | — | n/a | **out-of-family marker only** — never mixed into the scaling fit |

All ladder engines are real (`WhisperAdapter`, `offline_fallback=False` in
this pass — a missing weight is an error, not a heuristic fallback). The fit
uses only the three Whisper rungs; Vosk is reported alongside as an
architecturally independent observation.
