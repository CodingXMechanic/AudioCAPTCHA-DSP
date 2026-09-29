# ASR Models

Evaluation engines for the ASR axis. The core design goal versus the base
paper (single white-box Kaldi DNN-HMM) is **three genuinely independent
families**, so that "attack success" means *black-box transfer*, not
overfitting to one recognizer.

---

## 1. Engines used in headline results

| Engine id | Model | Family / lineage | Type | Weights |
|---|---|---|---|---|
| `whisper_tiny` | openai-whisper **tiny** (39 M) | Attention encoder-decoder | **real** | `~/.cache/whisper/tiny.pt` (auto-download) |
| `whisper_small` | openai-whisper **small** (244 M) | Attention encoder-decoder (same family as tiny, larger scale point) | **real** | `~/.cache/whisper/small.pt` (auto-download) |
| `vosk_small_en` | **vosk-model-small-en-us-0.15** (~40 MB) | **Kaldi nnet3** — same toolkit lineage as the base paper's WSJ-recipe DNN-HMM | **real** | `data/raw/vosk/vosk-model-small-en-us-0.15` |
| `wav2vec2_base` | **facebook/wav2vec2-base-960h** (95 M) | **Self-supervised** (wav2vec 2.0) encoder + CTC head — the family of the SSL papers in the survey | **real** | `~/.cache/huggingface/hub/models--facebook--wav2vec2-base-960h` (via the optional `ssl` extra: `pip install -e ".[ssl]"`) |

All engines:

- transcribe any `Signal` (mono conversion + resample to 16 kHz inside the
  adapter — 44.1 kHz inputs verified);
- run deterministically (Whisper `temperature=0`, `fp16=False` on CPU;
  Vosk lattice decoding is deterministic; wav2vec2 uses greedy CTC decoding);
- are **loaded once per worker** and reused (`_worker_init` in
  `experiments/comparative.py`);
- fail loudly: `offline_fallback=False` for Whisper/Vosk and an explicit
  `RuntimeError` in the wav2vec2 adapter mean a missing weight aborts the
  run — never a silent heuristic fallback.

### Why Vosk specifically

The base paper's ASR was the default Kaldi WSJ recipe. Vosk's small English
model is Kaldi-lineage (nnet3 + lattice decoding), i.e. a *modern relative of
the paper's own recognizer*. Cross-family results therefore read as
"attention-based ↔ Kaldi-lineage ↔ self-supervised" transfer, which brackets
the paper's setup from both sides.

### Why wav2vec2

The survey's self-supervised papers (Wav2Vec2/BYOL-A/HuBERT lineage) were
cited but never evaluated — the one-sidedness `LIMITATIONS.md` flagged. Adding
wav2vec2-base turns the comparison into a three-family design: a
supervised attention model, a Kaldi-lineage hybrid, and an SSL model whose
pretraining objective never saw transcripts. Whisper tiny **and** small
provide the within-family scale point (244 M vs 39 M) so robustness gains
from capacity are visible instead of confounded with architecture.

### Headline metric

```
family_delta(f) = mean over engines f of ( WER(condition) − WER(original) )
cross_delta_wer = mean over families f of family_delta(f)
```

Families: `whisper` (tiny + small), `kaldi` (vosk), `ssl` (wav2vec2). Each
lineage therefore weighs exactly once — an attack that fools only Whisper at
both sizes cannot outrank one that hurts all three families. This is the
`attack_rank` ordering key (`_macro_family_delta` /
`asr_family_groups` in `experiments/comparative.py`; `scripts/make_figures.py`
mirrors the same rule for per-utterance deltas).

---

## 2. Engines present but **not** used for conclusions

| Engine | What it is | How it is handled |
|---|---|---|
| `IndependentASREngine` (`asr/engine.py`) | **Heuristic, metadata-driven** — WER is a function of `signal.metadata`, audio content is never recognized | Tagged `asr_kind: heuristic_proxy` in manifests; **excluded from ΔWER/headline claims**; retained only as a deterministic plumbing test double |
| `MockASREngine` | Test double | Tests/CI only |
| Whisper `base` | Same family as tiny/small | Used only as the shelf-life capacity-ladder middle rung (`experiments/shelf_life.py`), not in the headline benchmark |
| HuBERT / WavLM / wav2vec2-large | Same SSL family at other scales | **Not evaluated** — documented residual in `LIMITATIONS.md` (`wav2vec2-base` carries the SSL family in the headline) |
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
and the default engine list are derived from that single registry. Required:
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
