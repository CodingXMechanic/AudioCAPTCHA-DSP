# ASR Models

Evaluation engines for the ASR axis. The core design goal versus the base
paper (single white-box Kaldi DNN-HMM) is **two genuinely independent
families**, so that "attack success" means *black-box transfer*, not
overfitting to one recognizer.

---

## 1. Engines used in headline results

| Engine id | Model | Family / lineage | Type | Weights |
|---|---|---|---|---|
| `whisper_tiny` | openai-whisper **tiny** | Attention encoder-decoder (encoder-decoder Transformers) | **real** | `~/.cache/whisper/tiny.pt` (auto-download) |
| `vosk_small_en` | **vosk-model-small-en-us-0.15** (~40 MB) | **Kaldi nnet3** — same toolkit lineage as the base paper's WSJ-recipe DNN-HMM | **real** | `data/raw/vosk/vosk-model-small-en-us-0.15` |

Both engines:

- transcribe any `Signal` (mono conversion + resample to 16 kHz inside the
  adapter — 44.1 kHz inputs verified);
- run deterministically (Whisper `temperature=0`, `fp16=False` on CPU;
  Vosk lattice decoding is deterministic);
- are **loaded once per worker** and reused (`_worker_init` in
  `experiments/comparative.py`).

### Why Vosk specifically

The base paper's ASR was the default Kaldi WSJ recipe. Vosk's small English
model is Kaldi-lineage (nnet3 + lattice decoding), i.e. a *modern relative of
the paper's own recognizer*. Cross-family results therefore read as
"attention-based ↔ Kaldi-lineage" transfer, which brackets the paper's setup
from both sides.

### Headline metric

```
cross_delta_wer = mean over engines of ( WER(condition) − WER(original) )
```

An attack that fools only one family scores roughly half of one that fools
both — this is the `attack_rank` ordering key.

---

## 2. Engines present but **not** used for conclusions

| Engine | What it is | How it is handled |
|---|---|---|
| `IndependentASREngine` (`asr/engine.py`) | **Heuristic, metadata-driven** — WER is a function of `signal.metadata`, audio content is never recognized | Tagged `asr_kind: heuristic_proxy` in manifests; **excluded from ΔWER/headline claims**; retained only as a deterministic plumbing test double |
| `MockASREngine` | Test double | Tests/CI only |
| Whisper `base`/`small` | Same family, larger | *Not present in this environment* (only tiny weights cached); a size-scaling study would be same-family, not independent |
| SSL models (wav2vec 2.0 / HuBERT — papers 19–21) | Independent family | **Not supported here** (`transformers` not installed); documented in `LIMITATIONS.md` |
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

Register in `comparative._worker_init` + `ROW_FIELDS` (`wer_<id>`,
`hyp_<id>` columns) and add an honest label to `ASR_LABELS`. Required:
`kind: real` must be earned — the heuristic engine's exclusion rule exists
because metadata-driven WER must never look like recognition.

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
