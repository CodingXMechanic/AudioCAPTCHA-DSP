# Threat Model

Who attacks what, with what knowledge, and what the defender does.
Structured so that every benchmark condition maps onto a threat-model cell.

---

## 1. Assets & security goal

- **Asset:** an audio CAPTCHA's security property — *bots fail, humans pass.*
- **Attacker goal (API security):** make an automated speech recognizer
  transcribe correctly (or at all), while the challenge remains answerable by
  humans. Formally: raise ASR success (`asr_sr` ↓ attack, WER ↑) while
  keeping human success high (`hsr` ↑).
- **Inverse reading (audio watermarking / steganography):** the same
  transforms can hide content from ASR while humans recover it — the
  `hag` gap serves both readings; which direction is "good" depends on the
  deployment (declared per experiment, not fixed by the code).

## 2. Attacker models (taxonomy → threat cell)

| Family | Attacker knowledge | Example cells | Base-paper relation |
|---|---|---|---|
| **A** baseline/identity | — | control/degradation sanity cells | — |
| **B** temporal, **C** multirate, **D** spectral, **E** noise/interference | **K0: knowledge-free (black box)** — signal-domain edits, no model access | band-stop, time-stretch, babble, codec | beyond the paper (paper: model-aware only) |
| **F** psychoacoustic | **K1: knows the human auditory model** — constrained optimization against masking thresholds | `masked_noise`, λ-sweep cells, matched-power control | **the paper's own threat model, generalized** (its "λ" and "None" columns) |
| **G** adversarial | **K2: query/gradient access to a surrogate ASR** — search-based, black-box-style objectives | `gradient_free`, `black_box`, `multi_objective`, `phoneme_guided` | ≈ paper's gradient attacks, but *transfer-oriented* (optimized on surrogate, evaluated on Whisper+Vosk) |
| **H** channel/deployment | **channel/relay attacker** — compression, packet loss, reverb, device response | `mp3`, `opus`, `packet_loss`, `room_reverb` | beyond the paper |

Transfer claims (RQ3) are exactly the K2→K0 statement: a transform tuned
against *any* surrogate must succeed against **unseen, architecturally
independent** engines — hence `cross_delta_wer` over Whisper ≠ Vosk.

## 3. Knowledge assumptions (explicit)

- **Human model knowledge:** attacker knows the psychoacoustic model used by
  the defender's imperceptibility criterion (masking thresholds, ATH).
- **No secret-key assumption:** none of the evaluated transforms relies on
  key secrecy; security is claimed only from the *human–machine gap*, which
  is the position of papers 3 and 11.
- **Corpus knowledge:** transcripts of the challenge texts are assumed known
  (CAPTCHA texts are not secret; the base paper publishes its targets too).

## 4. Defender model

| Layer | Mechanism | Where |
|---|---|---|
| Preprocessing defenses | loudness normalization, resampling, spectral denoising, codec round-trip, replay simulation (+ identity) | `asr/defense.py`; defense-aware metrics `DSR`, `CMFR`, `prr` |
| Channel mitigations (natural) | family-H transforms evaluated *as conditions* show which deployment channels erode attacks | `channel.*` |
| CAPTCHA-level | challenge generation with per-condition parameters recorded (replay-safe provenance) | `captcha/generator.py` |

**Out of scope for the defender:** adversarial training of the ASR,
certified robustness (papers 6/8 need white-box training access).

## 5. Adversary capabilities & limits

| Assumed | Not assumed |
|---|---|
| Can choose any registered DSP transform + parameters (incl. composite chains) | physical-world recording (replay cells are simulations) |
| Unlimited offline compute for transform selection | real-time constraint during CAPTCHA serving (cost axis logged, not enforced) |
| May know the exact subset protocol | access to WSJ/licensed corpora beyond public substitutes |
| May combine transforms (composite conditions supported by `runner`) | gradients through Whisper/Vosk internals (engines treated as black boxes end-to-end) |

## 6. Required attacker catalog (WHAT-REMAINS §14)

Every attacker type the spec requires, mapped to this framework's cells and
honest evaluation status:

| Attacker type | Definition | Our evaluation cell | Status |
|---|---|---|---|
| **White-box** | Full access to ASR internals + gradients (the base paper's setting) | Not reimplemented: our design is deliberately black-box; the base paper's white-box results are the comparison point (`docs/GAP_ANALYSIS.md`) | *Not supported (by design)* |
| **Gray-box** | Knows architecture/training lineage, not the exact weights — attacks a **surrogate** | Family G (`adversarial.gradient_free/black_box/multi_objective/phoneme_guided`), `novel.defense_robust`; success = transfer to engines never seen during search | **Implemented** |
| **Black-box** | No model access: knowledge-free signal edits or query-only probing | Families B–E, H (K0) + query-based family G (K2); measured by `cross_delta_wer` | **Implemented** |
| **Replay** | Records challenge audio and resubmits it | `channel.recording_replay` (speaker→room→mic chain), `ReplaySimulationDefense` | **Implemented (simulated chain; physical capture not measured)** |
| **Denoising** | Strips imperceptible perturbations by denoising before recognition | `SpectralDenoisingDefense` + `channel.noise_suppression` as attacker-side preprocessing; measured by `DSR`/`CMFR`/`prr` | **Implemented (defense sweep runs on demand)** |
| **Codec** | Transcodes through perceptual codecs to destroy hidden energy | `channel.mp3/opus/aac/codec_simulation` + `CodecSimulationDefense`; H-axis and ASR-axis measured per codec | **Implemented** |
| **Ensemble ASR** | Jointly targets/evaluates multiple recognizers | Headline metric is a **2-family ensemble** (Whisper + Vosk); `adversarial.multi_objective` searches against a multi-engine objective | **Implemented (2 families; third SSL family *not supported* here)** |
| **Adaptive** | Knows the defense pipeline and tailors the attack to beat it | `novel.defense_robust` (defense-aware composite); defense-aware HAG fields (`DSR`, `CMFR`) | **Implemented (single adaptive design; automated arms-race loop *Planned*)** |
| **Query-limited** | Bounded number of ASR queries during search | `query_budget` parameters on all family-G searches (e.g. 6 / 24 / 48) | **Implemented** |

## 7. Mapping benchmark rows → threat cells

Every row carries `family`, `transform_key`, `params_json` — sufficient to
place it in the table above. The Pareto/`hag` rankings answer:
*which threat cell yields the largest human–machine gap under this
defender set.*

## 8. Non-claims

- No claim of security against human-in-the-loop solvers (CAPTCHA mode
  user study pending — `HUMAN_STUDY_PROTOCOL.md`).
- No claim against future ASR families beyond the two evaluated
  (see `LIMITATIONS.md`).
- "Security" is always conditional on the labelled corpus/engine/defense
  set in the run manifests.
