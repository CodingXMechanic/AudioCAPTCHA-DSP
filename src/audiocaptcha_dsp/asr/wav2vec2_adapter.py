"""wav2vec 2.0 ASR adapter — an independent self-supervised (SSL) family.

wav2vec 2.0 (Baevski et al., NeurIPS 2020; papers 19-21 of the survey) is a
self-supervised speech representation model with a CTC head fine-tuned on
LibriSpeech.  In the targeted transfer-validation run it adds a *third*
architecture to the benchmark alongside Whisper (attention encoder-decoder)
and Vosk (Kaldi nnet3): a convolutional feature encoder + Transformer
encoder decoding with connectionist temporal classification, so the
cross-family transfer claim on the top-ranked conditions spans attention,
Kaldi-lineage and SSL models.

Model
-----
``facebook/wav2vec2-base-960h`` (~360 MB) is fetched from the Hugging Face
Hub on first use and cached under ``~/.cache/huggingface``.  Decoding is
greedy CTC argmax (deterministic), model runs in ``eval()`` mode under
``torch.no_grad()`` on CPU by default.

Unlike the heuristic ``IndependentASREngine`` (test double only, excluded
from every delta-WER claim), this adapter performs real recognition; a
missing dependency or missing weights raises instead of degrading to a
silent fallback, so a broken environment cannot masquerade as a strong or
weak recognizer in published numbers.
"""
from __future__ import annotations

import logging
import re
import time

import numpy as np

from audiocaptcha_dsp.asr.engine import ASREngine
from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.types import TranscriptionResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "facebook/wav2vec2-base-960h"
TARGET_SAMPLE_RATE = 16000


def clean_ctc_text(text: str) -> str:
    """Turn raw CTC output into a readable transcript.

    The wav2vec2 vocab uses ``|`` as the word delimiter; collapse those to
    single spaces and trim.  Casing/punctuation are left as the model emits
    them (``normalize_transcript`` is applied later, at WER time).
    """
    text = text.replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


class Wav2Vec2Adapter(ASREngine):
    """wav2vec 2.0 (SSL, CTC head) ASR adapter with deterministic decoding."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        super().__init__(name="wav2vec2_base")
        self.model_id = model_id
        self.device = device
        self.local_files_only = local_files_only
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading wav2vec2 model: %s on %s", self.model_id, self.device)
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as e:  # transformers is an optional dependency
            raise RuntimeError(
                "wav2vec2 engine requires the optional 'transformers' package "
                "(pip install -e '.[ssl]'); refusing to substitute a "
                "non-recognizing fallback for a benchmark engine"
            ) from e
        try:
            self._processor = Wav2Vec2Processor.from_pretrained(
                self.model_id, local_files_only=self.local_files_only
            )
            self._model = Wav2Vec2ForCTC.from_pretrained(
                self.model_id, local_files_only=self.local_files_only
            )
        except Exception as e:
            raise RuntimeError(
                f"could not load wav2vec2 weights for {self.model_id!r} "
                f"(cached under the Hugging Face hub cache): {e}"
            ) from e
        self._model.to(self.device)
        self._model.eval()
        logger.info("wav2vec2 model '%s' loaded successfully.", self.model_id)

    # ------------------------------------------------------------- transcribe
    def transcribe(self, signal: Signal) -> TranscriptionResult:
        if self._model is None:
            self.load()

        import torch
        import librosa

        start_time = time.perf_counter()
        mono = signal.to_mono().waveform.astype(np.float32)

        # The processor expects 16 kHz mono
        if signal.sample_rate != TARGET_SAMPLE_RATE:
            mono = librosa.resample(
                mono, orig_sr=signal.sample_rate, target_sr=TARGET_SAMPLE_RATE
            )
        if mono.size == 0:
            return TranscriptionResult(text="", confidence=0.0, language="en")

        with torch.no_grad():
            inputs = self._processor(
                mono, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
            ).input_values.to(self.device)
            logits = self._model(inputs).logits  # (1, T, vocab)
            # Confidence: mean over frames of the max posterior (CTC greedy)
            probs = torch.softmax(logits, dim=-1)
            confidence = float(probs.max(dim=-1).values.mean().item())
            token_ids = logits.argmax(dim=-1)

        raw_text = self._processor.batch_decode(token_ids)[0]
        text = clean_ctc_text(raw_text)

        elapsed = time.perf_counter() - start_time
        logger.debug(
            "wav2vec2 transcribed in %.2fs (conf %.3f): '%s'",
            elapsed, confidence, text[:80],
        )
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            segments=[{
                "start": 0.0,
                "end": signal.duration_seconds,
                "text": text,
            }],
            language="en",
        )
