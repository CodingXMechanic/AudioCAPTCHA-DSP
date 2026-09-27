"""Vosk ASR adapter — an independent (non-Whisper) recognition family.

Vosk wraps Kaldi models (nnet3 / lattice decoding), i.e. the same toolkit
lineage as the base paper's white-box ASR (Schönherr et al. 2018 used the
default Kaldi WSJ recipe DNN-HMM).  Pairing Whisper (attention-based
encoder-decoder) with Vosk (Kaldi nnet3) gives the benchmark two genuinely
independent ASR families, which is what cross-model transfer claims require.

Model
-----
Download ``vosk-model-small-en-us-0.15`` (~40 MB) from
https://alphacephei.com/vosk/models and unpack it under ``model_root``
(default ``data/raw/vosk/``).  If neither the package nor the model is
available the adapter degrades to a graceful fallback (empty transcript,
flagged via ``is_fallback``) unless ``offline_fallback=False``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from audiocaptcha_dsp.asr.engine import ASREngine
from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.types import TranscriptionResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "vosk-model-small-en-us-0.15"


class VoskAdapter(ASREngine):
    """Independent ASR engine backed by Vosk (Kaldi-lineage models)."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        model_root: str | Path = "data/raw/vosk",
        model_name: str = DEFAULT_MODEL_NAME,
        sample_rate: int = 16000,
        offline_fallback: bool = True,
    ) -> None:
        super().__init__(name="vosk_small_en")
        self.model_dir = Path(model_dir) if model_dir else Path(model_root) / model_name
        self.sample_rate = sample_rate
        self.offline_fallback = offline_fallback
        self._model = None
        self.is_fallback = False

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import vosk

            vosk.SetLogLevel(-1)
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Vosk model directory not found: {self.model_dir} "
                    f"(download {DEFAULT_MODEL_NAME} from https://alphacephei.com/vosk/models)"
                )
            self._model = vosk.Model(str(self.model_dir))
            logger.info("Vosk model loaded from %s", self.model_dir)
        except Exception as e:
            logger.warning("Failed to load Vosk model: %s", e)
            if not self.offline_fallback:
                raise
            self._model = "fallback"
            self.is_fallback = True

    # ------------------------------------------------------------- transcribe
    def transcribe(self, signal: Signal) -> TranscriptionResult:
        if self._model is None:
            self.load()

        if self._model == "fallback":
            return TranscriptionResult(text="", confidence=0.0, language="en")

        import vosk

        pcm = self._to_int16_pcm(signal)
        if pcm.size == 0:
            return TranscriptionResult(text="", confidence=0.0, language="en")

        rec = vosk.KaldiRecognizer(self._model, self.sample_rate)
        rec.SetWords(False)

        chunks: list[str] = []
        step = 2 * self.sample_rate  # 2 s frames keep latency bounded
        for start in range(0, pcm.size, step):
            if rec.AcceptWaveform(pcm[start : start + step].tobytes()):
                chunks.append(json.loads(rec.Result()).get("text", ""))
        chunks.append(json.loads(rec.FinalResult()).get("text", ""))
        text = " ".join(t for t in chunks if t).strip()

        return TranscriptionResult(
            text=text,
            confidence=None if not text else 1.0,
            language="en",
            segments=[{"start": 0.0, "end": signal.duration_seconds, "text": text}],
        )

    # ----------------------------------------------------------------- helpers
    def _to_int16_pcm(self, signal: Signal) -> np.ndarray:
        """Mono float [-1, 1] → int16 at the model's sample rate."""
        mono = signal.to_mono().waveform.astype(np.float64)
        if mono.size == 0:
            return np.zeros(0, dtype=np.int16)
        sr = signal.sample_rate
        if sr != self.sample_rate:
            from scipy.signal import resample_poly

            g = np.gcd(int(sr), int(self.sample_rate))
            mono = resample_poly(mono, self.sample_rate // g, int(sr) // g)
        peak = float(np.max(np.abs(mono))) or 1.0
        if peak > 1.0:  # guard against >0 dBFS transforms
            mono = mono / peak
        return (mono * 32767.0).astype(np.int16)
