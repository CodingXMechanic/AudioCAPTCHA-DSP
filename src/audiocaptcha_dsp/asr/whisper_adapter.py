from __future__ import annotations

import logging

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.types import TranscriptionResult
from audiocaptcha_dsp.asr.engine import ASREngine

logger = logging.getLogger(__name__)


class WhisperAdapter(ASREngine):
    def __init__(self, model_size: str = "tiny", device: str = "cpu", language: str = "en") -> None:
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model = None

    def load(self) -> None:
        logger.info("Loading Whisper model: %s on %s", self.model_size, self.device)

    def transcribe(self, signal: Signal) -> TranscriptionResult:
        raise NotImplementedError
