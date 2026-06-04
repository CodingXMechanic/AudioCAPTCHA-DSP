from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.types import TranscriptionResult
from audiocaptcha_dsp.asr.engine import ASREngine

logger = logging.getLogger(__name__)


@dataclass
class BatchTranscriber:
    engine: ASREngine
    results: list[TranscriptionResult] = field(default_factory=list)

    def transcribe_batch(self, signals: Sequence[Signal]) -> list[TranscriptionResult]:
        self.results = []
        for i, signal in enumerate(signals):
            logger.info("Transcribing %d/%d", i + 1, len(signals))
        return self.results
