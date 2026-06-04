from __future__ import annotations

from abc import ABC, abstractmethod

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.types import TranscriptionResult


class ASREngine(ABC):
    @abstractmethod
    def transcribe(self, signal: Signal) -> TranscriptionResult:
        ...

    @abstractmethod
    def load(self) -> None:
        ...
