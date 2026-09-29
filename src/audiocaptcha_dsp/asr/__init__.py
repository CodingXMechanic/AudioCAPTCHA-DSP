from __future__ import annotations

from audiocaptcha_dsp.asr.engine import ASREngine, MockASREngine, normalize_transcript
from audiocaptcha_dsp.asr.whisper_adapter import WhisperAdapter
from audiocaptcha_dsp.asr.vosk_adapter import VoskAdapter
from audiocaptcha_dsp.asr.wav2vec2_adapter import Wav2Vec2Adapter
from audiocaptcha_dsp.asr.batch import BatchTranscriber
from audiocaptcha_dsp.asr.defense import (
    ASRDefense,
    IdentityDefense,
    LoudnessNormDefense,
    ResamplingDefense,
    SpectralDenoisingDefense,
    CodecSimulationDefense,
    ReplaySimulationDefense,
    DefensePipeline,
)

__all__ = [
    "ASREngine",
    "MockASREngine",
    "WhisperAdapter",
    "VoskAdapter",
    "Wav2Vec2Adapter",
    "BatchTranscriber",
    "normalize_transcript",
    "ASRDefense",
    "IdentityDefense",
    "LoudnessNormDefense",
    "ResamplingDefense",
    "SpectralDenoisingDefense",
    "CodecSimulationDefense",
    "ReplaySimulationDefense",
    "DefensePipeline",
]
