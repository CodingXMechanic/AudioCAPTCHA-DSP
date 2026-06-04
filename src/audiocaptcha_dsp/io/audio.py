from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from audiocaptcha_dsp.core.signal import Signal

logger = logging.getLogger(__name__)


def load_audio(path: Path | str, target_sr: int | None = None) -> Signal:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    waveform, sr = sf.read(str(path), dtype="float64", always_2d=False)
    if waveform.ndim == 2 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(axis=1)
    if target_sr is not None and target_sr != sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    if waveform.ndim == 1:
        waveform = waveform.reshape(1, -1)
    return Signal(
        waveform=waveform,
        sample_rate=int(sr),
        metadata={"source_path": str(path), "source_sr": sr},
    )


def save_audio(signal: Signal, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = signal.to_mono().waveform
    sf.write(str(path), wav, signal.sample_rate, subtype="PCM_16")
    logger.info("Saved audio to %s", path)
