from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from audiocaptcha_dsp.core.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class AudioDataset:
    root_dir: Path | None = None
    signals: list[Signal] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42), repr=False)

    def load(self) -> tuple[list[Signal], list[str]]:
        if self.signals:
            return self.signals, self.transcripts
        if self.root_dir is not None and self.root_dir.exists():
            return self._load_from_disk()
        return self._generate_synthetic()

    def sample(self, n: int, seed: int = 42) -> tuple[list[Signal], list[str]]:
        signals, transcripts = self.load()
        if n >= len(signals):
            return signals, transcripts
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(signals), size=n, replace=False)
        indices.sort()
        return [signals[i] for i in indices], [transcripts[i] for i in indices]

    def _load_from_disk(self) -> tuple[list[Signal], list[str]]:
        import soundfile as sf
        signals = []
        transcripts = []
        assert self.root_dir is not None
        for wav_path in sorted(self.root_dir.glob("**/*.wav")):
            try:
                wav, sr = sf.read(str(wav_path), dtype="float64", always_2d=False)
                if wav.ndim == 2 and wav.shape[1] == 1:
                    wav = wav.squeeze(axis=1)
                signals.append(Signal(waveform=wav, sample_rate=int(sr), metadata={"source": str(wav_path)}))
                txt_path = wav_path.with_suffix(".txt")
                if txt_path.exists():
                    transcripts.append(txt_path.read_text(encoding="utf-8").strip())
                else:
                    transcripts.append("")
            except Exception as e:
                logger.warning("Failed to load %s: %s", wav_path, e)
        logger.info("Loaded %d signals from %s", len(signals), self.root_dir)
        return signals, transcripts

    def _generate_synthetic(self) -> tuple[list[Signal], list[str]]:
        sr = 16000
        duration = 2.0
        n_samples = int(sr * duration)
        signals = []
        transcripts = []
        base_frequencies = [220.0, 330.0, 440.0, 550.0, 660.0, 880.0, 1100.0, 1320.0]
        for i, freq in enumerate(base_frequencies):
            t = np.linspace(0, duration, n_samples, endpoint=False)
            envelope = np.exp(-2.0 * t / duration)
            waveform = 0.5 * envelope * np.sin(2.0 * np.pi * freq * t)
            noise = self._rng.normal(0, 0.005, size=n_samples)
            waveform = (waveform + noise).astype(np.float64)
            signals.append(Signal(
                waveform=waveform,
                sample_rate=sr,
                metadata={"source": "synthetic", "base_frequency": freq, "sample_id": f"synth_{i:04d}"},
            ))
            transcripts.append(f"synthetic_tone_{freq:.0f}hz")
        logger.info("Generated %d synthetic signals", len(signals))
        return signals, transcripts

    @classmethod
    def synthetic(cls, n_signals: int = 8, seed: int = 42, duration: float = 2.0, sr: int = 16000) -> AudioDataset:
        dataset = cls(root_dir=None, _rng=np.random.default_rng(seed))
        n_samples = int(sr * duration)
        signals = []
        transcripts = []
        rng = np.random.default_rng(seed)
        base_frequencies = np.linspace(150.0, 2000.0, n_signals)
        for i, freq in enumerate(base_frequencies):
            t = np.linspace(0, duration, n_samples, endpoint=False)
            envelope = np.exp(-2.0 * t / duration)
            waveform = 0.5 * envelope * np.sin(2.0 * np.pi * freq * t)
            noise = rng.normal(0, 0.005, size=n_samples)
            waveform = (waveform + noise).astype(np.float64)
            signals.append(Signal(
                waveform=waveform,
                sample_rate=sr,
                metadata={"source": "synthetic", "base_frequency": float(freq), "sample_id": f"synth_{i:04d}"},
            ))
            transcripts.append(f"synthetic_tone_{freq:.0f}hz")
        dataset.signals = signals
        dataset.transcripts = transcripts
        return dataset
