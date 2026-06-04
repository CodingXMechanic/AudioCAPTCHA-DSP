from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal


@pytest.fixture
def sample_rate() -> int:
    return 16000


@pytest.fixture
def duration_seconds() -> float:
    return 1.0


@pytest.fixture
def clean_signal(sample_rate: int, duration_seconds: float) -> Signal:
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    return Signal(waveform=waveform, sample_rate=sample_rate, metadata={"source": "synthetic"})


@pytest.fixture
def silent_signal(sample_rate: int, duration_seconds: float) -> Signal:
    return Signal(
        waveform=np.zeros(int(sample_rate * duration_seconds)),
        sample_rate=sample_rate,
        metadata={"source": "silent"},
    )


@pytest.fixture
def stereo_signal(sample_rate: int, duration_seconds: float) -> Signal:
    n = int(sample_rate * duration_seconds)
    waveform = np.random.randn(2, n).astype(np.float64) * 0.1
    return Signal(waveform=waveform, sample_rate=sample_rate, metadata={"source": "stereo_noise"})
