from __future__ import annotations

import numpy as np

from audiocaptcha_dsp.core.types import ThresholdModel


def absolute_threshold(
    freq_hz: np.ndarray,
    model: ThresholdModel = ThresholdModel.ISO226,
) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    raise NotImplementedError
