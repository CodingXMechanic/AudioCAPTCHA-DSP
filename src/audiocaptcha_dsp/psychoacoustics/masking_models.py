from __future__ import annotations

import numpy as np

from audiocaptcha_dsp.core.types import MaskingModel


def simultaneous_masking_threshold(
    signal: np.ndarray,
    sr: int,
    model: MaskingModel = MaskingModel.SIMULTANEOUS,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    raise NotImplementedError
