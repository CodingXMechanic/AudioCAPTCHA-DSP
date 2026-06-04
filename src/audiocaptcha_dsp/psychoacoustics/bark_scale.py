from __future__ import annotations

import numpy as np

from audiocaptcha_dsp.core.types import BarkScaleMethod


def hz_to_bark(freq_hz: np.ndarray, method: BarkScaleMethod = BarkScaleMethod.TRAUNMULLER) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    if method == BarkScaleMethod.TRAUNMULLER:
        return 26.81 * freq_hz / (1960.0 + freq_hz) - 0.53
    elif method == BarkScaleMethod.ZWICKER:
        return 13.0 * np.arctan(0.00076 * freq_hz) + 3.5 * np.arctan((freq_hz / 7500.0) ** 2)
    elif method == BarkScaleMethod.WANG:
        return 6.0 * np.arcsinh(freq_hz / 600.0)
    raise ValueError(f"Unknown Bark scale method: {method}")


def bark_to_hz(bark: np.ndarray, method: BarkScaleMethod = BarkScaleMethod.TRAUNMULLER) -> np.ndarray:
    bark = np.asarray(bark, dtype=np.float64)
    if method == BarkScaleMethod.TRAUNMULLER:
        return 1960.0 * (bark + 0.53) / (26.81 - bark - 0.53)
    elif method == BarkScaleMethod.ZWICKER:
        raise NotImplementedError("Zwicker inverse requires numerical inversion")
    elif method == BarkScaleMethod.WANG:
        return 600.0 * np.sinh(bark / 6.0)
    raise ValueError(f"Unknown Bark scale method: {method}")
