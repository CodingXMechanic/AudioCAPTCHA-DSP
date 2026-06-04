from __future__ import annotations

import numpy as np


def specific_loudness(
    power_spectrum: np.ndarray,
    freq_bins: np.ndarray,
) -> np.ndarray:
    power_spectrum = np.asarray(power_spectrum, dtype=np.float64)
    freq_bins = np.asarray(freq_bins, dtype=np.float64)
    raise NotImplementedError
