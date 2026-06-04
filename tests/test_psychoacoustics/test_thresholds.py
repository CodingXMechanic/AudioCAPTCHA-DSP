from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.psychoacoustics.thresholds import absolute_threshold
from audiocaptcha_dsp.core.types import ThresholdModel


class TestAbsoluteThreshold:
    def test_returns_array(self) -> None:
        freqs = np.array([250.0, 1000.0, 4000.0])
        with pytest.raises(NotImplementedError):
            absolute_threshold(freqs)
