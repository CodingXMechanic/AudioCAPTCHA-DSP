from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.psychoacoustics.masking_models import simultaneous_masking_threshold
from audiocaptcha_dsp.core.types import MaskingModel


class TestSimultaneousMasking:
    def test_returns_ndarray(self) -> None:
        signal = np.random.randn(16000)
        with pytest.raises(NotImplementedError):
            simultaneous_masking_threshold(signal, sr=16000)
