from __future__ import annotations

import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.composite.chain import TransformChain


class TestTransformChain:
    def test_empty_chain_returns_input(self, clean_signal: Signal) -> None:
        chain = TransformChain(name="empty")
        result = chain(clean_signal)
        assert isinstance(result, Signal)

    def test_chain_applies_in_order(self, clean_signal: Signal) -> None:
        chain = TransformChain(name="identity_chain")
        chain.transforms = [lambda s: s, lambda s: s]
        result = chain(clean_signal)
        assert isinstance(result, Signal)
        assert result.sample_rate == clean_signal.sample_rate
