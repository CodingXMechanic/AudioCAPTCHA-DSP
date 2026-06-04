from __future__ import annotations

import pytest
import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.pipeline import Pipeline


class TestSignal:
    def test_creation(self, clean_signal: Signal) -> None:
        assert clean_signal.waveform.ndim == 1
        assert clean_signal.sample_rate == 16000

    def test_duration(self, clean_signal: Signal) -> None:
        assert pytest.approx(clean_signal.duration_seconds, rel=1e-3) == 1.0

    def test_rms(self, clean_signal: Signal) -> None:
        assert clean_signal.rms > 0.0

    def test_clone_is_independent(self, clean_signal: Signal) -> None:
        cloned = clean_signal.clone()
        cloned.waveform[0] = 999.0
        assert clean_signal.waveform[0] != 999.0

    def test_mono_channel_count(self, clean_signal: Signal) -> None:
        assert clean_signal.num_channels == 1

    def test_stereo_to_mono(self, stereo_signal: Signal) -> None:
        assert stereo_signal.num_channels == 2
        mono = stereo_signal.to_mono()
        assert mono.num_channels == 1
        assert mono.waveform.ndim == 1

    def test_invalid_waveform_dimensions(self, sample_rate: int) -> None:
        bad = np.zeros((2, 3, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="1D or 2D"):
            Signal(waveform=bad, sample_rate=sample_rate)


class TestPipeline:
    def test_add_transform(self, clean_signal: Signal) -> None:
        pipeline = Pipeline(name="test")
        pipeline.add(lambda s: s)
        assert len(pipeline.transforms) == 1

    def test_apply_identity(self, clean_signal: Signal) -> None:
        pipeline = Pipeline(name="identity")
        pipeline.add(lambda s: s)
        result = pipeline.apply(clean_signal)
        assert isinstance(result, Signal)
        assert result.sample_rate == clean_signal.sample_rate

    def test_apply_chain(self, clean_signal: Signal) -> None:
        pipeline = Pipeline(name="chain")
        pipeline.add(lambda s: s)
        pipeline.add(lambda s: s)
        result = pipeline.apply(clean_signal)
        assert isinstance(result, Signal)

    def test_from_config(self, tmp_path: pytest.TempPathFactory) -> None:
        import yaml
        config_path = tmp_path / "test_pipeline.yaml"
        config_path.write_text(yaml.dump({"name": "config_test"}))
        pipeline = Pipeline.from_config(config_path)
        assert pipeline.name == "config_test"
