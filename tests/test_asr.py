from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.asr import (
    MockASREngine,
    WhisperAdapter,
    BatchTranscriber,
    normalize_transcript,
    DefensePipeline,
    IdentityDefense,
    LoudnessNormDefense,
    ResamplingDefense,
    SpectralDenoisingDefense,
    CodecSimulationDefense,
    ReplaySimulationDefense,
)


class TestASR:
    def test_normalize_transcript(self) -> None:
        assert normalize_transcript("Hello, World! 123") == "hello world 123"
        assert normalize_transcript("  Multiple   Spaces  ") == "multiple spaces"
        assert normalize_transcript("") == ""

    def test_mock_asr_engine(self) -> None:
        sig = Signal(waveform=np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)), sample_rate=16000)
        engine = MockASREngine(target_transcript="test challenge speech")
        res = engine.transcribe(sig)
        assert res.text == "test challenge speech"
        assert res.confidence > 0.8
        assert res.language == "en"

    def test_mock_asr_silence(self) -> None:
        sig = Signal(waveform=np.zeros(16000), sample_rate=16000)
        engine = MockASREngine()
        res = engine.transcribe(sig)
        assert res.text == ""
        assert res.confidence == 0.0

    def test_whisper_adapter_fallback(self) -> None:
        if not _whisper_weights_available():
            pytest.skip("Whisper tiny weights not cached (CI must not download models)")
        sig = Signal(waveform=0.1 * np.random.randn(16000), sample_rate=16000, metadata={"transcript": "hello"})
        adapter = WhisperAdapter(model_size="tiny", offline_fallback=True)
        # Should not raise exception
        res = adapter.transcribe(sig)
        assert isinstance(res.text, str)
        assert res.confidence is not None

    def test_batch_transcriber(self) -> None:
        sig1 = Signal(waveform=0.1 * np.random.randn(8000), sample_rate=16000)
        sig2 = Signal(waveform=0.1 * np.random.randn(8000), sample_rate=16000)
        engine = MockASREngine(target_transcript="batch test")
        batch = BatchTranscriber(engine=engine)
        results = batch.transcribe_batch([sig1, sig2])
        assert len(results) == 2
        assert len(batch.latencies) == 2
        assert batch.mean_latency() >= 0.0

    def test_all_defenses(self) -> None:
        t = np.linspace(0, 1, 16000, endpoint=False)
        sig = Signal(waveform=0.5 * np.sin(2 * np.pi * 440 * t), sample_rate=16000)
        defenses = DefensePipeline.all_defenses()
        assert len(defenses) == 6
        
        for defense in defenses:
            defended = defense(sig)
            assert isinstance(defended, Signal)
            assert defended.sample_rate == sig.sample_rate
            assert len(defended.waveform) == len(sig.waveform)
            assert not np.any(np.isnan(defended.waveform))
            assert not np.any(np.isinf(defended.waveform))
            assert "defense" in defended.metadata


def _whisper_weights_available() -> bool:
    """Whisper caches tiny.pt under ~/.cache/whisper; CI must not download it."""
    from pathlib import Path

    return (Path.home() / ".cache" / "whisper" / "tiny.pt").exists()


def _vosk_model_available() -> bool:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    return any(
        (base / "data" / "raw" / "vosk"
         / "vosk-model-small-en-us-0.15").exists()
        for base in (Path.cwd(), repo)
    )


class TestVoskAdapter:
    """Independent (Kaldi-lineage) ASR family used by the benchmark."""

    def test_missing_model_graceful_fallback(self, tmp_path) -> None:
        from audiocaptcha_dsp.asr import VoskAdapter

        adapter = VoskAdapter(model_dir=tmp_path / "nope", offline_fallback=True)
        sig = Signal(waveform=0.1 * np.random.randn(16000), sample_rate=16000)
        res = adapter.transcribe(sig)
        assert adapter.is_fallback is True
        assert res.text == ""

    def test_missing_model_strict_raises(self, tmp_path) -> None:
        from audiocaptcha_dsp.asr import VoskAdapter

        adapter = VoskAdapter(model_dir=tmp_path / "nope", offline_fallback=False)
        with pytest.raises(FileNotFoundError):
            adapter.load()

    @pytest.mark.skipif(not _vosk_model_available(),
                        reason="Vosk model not downloaded")
    def test_transcribes_speech_like_signal(self) -> None:
        from pathlib import Path

        import soundfile as sf

        from audiocaptcha_dsp.asr import VoskAdapter

        files = sorted(
            (Path(__file__).resolve().parents[1]
             / "data/raw/LibriSpeech/test-clean").glob("*/*/*.flac")
        )
        if not files:
            pytest.skip("LibriSpeech test-clean not present")
        wav, sr = sf.read(str(files[0]), dtype="float64")
        adapter = VoskAdapter()
        res = adapter.transcribe(Signal(waveform=wav, sample_rate=sr))
        assert adapter.is_fallback is False
        assert isinstance(res.text, str)
        assert len(res.text.split()) >= 3  # real recognizer on clean speech

    @pytest.mark.skipif(not _vosk_model_available(),
                        reason="Vosk model not downloaded")
    def test_resamples_non_16k_input(self) -> None:
        from audiocaptcha_dsp.asr import VoskAdapter

        adapter = VoskAdapter()
        sig = Signal(
            waveform=0.1 * np.random.randn(44100), sample_rate=44100
        )
        res = adapter.transcribe(sig)  # must not raise
        assert isinstance(res.text, str)
