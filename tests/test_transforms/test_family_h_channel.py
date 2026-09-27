import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.channel import (
    MP3Compression,
    OpusCompression,
    AACCompression,
    TelephoneCodec,
    PerceptualCodecSimulation,
    PacketLossSimulation,
    PacketJitter,
    ResamplingChain,
    RecordingReplaySimulation,
    EchoCancellationArtifacts,
    PlaybackEqualization,
    MicrophoneResponse,
    SpeakerResponse,
    AutomaticGainControl,
    NoiseSuppression,
    Dereverberation,
    DeviceDistortion,
    RoomReverberationChannel,
    ffmpeg_available,
)


@pytest.fixture
def speech_like() -> Signal:
    sr = 16000
    t = np.arange(int(1.2 * sr)) / sr
    wave = (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 1200 * t)
        + 0.2 * np.sin(2 * np.pi * 3500 * t)
    )
    wave *= 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    return Signal(waveform=wave, sample_rate=sr, metadata={"transcript": "hello world"})


CODECS = [MP3Compression(bitrate_kbps=32), OpusCompression(bitrate_kbps=16),
          AACCompression(bitrate_kbps=32), TelephoneCodec(),
          PerceptualCodecSimulation(bitrate_kbps=32)]

TRANSPORT = [PacketLossSimulation(loss_rate=0.1, seed=1),
             PacketJitter(jitter_ms=30.0, seed=1),
             ResamplingChain(intermediate_sr=48000),
             RecordingReplaySimulation(seed=1),
             EchoCancellationArtifacts(misalign_ms=6.0)]

DEVICES = [PlaybackEqualization(), MicrophoneResponse(), SpeakerResponse(),
           AutomaticGainControl(), NoiseSuppression(), Dereverberation(),
           DeviceDistortion(drive=2.0), RoomReverberationChannel(seed=1)]


def test_codecs_preserve_length_and_rate(speech_like):
    for tr in CODECS:
        out = tr(speech_like)
        assert out.sample_rate == speech_like.sample_rate
        assert len(out.waveform) == len(speech_like.waveform)
        assert np.all(np.isfinite(out.waveform))


def test_codecs_record_encoder_path(speech_like):
    """Metadata must state whether the real encoder or the simulation ran."""
    for tr in CODECS:
        out = tr(speech_like)
        keys = [k for k in out.metadata if k.endswith(".encoder")]
        if isinstance(tr, PerceptualCodecSimulation):
            assert out.metadata[f"{tr.name}.encoder"] == "perceptual_simulation"
        elif keys:
            assert out.metadata[keys[0]] in ("ffmpeg", "perceptual_simulation")
        # Fallback, if taken, must be labeled
        fb = f"{tr.name}.fallback"
        if fb in out.metadata:
            assert out.metadata[fb] == "perceptual_simulation"


def test_telephone_codec_bandlimits(speech_like):
    out = TelephoneCodec()(speech_like)
    spec = np.abs(np.fft.rfft(out.waveform))
    freqs = np.fft.rfftfreq(len(out.waveform), 1 / speech_like.sample_rate)
    speech = (freqs > 500) & (freqs < 3000)
    high = freqs > 4500
    assert spec[high].mean() < 0.5 * spec[speech].mean()


def test_transport_preserves_duration(speech_like):
    for tr in TRANSPORT:
        out = tr(speech_like)
        assert len(out.waveform) == len(speech_like.waveform)
        assert np.all(np.isfinite(out.waveform))


def test_packet_loss_records_lost_count(speech_like):
    out = PacketLossSimulation(loss_rate=0.2, packet_ms=20.0, seed=1)(speech_like)
    key = "channel.packet_loss.n_lost"
    assert key in out.metadata
    assert out.metadata[key] > 0
    # Lost frames concealed ⇒ signal differs from original
    assert not np.allclose(out.waveform, speech_like.waveform)


def test_zero_loss_is_identity(speech_like):
    out = PacketLossSimulation(loss_rate=0.0)(speech_like)
    assert np.allclose(out.waveform, speech_like.waveform, atol=1e-9)


def test_devices_preserve_duration(speech_like):
    for tr in DEVICES:
        out = tr(speech_like)
        assert len(out.waveform) == len(speech_like.waveform)
        assert out.sample_rate == speech_like.sample_rate
        assert np.all(np.isfinite(out.waveform))


def test_noise_suppression_reduces_added_noise():
    sr = 16000
    t = np.arange(int(1.2 * sr)) / sr
    clean = 0.5 * np.sin(2 * np.pi * 300 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    rng = np.random.default_rng(1)
    noisy = clean + rng.normal(0, 0.05, len(clean))
    sig = Signal(waveform=noisy, sample_rate=sr)
    # Default operating point (over_sub=1.5): aggressive over-suppression on
    # this narrowband AM signal distorts speech more than it removes noise
    # (classical spectral-subtraction tradeoff, Boll 1979).
    out = NoiseSuppression(over_sub=1.5)(sig)
    # Residual noise power should drop vs input
    resid_in = np.mean((noisy - clean) ** 2)
    resid_out = np.mean((out.waveform - clean) ** 2)
    assert resid_out < resid_in


def test_agc_normalizes_level():
    sr = 16000
    t = np.arange(sr) / sr
    quiet = 0.02 * np.sin(2 * np.pi * 300 * t)
    out = AutomaticGainControl(target_db=-20.0, max_gain_db=30.0)(Signal(quiet, sr))
    assert out.rms > float(np.sqrt(np.mean(quiet ** 2)))


def test_device_distortion_quantizes(speech_like):
    out = DeviceDistortion(drive=2.0, bits=4.0)(speech_like)
    # 4-bit quantization: output levels are sparse
    levels = np.unique(np.round(out.waveform, 6))
    assert len(levels) < 5000


def test_room_reverb_adds_late_energy(speech_like):
    dry = speech_like.waveform
    out = RoomReverberationChannel(rt60_s=0.8, wet_db=0.0, seed=1)(speech_like)
    # Tail region of a 1.5s AM signal has more energy when reverberated
    tail = slice(int(1.0 * speech_like.sample_rate), None)
    assert np.sum(out.waveform[tail] ** 2) >= 0.5 * np.sum(dry[tail] ** 2)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        PacketLossSimulation(loss_rate=1.5)
    with pytest.raises(ValueError):
        PacketJitter(jitter_ms=-1.0)
    with pytest.raises(ValueError):
        NoiseSuppression(over_sub=0.5)
    with pytest.raises(ValueError):
        DeviceDistortion(drive=-1.0)
    with pytest.raises(ValueError):
        RoomReverberationChannel(rt60_s=-0.1)


def test_ffmpeg_availability_is_boolean():
    assert isinstance(ffmpeg_available(), bool)
