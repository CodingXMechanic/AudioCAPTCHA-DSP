"""Family H: transport/channel-deployment transformations.
============================================================
WHAT-REMAINS.txt §3.H gaps implemented here:

- RTP-style packet loss with packet-loss concealment (PLC)
- Packet jitter / reordering
- Deployment resampling chains (8k→48k→16k)
- Recording-and-replay (speaker→room→microphone) chain
- Imperfect echo-cancellation (AEC) residual artifacts

References
----------
- Rosenberg (1999). RTP payload format for audio/video concealing losses
  (PLC practice). RFC 2198.
- ITU-T G.117 — echo control; 3GPP TS 26.131 — terminal acoustic specs.
- ITU-T P.340 — transmission characteristics of hands-free terminals.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform


def _safe(y: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _as_mono_1d(signal: Signal) -> np.ndarray:
    x = np.asarray(signal.waveform, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=0)
    return x


class PacketLossSimulation(BaseTransform):
    """RTP-style packet loss with simple packet-loss concealment (PLC).

    The stream is cut into ``packet_ms`` frames; a fraction ``loss_rate``
    of frames is marked lost. Concealment repeats the previous frame with
    a decaying fade (comfort-noise tail), modeling G.711-style PLC.
    Duration is exactly preserved.
    """

    def __init__(
        self,
        loss_rate: float = 0.05,
        packet_ms: float = 20.0,
        name: str = "channel.packet_loss",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if not (0.0 <= loss_rate < 1.0):
            raise ValueError("loss_rate must be in [0, 1)")
        if packet_ms <= 0:
            raise ValueError("packet_ms must be > 0")
        self.loss_rate = float(loss_rate)
        self.packet_ms = float(packet_ms)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        if self.loss_rate == 0.0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        frame = max(1, int(self.packet_ms * sr / 1000.0))
        n_frames = len(x) // frame
        if n_frames < 2:
            return signal.clone()
        lost = rng.random(n_frames) < self.loss_rate
        y = x.copy()
        prev_valid = None
        for i in range(n_frames):
            seg_slice = slice(i * frame, (i + 1) * frame)
            if not lost[i]:
                prev_valid = y[seg_slice].copy()
                continue
            if prev_valid is None:
                # No previous frame to conceal with: silence + slight noise
                y[seg_slice] = rng.normal(0, np.std(x) * 0.05, frame)
                prev_valid = y[seg_slice].copy()
                continue
            # PLC: repeat previous frame with exponential decay + low noise
            t = np.linspace(0.0, 1.0, frame)
            concealed = prev_valid * np.exp(-3.0 * t)
            concealed += rng.normal(0, np.std(x) * 0.02, frame)
            y[seg_slice] = concealed
            prev_valid = concealed
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={
                **signal.metadata,
                f"{self.name}.loss_rate": self.loss_rate,
                f"{self.name}.n_lost": int(np.count_nonzero(lost)),
                f"{self.name}.packet_ms": self.packet_ms,
            },
        )


class PacketJitter(BaseTransform):
    """Packet jitter / reordering: delayed frames are replayed out of order.

    Each ``packet_ms`` frame receives a random delay in ``[0, jitter_ms]``
    (quantized to frame multiples), modeling network jitter buffers that
    occasionally reorder or late-deliver packets. Trailing edge is padded
    to preserve duration.
    """

    def __init__(
        self,
        jitter_ms: float = 30.0,
        packet_ms: float = 20.0,
        name: str = "channel.packet_jitter",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if jitter_ms < 0 or packet_ms <= 0:
            raise ValueError("jitter_ms >= 0 and packet_ms > 0 required")
        self.jitter_ms = float(jitter_ms)
        self.packet_ms = float(packet_ms)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        if self.jitter_ms == 0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        frame = max(1, int(self.packet_ms * sr / 1000.0))
        max_delay_frames = int(self.jitter_ms / self.packet_ms) + 1
        n_frames = len(x) // frame
        if n_frames < 3:
            return signal.clone()
        y = np.zeros(len(x) + max_delay_frames * frame)
        for i in range(n_frames):
            delay_frames = int(rng.integers(0, max_delay_frames + 1))
            dest = (i + delay_frames) * frame
            src = i * frame
            end = min(dest + frame, len(y))
            y[dest: end] = x[src: src + (end - dest)]
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)]
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={
                **signal.metadata,
                f"{self.name}.jitter_ms": self.jitter_ms,
                f"{self.name}.packet_ms": self.packet_ms,
            },
        )


class ResamplingChain(BaseTransform):
    """Deployment resampling chain: up to a device rate, then back.

    Models the 8 kHz → 48 kHz → 16 kHz sample-rate conversions a voice
    pipeline applies between capture and processing (polyphase, with
    intermediate band-limiting at each stage).
    """

    def __init__(
        self,
        intermediate_sr: int = 48000,
        name: str = "channel.resampling_chain",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if intermediate_sr <= 0:
            raise ValueError("intermediate_sr must be > 0")
        self.intermediate_sr = int(intermediate_sr)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        if self.intermediate_sr == sr:
            return signal.clone()
        y = scipy.signal.resample_poly(x, up=self.intermediate_sr, down=sr)
        y = scipy.signal.resample_poly(y, up=sr, down=self.intermediate_sr)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)]
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.intermediate_sr": self.intermediate_sr},
        )


class RecordingReplaySimulation(BaseTransform):
    """Speaker → room → microphone recording-and-replay chain.

    Full deployment degradation: loudspeaker response tilt + room
    reverberation (convolution with a synthetic decaying RIR) + additive
    sensor noise at the requested SNR + gentle AGC-like level normalization.
    This models the *real-world* CAPTCHA relay attack path (play a
    recorded challenge through a second device).
    """

    def __init__(
        self,
        rt60_s: float = 0.4,
        snr_db: float = 25.0,
        name: str = "channel.recording_replay",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if rt60_s <= 0:
            raise ValueError("rt60_s must be > 0")
        self.rt60_s = float(rt60_s)
        self.snr_db = float(snr_db)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        from audiocaptcha_dsp.transforms.noise.interference import synthetic_rir

        rng = np.random.default_rng(self.seed)
        x = _as_mono_1d(signal)
        sr = signal.sample_rate

        # 1) Loudspeaker response: gentle high-pass + presence dip
        nyq = sr / 2.0
        b_hp, a_hp = scipy.signal.butter(2, 200.0 / nyq, btype="high")
        y = scipy.signal.lfilter(b_hp, a_hp, x)

        # 2) Room convolution
        ir = synthetic_rir(sr, rt60_s=self.rt60_s, seed=self.seed)
        wet = scipy.signal.fftconvolve(y, ir)[: len(y)]
        peak = np.max(np.abs(wet)) + 1e-12
        wet = wet / peak * (np.max(np.abs(y)) + 1e-12)
        y = 0.65 * y + 0.35 * wet

        # 3) Sensor noise
        noise = rng.normal(0, 1, len(y))
        sos = scipy.signal.butter(2, [100.0 / nyq, min(0.98, 8000.0 / nyq)],
                                  btype="band", output="sos")
        noise = scipy.signal.sosfiltfilt(sos, noise)
        p_sig = np.mean(y ** 2) + 1e-18
        p_noi = np.mean(noise ** 2) + 1e-18
        noise *= np.sqrt(p_sig / (p_noi * 10.0 ** (self.snr_db / 10.0)))
        y = y + noise

        # 4) Gentle level normalization (AGC-like, bounded)
        target_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        gain = np.clip(target_rms / (np.sqrt(np.mean(y ** 2)) + 1e-12), 0.5, 2.0)
        y = y * gain
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={
                **signal.metadata,
                f"{self.name}.rt60_s": self.rt60_s,
                f"{self.name}.snr_db": self.snr_db,
            },
        )


class EchoCancellationArtifacts(BaseTransform):
    """Imperfect echo-cancellation residual (misaligned subtraction).

    An AEC that estimates the echo path with a wrong delay ``τ'`` subtracts
    a misaligned copy of the signal: ``y = x − g·x(t−τ′)``. The residual
    comb-filtered difference is the artifact heard after an imperfect
    canceller (double-talk / mismatch case; ITU-T G.168-style scenario).
    """

    def __init__(
        self,
        misalign_ms: float = 6.0,
        subtraction_gain: float = 0.6,
        name: str = "channel.aec_artifacts",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if misalign_ms < 0 or not (0.0 <= subtraction_gain <= 1.0):
            raise ValueError("misalign_ms >= 0 and subtraction_gain in [0, 1] required")
        self.misalign_ms = float(misalign_ms)
        self.subtraction_gain = float(subtraction_gain)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        delay = int(self.misalign_ms * sr / 1000.0)
        if delay <= 0:
            return signal.clone()
        delayed = np.zeros_like(x)
        delayed[delay:] = x[: len(x) - delay]
        y = x - self.subtraction_gain * delayed
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={
                **signal.metadata,
                f"{self.name}.misalign_ms": self.misalign_ms,
                f"{self.name}.subtraction_gain": self.subtraction_gain,
            },
        )
