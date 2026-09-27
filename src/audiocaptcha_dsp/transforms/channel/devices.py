"""Family H: playback/capture device transforms and defense simulations.
=========================================================================
WHAT-REMAINS.txt §3.H / §4 (defenses) items implemented here:

- Playback device equalization curve
- Microphone frequency response
- Loudspeaker frequency response
- Automatic gain control (AGC) dynamics
- Single-channel noise suppression (spectral subtraction) — *defense*
- Envelope-based dereverberation — *defense*
- Device-dependent nonlinear distortion
- Room reverberation as a channel condition

Defense models here are explicitly *simulations* of common front-end
processing (what a deployed CAPTCHA service might apply before ASR); they
are used in the defense-survival analysis (§4).

References
----------
- Boll (1979). Suppression of acoustic noise in speech using spectral
  subtraction. IEEE TASSP.
- Ephraim & Malah (1984). Speech enhancement using a minimum mean-square
  error short-time spectral amplitude estimator. IEEE TASSP.
- Allen & Berkley (1979). Image method for efficiently simulating room
  acoustics. JASA.
- ITU-T P.340 / P.51 — transmission characteristics of terminals.
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


def _shaped_response(x: np.ndarray, sr: int, shelf_low_db: float, shelf_high_db: float,
                     dip_db: float = 0.0, dip_hz: float = 2500.0) -> np.ndarray:
    """Apply a smooth bass/treble shelf + optional mid dip via zero-phase EQ."""
    nyq = sr / 2.0
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr) if len(x) > 4 else np.array([0.0])
    gain_db = (
        shelf_low_db * (1.0 / (1.0 + (freqs / 300.0) ** 2))       # bass shelf
        + shelf_high_db * (1.0 / (1.0 + (3000.0 / np.maximum(freqs, 1.0)) ** 2))  # treble shelf
    )
    if dip_db != 0.0:
        gain_db = gain_db + dip_db / (1.0 + ((freqs - dip_hz) / (dip_hz * 0.5)) ** 2)
    gain = 10.0 ** (gain_db / 20.0)
    spec = np.fft.rfft(x)
    return np.fft.irfft(spec * gain[: len(spec)], n=len(x))


class PlaybackEqualization(BaseTransform):
    """Playback device equalization: consumer-speaker coloration curve.

    Bass roll-off + treble loss + mild presence bump, characteristic of
    laptop/phone speakers reproducing a CAPTCHA prompt.
    """

    def __init__(
        self,
        bass_db: float = -6.0,
        treble_db: float = -4.0,
        presence_db: float = 2.0,
        name: str = "channel.playback_eq",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.bass_db = float(bass_db)
        self.treble_db = float(treble_db)
        self.presence_db = float(presence_db)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        y = _shaped_response(x, signal.sample_rate, self.bass_db, self.treble_db,
                             dip_db=self.presence_db, dip_hz=2500.0)
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.bass_db": self.bass_db,
                      f"{self.name}.treble_db": self.treble_db},
        )


class MicrophoneResponse(BaseTransform):
    """Microphone frequency response: band-limited capture with mild resonance.

    Consumer lavalier/headset mics: −12 dB/oct low-end roll-off below
    ~100 Hz, attenuation above ~7 kHz, and a small condenser resonance.
    """

    def __init__(
        self,
        low_hz: float = 100.0,
        high_hz: float = 7000.0,
        resonance_db: float = 3.0,
        name: str = "channel.microphone",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if not (0 <= self._lo(low_hz) < high_hz):
            raise ValueError("need 0 <= low_hz < high_hz")
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.resonance_db = float(resonance_db)
        self.seed = seed

    @staticmethod
    def _lo(v: float) -> float:
        return v

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        nyq = sr / 2.0
        high = min(self.high_hz, nyq * 0.99)
        if nyq > self.low_hz * 2:
            b, a = scipy.signal.butter(2, [self.low_hz / nyq, high / nyq], btype="band")
            y = scipy.signal.filtfilt(b, a, x)
        else:  # pragma: no cover - pathological sample rates
            y = x.copy()
        y = y + _shaped_response(y, sr, 0.0, 0.0, dip_db=self.resonance_db, dip_hz=high * 0.9)
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.band": f"{self.low_hz}-{self.high_hz}"},
        )


class SpeakerResponse(BaseTransform):
    """Loudspeaker frequency response: small-driver roll-off and port resonance."""

    def __init__(
        self,
        low_hz: float = 150.0,
        rolloff_db_per_oct: float = 12.0,
        resonance_db: float = 4.0,
        resonance_hz: float = 180.0,
        name: str = "channel.speaker",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if low_hz <= 0 or rolloff_db_per_oct <= 0:
            raise ValueError("low_hz and rolloff_db_per_oct must be > 0")
        self.low_hz = float(low_hz)
        self.rolloff_db_per_oct = float(rolloff_db_per_oct)
        self.resonance_db = float(resonance_db)
        self.resonance_hz = float(resonance_hz)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        nyq = sr / 2.0
        order = max(2, int(round(self.rolloff_db_per_oct / 6.0)))
        if self.low_hz < nyq:
            b, a = scipy.signal.butter(order, self.low_hz / nyq, btype="high")
            y = scipy.signal.filtfilt(b, a, x)
        else:  # pragma: no cover - pathological sample rates
            y = x.copy()
        y = _shaped_response(y, sr, self.resonance_db, 0.0, dip_db=0.0,
                             dip_hz=self.resonance_hz)
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.low_hz": self.low_hz},
        )


class AutomaticGainControl(BaseTransform):
    """Automatic gain control (AGC) dynamics.

    Envelope follower with ``attack_db_per_s`` / ``release_db_per_s`` (or
    time constants) driving a gain that pulls RMS toward ``target_db``
    within ``max_gain_db`` bounds — the level-riding behavior of phone/
    browser capture chains.
    """

    def __init__(
        self,
        target_db: float = -20.0,
        max_gain_db: float = 18.0,
        attack_s: float = 0.010,
        release_s: float = 0.200,
        name: str = "channel.agc",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.target_db = float(target_db)
        self.max_gain_db = float(max_gain_db)
        self.attack_s = float(attack_s)
        self.release_s = float(release_s)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        frame = max(1, int(0.010 * sr))
        n_frames = max(1, len(x) // frame)
        target_amp = 10.0 ** (self.target_db / 20.0)
        max_gain = 10.0 ** (self.max_gain_db / 20.0)
        min_gain = 10.0 ** (-self.max_gain_db / 20.0)

        attack_c = np.exp(-1.0 / max(1.0, self.attack_s * 1000.0))
        release_c = np.exp(-1.0 / max(1.0, self.release_s * 1000.0))

        gain_curve = np.ones(len(x))
        state = 1.0
        for i in range(n_frames):
            seg = x[i * frame: (i + 1) * frame]
            rms = np.sqrt(np.mean(seg ** 2) + 1e-12)
            desired = float(np.clip(target_amp / rms, min_gain, max_gain))
            coeff = attack_c if desired < state else release_c
            state = coeff * state + (1.0 - coeff) * desired
            gain_curve[i * frame: (i + 1) * frame] = state
        gain_curve[n_frames * frame:] = state
        y = x * gain_curve
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.target_db": self.target_db,
                      f"{self.name}.max_gain_db": self.max_gain_db},
        )


class NoiseSuppression(BaseTransform):
    """Single-channel noise suppression (spectral subtraction) — a *defense*.

    Noise PSD is estimated from the quietest ``noise_percentile``%% of
    frames (min-statistics style); magnitude subtraction with ``over_sub``
    over-suppression and a spectral floor suppresses stationary noise.
    Common pre-ASR front-end processing (Boll 1979; Ephraim & Malah 1984).
    """

    def __init__(
        self,
        over_sub: float = 1.5,
        spectral_floor: float = 0.05,
        noise_percentile: float = 10.0,
        name: str = "channel.noise_suppression",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if over_sub < 1.0:
            raise ValueError("over_sub must be >= 1.0 (subtract at least the estimate)")
        if not (0.0 <= spectral_floor < 1.0):
            raise ValueError("spectral_floor must be in [0, 1)")
        self.over_sub = float(over_sub)
        self.spectral_floor = float(spectral_floor)
        self.noise_percentile = float(noise_percentile)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        n_fft, hop = 1024, 512
        if len(x) < n_fft:
            return signal.clone()
        win = np.hanning(n_fft)
        n_frames = 1 + (len(x) - n_fft) // hop
        specs = np.empty((n_frames, n_fft // 2 + 1), dtype=complex)
        mags = np.empty((n_frames, n_fft // 2 + 1))
        for i in range(n_frames):
            frame = x[i * hop: i * hop + n_fft] * win
            specs[i] = np.fft.rfft(frame)
            mags[i] = np.abs(specs[i])
        # Noise estimate from the quietest frames (per-bin percentile)
        frame_energy = mags.mean(axis=1)
        thresh = np.percentile(frame_energy, self.noise_percentile)
        noise_frames = mags[frame_energy <= thresh]
        if len(noise_frames) == 0:
            noise_frames = mags[:1]
        noise_mag = np.percentile(noise_frames, 50.0, axis=0)

        sub = np.maximum(mags - self.over_sub * noise_mag[None, :],
                         self.spectral_floor * mags)
        phase = np.exp(1j * np.angle(specs))
        enhanced = sub * phase

        out = np.zeros(len(x) + n_fft)
        wsum = np.zeros_like(out)
        for i in range(n_frames):
            y_frame = np.fft.irfft(enhanced[i], n=n_fft)
            out[i * hop: i * hop + n_fft] += y_frame * win
            wsum[i * hop: i * hop + n_fft] += win ** 2
        wsum = np.maximum(wsum, 1e-3 * float(wsum.max()))
        y = out / wsum
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)]
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.over_sub": self.over_sub,
                      f"{self.name}.defense": True},
        )


class Dereverberation(BaseTransform):
    """Envelope-based dereverberation — a *defense* (simulation).

    Late reverberation is approximated by the difference between the
    signal and its excitation envelope (inverse filtering in the STFT):
    spectral gain decays with the estimated reverberant tail energy,
    suppressing late reverb while keeping early components.
    """

    def __init__(
        self,
        suppression_db: float = -8.0,
        tail_ms: float = 60.0,
        name: str = "channel.dereverberation",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if tail_ms <= 0:
            raise ValueError("tail_ms must be > 0")
        self.suppression_db = float(suppression_db)
        self.tail_ms = float(tail_ms)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        n_fft, hop = 2048, 1024
        if len(x) < n_fft:
            return signal.clone()
        win = np.hanning(n_fft)
        n_frames = 1 + (len(x) - n_fft) // hop
        specs = np.empty((n_frames, n_fft // 2 + 1), dtype=complex)
        for i in range(n_frames):
            specs[i] = np.fft.rfft(x[i * hop: i * hop + n_fft] * win)
        mag = np.abs(specs)
        # Smoothed spectral magnitude = excitation estimate; the residual
        # between frames and its local decay is treated as late-reverb energy.
        kernel = max(2, int(self.tail_ms * sr / 1000.0 / hop))
        decay = np.empty_like(mag)
        for i in range(n_frames):
            lo = max(0, i - kernel)
            decay[i] = mag[lo: i + 1].mean(axis=0) if i > lo else mag[i]
        residual = np.clip(1.0 - decay / (mag + 1e-12), 0.0, 1.0)
        gain_db = self.suppression_db * residual
        gain = 10.0 ** (gain_db / 20.0)
        enhanced = mag * gain * np.exp(1j * np.angle(specs))

        out = np.zeros(len(x) + n_fft)
        wsum = np.zeros_like(out)
        for i in range(n_frames):
            y_frame = np.fft.irfft(enhanced[i], n=n_fft)
            out[i * hop: i * hop + n_fft] += y_frame * win
            wsum[i * hop: i * hop + n_fft] += win ** 2
        wsum = np.maximum(wsum, 1e-3 * float(wsum.max()))
        y = out / wsum
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)]
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.suppression_db": self.suppression_db,
                      f"{self.name}.defense": True},
        )


class DeviceDistortion(BaseTransform):
    """Device-dependent nonlinear distortion (memoryless + quantization).

    Combines a soft-clip nonlinearity (``drive``), weak even-harmonic
    asymmetry (``asymmetry``), and coarse output quantization — the
    saturating, noisy output of cheap DACs / overdriven amplifiers.
    """

    def __init__(
        self,
        drive: float = 1.5,
        asymmetry: float = 0.1,
        bits: float = 8.0,
        name: str = "channel.device_distortion",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if drive <= 0:
            raise ValueError("drive must be > 0")
        if not (0 <= asymmetry < 1):
            raise ValueError("asymmetry must be in [0, 1)")
        if bits <= 0:
            raise ValueError("bits must be > 0")
        self.drive = float(drive)
        self.asymmetry = float(asymmetry)
        self.bits = float(bits)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        peak = np.max(np.abs(x)) + 1e-12
        x_n = x / peak
        driven = np.tanh(self.drive * x_n)
        # Even-harmonic asymmetry (DC-blocked)
        driven = driven + self.asymmetry * driven ** 2
        driven = driven - np.mean(driven)
        # Quantization
        levels = 2.0 ** self.bits
        y = np.round(driven * (levels / 2 - 1)) / (levels / 2 - 1)
        y = y * peak
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.drive": self.drive,
                      f"{self.name}.bits": self.bits},
        )


class RoomReverberationChannel(BaseTransform):
    """Room reverberation as a deployment channel condition (synthetic RIR).

    Convolution with an image-source + diffuse-tail synthetic RIR at
    ``rt60_s``, mixed at ``wet_db`` relative to the direct path — the
    room in which a CAPTCHA is played and re-recorded.
    """

    def __init__(
        self,
        rt60_s: float = 0.6,
        wet_db: float = -6.0,
        name: str = "channel.room_reverb",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if rt60_s <= 0:
            raise ValueError("rt60_s must be > 0")
        self.rt60_s = float(rt60_s)
        self.wet_db = float(wet_db)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        from audiocaptcha_dsp.transforms.noise.interference import synthetic_rir

        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        ir = synthetic_rir(sr, rt60_s=self.rt60_s, seed=self.seed)
        wet = scipy.signal.fftconvolve(x, ir)[: len(x)]
        wet = wet / (np.sqrt(np.mean(wet ** 2)) + 1e-12) * (np.sqrt(np.mean(x ** 2)) + 1e-12)
        y = x + 10.0 ** (self.wet_db / 20.0) * wet
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.rt60_s": self.rt60_s,
                      f"{self.name}.wet_db": self.wet_db},
        )
