"""
Family E extensions: structured interference and room impulse responses.
=========================================================================
WHAT-REMAINS.txt §3.E gaps implemented here:

- Babble noise (multi-talker)
- Harmonic (buzz) interference
- Transient masking bursts at detected onsets
- Nonstationary / dynamic background noise
- Delay-and-add interference (multi-tap)
- Room impulse-response convolution (image-source-style synthetic RIR)

These are *ordinary noise-robustness conditions* (WHAT-REMAINS.txt §3.E):
they are distinct from psychoacoustic masking experiments of family F.

References
----------
- Cooke (2006). A glimpsing model of speech perception in noise. JASA —
  dynamic/babble masking structure.
- Boll (1979). Suppression of acoustic noise in speech using spectral
  subtraction; used here as the standard nonstationary-noise baselines.
- Allen & Berkley (1979). Image method for efficient simulation of room
  acoustics (image-source RIR).
- Schroeder (1962). Natural sounding artificial reverberation.
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


def _scale_to_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise so x + noise has target SNR (energy-based)."""
    p_sig = float(np.mean(x ** 2))
    p_noi = float(np.mean(noise ** 2))
    if p_noi < 1e-18:
        return noise
    if p_sig < 1e-18:
        return noise
    target_p = p_sig / (10.0 ** (snr_db / 10.0))
    return noise * np.sqrt(target_p / p_noi)


def synthetic_rir(
    sr: int,
    rt60_s: float = 0.5,
    n_taps: int = 16,
    room_dims: tuple[float, float, float] = (6.0, 5.0, 3.0),
    source_pos: tuple[float, float, float] = (2.0, 2.0, 1.5),
    mic_pos: tuple[float, float, float] = (4.0, 3.5, 1.5),
    seed: int | None = None,
) -> np.ndarray:
    """Synthesize a compact room impulse response (direct path + image taps).

    Uses the image-source principle for first-order reflections plus an
    exponentially decaying diffuse tail consistent with ``rt60_s``
    (Allen & Berkley 1979; Schroeder 1962).
    """
    rng = np.random.default_rng(seed)
    c = 343.0
    # Direct path
    d0 = float(np.linalg.norm(np.subtract(source_pos, mic_pos)))
    ir_len = int(rt60_s * sr) + int(0.05 * sr)
    ir = np.zeros(ir_len)
    direct_idx = max(0, int(d0 / c * sr))
    ir[direct_idx] = 1.0

    # Image sources across 6 walls (first order)
    images = []
    sx, sy, sz = source_pos
    mx, my, mz = mic_pos
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                if ix == iy == iz == 0:
                    continue
                px = sx if ix == 0 else (ix * room_dims[0] - sx if ix > 0 else -sx)
                # mirror coordinates across each axis
                qx = sx if ix == 0 else ((-sx) if ix < 0 else (2 * room_dims[0] - sx))
                qy = sy if iy == 0 else ((-sy) if iy < 0 else (2 * room_dims[1] - sy))
                qz = sz if iz == 0 else ((-sz) if iz < 0 else (2 * room_dims[2] - sz))
                d = float(np.linalg.norm((qx - mx, qy - my, qz - mz)))
                images.append((d, abs(ix) + abs(iy) + abs(iz)))
    images.sort(key=lambda t: t[0])
    for d, order in images[: max(0, n_taps)]:
        if d <= 0:
            continue
        idx = int(d / c * sr)
        if idx >= ir_len:
            continue
        reflection_loss = 0.55 ** order
        ir[idx] += reflection_loss * rng.uniform(0.7, 1.0)

    # Diffuse exponentially decaying tail
    t = np.arange(ir_len) / sr
    decay = 10.0 ** (-3.0 * t / max(rt60_s, 1e-3))
    tail = rng.normal(0, 1, ir_len) * decay
    # Apply band-limit to tail (rooms are not white)
    sos = scipy.signal.butter(2, [200.0 / (sr / 2), min(0.99, 7000.0 / (sr / 2))],
                              btype="band", output="sos")
    tail = scipy.signal.sosfiltfilt(sos, tail)
    ir += 0.35 * tail
    ir = ir / (np.max(np.abs(ir)) + 1e-12)
    return ir


class BabbleNoise(BaseTransform):
    """Multi-talker babble: overlapping modulated speech-shaped sources.

    ``n_talkers`` independent speech-shaped noise sources, each with
    syllabic amplitude modulation (3–7 Hz) and random spectral tilt,
    mixed to the requested SNR. This models the classic "cocktail party"
    masking background (Cooke 2006).
    """

    def __init__(
        self,
        snr_db: float = 10.0,
        n_talkers: int = 10,
        name: str = "noise.babble",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if n_talkers < 1:
            raise ValueError("n_talkers must be >= 1")
        self.snr_db = snr_db
        self.n_talkers = n_talkers
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n = len(x)
        noise = np.zeros(n)
        sos = scipy.signal.butter(2, [200.0 / (sr / 2), min(0.99, 5000.0 / (sr / 2))],
                                  btype="band", output="sos")
        for k in range(self.n_talkers):
            base = scipy.signal.sosfilt(sos, rng.normal(0, 1, n))
            # Speech-shaped tilt: gentle high-pass (per talker random slope)
            wn = float(np.clip(rng.uniform(0.4, 1.6) * 0.99, 0.01, 0.98))
            tilt_b, tilt_a = scipy.signal.butter(1, wn, btype="high")
            src = scipy.signal.lfilter(tilt_b, tilt_a, base)
            # Syllabic AM at 3–7 Hz with random phase and depth
            rate = rng.uniform(3.0, 7.0)
            depth = rng.uniform(0.5, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arange(n) / sr
            am = 1.0 + depth * np.sin(2 * np.pi * rate * t + phase)
            noise += src * am
        noise = _scale_to_snr(x, noise, self.snr_db)
        y = x + noise
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db,
                      f"{self.name}.n_talkers": self.n_talkers},
        )


class HarmonicInterference(BaseTransform):
    """Harmonic (buzz-like) interference: fixed F0 with N harmonics.

    Unlike a pure tone, a rich harmonic comb overlaps speech harmonics and
    masquerades as competing voicing.
    """

    def __init__(
        self,
        f0_hz: float = 120.0,
        n_harmonics: int = 12,
        snr_db: float = 15.0,
        name: str = "noise.harmonic_interference",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if f0_hz <= 0 or n_harmonics < 1:
            raise ValueError("f0_hz > 0 and n_harmonics >= 1 required")
        self.f0_hz = f0_hz
        self.n_harmonics = n_harmonics
        self.snr_db = snr_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n = len(x)
        t = np.arange(n) / sr
        noise = np.zeros(n)
        for h in range(1, self.n_harmonics + 1):
            f = self.f0_hz * h
            if f >= sr / 2.0:
                break
            amp = 1.0 / h  # natural harmonic rolloff
            phase = rng.uniform(0, 2 * np.pi)
            noise += amp * np.sin(2 * np.pi * f * t + phase)
        noise = _scale_to_snr(x, noise, self.snr_db)
        y = x + noise
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.f0_hz": self.f0_hz,
                      f"{self.name}.snr_db": self.snr_db},
        )


class TransientMasking(BaseTransform):
    """Inject masking transients at detected onsets (energy-derivative peaks).

    Short broadband bursts are placed at the ``n_transients`` strongest
    onsets; they rise above the local speech level by ``level_db`` to mask
    adjacent consonant onsets (temporal masking from onsets).
    """

    def __init__(
        self,
        level_db: float = 6.0,
        n_transients: int = 5,
        burst_ms: float = 25.0,
        name: str = "noise.transient_masking",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if n_transients < 0 or burst_ms <= 0:
            raise ValueError("n_transients >= 0 and burst_ms > 0 required")
        self.level_db = level_db
        self.n_transients = n_transients
        self.burst_ms = burst_ms
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        if self.n_transients == 0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        frame = max(1, int(0.010 * sr))
        n_frames = max(1, len(x) // frame)
        energy = np.array([
            np.mean(x[i * frame: (i + 1) * frame] ** 2) for i in range(n_frames)
        ])
        onset = np.diff(energy, prepend=energy[0])
        order = np.argsort(onset)[::-1][: self.n_transients]
        burst_len = max(1, int(self.burst_ms * sr / 1000.0))
        y = x.copy()
        local_amp = np.sqrt(np.mean(x ** 2)) * 10.0 ** (self.level_db / 20.0)
        for f in order:
            start = int(np.clip(f * frame, 0, len(x) - burst_len))
            if burst_len <= 0 or start + burst_len > len(y):
                continue
            burst = rng.normal(0, 1, burst_len)
            env = np.exp(-np.linspace(0, 4, burst_len))
            y[start: start + burst_len] += local_amp * burst * env
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.level_db": self.level_db,
                      f"{self.name}.n_transients": self.n_transients},
        )


class NonstationaryNoise(BaseTransform):
    """Dynamic (nonstationary) background noise.

    Time-varying spectrum: the noise switches between colored segments
    (white/pink/brown) every ``segment_s`` seconds with crossfades, plus a
    slow amplitude envelope — mimicking changing real-world backgrounds.
    """

    def __init__(
        self,
        snr_db: float = 10.0,
        segment_s: float = 0.5,
        name: str = "noise.nonstationary",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if segment_s <= 0:
            raise ValueError("segment_s must be > 0")
        self.snr_db = snr_db
        self.segment_s = segment_s
        self.seed = seed

    @staticmethod
    def _colored(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
        w = rng.normal(0, 1, n)
        if kind == "white":
            return w
        if kind == "pink":
            # Voss-like approximation via 1/f filtering in frequency domain
            X = np.fft.rfft(w)
            f = np.fft.rfftfreq(n)
            f[0] = f[1] if len(f) > 1 else 1.0
            return np.fft.irfft(X / np.sqrt(f), n=n)
        # brown
        c = np.cumsum(w)
        c -= c.mean()
        return c / (np.std(c) + 1e-12)

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n = len(x)
        seg = max(1, int(self.segment_s * sr))
        kinds = ["white", "pink", "brown"]
        noise = np.zeros(n)
        pos = 0
        fade = max(1, seg // 8)
        idx = 0
        while pos < n:
            end = min(n, pos + seg)
            kind = kinds[idx % len(kinds)]
            chunk = self._colored(end - pos, kind, rng)
            noise[pos: end] = chunk
            # crossfade boundary: fade previous chunk out, new chunk in
            if pos > 0 and fade > 0:
                f = min(fade, pos, end - pos)
                if f > 0:
                    ramp = np.linspace(0, 1, f)
                    noise[pos - f: pos] *= (1 - ramp)
                    noise[pos: pos + f] *= ramp
            pos = end
            idx += 1
        # Slow amplitude envelope
        t = np.arange(n) / sr
        env = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, np.pi)))
        noise *= env
        noise = _scale_to_snr(x, noise, self.snr_db)
        y = x + noise
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db,
                      f"{self.name}.segment_s": self.segment_s},
        )


class DelayAndAddInterference(BaseTransform):
    """Delay-and-add (multi-tap) interference.

    Adds ``n_taps`` copies of the signal at random delays with geometric
    decaying gains — comb-filtering interference distinct from a single
    discrete echo.
    """

    def __init__(
        self,
        n_taps: int = 3,
        max_delay_ms: float = 40.0,
        tap_gain: float = 0.4,
        name: str = "noise.delay_add",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if n_taps < 1 or max_delay_ms < 0:
            raise ValueError("n_taps >= 1 and max_delay_ms >= 0 required")
        self.n_taps = n_taps
        self.max_delay_ms = max_delay_ms
        self.tap_gain = tap_gain
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n = len(x)
        y = x.copy()
        for k in range(self.n_taps):
            delay = int(rng.uniform(1, max(2, self.max_delay_ms * sr / 1000.0)))
            gain = self.tap_gain * (0.6 ** k)
            delayed = np.zeros(n)
            if delay < n:
                delayed[delay:] = x[: n - delay]
            y += gain * delayed
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.n_taps": self.n_taps,
                      f"{self.name}.max_delay_ms": self.max_delay_ms},
        )


class RoomImpulseResponse(BaseTransform):
    """Convolution with a synthetic room impulse response (direct + images).

    Physical room acoustics condition: direct path, first-order image
    reflections, and an exponentially decaying diffuse tail set by
    ``rt60_s`` (Allen & Berkley 1979).
    """

    def __init__(
        self,
        rt60_s: float = 0.6,
        wet_db: float = -6.0,
        n_taps: int = 16,
        name: str = "noise.rir",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if rt60_s <= 0:
            raise ValueError("rt60_s must be > 0")
        self.rt60_s = rt60_s
        self.wet_db = wet_db
        self.n_taps = n_taps
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        ir = synthetic_rir(sr, rt60_s=self.rt60_s, n_taps=self.n_taps, seed=self.seed)
        wet = scipy.signal.fftconvolve(x, ir)[: len(x)]
        wet = wet / (np.sqrt(np.mean(wet ** 2)) + 1e-12) * (np.sqrt(np.mean(x ** 2)) + 1e-12)
        wet_gain = 10.0 ** (self.wet_db / 20.0)
        # Direct path stays at unity; wet component added at wet_db relative level
        y = x + wet_gain * wet
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.rt60_s": self.rt60_s,
                      f"{self.name}.wet_db": self.wet_db},
        )
