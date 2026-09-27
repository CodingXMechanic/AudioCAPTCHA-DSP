"""
Family C: resampling and multirate transformations.
====================================================
WHAT-REMAINS.txt §3.C gaps implemented here:

- integer downsampling/upsampling (polyphase, anti-aliased)
- rational resampling (p/q)
- non-integer resampling
- time-varying resampling
- sample-rate drift (clock mismatch, ppm)
- bandwidth limitation with tunable rolloff
- anti-aliasing filter variation (degraded decimation filters)
- controlled (intentional) aliasing
- narrowband telephone simulation (300–3400 Hz @ 8 kHz)
- wideband→narrowband conversion round-trip
- codec-inspired multirate artifacts (block/decimation artifacts)

All transforms preserve ``sample_rate`` metadata semantics explicitly:
transforms that change the *represented* rate without changing playback
rate resample back to the input rate (declared via metadata), matching the
platform rule "preserve or explicitly declare sample rate/duration
changes" (WHAT-REMAINS.txt §13).

References
----------
- Crochiere & Rabiner (1983). Multirate Digital Signal Processing.
- Smith (1997). Digital Audio Resampling (CCRMA).
- 3GPP TS 26.071 (AMR narrowband) — telephone-band conditions.
- ITU-T G.711/G.726 — legacy telephony quantization.
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


def _resample_back(y: np.ndarray, orig_sr: int, cur_sr: int, n_out: int) -> np.ndarray:
    """Resample y from cur_sr back to orig_sr and trim/pad to n_out samples."""
    if cur_sr != orig_sr:
        y = scipy.signal.resample_poly(y, up=orig_sr, down=cur_sr)
    if len(y) < n_out:
        y = np.pad(y, (0, n_out - len(y)))
    else:
        y = y[:n_out]
    return y


class IntegerResampling(BaseTransform):
    """Integer-factor downsampling/upsampling with anti-alias filtering.

    Downsamples by ``down_factor`` (FIR anti-alias, then decimation) and
    upsamples back by the same factor, returning the signal at the original
    nominal sample rate but band-limited to ``sr/(2*down_factor)`` — the
    classic multirate "rate change without rate declaration" artifact.
    """

    def __init__(
        self,
        down_factor: int = 2,
        anti_alias: bool = True,
        name: str = "multirate.integer",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if down_factor < 1:
            raise ValueError("down_factor must be >= 1")
        self.down_factor = int(down_factor)
        self.anti_alias = anti_alias
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.down_factor == 1:
            return signal.clone()
        x = np.asarray(signal.waveform, dtype=np.float64)
        n = len(x)
        if self.anti_alias:
            # Butterworth anti-alias low-pass at the new Nyquist
            nyq = signal.sample_rate / 2.0
            cutoff = (signal.sample_rate / (2 * self.down_factor)) / nyq
            b, a = scipy.signal.butter(4, min(0.99, cutoff), btype="low")
            x_aa = scipy.signal.filtfilt(b, a, x)
        else:
            x_aa = x
        x_down = x_aa[:: self.down_factor]
        y = np.interp(
            np.arange(n) / self.down_factor,
            np.arange(len(x_down)),
            x_down,
        )
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.down_factor": self.down_factor,
                      f"{self.name}.anti_alias": self.anti_alias},
        )


class RationalResampling(BaseTransform):
    """Rational p/q resampling with polyphase filtering, then rate restoration.

    Performs an up/down conversion (up_factor/down_factor) with ``sosfilt``
    anti-imaging/anti-alias filtering, then returns to the original rate —
    exposing polyphase imaging/aliasing residues at the original nominal rate.
    """

    def __init__(
        self,
        up_factor: int = 3,
        down_factor: int = 2,
        name: str = "multirate.rational",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if up_factor < 1 or down_factor < 1:
            raise ValueError("up_factor and down_factor must be >= 1")
        self.up_factor = int(up_factor)
        self.down_factor = int(down_factor)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.up_factor == 1 and self.down_factor == 1:
            return signal.clone()
        x = np.asarray(signal.waveform, dtype=np.float64)
        n = len(x)
        y = scipy.signal.resample_poly(x, up=self.up_factor, down=self.down_factor)
        # Back to original nominal rate (same rational ratio) — preserves length
        y = scipy.signal.resample_poly(y, up=self.down_factor, down=self.up_factor)
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        y = y[:n]
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata,
                      f"{self.name}.up": self.up_factor,
                      f"{self.name}.down": self.down_factor},
        )


class NonIntegerResampling(BaseTransform):
    """Resample to a non-integer-related target rate and back.

    Simulates mixing device rate mismatches (e.g. 16 kHz → 11.025 kHz →
    16 kHz) where the conversion ratio is not an integer.
    """

    def __init__(
        self,
        target_sr: int = 11025,
        name: str = "multirate.non_integer",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if target_sr <= 0:
            raise ValueError("target_sr must be > 0")
        self.target_sr = int(target_sr)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.target_sr == signal.sample_rate:
            return signal.clone()
        x = np.asarray(signal.waveform, dtype=np.float64)
        n = len(x)
        # Forward: convert to target_sr; backward: return to the input rate
        y = scipy.signal.resample_poly(x, up=self.target_sr, down=signal.sample_rate)
        y = scipy.signal.resample_poly(y, up=signal.sample_rate, down=self.target_sr)
        y = _resample_back(y, signal.sample_rate, signal.sample_rate, n)
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.target_sr": self.target_sr},
        )


class TimeVaryingResampling(BaseTransform):
    """Smooth time-varying resampling: playback rate sweeps rate_start→rate_end.

    Emulates clock skew / wow-and-flutter: instantaneous rate varies
    linearly across the utterance; output is returned at the original
    duration, producing a slowly varying pitch and time warp.
    """

    def __init__(
        self,
        rate_start: float = 1.0,
        rate_end: float = 1.1,
        name: str = "multirate.time_varying",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if rate_start <= 0 or rate_end <= 0:
            raise ValueError("rates must be > 0")
        self.rate_start = rate_start
        self.rate_end = rate_end
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        n = len(x)
        u = np.linspace(0.0, 1.0, n, endpoint=False)
        rate = self.rate_start + (self.rate_end - self.rate_start) * u
        # Cumulative source position map
        cum = np.cumsum(rate)
        cum = cum / cum[-1] * (n - 1)
        y = np.interp(cum, np.arange(n), x)
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata,
                      f"{self.name}.rate_start": self.rate_start,
                      f"{self.name}.rate_end": self.rate_end},
        )


class SampleRateDrift(BaseTransform):
    """Sample-rate clock drift in parts-per-million (ppm).

    A real device clock offset of ``drift_ppm`` resamples the signal by
    (1 + ppm/1e6) and restores the frame length, producing a small
    cumulative time/pitch shift plus resampling artifacts.
    """

    def __init__(
        self,
        drift_ppm: float = 500.0,
        name: str = "multirate.drift",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.drift_ppm = float(drift_ppm)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        n = len(x)
        factor = 1.0 + self.drift_ppm / 1e6
        if abs(self.drift_ppm) < 1e-9:
            return signal.clone()
        # resample_poly accepts ints: approximate the factor as 1000 : round(1000/factor)
        y = scipy.signal.resample_poly(x, up=1000, down=int(round(1000 / factor)))
        y = _resample_back(y, signal.sample_rate, signal.sample_rate, n)
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.drift_ppm": self.drift_ppm},
        )


class BandwidthLimitation(BaseTransform):
    """Hard bandwidth limitation with configurable transition rolloff.

    Brick-wall-ish low-pass at ``cutoff_hz`` with a Butterworth filter of
    order ``order``; models band-limited transmission channels.
    """

    def __init__(
        self,
        cutoff_hz: float = 7000.0,
        rolloff_db_per_oct: float = 24.0,
        name: str = "multirate.bandwidth",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if cutoff_hz <= 0 or rolloff_db_per_oct <= 0:
            raise ValueError("cutoff_hz and rolloff_db_per_oct must be > 0")
        self.cutoff_hz = cutoff_hz
        self.rolloff_db_per_oct = rolloff_db_per_oct
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        nyq = signal.sample_rate / 2.0
        if self.cutoff_hz >= nyq:
            return signal.clone()
        order = max(2, int(round(self.rolloff_db_per_oct / 6.0)))
        b, a = scipy.signal.butter(order, self.cutoff_hz / nyq, btype="low")
        y = scipy.signal.filtfilt(b, a, np.asarray(signal.waveform, dtype=np.float64))
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.cutoff_hz": self.cutoff_hz},
        )


class AntiAliasVariation(BaseTransform):
    """Decimate with a deliberately degraded (low-order) anti-alias filter.

    First-order (or ``alias_filter_order``) filtering before decimation
    leaves aliasing residues above the new Nyquist — the difference between
    high-quality and cheap sample-rate converters.
    """

    def __init__(
        self,
        down_factor: int = 2,
        alias_filter_order: int = 1,
        name: str = "multirate.antialias_variation",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if down_factor < 1 or alias_filter_order < 0:
            raise ValueError("down_factor >= 1 and alias_filter_order >= 0 required")
        self.down_factor = int(down_factor)
        self.alias_filter_order = int(alias_filter_order)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.down_factor == 1:
            return signal.clone()
        x = np.asarray(signal.waveform, dtype=np.float64)
        n = len(x)
        nyq = signal.sample_rate / 2.0
        cutoff = (signal.sample_rate / (2 * self.down_factor)) / nyq
        cutoff = float(np.clip(cutoff, 1e-4, 0.99))
        if self.alias_filter_order == 0:
            x_f = x
        else:
            b, a = scipy.signal.butter(self.alias_filter_order, cutoff, btype="low")
            x_f = scipy.signal.lfilter(b, a, x)
        x_down = x_f[:: self.down_factor]
        y = np.interp(np.arange(n) / self.down_factor, np.arange(len(x_down)), x_down)
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata,
                      f"{self.name}.down_factor": self.down_factor,
                      f"{self.name}.order": self.alias_filter_order},
        )


class ControlledAliasing(BaseTransform):
    """Intentional aliasing: decimate without any anti-alias filtering.

    Content above the reduced Nyquist folds back into the audible band.
    Strength scales with ``down_factor``.
    """

    def __init__(
        self,
        down_factor: int = 3,
        name: str = "multirate.aliasing",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if down_factor < 1:
            raise ValueError("down_factor must be >= 1")
        self.down_factor = int(down_factor)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.down_factor == 1:
            return signal.clone()
        x = np.asarray(signal.waveform, dtype=np.float64)
        n = len(x)
        x_down = x[:: self.down_factor]  # no filtering: spectral folding
        # Smoothly reconstruct to original length (folding artifacts remain)
        t_src = np.arange(len(x_down)) * self.down_factor
        y = np.interp(np.arange(n), t_src, x_down)
        return Signal(
            waveform=_safe(y),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.down_factor": self.down_factor},
        )


class NarrowbandTelephone(BaseTransform):
    """Narrowband telephone simulation: 300–3400 Hz band, 8 kHz internal rate.

    Chain: band-pass → decimate to 8 kHz → upsample to input rate. Models
    PSTN/legacy handset channels (ITU-T G.711 reference condition).
    """

    def __init__(
        self,
        low_hz: float = 300.0,
        high_hz: float = 3400.0,
        name: str = "multirate.narrowband",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if not (0 <= low_hz < high_hz):
            raise ValueError("need 0 <= low_hz < high_hz")
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        nyq = sr / 2.0
        hi = min(self.high_hz, nyq - 100.0) if nyq > 500 else nyq * 0.99
        b, a = scipy.signal.butter(4, [self.low_hz / nyq, hi / nyq], btype="band")
        y = scipy.signal.filtfilt(b, a, x)
        # Internal 8 kHz stage when source rate is higher
        if sr > 8000:
            y8 = scipy.signal.resample_poly(y, up=1, down=int(sr // 8000))
            y = scipy.signal.resample_poly(y8, up=int(sr // 8000), down=1)
            if len(y) < len(x):
                y = np.pad(y, (0, len(x) - len(y)))
            y = y[: len(x)]
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.band": f"{self.low_hz}-{self.high_hz}"},
        )


class WidebandToNarrowband(BaseTransform):
    """Wideband → narrowband → wideband conversion with rate change.

    16 kHz wideband speech passes through an 8 kHz narrowband stage
    (band-limit + halved rate), then is upsampled back — the standard
    telephone-network degradation applied to wideband sources.
    """

    def __init__(
        self,
        narrow_sr: int = 8000,
        cutoff_hz: float = 3400.0,
        name: str = "multirate.wideband_to_narrowband",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if narrow_sr <= 0 or cutoff_hz <= 0:
            raise ValueError("narrow_sr and cutoff_hz must be > 0")
        self.narrow_sr = int(narrow_sr)
        self.cutoff_hz = cutoff_hz
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        nyq = sr / 2.0
        cutoff = min(self.cutoff_hz, nyq * 0.99)
        b, a = scipy.signal.butter(4, cutoff / nyq, btype="low")
        y = scipy.signal.filtfilt(b, a, x)
        if sr > self.narrow_sr:
            factor = int(round(sr / self.narrow_sr))
            y_n = scipy.signal.resample_poly(y, up=1, down=factor)
            y = scipy.signal.resample_poly(y_n, up=factor, down=1)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)]
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.narrow_sr": self.narrow_sr},
        )


class CodecMultirateArtifacts(BaseTransform):
    """Codec-inspired multirate artifacts: block-based subband quantization.

    Simulates the block/subband structure of perceptual codecs (MP3's 32
    subbands × 18-sample granules): STFT frames of ``block_size_ms`` are
    coarsely quantized per frequency band, producing blockwise spectral
    floor modulation and pre-echo-like smearing without any external encoder.
    """

    def __init__(
        self,
        block_size_ms: float = 1.5,
        quantization_bits: float = 4.0,
        name: str = "multirate.codec_artifacts",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if block_size_ms <= 0 or quantization_bits <= 0:
            raise ValueError("block_size_ms and quantization_bits must be > 0")
        self.block_size_ms = block_size_ms
        self.quantization_bits = quantization_bits
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n_fft = int(2 * round(self.block_size_ms * sr / 1000.0))
        n_fft = max(64, 1 << int(np.ceil(np.log2(n_fft))))
        hop = n_fft // 2
        win = np.hanning(n_fft)
        if len(x) < n_fft:
            return signal.clone()
        n_frames = 1 + (len(x) - n_fft) // hop
        out = np.zeros(len(x))
        wsum = np.zeros(len(x))
        step = 2.0 ** -self.quantization_bits
        for i in range(n_frames):
            seg = x[i * hop: i * hop + n_fft] * win
            spec = np.fft.rfft(seg)
            mag = np.abs(spec)
            peak = mag.max() if mag.max() > 0 else 1.0
            # Coarser quantization for low-energy bins (perceptual floor)
            q = step * peak
            mag_q = np.round(mag / np.maximum(q, 1e-12)) * np.maximum(q, 1e-12)
            spec_q = mag_q * np.exp(1j * np.angle(spec))
            y_frame = np.fft.irfft(spec_q, n=n_fft)
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
            metadata={**signal.metadata,
                      f"{self.name}.block_ms": self.block_size_ms,
                      f"{self.name}.bits": self.quantization_bits},
        )
