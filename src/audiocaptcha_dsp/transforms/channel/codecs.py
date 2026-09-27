"""Family H: real codec transforms (ffmpeg round-trips) with honest fallbacks.
================================================================================
WHAT-REMAINS.txt §3.H gaps implemented here:

- MP3 compression (real ffmpeg encode/decode round-trip)
- Opus compression (real ffmpeg round-trip)
- AAC-like compression (real ffmpeg round-trip)
- Telephone codec simulation (μ-law + 8 kHz narrowband)
- Masking-threshold perceptual codec simulation (no external encoder)

Design
------
* :func:`ffmpeg_available` probes ``ffmpeg`` on PATH (cached). When ffmpeg
  exists, the codec classes perform a genuine lossy round-trip through the
  actual encoder — bitrates are honored exactly as the encoder's VBR/CBR
  settings allow.
* When ffmpeg is unavailable (or the encode fails), the classes fall back
  to :class:`PerceptualCodecSimulation` and set metadata
  ``"<name>.fallback" = "perceptual_simulation"`` so no run ever silently
  claims a real codec result it did not produce (integrity rule §13:
  record which experiment path actually ran).

References
----------
- MPEG-1 Audio Layer III (ISO/IEC 11172-3, 1993) — MP3.
- Valin et al. (2012). Guide to Opus, RFC 6716 (IETF, 2012).
- ISO/IEC 14496-3 — AAC.
- ITU-T G.711 (1988) — μ-law companding.
- Noll (1976). Digital coding of audio (MDCT/subband perceptual coding).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform

logger = logging.getLogger(__name__)


def _safe(y: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _as_mono_1d(signal: Signal) -> np.ndarray:
    x = np.asarray(signal.waveform, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=0)
    return x


@lru_cache(maxsize=1)
def ffmpeg_available() -> bool:
    """Return True when an ``ffmpeg`` executable is on PATH (cached)."""
    return shutil.which("ffmpeg") is not None


def _run_ffmpeg(args: list[str], timeout: float = 30.0) -> bool:
    """Run ffmpeg quietly; return True on success (rc == 0)."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return False
    try:
        proc = subprocess.run(
            [exe, "-y", "-hide_banner", "-loglevel", "error", *args],
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode == 0
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("ffmpeg invocation failed: %s", exc)
        return False


def _load_audio(path: str, target_sr: int) -> np.ndarray | None:
    """Load audio via soundfile (or ffmpeg → wav fallback); mono float64."""
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float64", always_2d=True)
        y = data.mean(axis=1)
        if sr != target_sr:
            import scipy.signal

            y = scipy.signal.resample_poly(y, up=target_sr, down=sr)
        return y
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("soundfile load failed for %s: %s", path, exc)
        return None


class PerceptualCodecSimulation(BaseTransform):
    """MDCT-less STFT quantization with a masking-flavored bit allocation.

    A codec *simulation* (no external encoder): the signal is analyzed in
    20 ms frames / 10 ms hops; per-frame MDCT-style (rfft) coefficients
    are quantized with a step size derived from the target bitrate and the
    frame's spectral peak; low-energy bins receive proportionally coarser
    quantization (an auditory-floor heuristic). Frames are overlap-added.

    This is explicitly labeled a *simulation* — never a real MP3/AAC/Opus
    bitstream — and is used as the fallback path when ffmpeg is absent.

    Parameters
    ----------
    bitrate_kbps : float
        Target bitrate driving the quantizer step (higher = finer).
    """

    def __init__(
        self,
        bitrate_kbps: float = 32.0,
        name: str = "channel.codec_simulation",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if bitrate_kbps <= 0:
            raise ValueError("bitrate_kbps must be > 0")
        self.bitrate_kbps = float(bitrate_kbps)
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        n_fft = 1024 if sr >= 16000 else 512
        hop = n_fft // 2
        if len(x) < n_fft:
            return signal.clone()
        win = np.hanning(n_fft)
        n_frames = 1 + (len(x) - n_fft) // hop
        out = np.zeros(len(x) + n_fft)
        wsum = np.zeros_like(out)

        # bits per coefficient ∝ bitrate (frame bits = bitrate * frame_dur)
        frame_bits = self.bitrate_kbps * 1000.0 * (n_fft / sr) / n_fft
        n_bins = n_fft // 2 + 1
        bits_per_bin = max(1.0, frame_bits)
        step_ratio = 2.0 ** (-bits_per_bin)

        for i in range(n_frames):
            seg = x[i * hop: i * hop + n_fft] * win
            spec = np.fft.rfft(seg)
            mag = np.abs(spec)
            peak = mag.max()
            if peak < 1e-12:
                out[i * hop: i * hop + n_fft] += seg
                wsum[i * hop: i * hop + n_fft] += win ** 2
                continue
            # Auditory-floor heuristic: quiet bins get coarser steps
            floor = peak * step_ratio
            step = np.maximum(floor, mag * step_ratio)
            mag_q = np.round(mag / step) * step
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
            metadata={
                **signal.metadata,
                f"{self.name}.bitrate_kbps": self.bitrate_kbps,
                f"{self.name}.encoder": "perceptual_simulation",
            },
        )


class _FFmpegCodecBase(BaseTransform):
    """Shared ffmpeg round-trip logic for MP3/AAC/Opus codecs."""

    #: ffmpeg codec/format arguments; subclasses override.
    codec_args: tuple[str, ...] = ()
    #: file extension of the encoded container
    ext: str = ".bin"

    def __init__(
        self,
        bitrate_kbps: float = 32.0,
        name: str = "channel.codec",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if bitrate_kbps <= 0:
            raise ValueError("bitrate_kbps must be > 0")
        self.bitrate_kbps = float(bitrate_kbps)
        self.seed = seed
        self.fallback = PerceptualCodecSimulation(bitrate_kbps=bitrate_kbps, name=name)

    def _encode_decode(self, x: np.ndarray, sr: int) -> np.ndarray | None:
        """Full round-trip via ffmpeg; None signals failure (caller falls back)."""
        if not ffmpeg_available():
            return None
        with tempfile.TemporaryDirectory(prefix="audiocap_codec_") as tmp:
            in_path = os.path.join(tmp, "in.wav")
            enc_path = os.path.join(tmp, "encoded" + self.ext)
            dec_path = os.path.join(tmp, "decoded.wav")
            try:
                import soundfile as sf

                sf.write(in_path, x, sr, subtype="PCM_16")
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("wav write failed: %s", exc)
                return None
            ok = _run_ffmpeg(
                ["-i", in_path, *self.codec_args,
                 "-b:a", f"{int(round(self.bitrate_kbps))}k", enc_path]
            )
            if not ok:
                return None
            ok = _run_ffmpeg(["-i", enc_path, "-ar", str(sr), dec_path])
            if not ok:
                return None
            return _load_audio(dec_path, sr)

    def __call__(self, signal: Signal) -> Signal:
        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        decoded = self._encode_decode(x, sr)
        if decoded is None:
            # Honest fallback: label the simulation path in metadata
            out = self.fallback(signal)
            return Signal(
                waveform=out.waveform,
                sample_rate=out.sample_rate,
                metadata={**out.metadata, f"{self.name}.fallback": "perceptual_simulation"},
            )
        y = decoded
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)]
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={
                **signal.metadata,
                f"{self.name}.bitrate_kbps": self.bitrate_kbps,
                f"{self.name}.encoder": "ffmpeg",
            },
        )


class MP3Compression(_FFmpegCodecBase):
    """MP3 (MPEG-1 Audio Layer III) lossy round-trip at the target bitrate."""

    codec_args = ("-c:a", "libmp3lame")
    ext = ".mp3"

    def __init__(self, bitrate_kbps: float = 32.0, name: str = "channel.mp3",
                 seed: int | None = None) -> None:
        super().__init__(bitrate_kbps=bitrate_kbps, name=name, seed=seed)


class OpusCompression(_FFmpegCodecBase):
    """Opus lossy round-trip (VoIP/streaming codec, ~6–510 kbps)."""

    codec_args = ("-c:a", "libopus")
    ext = ".opus"

    def __init__(self, bitrate_kbps: float = 16.0, name: str = "channel.opus",
                 seed: int | None = None) -> None:
        super().__init__(bitrate_kbps=bitrate_kbps, name=name, seed=seed)


class AACCompression(_FFmpegCodecBase):
    """AAC-like (MPEG-4 Part 3) lossy round-trip at the target bitrate."""

    codec_args = ("-c:a", "aac")
    ext = ".m4a"

    def __init__(self, bitrate_kbps: float = 32.0, name: str = "channel.aac",
                 seed: int | None = None) -> None:
        super().__init__(bitrate_kbps=bitrate_kbps, name=name, seed=seed)


class TelephoneCodec(BaseTransform):
    """G.711 μ-law telephone simulation: 8 kHz + μ-law quantization.

    Chain: band-limit (300–3400 Hz) → decimate to 8 kHz → μ-law (8-bit)
    quantization/expand → upsample back to the input rate. Models the
    PSTN path (ITU-T G.711) without any external encoder.
    """

    def __init__(self, name: str = "channel.telephone_codec", seed: int | None = None) -> None:
        super().__init__(name=name)
        self.seed = seed

    @staticmethod
    def _mu_law(x: np.ndarray, mu: int = 255) -> np.ndarray:
        """μ-law encode/decode round-trip (G.711), input in [-1, 1]."""
        x = np.clip(x, -1.0, 1.0)
        magnitude = np.log1p(mu * np.abs(x)) / np.log1p(mu)
        quantized = np.round(magnitude * (mu - 1)) / (mu - 1)
        restored = np.sign(x) * (np.expm1(quantized * np.log1p(mu)) / mu)
        return restored

    def __call__(self, signal: Signal) -> Signal:
        import scipy.signal

        x = _as_mono_1d(signal)
        sr = signal.sample_rate
        nyq = sr / 2.0
        hi = min(3400.0, nyq * 0.99)
        if nyq > 3600.0:
            b, a = scipy.signal.butter(4, [300.0 / nyq, hi / nyq], btype="band")
            x = scipy.signal.filtfilt(b, a, x)
        # Internal 8 kHz stage
        peak = np.max(np.abs(x)) + 1e-12
        x_norm = x / peak
        if sr > 8000:
            factor = int(round(sr / 8000))
            y = scipy.signal.resample_poly(x_norm, up=1, down=factor)
            y = self._mu_law(y)
            y = scipy.signal.resample_poly(y, up=factor, down=1)
        else:
            y = self._mu_law(x_norm)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)] * peak
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.codec": "g711_mu_law",
                      f"{self.name}.internal_rate": 8000},
        )
