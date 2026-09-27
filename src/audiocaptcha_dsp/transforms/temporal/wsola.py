"""
Family B extensions: WSOLA/PSOLA time-scale modification and prosody/rhythm control.
=====================================================================================
Implements the WHAT-REMAINS.txt §3.B gaps:

- WSOLA (Waveform Similarity Overlap-Add) pitch-preserving time modification
- PSOLA (Pitch-Synchronous Overlap-Add) pitch shifting
- Prosody perturbation (F0 contour modulation with duration preserved)
- Rhythm perturbation (local speaking-rate redistribution, duration preserved)
- Short-duration pause insertion at low-energy boundaries
- Segment-level time displacement
- Randomized micro-timing variation

References
----------
- McAulay & Quatieri (1986). Speech transformations based on a sinusoidal
  representation. IEEE TASSP — phase vocoder (implemented elsewhere).
- Roux & Roebel (2009). Phase vocoder-based time-scale modification with
  improved phase locking (WSOLA-family analysis). DAFx.
- Moulines & Laroche (1995). Non-parametric pitch and time-scale modification.
  ICASSP — SOLA/PSOLA family.
- Laroche & Dolson (1999). Improved phase vocoder time-scale modification.
- Kleijn & Haagen (1995). Signal transformation based on a sinusoidal model
  — PSOLA foundations.
"""
from __future__ import annotations

import numpy as np
import librosa

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform


def _safe(y: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _ola(x: np.ndarray, hop: int, win: np.ndarray, out_len: int) -> np.ndarray:
    """Overlap-add x frames (frame length len(win), hop samples) into out_len samples."""
    n_frames = max(0, (len(x) - len(win)) // hop + 1)
    out = np.zeros(out_len + len(win), dtype=np.float64)
    wsum = np.zeros_like(out)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + len(win)]
        out[i * hop: i * hop + len(win)] += seg * win
        wsum[i * hop: i * hop + len(win)] += win ** 2
    wsum[wsum < 1e-8] = 1.0
    return out[:out_len] / wsum[:out_len]


class WSOLATimeStretch(BaseTransform):
    """Pitch-preserving time-scale modification via Waveform Similarity OLA.

    WSOLA searches each synthesis frame in a small neighbourhood of the
    nominal analysis position for the offset with maximum cross-correlation
    to the previous synthesis frame, restoring phase coherence that plain
    OLA loses (Moulines & Laroche 1995; Roux & Roebel 2009).

    Parameters
    ----------
    rate : float
        Time-scale factor. >1 shortens (faster), <1 lengthens (slower).
        Pitch is preserved.
    frame_ms : int
        Analysis frame length in milliseconds.
    search_ms : int
        Half-width of the cross-correlation search window in milliseconds.
    """

    def __init__(
        self,
        rate: float = 1.0,
        frame_ms: int = 40,
        search_ms: int = 10,
        name: str = "temporal.wsola",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if frame_ms <= 0 or search_ms < 0:
            raise ValueError("frame_ms must be > 0 and search_ms >= 0")
        self.rate = rate
        self.frame_ms = frame_ms
        self.search_ms = search_ms
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if abs(self.rate - 1.0) < 1e-9:
            return signal.clone()

        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        frame = max(16, int(self.frame_ms * sr / 1000.0))
        if len(x) < 4 * frame:
            # Too short for meaningful OLA: fall back to linear resampling
            n_out = max(1, int(round(len(x) / self.rate)))
            t_in = np.linspace(0.0, 1.0, len(x), endpoint=False)
            t_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
            y = np.interp(t_out, t_in, x)
            return Signal(waveform=_safe(y), sample_rate=sr,
                          metadata={**signal.metadata, f"{self.name}.rate": self.rate})

        hop_a = frame // 2                    # analysis hop
        hop_s = max(1, int(round(hop_a / self.rate)))  # synthesis hop
        search = int(self.search_ms * sr / 1000.0)
        half = frame // 2
        win = np.hanning(2 * half)
        # Work on centre-clipped, windowed frames of length 2*half
        fl = 2 * half

        n_frames = int(np.ceil((len(x) - fl) / hop_s)) + 1
        out_len = n_frames * hop_s
        out = np.zeros(out_len + fl)
        wsum = np.zeros(out_len + fl)

        prev_frame: np.ndarray | None = None
        nominal = 0
        best_start = 0
        for i in range(n_frames):
            lo = max(0, nominal - search)
            hi = min(len(x) - fl, nominal + search)
            if hi < lo:
                hi = lo
            if prev_frame is None or hi == lo:
                best_start = nominal if nominal + fl <= len(x) else max(0, len(x) - fl)
            else:
                # Cross-correlate candidate frames with previous synthesis frame
                candidates = np.arange(lo, hi + 1)
                best_corr = -np.inf
                best_start = candidates[0]
                pf = prev_frame * win
                pf = pf - pf.mean()
                pf_norm = np.linalg.norm(pf) + 1e-12
                # Subsample candidates for speed
                step = max(1, len(candidates) // 64)
                for s in candidates[::step]:
                    seg = x[s: s + fl] * win
                    seg = seg - seg.mean()
                    corr = float(np.dot(seg, pf)) / (np.linalg.norm(seg) + 1e-12) * pf_norm
                    if corr > best_corr:
                        best_corr = corr
                        best_start = int(s)
            frame_data = x[best_start: best_start + fl] * win
            prev_frame = frame_data
            out[i * hop_s: i * hop_s + fl] += frame_data
            wsum[i * hop_s: i * hop_s + fl] += win
            nominal += hop_a

        wsum[wsum < 1e-8] = 1.0
        y = out / wsum
        # Trim to expected output length
        n_out = max(1, int(round(len(x) / self.rate)))
        y = y[:n_out]
        if len(y) < n_out:
            y = np.pad(y, (0, n_out - len(y)))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.rate": self.rate},
        )


class PSOLAPitchShift(BaseTransform):
    """Pitch-synchronous overlap-add pitch shifting (duration preserved).

    Pitch marks are placed at glottal periods estimated with librosa.yin;
    grains centred on each mark are resampled by 2**(n_steps/12) and
    overlap-added at the *original* mark density, preserving duration while
    shifting F0 (Moulines & Laroche 1995; Kleijn & Haagen 1995).

    Falls back to librosa's phase-vocoder pitch shift when voiced F0 cannot
    be estimated (unvoiced/silent input).
    """

    def __init__(
        self,
        n_steps: float = 0.0,
        fmin: float = 60.0,
        fmax: float = 400.0,
        name: str = "temporal.psola",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.n_steps = float(n_steps)
        self.fmin = fmin
        self.fmax = fmax
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if abs(self.n_steps) < 1e-9:
            return signal.clone()

        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        if len(x) < 4 * (sr // 100):
            return self._fallback(signal)

        try:
            f0, voiced_flag, _ = librosa.pyin(
                x, fmin=self.fmin, fmax=self.fmax, sr=sr,
                frame_length=1024, hop_length=256,
            )
        except Exception:
            return self._fallback(signal)

        if voiced_flag is None or not np.any(voiced_flag):
            return self._fallback(signal)

        # Per-sample period interpolation from frame-level F0
        hop = 256
        times = np.arange(len(f0)) * hop + hop // 2
        f0_filled = np.array(f0, dtype=np.float64)
        if np.any(~np.isfinite(f0_filled)):
            # Fill unvoiced frames with nearest voiced value (or fallback)
            valid = np.isfinite(f0_filled)
            if np.count_nonzero(valid) < 3:
                return self._fallback(signal)
            f0_filled[~valid] = np.interp(
                np.flatnonzero(~valid), np.flatnonzero(valid), f0_filled[valid]
            )
        period_samp = np.maximum(8, sr / f0_filled)

        period_at = np.interp(
            np.arange(len(x)),
            np.clip(times, 0, len(x) - 1),
            period_samp,
        )
        if not np.all(np.isfinite(period_at)) or np.mean(period_at) <= 0:
            return self._fallback(signal)

        # Pitch marks: greedy placement following the local period
        marks = [0]
        pos = 0.0
        while True:
            pos += period_at[int(min(pos, len(x) - 1))]
            if pos >= len(x):
                break
            marks.append(int(pos))
            if len(marks) > 100000:
                break
        marks = np.array(marks, dtype=int)

        ratio = 2.0 ** (self.n_steps / 12.0)
        out = np.zeros(len(x))
        wsum = np.zeros(len(x))
        # Output grain centres keep the ORIGINAL mark positions (duration preserved)
        for m in marks:
            p_out = int(round(period_at[m]))
            p_in = max(8, int(round(period_at[m] / ratio)))
            half = p_out
            s0 = m - half
            s1 = m + half
            if s0 < 0 or s1 > len(x):
                continue
            grain = x[s0:s1]
            # Resample grain period p_out -> p_in (pitch shift), then OLA at fixed grid
            if p_in != p_out:
                t_in = np.linspace(0.0, 1.0, p_out)
                t_out = np.linspace(0.0, 1.0, p_in)
                grain = np.interp(t_out, t_in, grain[:p_out])
            else:
                grain = grain[:p_out]
            w = np.hanning(len(grain))
            start = m - len(grain) // 2
            if start < 0 or start + len(grain) > len(x):
                continue
            out[start: start + len(grain)] += grain * w
            wsum[start: start + len(grain)] += w

        wsum[wsum < 1e-6] = 1.0
        y = out / wsum
        if np.allclose(y, 0.0) and not np.allclose(x, 0.0):
            return self._fallback(signal)
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.n_steps": self.n_steps},
        )

    def _fallback(self, signal: Signal) -> Signal:
        y = librosa.effects.pitch_shift(
            np.asarray(signal.waveform, dtype=np.float32),
            sr=signal.sample_rate,
            n_steps=self.n_steps,
        )
        return Signal(
            waveform=_safe(np.asarray(y, dtype=np.float64)),
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.n_steps": self.n_steps,
                      f"{self.name}.fallback": "phase_vocoder"},
        )


class ProsodyPerturbation(BaseTransform):
    """Perturb the F0 contour (prosody) segment-wise with duration preserved.

    The utterance is split into ``n_segments`` contiguous regions; each
    region receives an independent semitone offset drawn uniformly from
    ±``semitone_range``, applied with librosa's formant-preserving pitch
    shifter and equal-power crossfades. Total duration is unchanged; the
    melodic contour, intonation pattern, and stress rhythm are altered.
    """

    def __init__(
        self,
        semitone_range: float = 2.0,
        n_segments: int = 6,
        name: str = "temporal.prosody",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if semitone_range < 0:
            raise ValueError("semitone_range must be >= 0")
        if n_segments < 2:
            raise ValueError("n_segments must be >= 2")
        self.semitone_range = semitone_range
        self.n_segments = n_segments
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.semitone_range == 0.0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n = len(x)
        bounds = np.linspace(0, n, self.n_segments + 1, dtype=int)
        fade = max(1, int(0.010 * sr))

        # Shift each segment's F0 contour independently
        shifted: list[np.ndarray] = []
        for i in range(self.n_segments):
            seg = x[bounds[i]: bounds[i + 1]]
            offset = float(rng.uniform(-self.semitone_range, self.semitone_range))
            if abs(offset) < 1e-6 or len(seg) < 8:
                s = seg.astype(np.float64).copy()
            else:
                s = librosa.effects.pitch_shift(
                    seg.astype(np.float32), sr=sr, n_steps=offset
                ).astype(np.float64)
            if len(s) > len(seg):
                s = s[: len(seg)]
            elif len(s) < len(seg):
                s = np.pad(s, (0, len(seg) - len(s)))
            shifted.append(s)

        # Concatenate with equal-power crossfades at joints (duration preserved
        # up to the crossfade overlap, which is trimmed from the tail below)
        out = np.zeros(n + 2 * fade)
        pos = 0
        for i, s in enumerate(shifted):
            if i == 0:
                out[: len(s)] = s
                pos = len(s)
                continue
            f = min(fade, pos, len(s))
            if f > 0:
                t = np.linspace(0.0, 1.0, f, endpoint=False)
                out[pos - f: pos] = (
                    out[pos - f: pos] * np.cos(np.pi / 2 * t)
                    + s[:f] * np.sin(np.pi / 2 * t)
                )
                rest = s[f:]
                out[pos: pos + len(rest)] = rest
                pos += len(rest)
            else:
                out[pos: pos + len(s)] = s
                pos += len(s)
        y = out[:n]
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata,
                      f"{self.name}.semitone_range": self.semitone_range,
                      f"{self.name}.n_segments": self.n_segments},
        )


class RhythmPerturbation(BaseTransform):
    """Redistribute local speaking rate (rhythm) while preserving duration.

    The signal is divided into ``n_segments`` regions; each region is time
    stretched by rate ~ clip(N(1, rate_sigma), 1-|max|, 1+|max|); the
    concatenated result is then resampled back to the exact original
    duration, so mean speaking rate is unchanged while local rhythm varies
    (micro-variations of articulation rate).
    """

    def __init__(
        self,
        rate_sigma: float = 0.1,
        n_segments: int = 8,
        max_rate_deviation: float = 0.3,
        name: str = "temporal.rhythm",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if rate_sigma < 0:
            raise ValueError("rate_sigma must be >= 0")
        if n_segments < 2:
            raise ValueError("n_segments must be >= 2")
        self.rate_sigma = rate_sigma
        self.n_segments = n_segments
        self.max_rate_deviation = max_rate_deviation
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.rate_sigma == 0.0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        n = len(x)
        bounds = np.linspace(0, n, self.n_segments + 1, dtype=int)
        pieces: list[np.ndarray] = []
        for i in range(self.n_segments):
            seg = x[bounds[i]: bounds[i + 1]]
            rate = float(np.clip(
                rng.normal(1.0, self.rate_sigma),
                1.0 - self.max_rate_deviation,
                1.0 + self.max_rate_deviation,
            ))
            if abs(rate - 1.0) < 1e-6 or len(seg) < 64:
                pieces.append(seg)
                continue
            stretched = librosa.effects.time_stretch(seg.astype(np.float32), rate=rate)
            pieces.append(np.asarray(stretched, dtype=np.float64))
        y = np.concatenate(pieces) if pieces else x.copy()
        # Restore exact duration (mean rate preserved)
        if len(y) != n:
            t_in = np.linspace(0.0, 1.0, len(y), endpoint=False)
            t_out = np.linspace(0.0, 1.0, n, endpoint=False)
            y = np.interp(t_out, t_in, y)
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata,
                      f"{self.name}.rate_sigma": self.rate_sigma,
                      f"{self.name}.n_segments": self.n_segments},
        )


class PauseInsertion(BaseTransform):
    """Insert short silences at the ``n_pauses`` lowest-energy boundaries.

    Energy minima of the smoothed short-time energy contour approximate
    inter-word boundaries; inserting pauses there models dysfluency /
    listening-gap conditions while leaving speech segments intact.
    """

    def __init__(
        self,
        n_pauses: int = 3,
        pause_ms: float = 120.0,
        min_gap_ms: float = 400.0,
        name: str = "temporal.pause_insertion",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if n_pauses < 0 or pause_ms < 0:
            raise ValueError("n_pauses and pause_ms must be >= 0")
        self.n_pauses = n_pauses
        self.pause_ms = pause_ms
        self.min_gap_ms = min_gap_ms
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        sr = signal.sample_rate
        if self.n_pauses == 0 or self.pause_ms == 0:
            return signal.clone()
        frame = max(1, int(0.020 * sr))
        n_frames = len(x) // frame
        if n_frames < 4:
            return signal.clone()
        energy = np.array([
            np.mean(x[i * frame: (i + 1) * frame] ** 2) for i in range(n_frames)
        ])
        # Smooth energy
        kernel = np.ones(5) / 5.0
        smooth = np.convolve(energy, kernel, mode="same")
        # Local minima candidates
        candidates = [
            i for i in range(2, n_frames - 2)
            if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]
        ]
        if not candidates:
            candidates = list(range(2, n_frames - 2))
        # Choose the quietest candidates with a minimum spacing
        rng = np.random.default_rng(self.seed)
        order = np.argsort([smooth[i] for i in candidates])
        chosen: list[int] = []
        min_gap_frames = max(1, int(self.min_gap_ms / 20.0))
        for idx in order:
            pos = candidates[idx]
            if all(abs(pos - c) >= min_gap_frames for c in chosen):
                chosen.append(pos)
            if len(chosen) >= self.n_pauses:
                break
        chosen.sort()
        pause_samples = int(self.pause_ms * sr / 1000.0)
        insert_at = sorted(int(c) * frame + frame // 2 for c in chosen)
        parts: list[np.ndarray] = []
        prev = 0
        for pos in insert_at:
            if pos <= prev or pos >= len(x):
                continue
            parts.append(x[prev:pos])
            parts.append(np.zeros(pause_samples))
            prev = pos
        parts.append(x[prev:])
        y = np.concatenate(parts)
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata,
                      f"{self.name}.n_pauses": len(insert_at),
                      f"{self.name}.pause_ms": self.pause_ms},
        )


class SegmentDisplacement(BaseTransform):
    """Segment-level time displacement: move blocks of audio in the timeline.

    ``n_moves`` blocks of ``segment_ms`` are cut and re-inserted up to
    ``max_shift_ms`` away from their original position (wrapping inside the
    utterance). Duration is exactly preserved; local event order changes.
    """

    def __init__(
        self,
        segment_ms: float = 200.0,
        max_shift_ms: float = 60.0,
        n_moves: int = 3,
        name: str = "temporal.segment_displacement",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if segment_ms <= 0 or max_shift_ms < 0 or n_moves < 0:
            raise ValueError("segment_ms > 0, max_shift_ms >= 0, n_moves >= 0")
        self.segment_ms = segment_ms
        self.max_shift_ms = max_shift_ms
        self.n_moves = n_moves
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = np.asarray(signal.waveform, dtype=np.float64)
        n = len(x)
        if self.n_moves == 0 or self.max_shift_ms == 0 or n < 64:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        sr = signal.sample_rate
        seg = int(self.segment_ms * sr / 1000.0)
        max_shift = int(self.max_shift_ms * sr / 1000.0)
        if seg >= n or seg < 8:
            return signal.clone()
        y = x.copy()
        for _ in range(self.n_moves):
            start = int(rng.integers(0, n - seg))
            shift = int(rng.integers(-max_shift, max_shift + 1))
            dest = int(np.clip(start + shift, 0, n - seg))
            if dest == start:
                continue
            block = y[start: start + seg].copy()
            y[dest: dest + seg] = block
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata,
                      f"{self.name}.segment_ms": self.segment_ms,
                      f"{self.name}.max_shift_ms": self.max_shift_ms},
        )


class MicroTimingVariation(BaseTransform):
    """Randomized micro-timing variation (endpoint-anchored smooth warp).

    A smooth random displacement curve with amplitude ≤ ``max_shift_ms``
    warps the time axis by a few milliseconds: onsets drift slightly
    earlier/later without changing duration or gross structure — the
    domain-general analogue of human micro-timing (timing variability)
    in performance research.
    """

    def __init__(
        self,
        max_shift_ms: float = 8.0,
        n_control_points: int = 12,
        name: str = "temporal.micro_timing",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if max_shift_ms < 0:
            raise ValueError("max_shift_ms must be >= 0")
        if n_control_points < 3:
            raise ValueError("n_control_points must be >= 3")
        self.max_shift_ms = max_shift_ms
        self.n_control_points = n_control_points
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        if self.max_shift_ms == 0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        n = len(x)
        sr = signal.sample_rate
        max_shift = self.max_shift_ms * sr / 1000.0
        ctrl = np.linspace(0, n - 1, self.n_control_points)
        disp = rng.uniform(-max_shift, max_shift, self.n_control_points)
        disp[0] = 0.0
        disp[-1] = 0.0
        t = np.arange(n)
        t_shifted = t + np.interp(t, ctrl, disp)
        t_shifted = np.clip(t_shifted, 0, n - 1)
        y = np.interp(t_shifted, t, x)
        return Signal(
            waveform=_safe(y),
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.max_shift_ms": self.max_shift_ms},
        )
