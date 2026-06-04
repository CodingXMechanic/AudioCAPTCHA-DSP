from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform


class PerturbedOverlapAdd(BaseTransform):
    def __init__(
        self,
        window_size: int = 512,
        hop_ratio: float = 0.25,
        window: str = "hann",
        jitter_ms: float = 0.0,
        name: str = "perturbed_overlap_add",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if window_size < 2:
            raise ValueError(f"Window size must be >= 2, got {window_size}")
        if not 0.0 < hop_ratio <= 1.0:
            raise ValueError(f"Hop ratio must be in (0, 1], got {hop_ratio}")
        self.window_size = window_size
        self.hop_size = max(1, int(round(window_size * hop_ratio)))
        self.window = window
        self.jitter_ms = jitter_ms
        self.seed = seed

    def _ola(self, waveform: np.ndarray, win: np.ndarray, hop: int) -> np.ndarray:
        n = waveform.shape[0]
        ws = win.shape[0]
        out_len = n + ws
        output = np.zeros(out_len, dtype=np.float64)
        window_sum = np.zeros(out_len, dtype=np.float64)

        num_frames = max(1, int(np.ceil((n - ws) / hop)) + 1)
        for i in range(num_frames):
            start = i * hop
            end = start + ws
            if start >= n:
                break
            actual_end = min(end, n)
            chunk = waveform[start:actual_end]
            if end <= n:
                output[start:end] += chunk * win
                window_sum[start:end] += win * win
            else:
                output[start:n] += chunk * win[: actual_end - start]
                window_sum[start:n] += win[: actual_end - start] ** 2

        nonzero = window_sum > 1e-12
        output[nonzero] /= window_sum[nonzero]

        trim_start = ws // 2
        trim_end = trim_start + n
        if trim_end > out_len:
            output = np.pad(output, (0, trim_end - out_len), mode="constant")
        return output[trim_end - n : trim_end]

    def _ola_jittered(self, waveform: np.ndarray, win: np.ndarray, hop: int, jitter_samples: int, rng: np.random.Generator) -> np.ndarray:
        n = waveform.shape[0]
        ws = win.shape[0]
        out_len = n + 2 * jitter_samples + ws
        output = np.zeros(out_len, dtype=np.float64)
        window_sum = np.zeros(out_len, dtype=np.float64)

        num_frames = max(1, int(np.ceil((n - ws) / hop)) + 1)
        for i in range(num_frames):
            base_start = i * hop
            offset = rng.integers(-jitter_samples, jitter_samples + 1) if jitter_samples > 0 else 0
            start = base_start + offset
            end = start + ws

            src_start = max(0, start)
            src_end = min(n, end)
            dst_start = src_start + (end - src_start) - (src_end - src_start) + start
            dst_start = max(0, start)
            dst_end = dst_start + (src_end - src_start)

            if src_start >= n or dst_start >= out_len:
                continue

            chunk = waveform[src_start:src_end]
            win_start = src_start - start
            win_end = win_start + (src_end - src_start)
            w = win[win_start:win_end]

            if dst_end > out_len:
                dst_end = out_len
                chunk = chunk[: dst_end - dst_start]
                w = w[: dst_end - dst_start]

            output[dst_start:dst_end] += chunk * w
            window_sum[dst_start:dst_end] += w * w

        nonzero = window_sum > 1e-12
        output[nonzero] /= window_sum[nonzero]

        trim_start = ws // 2
        trim_end = trim_start + n
        if trim_end > out_len:
            output = np.pad(output, (0, trim_end - out_len), mode="constant")
        return output[trim_end - n : trim_end]

    def __call__(self, signal: Signal) -> Signal:
        win = get_window(self.window, self.window_size)
        sr = signal.sample_rate
        jitter_samples = int(round(self.jitter_ms / 1000.0 * sr))

        if signal.waveform.ndim == 1:
            mono = signal.waveform
        else:
            mono = signal.waveform.mean(axis=0)

        if self.jitter_ms > 0 and jitter_samples > 0:
            rng = np.random.default_rng(self.seed)
            processed = self._ola_jittered(mono, win, self.hop_size, jitter_samples, rng)
        else:
            processed = self._ola(mono, win, self.hop_size)

        if signal.waveform.ndim == 2:
            gain = np.linalg.norm(signal.waveform) / (np.linalg.norm(processed) + 1e-12)
            processed = (processed * gain).astype(np.float64)
            output_waveform = np.broadcast_to(processed, signal.waveform.shape).copy()
        else:
            output_waveform = processed.astype(np.float64)

        output = Signal(
            waveform=output_waveform,
            sample_rate=sr,
            metadata={
                **signal.metadata,
                "ola_window": self.window,
                "ola_window_size": self.window_size,
                "ola_hop_size": self.hop_size,
                "ola_jitter_ms": self.jitter_ms,
                "ola_jitter_samples": jitter_samples,
                "ola_seed": self.seed,
            },
        )
        return output
