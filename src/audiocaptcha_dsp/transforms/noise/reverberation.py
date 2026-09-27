from __future__ import annotations
import warnings
import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform

def _validate_nan_inf(waveform: np.ndarray, transform_name: str) -> np.ndarray:
    if not np.isfinite(waveform).all():
        warnings.warn(f"NaN or Inf encountered in {transform_name}. Replacing with zeros.")
        return np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    return waveform

class Reverberation(BaseTransform):
    """
    Synthetic reverb using Schroeder-style all-pass + comb filter network.
    Reference: Schroeder, M. R. (1962). "Natural Sounding Artificial Reverberation".
    """
    def __init__(
        self,
        room_size: float = 0.4,
        damping: float = 0.5,
        wet_level: float = 0.3,
        dry_level: float = 0.7,
        seed: int | None = None,
        name: str = "noise.reverberation"
    ) -> None:
        super().__init__(name=name)
        if not (0 <= room_size <= 1):
            raise ValueError("Room size must be between 0 and 1.")
        if not (0 <= damping <= 1):
            raise ValueError("Damping must be between 0 and 1.")
        self.room_size = room_size
        self.damping = damping
        self.wet_level = wet_level
        self.dry_level = dry_level
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        x = signal.waveform
        sr = signal.sample_rate
        
        scale = sr / 44100.0
        
        # 4 comb delays, 2 all-pass delays
        comb_delays = [int(d * scale * self.room_size) for d in (1557, 1617, 1491, 1422)]
        allpass_delays = [int(d * scale * self.room_size) for d in (225, 556)]
        
        comb_delays = [max(1, d) for d in comb_delays]
        allpass_delays = [max(1, d) for d in allpass_delays]
        
        def comb_filter(sig, delay, damp):
            out = np.zeros_like(sig)
            buf = np.zeros(delay)
            idx = 0
            filter_store = 0.0
            
            for i in range(len(sig)):
                buf_out = buf[idx]
                filter_store = (buf_out * (1 - damp)) + (filter_store * damp)
                out[i] = buf_out
                
                buf[idx] = sig[i] + (filter_store * 0.84) # 0.84 is typical feedback gain
                idx = (idx + 1) % delay
            return out
            
        def allpass_filter(sig, delay):
            out = np.zeros_like(sig)
            buf = np.zeros(delay)
            idx = 0
            for i in range(len(sig)):
                buf_out = buf[idx]
                
                in_val = sig[i]
                buf[idx] = in_val + buf_out * 0.5
                out[i] = buf_out - buf[idx] * 0.5
                
                idx = (idx + 1) % delay
            return out

        if x.ndim == 1:
            wet = np.zeros_like(x)
            for d in comb_delays:
                wet += comb_filter(x, d, self.damping)
            
            for d in allpass_delays:
                wet = allpass_filter(wet, d)
                
            y = self.dry_level * x + self.wet_level * wet
        else:
            y = np.zeros_like(x)
            for ch in range(x.shape[0]):
                wet = np.zeros_like(x[ch])
                for d in comb_delays:
                    wet += comb_filter(x[ch], d, self.damping)
                for d in allpass_delays:
                    wet = allpass_filter(wet, d)
                y[ch] = self.dry_level * x[ch] + self.wet_level * wet
                
        y = _validate_nan_inf(y, self.name)
        
        return Signal(
            waveform=y,
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.room_size": self.room_size}
        )

class SimpleEcho(BaseTransform):
    """
    Single echo.
    """
    def __init__(self, delay_ms: float = 200.0, gain: float = 0.3, seed: int | None = None, name: str = "noise.echo") -> None:
        super().__init__(name=name)
        if delay_ms < 0:
            raise ValueError("Delay must be non-negative.")
        self.delay_ms = delay_ms
        self.gain = gain
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        sr = signal.sample_rate
        delay_samples = int(sr * (self.delay_ms / 1000.0))
        
        if delay_samples == 0:
            new_waveform = signal.waveform * (1 + self.gain)
        else:
            pad_width = []
            for i in range(signal.waveform.ndim):
                if i == signal.waveform.ndim - 1:
                    pad_width.append((delay_samples, 0))
                else:
                    pad_width.append((0, 0))
                    
            delayed = np.pad(signal.waveform, pad_width, mode='constant', constant_values=0.0)
            
            # Truncate to original length to preserve shape
            if delayed.ndim == 1:
                delayed = delayed[:signal.waveform.shape[-1]]
            else:
                delayed = delayed[..., :signal.waveform.shape[-1]]
                
            new_waveform = signal.waveform + self.gain * delayed
            
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.delay_ms": self.delay_ms}
        )
