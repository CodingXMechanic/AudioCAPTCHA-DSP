from __future__ import annotations
import warnings
import numpy as np
from scipy.signal import resample_poly

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform

def _validate_nan_inf(waveform: np.ndarray, transform_name: str) -> np.ndarray:
    if not np.isfinite(waveform).all():
        warnings.warn(f"NaN or Inf encountered in {transform_name}. Replacing with zeros.")
        return np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    return waveform

class IdentityTransform(BaseTransform):
    """
    Returns an exact clone of the input signal.
    Reference: Trivially defined for baseline testing.
    """
    def __init__(self, name: str = "baseline.identity") -> None:
        super().__init__(name=name)

    def __call__(self, signal: Signal) -> Signal:
        output = signal.clone()
        output.metadata["identity_applied"] = True
        return output

class GainTransform(BaseTransform):
    """
    Applies a fixed gain in decibels (dB).
    Reference: standard linear scaling by 10^(dB/20).
    """
    def __init__(self, gain_db: float = 0.0, name: str = "baseline.gain") -> None:
        super().__init__(name=name)
        if not np.isfinite(gain_db):
            raise ValueError(f"Gain must be finite, got {gain_db}")
        self.gain_db = gain_db

    def __call__(self, signal: Signal) -> Signal:
        scale = 10.0 ** (self.gain_db / 20.0)
        new_waveform = signal.waveform * scale
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.gain_db": self.gain_db}
        )

class PeakNormalize(BaseTransform):
    """
    Normalizes the signal to a target peak decibel level.
    Reference: Digital peak normalization.
    """
    def __init__(self, target_db: float = -3.0, name: str = "baseline.peak_normalize") -> None:
        super().__init__(name=name)
        if not np.isfinite(target_db) or target_db > 0.0:
            raise ValueError(f"Target dB must be finite and <= 0.0, got {target_db}")
        self.target_db = target_db

    def __call__(self, signal: Signal) -> Signal:
        peak = np.max(np.abs(signal.waveform))
        if peak == 0.0:
            return signal.clone()
        
        target_linear = 10.0 ** (self.target_db / 20.0)
        scale = target_linear / peak
        new_waveform = signal.waveform * scale
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.target_db": self.target_db}
        )

class RMSNormalize(BaseTransform):
    """
    Normalizes the signal to a target RMS level.
    Reference: Root-Mean-Square normalization.
    """
    def __init__(self, target_rms: float = 0.1, name: str = "baseline.rms_normalize") -> None:
        super().__init__(name=name)
        if not np.isfinite(target_rms) or target_rms <= 0.0:
            raise ValueError(f"Target RMS must be finite and positive, got {target_rms}")
        self.target_rms = target_rms

    def __call__(self, signal: Signal) -> Signal:
        rms = np.sqrt(np.mean(signal.waveform**2))
        if rms == 0.0:
            return signal.clone()
        
        scale = self.target_rms / rms
        new_waveform = signal.waveform * scale
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.target_rms": self.target_rms}
        )

class LoudnessNormalize(BaseTransform):
    """
    Simple integrated loudness normalization using RMS as a proxy.
    Reference: ITU-R BS.1770-4 (Approximation via RMS).
    """
    def __init__(self, target_lufs: float = -23.0, name: str = "baseline.loudness_normalize") -> None:
        super().__init__(name=name)
        if not np.isfinite(target_lufs):
            raise ValueError(f"Target LUFS must be finite, got {target_lufs}")
        self.target_lufs = target_lufs

    def __call__(self, signal: Signal) -> Signal:
        rms = np.sqrt(np.mean(signal.waveform**2))
        if rms == 0.0:
            return signal.clone()
        
        current_lufs = 20 * np.log10(rms + 1e-12)
        gain_db = self.target_lufs - current_lufs
        scale = 10.0 ** (gain_db / 20.0)
        
        new_waveform = signal.waveform * scale
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.target_lufs": self.target_lufs}
        )

class DynamicRangeCompressor(BaseTransform):
    """
    IIR-based dynamic range compressor.
    Reference: Udo Zölzer, DAFX: Digital Audio Effects (2nd Edition).
    """
    def __init__(
        self,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        makeup_gain_db: float = 0.0,
        name: str = "baseline.compressor"
    ) -> None:
        super().__init__(name=name)
        if ratio < 1.0:
            raise ValueError(f"Ratio must be >= 1.0, got {ratio}")
        if attack_ms <= 0.0 or release_ms <= 0.0:
            raise ValueError("Attack and release times must be positive.")
            
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.makeup_gain_db = makeup_gain_db

    def __call__(self, signal: Signal) -> Signal:
        sr = signal.sample_rate
        attack_coeff = np.exp(-1.0 / (sr * (self.attack_ms / 1000.0)))
        release_coeff = np.exp(-1.0 / (sr * (self.release_ms / 1000.0)))
        
        x = signal.waveform
        original_shape = x.shape
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        y = np.zeros_like(x)
        makeup = 10.0 ** (self.makeup_gain_db / 20.0)
        
        for ch in range(x.shape[0]):
            env = 0.0
            for i in range(x.shape[1]):
                abs_x = abs(x[ch, i])
                level_db = 20 * np.log10(abs_x + 1e-12)
                
                # Gain computation in dB
                if level_db > self.threshold_db:
                    gain_db = (1.0 / self.ratio - 1.0) * (level_db - self.threshold_db)
                else:
                    gain_db = 0.0
                    
                target_gain = 10.0 ** (gain_db / 20.0)
                
                if target_gain < env:
                    env = attack_coeff * env + (1 - attack_coeff) * target_gain
                else:
                    env = release_coeff * env + (1 - release_coeff) * target_gain
                    
                y[ch, i] = x[ch, i] * env * makeup
                
        if len(original_shape) == 1:
            y = y.flatten()
            
        y = _validate_nan_inf(y, self.name)
        
        return Signal(
            waveform=y,
            sample_rate=sr,
            metadata={**signal.metadata, f"{self.name}.threshold_db": self.threshold_db}
        )

class Limiter(BaseTransform):
    """
    Hard limiter (clip).
    Reference: Basic hard clipping function.
    """
    def __init__(self, threshold_db: float = -1.0, name: str = "baseline.limiter") -> None:
        super().__init__(name=name)
        if threshold_db > 0.0:
            raise ValueError(f"Limiter threshold must be <= 0.0, got {threshold_db}")
        self.threshold_db = threshold_db

    def __call__(self, signal: Signal) -> Signal:
        threshold_linear = 10.0 ** (self.threshold_db / 20.0)
        new_waveform = np.clip(signal.waveform, -threshold_linear, threshold_linear)
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.threshold_db": self.threshold_db}
        )

class SilencePadding(BaseTransform):
    """
    Pads the signal with silence.
    Reference: Standard zero-padding in time domain.
    """
    def __init__(self, pad_ms: float = 500.0, location: str = "both", name: str = "baseline.silence_padding") -> None:
        super().__init__(name=name)
        if pad_ms < 0:
            raise ValueError(f"pad_ms must be non-negative, got {pad_ms}")
        if location not in ('start', 'end', 'both'):
            raise ValueError(f"location must be 'start', 'end', or 'both', got {location}")
        self.pad_ms = pad_ms
        self.location = location

    def __call__(self, signal: Signal) -> Signal:
        pad_samples = int(signal.sample_rate * (self.pad_ms / 1000.0))
        
        pad_width = []
        for i in range(signal.waveform.ndim):
            if i == signal.waveform.ndim - 1:
                if self.location == 'start':
                    pad_width.append((pad_samples, 0))
                elif self.location == 'end':
                    pad_width.append((0, pad_samples))
                else: # both
                    pad_width.append((pad_samples, pad_samples))
            else:
                pad_width.append((0, 0))
                
        new_waveform = np.pad(signal.waveform, pad_width, mode='constant', constant_values=0.0)
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.pad_ms": self.pad_ms, f"{self.name}.location": self.location}
        )

class SampleRateConverter(BaseTransform):
    """
    Resamples the signal to a new sample rate using polyphase filtering.
    Reference: scipy.signal.resample_poly
    """
    def __init__(self, target_sr: int = 8000, name: str = "baseline.sample_rate_converter") -> None:
        super().__init__(name=name)
        if target_sr <= 0:
            raise ValueError(f"Target SR must be positive, got {target_sr}")
        self.target_sr = target_sr

    def __call__(self, signal: Signal) -> Signal:
        original_sr = signal.sample_rate
        if self.target_sr == original_sr:
            return signal.clone()
        
        downsampled = resample_poly(signal.waveform, self.target_sr, original_sr, axis=-1)
        resampled_back = resample_poly(downsampled, original_sr, self.target_sr, axis=-1)
        
        resampled_back = _validate_nan_inf(resampled_back, self.name)
        
        return Signal(
            waveform=resampled_back,
            sample_rate=original_sr,
            metadata={**signal.metadata, f"{self.name}.target_sr": self.target_sr}
        )

class BitDepthConverter(BaseTransform):
    """
    Quantizes the signal to a target bit depth.
    Reference: Uniform scalar quantization.
    """
    def __init__(self, bits: int = 8, name: str = "baseline.bit_depth_converter") -> None:
        super().__init__(name=name)
        if bits < 1 or bits > 32:
            raise ValueError(f"Bits must be between 1 and 32, got {bits}")
        self.bits = bits

    def __call__(self, signal: Signal) -> Signal:
        levels = 2 ** self.bits
        
        # Normalize to -1 to 1
        peak = np.max(np.abs(signal.waveform))
        if peak == 0.0:
            return signal.clone()
            
        norm = signal.waveform / peak
        
        # Quantize
        quantized = np.round(norm * (levels / 2 - 1)) / (levels / 2 - 1)
        
        # Denormalize
        new_waveform = quantized * peak
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.bits": self.bits}
        )

class MuLawCompanding(BaseTransform):
    """
    Mu-law encoding and decoding as a transformation.
    Reference: ITU-T G.711 mu-law companding algorithm.
    """
    def __init__(self, mu: int = 255, name: str = "baseline.mu_law_companding") -> None:
        super().__init__(name=name)
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        self.mu = mu

    def __call__(self, signal: Signal) -> Signal:
        x = signal.waveform
        peak = np.max(np.abs(x))
        if peak == 0.0:
            return signal.clone()
            
        x_norm = x / peak
        
        # Encode
        encoded = np.sign(x_norm) * np.log1p(self.mu * np.abs(x_norm)) / np.log1p(self.mu)
        
        # Quantize to 8-bit typical for mu-law
        quantized = np.round(encoded * 127) / 127
        
        # Decode
        decoded = np.sign(quantized) * (1.0 / self.mu) * ((1.0 + self.mu)**np.abs(quantized) - 1.0)
        
        new_waveform = decoded * peak
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.mu": self.mu}
        )
