from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from scipy.signal import resample_poly, stft, istft

from audiocaptcha_dsp.core.signal import Signal

logger = logging.getLogger(__name__)


class ASRDefense(ABC):
    """Abstract base class for preprocessing defense and enhancement pipelines."""
    
    def __init__(self, name: str = "base_defense") -> None:
        self.name = name

    @abstractmethod
    def __call__(self, signal: Signal) -> Signal:
        """Apply defense preprocessing to audio before ASR decoding."""
        ...


class IdentityDefense(ASRDefense):
    """Pass-through condition (no defense applied)."""
    
    def __init__(self) -> None:
        super().__init__(name="defense.raw")

    def __call__(self, signal: Signal) -> Signal:
        out = signal.clone()
        out.metadata = {**signal.metadata, "defense": self.name}
        return out


class LoudnessNormDefense(ASRDefense):
    """Loudness / RMS normalization defense to counter dynamic-range tampering."""
    
    def __init__(self, target_rms: float = 0.1) -> None:
        super().__init__(name="defense.loudness_norm")
        self.target_rms = target_rms

    def __call__(self, signal: Signal) -> Signal:
        waveform = signal.waveform.copy()
        current_rms = float(np.sqrt(np.mean(waveform ** 2)))
        if current_rms > 1e-8:
            waveform = waveform * (self.target_rms / current_rms)
        return Signal(waveform=waveform, sample_rate=signal.sample_rate, metadata={**signal.metadata, "defense": self.name})


class ResamplingDefense(ASRDefense):
    """Multirate bottleneck defense (anti-aliasing bandlimit, downsampling, and upsampling)."""
    
    def __init__(self, bottleneck_sr: int = 8000) -> None:
        super().__init__(name="defense.resampling")
        self.bottleneck_sr = bottleneck_sr

    def __call__(self, signal: Signal) -> Signal:
        orig_sr = signal.sample_rate
        if orig_sr == self.bottleneck_sr:
            return signal.clone()
            
        import librosa
        mono = signal.to_mono().waveform
        # Downsample to bottleneck then upsample back to original
        down = librosa.resample(mono, orig_sr=orig_sr, target_sr=self.bottleneck_sr)
        up = librosa.resample(down, orig_sr=self.bottleneck_sr, target_sr=orig_sr)
        
        # Ensure length matches original
        min_len = min(len(mono), len(up))
        out = np.zeros_like(mono)
        out[:min_len] = up[:min_len]
        
        return Signal(waveform=out, sample_rate=orig_sr, metadata={**signal.metadata, "defense": self.name, "bottleneck_sr": self.bottleneck_sr})


class SpectralDenoisingDefense(ASRDefense):
    """Spectral subtraction speech enhancement defense against additive perturbation."""
    
    def __init__(self, oversubtraction: float = 1.5, spectral_floor: float = 0.05) -> None:
        super().__init__(name="defense.spectral_denoise")
        self.oversubtraction = oversubtraction
        self.spectral_floor = spectral_floor

    def __call__(self, signal: Signal) -> Signal:
        mono = signal.to_mono().waveform
        sr = signal.sample_rate
        
        nperseg = min(512, len(mono))
        if nperseg < 64:
            return signal.clone()
            
        f, t, zxx = stft(mono, fs=sr, nperseg=nperseg)
        mag = np.abs(zxx)
        phase = np.angle(zxx)
        
        # Estimate noise floor from lowest energy frames
        frame_energies = np.sum(mag ** 2, axis=0)
        n_noise_frames = max(1, int(0.1 * len(frame_energies)))
        noise_frame_indices = np.argsort(frame_energies)[:n_noise_frames]
        noise_profile = np.mean(mag[:, noise_frame_indices], axis=1, keepdims=True)
        
        # Spectral subtraction with oversubtraction factor
        subtracted = mag - self.oversubtraction * noise_profile
        floored = np.maximum(subtracted, self.spectral_floor * mag)
        
        recon = floored * np.exp(1j * phase)
        _, enhanced = istft(recon, fs=sr, nperseg=nperseg)
        
        # Match original length
        out = np.zeros_like(mono)
        min_len = min(len(mono), len(enhanced))
        out[:min_len] = enhanced[:min_len]
        
        return Signal(waveform=out, sample_rate=sr, metadata={**signal.metadata, "defense": self.name})


class CodecSimulationDefense(ASRDefense):
    """Codec compression defense simulation (mu-law non-linear companding + quantization)."""
    
    def __init__(self, mu: int = 255) -> None:
        super().__init__(name="defense.codec_sim")
        self.mu = mu

    def __call__(self, signal: Signal) -> Signal:
        x = signal.to_mono().waveform
        x_max = np.max(np.abs(x))
        if x_max < 1e-8:
            return signal.clone()
        x_norm = x / x_max
        # Mu-law compression
        y = np.sign(x_norm) * np.log(1.0 + self.mu * np.abs(x_norm)) / np.log(1.0 + self.mu)
        # Quantization to 8-bit
        quantized = np.round(y * 127.0) / 127.0
        # Mu-law expansion
        x_rec = np.sign(quantized) * ((1.0 + self.mu) ** np.abs(quantized) - 1.0) / self.mu
        out = x_rec * x_max
        return Signal(waveform=out, sample_rate=signal.sample_rate, metadata={**signal.metadata, "defense": self.name})


class ReplaySimulationDefense(ASRDefense):
    """Acoustic room impulse response replay simulation."""
    
    def __init__(self, decay: float = 0.4, delay_ms: float = 30.0) -> None:
        super().__init__(name="defense.replay_sim")
        self.decay = decay
        self.delay_ms = delay_ms

    def __call__(self, signal: Signal) -> Signal:
        mono = signal.to_mono().waveform
        sr = signal.sample_rate
        delay_samples = int(round(self.delay_ms / 1000.0 * sr))
        
        replayed = mono.copy()
        if delay_samples < len(mono):
            replayed[delay_samples:] += self.decay * mono[:-delay_samples]
            
        return Signal(waveform=replayed, sample_rate=sr, metadata={**signal.metadata, "defense": self.name})


class DefensePipeline:
    """Manager to evaluate audio transformations against all defense conditions."""
    
    DEFENSES: list[type[ASRDefense]] = [
        IdentityDefense,
        LoudnessNormDefense,
        ResamplingDefense,
        SpectralDenoisingDefense,
        CodecSimulationDefense,
        ReplaySimulationDefense,
    ]

    @classmethod
    def all_defenses(cls) -> list[ASRDefense]:
        return [d() for d in cls.DEFENSES]
