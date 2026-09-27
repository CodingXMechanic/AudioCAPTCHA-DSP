import numpy as np
import librosa
from audiocaptcha_dsp.transforms.base import BaseTransform
from audiocaptcha_dsp.core.signal import Signal

class MelBandMasking(BaseTransform):
    """Mask random mel frequency bands."""
    def __init__(self, max_mask_bands=2, n_mels=80, name='spectral.mel_masking', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if max_mask_bands < 0 or n_mels <= 0:
            raise ValueError("Invalid parameters for MelBandMasking")
        self.max_mask_bands = max_mask_bands
        self.n_mels = n_mels

    def __call__(self, signal: Signal) -> Signal:
        if self.max_mask_bands == 0:
            return signal.clone()
            
        rng = np.random.default_rng(self.seed)
        
        mel_freqs = librosa.mel_frequencies(n_mels=self.n_mels + 2, fmin=0.0, fmax=signal.sample_rate/2)
        
        spec = np.fft.rfft(signal.waveform)
        freqs = np.fft.rfftfreq(len(signal.waveform), d=1.0/signal.sample_rate)
        
        for _ in range(self.max_mask_bands):
            f_idx = rng.integers(0, self.n_mels)
            low_hz = mel_freqs[f_idx]
            high_hz = mel_freqs[f_idx + 2]
            
            mask = (freqs >= low_hz) & (freqs <= high_hz)
            spec[mask] = 0.0
            
        y = np.fft.irfft(spec, n=len(signal.waveform))
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class BarkBandMasking(BaseTransform):
    """Mask random Bark-scale frequency bands."""
    def __init__(self, max_mask_bands=2, name='spectral.bark_masking', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if max_mask_bands < 0:
            raise ValueError("max_mask_bands must be >= 0")
        self.max_mask_bands = max_mask_bands

    def __call__(self, signal: Signal) -> Signal:
        if self.max_mask_bands == 0:
            return signal.clone()
            
        rng = np.random.default_rng(self.seed)
        spec = np.fft.rfft(signal.waveform)
        freqs = np.fft.rfftfreq(len(signal.waveform), d=1.0/signal.sample_rate)
        
        def hz_to_bark(hz):
            return 13 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500)**2)
        def bark_to_hz(z):
            return 1960 * (z + 0.53) / (26.28 - z)
            
        max_bark = hz_to_bark(signal.sample_rate / 2)
        n_bands = int(np.ceil(max_bark))
        if n_bands == 0:
            return signal.clone()
            
        for _ in range(self.max_mask_bands):
            z = rng.integers(0, n_bands)
            low_hz = bark_to_hz(z) if z > 0 else 0.0
            high_hz = bark_to_hz(z + 1)
            
            mask = (freqs >= low_hz) & (freqs <= high_hz)
            spec[mask] = 0.0
            
        y = np.fft.irfft(spec, n=len(signal.waveform))
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class CriticalBandAttenuation(BaseTransform):
    """Attenuate specific critical bands."""
    def __init__(self, band_indices=None, attenuation_db=20.0, name='spectral.critical_band_attenuation', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if attenuation_db < 0:
            raise ValueError("attenuation_db must be >= 0")
        self.band_indices = band_indices if band_indices is not None else [5, 10, 15]
        self.attenuation_db = attenuation_db

    def __call__(self, signal: Signal) -> Signal:
        if self.attenuation_db == 0.0 or not self.band_indices:
            return signal.clone()
            
        spec = np.fft.rfft(signal.waveform)
        freqs = np.fft.rfftfreq(len(signal.waveform), d=1.0/signal.sample_rate)
        
        def bark_to_hz(z):
            return 1960 * (z + 0.53) / (26.28 - z)
            
        linear_gain = 10 ** (-self.attenuation_db / 20.0)
        
        for z in self.band_indices:
            low_hz = bark_to_hz(z) if z > 0 else 0.0
            high_hz = bark_to_hz(z + 1)
            
            mask = (freqs >= low_hz) & (freqs <= high_hz)
            spec[mask] *= linear_gain
            
        y = np.fft.irfft(spec, n=len(signal.waveform))
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)
