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

class WhiteNoise(BaseTransform):
    """
    Add white Gaussian noise at a specific SNR.
    Reference: Standard AWGN channel model.
    """
    def __init__(self, snr_db: float = 20.0, seed: int | None = None, name: str = "noise.white") -> None:
        super().__init__(name=name)
        if not np.isfinite(snr_db):
            raise ValueError("SNR must be a finite number.")
        self.snr_db = snr_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        noise = rng.standard_normal(signal.waveform.shape)
        
        sig_rms = np.sqrt(np.mean(signal.waveform**2))
        if sig_rms == 0.0:
            noise_rms = 0.01
        else:
            noise_rms = sig_rms / (10.0 ** (self.snr_db / 20.0))
            
        current_noise_rms = np.sqrt(np.mean(noise**2))
        if current_noise_rms > 0:
            noise = noise * (noise_rms / current_noise_rms)
        
        new_waveform = signal.waveform + noise
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db}
        )

class PinkNoise(BaseTransform):
    """
    Add pink noise (1/f spectrum) at a specific SNR.
    Reference: Voss, R. F. (1978). "Linearity of 1/f noise mechanisms".
    """
    def __init__(self, snr_db: float = 20.0, seed: int | None = None, name: str = "noise.pink") -> None:
        super().__init__(name=name)
        if not np.isfinite(snr_db):
            raise ValueError("SNR must be a finite number.")
        self.snr_db = snr_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        N = signal.waveform.shape[-1]
        
        # Generate white noise
        white_noise = rng.standard_normal(signal.waveform.shape)
        
        # FFT
        X = np.fft.rfft(white_noise, axis=-1)
        
        # 1/sqrt(f) filter
        freqs = np.fft.rfftfreq(N)
        freqs[0] = freqs[1]  # avoid division by zero
        filter_shape = 1.0 / np.sqrt(freqs)
        
        X_filtered = X * filter_shape
        
        # IFFT
        noise = np.fft.irfft(X_filtered, n=N, axis=-1)
        
        sig_rms = np.sqrt(np.mean(signal.waveform**2))
        if sig_rms == 0.0:
            noise_rms = 0.01
        else:
            noise_rms = sig_rms / (10.0 ** (self.snr_db / 20.0))
            
        current_noise_rms = np.sqrt(np.mean(noise**2))
        if current_noise_rms > 0:
            noise = noise * (noise_rms / current_noise_rms)
        
        new_waveform = signal.waveform + noise
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db}
        )

class BrownNoise(BaseTransform):
    """
    Add brown/red noise (1/f^2 spectrum) at a specific SNR.
    Reference: Standard Brownian motion noise model.
    """
    def __init__(self, snr_db: float = 20.0, seed: int | None = None, name: str = "noise.brown") -> None:
        super().__init__(name=name)
        if not np.isfinite(snr_db):
            raise ValueError("SNR must be a finite number.")
        self.snr_db = snr_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        
        white_noise = rng.standard_normal(signal.waveform.shape)
        noise = np.cumsum(white_noise, axis=-1)
        
        # Detrending (removing DC)
        noise = noise - np.mean(noise, axis=-1, keepdims=True)
        
        sig_rms = np.sqrt(np.mean(signal.waveform**2))
        if sig_rms == 0.0:
            noise_rms = 0.01
        else:
            noise_rms = sig_rms / (10.0 ** (self.snr_db / 20.0))
            
        current_noise_rms = np.sqrt(np.mean(noise**2))
        if current_noise_rms > 0:
            noise = noise * (noise_rms / current_noise_rms)
        
        new_waveform = signal.waveform + noise
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db}
        )

class BandLimitedNoise(BaseTransform):
    """
    Add bandpass filtered noise at a specific SNR.
    """
    def __init__(self, snr_db: float = 20.0, low_hz: float = 300.0, high_hz: float = 3400.0, seed: int | None = None, name: str = "noise.band_limited") -> None:
        super().__init__(name=name)
        if not np.isfinite(snr_db):
            raise ValueError("SNR must be a finite number.")
        if low_hz >= high_hz or low_hz < 0:
            raise ValueError("Invalid frequency band.")
        self.snr_db = snr_db
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        from scipy.signal import butter, sosfilt
        rng = np.random.default_rng(self.seed)
        noise = rng.standard_normal(signal.waveform.shape)
        
        nyq = signal.sample_rate / 2.0
        low = self.low_hz / nyq
        high = min(self.high_hz / nyq, 0.999)
        
        sos = butter(4, [low, high], btype='bandpass', output='sos')
        if noise.ndim == 1:
            noise = sosfilt(sos, noise)
        else:
            for ch in range(noise.shape[0]):
                noise[ch] = sosfilt(sos, noise[ch])
                
        sig_rms = np.sqrt(np.mean(signal.waveform**2))
        if sig_rms == 0.0:
            noise_rms = 0.01
        else:
            noise_rms = sig_rms / (10.0 ** (self.snr_db / 20.0))
            
        current_noise_rms = np.sqrt(np.mean(noise**2))
        if current_noise_rms > 0:
            noise = noise * (noise_rms / current_noise_rms)
        
        new_waveform = signal.waveform + noise
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db}
        )

class SpeechShapedNoise(BaseTransform):
    """
    Noise shaped to match long-term speech spectrum.
    Reference: ITU-T P.50 approximation.
    """
    def __init__(self, snr_db: float = 20.0, seed: int | None = None, name: str = "noise.speech_shaped") -> None:
        super().__init__(name=name)
        if not np.isfinite(snr_db):
            raise ValueError("SNR must be a finite number.")
        self.snr_db = snr_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        N = signal.waveform.shape[-1]
        
        white_noise = rng.standard_normal(signal.waveform.shape)
        X = np.fft.rfft(white_noise, axis=-1)
        freqs = np.fft.rfftfreq(N, d=1.0/signal.sample_rate)
        
        # ITU-T P.50 approximation
        filter_shape = np.ones_like(freqs)
        f_low = 1000.0
        f_high = 4000.0
        
        mask_low = freqs < f_low
        if np.any(mask_low):
            filter_shape[mask_low] = freqs[mask_low] / f_low
            
        mask_high = freqs > f_high
        if np.any(mask_high):
            filter_shape[mask_high] = (f_high / freqs[mask_high])**2
            
        X_filtered = X * filter_shape
        noise = np.fft.irfft(X_filtered, n=N, axis=-1)
        
        sig_rms = np.sqrt(np.mean(signal.waveform**2))
        if sig_rms == 0.0:
            noise_rms = 0.01
        else:
            noise_rms = sig_rms / (10.0 ** (self.snr_db / 20.0))
            
        current_noise_rms = np.sqrt(np.mean(noise**2))
        if current_noise_rms > 0:
            noise = noise * (noise_rms / current_noise_rms)
        
        new_waveform = signal.waveform + noise
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db}
        )

class ImpulsiveNoise(BaseTransform):
    """
    Random clicks/pops at given rate.
    """
    def __init__(self, rate: float = 0.001, amplitude: float = 0.5, seed: int | None = None, name: str = "noise.impulsive") -> None:
        super().__init__(name=name)
        if rate < 0 or rate > 1:
            raise ValueError("Rate must be between 0 and 1.")
        self.rate = rate
        self.amplitude = amplitude
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        noise = np.zeros_like(signal.waveform)
        mask = rng.random(noise.shape) < self.rate
        noise[mask] = rng.choice([-self.amplitude, self.amplitude], size=np.count_nonzero(mask))
        
        new_waveform = signal.waveform + noise
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.rate": self.rate}
        )

class TonalInterference(BaseTransform):
    """
    Single sinusoidal interferer.
    """
    def __init__(self, frequency_hz: float = 1000.0, snr_db: float = 20.0, seed: int | None = None, name: str = "noise.tonal") -> None:
        super().__init__(name=name)
        if not np.isfinite(snr_db):
            raise ValueError("SNR must be a finite number.")
        if frequency_hz <= 0:
            raise ValueError("Frequency must be positive.")
        self.frequency_hz = frequency_hz
        self.snr_db = snr_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        N = signal.waveform.shape[-1]
        t = np.arange(N) / signal.sample_rate
        
        phase = rng.uniform(0, 2*np.pi)
        tone = np.sin(2 * np.pi * self.frequency_hz * t + phase)
        if signal.waveform.ndim > 1:
            tone = np.broadcast_to(tone, signal.waveform.shape)
            
        sig_rms = np.sqrt(np.mean(signal.waveform**2))
        if sig_rms == 0.0:
            noise_rms = 0.01
        else:
            noise_rms = sig_rms / (10.0 ** (self.snr_db / 20.0))
            
        current_noise_rms = np.sqrt(np.mean(tone**2))
        if current_noise_rms > 0:
            tone = tone * (noise_rms / current_noise_rms)
        
        new_waveform = signal.waveform + tone
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db}
        )


class VioletNoise(BaseTransform):
    """Add violet noise (increasing power with frequency) at a specific SNR.
    Reference: Standard violet noise model for perceptual testing.
    """
    def __init__(self, snr_db: float = 20.0, seed: int | None = None, name: str = "noise.violet") -> None:
        super().__init__(name=name)
        if not np.isfinite(snr_db):
            raise ValueError("SNR must be a finite number.")
        self.snr_db = snr_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        N = signal.waveform.shape[-1]
        
        # Generate white noise
        white_noise = rng.standard_normal(signal.waveform.shape)
        
        # FFT
        X = np.fft.rfft(white_noise, axis=-1)
        
        # f filter (violet noise has increasing power with frequency)
        freqs = np.fft.rfftfreq(N)
        freqs[0] = freqs[1]  # avoid division by zero
        filter_shape = freqs  # Linear increase with frequency
        
        X_filtered = X * filter_shape
        
        # IFFT
        noise = np.fft.irfft(X_filtered, n=N, axis=-1)
        
        sig_rms = np.sqrt(np.mean(signal.waveform**2))
        if sig_rms == 0.0:
            noise_rms = 0.01
        else:
            noise_rms = sig_rms / (10.0 ** (self.snr_db / 20.0))
            
        current_noise_rms = np.sqrt(np.mean(noise**2))
        if current_noise_rms > 0:
            noise = noise * (noise_rms / current_noise_rms)
        
        new_waveform = signal.waveform + noise
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db}
        )


class Chirp(BaseTransform):
    """Add a chirp signal (frequency sweep) at a specific SNR.
    
    A chirp is a signal in which the frequency increases or decreases with time.
    Useful for testing ASR robustness to frequency-modulated interference.
    """
    def __init__(self, start_hz: float = 200.0, end_hz: float = 3400.0, duration_sec: float | None = None, snr_db: float = 20.0, seed: int | None = None, name: str = "noise.chirp") -> None:
        super().__init__(name=name)
        if not np.isfinite(snr_db):
            raise ValueError("SNR must be a finite number.")
        if start_hz <= 0 or end_hz <= 0:
            raise ValueError("Frequencies must be positive.")
        self.start_hz = start_hz
        self.end_hz = end_hz
        self.duration_sec = duration_sec
        self.snr_db = snr_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        N = signal.waveform.shape[-1]
        sr = signal.sample_rate
        
        if self.duration_sec is None:
            self.duration_sec = N / sr
        
        t = np.linspace(0, self.duration_sec, N, endpoint=False)
        
        # Linear frequency sweep
        phase = 2 * np.pi * (self.start_hz * t + (self.end_hz - self.start_hz) * t**2 / (2 * self.duration_sec))
        chirp_signal = np.sin(phase)
        
        sig_rms = np.sqrt(np.mean(signal.waveform**2))
        if sig_rms == 0.0:
            chirp_rms = 0.01
        else:
            chirp_rms = sig_rms / (10.0 ** (self.snr_db / 20.0))
        
        current_chirp_rms = np.sqrt(np.mean(chirp_signal**2))
        if current_chirp_rms > 0:
            chirp_signal = chirp_signal * (chirp_rms / current_chirp_rms)
        
        new_waveform = signal.waveform + chirp_signal
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db, f"{self.name}.start_hz": self.start_hz, f"{self.name}.end_hz": self.end_hz}
        )


class ModulatedNoise(BaseTransform):
    """Add modulated noise (e.g., amplitude-modulated or frequency-modulated)."""
    def __init__(self, snr_db: float = 20.0, modulation_freq_hz: float = 5.0, modulation_index: float = 0.5, seed: int | None = None, name: str = "noise.modulated") -> None:
        super().__init__(name=name)
        if not np.isfinite(snr_db):
            raise ValueError("SNR must be a finite number.")
        if modulation_freq_hz <= 0:
            raise ValueError("Modulation frequency must be positive.")
        self.snr_db = snr_db
        self.modulation_freq_hz = modulation_freq_hz
        self.modulation_index = modulation_index
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        N = signal.waveform.shape[-1]
        sr = signal.sample_rate
        
        # Generate base white noise
        white_noise = rng.standard_normal(N)
        
        # Apply amplitude modulation
        t = np.arange(N) / sr
        env = 1.0 + self.modulation_index * np.sin(2 * np.pi * self.modulation_freq_hz * t)
        
        # Scale noise by envelope
        modulated_noise = white_noise * env
        
        # Normalize to target SNR
        sig_rms = np.sqrt(np.mean(signal.waveform**2))
        if sig_rms == 0.0:
            noise_rms = 0.01
        else:
            noise_rms = sig_rms / (10.0 ** (self.snr_db / 20.0))
        
        current_noise_rms = np.sqrt(np.mean(modulated_noise**2))
        if current_noise_rms > 0:
            modulated_noise = modulated_noise * (noise_rms / current_noise_rms)
        
        new_waveform = signal.waveform + modulated_noise
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db, f"{self.name}.mod_freq_hz": self.modulation_freq_hz, f"{self.name}.mod_index": self.modulation_index}
        )


class CompetingSpeakerNoise(BaseTransform):
    """Add competing speaker noise (multiple simultaneous talkers)."""
    def __init__(self, snr_db: float = 20.0, n_speakers: int = 2, seed: int | None = None, name: str = "noise.competing_speaker") -> None:
        super().__init__(name=name)
        if not np.isfinite(snr_db):
            raise ValueError("SNR must be a finite number.")
        if n_speakers < 1:
            raise ValueError("n_speakers must be >= 1.")
        self.snr_db = snr_db
        self.n_speakers = n_speakers
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        N = signal.waveform.shape[-1]
        sr = signal.sample_rate
        
        # Generate multiple speaker segments
        mixed = np.zeros(N)
        for _ in range(self.n_speakers):
            # Random pitch between 100-300 Hz
            f0 = rng.uniform(100, 300)
            # Random duration segment
            segment_len = rng.integers(int(0.5 * sr), int(2.0 * sr))
            start = rng.integers(0, max(1, N - segment_len))
            
            # Generate vowel-like segment
            segment_t = np.linspace(start / sr, (start + segment_len) / sr, segment_len, endpoint=False)
            segment = rng.normal(0, 0.5, segment_len) * np.sin(2 * np.pi * f0 * segment_t)
            
            # Add with random overlap
            end = min(N, start + segment_len)
            mixed[start:end] += segment[:end-start]
        
        # Normalize to target SNR
        sig_rms = np.sqrt(np.mean(signal.waveform**2))
        if sig_rms == 0.0:
            mixed_rms = 0.01
        else:
            mixed_rms = sig_rms / (10.0 ** (self.snr_db / 20.0))
        
        current_mixed_rms = np.sqrt(np.mean(mixed**2))
        if current_mixed_rms > 0:
            mixed = mixed * (mixed_rms / current_mixed_rms)
        
        new_waveform = signal.waveform + mixed
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.snr_db": self.snr_db, f"{self.name}.n_speakers": self.n_speakers}
        )


class ImpulsiveNoiseClicks(BaseTransform):
    """Add discrete clicks and pops (impulsive noise)."""
    def __init__(self, click_rate: float = 0.01, click_amplitude: float = 0.5, seed: int | None = None, name: str = "noise.clicks") -> None:
        super().__init__(name=name)
        if click_rate < 0 or click_rate > 1:
            raise ValueError("click_rate must be between 0 and 1.")
        self.click_rate = click_rate
        self.click_amplitude = click_amplitude
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        noise = np.zeros_like(signal.waveform)
        
        n_clicks = max(1, int(signal.waveform.shape[-1] * self.click_rate))
        click_positions = rng.choice(signal.waveform.shape[-1], size=n_clicks, replace=False)
        
        for pos in click_positions:
            # Create a short click
            click_len = min(100, len(signal.waveform) - pos)
            click = self.click_amplitude * rng.uniform(-1, 1, click_len)
            noise[pos:pos+click_len] += click
        
        new_waveform = signal.waveform + noise
        new_waveform = _validate_nan_inf(new_waveform, self.name)
        
        return Signal(
            waveform=new_waveform,
            sample_rate=signal.sample_rate,
            metadata={**signal.metadata, f"{self.name}.click_rate": self.click_rate, f"{self.name}.click_amp": self.click_amplitude}
        )