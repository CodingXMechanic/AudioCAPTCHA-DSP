import numpy as np
import scipy.signal
from audiocaptcha_dsp.transforms.base import BaseTransform
from audiocaptcha_dsp.core.signal import Signal

class BandpassFilter(BaseTransform):
    """Butterworth bandpass filter."""
    def __init__(self, low_hz=300.0, high_hz=3400.0, order=4, name='spectral.bandpass', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if low_hz < 0 or high_hz <= low_hz or order <= 0:
            raise ValueError("Invalid parameters for BandpassFilter")
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order

    def __call__(self, signal: Signal) -> Signal:
        nyq = 0.5 * signal.sample_rate
        low = self.low_hz / nyq
        high = self.high_hz / nyq
        b, a = scipy.signal.butter(self.order, [low, high], btype='band')
        y = scipy.signal.lfilter(b, a, signal.waveform)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class LowpassFilter(BaseTransform):
    """Butterworth lowpass filter."""
    def __init__(self, cutoff_hz=4000.0, order=4, name='spectral.lowpass', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if cutoff_hz <= 0 or order <= 0:
            raise ValueError("Invalid parameters for LowpassFilter")
        self.cutoff_hz = cutoff_hz
        self.order = order

    def __call__(self, signal: Signal) -> Signal:
        nyq = 0.5 * signal.sample_rate
        if self.cutoff_hz >= nyq:
            return signal.clone()
        norm_cutoff = self.cutoff_hz / nyq
        b, a = scipy.signal.butter(self.order, norm_cutoff, btype='low')
        y = scipy.signal.lfilter(b, a, signal.waveform)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class HighpassFilter(BaseTransform):
    """Butterworth highpass filter."""
    def __init__(self, cutoff_hz=80.0, order=4, name='spectral.highpass', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if cutoff_hz <= 0 or order <= 0:
            raise ValueError("Invalid parameters for HighpassFilter")
        self.cutoff_hz = cutoff_hz
        self.order = order

    def __call__(self, signal: Signal) -> Signal:
        nyq = 0.5 * signal.sample_rate
        if self.cutoff_hz >= nyq:
            y = np.zeros_like(signal.waveform)
            return Signal(waveform=y, sample_rate=signal.sample_rate)
        norm_cutoff = self.cutoff_hz / nyq
        b, a = scipy.signal.butter(self.order, norm_cutoff, btype='high')
        y = scipy.signal.lfilter(b, a, signal.waveform)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class SpectralTilt(BaseTransform):
    """Apply linear spectral tilt (pre-emphasis or de-emphasis) in dB/octave."""
    def __init__(self, tilt_db_per_octave=6.0, pivot_hz=1000.0, name='spectral.tilt', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if pivot_hz <= 0:
            raise ValueError("pivot_hz must be > 0")
        self.tilt_db_per_octave = tilt_db_per_octave
        self.pivot_hz = pivot_hz

    def __call__(self, signal: Signal) -> Signal:
        if self.tilt_db_per_octave == 0.0:
            return signal.clone()
        
        freqs = np.fft.rfftfreq(len(signal.waveform), d=1.0/signal.sample_rate)
        spec = np.fft.rfft(signal.waveform)
        
        freqs = np.maximum(freqs, 1e-10)
        octaves = np.log2(freqs / self.pivot_hz)
        db_shift = self.tilt_db_per_octave * octaves
        linear_multiplier = 10 ** (db_shift / 20.0)
        
        spec_tilted = spec * linear_multiplier
        y = np.fft.irfft(spec_tilted, n=len(signal.waveform))
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class SpectralSmoothing(BaseTransform):
    """Smooth spectrum by convolution with a Hann window."""
    def __init__(self, smoothing_bins=10, name='spectral.smoothing', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if smoothing_bins < 1:
            raise ValueError("smoothing_bins must be >= 1")
        self.smoothing_bins = smoothing_bins

    def __call__(self, signal: Signal) -> Signal:
        if self.smoothing_bins == 1:
            return signal.clone()
            
        spec = np.fft.rfft(signal.waveform)
        mag = np.abs(spec)
        phase = np.angle(spec)
        
        window = np.hanning(self.smoothing_bins)
        window /= np.sum(window)
        
        mag_smoothed = np.convolve(mag, window, mode='same')
        spec_smoothed = mag_smoothed * np.exp(1j * phase)
        
        y = np.fft.irfft(spec_smoothed, n=len(signal.waveform))
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class FrequencyBinDropout(BaseTransform):
    """Randomly zero out frequency bins in STFT representation."""
    def __init__(self, dropout_rate=0.1, name='spectral.freq_dropout', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if not (0 <= dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0 and 1")
        self.dropout_rate = dropout_rate

    def __call__(self, signal: Signal) -> Signal:
        if self.dropout_rate == 0.0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        spec = np.fft.rfft(signal.waveform)
        
        mask = rng.random(len(spec)) >= self.dropout_rate
        spec *= mask
        
        y = np.fft.irfft(spec, n=len(signal.waveform))
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class PhaseRandomization(BaseTransform):
    """Randomize phase spectrum while preserving magnitude spectrum."""
    def __init__(self, randomization_strength=1.0, name='spectral.phase_randomization', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if not (0 <= randomization_strength <= 1.0):
            raise ValueError("randomization_strength must be between 0 and 1")
        self.randomization_strength = randomization_strength

    def __call__(self, signal: Signal) -> Signal:
        if self.randomization_strength == 0.0:
            return signal.clone()
            
        rng = np.random.default_rng(self.seed)
        spec = np.fft.rfft(signal.waveform)
        mag = np.abs(spec)
        orig_phase = np.angle(spec)
        
        rand_phase = rng.uniform(-np.pi, np.pi, len(spec))
        
        diff = rand_phase - orig_phase
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
        new_phase = orig_phase + self.randomization_strength * diff
        
        spec_new = mag * np.exp(1j * new_phase)
        y = np.fft.irfft(spec_new, n=len(signal.waveform))
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class CombFilter(BaseTransform):
    """Apply comb filter (delay line with feedback)."""
    def __init__(self, delay_ms=10.0, gain=0.5, name='spectral.comb_filter', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if delay_ms <= 0:
            raise ValueError("delay_ms must be > 0")
        self.delay_ms = delay_ms
        self.gain = gain

    def __call__(self, signal: Signal) -> Signal:
        delay_samples = int(self.delay_ms * signal.sample_rate / 1000.0)
        if delay_samples == 0:
            return signal.clone()
            
        y = np.zeros_like(signal.waveform)
        for i in range(len(y)):
            if i < delay_samples:
                y[i] = signal.waveform[i]
            else:
                y[i] = signal.waveform[i] + self.gain * y[i - delay_samples]
                
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class HarmonicAttenuation(BaseTransform):
    """Attenuate harmonics of a fundamental frequency."""
    def __init__(self, fundamental_hz=100.0, num_harmonics=5, attenuation_db=10.0, bandwidth_hz=50.0, name='spectral.harmonic_attenuation', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if fundamental_hz <= 0 or num_harmonics < 1 or bandwidth_hz <= 0:
            raise ValueError("Invalid parameters for HarmonicAttenuation")
        self.fundamental_hz = fundamental_hz
        self.num_harmonics = num_harmonics
        self.attenuation_db = attenuation_db
        self.bandwidth_hz = bandwidth_hz

    def __call__(self, signal: Signal) -> Signal:
        nyq = 0.5 * signal.sample_rate
        y = signal.waveform.copy()
        
        linear_gain = 10 ** (-self.attenuation_db / 20.0)
        
        for k in range(1, self.num_harmonics + 1):
            fc = k * self.fundamental_hz
            if fc >= nyq:
                break
                
            Q = fc / self.bandwidth_hz
            if Q <= 0:
                continue
                
            w0 = fc / nyq
            b, a = scipy.signal.iirnotch(w0, Q)
            
            notched = scipy.signal.lfilter(b, a, y)
            y = notched * (1 - linear_gain) + y * linear_gain
            
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)
