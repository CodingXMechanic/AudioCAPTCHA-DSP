import numpy as np
import librosa
from audiocaptcha_dsp.transforms.base import BaseTransform
from audiocaptcha_dsp.core.signal import Signal
import scipy.interpolate

class TimeStretch(BaseTransform):
    """Phase-vocoder based time stretching without pitch change.
    Ref: Laroche & Dolson (1999). Improved phase vocoder time-scale modification."""
    def __init__(self, rate=1.0, n_fft=512, hop_length=None, name='temporal.time_stretch', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, signal: Signal) -> Signal:
        if self.rate == 1.0:
            return signal.clone()
        y = librosa.effects.time_stretch(signal.waveform, rate=self.rate, n_fft=self.n_fft, hop_length=self.hop_length)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class PitchShift(BaseTransform):
    """Shift pitch by n_steps semitones without changing speed.
    Ref: Moulines & Laroche (1995). SOLA time-scale modification."""
    def __init__(self, n_steps=0.0, n_fft=512, hop_length=None, name='temporal.pitch_shift', seed=None):
        super().__init__(name=name)
        self.seed = seed
        self.n_steps = n_steps
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, signal: Signal) -> Signal:
        if self.n_steps == 0.0:
            return signal.clone()
        y = librosa.effects.pitch_shift(signal.waveform, sr=signal.sample_rate, n_steps=self.n_steps, 
                                        n_fft=self.n_fft, hop_length=self.hop_length)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class SpeedPerturbation(BaseTransform):
    """Change speed (and pitch) by a factor. Simple resampling approach."""
    def __init__(self, factor=1.0, name='temporal.speed_perturbation', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if factor <= 0:
            raise ValueError("factor must be > 0")
        self.factor = factor

    def __call__(self, signal: Signal) -> Signal:
        if self.factor == 1.0:
            return signal.clone()
        new_sr = int(signal.sample_rate / self.factor)
        y = librosa.resample(signal.waveform, orig_sr=signal.sample_rate, target_sr=new_sr)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class TimeMasking(BaseTransform):
    """SpecAugment-style time masking: zero out T consecutive time samples.
    Ref: Park et al. (2019). SpecAugment."""
    def __init__(self, max_mask_ms=100.0, num_masks=1, name='temporal.time_masking', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if max_mask_ms < 0 or num_masks < 0:
            raise ValueError("max_mask_ms and num_masks must be >= 0")
        self.max_mask_ms = max_mask_ms
        self.num_masks = num_masks

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        y = signal.waveform.copy()
        for _ in range(self.num_masks):
            t = rng.uniform(0, self.max_mask_ms)
            mask_samples = int(t * signal.sample_rate / 1000.0)
            if mask_samples > 0 and mask_samples < len(y):
                start = rng.integers(0, len(y) - mask_samples)
                y[start:start + mask_samples] = 0.0
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class TemporalDropout(BaseTransform):
    """Randomly zero out short segments of audio."""
    def __init__(self, dropout_rate=0.05, segment_ms=10.0, name='temporal.dropout', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if not (0 <= dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0 and 1")
        if segment_ms <= 0:
            raise ValueError("segment_ms must be > 0")
        self.dropout_rate = dropout_rate
        self.segment_ms = segment_ms

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        segment_samples = max(1, int(self.segment_ms * signal.sample_rate / 1000.0))
        num_segments = int(np.ceil(len(signal.waveform) / segment_samples))
        y = signal.waveform.copy()
        
        for i in range(num_segments):
            if rng.random() < self.dropout_rate:
                start = i * segment_samples
                end = min(len(y), (i + 1) * segment_samples)
                y[start:end] = 0.0
        return Signal(waveform=y, sample_rate=signal.sample_rate)

class LocalTimeWarping(BaseTransform):
    """Non-uniform time warping using cubic interpolation of random control points."""
    def __init__(self, warp_factor=0.1, num_control_points=5, name='temporal.local_warp', seed=None):
        super().__init__(name=name)
        self.seed = seed
        if not (0 <= warp_factor < 1.0):
            raise ValueError("warp_factor must be between 0 and 1")
        if num_control_points < 2:
            raise ValueError("num_control_points must be >= 2")
        self.warp_factor = warp_factor
        self.num_control_points = num_control_points

    def __call__(self, signal: Signal) -> Signal:
        if self.warp_factor == 0.0:
            return signal.clone()
        rng = np.random.default_rng(self.seed)
        n = len(signal.waveform)
        
        orig_points = np.linspace(0, n - 1, self.num_control_points)
        warped_points = orig_points.copy()
        max_displacement = (n / self.num_control_points) * self.warp_factor
        
        if self.num_control_points > 2:
            displacements = rng.uniform(-max_displacement, max_displacement, self.num_control_points - 2)
            warped_points[1:-1] += displacements
            
        warped_points = np.sort(warped_points)
        t_orig = np.arange(n)
        cs = scipy.interpolate.CubicSpline(orig_points, warped_points)
        t_warped = cs(t_orig)
        t_warped = np.clip(t_warped, 0, n - 1)
        
        y = np.interp(t_warped, t_orig, signal.waveform)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        return Signal(waveform=y, sample_rate=signal.sample_rate)
