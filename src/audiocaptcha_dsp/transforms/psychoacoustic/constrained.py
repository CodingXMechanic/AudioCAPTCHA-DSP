import numpy as np
from audiocaptcha_dsp.transforms.base import BaseTransform
from audiocaptcha_dsp.core.signal import Signal

try:
    from audiocaptcha_dsp.psychoacoustics.masking_models import (
        simultaneous_masking_threshold,
        compute_masking_budget,
        compute_perturbation_budget,
    )
    from audiocaptcha_dsp.psychoacoustics.bark_scale import hz_to_bark, bark_to_hz
    _HAVE_MODELS = True
except ImportError:  # pragma: no cover - defensive fallback
    _HAVE_MODELS = False

    def hz_to_bark(hz):
        return 13 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500) ** 2)

    def bark_to_hz(z):
        return 1960 * (z + 0.53) / (26.28 - z)

    def simultaneous_masking_threshold(signal, sr):
        """Fallback: 30 dB below per-bin magnitude (very conservative)."""
        mag = np.abs(np.fft.rfft(np.asarray(signal, dtype=np.float64)))
        return 20 * np.log10(mag + 1e-12) - 30.0

    def compute_masking_budget(signal, sr, margin_db=6.0):
        mag = np.abs(np.fft.rfft(np.asarray(signal, dtype=np.float64)))
        thresh_db = simultaneous_masking_threshold(signal, sr) - margin_db
        return (10.0 ** (thresh_db / 10.0))

    # Fallback is already in signal (FFT-bin) units — ATH mixing not present
    compute_perturbation_budget = compute_masking_budget


def _safe(y: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


class PsychoacousticNoiseInjection(BaseTransform):
    """Inject noise constrained below the psychoacoustic masking threshold.

    The per-bin perturbation *power* never exceeds the spreading-function
    masking budget of the input signal computed with a safety margin λ
    (``margin_db``), mirroring the MP3-style hearing-threshold constraint
    of Schönherr et al. (2018, arXiv:1808.05665). Conservative language:
    "psychoacoustically constrained" (WHAT-REMAINS.txt §3.F).
    """

    def __init__(self, margin_db=10.0, seed=None, name="psychoacoustic.masked_noise"):
        super().__init__(name=name)
        self.seed = seed
        if margin_db < 0:
            raise ValueError("margin_db must be >= 0")
        self.margin_db = margin_db

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        if x.ndim != 1:
            x = x.mean(axis=0)
        spec = np.fft.rfft(x)

        # Per-bin linear power budget below the masking threshold (with margin)
        budget_power = compute_perturbation_budget(x, signal.sample_rate, margin_db=self.margin_db)
        budget_power = np.asarray(budget_power, dtype=np.float64)
        if budget_power.shape != spec.shape:
            n = min(len(budget_power), len(spec))
            trimmed = np.zeros(len(spec), dtype=np.float64)
            trimmed[:n] = budget_power[:n]
            budget_power = trimmed
        allowed_mag = np.sqrt(np.maximum(budget_power, 0.0))

        noise_phase = rng.uniform(-np.pi, np.pi, len(spec))
        noise_spec = allowed_mag * np.exp(1j * noise_phase)

        y = np.fft.irfft(spec + noise_spec, n=len(x))
        return Signal(waveform=_safe(y), sample_rate=signal.sample_rate,
                      metadata={**signal.metadata, f"{self.name}.margin_db": self.margin_db})


class BarkScalePerturbation(BaseTransform):
    """Perturbation allocated according to a Bark-scale masking budget.

    Perturbation magnitude per FFT bin is the Bark-spread masking budget
    scaled by ``perturbation_scale`` and reduced by ``margin_db``.
    """

    def __init__(self, perturbation_scale=0.1, margin_db=6.0, seed=None,
                 name="psychoacoustic.bark_perturbation"):
        super().__init__(name=name)
        self.seed = seed
        if perturbation_scale < 0 or margin_db < 0:
            raise ValueError("perturbation_scale and margin_db must be >= 0")
        self.perturbation_scale = perturbation_scale
        self.margin_db = margin_db

    def __call__(self, signal: Signal) -> Signal:
        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        if x.ndim != 1:
            x = x.mean(axis=0)
        spec = np.fft.rfft(x)

        budget_power = compute_perturbation_budget(x, signal.sample_rate, margin_db=self.margin_db)
        budget_power = np.asarray(budget_power, dtype=np.float64)
        if budget_power.shape != spec.shape:
            n = min(len(budget_power), len(spec))
            trimmed = np.zeros(len(spec), dtype=np.float64)
            trimmed[:n] = budget_power[:n]
            budget_power = trimmed

        amp = np.sqrt(np.maximum(budget_power, 0.0)) * self.perturbation_scale
        noise_phase = rng.uniform(-np.pi, np.pi, len(spec))
        y = np.fft.irfft(spec + amp * np.exp(1j * noise_phase), n=len(x))
        return Signal(waveform=_safe(y), sample_rate=signal.sample_rate,
                      metadata={**signal.metadata,
                               f"{self.name}.scale": self.perturbation_scale,
                               f"{self.name}.margin_db": self.margin_db})
