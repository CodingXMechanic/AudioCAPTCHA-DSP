from __future__ import annotations

import logging
import time
from itertools import product
from pathlib import Path
from typing import Any

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.types import (
    BenchmarkResult,
    ExperimentConfig,
    ExperimentResult,
    Transform,
)
from audiocaptcha_dsp.evaluation.dataset import AudioDataset
from audiocaptcha_dsp.evaluation.metrics import compute_metrics
from audiocaptcha_dsp.evaluation.stats import ConditionSummary, summarize_condition
from audiocaptcha_dsp.io.artifacts import save_experiment_manifest, save_signal_wav
from audiocaptcha_dsp.transforms.base import BaseTransform
from audiocaptcha_dsp.transforms.composite.chain import TransformChain
from audiocaptcha_dsp.transforms.spectral.masking import MaskingInjection
from audiocaptcha_dsp.transforms.spectral.notch import SpectralNotch
from audiocaptcha_dsp.transforms.spectral.warping import FrequencyWarping
from audiocaptcha_dsp.transforms.temporal.jitter import TemporalJitter
from audiocaptcha_dsp.transforms.temporal.overlap_add import PerturbedOverlapAdd
from audiocaptcha_dsp.transforms.temporal.resampler import Resampler

from audiocaptcha_dsp.transforms.baseline.gain import (
    IdentityTransform, GainTransform, PeakNormalize, RMSNormalize, LoudnessNormalize,
    DynamicRangeCompressor, Limiter, SilencePadding, SampleRateConverter, BitDepthConverter, MuLawCompanding
)
from audiocaptcha_dsp.transforms.noise.additive import (
    WhiteNoise, PinkNoise, BrownNoise, BandLimitedNoise, SpeechShapedNoise, ImpulsiveNoise, TonalInterference
)
from audiocaptcha_dsp.transforms.noise.reverberation import Reverberation, SimpleEcho

# Extended temporal
from audiocaptcha_dsp.transforms.temporal.stretching import TimeStretch, PitchShift, SpeedPerturbation, TimeMasking, TemporalDropout, LocalTimeWarping
# Extended spectral filtering
from audiocaptcha_dsp.transforms.spectral.filtering import BandpassFilter, LowpassFilter, HighpassFilter, SpectralTilt, SpectralSmoothing, FrequencyBinDropout, PhaseRandomization, CombFilter, HarmonicAttenuation
# Extended mel masking
from audiocaptcha_dsp.transforms.spectral.mel_masking import MelBandMasking, BarkBandMasking, CriticalBandAttenuation
# Psychoacoustic constrained
from audiocaptcha_dsp.transforms.psychoacoustic.constrained import PsychoacousticNoiseInjection, BarkScalePerturbation
# Novel transforms (our paper's proposed contributions)
from audiocaptcha_dsp.transforms.novel import (
    PhonemeAwarePerturbation,
    PhonemeSegmentDropout,
    MultiDomainPerturbation,
    AdaptiveFormantPerturbation,
    CAPTCHAOptimalTransform,
    DefenseRobustTransform,
)

logger = logging.getLogger(__name__)

_TRANSFORM_REGISTRY: dict[str, type] = {
    "temporal.jitter": TemporalJitter,
    "temporal.resampler": Resampler,
    "temporal.overlap_add": PerturbedOverlapAdd,
    "spectral.masking": MaskingInjection,
    "spectral.notch": SpectralNotch,
    "spectral.warping": FrequencyWarping,
    "temporal.time_stretch": TimeStretch,
    "temporal.pitch_shift": PitchShift,
    "temporal.speed_perturbation": SpeedPerturbation,
    "temporal.time_masking": TimeMasking,
    "temporal.dropout": TemporalDropout,
    "temporal.local_warp": LocalTimeWarping,
    "spectral.bandpass": BandpassFilter,
    "spectral.lowpass": LowpassFilter,
    "spectral.highpass": HighpassFilter,
    "spectral.tilt": SpectralTilt,
    "spectral.smoothing": SpectralSmoothing,
    "spectral.freq_dropout": FrequencyBinDropout,
    "spectral.phase_randomization": PhaseRandomization,
    "spectral.comb_filter": CombFilter,
    "spectral.harmonic_attenuation": HarmonicAttenuation,
    "spectral.mel_masking": MelBandMasking,
    "spectral.bark_masking": BarkBandMasking,
    "spectral.critical_band_attenuation": CriticalBandAttenuation,
    "psychoacoustic.masked_noise": PsychoacousticNoiseInjection,
    "psychoacoustic.bark_perturbation": BarkScalePerturbation,
    "novel.phoneme_aware": PhonemeAwarePerturbation,
    "novel.phoneme_dropout": PhonemeSegmentDropout,
    "novel.multi_domain": MultiDomainPerturbation,
    "novel.adaptive_formant": AdaptiveFormantPerturbation,
    "novel.captcha_optimal": CAPTCHAOptimalTransform,
    "novel.defense_robust": DefenseRobustTransform,
    "baseline.identity": IdentityTransform,
    "baseline.gain": GainTransform,
    "baseline.peak_normalize": PeakNormalize,
    "baseline.rms_normalize": RMSNormalize,
    "baseline.loudness_normalize": LoudnessNormalize,
    "baseline.compressor": DynamicRangeCompressor,
    "baseline.limiter": Limiter,
    "baseline.silence_padding": SilencePadding,
    "baseline.sample_rate_converter": SampleRateConverter,
    "baseline.bit_depth_converter": BitDepthConverter,
    "baseline.mu_law_companding": MuLawCompanding,
    "noise.white": WhiteNoise,
    "noise.pink": PinkNoise,
    "noise.brown": BrownNoise,
    "noise.band_limited": BandLimitedNoise,
    "noise.speech_shaped": SpeechShapedNoise,
    "noise.impulsive": ImpulsiveNoise,
    "noise.tonal": TonalInterference,
    "noise.reverberation": Reverberation,
    "noise.echo": SimpleEcho,
}


def resolve_transform(type_name: str, params: dict[str, Any] | None = None) -> Transform:
    params = params or {}
    if type_name in _TRANSFORM_REGISTRY:
        cls = _TRANSFORM_REGISTRY[type_name]
        filtered = {k: v for k, v in params.items() if k not in ("condition_index", "sample_index")}
        return cls(**filtered)
    # Fall back to the central taxonomy registry (families A–H, full catalog).
    # Registry default parameters act as defaults; explicit params win.
    from audiocaptcha_dsp.transforms.registry import get_registry

    reg = get_registry()
    if type_name in reg:
        spec = reg[type_name]
        filtered = {
            k: v for k, v in params.items()
            if k not in ("condition_index", "sample_index", "type", "name")
        }
        merged = {**spec.params, **filtered}
        return spec.cls(**merged)
    raise ValueError(
        f"Unknown transform type: '{type_name}'. Available: "
        f"{sorted(set(_TRANSFORM_REGISTRY) | set(reg))}"
    )


def build_transform_chain(specs: list[dict[str, Any]]) -> TransformChain:
    transforms = []
    for spec in specs:
        t = resolve_transform(spec.get("type", ""), spec.get("parameters", {}))
        transforms.append(t)
    return TransformChain(transforms=transforms, name="experiment_chain")


def _build_condition_grid(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    if not parameters:
        return [{}]
    keys = sorted(parameters.keys())
    values = [parameters[k] for k in keys]
    combinations = list(product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, save_wav: bool = False) -> BenchmarkResult:
        logger.info("Starting experiment %s (%s)", self.config.experiment_id, self.config.name)
        start_time = time.perf_counter()

        dataset = AudioDataset.synthetic(n_signals=self.config.num_samples, seed=self.config.seed)
        signals, transcripts = dataset.load()

        if self.config.is_composite:
            condition_grid = self.config.composite_conditions
        else:
            condition_grid = _build_condition_grid(self.config.parameters)

        logger.info("Condition grid: %d conditions", len(condition_grid))

        transform_specs = []
        for spec in self.config.transform_specs:
            transform_specs.append({"type": spec.transform_type, "parameters": spec.parameters})

        all_results: list[ExperimentResult] = []
        per_condition: list[dict[str, Any]] = []

        for cond_idx, cond_params in enumerate(condition_grid):
            logger.info("Condition %d/%d: %s", cond_idx + 1, len(condition_grid), cond_params)

            if self.config.is_composite:
                merged_specs = self._merge_composite_specs(transform_specs, cond_params)
            else:
                merged_specs = self._merge_single_specs(transform_specs, cond_params)

            chain = build_transform_chain(merged_specs)
            chain_names = [ts["type"] for ts in merged_specs]

            condition_metrics: list[dict[str, float]] = []

            for sample_idx, signal in enumerate(signals):
                sample_start = time.perf_counter()
                processed = chain(signal)
                sample_duration = time.perf_counter() - sample_start
                metrics = compute_metrics(signal, processed)
                metrics["condition_index"] = cond_idx
                metrics["condition_params"] = str(cond_params)
                condition_metrics.append(metrics)

                result = ExperimentResult(
                    experiment_id=self.config.experiment_id,
                    condition_index=cond_idx,
                    condition_params=cond_params,
                    sample_index=sample_idx,
                    sample_id=signal.metadata.get("sample_id", f"sample_{sample_idx:04d}"),
                    metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k not in ("condition_index",)},
                    transform_chain=chain_names,
                    duration_seconds=sample_duration,
                    seed=self.config.seed,
                )
                all_results.append(result)

                if save_wav:
                    wav_dir = self.output_dir / self.config.experiment_id / f"cond_{cond_idx:03d}"
                    save_signal_wav(processed, wav_dir / f"{result.sample_id}_processed.wav")

            summary = summarize_condition(condition_metrics, ci=0.95, seed=self.config.seed)
            summary.condition_index = cond_idx
            summary.condition_params = cond_params
            summary.n_samples = len(signals)
            per_condition.append(summary.to_dict())

        total_duration = time.perf_counter() - start_time

        benchmark = BenchmarkResult(
            experiment_id=self.config.experiment_id,
            total_conditions=len(condition_grid),
            total_samples=len(signals) * len(condition_grid),
            total_duration_seconds=total_duration,
            per_condition=per_condition,
            results=all_results,
        )

        manifest_path = self.output_dir / self.config.experiment_id / "manifest.json"
        save_experiment_manifest(benchmark, manifest_path)

        logger.info(
            "Experiment %s complete: %d conditions, %d samples, %.2fs",
            self.config.experiment_id,
            benchmark.total_conditions,
            benchmark.total_samples,
            total_duration,
        )
        return benchmark

    @staticmethod
    def _merge_single_specs(transform_specs: list[dict[str, Any]], cond_params: dict[str, Any]) -> list[dict[str, Any]]:
        merged = []
        for ts in transform_specs:
            m = dict(ts)
            m_params = dict(m.get("parameters", {}))
            m_params.update(cond_params)
            m["parameters"] = m_params
            merged.append(m)
        return merged

    @staticmethod
    def _merge_composite_specs(transform_specs: list[dict[str, Any]], cond_params: dict[str, Any]) -> list[dict[str, Any]]:
        merged = []
        for ts in transform_specs:
            m = dict(ts)
            m_params: dict[str, Any] = {}
            orig_params = m.get("parameters", {})
            for pkey, pval in orig_params.items():
                if pkey in cond_params:
                    m_params[pkey] = cond_params[pkey]
                else:
                    m_params[pkey] = pval if not isinstance(pval, list) else pval[0]
            m["parameters"] = m_params
            merged.append(m)
        return merged
