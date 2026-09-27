"""
Family G: adversarial and representation-aware transformations (authorized research).
=====================================================================================
WHAT-REMAINS.txt §3.G items implemented here:

- Gradient-free parameter search against an ASR-objective surrogate
- Black-box transformation search with an explicit query budget
- Phoneme/representation-guided perturbation allocation
- Multi-objective optimization over perceptual + ASR objectives
  (non-dominated sorting, NSGA-II-style)

Threat-model note: the search is *model-agnostic* — the objective is
pluggable and never hard-coded to a single ASR family (WHAT-REMAINS.txt
§3.G: "Do not hard-code the system to attack only Whisper").

References
----------
- Spall (1992). SPSA — a stochastic approximation algorithm for
  optimization and stochastic simulation. IEEE TAC.
- Deb et al. (2002). A fast and elitist multiobjective genetic algorithm:
  NSGA-II. IEEE TEC.
- Papernot et al. (2016). Black-box attacks against deep learning systems
  (query-budgeted black-box search). arXiv:1602.02697.
- Paper 10 (arXiv:2307.12498): phoneme-level perturbations for ASR.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform

logger = logging.getLogger(__name__)

# Objective signature: (original, candidate) -> scalar to MAXIMIZE
Objective = Callable[[Signal, Signal], float]


def _default_objective(original: Signal, candidate: Signal) -> float:
    """Default surrogate objective: intelligibility loss at bounded distortion.

    Maximizes ``1 − STOI-proxy`` subject to a mild SNR floor, i.e. the
    strongest perceptible-but-intelligible distortion found by the search.
    Replace with a real ASR-WER objective for ASR-targeted experiments.
    """
    from audiocaptcha_dsp.evaluation.metrics import compute_stoi_proxy, compute_snr

    orig = original.to_mono().waveform
    cand = candidate.to_mono().waveform
    stoi = compute_stoi_proxy(orig, cand, sr=original.sample_rate)
    snr = compute_snr(orig, cand)
    penalty = 0.0 if (np.isfinite(snr) and snr >= 10.0) else 0.5
    return (1.0 - stoi) - penalty


def _default_psychoacoustic_factory(**params: Any) -> BaseTransform:
    """Default search space: psychoacoustic noise injection margin (λ).

    Lets family-G transforms run standalone (registry instantiation)
    while remaining *pluggable* — callers may pass any
    ``transform_factory``/``param_bounds`` to target another family.
    """
    from audiocaptcha_dsp.transforms.psychoacoustic.constrained import (
        PsychoacousticNoiseInjection,
    )

    return PsychoacousticNoiseInjection(**params)


#: Fallback black-box candidate library (registered keys + param grids)
DEFAULT_BLACKBOX_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    ("psychoacoustic.masked_noise", {"margin_db": 6.0}),
    ("psychoacoustic.masked_noise", {"margin_db": 20.0}),
    ("noise.white", {"snr_db": 20.0}),
    ("noise.white", {"snr_db": 10.0}),
    ("noise.pink", {"snr_db": 15.0}),
    ("spectral.lowpass", {"cutoff_hz": 6000.0}),
    ("spectral.lowpass", {"cutoff_hz": 3000.0}),
    ("multirate.narrowband", {}),
    ("temporal.speed_perturbation", {"factor": 1.1}),
    ("channel.mp3", {"bitrate_kbps": 32}),
]


class GradientFreeParameterSearch(BaseTransform):
    """Gradient-free (SPSA-style) parameter search over a transform family.

    Parameters
    ----------
    transform_factory : callable
        ``params -> BaseTransform``; builds the candidate transform for a
        given parameter vector.
    param_bounds : dict[str, tuple[float, float]]
        Box bounds for each searched parameter.
    objective : callable, optional
        ``(original, candidate) -> float`` to maximize. Defaults to the
        perceptual surrogate objective.
    n_iterations : int
        Number of SPSA iterations (2 objective calls each).
    query_budget : int
        Hard cap on objective evaluations (models a query-limited attacker).

    Notes
    -----
    This is the authorized-research, gradient-free analogue of
    white-box attacks: no model internals are accessed, only objective
    queries. ``search()`` caches the best parameters per transform.
    """

    def __init__(
        self,
        transform_factory: Callable[..., BaseTransform] | None = None,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        objective: Objective | None = None,
        n_iterations: int = 20,
        query_budget: int = 64,
        perturbation: float = 0.1,
        seed: int | None = None,
        name: str = "adversarial.gradient_free",
    ) -> None:
        super().__init__(name=name)
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        if query_budget < 2:
            raise ValueError("query_budget must be >= 2")
        # Standalone default: search the psychoacoustic λ-margin (runnable
        # without a custom factory; callers may override both arguments).
        self.transform_factory = transform_factory or _default_psychoacoustic_factory
        self.param_bounds = dict(
            param_bounds if param_bounds is not None else {"margin_db": (0.0, 40.0)}
        )
        self.objective = objective or _default_objective
        self.n_iterations = n_iterations
        self.query_budget = query_budget
        self.perturbation = perturbation
        self.seed = seed
        self.best_params_: dict[str, float] | None = None
        self.best_score_: float = -np.inf
        self.n_queries_: int = 0
        self.history_: list[float] = []

    # -- helpers -----------------------------------------------------------
    def _x0(self) -> np.ndarray:
        if not self.param_bounds:
            return np.zeros(0)
        return np.array(
            [(lo + hi) / 2.0 for lo, hi in self.param_bounds.values()], dtype=np.float64
        )

    def _clip(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for i, (lo, hi) in enumerate(self.param_bounds.values()):
            out[i] = float(np.clip(out[i], lo, hi))
        return out

    def _params_dict(self, x: np.ndarray) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self.param_bounds.keys(), x)}

    def _evaluate(self, original: Signal, params: dict[str, float]) -> float:
        if self.n_queries_ >= self.query_budget:
            return -np.inf
        if self.transform_factory is None:
            raise ValueError("transform_factory is required to evaluate candidates")
        candidate = self.transform_factory(**params)(original)
        self.n_queries_ += 1
        score = float(self.objective(original, candidate))
        if not np.isfinite(score):
            score = -np.inf
        if score > self.best_score_:
            self.best_score_ = score
            self.best_params_ = dict(params)
        self.history_.append(score)
        return score

    # -- public API --------------------------------------------------------
    def search(self, signal: Signal) -> dict[str, Any]:
        """Run the SPSA search on ``signal``; return a result summary."""
        if not self.param_bounds:
            raise ValueError("param_bounds must be provided before search()")
        rng = np.random.default_rng(self.seed)
        x = self._x0()
        self.n_queries_ = 0
        self.history_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf

        base = self._evaluate(signal, self._params_dict(x))
        best = base
        for _ in range(self.n_iterations):
            if self.n_queries_ >= self.query_budget:
                break
            delta = rng.choice([-1.0, 1.0], size=len(x))
            eps = self.perturbation * (np.max([hi - lo for lo, hi in self.param_bounds.values()]) + 1e-9)
            x_plus = self._clip(x + eps * delta)
            x_minus = self._clip(x - eps * delta)
            s_plus = self._evaluate(signal, self._params_dict(x_plus))
            if self.n_queries_ >= self.query_budget:
                break
            s_minus = self._evaluate(signal, self._params_dict(x_minus))
            if not np.isfinite(s_plus) or not np.isfinite(s_minus):
                continue
            # SPSA gradient estimate
            g = (s_plus - s_minus) / (2.0 * eps) * delta
            step = self.perturbation * (np.max([hi - lo for lo, hi in self.param_bounds.values()]) + 1e-9)
            x_new = self._clip(x + step * g)
            s_new = self._evaluate(signal, self._params_dict(x_new))
            if s_new > best:
                best = s_new
                x = x_new
            else:
                x = x  # stay (SPSA with accept-if-better)
        if self.best_params_ is None:
            self.best_params_ = self._params_dict(x)
        return {
            "transform": self.name,
            "best_params": self.best_params_,
            "best_score": float(self.best_score_),
            "n_queries": self.n_queries_,
            "query_budget": self.query_budget,
            "history": list(self.history_),
        }

    def __call__(self, signal: Signal) -> Signal:
        """Apply the current best parameterization (searching on first use)."""
        if self.best_params_ is None:
            self.search(signal)
        assert self.best_params_ is not None
        if self.transform_factory is None:
            raise ValueError("transform_factory is required for __call__")
        params = {k: self.best_params_.get(k, v[0]) for k, v in self.param_bounds.items()}
        return self.transform_factory(**params)(signal)


class BlackBoxTransformSearch(BaseTransform):
    """Black-box search over a fixed library of candidate transforms.

    Evaluates a query-budgeted set of (transform_key, params) candidates
    using only the objective interface, and keeps the top-``top_k``. On
    ``__call__`` the best candidate found so far is applied — modeling a
    query-limited black-box adversary (Papernot et al. 2016).
    """

    def __init__(
        self,
        candidates: list[tuple[str, dict[str, Any]]] | None = None,
        objective: Objective | None = None,
        query_budget: int = 32,
        top_k: int = 3,
        seed: int | None = None,
        name: str = "adversarial.black_box",
    ) -> None:
        super().__init__(name=name)
        if query_budget < 1:
            raise ValueError("query_budget must be >= 1")
        self.candidates = list(candidates) if candidates else list(DEFAULT_BLACKBOX_CANDIDATES)
        self.objective = objective or _default_objective
        self.query_budget = query_budget
        self.top_k = top_k
        self.seed = seed
        self.results_: list[dict[str, Any]] = []
        self.best_key_: str | None = None

    def _build(self, key: str, params: dict[str, Any]) -> BaseTransform:
        from audiocaptcha_dsp.transforms.registry import build_transform

        return build_transform(key, params)

    def search(self, signal: Signal) -> list[dict[str, Any]]:
        from audiocaptcha_dsp.transforms.registry import get_registry

        reg = get_registry()
        self.results_ = []
        queries = 0
        for key, params in self.candidates:
            if queries >= self.query_budget:
                break
            if key not in reg:
                logger.warning("Black-box search: unknown candidate %s skipped", key)
                continue
            try:
                cand = self._build(key, params)(signal)
                score = float(self.objective(signal, cand))
            except Exception as exc:  # candidate failed: record, don't crash
                logger.warning("Candidate %s failed: %s", key, exc)
                score = -np.inf
            queries += 1
            self.results_.append({"candidate": key, "params": dict(params), "score": score})
        self.results_.sort(key=lambda r: r["score"], reverse=True)
        self.results_ = self.results_[: self.top_k]
        self.best_key_ = self.results_[0]["candidate"] if self.results_ else None
        return list(self.results_)

    def __call__(self, signal: Signal) -> Signal:
        if not self.results_:
            self.search(signal)
        if not self.results_:
            return signal.clone()
        best = self.results_[0]
        return self._build(best["candidate"], best["params"])(signal)


class PhonemeGuidedAllocation(BaseTransform):
    """Representation-guided perturbation allocation (phonetic energy regions).

    Pseudo-phoneme segmentation is derived from the energy envelope +
    zero-crossing profile: high-energy low-ZCR frames are "voiced", low-
    energy/high-ZCR frames are "fricative-like", short gaps are "stop-like".
    Perturbation power budget is allocated to the selected region class —
    consonant-like regions (where ASR front-ends are most brittle,
    Papers 10/21) or vowel regions (where masking power is greatest).
    """

    def __init__(
        self,
        target_region: str = "consonant",
        strength: float = 0.5,
        margin_db: float = 6.0,
        name: str = "adversarial.phoneme_guided",
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name)
        if target_region not in ("consonant", "vowel", "both"):
            raise ValueError("target_region must be 'consonant', 'vowel' or 'both'")
        if strength < 0 or margin_db < 0:
            raise ValueError("strength and margin_db must be >= 0")
        self.target_region = target_region
        self.strength = strength
        self.margin_db = margin_db
        self.seed = seed

    def __call__(self, signal: Signal) -> Signal:
        from audiocaptcha_dsp.psychoacoustics.masking_models import compute_perturbation_budget

        rng = np.random.default_rng(self.seed)
        x = np.asarray(signal.waveform, dtype=np.float64)
        if x.ndim != 1:
            x = x.mean(axis=0)
        sr = signal.sample_rate
        n_fft, hop = 1024, 160  # ~10 ms hop for phonetic granularity
        if len(x) < n_fft:
            return signal.clone()
        win = np.hanning(n_fft)
        n_frames = 1 + (len(x) - n_fft) // hop

        # Frame features
        rms = np.empty(n_frames)
        zcr = np.empty(n_frames)
        for i in range(n_frames):
            seg = x[i * hop: i * hop + n_fft]
            rms[i] = np.sqrt(np.mean(seg ** 2) + 1e-12)
            zcr[i] = np.mean(np.abs(np.diff(np.sign(seg)))) + 1e-6
        rms_med = np.median(rms)
        zcr_med = np.median(zcr)

        voiced = (rms > rms_med) & (zcr < zcr_med)
        fricative = (~voiced) & (zcr > zcr_med)
        stop_like = rms <= 0.3 * rms_med

        if self.target_region == "consonant":
            region = fricative | stop_like
        elif self.target_region == "vowel":
            region = voiced
        else:
            region = np.ones(n_frames, dtype=bool)

        out = np.zeros(len(x) + n_fft)
        wsum = np.zeros_like(out)
        for i in range(n_frames):
            seg = x[i * hop: i * hop + n_fft] * win
            spec = np.fft.rfft(seg)
            if not region[i]:
                y_frame = np.fft.irfft(spec, n=n_fft)
            else:
                # Per-frame budget in frame-length rfft units (matches spec)
                b = np.asarray(
                    compute_perturbation_budget(seg, sr, margin_db=self.margin_db),
                    dtype=np.float64,
                )
                amp = np.sqrt(np.maximum(b, 0.0)) * self.strength
                phase = rng.uniform(-np.pi, np.pi, len(spec))
                y_frame = np.fft.irfft(spec + amp * np.exp(1j * phase), n=n_fft)
            out[i * hop: i * hop + n_fft] += y_frame * win
            wsum[i * hop: i * hop + n_fft] += win ** 2

        wsum = np.maximum(wsum, 1e-3 * float(wsum.max()))
        y = out / wsum
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = y[: len(x)]
        return Signal(
            waveform=y if np.all(np.isfinite(y)) else np.nan_to_num(y),
            sample_rate=sr,
            metadata={**signal.metadata,
                      f"{self.name}.region": self.target_region,
                      f"{self.name}.strength": self.strength,
                      f"{self.name}.n_region_frames": int(np.count_nonzero(region))},
        )


class MultiObjectiveSearch(BaseTransform):
    """NSGA-II-style multi-objective parameter search.

    Objectives (maximized simultaneously):

    1. ``asr_objective`` — ASR degradation (WER) or surrogate (default:
       1 − STOI-proxy).
    2. ``quality_objective`` — perceptual quality proxy (default: STOI-proxy).

    Returns the non-dominated parameter set (Pareto front) plus the single
    best compromise solution under a configurable preference weight vector
    (user-defined preference, not objective truth — WHAT-REMAINS.txt §8).
    """

    def __init__(
        self,
        transform_factory: Callable[..., BaseTransform] | None = None,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        asr_objective: Objective | None = None,
        quality_objective: Objective | None = None,
        population: int = 12,
        generations: int = 8,
        query_budget: int = 128,
        w_asr: float = 0.5,
        seed: int | None = None,
        name: str = "adversarial.multi_objective",
    ) -> None:
        super().__init__(name=name)
        if population < 4:
            raise ValueError("population must be >= 4")
        if generations < 1:
            raise ValueError("generations must be >= 1")
        # Standalone default mirrors GradientFreeParameterSearch (λ-margin).
        self.transform_factory = transform_factory or _default_psychoacoustic_factory
        self.param_bounds = dict(
            param_bounds if param_bounds is not None else {"margin_db": (0.0, 40.0)}
        )
        self.asr_objective = asr_objective or _default_objective
        self.quality_objective = quality_objective or (
            lambda o, c: -_default_objective(o, c)  # keep quality (lower distortion)
        )
        self.population = population
        self.generations = generations
        self.query_budget = query_budget
        self.w_asr = w_asr
        self.seed = seed
        self.pareto_front_: list[dict[str, Any]] = []
        self.n_queries_: int = 0
        self.best_params_: dict[str, float] | None = None

    def _sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.array(
            [rng.uniform(lo, hi) for lo, hi in self.param_bounds.values()],
            dtype=np.float64,
        )

    def _params_dict(self, x: np.ndarray) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self.param_bounds.keys(), x)}

    def _evaluate(self, original: Signal, x: np.ndarray) -> tuple[float, float] | None:
        if self.n_queries_ >= self.query_budget:
            return None
        if self.transform_factory is None:
            raise ValueError("transform_factory is required")
        params = self._params_dict(x)
        try:
            cand = self.transform_factory(**params)(original)
            f1 = float(self.asr_objective(original, cand))
            f2 = float(self.quality_objective(original, cand))
        except Exception as exc:
            logger.warning("Multi-objective candidate failed: %s", exc)
            return None
        self.n_queries_ += 1
        if not (np.isfinite(f1) and np.isfinite(f2)):
            return None
        return f1, f2

    @staticmethod
    def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return a[0] >= b[0] and a[1] >= b[1] and (a[0] > b[0] or a[1] > b[1])

    def search(self, signal: Signal) -> dict[str, Any]:
        if not self.param_bounds:
            raise ValueError("param_bounds must be provided before search()")
        rng = np.random.default_rng(self.seed)
        self.n_queries_ = 0

        # Initialize population
        xs = [self._sample(rng) for _ in range(self.population)]
        fitness: list[tuple[float, float] | None] = [self._evaluate(signal, x) for x in xs]

        for _ in range(self.generations):
            if self.n_queries_ >= self.query_budget:
                break
            # Variation: blend crossover + gaussian mutation
            new_xs = []
            order = rng.permutation(len(xs))
            for i in range(0, len(xs) - 1, 2):
                a, b = xs[order[i]], xs[order[i + 1]]
                gamma = rng.uniform(0.0, 1.0, size=len(a))
                c1 = gamma * a + (1 - gamma) * b
                c2 = gamma * b + (1 - gamma) * a
                spread = np.array([hi - lo for lo, hi in self.param_bounds.values()])
                c1 = c1 + rng.normal(0, 0.1, size=len(c1)) * spread
                c2 = c2 + rng.normal(0, 0.1, size=len(c2)) * spread
                new_xs.extend([c1, c2])
            # Clip to bounds
            clipped = []
            for x in new_xs:
                xc = x.copy()
                for i, (lo, hi) in enumerate(self.param_bounds.values()):
                    xc[i] = np.clip(xc[i], lo, hi)
                clipped.append(xc)
            new_fit = [self._evaluate(signal, x) for x in clipped]

            # Merge and select by non-domination + crowding (simplified)
            all_x = xs + clipped
            all_f = [f for f in fitness + new_fit]
            valid = [(x, f) for x, f in zip(all_x, all_f) if f is not None]
            if not valid:
                break
            fronts = self._fast_non_dominated(valid)
            selected: list[tuple[np.ndarray, tuple[float, float]]] = []
            for front in fronts:
                for item in front:
                    if len(selected) >= self.population:
                        break
                    selected.append(item)
                if len(selected) >= self.population:
                    break
            xs = [s[0] for s in selected]
            fitness = [s[1] for s in selected]

        valid = [(x, f) for x, f in zip(xs, fitness) if f is not None]
        if not valid:
            raise RuntimeError("Multi-objective search: no feasible candidates evaluated")
        front1 = self._fast_non_dominated(valid)[0]
        self.pareto_front_ = [
            {"params": self._params_dict(x), "objectives": {"asr": f[0], "quality": f[1]}}
            for x, f in front1
        ]
        # Weighted compromise (user preference)
        def scalar(item):
            _, f = item
            w = self.w_asr
            # Normalize each objective by its front range
            return w * f[0] + (1 - w) * f[1]

        best = max(valid, key=scalar)
        self.best_params_ = self._params_dict(best[0])
        return {
            "transform": self.name,
            "pareto_front": list(self.pareto_front_),
            "best_params": self.best_params_,
            "best_objectives": {"asr": best[1][0], "quality": best[1][1]},
            "w_asr": self.w_asr,
            "n_queries": self.n_queries_,
        }

    def _fast_non_dominated(
        self, items: list[tuple[np.ndarray, tuple[float, float]]]
    ) -> list[list[tuple[np.ndarray, tuple[float, float]]]]:
        remaining = list(items)
        fronts: list[list[tuple[np.ndarray, tuple[float, float]]]] = []
        while remaining:
            front = []
            for cand in remaining:
                dominated = any(
                    self._dominates(other[1], cand[1])
                    for other in remaining
                    if other is not cand
                )
                if not dominated:
                    front.append(cand)
            if not front:  # all mutually dominated (shouldn't happen)
                front = [remaining[0]]
            fronts.append(front)
            front_set = {id(f[0]) for f in front}
            remaining = [r for r in remaining if id(r[0]) not in front_set]
        return fronts

    def __call__(self, signal: Signal) -> Signal:
        if self.best_params_ is None:
            self.search(signal)
        assert self.best_params_ is not None
        if self.transform_factory is None:
            raise ValueError("transform_factory is required for __call__")
        return self.transform_factory(**self.best_params_)(signal)
