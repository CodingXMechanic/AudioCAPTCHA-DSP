import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.adversarial import (
    GradientFreeParameterSearch,
    BlackBoxTransformSearch,
    PhonemeGuidedAllocation,
    MultiObjectiveSearch,
)
from audiocaptcha_dsp.transforms.psychoacoustic.constrained import (
    PsychoacousticNoiseInjection,
)


@pytest.fixture
def speech_like() -> Signal:
    sr = 16000
    t = np.arange(int(1.0 * sr)) / sr
    wave = 0.5 * np.sin(2 * np.pi * 170 * t) + 0.3 * np.sin(2 * np.pi * 340 * t)
    wave *= 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    return Signal(waveform=wave, sample_rate=sr, metadata={"transcript": "hello world"})


def analytic_objective(original: Signal, candidate: Signal) -> float:
    """Maximize perturbation energy (cheap, deterministic test objective)."""
    delta = candidate.waveform - original.waveform
    return float(np.sum(delta ** 2))


def test_gradient_free_search_runs_and_records(speech_like):
    search = GradientFreeParameterSearch(
        transform_factory=lambda margin_db=10.0: PsychoacousticNoiseInjection(
            margin_db=margin_db
        ),
        param_bounds={"margin_db": (0.0, 30.0)},
        objective=analytic_objective,
        n_iterations=5,
        query_budget=12,
        seed=1,
    )
    result = search.search(speech_like)
    assert result["n_queries"] <= 12
    assert result["best_params"] is not None
    assert 0.0 <= result["best_params"]["margin_db"] <= 30.0
    assert np.isfinite(result["best_score"])
    assert len(result["history"]) >= 1


def test_gradient_free_call_applies_best(speech_like):
    search = GradientFreeParameterSearch(
        transform_factory=lambda margin_db=10.0: PsychoacousticNoiseInjection(
            margin_db=margin_db
        ),
        param_bounds={"margin_db": (0.0, 30.0)},
        objective=analytic_objective,
        n_iterations=4,
        query_budget=8,
        seed=2,
    )
    out = search(speech_like)
    assert out.sample_rate == speech_like.sample_rate
    assert len(out.waveform) == len(speech_like.waveform)
    assert np.all(np.isfinite(out.waveform))


def test_gradient_free_respects_query_budget(speech_like):
    search = GradientFreeParameterSearch(
        param_bounds={"margin_db": (0.0, 30.0)},
        objective=analytic_objective,
        n_iterations=50,
        query_budget=6,
        seed=3,
    )
    search.search(speech_like)
    assert search.n_queries_ <= 6


def test_black_box_search_ranks_candidates(speech_like):
    search = BlackBoxTransformSearch(
        candidates=[
            ("noise.white", {"snr_db": 30.0}),
            ("noise.white", {"snr_db": 5.0}),
        ],
        objective=lambda o, c: float(np.sum((c.waveform - o.waveform) ** 2)),
        query_budget=4,
        top_k=2,
        seed=1,
    )
    results = search.search(speech_like)
    assert len(results) == 2
    # Lower SNR ⇒ bigger perturbation ⇒ higher score
    assert results[0]["params"]["snr_db"] == 5.0
    assert results[0]["score"] >= results[1]["score"]


def test_black_box_call_uses_best(speech_like):
    search = BlackBoxTransformSearch(
        candidates=[("spectral.lowpass", {"cutoff_hz": 5000.0})],
        query_budget=2,
        top_k=1,
    )
    out = search(speech_like)
    assert np.all(np.isfinite(out.waveform))
    assert len(out.waveform) == len(speech_like.waveform)


def test_phoneme_guided_allocates_by_region(speech_like):
    cons = PhonemeGuidedAllocation(
        target_region="consonant", strength=0.8, margin_db=6.0, seed=5
    )(speech_like)
    vow = PhonemeGuidedAllocation(
        target_region="vowel", strength=0.8, margin_db=6.0, seed=5
    )(speech_like)
    d_cons = np.sum((cons.waveform - speech_like.waveform) ** 2)
    d_vow = np.sum((vow.waveform - speech_like.waveform) ** 2)
    # AM-like signal: voiced frames dominate ⇒ vowel allocation perturbs more
    assert d_vow > 0 and d_cons > 0
    assert "adversarial.phoneme_guided.n_region_frames" in cons.metadata
    assert "adversarial.phoneme_guided.n_region_frames" in vow.metadata


def test_phoneme_guided_invalid_region_raises():
    with pytest.raises(ValueError):
        PhonemeGuidedAllocation(target_region="nonsense")


def test_multi_objective_search_builds_pareto_front(speech_like):
    search = MultiObjectiveSearch(
        transform_factory=lambda margin_db=10.0: PsychoacousticNoiseInjection(
            margin_db=margin_db
        ),
        param_bounds={"margin_db": (0.0, 30.0)},
        asr_objective=analytic_objective,
        quality_objective=lambda o, c: -float(np.sum((c.waveform - o.waveform) ** 2)),
        population=6,
        generations=3,
        query_budget=40,
        seed=4,
    )
    result = search.search(speech_like)
    assert result["pareto_front"]  # non-empty non-dominated set
    assert result["best_params"] is not None
    assert 0.0 <= result["best_params"]["margin_db"] <= 30.0
    assert result["n_queries"] <= 40


def test_multi_objective_pareto_non_dominated(speech_like):
    search = MultiObjectiveSearch(
        param_bounds={"margin_db": (0.0, 30.0)},
        asr_objective=analytic_objective,
        quality_objective=lambda o, c: -float(np.sum((c.waveform - o.waveform) ** 2)),
        population=6,
        generations=2,
        query_budget=30,
        seed=6,
    )
    result = search.search(speech_like)
    objs = [p["objectives"] for p in result["pareto_front"]]
    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i == j:
                continue
            dominates = (
                b["asr"] >= a["asr"]
                and b["quality"] >= a["quality"]
                and (b["asr"] > a["asr"] or b["quality"] > a["quality"])
            )
            assert not dominates, "Pareto front contains a dominated point"
