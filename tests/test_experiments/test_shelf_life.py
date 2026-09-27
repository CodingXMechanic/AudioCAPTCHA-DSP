"""Shelf-life forecasting (novelty N11): conditions, fits, bootstrap, report."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from audiocaptcha_dsp.experiments.shelf_life import (
    CAPACITY_LADDER,
    DEFAULT_DOUBLING_MONTHS,
    DEFAULT_MARGINS_DB,
    PARAMS_BY_MODEL,
    BootstrapResult,
    CapacityPoint,
    ScalingFit,
    build_shelf_conditions,
    compute_capacity_points,
    evaluate_shelf_life,
    fit_power_law,
    paired_bootstrap_scaling,
    shelf_life_months,
    write_shelf_report,
)

HSR_LBL_FRAG = "illustrative"


# --------------------------------------------------------------------------
# Synthetic row fixtures
# --------------------------------------------------------------------------
def _row(cid: str, utt: str, model: str, wer: float, *,
         kind: str = "sweep", stoi: float = 0.9, margin: str = "",
         status: str = "ok", error: str = "") -> dict:
    return {
        "condition_id": cid, "kind": kind, "transform_key": "psychoacoustic.masked_noise",
        "family": "F", "params_json": "{}", "margin_db": margin,
        "utt_id": utt, "spk": "spk1", "dur_s": "2.0",
        "status": status, "error": error, "ref_text": "hello world",
        "stoi_proxy": f"{stoi:g}", "snr_db": "10",
        "model_key": model,
        "params_m": f"{PARAMS_BY_MODEL.get(model, 0):g}",
        "wer": f"{wer:g}" if wer == wer else "", "hyp": "",
    }


def _synthetic_rows(
    a: float = 40.0, b: float = 1.2, n_pairs: int = 40, noise: float = 0.0,
    seed: int = 7,
) -> list[dict]:
    """WER decays with capacity: WER = a * C^-b (+ noise), original lower."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    models = [m["model_key"] for m in CAPACITY_LADDER]
    for p in range(n_pairs):
        utt = f"u{p:03d}"
        for mk in models:
            c = PARAMS_BY_MODEL[mk]
            w_att = a * c ** (-b) + noise * rng.normal()
            w_att = float(np.clip(w_att, 0.01, 3.0))
            w_orig = max(0.02, 0.07 * c ** (-0.3) + noise * rng.normal() / 4)
            rows.append(_row("original", utt, mk, w_orig, kind="original"))
            rows.append(_row("psychoacoustic.masked_noise#margin=0", utt, mk,
                             w_att, margin="0"))
            rows.append(_row("control.no_hearing_threshold", utt, mk,
                             min(1.5, w_att * 1.4 + 0.05), kind="control",
                             margin=""))
    return rows


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------
class TestConditions:
    def test_policy_conditions_shape(self) -> None:
        conds = build_shelf_conditions((0.0, 10.0, 20.0, 40.0), seed=42)
        ids = [c.condition_id for c in conds]
        assert ids[0] == "original"
        assert ids[-1] == "control.no_hearing_threshold"
        margins = {c.margin_db for c in conds if c.kind == "sweep"}
        assert margins == {0.0, 10.0, 20.0, 40.0}
        assert all(c.params.get("seed", 42) == 42 for c in conds
                   if c.kind == "sweep")

    def test_duplicate_margins_collapse(self) -> None:
        conds = build_shelf_conditions((0.0, 0.0, 20.0))
        sweep = [c for c in conds if c.kind == "sweep"]
        assert len(sweep) == 2

    def test_unknown_transform_raises(self) -> None:
        with pytest.raises(ValueError):
            build_shelf_conditions((0.0,), transform_key="nope.nope")


# --------------------------------------------------------------------------
# Scaling fit
# --------------------------------------------------------------------------
class TestFit:
    def test_recovers_exact_power_law(self) -> None:
        a_true, b_true = 40.0, 1.25
        x = np.array([m["params_m"] for m in CAPACITY_LADDER])
        y = a_true * x ** (-b_true)
        fit = fit_power_law(x, y)
        assert fit.b == pytest.approx(b_true, rel=1e-6)
        assert fit.a == pytest.approx(a_true, rel=1e-6)
        assert fit.r2 == pytest.approx(1.0, abs=1e-9)
        assert fit.n_points == 3

    def test_needs_two_distinct_points(self) -> None:
        with pytest.raises(ValueError):
            fit_power_law(np.array([39.0]), np.array([0.5]))
        with pytest.raises(ValueError):
            fit_power_law(np.array([39.0, 39.0]), np.array([0.5, 0.6]))

    def test_break_capacity_closed_form(self) -> None:
        a, b = 40.0, 1.2
        fit = ScalingFit(a=a, b=b, r2=1.0, n_points=3)
        c_star = fit.break_capacity(0.3)
        assert c_star is not None
        assert float(fit.predict(c_star)) == pytest.approx(0.3, rel=1e-9)

    def test_no_break_when_exponent_nonpositive(self) -> None:
        assert ScalingFit(a=1.0, b=0.0, r2=0.0, n_points=3).break_capacity(0.3) is None
        assert ScalingFit(a=1.0, b=-0.5, r2=0.0, n_points=3).break_capacity(0.3) is None


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
class TestBootstrap:
    def _vectors(self, noise: float = 0.02, n: int = 60, seed: int = 3):
        rng = np.random.default_rng(seed)
        x = np.array([m["params_m"] for m in CAPACITY_LADDER])
        vecs = {}
        for mk in PARAMS_BY_MODEL:
            c = PARAMS_BY_MODEL[mk]
            vecs[mk] = np.clip(40.0 * c ** (-1.2)
                               + noise * rng.standard_normal(n), 0.01, 3)
        return x, vecs

    def test_ci_brackets_point_estimate(self) -> None:
        x, vecs = self._vectors()
        boot = paired_bootstrap_scaling(x, vecs, n_boot=300, seed=1)
        assert isinstance(boot, BootstrapResult)
        lo, hi = boot.b_ci
        assert lo <= boot.fit.b <= hi
        assert boot.n_break_valid > 0.8 * 300
        assert boot.break_ci is not None
        assert boot.break_ci[0] <= boot.break_ci[1]

    def test_deterministic_for_seed(self) -> None:
        x, vecs = self._vectors()
        b1 = paired_bootstrap_scaling(x, vecs, n_boot=100, seed=9)
        b2 = paired_bootstrap_scaling(x, vecs, n_boot=100, seed=9)
        assert b1.b_ci == b2.b_ci
        assert np.allclose(b1.grid_lo, b2.grid_lo, equal_nan=True)

    def test_rejects_unaligned_vectors(self) -> None:
        x = np.array([39.0, 74.0, 244.0])
        with pytest.raises(ValueError):
            paired_bootstrap_scaling(x, {
                "whisper_tiny": np.ones(5),
                "whisper_base": np.ones(4),
            })

    def test_grid_band_is_monotone_in_capacity(self) -> None:
        # WER must decrease with capacity under a positive-exponent fit
        x, vecs = self._vectors(noise=0.0)
        boot = paired_bootstrap_scaling(x, vecs, n_boot=50, seed=2)
        mid = np.nanmedian(np.vstack([boot.grid_lo, boot.grid_hi]), axis=0)
        assert np.all(np.diff(mid) <= 1e-9)


# --------------------------------------------------------------------------
# Capacity points
# --------------------------------------------------------------------------
class TestCapacityPoints:
    def test_points_from_synthetic_rows(self) -> None:
        rows = _synthetic_rows(a=40.0, b=1.2)
        pts = compute_capacity_points(rows, wer_threshold=0.3)
        assert len(pts) == 3
        assert [p.params_m for p in pts] == [39.0, 74.0, 244.0]
        # attacked WER decays with capacity
        assert pts[0].attacked_wer > pts[1].attacked_wer > pts[2].attacked_wer
        # human proxy identical across models (model-independent axis)
        assert len({round(p.hsr, 9) for p in pts}) == 1
        for p in pts:
            assert p.gap == pytest.approx(p.hsr - p.asr_sr, abs=1e-9)
            assert 0.0 <= p.asr_sr <= 1.0

    def test_error_rows_excluded(self) -> None:
        rows = _synthetic_rows(n_pairs=4) + [
            _row("psychoacoustic.masked_noise#margin=0", "u000",
                 "whisper_tiny", float("nan"), status="error", error="boom")
        ]
        pts = compute_capacity_points(rows)
        assert all(p.n > 0 for p in pts)


# --------------------------------------------------------------------------
# Forecast + report
# --------------------------------------------------------------------------
class TestForecast:
    def test_shelf_life_months_monotone(self) -> None:
        ref = 244.0
        fast = shelf_life_months(10_000.0, ref, doubling_months=6.0)
        slow = shelf_life_months(10_000.0, ref, doubling_months=24.0)
        assert fast < slow
        # already-broken policy -> zero months
        assert shelf_life_months(100.0, ref, 12.0) == 0.0
        assert shelf_life_months(None, ref, 12.0) is None
        assert not math.isfinite(shelf_life_months(float("nan"), ref, 12.0)
                                 or float("nan"))

    def test_evaluate_and_report_end_to_end(self, tmp_path) -> None:
        rows = _synthetic_rows(a=60.0, b=1.1, n_pairs=30, noise=0.01)
        summary = evaluate_shelf_life(
            rows, w_break=0.3, doubling_months=DEFAULT_DOUBLING_MONTHS,
            n_boot=200, seed=11,
        )
        # JSON-serializable
        blob = json.dumps(summary, default=str)
        assert "illustrative" in blob
        assert len(summary["capacity_points"]) == 3
        assert set(summary["fits"]) == {
            "psychoacoustic.masked_noise#margin=0",
            "control.no_hearing_threshold",
        }
        fit = summary["fits"]["psychoacoustic.masked_noise#margin=0"]
        assert fit["fit"]["b"] > 0
        assert fit["break_capacity_m_params"] is not None
        assert set(fit["forecast_months"]) == {"6", "12", "24"}
        assert summary["hsr_label"] and HSR_LBL_FRAG in summary["hsr_label"]

        report = write_shelf_report(summary, tmp_path)
        text = report.read_text(encoding="utf-8")
        assert HSR_LBL_FRAG in text
        assert "exponent b" in text
        assert (tmp_path / "shelf_life.json").exists()
        # markdown table has a separator row matching its header columns
        lines = text.splitlines()
        header = next(l for l in lines if l.startswith("| Condition |"))
        sep = lines[lines.index(header) + 1]
        assert sep.startswith("|---")
        assert header.count("|") == sep.count("|")

    def test_control_without_decay_reports_no_break(self) -> None:
        # control: high WER regardless of capacity -> b ~ 0 -> no finite break
        rng = np.random.default_rng(5)
        rows: list[dict] = []
        for p in range(30):
            utt = f"u{p:03d}"
            for mk in PARAMS_BY_MODEL:
                rows.append(_row("original", utt, mk, 0.08, kind="original"))
                rows.append(_row("control.no_hearing_threshold", utt, mk,
                                 1.2 + 0.02 * rng.normal(), kind="control"))
                rows.append(_row("psychoacoustic.masked_noise#margin=0", utt, mk,
                                 0.4, margin="0"))
        summary = evaluate_shelf_life(rows, n_boot=100, seed=4)
        ctl = summary["fits"]["control.no_hearing_threshold"]
        # b≈0 → either no break or a break CI spanning far beyond ladder;
        # the important part is the code path stays finite and honest
        if ctl["break_capacity_m_params"] is None:
            assert ctl["forecast_months"]["12"] is None
        else:
            assert ctl["fit"]["b"] < 0.5


# --------------------------------------------------------------------------
# Optional integration smoke (skipped without weights + corpus; CI-safe)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not all(
        (__import__("pathlib").Path.home() / ".cache" / "whisper"
         / f"{m['size']}.pt").exists()
        for m in CAPACITY_LADDER
    ),
    reason="Whisper ladder weights not cached (CI must not download models)",
)
def test_shelf_life_runner_smoke(tmp_path) -> None:
    from pathlib import Path

    data_root = Path("data/raw/LibriSpeech/test-clean")
    if not data_root.exists():
        pytest.skip("LibriSpeech test-clean not present")
    from audiocaptcha_dsp.experiments.shelf_life import run_shelf_life

    rows_path = run_shelf_life(
        out_dir=tmp_path,
        dataset="librispeech",
        data_root=data_root,
        subset="A",
        seed=42,
        max_utterances=2,
        margins_db=(0.0,),
        workers=1,
        log=lambda m: None,
    )
    assert rows_path.exists()
    from audiocaptcha_dsp.experiments.shelf_life import read_shelf_rows

    rows = read_shelf_rows(rows_path)
    # (orig + masked λ0 + control) x 2 utts x 3 models
    assert len(rows) == 3 * 2 * 3
    assert {r["model_key"] for r in rows} == set(PARAMS_BY_MODEL)
    assert (tmp_path / "dataset_manifest.json").exists()
    assert (tmp_path / "run_manifest.json").exists()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text("utf-8"))
    assert "illustrative" in manifest["hsr_label"]

    # resume: nothing to do
    again = run_shelf_life(
        out_dir=tmp_path, dataset="librispeech", data_root=data_root,
        subset="A", seed=42, max_utterances=2, margins_db=(0.0,),
        workers=1, log=lambda m: None,
    )
    assert len(read_shelf_rows(again)) == len(rows)
