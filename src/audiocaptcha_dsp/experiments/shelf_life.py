"""
CAPTCHA Shelf-Life Forecasting (novel contribution N11)
========================================================
Original to this work — not adopted from the 23-paper survey or the base
paper.

Idea
----
A psychoacoustic CAPTCHA policy's security is a *function of the attacker's
ASR capacity*.  We measure the human–ASR gap as recognizers grow along a
controlled capacity ladder (Whisper tiny → base → small; one architecture,
three sizes), fit a neural-scaling-style law to the attacked-ASR WER,

    WER(C) = a * C ** (-b)        (C = model parameters, log-log linear),

and forecast the *break capacity* C* at which expected WER on attacked audio
falls to a security threshold — then translate C* into months-to-break under
explicit capacity-doubling scenarios.  The output is a "shelf life" for the
evaluated psychoacoustic policy: when it is predicted to stop separating
humans from machines as models scale.

Honesty rules (this file's contract)
------------------------------------
* The human axis is the STOI-derived illustrative proxy — never a measured
  HSR (``HSR_LABEL`` is written into every report).
* Fits use only |ladder| = 3 capacity points (df = 1); uncertainty comes from
  a paired bootstrap over evaluation pairs and extrapolations beyond the
  observed range are flagged, not sold as predictions.
* Doubling scenarios are *assumptions*, stated as scenarios, never as
  trends we measured.
* Only Whisper family sizes enter the fit (architecture controlled); other
  engines may be reported as out-of-family markers but never mixed into the
  scaling axis.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.experiments.comparative import (
    Condition,
    ORIGINAL_ID,
    apply_no_threshold_control,
    select_samples,
    write_dataset_manifest,
    _git_commit,
    _package_versions,
    _tasks_for,
)
from audiocaptcha_dsp.transforms.registry import get_registry
from audiocaptcha_dsp.asr import normalize_transcript
from audiocaptcha_dsp.evaluation.metrics import compute_metrics, compute_wer

# --------------------------------------------------------------------------
# Capacity ladder (one architecture, three sizes)
# --------------------------------------------------------------------------
CAPACITY_LADDER: tuple[dict[str, Any], ...] = (
    {"model_key": "whisper_tiny", "size": "tiny", "params_m": 39.0},
    {"model_key": "whisper_base", "size": "base", "params_m": 74.0},
    {"model_key": "whisper_small", "size": "small", "params_m": 244.0},
)
PARAMS_BY_MODEL: dict[str, float] = {
    m["model_key"]: float(m["params_m"]) for m in CAPACITY_LADDER
}

#: Default margins (λ, dB above the hearing threshold) of the tested policy.
DEFAULT_MARGINS_DB: tuple[float, ...] = (0.0, 10.0, 20.0, 40.0)

#: Security threshold: expected WER on attacked audio at/below which the
#: policy is considered broken (same default as SecurityEvaluator.asr_sr).
DEFAULT_W_BREAK = 0.3

#: Capacity-doubling scenarios (months per doubling) for shelf-life translation.
DEFAULT_DOUBLING_MONTHS: tuple[float, ...] = (6.0, 12.0, 24.0)

CONTROL_ID = "control.no_hearing_threshold"

SHELF_ROW_FIELDS = [
    "condition_id", "kind", "transform_key", "family", "params_json",
    "margin_db", "utt_id", "spk", "dur_s", "status", "error",
    "ref_text", "stoi_proxy", "snr_db",
    "model_key", "params_m", "wer", "hyp",
]


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------
def build_shelf_conditions(
    margins_db: tuple[float, ...] | list[float] = DEFAULT_MARGINS_DB,
    seed: int = 42,
    transform_key: str = "psychoacoustic.masked_noise",
) -> list[Condition]:
    """Original + λ-grid for one psychoacoustic policy + matched-power control."""
    from audiocaptcha_dsp.experiments.comparative import _with_seed

    reg = get_registry()
    if transform_key not in reg:
        raise ValueError(f"unknown transform key: {transform_key!r}")
    spec = reg[transform_key]
    conds: list[Condition] = [
        Condition(ORIGINAL_ID, "original", "baseline.identity", "-"),
    ]
    seen: set[tuple[str, str]] = {(ORIGINAL_ID, "")}
    for m in sorted(set(float(m) for m in margins_db)):
        params = dict(spec.params)
        params["margin_db"] = m
        params = _with_seed(transform_key, params, seed)
        sig = json.dumps(sorted(params.items()), default=str)
        key = (transform_key, sig)
        if key in seen:
            continue
        seen.add(key)
        conds.append(Condition(
            f"{transform_key}#margin={m:g}", "sweep", transform_key,
            spec.family, params, margin_db=m,
        ))
    conds.append(Condition(
        CONTROL_ID, "control", "psychoacoustic.masked_noise", "F",
        {"margin_db": 0.0, "seed": seed, "power_match": "lambda0"},
    ))
    return conds


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
_G: dict[str, Any] = {}


def _shelf_worker_init(model_keys: list[str], torch_threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, torch_threads)))
    if torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(torch_threads)
        except Exception:
            pass
    from audiocaptcha_dsp.asr import WhisperAdapter

    sizes = {m["model_key"]: m["size"] for m in CAPACITY_LADDER}
    engines = {}
    for key in model_keys:
        engines[key] = WhisperAdapter(
            model_size=sizes[key], offline_fallback=False
        )
    for e in engines.values():
        e.load()
    _G["engines"] = engines


def _shelf_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    import soundfile as sf

    cond = task["condition"]
    base: dict[str, Any] = {
        "condition_id": cond["condition_id"],
        "kind": cond["kind"],
        "transform_key": cond["transform_key"],
        "family": cond["family"],
        "params_json": json.dumps(cond["params"], sort_keys=True, default=str),
        "margin_db": "" if cond["margin_db"] is None else cond["margin_db"],
        "utt_id": task["utt_id"],
        "spk": task["spk"],
        "dur_s": task["dur_s"],
        "status": "ok",
        "error": "",
        "ref_text": task["transcript"],
    }
    rows: list[dict[str, Any]] = []
    try:
        wav, sr = sf.read(task["path"], dtype="float64")
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=-1)
        original = Signal(waveform=np.asarray(wav), sample_rate=int(sr))

        if cond["kind"] == "original":
            processed = original
        elif cond["kind"] == "control":
            processed = apply_no_threshold_control(
                original, seed=int(cond["params"].get("seed", 42))
            )
        else:
            from audiocaptcha_dsp.experiments.runner import resolve_transform

            processed = resolve_transform(
                cond["transform_key"], dict(cond["params"])
            )(original)

        metrics = compute_metrics(original, processed)
        for key, col in (("stoi_proxy", "stoi_proxy"), ("snr_db", "snr_db")):
            v = metrics.get(key)
            base[col] = "" if v is None or not np.isfinite(v) else f"{float(v):.6g}"

        ref = normalize_transcript(task["transcript"])
        wanted = set(task.get("model_keys") or list(_G.get("engines", {})))
        for name, eng in _G.get("engines", {}).items():
            if name not in wanted:
                continue
            hyp = eng.transcribe(processed).text
            row = dict(base)
            row["model_key"] = name
            row["params_m"] = f"{PARAMS_BY_MODEL[name]:g}"
            row["hyp"] = hyp
            row["wer"] = f"{compute_wer(ref, normalize_transcript(hyp)):.6f}"
            rows.append(row)
    except Exception as e:  # noqa: BLE001 - one failure must not kill the run
        row = dict(base)
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        row.setdefault("model_key", "")
        row.setdefault("params_m", "")
        row.setdefault("wer", "")
        row.setdefault("hyp", "")
        rows.append(row)
    return rows


def _load_shelf_done(rows_path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if rows_path.exists():
        with rows_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["condition_id"], r["utt_id"], r.get("model_key", "")))
    return done


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_shelf_life(
    out_dir: Path,
    dataset: str = "librispeech",
    data_root: Path = Path("data/raw/LibriSpeech/test-clean"),
    subset: str = "A",
    seed: int = 42,
    max_utterances: int | None = None,
    margins_db: tuple[float, ...] = DEFAULT_MARGINS_DB,
    model_keys: tuple[str, ...] = tuple(m["model_key"] for m in CAPACITY_LADDER),
    workers: int = 2,
    log: Callable[[str], None] = print,
) -> Path:
    """Run (or resume) the capacity-ladder pass. Returns rows.csv path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_shelf_conditions(margins_db, seed=seed)
    log(f"[conditions] {len(conditions)} "
        f"(margins={list(margins_db)}, models={list(model_keys)})")

    samples, info = select_samples(
        dataset, Path(data_root), subset=subset, seed=seed,
        max_utterances=max_utterances,
    )
    write_dataset_manifest(out_dir, samples, info)
    log(f"[dataset] {info['protocol']} | {len(samples)} samples | "
        f"corpus={info['corpus']}")

    # run manifest (novelty-specific, same schema spirit as the benchmark)
    import platform

    manifest = {
        "script": "run_shelf_life",
        "novelty": "N11 shelf-life forecasting (gap-vs-capacity scaling)",
        "created": datetime.now(timezone.utc).isoformat(),
        "args": {
            "dataset": dataset, "data_root": str(data_root),
            "subset": subset, "seed": seed,
            "max_utterances": max_utterances,
            "margins_db": list(margins_db), "models": list(model_keys),
            "workers": workers,
        },
        "capacity_ladder": [dict(m) for m in CAPACITY_LADDER],
        "w_break": DEFAULT_W_BREAK,
        "doubling_scenarios_months": list(DEFAULT_DOUBLING_MONTHS),
        "hsr_label": _hsr_label(),
        "base_paper": "Schönherr et al. 2018, arXiv:1808.05665",
        "git_commit": _git_commit(),
        "python": sys_version(),
        "platform": platform.platform(),
        "versions": _package_versions(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    rows_path = out_dir / "rows.csv"
    done = _load_shelf_done(rows_path)
    all_tasks = _tasks_for(conditions, samples)
    tasks: list[dict[str, Any]] = []
    for t in all_tasks:
        for mk in model_keys:
            if (t["condition"]["condition_id"], t["utt_id"], mk) not in done:
                tasks.append({**t, "model_key": mk})
    log(f"[tasks] {len(all_tasks) * len(model_keys)} total, "
        f"{len(done)} already done, {len(tasks)} to run")
    if not tasks:
        log("[done] nothing to run")
        return rows_path

    t0 = time.time()
    n_done = 0

    def _write(rows: list[dict[str, Any]]) -> None:
        nonlocal n_done
        with rows_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=SHELF_ROW_FIELDS,
                               extrasaction="ignore")
            if rows_path.stat().st_size == 0:
                w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in SHELF_ROW_FIELDS})
        n_done += 1
        if n_done % 25 == 0 or n_done == len(tasks):
            el = time.time() - t0
            rate = n_done / el if el > 0 else 0
            eta = (len(tasks) - n_done) / rate if rate > 0 else 0
            log(f"[progress] {n_done}/{len(tasks)} "
                f"({100 * n_done / len(tasks):.1f}%) | {rate:.2f} task/s | "
                f"elapsed {el / 60:.1f} min | ETA {eta / 60:.1f} min")

    # one task per (condition, utt) transcription once per model inside the
    # worker, so re-submit tasks grouped by (condition, utt):
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in tasks:
        grouped.setdefault(
            (t["condition"]["condition_id"], t["utt_id"]), []
        ).append(t["model_key"])
    exec_tasks = []
    for t in tasks:
        mk = grouped.pop((t["condition"]["condition_id"], t["utt_id"]), None)
        if mk is not None:
            exec_tasks.append({**t, "model_keys": mk})

    if workers <= 1:
        _shelf_worker_init(list(model_keys), torch_threads=0)
        for t in exec_tasks:
            _write(_shelf_task(t))
    else:
        torch_threads = max(1, (os.cpu_count() or 4) // workers)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_shelf_worker_init,
            initargs=(list(model_keys), torch_threads),
        ) as pool:
            futs = [pool.submit(_shelf_task, t) for t in exec_tasks]
            for fut in as_completed(futs):
                _write(fut.result())

    log(f"[run] finished in {(time.time() - t0) / 60:.1f} min")
    return rows_path


def sys_version() -> str:
    import sys

    return sys.version


def _hsr_label() -> str:
    from audiocaptcha_dsp.experiments.comparative import HSR_LABEL

    return HSR_LABEL


# --------------------------------------------------------------------------
# Analysis: capacity points
# --------------------------------------------------------------------------
@dataclass
class CapacityPoint:
    model_key: str
    params_m: float
    n: int
    baseline_wer: float          # mean WER on original audio
    attacked_wer: float          # mean WER over attacked (condition, utt) pairs
    delta_wer: float             # attacked - baseline
    asr_sr: float                # share of attacked pairs with wer <= threshold
    hsr: float                   # illustrative human proxy (model-independent)
    gap: float                   # hsr - asr_sr (per WHAT-REMAINS §6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key, "params_m": self.params_m,
            "n": self.n, "baseline_wer": self.baseline_wer,
            "attacked_wer": self.attacked_wer, "delta_wer": self.delta_wer,
            "asr_sr": self.asr_sr, "hsr": self.hsr, "gap": self.gap,
        }


def read_shelf_rows(rows_path: Path) -> list[dict[str, Any]]:
    with Path(rows_path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v


def compute_capacity_points(
    rows: list[dict[str, Any]],
    wer_threshold: float = DEFAULT_W_BREAK,
) -> list[CapacityPoint]:
    """Observed (human-proxy, ASR) points along the capacity ladder."""
    ok = [r for r in rows if r.get("status") == "ok"]
    orig = [r for r in ok if r["condition_id"] == ORIGINAL_ID]
    attacked = [r for r in ok if r["condition_id"] != ORIGINAL_ID]

    # human proxy: dedupe (condition, utt) — identical across model rows
    stoi_by_pair: dict[tuple[str, str], float] = {}
    for r in attacked:
        v = _f(r.get("stoi_proxy"))
        if np.isfinite(v):
            stoi_by_pair.setdefault((r["condition_id"], r["utt_id"]), v)
    hsr_vals = [float(np.clip(0.6 + 0.4 * v, 0.0, 1.0))
                for v in stoi_by_pair.values()]
    hsr = float(np.mean(hsr_vals)) if hsr_vals else float("nan")

    points: list[CapacityPoint] = []
    for m in CAPACITY_LADDER:
        mk = m["model_key"]
        o = [_f(r["wer"]) for r in orig if r["model_key"] == mk]
        a = [_f(r["wer"]) for r in attacked if r["model_key"] == mk]
        o = [v for v in o if np.isfinite(v)]
        a = [v for v in a if np.isfinite(v)]
        if not a:
            continue
        baseline = float(np.mean(o)) if o else float("nan")
        attacked_mean = float(np.mean(a))
        asr_sr = float(np.mean([v <= wer_threshold for v in a]))
        points.append(CapacityPoint(
            model_key=mk, params_m=float(m["params_m"]), n=len(a),
            baseline_wer=baseline, attacked_wer=attacked_mean,
            delta_wer=attacked_mean - baseline,
            asr_sr=asr_sr, hsr=hsr, gap=hsr - asr_sr,
        ))
    return points


# --------------------------------------------------------------------------
# Analysis: power-law scaling fit + paired bootstrap
# --------------------------------------------------------------------------
@dataclass
class ScalingFit:
    a: float                     # coefficient, WER = a * C^-b
    b: float                     # scaling exponent (positive = WER decays)
    r2: float
    n_points: int

    def predict(self, params_m: float | np.ndarray) -> float | np.ndarray:
        return self.a * np.asarray(params_m, dtype=float) ** (-self.b)

    #: Beyond this (a trillion-trillion parameters) a "break" is meaningless;
    #: :meth:`break_capacity` reports None instead of an astronomical value.
    MAX_BREAK_PARAMS_M = 1e12

    def break_capacity(self, w_break: float) -> float | None:
        """Capacity at which predicted attacked WER falls to ``w_break``.

        Returns ``None`` when there is no finite, meaningful break: non-positive
        exponent, or a break capacity beyond ``MAX_BREAK_PARAMS_M``.
        """
        if not (np.isfinite(self.a) and np.isfinite(self.b)) or self.b <= 0:
            return None
        if self.a <= 0 or w_break <= 0:
            return None
        try:
            log_c = (math.log(self.a) - math.log(w_break)) / self.b
        except (ValueError, ZeroDivisionError):
            return None
        if not math.isfinite(log_c) or log_c > math.log(self.MAX_BREAK_PARAMS_M):
            return None
        c = math.exp(log_c)
        return float(c) if c > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "r2": self.r2,
                "n_points": self.n_points}


def fit_power_law(params_m: np.ndarray, wer: np.ndarray) -> ScalingFit:
    """log-log ordinary least squares: log WER = log a - b log C."""
    x = np.asarray(params_m, dtype=float)
    y = np.asarray(wer, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.unique(x).size < 2:
        raise ValueError("need >=2 distinct positive capacity points")
    lx, ly = np.log(x), np.log(y)
    b_neg, log_a = np.polyfit(lx, ly, 1)   # ly = b_neg * lx + log_a
    b = -b_neg
    a = float(np.exp(log_a))
    pred = log_a + b_neg * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return ScalingFit(a=a, b=float(b), r2=float(r2), n_points=int(x.size))


@dataclass
class BootstrapResult:
    fit: ScalingFit
    b_ci: tuple[float, float]
    a_ci: tuple[float, float]
    break_ci: tuple[float, float] | None     # percentile CI over C* | b>0
    n_boot: int
    n_break_valid: int
    grid: np.ndarray = field(default_factory=lambda: np.array([]))
    grid_lo: np.ndarray = field(default_factory=lambda: np.array([]))
    grid_hi: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit": self.fit.to_dict(),
            "b_ci": list(self.b_ci), "a_ci": list(self.a_ci),
            "break_ci": list(self.break_ci) if self.break_ci else None,
            "n_boot": self.n_boot, "n_break_valid": self.n_break_valid,
        }


def paired_bootstrap_scaling(
    params_m: np.ndarray,
    per_model_wer: dict[str, np.ndarray],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    w_break: float = DEFAULT_W_BREAK,
    grid: np.ndarray | None = None,
) -> BootstrapResult:
    """Paired bootstrap: resample evaluation pairs (shared across models).

    ``per_model_wer`` maps model key → vector of per-pair WERs, **aligned**
    (same pair order) across models.
    """
    x = np.asarray(params_m, dtype=float)
    models = sorted(per_model_wer, key=lambda k: PARAMS_BY_MODEL.get(k, 0.0))
    mats = [np.asarray(per_model_wer[m], dtype=float) for m in models]
    lengths = {m.size for m in mats}
    if len(lengths) != 1:
        raise ValueError("per-model WER vectors must be aligned")
    stack = np.vstack(mats)                      # (M, P)
    n_pairs = stack.shape[1]
    if n_pairs < 2:
        raise ValueError("need >= 2 evaluation pairs for bootstrap")

    base_means = np.nanmean(stack, axis=1)
    base_means = np.where(np.isfinite(base_means) & (base_means > 0),
                          base_means, np.nan)
    valid = np.isfinite(base_means)
    fit = fit_power_law(x[valid], base_means[valid])

    rng = np.random.default_rng(seed)
    bs: list[float] = []
    as_: list[float] = []
    breaks: list[float] = []
    if grid is None:
        grid = np.geomspace(max(x.max(), 1.0), max(x.max(), 1.0) * 100.0, 60)
    preds = np.empty((n_boot, grid.size), dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        means = np.nanmean(stack[:, idx], axis=1)
        means = np.where(np.isfinite(means) & (means > 0), means, np.nan)
        ok = np.isfinite(means)
        if ok.sum() < 2:
            continue
        try:
            f = fit_power_law(x[ok], means[ok])
        except ValueError:
            continue
        bs.append(f.b)
        as_.append(f.a)
        preds[i, :] = f.predict(grid)
        c = f.break_capacity(w_break)
        if c is not None:
            breaks.append(c)

    bs_arr = np.asarray(bs, dtype=float)
    as_arr = np.asarray(as_, dtype=float)
    lo_i, hi_i = int(alpha / 2 * n_boot), int((1 - alpha / 2) * n_boot)

    def _pct(v: np.ndarray) -> tuple[float, float]:
        if v.size == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(v, 100 * alpha / 2)),
                float(np.percentile(v, 100 * (1 - alpha / 2))))

    brk_ci = None
    if breaks:
        brk_ci = _pct(np.asarray(breaks, dtype=float))

    valid_rows = preds[np.all(np.isfinite(preds), axis=1)] if n_boot else preds
    if valid_rows.size:
        grid_lo = np.nanpercentile(valid_rows, 100 * alpha / 2, axis=0)
        grid_hi = np.nanpercentile(valid_rows, 100 * (1 - alpha / 2), axis=0)
    else:
        grid_lo = np.full(grid.shape, np.nan)
        grid_hi = np.full(grid.shape, np.nan)

    return BootstrapResult(
        fit=fit, b_ci=_pct(bs_arr), a_ci=_pct(as_arr), break_ci=brk_ci,
        n_boot=n_boot, n_break_valid=len(breaks),
        grid=grid, grid_lo=grid_lo, grid_hi=grid_hi,
    )


def shelf_life_months(
    break_capacity: float | None,
    reference_capacity_m: float,
    doubling_months: float,
) -> float | None:
    """Months from ``reference_capacity_m`` until C* under a doubling scenario."""
    if break_capacity is None or not np.isfinite(break_capacity):
        return None
    if break_capacity <= reference_capacity_m:
        return 0.0
    return float(doubling_months * math.log2(break_capacity / reference_capacity_m))


# --------------------------------------------------------------------------
# Per-condition analysis + report
# --------------------------------------------------------------------------
def evaluate_shelf_life(
    rows: list[dict[str, Any]],
    w_break: float = DEFAULT_W_BREAK,
    doubling_months: tuple[float, ...] = DEFAULT_DOUBLING_MONTHS,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Full analysis: capacity points + per-condition scaling fits + forecasts."""
    ok = [r for r in rows if r.get("status") == "ok"]
    capacity_points = compute_capacity_points(ok, wer_threshold=w_break)

    # (condition, utt) → {model: wer} for attacked conditions
    pair_wer: dict[tuple[str, str], dict[str, float]] = {}
    pair_meta: dict[tuple[str, str], tuple[str, str, str]] = {}
    for r in ok:
        if r["condition_id"] == ORIGINAL_ID:
            continue
        w = _f(r.get("wer"))
        if not np.isfinite(w):
            continue
        key = (r["condition_id"], r["utt_id"])
        pair_wer.setdefault(key, {})[r["model_key"]] = w
        pair_meta[key] = (r["condition_id"], r.get("margin_db", ""),
                          r.get("kind", ""))

    cond_ids = sorted({k[0] for k in pair_wer})
    reference = PARAMS_BY_MODEL[
        max(CAPACITY_LADDER, key=lambda m: m["params_m"])["model_key"]
    ]

    fits: dict[str, Any] = {}
    for cid in cond_ids:
        keys = sorted(k for k in pair_wer if k[0] == cid)
        per_model: dict[str, np.ndarray] = {}
        for m in CAPACITY_LADDER:
            mk = m["model_key"]
            vals = [pair_wer[k].get(mk, np.nan) for k in keys]
            arr = np.asarray(vals, dtype=float)
            if np.isfinite(arr).any():
                per_model[mk] = arr
        if len(per_model) < 2:
            continue
        x = np.asarray([PARAMS_BY_MODEL[m] for m in sorted(
            per_model, key=lambda k: PARAMS_BY_MODEL[k])], dtype=float)
        vectors = {k: np.nan_to_num(v, nan=float("nan"))
                   for k, v in per_model.items()}
        # drop pairs that are not aligned (any NaN → drop that pair)
        stack = np.vstack([vectors[k] for k in sorted(
            vectors, key=lambda kk: PARAMS_BY_MODEL[kk])])
        keep = np.all(np.isfinite(stack), axis=0)
        if keep.sum() < 2:
            continue
        vectors = {k: v[keep] for k, v in vectors.items()}
        try:
            boot = paired_bootstrap_scaling(
                x, vectors, n_boot=n_boot, seed=seed, w_break=w_break,
            )
        except ValueError:
            continue
        margin = next(
            (m for k, v in pair_meta.items() if k[0] == cid for m in [v[1]]), ""
        )
        c_break = boot.fit.break_capacity(w_break)
        fits[cid] = {
            "condition_id": cid,
            "margin_db": margin,
            "n_pairs": int(keep.sum()),
            **boot.to_dict(),
            "break_capacity_m_params": c_break,
            "break_ci_m_params": list(boot.break_ci) if boot.break_ci else None,
            "forecast_months": {
                f"{dm:g}": shelf_life_months(c_break, reference, dm)
                for dm in doubling_months
            },
        }

    summary = {
        "novelty": "N11 shelf-life forecasting (gap-vs-capacity scaling)",
        "created": datetime.now(timezone.utc).isoformat(),
        "hsr_label": _hsr_label(),
        "w_break": w_break,
        "reference_capacity_m": reference,
        "doubling_scenarios_months": list(doubling_months),
        "capacity_ladder": [dict(m) for m in CAPACITY_LADDER],
        "capacity_points": [p.to_dict() for p in capacity_points],
        "fits": fits,
        "n_pairs_total": len(pair_wer),
        "caveats": [
            "fits use 3 capacity points (one architecture, three sizes)",
            "extrapolation beyond the observed range is scenario-based",
            "human axis is the illustrative STOI-derived proxy (no human study)",
        ],
    }
    return summary


def write_shelf_report(summary: dict[str, Any], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "shelf_life.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines: list[str] = []
    a = lines.append
    a("# Shelf-Life Forecast — CAPTCHA gap vs ASR capacity")
    a("")
    a("**Novelty N11 (original to this work)**: scaling-law forecast of when a")
    a("psychoacoustic CAPTCHA policy stops separating humans from machines as")
    a("the attacker's ASR grows.")
    a("")
    a(f"_HSR note: {summary['hsr_label']}_")
    a("")
    a("## Capacity ladder (observed)")
    a("")
    a("| Model | Params (M) | Baseline WER | Attacked WER | ΔWER | ASR-SR | Gap |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for p in summary["capacity_points"]:
        a(f"| {p['model_key']} | {p['params_m']:g} | {p['baseline_wer']:.3f} "
          f"| {p['attacked_wer']:.3f} | {p['delta_wer']:+.3f} "
          f"| {p['asr_sr']:.3f} | {p['gap']:+.3f} |")
    a("")
    a(f"Security threshold W_break = {summary['w_break']:g} "
      f"(policy broken when predicted attacked WER ≤ W_break).")
    a("")
    a("## Scaling fits and forecasts")
    a("")
    month_cols = summary["doubling_scenarios_months"]
    a("| Condition | λ (dB) | n pairs | exponent b [95% CI] | R² | "
      "C* (M params) [95% CI] | "
      + " | ".join(f"months @{dm:g}m/doubling" for dm in month_cols) + " |")
    a("|---|---|---:|---|---:|---|" + "---:|" * len(month_cols))
    for cid, f in sorted(
        summary["fits"].items(),
        key=lambda kv: (kv[1].get("margin_db") or "", kv[0]),
    ):
        b_lo, b_hi = f["b_ci"]
        c_break = f["break_capacity_m_params"]
        c_ci = f.get("break_ci_m_params")
        c_txt = "no finite break" if c_break is None else (
            f"{c_break:,.0f}" + (
                f" [{c_ci[0]:,.0f}–{c_ci[1]:,.0f}]" if c_ci else ""))
        months = " | ".join(
            ("–" if f["forecast_months"][f"{dm:g}"] is None
             else f"{f['forecast_months'][f'{dm:g}']:.0f}")
            for dm in summary["doubling_scenarios_months"]
        )
        a(f"| {cid} | {f.get('margin_db') or '–'} | {f['n_pairs']} "
          f"| {f['fit']['b']:.2f} [{b_lo:.2f}, {b_hi:.2f}] "
          f"| {f['fit']['r2']:.3f} | {c_txt} | {months} |")
    a("")
    a("## Reading this table")
    a("")
    a("- **C***: model size (M parameters) at which the fitted attacked-WER")
    a("  crosses W_break — the policy's predicted break point. *No finite")
    a("  break* = non-positive exponent or a break beyond 1e12 M parameters.")
    a("- **months**: time to reach C* from the largest evaluated model under")
    a("  an assumed capacity-doubling rate (scenario, not a trend we measured).")
    a("- `–` = no finite break or beyond the horizon shown.")
    a("")
    a("## Caveats")
    a("")
    for c in summary["caveats"]:
        a(f"- {c}")
    a("- corpus/engine provenance: see run_manifest.json / dataset_manifest.json")
    path = out_dir / "shelf_life.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
