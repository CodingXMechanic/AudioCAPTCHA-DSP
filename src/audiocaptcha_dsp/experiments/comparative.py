"""Comparative benchmark: Human Intelligibility vs ASR attack effectiveness.

Ranks every registered transform (families A–H) on two axes:

* **Human-intelligibility axis** — STOI proxy (honest, signal-based) and the
  SecurityEvaluator HSR (**labelled illustrative**: it is STOI-derived, no
  human study has been conducted).
* **ASR-effectiveness axis** — WER increase on *three genuinely independent
  ASR families*: OpenAI Whisper tiny + small (attention encoder-decoder,
  two sizes of one family), Vosk small-en (Kaldi nnet3 lineage — the same
  toolkit family as the base paper's white-box DNN-HMM) and wav2vec 2.0
  base (self-supervised encoder, CTC head).  The headline attack metric is
  the family-macro average of cross-family delta-WER, so each lineage
  weighs exactly once regardless of how many sizes it is measured at.

Dataset fidelity (see docs/BASE_PAPER_DATASET.md): the sample selection
reuses ``WSJAdapter.load_base_paper_subset`` — the exact base-paper
protocol (subset A = 70 utterances / 10 speakers; B = 72+70; C = 150+72,
≤6 phones/s) — on whichever corpus is available.  Every run writes a
``dataset_manifest.json`` so all conclusions are traceable.

The λ-sweep mirrors the base paper's hearing-threshold margin experiment
(λ ∈ {0, 5, 10, …, 50} dB) plus a "no hearing threshold" control whose
perturbation power is matched to λ = 0 (the paper's "None" column).
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.evaluation.metrics import compute_metrics, compute_wer
from audiocaptcha_dsp.asr.engine import normalize_transcript
from audiocaptcha_dsp.evaluation.dataset import (
    BASE_PAPER_SUBSETS,
    LibriSpeechAdapter,
    WSJAdapter,
    SpeechSample,
)
from audiocaptcha_dsp.experiments.runner import resolve_transform
from audiocaptcha_dsp.transforms.registry import get_registry

# --------------------------------------------------------------------------
# Labels that MUST accompany every published number from this benchmark.
# --------------------------------------------------------------------------
HSR_LABEL = "illustrative (STOI-derived proxy; no human study conducted)"
ASR_LABELS: dict[str, dict[str, str]] = {
    "whisper_tiny": {
        "model": "openai-whisper tiny",
        "family": "Whisper (attention encoder-decoder)",
        "group": "whisper",
        "kind": "real",
    },
    "whisper_small": {
        "model": "openai-whisper small",
        "family": "Whisper (attention encoder-decoder)",
        "group": "whisper",
        "kind": "real",
    },
    "vosk_small_en": {
        "model": "vosk-model-small-en-us-0.15",
        "family": "Vosk/Kaldi nnet3 (lineage of the base paper's DNN-HMM toolkit)",
        "group": "kaldi",
        "kind": "real",
    },
    "wav2vec2_base": {
        "model": "facebook/wav2vec2-base-960h",
        "family": "wav2vec 2.0 (self-supervised, CTC head)",
        "group": "ssl",
        "kind": "real",
    },
}
DEFAULT_ENGINES = list(ASR_LABELS)

DEFAULT_TRANSFORMS = [
    "noise.white", "noise.babble", "noise.pink", "noise.band_limited",
    "noise.speech_shaped", "noise.competing_speaker", "noise.modulated",
    "temporal.jitter", "temporal.time_stretch", "temporal.pitch_shift",
    "temporal.speed_perturbation", "temporal.segment_displacement",
    "temporal.time_masking",
    "multirate.narrowband", "multirate.aliasing", "multirate.codec_artifacts",
    "multirate.rational", "multirate.drift",
    "spectral.masking", "spectral.bark_masking", "spectral.phase_randomization",
    "spectral.formant_suppression", "spectral.harmonic_attenuation",
    "spectral.hole", "spectral.comb_filter",
    "noise.tonal", "noise.echo", "noise.harmonic_interference", "noise.delay_add",
    "psychoacoustic.masked_noise", "psychoacoustic.bark_perturbation",
    "psychoacoustic.erb_perturbation", "psychoacoustic.temporal_masking",
    "psychoacoustic.signal_threshold", "psychoacoustic.loudness_preserving",
    "psychoacoustic.frame_budget", "psychoacoustic.speech_aware",
    "psychoacoustic.combined_masking",
    "adversarial.black_box", "adversarial.gradient_free",
    "adversarial.multi_objective", "adversarial.phoneme_guided",
    "novel.phoneme_aware", "novel.phoneme_dropout", "novel.multi_domain",
    "novel.adaptive_formant", "novel.captcha_optimal", "novel.defense_robust",
    "channel.mp3", "channel.opus", "channel.packet_loss", "channel.room_reverb",
    "channel.noise_suppression", "channel.dereverberation",
    "channel.device_distortion",
]

SWEEP_TARGETS = [
    "psychoacoustic.masked_noise",
    "psychoacoustic.bark_perturbation",
    "psychoacoustic.temporal_masking",
    "psychoacoustic.signal_threshold",
]
SWEEP_MARGINS_DB = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
CONTROL_ID = "control.no_hearing_threshold"
ORIGINAL_ID = "original"

ROW_FIELDS = [
    "condition_id", "kind", "transform_key", "family", "params_json",
    "margin_db", "utt_id", "spk", "dur_s", "status", "error",
    "snr_db", "stoi_proxy", "mbsd", "si_sdr_db", "rms_ratio", "ref_text",
    *[f"wer_{eid}" for eid in ASR_LABELS],
    *[f"hyp_{eid}" for eid in ASR_LABELS],
]


def asr_family_groups(
    engines: list[str] | None = None,
) -> dict[str, list[str]]:
    """Group engine ids by ASR lineage (``whisper`` / ``kaldi`` / ``ssl``).

    The headline cross-ΔWER macro-averages over these groups, so a family
    evaluated at two model sizes (Whisper tiny + small) does not outweigh a
    family evaluated once.  Unknown ids fall back to a group of their own.
    """
    groups: dict[str, list[str]] = {}
    for eid in (engines if engines is not None else list(ASR_LABELS)):
        group = ASR_LABELS.get(eid, {}).get("group", eid)
        groups.setdefault(group, []).append(eid)
    return groups


def _macro_family_delta(deltas: dict[str, float]) -> float:
    """Macro-average per-engine ΔWER over ASR families.

    ``deltas`` maps ``wer_<engine>`` columns to their ΔWER.  Engines of one
    lineage (Whisper tiny + small) are averaged first, then the family
    means are averaged — each independent family therefore weighs exactly
    once, and with one engine per family this equals the plain mean.
    """
    if not deltas:
        return float("nan")
    engine_ids = [c.removeprefix("wer_") for c in deltas]
    family_means = [
        float(np.mean([deltas[f"wer_{e}"] for e in members]))
        for members in asr_family_groups(engine_ids).values()
    ]
    return float(np.mean(family_means))


# --------------------------------------------------------------------------
# Condition construction
# --------------------------------------------------------------------------
@dataclass
class Condition:
    condition_id: str
    kind: str  # original | transform | sweep | control
    transform_key: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    margin_db: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "kind": self.kind,
            "transform_key": self.transform_key,
            "family": self.family,
            "params": self.params,
            "margin_db": self.margin_db,
        }


def _with_seed(key: str, params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Add a fixed seed to transforms that accept one (reproducibility)."""
    reg = get_registry()
    try:
        sig = inspect.signature(reg[key].cls.__init__)
    except (ValueError, TypeError):
        return params
    if "seed" in sig.parameters:
        params = dict(params)
        params.setdefault("seed", seed)
    return params


def build_conditions(
    transform_keys: list[str],
    sweep: bool = True,
    seed: int = 42,
) -> list[Condition]:
    """Original control + per-transform defaults + λ-sweep + 'None' control.

    Duplicate parameter sets (e.g. masked_noise default margin 20 dB vs the
    sweep grid) are de-duplicated so no condition is evaluated twice.
    """
    reg = get_registry()
    conds: list[Condition] = [
        Condition(ORIGINAL_ID, "original", "baseline.identity", "-")
    ]
    seen: set[tuple[str, str]] = {(ORIGINAL_ID, "")}

    def _add(c: Condition) -> None:
        sig = json.dumps(sorted(c.params.items()), default=str)
        key = (c.transform_key, sig)
        if key in seen:
            return
        seen.add(key)
        conds.append(c)

    for key in transform_keys:
        if key == "baseline.identity":
            continue
        if key == CONTROL_ID:
            _add(Condition(
                CONTROL_ID, "control", "psychoacoustic.masked_noise", "F",
                {"margin_db": 0.0, "seed": seed, "power_match": "lambda0"},
            ))
            continue
        if key not in reg:
            raise ValueError(f"unknown transform key: {key!r}")
        spec = reg[key]
        params = _with_seed(key, dict(spec.params), seed)
        _add(Condition(key, "transform", key, spec.family, params))

    if sweep:
        for key in SWEEP_TARGETS:
            spec = reg[key]
            for m in SWEEP_MARGINS_DB:
                params = dict(spec.params)
                params["margin_db"] = m
                params = _with_seed(key, params, seed)
                _add(Condition(
                    f"{key}#margin={m:g}", "sweep", key, spec.family,
                    params, margin_db=m,
                ))

    # Paper's "None" column: perturbation with NO hearing-threshold
    # constraint, power-matched to the λ = 0 (at-threshold) injection.
    _add(Condition(
        CONTROL_ID, "control", "psychoacoustic.masked_noise", "F",
        {"margin_db": 0.0, "seed": seed, "power_match": "lambda0"},
        margin_db=None,
    ))
    return conds


# --------------------------------------------------------------------------
# The "None" (no hearing threshold) control transform
# --------------------------------------------------------------------------
def apply_no_threshold_control(
    signal: Signal, seed: int = 42, reference_margin_db: float = 0.0
) -> Signal:
    """White noise with the same total power as the at-threshold (λ=0)
    psychoacoustic injection — the base paper's unconstrained 'None' column."""
    ref_t = resolve_transform(
        "psychoacoustic.masked_noise",
        {"margin_db": reference_margin_db, "seed": seed},
    )
    ref = ref_t(signal)
    pert = np.asarray(ref.waveform, dtype=np.float64) - signal.waveform
    p_target = float(np.sum(pert**2))
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(signal.waveform.shape[0])
    p_n = float(np.sum(n**2)) or 1.0
    if p_target > 0:
        n *= math.sqrt(p_target / p_n)
    return Signal(waveform=signal.waveform + n, sample_rate=signal.sample_rate)


# --------------------------------------------------------------------------
# Dataset selection + manifests
# --------------------------------------------------------------------------
def select_samples(
    dataset: str,
    data_root: Path,
    subset: str = "A",
    seed: int = 42,
    music_dir: Path | None = None,
    max_utterances: int | None = None,
) -> tuple[list[SpeechSample], dict[str, Any]]:
    """Load a corpus and apply the *exact base-paper subset protocol*."""
    if subset not in BASE_PAPER_SUBSETS:
        raise ValueError(f"subset must be one of {sorted(BASE_PAPER_SUBSETS)}")

    if dataset == "librispeech":
        pool = LibriSpeechAdapter(data_root).load()
        if not pool:
            raise FileNotFoundError(f"no LibriSpeech audio under {data_root}")
        samples = WSJAdapter(data_root).load_base_paper_subset(
            subset, seed=seed, music_dir=music_dir, pool=pool
        )
        corpus = "librispeech test-clean (stand-in for WSJ: LDC-licensed)"
        adapter_name = "LibriSpeechAdapter + base-paper subset protocol"
    elif dataset == "wsj":
        samples = WSJAdapter(data_root).load_base_paper_subset(
            subset, seed=seed, music_dir=music_dir
        )
        corpus = "wsj (exact base-paper corpus)"
        adapter_name = "WSJAdapter"
    else:
        raise ValueError(f"unsupported dataset: {dataset!r}")

    if not samples:
        raise RuntimeError(f"subset {subset} selection returned no samples")
    if max_utterances is not None:
        samples = samples[:max_utterances]
    return samples, {
        "corpus": corpus,
        "adapter": adapter_name,
        "root": str(data_root),
        "protocol": f"base-paper-{subset}",
        "subset_spec": BASE_PAPER_SUBSETS[subset],
        "seed": seed,
        "music_dir": str(music_dir) if music_dir else (
            "omitted (no music corpus supplied)" if subset in ("B", "C") else None
        ),
    }


def _sha256_manifest(samples: list[SpeechSample]) -> str:
    h = hashlib.sha256()
    for s in sorted(samples, key=lambda s: s.utterance_id):
        h.update(f"{s.utterance_id}|{s.speaker_id}|{s.transcript}\n".encode())
    return h.hexdigest()


def write_dataset_manifest(
    out_dir: Path, samples: list[SpeechSample], info: dict[str, Any]
) -> Path:
    spec = info.get("protocol", "")
    sel = {"mode": "balanced" if spec == "base-paper-A" else
           ("balanced" if spec == "base-paper-B" else "random"),
           "seed": info.get("seed", 42)}
    n_speakers = info.get("subset_spec", {}).get("n_speakers", 0)
    sel.update({"total": len(samples), "n_speakers": n_speakers})
    manifest = {
        "corpus": info["corpus"],
        "base_paper": "Schönherr et al. 2018, arXiv:1808.05665",
        "protocol": info["protocol"],
        "adapter": info["adapter"],
        "root": info["root"],
        "selection": sel,
        "speakers": sorted({s.speaker_id for s in samples}),
        "utterances": [
            {
                "id": s.utterance_id,
                "speaker": s.speaker_id,
                "duration": round(
                    s.duration or len(s.waveform) / max(s.sample_rate, 1), 3
                ),
                "transcript": s.transcript,
                "source_path": s.metadata.get("source_path"),
            }
            for s in samples
        ],
        "n_samples": len(samples),
        "corpus_sha256": _sha256_manifest(samples),
        "music": info.get("music_dir"),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    path = out_dir / "dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip() or None
    except Exception:
        return None


def write_run_manifest(
    out_dir: Path, args: dict[str, Any], conditions: list[Condition]
) -> Path:
    import platform

    manifest = {
        "script": "run_comparative_benchmark",
        "created": datetime.now(timezone.utc).isoformat(),
        "args": args,
        "n_conditions": len(conditions),
        "conditions": [c.to_dict() for c in conditions],
        "asr_engines": {
            k: ASR_LABELS[k] for k in args.get("engines", []) if k in ASR_LABELS
        },
        "hsr_label": HSR_LABEL,
        "metrics": {
            "human_axis": "stoi_proxy (signal-based, honest)",
            "attack_axis": "cross-family delta-WER (mean over independent ASR families)",
        },
        "base_paper": "Schönherr et al. 2018, arXiv:1808.05665",
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": _package_versions(),
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _package_versions() -> dict[str, str]:
    import importlib

    out: dict[str, str] = {}
    for name in ("numpy", "scipy", "pandas", "torch", "whisper", "vosk",
                 "jiwer", "librosa", "soundfile", "transformers"):
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = "not installed"
    return out


# --------------------------------------------------------------------------
# Worker (parallel execution; also used sequentially with workers=0)
# --------------------------------------------------------------------------
_G: dict[str, Any] = {}


def _make_engine(name: str) -> Any:
    """Instantiate one registered ASR engine by id (see ``ASR_LABELS``)."""
    if name.startswith("whisper_"):
        from audiocaptcha_dsp.asr import WhisperAdapter

        return WhisperAdapter(
            model_size=name.split("_", 1)[1], offline_fallback=False
        )
    if name == "vosk_small_en":
        from audiocaptcha_dsp.asr import VoskAdapter

        return VoskAdapter(offline_fallback=False)
    if name == "wav2vec2_base":
        from audiocaptcha_dsp.asr import Wav2Vec2Adapter

        return Wav2Vec2Adapter()
    raise ValueError(f"unknown ASR engine: {name!r}")


def _worker_init(asr_names: list[str], torch_threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, torch_threads)))
    if torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(torch_threads)
        except Exception:
            pass
    engines: dict[str, Any] = {}
    for name in asr_names:
        engines[name] = _make_engine(name)
    for e in engines.values():
        e.load()
    _G["engines"] = engines


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    import soundfile as sf

    cond = task["condition"]
    row: dict[str, Any] = {
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
    for f in ROW_FIELDS:
        row.setdefault(f, "")
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
            t = resolve_transform(cond["transform_key"], dict(cond["params"]))
            processed = t(original)

        metrics = compute_metrics(original, processed)
        for key, col in (
            ("snr_db", "snr_db"), ("stoi_proxy", "stoi_proxy"),
            ("mbsd", "mbsd"), ("si_sdr_db", "si_sdr_db"),
            ("rms_ratio", "rms_ratio"),
        ):
            v = metrics.get(key)
            row[col] = "" if v is None or not np.isfinite(v) else f"{float(v):.6g}"

        ref = normalize_transcript(task["transcript"])
        for name, eng in _G.get("engines", {}).items():
            hyp = eng.transcribe(processed).text
            row[f"hyp_{name}"] = hyp
            row[f"wer_{name}"] = (
                f"{compute_wer(ref, normalize_transcript(hyp)):.6f}"
            )
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        row["_trace"] = traceback.format_exc(limit=3)
    return row


def _tasks_for(
    conditions: list[Condition], samples: list[SpeechSample]
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for cond in conditions:
        for s in samples:
            path = s.metadata.get("source_path")
            if not path:
                raise RuntimeError(
                    f"sample {s.utterance_id} has no source_path in metadata "
                    "(required for parallel workers)"
                )
            dur = s.duration or (len(s.waveform) / max(s.sample_rate, 1))
            tasks.append({
                "condition": cond.to_dict(),
                "utt_id": s.utterance_id,
                "spk": s.speaker_id,
                "dur_s": round(float(dur), 3),
                "path": path,
                "transcript": s.transcript,
            })
    return tasks


def _load_done_keys(rows_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if rows_path.exists():
        with rows_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["condition_id"], r["utt_id"]))
    return done


def _check_engine_set(rows_path: Path, engines: list[str]) -> None:
    """Refuse to resume a run produced by a *different* engine set.

    ``rows.csv`` accumulates one row per (condition, utterance); the headline
    cross-ΔWER is a macro-average over the evaluated ASR families.  Mixing rows
    from two engine sets in one file would silently change that definition
    half-way through the table, so a mismatch stops the run instead.
    """
    if not rows_path.exists() or rows_path.stat().st_size == 0:
        return
    produced: set[str] = set()
    with rows_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for eid in ASR_LABELS:
                if r.get(f"wer_{eid}", "").strip():
                    produced.add(eid)
    if produced != set(engines):
        raise RuntimeError(
            f"{rows_path} holds rows for engines {sorted(produced)} but this "
            f"run requests {sorted(engines)}; cross-ΔWER must be computed "
            "from one engine set. Move rows.csv aside (or choose a fresh "
            "--out) to start a clean run."
        )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_benchmark(
    out_dir: Path,
    dataset: str = "librispeech",
    data_root: Path = Path("data/raw/LibriSpeech/test-clean"),
    subset: str = "A",
    seed: int = 42,
    music_dir: Path | None = None,
    max_utterances: int | None = None,
    transforms: str = "curated",
    sweep: bool = True,
    engines: list[str] | None = None,
    workers: int = 3,
    log: Any = print,
) -> Path:
    """Run (or resume) the comparative benchmark.  Returns summary.csv path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engines = list(engines) if engines is not None else list(DEFAULT_ENGINES)
    unknown = [e for e in engines if e not in ASR_LABELS]
    if unknown:
        raise ValueError(
            f"unknown ASR engine(s) {unknown}; available: {list(ASR_LABELS)}"
        )
    rows_path = out_dir / "rows.csv"
    _check_engine_set(rows_path, engines)  # before any dataset/ASR work

    # ---- transforms -----------------------------------------------------
    reg = get_registry()
    if transforms == "curated":
        keys = [k for k in DEFAULT_TRANSFORMS if k in reg]
    elif transforms == "all":
        keys = [k for k in sorted(reg) if k != "baseline.identity"]
    else:
        keys = [k.strip() for k in transforms.split(",") if k.strip()]
    conditions = build_conditions(keys, sweep=sweep, seed=seed)
    log(f"[conditions] {len(conditions)} "
        f"(transforms={len(keys)}, sweep={'on' if sweep else 'off'})")

    # ---- dataset --------------------------------------------------------
    samples, info = select_samples(
        dataset, Path(data_root), subset=subset, seed=seed,
        music_dir=music_dir, max_utterances=max_utterances,
    )
    manifest_path = write_dataset_manifest(out_dir, samples, info)
    log(f"[dataset] {info['protocol']} | {len(samples)} samples | "
        f"{len({s.speaker_id for s in samples})} speakers | "
        f"manifest={manifest_path.name}")

    args_dump = {
        "dataset": dataset, "data_root": str(data_root), "subset": subset,
        "seed": seed, "max_utterances": max_utterances,
        "transforms": transforms, "sweep": sweep, "engines": engines,
        "workers": workers, "music_dir": str(music_dir) if music_dir else None,
    }
    write_run_manifest(out_dir, args_dump, conditions)

    # ---- tasks + resume -------------------------------------------------
    done = _load_done_keys(rows_path)
    all_tasks = _tasks_for(conditions, samples)
    tasks = [t for t in all_tasks
             if (t["condition"]["condition_id"], t["utt_id"]) not in done]
    log(f"[tasks] {len(all_tasks)} total, {len(done)} already done, "
        f"{len(tasks)} to run")
    if not tasks:
        log("[done] nothing to run; aggregating existing rows")
        return aggregate_results(out_dir, log=log)

    t0 = time.time()
    n_done = 0

    def _write(row: dict[str, Any]) -> None:
        nonlocal n_done
        row.pop("_trace", None)
        with rows_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
            if not rows_path.exists() or rows_path.stat().st_size == 0:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in ROW_FIELDS})
        n_done += 1
        if n_done % 25 == 0 or n_done == len(tasks):
            el = time.time() - t0
            rate = n_done / el if el > 0 else 0
            eta = (len(tasks) - n_done) / rate if rate > 0 else 0
            log(f"[progress] {n_done}/{len(tasks)} "
                f"({100 * n_done / len(tasks):.1f}%) | "
                f"{rate:.2f} task/s | elapsed {el / 60:.1f} min | "
                f"ETA {eta / 60:.1f} min")

    if workers <= 1:
        _worker_init(engines, torch_threads=0)
        for t in tasks:
            _write(_run_task(t))
    else:
        torch_threads = max(1, (os.cpu_count() or 4) // workers)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(engines, torch_threads),
        ) as pool:
            futs = [pool.submit(_run_task, t) for t in tasks]
            for fut in as_completed(futs):
                _write(fut.result())

    log(f"[run] finished in {(time.time() - t0) / 60:.1f} min")
    return aggregate_results(out_dir, log=log)


# --------------------------------------------------------------------------
# Aggregation + ranking
# --------------------------------------------------------------------------
def aggregate_results(out_dir: Path, log: Any = print) -> Path:
    import pandas as pd

    from audiocaptcha_dsp.evaluation.hag_metrics import SecurityEvaluator

    out_dir = Path(out_dir)
    rows_path = out_dir / "rows.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"no rows.csv under {out_dir}")
    df = pd.read_csv(rows_path, dtype=str, keep_default_na=False)
    if df.empty:
        raise RuntimeError("rows.csv is empty")

    wer_cols = [f"wer_{eid}" for eid in ASR_LABELS]
    num_cols = ["dur_s", "snr_db", "stoi_proxy", "mbsd", "si_sdr_db",
                "rms_ratio", "margin_db", *wer_cols]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    df["ok"] = df["status"] == "ok"
    ok = df[df["ok"]]

    engine_cols = [c for c in wer_cols
                   if c in ok.columns and ok[c].notna().any()]
    baselines = {c: float(ok.loc[ok.condition_id == ORIGINAL_ID, c].mean())
                 for c in engine_cols}
    for c, v in baselines.items():
        if not np.isfinite(v):
            raise RuntimeError(
                f"baseline WER for {c} is unavailable — the 'original' "
                "condition has no successful ASR rows; cannot compute ΔWER"
            )
    log(f"[baseline WER] " + ", ".join(
        f"{c.replace('wer_', '')}={v:.3f}" for c, v in baselines.items()))

    records: list[dict[str, Any]] = []
    evaluator = SecurityEvaluator()
    for cid, g_all in df.groupby("condition_id", sort=False):
        meta = g_all.iloc[0]
        g = g_all[g_all["ok"]]
        rec: dict[str, Any] = {
            "condition_id": cid,
            "kind": meta["kind"],
            "transform_key": meta["transform_key"],
            "family": meta["family"],
            "params_json": meta["params_json"],
            "margin_db": meta["margin_db"],
            "n_ok": int(len(g)),
            "n_err": int(len(g_all) - len(g)),
            "dur_s_mean": g["dur_s"].mean(),
            "stoi_mean": g["stoi_proxy"].mean(),
            "stoi_std": g["stoi_proxy"].std(ddof=0),
            "snr_mean": g.loc[np.isfinite(g["snr_db"]), "snr_db"].mean(),
            "mbsd_mean": g["mbsd"].mean(),
            "si_sdr_mean": g["si_sdr_db"].mean(),
        }
        if g.empty:  # every row errored: report the condition, no metrics
            rec["cross_delta_wer"] = np.nan
            records.append(rec)
            continue
        deltas: dict[str, float] = {}
        for c in engine_cols:
            m = g[c].mean()
            d = m - baselines[c]
            rec[f"{c}_mean"] = m
            rec[f"{c}_delta"] = d
            deltas[c] = d
        rec["cross_delta_wer"] = _macro_family_delta(deltas)

        # SecurityEvaluator → HSR (illustrative), ASR-SR, HAG, CSS
        if cid != ORIGINAL_ID and np.isfinite(rec["snr_mean"]):
            dsp = [
                {"stoi_proxy": float(r.stoi_proxy), "mbsd": float(r.mbsd),
                 "snr_db": float(r.snr_db)}
                for r in g.itertuples()
                if np.isfinite(r.stoi_proxy) and np.isfinite(r.snr_db)
            ]
            asr_res = {c.replace("wer_", ""): g[c].dropna().tolist()
                       for c in engine_cols}
            try:
                ev = evaluator.evaluate(
                    transform_name=str(meta["transform_key"]),
                    condition_params=json.loads(meta["params_json"]),
                    dsp_metrics=dsp,
                    asr_results=asr_res or None,
                    original_wer=baselines.get(
                        engine_cols[0], 0.0) if engine_cols else 0.0,
                ).to_dict()
                rec["hsr"] = ev.get("hsr")
                rec["asr_sr"] = ev.get("asr_sr")
                rec["hag"] = ev.get("hag")
                rec["css"] = ev.get("css")
            except Exception as e:  # keep the run alive; leave blanks
                log(f"[warn] SecurityEvaluator failed for {cid}: {e}")
        records.append(rec)

    summary = pd.DataFrame(records)

    # ---- ranks (transforms only; original is the control row) ----------
    ranked = summary[summary.kind != "original"].copy()
    if not ranked.empty:
        ranked["human_rank"] = ranked["stoi_mean"].rank(
            ascending=False, method="min")
        if ranked["cross_delta_wer"].notna().any():
            ranked["attack_rank"] = ranked["cross_delta_wer"].rank(
                ascending=False, method="min")
        else:
            ranked["attack_rank"] = np.nan
        # WHAT-REMAINS §8 views: highest human-ASR gap (HAG from the
        # SecurityEvaluator) and best perceptual quality (lowest MBSD)
        if "hag" in ranked and ranked["hag"].notna().any():
            ranked["gap_rank"] = ranked["hag"].rank(
                ascending=False, method="min")
        else:
            ranked["gap_rank"] = np.nan
        if ranked["mbsd_mean"].notna().any():
            ranked["quality_rank"] = ranked["mbsd_mean"].rank(
                ascending=True, method="min")
        else:
            ranked["quality_rank"] = np.nan
        ranked["pareto"] = [
            _is_nondominated(row.stoi_mean, row.cross_delta_wer, ranked)
            for row in ranked.itertuples()
        ]
        summary = summary.merge(
            ranked[["condition_id", "human_rank", "attack_rank",
                    "gap_rank", "quality_rank", "pareto"]],
            on="condition_id", how="left",
        )
        summary["pareto"] = summary["pareto"].fillna(False).astype(bool)
    else:
        summary["human_rank"] = np.nan
        summary["attack_rank"] = np.nan
        summary["gap_rank"] = np.nan
        summary["quality_rank"] = np.nan
        summary["pareto"] = False

    summary = summary.sort_values(
        "attack_rank", na_position="last"
    ).reset_index(drop=True)
    summary["hsr_label"] = HSR_LABEL

    csv_path = out_dir / "summary.csv"
    summary.to_csv(csv_path, index=False)

    summary_json = {
        "created": datetime.now(timezone.utc).isoformat(),
        "hsr_label": HSR_LABEL,
        "baseline_wer": baselines,
        "n_rows": int(len(df)),
        "n_ok": int(df["ok"].sum()),
        "n_err": int((~df["ok"]).sum()),
        "conditions": summary.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonify(summary_json), indent=2), encoding="utf-8"
    )

    _write_ranking_md(out_dir, summary, baselines, df)
    log(f"[aggregate] {csv_path} | {len(summary)} conditions | "
        f"{int(df['ok'].sum())} ok / {int((~df['ok']).sum())} error rows")
    return csv_path


def _is_nondominated(stoi: float, delta: float, df: "Any") -> bool:
    """Maximize both axes: nondominated iff no other row is >= in both."""
    if not np.isfinite(delta):
        return False
    others = df[(df.stoi_mean >= stoi) & (df.cross_delta_wer >= delta)]
    return len(others) == 1  # only itself


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _fmt(v: Any, nd: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return f"{f:.{nd}f}"


def _write_ranking_md(
    out_dir: Path, summary: "Any", baselines: dict, df: "Any"
) -> Path:
    lines: list[str] = []
    a = lines.append
    a("# Comparative Ranking — Human Intelligibility vs ASR Effectiveness")
    a("")
    a("> **Dataset provenance:** `dataset_manifest.json` in this directory "
      "(base-paper subset protocol, Schönherr et al. 2018, arXiv:1808.05665).")
    a(f"> **HSR values are {HSR_LABEL}.**")
    delta_cols = [c for c in summary.columns
                  if c.startswith("wer_") and c.endswith("_delta")]
    engine_ids = [c[len("wer_"):-len("_delta")] for c in delta_cols]
    fam_names = sorted({
        ASR_LABELS.get(e, {}).get("family", e) for e in engine_ids
    })
    a("> **ASR engines:** "
      + ", ".join(e.replace("_", " ") for e in engine_ids)
      + f" — {len(fam_names)} independent famil"
      + ("y" if len(fam_names) == 1 else "ies") + ": "
      + "; ".join(fam_names) + ". All are real recognizers.")
    a("")
    a(f"Rows: {int(df['ok'].sum())} ok, {int((~df['ok']).sum())} error. "
      f"Baseline WER: " + ", ".join(
          f"{k.replace('wer_', '')}={_fmt(v)}" for k, v in baselines.items()))
    a("")
    a("## Overall ranking (attack = cross-family ΔWER, human = STOI)")
    a("")
    a("| atk | condition | fam | kind | STOI | "
      + " | ".join(f"ΔWER {e.replace('_', ' ')}" for e in engine_ids)
      + " | cross-ΔWER | SNR dB | HSR (ill.) | pareto |")
    a("|---:|---|:-:|:-:|---:|" + "---:|" * len(delta_cols)
      + "---:|---:|---:|:-:|")
    ranked = summary.sort_values("attack_rank", na_position="last")
    for r in ranked.itertuples():
        a("| " + " | ".join([
            _fmt(r.attack_rank, 0) if np.isfinite(getattr(r, "attack_rank"))
            else "—",
            f"`{r.condition_id}`", r.family, r.kind,
            _fmt(r.stoi_mean),
            *(_fmt(getattr(r, c, np.nan)) for c in delta_cols),
            _fmt(r.cross_delta_wer),
            _fmt(r.snr_mean, 1),
            _fmt(getattr(r, "hsr", np.nan)),
            "★" if getattr(r, "pareto", False) else "",
        ]) + " |")
    a("")
    a("## Human-intelligibility view (top 15 by STOI)")
    a("")
    a("| # | condition | fam | STOI | HSR (ill.) | cross-ΔWER |")
    a("|---:|---|:-:|---:|---:|---:|")
    for i, r in enumerate(
        summary.sort_values("stoi_mean", ascending=False).head(15).itertuples()
    ):
        a(f"| {i + 1} | `{r.condition_id}` | {r.family} | {_fmt(r.stoi_mean)} "
          f"| {_fmt(getattr(r, 'hsr', np.nan))} "
          f"| {_fmt(r.cross_delta_wer)} |")
    a("")

    # ---- human-ASR gap view (WHAT-REMAINS §8: highest human-ASR gap) ---
    a("## Human-ASR gap view (top 10 by HAG)")
    a("")
    a("| rank | condition | fam | HAG | HSR (ill.) | ASR-SR | STOI | "
      "cross-ΔWER |")
    a("|---:|---|:-:|---:|---:|---:|---:|---:|")
    if "gap_rank" in summary.columns and summary["gap_rank"].notna().any():
        for r in (
            summary.dropna(subset=["gap_rank"])
            .nsmallest(10, "gap_rank").itertuples()
        ):
            a(f"| {int(r.gap_rank)} | `{r.condition_id}` | {r.family} "
              f"| {_fmt(getattr(r, 'hag', np.nan))} "
              f"| {_fmt(getattr(r, 'hsr', np.nan))} "
              f"| {_fmt(getattr(r, 'asr_sr', np.nan))} "
              f"| {_fmt(r.stoi_mean)} | {_fmt(r.cross_delta_wer)} |")
    else:
        a("| — | (HAG unavailable for this run) | | | | | | |")
    a("")

    # ---- λ-sweep mirror ------------------------------------------------
    sweep = summary[summary.kind == "sweep"]
    if not sweep.empty:
        a("## λ-sweep mirror (hearing-threshold margin, base-paper grid)")
        a("")
        a("Paper: WER/attack success vs λ ∈ {0…50} dB; `None` = unconstrained "
          "perturbation (our power-matched control row).")
        a("")
        for key in SWEEP_TARGETS:
            sub = sweep[sweep.transform_key == key].sort_values("margin_db")
            if sub.empty:
                continue
            ctrl = summary[summary.condition_id == CONTROL_ID]
            a(f"### `{key}`")
            a("")
            a("| λ (dB) | STOI | "
              + " | ".join(f"ΔWER {e.replace('_', ' ')}" for e in engine_ids)
              + " | cross-ΔWER | SNR dB |")
            a("|---:|---:|" + "---:|" * len(delta_cols) + "---:|---:|")
            for r in sub.itertuples():
                a("| " + " | ".join([
                    f"{r.margin_db:g}", _fmt(r.stoi_mean),
                    *(_fmt(getattr(r, c, np.nan)) for c in delta_cols),
                    _fmt(r.cross_delta_wer), _fmt(r.snr_mean, 1),
                ]) + " |")
            if not ctrl.empty:
                r = ctrl.iloc[0]
                a("| " + " | ".join([
                    "None", _fmt(r.stoi_mean),
                    *(_fmt(r.get(c, np.nan)) for c in delta_cols),
                    _fmt(r.cross_delta_wer), _fmt(r.snr_mean, 1),
                ]) + " |")
            a("")

    # ---- family summary ------------------------------------------------
    a("## Family-level summary")
    a("")
    a("| family | n | mean STOI | mean cross-ΔWER | best attack condition |")
    a("|:-:|---:|---:|---:|:--|")
    fam = summary[summary.kind != "original"]
    for f, g in fam.groupby("family"):
        best = g.loc[g["cross_delta_wer"].idxmax()] if g["cross_delta_wer"].notna().any() else None
        a(f"| {f} | {len(g)} | {_fmt(g.stoi_mean.mean())} "
          f"| {_fmt(g.cross_delta_wer.mean())} "
          f"| `{best.condition_id}` |" if best is not None
          else f"| {f} | {len(g)} | {_fmt(g.stoi_mean.mean())} | — | — |")
    a("")
    a("_Generated by `scripts/run_comparative_benchmark.py`; regenerate with "
      "a rerun (rows.csv is resumable)._")

    path = out_dir / "ranking.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
