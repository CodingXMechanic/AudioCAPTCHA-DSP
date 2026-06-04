from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.types import BenchmarkResult, ExperimentResult

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = "1.0"


def save_artifact(data: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _make_serializable(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info("Saved artifact to %s", path)


def load_artifact(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_signal_wav(signal: Signal, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = signal.to_mono().waveform
    sf.write(str(path), mono, signal.sample_rate, subtype="PCM_16")
    logger.info("Saved WAV to %s", path)
    return path


def save_experiment_manifest(benchmark: BenchmarkResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": _MANIFEST_VERSION,
        "experiment_id": benchmark.experiment_id,
        "timestamp": time.time(),
        "summary": benchmark.summary(),
        "results": [r.to_dict() for r in benchmark.results],
    }
    save_artifact(manifest, path)


def load_experiment_manifest(path: Path | str) -> dict[str, Any]:
    return load_artifact(path)


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(_make_serializable(item) for item in obj)
    return obj
