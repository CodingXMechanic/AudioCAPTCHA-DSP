#!/usr/bin/env python
"""Comparative benchmark CLI — see experiments/comparative.py for details.

Examples
--------
# Headline run: base-paper subset A protocol on the public stand-in corpus,
# curated transform set, λ-sweep mirror, Whisper + Vosk:
python scripts/run_comparative_benchmark.py --out results/comparative/main

# Full catalog (all 119 transforms), resumable - rerun to continue:
python scripts/run_comparative_benchmark.py --transforms all \
    --out results/comparative/main --workers 4

# DSP-only smoke test (no ASR):
python scripts/run_comparative_benchmark.py --asr none \
    --max-utterances 2 --no-sweep --transforms noise.white \
    --out results/comparative/smoke
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dataset", choices=["librispeech", "wsj"],
                   default="librispeech")
    p.add_argument("--data-root", type=Path,
                   default=Path("data/raw/LibriSpeech/test-clean"))
    p.add_argument("--subset", choices=["A", "B", "C"], default="A",
                   help="base-paper subset protocol (A=70/10spk, B=72+70, "
                        "C=150+72)")
    p.add_argument("--music-dir", type=Path, default=None,
                   help="music corpus for subsets B/C (paper: unnamed corpus)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-utterances", type=int, default=None,
                   help="cap the subset (default: full subset)")
    p.add_argument("--transforms", default="curated",
                   help="'curated', 'all', or comma-separated registry keys")
    p.add_argument("--no-sweep", action="store_true",
                   help="disable the λ-sweep mirror + control condition")
    p.add_argument("--asr", default="whisper_tiny,vosk_small_en",
                   help="comma-separated engines, or 'none'")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", type=Path, default=None,
                   help="default: results/comparative/run_<timestamp>")
    args = p.parse_args(argv)

    engines = [] if args.asr.strip().lower() == "none" else [
        e.strip() for e in args.asr.split(",") if e.strip()
    ]
    for e in engines:
        if e not in ("whisper_tiny", "vosk_small_en"):
            p.error(f"unknown engine {e!r} (available: whisper_tiny, vosk_small_en)")

    out = args.out or Path(
        "results/comparative"
        f"/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    from audiocaptcha_dsp.experiments.comparative import run_benchmark

    summary = run_benchmark(
        out_dir=out,
        dataset=args.dataset,
        data_root=args.data_root,
        subset=args.subset,
        seed=args.seed,
        music_dir=args.music_dir,
        max_utterances=args.max_utterances,
        transforms=args.transforms,
        sweep=not args.no_sweep,
        engines=engines,
        workers=args.workers,
    )
    print(f"\nSummary: {summary}")
    print(f"Ranking: {Path(summary).parent / 'ranking.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
