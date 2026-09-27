#!/usr/bin/env python
"""
Shelf-life forecasting pass (novelty N11)
=========================================
Runs the capacity-ladder evaluation (Whisper tiny/base/small) over a compact
psychoacoustic policy set, then fits the attacked-WER scaling law, forecasts
the break capacity, and writes shelf_life.{json,md} + manifests.

Usage
-----
    # full pass (default: subset A, margins 0/10/20/40 + control, 3 models)
    python scripts/run_shelf_life.py --out results/shelf_life --workers 2

    # re-analysis only (rows already present)
    python scripts/run_shelf_life.py --out results/shelf_life --analysis-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audiocaptcha_dsp.experiments.shelf_life import (  # noqa: E402
    DEFAULT_DOUBLING_MONTHS,
    DEFAULT_MARGINS_DB,
    DEFAULT_W_BREAK,
    evaluate_shelf_life,
    read_shelf_rows,
    run_shelf_life,
    write_shelf_report,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="CAPTCHA shelf-life forecasting (gap-vs-capacity scaling)"
    )
    p.add_argument("--dataset", default="librispeech",
                   choices=["librispeech", "wsj"])
    p.add_argument("--data-root", type=Path,
                   default=Path("data/raw/LibriSpeech/test-clean"))
    p.add_argument("--subset", default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-utterances", type=int, default=None,
                   help="cap utterances (default: full subset, 70)")
    p.add_argument("--margins", default=",".join(f"{m:g}" for m in DEFAULT_MARGINS_DB),
                   help="comma-separated λ grid in dB above threshold")
    p.add_argument("--w-break", type=float, default=DEFAULT_W_BREAK,
                   help="security threshold: broken when attacked WER <= this")
    p.add_argument("--doubling-months", default="6,12,24",
                   help="capacity-doubling scenarios (months per doubling)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("results/shelf_life"))
    p.add_argument("--analysis-only", action="store_true")
    args = p.parse_args(argv)

    margins = tuple(float(x) for x in args.margins.split(",") if x.strip())
    doubling = tuple(float(x) for x in args.doubling_months.split(",")
                     if x.strip()) or DEFAULT_DOUBLING_MONTHS

    if not args.analysis_only:
        run_shelf_life(
            out_dir=args.out,
            dataset=args.dataset,
            data_root=args.data_root,
            subset=args.subset,
            seed=args.seed,
            max_utterances=args.max_utterances,
            margins_db=margins,
            workers=args.workers,
        )

    rows_path = args.out / "rows.csv"
    if not rows_path.exists():
        print(f"[error] no rows at {rows_path}", file=sys.stderr)
        return 1

    rows = read_shelf_rows(rows_path)
    summary = evaluate_shelf_life(
        rows, w_break=args.w_break, doubling_months=doubling,
        n_boot=args.n_boot, seed=args.seed,
    )
    report = write_shelf_report(summary, args.out)
    print(f"[report] {report}")
    print(f"[hsr] {summary['hsr_label']}")
    for cid, f in sorted(summary["fits"].items()):
        c = f["break_capacity_m_params"]
        c_txt = "no break" if c is None else f"{c:,.0f} M params"
        print(f"[fit] {cid}: b={f['fit']['b']:.2f} "
              f"R2={f['fit']['r2']:.3f} -> C* = {c_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
