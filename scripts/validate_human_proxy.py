#!/usr/bin/env python
"""Validate the HSR human proxy against real listener scores (TMHINT-QI).

Layer 2 of the human-axis validation package: correlates
``compute_stoi_proxy`` -- the exact metric behind the HSR formula
``hsr = clip(0.6 + 0.4 * stoi, 0, 1)`` -- with per-utterance *mean human
word-intelligibility ratings* from the public TMHINT-QI corpus
(Chen & Tsao, "InQSS", Interspeech 2022; 226 listeners, Mandarin HINT
sentences, noise + speech-enhancement degradations).

Scope and non-claims:
  * Validation only: it demonstrates that the proxy tracks real listener
    judgments. It never produces human scores for our own conditions and
    its numbers are never mixed with benchmark/condition results.
  * Mandarin content, remote (web) listening test, roughly 1.6 raters per
    file -- stated as caveats wherever the numbers are reported.
  * Rows whose ``method`` is ``None`` (pretest) or ``clean`` (reference
    rated alone) are excluded: STOI(clean, clean) == 1 is a tautology.

Outputs (``results/validation/``):
  * ``human_proxy_per_file.csv`` -- per-utterance scores, both STOI variants
  * ``human_proxy_summary.csv``  -- Pearson/Spearman per split and metric

Usage::

    python scripts/validate_human_proxy.py --split test
    python scripts/validate_human_proxy.py --split all --jobs 4
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from audiocaptcha_dsp.evaluation.metrics import compute_stoi_proxy
from audiocaptcha_dsp.io.audio import load_audio

SR = 16000
DEGRADED_METHODS = ("Noisy", "MMSE", "FCN", "DDAE", "KLT", "Trans")


def find_wav(root: Path, split: str, file_name: str) -> Path | None:
    for folder in (split, "test", "train"):
        p = root / folder / f"{file_name}.wav"
        if p.exists():
            return p
    return None


def resolve_clean_ref(root: Path, split: str, uttr: str) -> Path | None:
    """Clean reference for an utterance: ``clean_<uttr>.wav`` (test) or
    ``<uttr>.wav`` fallback, searched in the utterance's own folder first."""
    for folder in (split, "test", "train"):
        for name in (f"clean_{uttr}.wav", f"{uttr}.wav"):
            p = root / folder / name
            if p.exists():
                return p
    return None


def score_pair(job: tuple[str, str]) -> tuple[float, float]:
    """(clean_path, degraded_path) -> (stoi_proxy, stoi_reference)."""
    clean_path, deg_path = job
    orig = np.asarray(load_audio(clean_path, target_sr=SR).to_mono().waveform)
    proc = np.asarray(load_audio(deg_path, target_sr=SR).to_mono().waveform)
    orig, proc = orig.reshape(-1), proc.reshape(-1)
    n = min(len(orig), len(proc))
    orig, proc = orig[:n], proc[:n]
    proxy = compute_stoi_proxy(orig, proc, sr=SR)
    ref = float("nan")
    try:
        from pystoi import stoi as _stoi

        ref = float(_stoi(orig, proc, SR, extended=False))
    except Exception:  # pragma: no cover - reference metric optional
        pass
    return float(proxy), ref


def _corr_block(
    frame: pd.DataFrame, metric: str, target: str, split: str, out: list[dict]
) -> None:
    sub = frame[[metric, target]].dropna()
    if len(sub) < 3 or sub[metric].std() == 0 or sub[target].std() == 0:
        return
    r, rp = pearsonr(sub[metric], sub[target])
    rho, sp = spearmanr(sub[metric], sub[target])
    out.append(
        {
            "split": split,
            "metric": metric,
            "target": target,
            "n": len(sub),
            "pearson_r": round(float(r), 4),
            "pearson_p": float(rp),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(sp),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/references/tmhintqi/TMHINTQI"),
        help="Extracted TMHINTQI directory",
    )
    parser.add_argument(
        "--split",
        choices=("test", "train", "all"),
        default="test",
        help="Which files to score (summary always stratifies test/train/all)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/validation"), help="Output dir"
    )
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes")
    parser.add_argument(
        "--limit", type=int, default=0, help="Debug: cap unique files scored"
    )
    args = parser.parse_args()

    root: Path = args.data
    raw_path = root / "raw_data.csv"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found", file=sys.stderr)
        return 1

    raw = pd.read_csv(raw_path)
    n_rows = len(raw)
    raw = raw[raw["method"].isin(DEGRADED_METHODS)].copy()
    print(
        f"rating rows: {n_rows} total -> {len(raw)} degraded "
        f"({n_rows - len(raw)} pretest/reference excluded)"
    )

    human = (
        raw.groupby("file_name")
        .agg(
            n_raters=("idx", "nunique"),
            human_intell=("intelligibility_score", "mean"),
            human_quality=("quality_score", "mean"),
            method=("method", "first"),
            snr=("snr", "first"),
            noise=("noise", "first"),
            uttr=("uttr", "first"),
        )
        .reset_index()
    )

    # Resolve paths and split membership.
    resolved = []
    for row in human.itertuples(index=False):
        split = "test" if (root / "test" / f"{row.file_name}.wav").exists() else (
            "train" if (root / "train" / f"{row.file_name}.wav").exists() else None
        )
        if split is None:
            continue
        if args.split != "all" and split != args.split:
            continue
        deg = root / split / f"{row.file_name}.wav"
        clean = resolve_clean_ref(root, split, row.uttr)
        if clean is None:
            continue
        resolved.append((row.file_name, split, str(clean), str(deg)))
    if args.limit:
        resolved = resolved[: args.limit]
    print(f"files to score: {len(resolved)} (split={args.split})")

    jobs = [(c, d) for _, _, c, d in resolved]
    t0 = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            scores = list(pool.map(score_pair, jobs, chunksize=32))
    else:
        scores = []
        for i, job in enumerate(jobs, 1):
            scores.append(score_pair(job))
            if i % 200 == 0:
                print(
                    f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    print(f"scored {len(scores)} pairs in {time.time() - t0:.0f}s")

    scored = pd.DataFrame(
        {
            "file_name": [f for f, _, _, _ in resolved],
            "split": [s for _, s, _, _ in resolved],
            "stoi_proxy": [s[0] for s in scores],
            "stoi_ref": [s[1] for s in scores],
        }
    )
    # Attach labels/aggregates computed earlier.
    labels = human.set_index("file_name")
    scored = scored.join(labels, on="file_name")
    # HSR: the published proxy formula (no clipping reachable: stoi in [0,1]).
    scored["hsr"] = np.clip(0.6 + 0.4 * scored["stoi_proxy"], 0.0, 1.0)
    scored["human_intell_norm"] = scored["human_intell"] / 10.0

    args.out.mkdir(parents=True, exist_ok=True)
    per_file = args.out / "human_proxy_per_file.csv"
    scored.to_csv(per_file, index=False)

    summary: list[dict] = []
    subsets = {"all": scored}
    for sp in scored["split"].unique():
        subsets[sp] = scored[scored["split"] == sp]
    for split_name, frame in subsets.items():
        for metric in ("stoi_proxy", "stoi_ref", "hsr"):
            _corr_block(frame, metric, "human_intell", split_name, summary)
        _corr_block(frame, "stoi_proxy", "human_quality", split_name, summary)
        _corr_block(frame, "stoi_proxy", "stoi_ref", split_name, summary)

    summary_df = pd.DataFrame(summary)
    summary_path = args.out / "human_proxy_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    with pd.option_context("display.width", 160, "display.max_rows", 50):
        print(summary_df.to_string(index=False))

    primary = summary_df[
        (summary_df["split"] == "test")
        & (summary_df["metric"] == "stoi_proxy")
        & (summary_df["target"] == "human_intell")
    ]
    if not primary.empty:
        p = primary.iloc[0]
        print(
            f"\nPRIMARY (test split, proxy vs human intelligibility): "
            f"r={p['pearson_r']} (p={p['pearson_p']:.2e}), "
            f"rho={p['spearman_rho']} (p={p['spearman_p']:.2e}), n={p['n']}"
        )
    print(f"\nwrote {per_file}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
