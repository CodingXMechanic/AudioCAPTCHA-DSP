#!/usr/bin/env python
"""Generate paper-quality figures from a comparative benchmark run.

Reads ``<run>/summary.csv`` (+ manifests) and writes PNG (300 dpi) + SVG
figures to an output directory:

1. ``fig1_pareto_human_vs_asr``  — human intelligibility (STOI) vs
   cross-family attack strength (ΔWER), Pareto front, family colours.
2. ``fig2_lambda_sweep_mirror``   — base-paper λ-grid mirror: ΔWER (two
   independent ASR families) + STOI vs hearing-threshold margin, with the
   unconstrained "None" control.
3. ``fig3_cross_family_transfer`` — Whisper ΔWER vs Vosk ΔWER (transfer
   across independent ASR families; y = x = perfect transfer).
4. ``fig4_family_summary``        — per-family attack strength (mean
   cross-ΔWER ± 95 % CI) and human cost (mean STOI).

Extended set (5–16, WHAT-REMAINS §12):

5. ``fig5_taxonomy_overview``        — transform taxonomy overview.
6. ``fig6_strength_and_gap_curves``  — WER / human success / human–ASR gap vs
   transformation strength.
7. ``fig7_quality_vs_degradation``   — perceptual quality (MBSD) vs ASR
   degradation.
8. ``fig8_parameter_model_heatmaps`` — heatmaps across transform parameters
   and ASR models.
9. ``fig9_speaker_variability``      — speaker-level variability.
10. ``fig10_word_confusion``         — confusion matrix (word level).
11. ``fig11_spectrograms``           — original/transformed spectrograms.
12. ``fig12_masking_thresholds``     — masking thresholds and perturbation
    margins (Bark-scale).
13. ``fig13_metric_correlation``     — metric correlation matrix.
14. ``fig14_forest_effect_sizes``    — forest plot for effect sizes (± 95% CI).
15. ``fig15_rank_stability``         — ranking stability under bootstrap
    resampling.
16. ``fig16_shelf_life_forecast``    — novelty N11: gap-vs-capacity scaling
    (needs ``--shelf-run`` rows; skipped when absent).

Every figure carries dataset / model / transform / parameter / statistical
metadata in a footer line.

Not generated here (documented gaps — see LIMITATIONS.md): CER-vs-strength
(CER not recorded), phoneme-level vulnerability (needs forced alignment),
defense-recovery and runtime-cost plots (not captured in headline rows).

Usage
-----
python scripts/make_figures.py --run results/comparative/main \
    --out results/figures [--shelf-run results/shelf_life]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY_COLORS = {
    "A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#C44E52",
    "E": "#8172B3", "F": "#937860", "G": "#DA8BC3", "H": "#8C8C8C",
    "-": "#000000",
}
FAMILY_NAMES = {
    "A": "A: baseline/level", "B": "B: temporal", "C": "C: multirate",
    "D": "D: spectral", "E": "E: additive/interference", "F": "F: psychoacoustic",
    "G": "G: adversarial", "H": "H: channel/codec", "-": "control",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _macro_delta(delta_by_engine: dict[str, float]) -> float:
    """Family-macro mean over engines — the headline cross-ΔWER rule.

    Mirrors ``experiments.comparative._macro_family_delta``: engines of one
    lineage (Whisper tiny + small) are averaged first, then the family
    means, so each independent ASR family weighs exactly once.
    """
    from audiocaptcha_dsp.experiments.comparative import asr_family_groups

    if not delta_by_engine:
        return float("nan")
    family_means = [
        np.nanmean([delta_by_engine[e] for e in members])
        for members in asr_family_groups(list(delta_by_engine)).values()
    ]
    return float(np.nanmean(family_means))


def _save(fig: plt.Figure, out: Path, name: str) -> list[Path]:
    paths = []
    for ext in ("png", "svg", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def fig_pareto(df: pd.DataFrame, out: Path, meta: dict) -> list[Path]:
    ranked = df[(df.kind != "original") & df.stoi_mean.notna()
                & df.cross_delta_wer.notna()].copy()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for fam, g in ranked.groupby("family"):
        ax.scatter(
            g.stoi_mean, g.cross_delta_wer, s=42, alpha=0.8,
            color=FAMILY_COLORS.get(fam, "#999999"),
            label=FAMILY_NAMES.get(fam, fam), edgecolors="white",
            linewidths=0.4,
        )

    # Pareto front (already-computed flag)
    front = ranked[ranked.pareto == True].sort_values("stoi_mean")  # noqa: E712
    if len(front) > 1:
        ax.plot(front.stoi_mean, front.cross_delta_wer, ls="-", lw=1.2,
                color="black", alpha=0.7, marker="none",
                label="Pareto front")

    # annotate only informative points: the strongest attacks + the best
    # stealthy attack (max strength among near-transparent samples)
    short = lambda s: (s.replace("psychoacoustic.", "psy.")
                        .replace("adversarial.", "adv.")
                        .replace("spectral.", "spec.")
                        .replace("temporal.", "temp.")
                        .replace("noise.", "n.")
                        .replace("channel.", "ch.")
                        .replace("multirate.", "mr."))
    annotated = pd.concat([
        ranked.nlargest(3, "cross_delta_wer"),
        ranked[ranked.stoi_mean >= 0.95].nlargest(1, "cross_delta_wer"),
    ]).drop_duplicates("condition_id")
    annotated = annotated.sort_values("cross_delta_wer", ascending=False)
    # reserve headroom so stacked labels cannot collide with the axes top
    _x0, _x1 = ax.get_xlim()
    _y0, _y1 = ax.get_ylim()
    ax.set_ylim(_y0, _y1 + 0.30 * (_y1 - _y0))
    for k, (_, r) in enumerate(annotated.iterrows()):
        # keep labels inside the canvas: right-anchor them when the point
        # sits in the right quarter of the x-range; one vertical layer per
        # label so near-equal ΔWER conditions never overlap
        flip = r.stoi_mean > (_x1 - 0.25 * (_x1 - _x0))
        ax.annotate(
            short(r.condition_id),
            (r.stoi_mean, r.cross_delta_wer),
            textcoords="offset points",
            xytext=(-6 if flip else 6, 5 + 12 * k), fontsize=7,
            ha="right" if flip else "left",
        )

    ax.set_xlabel("Human intelligibility  →  mean STOI proxy (higher = clearer to humans)")
    ax.set_ylabel("ASR attack strength  →  cross-family $\\Delta$WER (higher = stronger)")
    ax.set_title(
        "Human intelligibility vs. ASR effectiveness\n"
        f"{meta.get('protocol', '')} — {meta.get('n', 0)} samples\n"
        f"{meta.get('engines', 'registered engines')}"
    )
    ax.legend(loc="lower left", fontsize=7.5, ncol=2,
              framealpha=0.85, edgecolor="white")
    _footer(fig, meta, "95% CI: normal approx (see fig4) | one point = one "
                       "condition")
    return _save(fig, out, "fig1_pareto_human_vs_asr")


def fig_lambda_sweep(df: pd.DataFrame, out: Path, meta: dict) -> list[Path]:
    sweeps = df[df.kind == "sweep"].copy()
    if sweeps.empty:
        return []
    targets = sorted(sweeps.transform_key.unique())
    ncol = 2
    nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 3.4 * nrow), squeeze=False)
    control = df[df.condition_id.str.startswith("control.", na=False)]

    shared_h: list = []
    shared_l: list = []
    for i, (ax, key) in enumerate(zip(axes.flat, targets)):
        g = sweeps[sweeps.transform_key == key].sort_values("margin_db")
        for c in df.columns:
            if not c.endswith("_delta"):
                continue
            eng = c.replace("wer_", "").replace("_delta", "")
            ax.plot(g.margin_db, g[c], marker="o", ms=4,
                    label=f"$\\Delta$WER {eng.replace('_', ' ')}")
        ax2 = ax.twinx()
        ax2.plot(g.margin_db, g.stoi_mean, color="black", ls="--",
                 marker="s", ms=3.5, label="STOI (human axis)")
        if i == 0:  # collect the shared legend from the first panel
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            shared_h, shared_l = h1 + h2, l1 + l2
        ax2.set_ylim(0, 1.05)
        ax2.grid(False)
        ax2.set_ylabel("STOI", color="black", fontsize=9)

        if not control.empty:
            cr = control.iloc[0]
            for c in [c for c in df.columns if c.endswith("_delta")]:
                ax.axhline(cr[c], color="gray", lw=0.8, ls=":")
            ax.text(0.99, 0.10, "'None' control", transform=ax.transAxes,
                    ha="right", fontsize=7, color="gray")

        ax.set_xlabel("hearing-threshold margin $\\lambda$ (dB)")
        ax.set_ylabel("$\\Delta$WER vs. original")
        ax.set_title(key)

    for ax in axes.flat[len(targets):]:
        ax.axis("off")
    fig.suptitle(
        "$\\lambda$-sweep mirror of the base paper "
        "(Schönherr et al. 2018): attack vs. human trade-off",
        fontsize=11, y=1.03,
    )
    fig.legend(shared_h, shared_l, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    _footer(fig, meta, "λ grid: " + ", ".join(
        f"{v:g}" for v in sorted(df[df.kind == 'sweep'].margin_db.dropna()
                                 .unique())) + " dB | matched-power control")
    return _save(fig, out, "fig2_lambda_sweep_mirror")


def fig_cross_family(df: pd.DataFrame, out: Path, meta: dict) -> list[Path]:
    cols = {"whisper": "wer_whisper_tiny_delta", "vosk": "wer_vosk_small_en_delta"}
    if not all(c in df.columns for c in cols.values()):
        return []
    ranked = df[(df.kind != "original")
                & df[cols["whisper"]].notna()
                & df[cols["vosk"]].notna()]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    lo = min(ranked[cols["whisper"]].min(), ranked[cols["vosk"]].min(), 0)
    hi = max(ranked[cols["whisper"]].max(), ranked[cols["vosk"]].max(), 0.1)
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1,
            color="black", alpha=0.6, label="perfect transfer (y = x)")

    for fam, g in ranked.groupby("family"):
        ax.scatter(g[cols["whisper"]], g[cols["vosk"]], s=40, alpha=0.8,
                   color=FAMILY_COLORS.get(fam, "#999999"),
                   edgecolors="white", linewidths=0.4,
                   label=FAMILY_NAMES.get(fam, fam))

    ax.set_xlabel("Whisper tiny $\\Delta$WER")
    ax.set_ylabel("Vosk (Kaldi-lineage) $\\Delta$WER")
    ax.set_title(
        "Cross-family attack transfer\n"
        "above y = x: stronger on Whisper; below: stronger on Vosk"
    )
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    _footer(fig, meta, "scatter over attacked conditions | y = x: perfect "
                       "transfer")
    return _save(fig, out, "fig3_cross_family_transfer")


def fig_family_summary(df: pd.DataFrame, out: Path, meta: dict) -> list[Path]:
    g = df[(df.kind != "original") & df.cross_delta_wer.notna()]
    fams = sorted(g.family.unique())
    means_a, ci_a, means_h, ci_h, ns = [], [], [], [], []
    for f in fams:
        sub = g[g.family == f]
        ns.append(len(sub))
        a = sub.cross_delta_wer.to_numpy()
        means_a.append(a.mean())
        ci_a.append(
            1.96 * a.std(ddof=1) / np.sqrt(len(a))
            if len(a) > 1 else 0.0
        )
        h = sub.stoi_mean.dropna().to_numpy()
        means_h.append(h.mean() if len(h) else np.nan)
        ci_h.append(
            1.96 * h.std(ddof=1) / np.sqrt(len(h)) if len(h) > 1 else 0.0
        )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    x = np.arange(len(fams))
    colors = [FAMILY_COLORS.get(f, "#999") for f in fams]

    ax1.bar(x, means_a, yerr=ci_a, color=colors, alpha=0.85, capsize=3,
            edgecolor="white")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{f}\n(n={n})" for f, n in zip(fams, ns)],
                        fontsize=8)
    ax1.set_ylabel("mean cross-family $\\Delta$WER  ± 95 % CI")
    ax1.set_title("Attack effectiveness by family")

    ax2.bar(x, means_h, yerr=ci_h, color=colors, alpha=0.85, capsize=3,
            edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(fams, fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("mean STOI proxy")
    ax2.set_title("Human intelligibility by family")

    fig.tight_layout()
    _footer(fig, meta, "95% CI: normal approx over conditions within family "
                       "| HSR not shown (illustrative proxy = STOI axis)")
    return _save(fig, out, "fig4_family_summary")


# ===========================================================================
# Extended figure set — WHAT-REMAINS §12 (fig5..fig16)
# ===========================================================================
def _footer(fig: plt.Figure, meta: dict, extra: str = "") -> None:
    """Dataset / model / parameter / statistical metadata on every figure."""
    bits: list[str] = []
    if meta.get("protocol"):
        bits.append(f"dataset: {meta['protocol']}")
    if meta.get("n"):
        bits.append(f"n={meta['n']}")
    if meta.get("corpus"):
        bits.append(f"corpus: {meta['corpus']}")
    bits.append(f"engines: {meta.get('engines', 'registered engines')}")
    if meta.get("seed") is not None:
        bits.append(f"seed={meta['seed']}")
    if extra:
        bits.append(extra)
    fig.text(0.005, 0.005, " | ".join(bits), fontsize=6.5, color="0.35",
             ha="left")


def _load_rows(run: Path) -> pd.DataFrame | None:
    p = Path(run) / "rows.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return None


def _short(key: str) -> str:
    return (key.replace("psychoacoustic.", "psy.").replace("adversarial.", "adv.")
            .replace("spectral.", "spec.").replace("temporal.", "temp.")
            .replace("noise.", "n.").replace("channel.", "ch.")
            .replace("multirate.", "mr.").replace("novel.", "nov.")
            .replace("baseline.", ""))


def fig_taxonomy(df: pd.DataFrame, out: Path, meta: dict) -> list[Path]:
    """fig5 — transform taxonomy overview (registry counts + run coverage)."""
    reg_counts: dict[str, int] = {}
    try:
        from audiocaptcha_dsp.transforms.registry import get_registry

        for key, spec in get_registry().items():
            fam = str(getattr(spec, "family", "-"))
            reg_counts[fam] = reg_counts.get(fam, 0) + 1
    except Exception:
        reg_counts = {}
    run_counts = df[df.kind != "original"].groupby("family").size().to_dict()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    fams = sorted(set(reg_counts) | set(run_counts))
    if not fams:
        plt.close(fig)
        return []
    x = np.arange(len(fams))
    colors = [FAMILY_COLORS.get(f, "#999") for f in fams]
    ax1.bar(x, [reg_counts.get(f, 0) for f in fams], color=colors, alpha=0.85,
            edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(fams)
    ax1.set_ylabel("# registered transforms")
    ax1.set_title("Registry: taxonomy families A–H")

    ax2.bar(x, [run_counts.get(f, 0) for f in fams], color=colors, alpha=0.85,
            edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(fams)
    ax2.set_ylabel("# evaluated conditions")
    ax2.set_title("This run: conditions per family")
    fig.tight_layout()
    _footer(fig, meta, "statistics: counts (no inference)")
    return _save(fig, out, "fig5_taxonomy_overview")


def fig_strength_curves(rows: pd.DataFrame | None, out: Path,
                        meta: dict) -> list[Path]:
    """fig6 — human success and human–ASR gap vs transformation strength λ."""
    if rows is None or rows.empty:
        return []
    r = rows[(rows.status == "ok") & (rows.kind == "sweep")].copy()
    if r.empty:
        return []
    engines = [c.replace("wer_", "") for c in rows.columns
               if c.startswith("wer_") and c != "wer_original"]
    if not engines:
        return []

    rows_out: list[dict] = []
    for (key, mb), g in r.groupby(["transform_key", "margin_db"]):
        stoi = pd.to_numeric(g.stoi_proxy, errors="coerce").dropna()
        hsr = (float(np.clip(0.6 + 0.4 * float(stoi.mean()), 0, 1))
               if len(stoi) else np.nan)
        wers = pd.concat([pd.to_numeric(g[f"wer_{e}"], errors="coerce")
                          for e in engines]).dropna()
        asr_sr = float((wers <= 0.3).mean()) if len(wers) else np.nan
        rows_out.append({
            "transform_key": key, "margin_db": mb, "hsr": hsr,
            "asr_sr": asr_sr,
            "gap": (hsr - asr_sr
                    if np.isfinite(hsr) and np.isfinite(asr_sr) else np.nan),
            "wer": float(wers.mean()) if len(wers) else np.nan,
        })
    g = pd.DataFrame(rows_out).set_index(["transform_key", "margin_db"])
    targets = sorted(r.transform_key.unique())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharex=True)
    for key in targets:
        sub = g.loc[key].sort_index()
        lbl = _short(key)
        axes[0].plot(sub.index, sub.wer, marker="o", ms=4, label=lbl)
        axes[1].plot(sub.index, sub.hsr, marker="o", ms=4, label=lbl)
        axes[2].plot(sub.index, sub.gap, marker="o", ms=4, label=lbl)
    axes[0].set_ylabel("mean WER (all engines pooled)")
    axes[0].set_title("(a) WER vs transformation strength")
    axes[1].set_ylabel("human success (STOI proxy)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("(b) Human success vs strength")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_ylabel("HAG = human − ASR success")
    axes[2].set_title("(c) Human–ASR gap curve")
    for ax in axes:
        ax.set_xlabel("hearing-threshold margin λ (dB)")
    axes[2].legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    _footer(fig, meta, "ASR success = share WER≤0.3 | HSR illustrative proxy | "
                       "95% CI: normal approx (not shown on curves)")
    return _save(fig, out, "fig6_strength_and_gap_curves")


def fig_quality_degradation(df: pd.DataFrame, out: Path,
                            meta: dict) -> list[Path]:
    """fig7 — perceptual quality (MBSD) versus ASR degradation (ΔWER)."""
    qcol = next((c for c in ("mbsd_mean", "mbsd") if c in df.columns), None)
    if qcol is None:
        return []
    g = df[(df.kind != "original") & df[qcol].notna()
           & df.cross_delta_wer.notna()]
    if g.empty:
        return []
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for fam, sub in g.groupby("family"):
        ax.scatter(sub[qcol], sub.cross_delta_wer, s=40, alpha=0.8,
                   color=FAMILY_COLORS.get(fam, "#999"), edgecolors="white",
                   linewidths=0.4, label=FAMILY_NAMES.get(fam, fam))
    ax.set_xlabel("perceptual quality damage — MBSD (higher = worse)")
    ax.set_ylabel("cross-family $\\Delta$WER (higher = stronger attack)")
    ax.set_title("Perceptual quality vs. ASR degradation")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _footer(fig, meta, "one point = one condition | correlation reported in "
                       "fig13")
    return _save(fig, out, "fig7_quality_vs_degradation")


def fig_heatmap(df: pd.DataFrame, rows: pd.DataFrame | None, out: Path,
                meta: dict) -> list[Path]:
    """fig8 — heatmaps: λ × transform (parameters) and transform × ASR model."""
    panels = []
    data1 = data2 = None
    label1 = label2 = None

    sweeps = df[df.kind == "sweep"]
    if not sweeps.empty and "margin_db" in sweeps.columns:
        piv = sweeps.pivot_table(index="margin_db", columns="transform_key",
                                 values="cross_delta_wer", aggfunc="mean")
        if piv.shape[0] >= 2 and piv.shape[1] >= 2:
            data1 = piv
            label1 = "cross-family ΔWER"

    if rows is not None and not rows.empty:
        engines = [c.replace("wer_", "") for c in rows.columns
                   if c.startswith("wer_")]
        orig = rows[rows.condition_id == "original"]
        if engines and not orig.empty:
            base = {e: pd.to_numeric(orig[f"wer_{e}"], errors="coerce").mean()
                    for e in engines}
            ok = rows[(rows.kind != "original") & (rows.status == "ok")]
            deltas = {}
            for cid, sub in ok.groupby("condition_id"):
                deltas[cid] = [
                    pd.to_numeric(sub[f"wer_{e}"], errors="coerce").mean()
                    - base.get(e, np.nan) for e in engines
                ]
            d2 = pd.DataFrame(deltas, index=engines).T
            if len(d2) > 6:
                d2 = d2.loc[d2.abs().max(axis=1).nlargest(24).index]
            if d2.shape[0] >= 2:
                data2 = d2
                label2 = "ΔWER vs original"

    if data1 is None and data2 is None:
        return []
    n = int(data1 is not None) + int(data2 is not None)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n + 1, 5.4), squeeze=False)
    axes = axes.flat
    i = 0
    if data1 is not None:
        ax = axes[i]
        im = ax.imshow(data1.to_numpy(dtype=float), aspect="auto",
                       cmap="RdYlBu_r")
        ax.set_xticks(range(len(data1.columns)))
        ax.set_xticklabels([_short(c) for c in data1.columns], rotation=45,
                           ha="right", fontsize=7)
        ax.set_yticks(range(len(data1.index)))
        ax.set_yticklabels([f"{v:g}" for v in data1.index])
        ax.set_xlabel("sweep transform (columns) — parameter λ (rows, dB)")
        ax.set_ylabel("λ margin (dB)")
        ax.set_title(f"(a) parameters × transforms: {label1}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        i += 1
    if data2 is not None:
        ax = axes[i]
        im = ax.imshow(data2.to_numpy(dtype=float), aspect="auto",
                       cmap="RdYlBu_r")
        ax.set_xticks(range(len(data2.columns)))
        ax.set_xticklabels(data2.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(data2.index)))
        ax.set_yticklabels([_short(v) for v in data2.index], fontsize=6.5)
        ax.set_ylabel("transform")
        ax.set_title(f"(b) transform × ASR model: {label2}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _footer(fig, meta, "cell values: mean over utterances")
    return _save(fig, out, "fig8_parameter_model_heatmaps")


def fig_speaker_variability(rows: pd.DataFrame | None, out: Path,
                            meta: dict) -> list[Path]:
    """fig9 — speaker-level variability (family × speaker ΔWER heatmap)."""
    if rows is None or rows.empty:
        return []
    engines = [c.replace("wer_", "") for c in rows.columns
               if c.startswith("wer_")]
    if not engines:
        return []
    orig = rows[rows.condition_id == "original"]
    if orig.empty:
        return []
    base = {u: {e: pd.to_numeric(r[f"wer_{e}"], errors="coerce")
                for e in engines}
            for u, r in orig.set_index("utt_id").iterrows()}
    ok = rows[(rows.kind != "original") & (rows.status == "ok")].copy()
    if ok.empty:
        return []

    def _row_delta(r: pd.Series) -> float:
        b = base.get(r.utt_id, {})
        vals = []
        for e in engines:
            w = pd.to_numeric(r.get(f"wer_{e}"), errors="coerce")
            bb = b.get(e, np.nan)
            if np.isfinite(w) and np.isfinite(bb):
                vals.append(w - bb)
        return float(np.mean(vals)) if vals else np.nan

    ok["delta"] = ok.apply(_row_delta, axis=1)
    pivot = ok.pivot_table(index="spk", columns="family", values="delta",
                           aggfunc="mean")
    if pivot.shape[0] < 2:
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto",
                   cmap="RdYlBu_r")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("taxonomy family")
    ax.set_ylabel("speaker")
    ax.set_title("Speaker-level variability: mean ΔWER by family × speaker")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _footer(fig, meta, "cell = mean ΔWER over that speaker's utterances")
    return _save(fig, out, "fig9_speaker_variability")


def fig_word_confusion(rows: pd.DataFrame | None, out: Path,
                       meta: dict) -> list[Path]:
    """fig10 — word-level confusion matrix (ref → hypothesis substitutions)."""
    if rows is None or rows.empty:
        return []
    import re
    from collections import Counter
    from difflib import SequenceMatcher

    engines = [c.replace("wer_", "").replace("hyp_", "")
               for c in rows.columns if c.startswith("hyp_")]
    hyp_cols = [f"hyp_{e}" for e in engines if f"hyp_{e}" in rows.columns]
    if not hyp_cols:
        return []
    tok = lambda s: re.findall(r"[a-z']+", str(s).lower())
    pairs: Counter = Counter()
    ok = rows[(rows.status == "ok") & (rows.kind != "original")]
    for _, r in ok.iterrows():
        ref = tok(r.ref_text)
        for col in hyp_cols:
            hyp = tok(r[col])
            if ref == hyp:
                continue
            sm = SequenceMatcher(None, ref, hyp, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    continue
                if tag == "replace":
                    for a, b in zip(ref[i1:i2], hyp[j1:j2]):
                        pairs[(a, b)] += 1
                elif tag == "delete":
                    for a in ref[i1:i2]:
                        pairs[(a, "∅")] += 1
                else:
                    for b in hyp[j1:j2]:
                        pairs[("∅", b)] += 1
    if not pairs:
        return []
    refs = [w for (w, _), _ in pairs.most_common(400)]
    hyps = [w for (_, w), _ in pairs.most_common(400)]
    top_r = [w for w, _ in Counter(refs).most_common(14)]
    top_h = [w for w, _ in Counter(hyps).most_common(14)]
    mat = np.zeros((len(top_r), len(top_h)))
    for (a, b), c in pairs.items():
        if a in top_r and b in top_h:
            mat[top_r.index(a), top_h.index(b)] += c
    if mat.sum() == 0:
        return []
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(top_h)))
    ax.set_xticklabels(top_h, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_r)))
    ax.set_yticklabels(top_r, fontsize=8)
    ax.set_xlabel("hypothesis word")
    ax.set_ylabel("reference word")
    ax.set_title("Word-level confusion (substitutions, deletions ∅, insertions ∅)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    fig.tight_layout()
    _footer(fig, meta, f"alignment: SequenceMatcher | pairs={int(mat.sum())} | "
                       "attacked conditions only, all engines pooled")
    return _save(fig, out, "fig10_word_confusion")


def fig_spectrograms(run: Path, out: Path, meta: dict) -> list[Path]:
    """fig11 — original vs transformed spectrograms of one utterance."""
    import json as _json

    man = Path(run) / "dataset_manifest.json"
    if not man.exists():
        return []
    try:
        import soundfile as sf

        from audiocaptcha_dsp.core.signal import Signal
        from audiocaptcha_dsp.experiments.runner import resolve_transform
    except Exception:
        return []
    m = _json.loads(man.read_text(encoding="utf-8"))
    utts = [u for u in m.get("utterances", []) if u.get("source_path")]
    if not utts:
        return []
    path = utts[0]["source_path"]
    if not Path(path).exists():
        return []
    wav, sr = sf.read(path, dtype="float64")
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=-1)
    original = Signal(waveform=np.asarray(wav), sample_rate=int(sr))
    cases = [("original (clean)", None, {})]
    for key, params in [
        ("psychoacoustic.masked_noise", {"margin_db": 0.0}),
        ("spectral.bark_masking", {}),
        ("channel.mp3", {}),
    ]:
        cases.append((key, key, params))

    panels = []
    for label, key, params in cases:
        sig = original
        if key is not None:
            try:
                sig = resolve_transform(key, dict(params))(original)
            except Exception:
                continue
        panels.append((label, sig))
    if len(panels) < 2:
        return []
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(8.6, 2.3 * len(panels)), sharex=True,
                             squeeze=False)
    for ax, (label, sig) in zip(axes.flat, panels):
        ax.specgram(sig.waveform, Fs=sig.sample_rate, NFFT=1024, noverlap=512,
                    cmap="magma")
        ax.set_ylabel("Hz", fontsize=8)
        ax.set_title(label, fontsize=9, loc="left")
    axes.flat[-1].set_xlabel("time (s)")
    fig.suptitle("Original vs transformed spectrograms", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _footer(fig, meta, "STFT: NFFT=1024, hop=512 | first utterance of manifest")
    return _save(fig, out, "fig11_spectrograms")


def fig_masking_thresholds(run: Path, out: Path, meta: dict) -> list[Path]:
    """fig12 — Bark-scale masking thresholds and perturbation margins."""
    import json as _json

    try:
        import soundfile as sf

        from audiocaptcha_dsp.psychoacoustics.thresholds import (
            absolute_threshold,
            compute_masking_threshold_db,
        )
    except Exception:
        return []
    # signal: first manifest utterance if present, else synthetic harmonics
    sig = None
    man = Path(run) / "dataset_manifest.json"
    if man.exists():
        m = _json.loads(man.read_text(encoding="utf-8"))
        utts = [u for u in m.get("utterances", []) if u.get("source_path")]
        if utts and Path(utts[0]["source_path"]).exists():
            wav, sr = sf.read(utts[0]["source_path"], dtype="float64")
            if getattr(wav, "ndim", 1) > 1:
                wav = wav.mean(axis=-1)
            sig = (np.asarray(wav), int(sr))
    if sig is None:
        t = np.arange(int(16000 * 1.5)) / 16000
        sig = (0.1 * sum(np.sin(2 * np.pi * f * t)
                         for f in (120, 240, 480, 960, 1920)), 16000)
    wav, sr = sig

    freqs = np.fft.rfftfreq(len(wav), 1 / sr)
    power = np.abs(np.fft.rfft(wav)) ** 2
    thresh_db = compute_masking_threshold_db(power, freqs, sr)
    ath = absolute_threshold(np.maximum(freqs, 20.0))
    db_spec = 10 * np.log10(np.maximum(power, 1e-20))
    db_spec -= db_spec.max()                      # relative dB spectrum
    # align threshold onto the same relative scale
    thresh_rel = thresh_db - thresh_db.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    f_ath = np.geomspace(20, 20000, 400)
    ax1.plot(f_ath, absolute_threshold(f_ath), color="black", lw=1.5,
             label="absolute threshold of hearing (ISO 226)")
    ax1.set_xscale("log")
    ax1.set_xlabel("frequency (Hz)")
    ax1.set_ylabel("dB SPL")
    ax1.set_title("(a) Absolute threshold of hearing")
    ax1.legend(fontsize=8)

    ax2.plot(freqs[1:], db_spec[1:], color="0.55", lw=0.5, alpha=0.6,
             label="signal spectrum (relative dB)")
    ax2.plot(freqs[1:], thresh_rel[1:], color="#C44E52", lw=1.4,
             label="overall masking threshold")
    for lam, ls in ((0.0, "--"), (10.0, ":"), (20.0, "-.")):
        ax2.plot(freqs[1:], thresh_rel[1:] - lam, lw=1.0, ls=ls,
                 color="#4C72B0", label=f"margin λ = {lam:g} dB")
    ax2.set_xscale("log")
    ax2.set_xlim(50, min(20000, freqs[-1]))
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("relative dB")
    ax2.set_title("(b) Masking thresholds and perturbation margins")
    ax2.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    _footer(fig, meta, "ATH: ISO 226 | threshold: ATH+simultaneous elevation | "
                       "perturbations are kept below λ")
    return _save(fig, out, "fig12_masking_thresholds")


def fig_metric_correlation(df: pd.DataFrame, out: Path,
                           meta: dict) -> list[Path]:
    """fig13 — correlation matrix of headline metrics across conditions."""
    wanted = [c for c in (
        "stoi_mean", "mbsd_mean", "snr_mean", "si_sdr_mean",
        "cross_delta_wer", "margin_db", "dur_mean",
    ) if c in df.columns]
    wanted += [c for c in df.columns
               if c.startswith("wer_") and c.endswith("_delta")
               and c not in wanted]
    if len(wanted) < 3:
        return []
    sub = df[df.kind != "original"][wanted].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr()
    if corr.isna().all().all():
        return []
    side = max(6.8, 0.85 * len(wanted) + 1.5)
    fig, ax = plt.subplots(figsize=(side, max(5.6, 0.75 * len(wanted) + 1.5)))
    im = ax.imshow(corr.to_numpy(dtype=float), cmap="coolwarm", vmin=-1,
                   vmax=1, aspect="auto")
    labels = [c.replace("_mean", "").replace("_delta", " Δ").replace("wer_", "WER ")
              for c in corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "black")
    ax.set_title("Metric correlation matrix (Pearson, per condition)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _footer(fig, meta, "Pearson r over conditions (attacked only) | "
                       "pairwise-complete observations")
    return _save(fig, out, "fig13_metric_correlation")


def fig_forest(rows: pd.DataFrame | None, out: Path,
               meta: dict) -> list[Path]:
    """fig14 — forest plot of attack effect sizes (ΔWER ± 95% CI)."""
    if rows is None or rows.empty:
        return []
    engines = [c.replace("wer_", "") for c in rows.columns
               if c.startswith("wer_")]
    if not engines:
        return []
    orig = rows[rows.condition_id == "original"]
    ok = rows[(rows.kind != "original") & (rows.status == "ok")]
    if orig.empty or ok.empty:
        return []
    base = {u: {e: pd.to_numeric(r.get(f"wer_{e}"), errors="coerce")
                for e in engines}
            for u, r in orig.set_index("utt_id").iterrows()}

    def _delta(r: pd.Series) -> float:
        b = base.get(r.utt_id)
        if b is None:
            return np.nan
        per_engine: dict[str, float] = {}
        for e in engines:
            v = pd.to_numeric(r.get(f"wer_{e}"), errors="coerce")
            bv = b.get(e, np.nan)
            if np.isfinite(v) and np.isfinite(bv):
                per_engine[e] = float(v - bv)
        if not per_engine:
            return np.nan
        return _macro_delta(per_engine)

    ok = ok.copy()
    ok["delta"] = ok.apply(_delta, axis=1)
    stats = []
    for cid, sub in ok.groupby("condition_id"):
        v = sub.delta.dropna().to_numpy()
        if v.size < 5:
            continue
        mu = v.mean()
        ci = 1.96 * v.std(ddof=1) / np.sqrt(v.size)
        fam = sub.family.iloc[0]
        stats.append((cid, fam, mu, ci, v.size))
    if not stats:
        return []
    # strongest attack first (descending mean ΔWER), keep the top 20
    stats.sort(key=lambda t: t[2], reverse=True)
    stats = stats[:20] if len(stats) > 20 else stats

    fig, ax = plt.subplots(figsize=(7.8, 0.34 * len(stats) + 1.8))
    y = np.arange(len(stats))
    ax.errorbar(
        [t[2] for t in stats], y,
        xerr=[t[3] for t in stats], fmt="o", ms=5, lw=1.2, capsize=3,
        color="#333333", ecolor="#888888",
    )
    for i, (cid, fam, mu, ci, n) in enumerate(stats):
        ax.plot(mu, y[i], "o", ms=6,
                color=FAMILY_COLORS.get(fam, "#999"))
    ax.set_yticks(y)
    ax.set_yticklabels([_short(t[0]) for t in stats], fontsize=7.5)
    ax.axvline(0, color="black", lw=0.9)
    ax.invert_yaxis()  # strongest condition on top (y = 0)
    ax.set_xlabel("effect size: ΔWER vs original ± 95% CI")
    ax.set_title("Forest plot: attack effect sizes (top conditions)")
    fig.tight_layout()
    _footer(fig, meta, "per-utterance ΔWER (family-macro over engines) | "
                       "95% CI: normal approx over utterances")
    return _save(fig, out, "fig14_forest_effect_sizes")


_FAMILY_PREFIXES = (
    ("noise.", "E"), ("temporal.", "B"), ("multirate.", "C"),
    ("spectral.", "D"), ("psychoacoustic.", "F"), ("adversarial.", "G"),
    ("novel.", "G"), ("channel.", "H"), ("control.", "-"),
)


def _family_of(condition_id: str) -> str:
    for prefix, fam in _FAMILY_PREFIXES:
        if condition_id.startswith(prefix):
            return fam
    return "-"


def fig_rank_stability(rows: pd.DataFrame | None, out: Path, meta: dict,
                       n_boot: int = 300, seed: int = 42) -> list[Path]:
    """fig15 — ranking stability under bootstrap resampling of utterances."""
    if rows is None or rows.empty:
        return []
    engines = [c.replace("wer_", "") for c in rows.columns
               if c.startswith("wer_")]
    if not engines:
        return []
    orig = rows[rows.condition_id == "original"]
    ok = rows[(rows.kind != "original") & (rows.status == "ok")]
    if orig.empty or ok.empty:
        return []
    base = {u: {e: pd.to_numeric(r.get(f"wer_{e}"), errors="coerce")
                for e in engines}
            for u, r in orig.set_index("utt_id").iterrows()}
    utts = sorted(set(ok.utt_id) & set(base))
    if len(utts) < 10:
        return []

    # (condition, utt) → delta (family-macro over engines, as in summary)
    deltas: dict[str, dict[str, float]] = {}
    for _, r in ok.iterrows():
        b = base.get(r.utt_id)
        if b is None:
            continue
        per_engine: dict[str, float] = {}
        for e in engines:
            v = pd.to_numeric(r.get(f"wer_{e}"), errors="coerce")
            bv = b.get(e, np.nan)
            if np.isfinite(v) and np.isfinite(bv):
                per_engine[e] = float(v - bv)
        if per_engine:
            deltas.setdefault(r.condition_id, {})[r.utt_id] = _macro_delta(
                per_engine
            )
    conds = [c for c, d in deltas.items() if len(d) >= 0.8 * len(utts)]
    if len(conds) < 5:
        return []

    def _rank(mat: np.ndarray) -> np.ndarray:
        return np.argsort(-np.nanmean(mat, axis=1), kind="stable")

    mat = np.array([[deltas[c].get(u, np.nan) for u in utts]
                    for c in conds])
    full_rank = np.empty(len(conds), dtype=int)
    full_rank[_rank(mat)] = np.arange(1, len(conds) + 1)  # 1-based ranks

    rng = np.random.default_rng(seed)
    top_k = min(10, len(conds))
    top_ids = [conds[i] for i in np.argsort(full_rank)[:top_k]]
    rank_samples = {c: [] for c in top_ids}
    for _ in range(n_boot):
        idx = rng.integers(0, len(utts), size=len(utts))
        sub = mat[:, idx]
        r = np.empty(len(conds), dtype=int)
        r[_rank(sub)] = np.arange(1, len(conds) + 1)  # 1-based ranks
        for c in top_ids:
            rank_samples[c].append(int(r[conds.index(c)]))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ys = np.arange(len(top_ids))
    for i, c in enumerate(top_ids):
        arr = np.asarray(rank_samples[c], dtype=float)
        lo, hi = np.percentile(arr, [5, 95])
        med = np.median(arr)
        ax.plot([lo, hi], [i, i], color="0.55", lw=4, solid_capstyle="round")
        ax.plot(med, i, "o", ms=6,
                color=FAMILY_COLORS.get(_family_of(c), "#999"))
        ax.plot(full_rank[conds.index(c)], i, "k|", ms=14, mew=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([_short(c) for c in top_ids], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("rank by mean ΔWER (1 = strongest)")
    ax.set_title(f"Ranking stability under bootstrap "
                 f"({n_boot} resamples of {len(utts)} utterances)")
    ax.text(0.99, 0.97, "bar = 5–95% rank | ● median | | full-data rank",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color="0.35")
    fig.tight_layout()
    _footer(fig, meta, f"bootstrap: {n_boot} resamples, seed={seed} | "
                       "paired over utterances")
    return _save(fig, out, "fig15_rank_stability")


def fig_shelf_life(shelf_run: Path, out: Path, meta: dict) -> list[Path]:
    """fig16 — novelty N11: gap-vs-capacity scaling and break forecast."""
    shelf_run = Path(shelf_run)
    rows_p = shelf_run / "rows.csv"
    if not rows_p.exists():
        return []
    try:
        from audiocaptcha_dsp.experiments.shelf_life import (
            CAPACITY_LADDER,
            DEFAULT_W_BREAK,
            ORIGINAL_ID,
            PARAMS_BY_MODEL,
            compute_capacity_points,
            paired_bootstrap_scaling,
            read_shelf_rows,
        )
    except Exception:
        return []
    srows = read_shelf_rows(rows_p)
    ok = [r for r in srows if r.get("status") == "ok"]
    if not ok:
        return []

    models = [m["model_key"] for m in CAPACITY_LADDER]
    x = np.array([m["params_m"] for m in CAPACITY_LADDER])
    cond_ids = sorted({r["condition_id"] for r in ok
                       if r["condition_id"] != ORIGINAL_ID})
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.4),
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(cond_ids), 1)))
    for ci, cid in enumerate(cond_ids):
        sub = [r for r in ok if r["condition_id"] == cid]
        per_model = {}
        for mk in models:
            vals = [float(r["wer"]) for r in sub
                    if r["model_key"] == mk and r.get("wer") not in (None, "")]
            if vals:
                per_model[mk] = np.asarray(vals, dtype=float)
        if len(per_model) < 2:
            continue
        try:
            keep = np.all(np.isfinite(np.vstack(
                [per_model[m] for m in sorted(per_model,
                                              key=PARAMS_BY_MODEL.get)])), axis=0)
        except ValueError:
            continue
        vectors = {m: v[keep] for m, v in per_model.items()}
        xv = np.array([PARAMS_BY_MODEL[m] for m in sorted(vectors,
                                                          key=PARAMS_BY_MODEL.get)])
        try:
            boot = paired_bootstrap_scaling(xv, vectors, n_boot=400, seed=42,
                                            w_break=DEFAULT_W_BREAK)
        except ValueError:
            continue
        col = palette[ci]
        ax1.plot(xv, boot.fit.predict(xv), color=col, lw=1.4,
                 label=_short(cid))
        ax1.scatter(xv, [np.mean(vectors[m]) for m in sorted(
            vectors, key=PARAMS_BY_MODEL.get)], color=col, s=28, zorder=3)
        if boot.grid.size and np.all(np.isfinite(boot.grid_lo)):
            ax1.fill_between(boot.grid, boot.grid_lo, boot.grid_hi,
                             color=col, alpha=0.15, lw=0)
        c_break = boot.fit.break_capacity(DEFAULT_W_BREAK)
        if c_break:
            ax1.scatter([c_break], [DEFAULT_W_BREAK], marker="*", s=90,
                        color=col, edgecolor="black", linewidth=0.5, zorder=4)
    ax1.set_xscale("log")
    ax1.axhline(DEFAULT_W_BREAK, color="black", ls="--", lw=1,
                label=f"break threshold W={DEFAULT_W_BREAK:g}")
    ax1.set_xlabel("ASR capacity C (M parameters, log scale)")
    ax1.set_ylabel("WER on attacked audio")
    ax1.set_title("(a) Scaling law fit + bootstrap band (400 resamples)")
    ax1.legend(fontsize=6.5, loc="upper right")

    pts = compute_capacity_points(ok, wer_threshold=DEFAULT_W_BREAK)
    if pts:
        xs = [p.params_m for p in pts]
        ax2.plot(xs, [p.gap for p in pts], marker="o", lw=1.6,
                 label="gap = HSR − ASR success")
        ax2.plot(xs, [p.asr_sr for p in pts], marker="s", lw=1.2,
                 label="ASR success rate")
        ax2.plot(xs, [p.hsr for p in pts], marker="^", lw=1.2, ls="--",
                 label="human proxy (HSR)")
        for p in pts:
            ax2.annotate(f"{p.model_key.replace('whisper_', '')}",
                         (p.params_m, p.gap), textcoords="offset points",
                         xytext=(5, -10), fontsize=7)
    ax2.set_xscale("log")
    # extra headroom at the bottom so the legend never sits on the curves
    ax2.set_ylim(-0.30, 1.05)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xlabel("ASR capacity C (M parameters)")
    ax2.set_ylabel("rate")
    ax2.set_title("(b) Human–ASR gap along the ladder")
    ax2.legend(fontsize=7, loc="lower center", ncol=2, framealpha=0.9)
    fig.suptitle("Shelf-life forecasting (novelty N11): human–ASR gap vs "
                 "attacker capacity", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _footer(fig, meta, "HSR illustrative proxy (no human study) | "
                       "fit: log-log OLS over 3 ladder points + paired bootstrap")
    return _save(fig, out, "fig16_shelf_life_forecast")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run", type=Path,
                   default=Path("results/comparative/main"))
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    p.add_argument("--shelf-run", type=Path,
                   default=Path("results/shelf_life"),
                   help="shelf-life pass (novelty N11); skipped if absent")
    args = p.parse_args(argv)

    summary_path = args.run / "summary.csv"
    if not summary_path.exists():
        print(f"missing {summary_path} — run the benchmark first")
        return 1
    df = pd.read_csv(summary_path)
    df["pareto"] = df["pareto"].fillna(False)

    meta: dict = {"protocol": "", "n": 0, "corpus": "", "seed": None,
                  "engines": "registered engines"}
    import json
    manifest = args.run / "dataset_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        meta.update({
            "protocol": m.get("protocol", ""), "n": m.get("n_samples", 0),
            "corpus": m.get("corpus", ""),
        })
    run_man = args.run / "run_manifest.json"
    if run_man.exists():
        rm = json.loads(run_man.read_text(encoding="utf-8"))
        args_dump = rm.get("args", {})
        if args_dump.get("seed") is not None:
            meta["seed"] = args_dump["seed"]
        if args_dump.get("engines"):
            meta["engines"] = " + ".join(
                str(e).replace("_", " ") for e in args_dump["engines"])

    args.out.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.run)

    written: list[Path] = []
    # figs 1–4 (original set)
    written += fig_pareto(df, args.out, meta)
    written += fig_lambda_sweep(df, args.out, meta)
    written += fig_cross_family(df, args.out, meta)
    written += fig_family_summary(df, args.out, meta)
    # figs 5–16 (WHAT-REMAINS §12 extended set)
    written += fig_taxonomy(df, args.out, meta)
    written += fig_strength_curves(rows, args.out, meta)
    written += fig_quality_degradation(df, args.out, meta)
    written += fig_heatmap(df, rows, args.out, meta)
    written += fig_speaker_variability(rows, args.out, meta)
    written += fig_word_confusion(rows, args.out, meta)
    written += fig_spectrograms(args.run, args.out, meta)
    written += fig_masking_thresholds(args.run, args.out, meta)
    written += fig_metric_correlation(df, args.out, meta)
    written += fig_forest(rows, args.out, meta)
    written += fig_rank_stability(rows, args.out, meta)
    written += fig_shelf_life(args.shelf_run, args.out, meta)

    for w in written:
        print(f"wrote {w}")
    if not written:
        print("no figures produced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
