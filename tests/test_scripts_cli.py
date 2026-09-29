"""Phase-13 CLI tests: script entry points run end-to-end on synthetic data.

These are the lightweight CLI checks — they need **no** ASR weights, no
datasets and no network (CI-safe). A synthetic benchmark run (rows.csv +
summary.csv + manifests) and a synthetic shelf-life pass are built in tmp
dirs and fed to the real scripts via subprocess.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


# --------------------------------------------------------------------------
# fixture builders
# --------------------------------------------------------------------------
ROW_FIELDS = [
    "condition_id", "kind", "transform_key", "family", "params_json",
    "margin_db", "utt_id", "spk", "dur_s", "status", "error",
    "snr_db", "stoi_proxy", "mbsd", "si_sdr_db", "rms_ratio", "ref_text",
    "wer_whisper_tiny", "wer_whisper_small",
    "wer_vosk_small_en", "wer_wav2vec2_base",
    "hyp_whisper_tiny", "hyp_whisper_small",
    "hyp_vosk_small_en", "hyp_wav2vec2_base",
]

SHELF_ROW_FIELDS = [
    "condition_id", "kind", "transform_key", "family", "params_json",
    "margin_db", "utt_id", "spk", "dur_s", "status", "error",
    "ref_text", "stoi_proxy", "snr_db",
    "model_key", "params_m", "wer", "hyp",
]

SUMMARY_COLS = [
    "condition_id", "kind", "transform_key", "family", "margin_db",
    "n_ok", "n_err", "stoi_mean", "stoi_std", "snr_mean", "mbsd_mean",
    "si_sdr_mean", "wer_whisper_tiny_mean", "wer_whisper_tiny_delta",
    "wer_whisper_small_mean", "wer_whisper_small_delta",
    "wer_vosk_small_en_mean", "wer_vosk_small_en_delta",
    "wer_wav2vec2_base_mean", "wer_wav2vec2_base_delta",
    "cross_delta_wer", "hsr", "asr_sr", "hag", "css",
    "human_rank", "attack_rank", "gap_rank", "quality_rank",
    "pareto", "hsr_label",
]


def _conditions() -> list[tuple[str, str, str, str, str]]:
    """(condition_id, kind, transform_key, family, margin_db)."""
    conds = [("original", "original", "baseline.identity", "-", "")]
    for t, fam in (("psychoacoustic.masked_noise", "F"),
                   ("psychoacoustic.bark_perturbation", "F")):
        for m in (0, 5, 10, 20):
            conds.append((f"{t}#margin={m}", "sweep", t, fam, str(m)))
    conds += [
        ("control.no_hearing_threshold", "control", "psychoacoustic.masked_noise",
         "F", ""),
        ("noise.white", "transform", "noise.white", "E", ""),
        ("temporal.jitter", "transform", "temporal.jitter", "B", ""),
        ("channel.mp3", "transform", "channel.mp3", "H", ""),
        ("adversarial.black_box", "transform", "adversarial.black_box", "G", ""),
        ("spectral.bark_masking", "transform", "spectral.bark_masking", "D", ""),
    ]
    return conds


def _make_run(run_dir: Path, n_utt: int = 12) -> Path:
    import numpy as np

    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    conds = _conditions()
    speakers = [f"spk{i}" for i in range(4)]
    utts = [f"utt{i:02d}" for i in range(n_utt)]

    rows = []
    for ci, (cid, kind, key, fam, mb) in enumerate(conds):
        strength = 0.0 if kind == "original" else 0.04 + 0.06 * (ci % 5)
        for u in utts:
            stoi = 0.97 if kind == "original" else float(
                np.clip(0.94 - 0.05 * (ci % 4), 0.5, 1.0))
            w_w = 0.06 + (strength if kind != "original" else 0) + \
                0.01 * rng.standard_normal()
            w_s = 0.04 + (0.5 * strength if kind != "original" else 0) + \
                0.01 * rng.standard_normal()
            w_v = 0.13 + (0.8 * strength if kind != "original" else 0) + \
                0.02 * rng.standard_normal()
            w_x = 0.05 + (0.7 * strength if kind != "original" else 0) + \
                0.01 * rng.standard_normal()
            hyp_w = "hello world" if w_w < 0.1 else "hallo word"
            hyp_s = "hello world" if w_s < 0.1 else "hallo word"
            hyp_v = "hello world" if w_v < 0.15 else "help world"
            hyp_x = "HELLO WORLD" if w_x < 0.1 else "HALLO WORD"
            rows.append({
                "condition_id": cid, "kind": kind, "transform_key": key,
                "family": fam, "params_json": "{}", "margin_db": mb,
                "utt_id": u, "spk": speakers[hash(u) % len(speakers)],
                "dur_s": "2.0", "status": "ok", "error": "",
                "snr_db": "20", "stoi_proxy": f"{stoi:.3f}",
                "mbsd": f"{0.1 + 0.05 * (ci % 3):.3f}",
                "si_sdr_db": "15", "rms_ratio": "1.0",
                "ref_text": "hello world",
                "wer_whisper_tiny": f"{max(w_w, 0.0):.3f}",
                "wer_whisper_small": f"{max(w_s, 0.0):.3f}",
                "wer_vosk_small_en": f"{max(w_v, 0.0):.3f}",
                "wer_wav2vec2_base": f"{max(w_x, 0.0):.3f}",
                "hyp_whisper_tiny": hyp_w, "hyp_whisper_small": hyp_s,
                "hyp_vosk_small_en": hyp_v, "hyp_wav2vec2_base": hyp_x,
            })
    with (run_dir / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)

    # summary.csv
    import numpy as _np

    def _rank(vals, reverse=True):
        order = sorted(range(len(vals)), key=lambda i: vals[i],
                       reverse=reverse)
        out = [0] * len(vals)
        for pos, idx in enumerate(order, start=1):
            out[idx] = pos
        return out

    srows = []
    for cid, kind, key, fam, mb in conds:
        sub = [r for r in rows if r["condition_id"] == cid]
        w_w = _np.array([float(r["wer_whisper_tiny"]) for r in sub])
        w_s = _np.array([float(r["wer_whisper_small"]) for r in sub])
        w_v = _np.array([float(r["wer_vosk_small_en"]) for r in sub])
        w_x = _np.array([float(r["wer_wav2vec2_base"]) for r in sub])
        st = _np.array([float(r["stoi_proxy"]) for r in sub])
        orig = [r for r in rows if r["condition_id"] == "original"]
        base_w = _np.array([float(r["wer_whisper_tiny"]) for r in orig])
        base_s = _np.array([float(r["wer_whisper_small"]) for r in orig])
        base_v = _np.array([float(r["wer_vosk_small_en"]) for r in orig])
        base_x = _np.array([float(r["wer_wav2vec2_base"]) for r in orig])
        d_w = float(w_w.mean() - base_w.mean())
        d_s = float(w_s.mean() - base_s.mean())
        d_v = float(w_v.mean() - base_v.mean())
        d_x = float(w_x.mean() - base_x.mean())
        # headline metric: macro-average over ASR families (whisper, kaldi,
        # ssl) — matches comparative.aggregate_results
        cross = ((d_w + d_s) / 2 + d_v + d_x) / 3
        srows.append({
            "condition_id": cid, "kind": kind, "transform_key": key,
            "family": fam, "margin_db": mb, "n_ok": len(sub), "n_err": 0,
            "stoi_mean": f"{st.mean():.3f}", "stoi_std": f"{st.std():.3f}",
            "snr_mean": "20", "mbsd_mean": f"{0.12:.3f}",
            "si_sdr_mean": "15",
            "wer_whisper_tiny_mean": f"{w_w.mean():.3f}",
            "wer_whisper_tiny_delta": f"{d_w:.3f}",
            "wer_whisper_small_mean": f"{w_s.mean():.3f}",
            "wer_whisper_small_delta": f"{d_s:.3f}",
            "wer_vosk_small_en_mean": f"{w_v.mean():.3f}",
            "wer_vosk_small_en_delta": f"{d_v:.3f}",
            "wer_wav2vec2_base_mean": f"{w_x.mean():.3f}",
            "wer_wav2vec2_base_delta": f"{d_x:.3f}",
            "cross_delta_wer": f"{cross:.3f}",
            "hsr": f"{min(0.6 + 0.4 * st.mean(), 1.0):.3f}",
            "asr_sr": "0.8", "hag": "0.1", "css": "0.5",
            "human_rank": "", "attack_rank": "", "gap_rank": "",
            "quality_rank": "", "pareto": "True" if kind != "original" else "",
            "hsr_label": "illustrative (STOI-derived proxy; no human study "
                         "conducted)",
        })
    attacked = [i for i, r in enumerate(srows) if r["kind"] != "original"]
    for pos, idx in enumerate(sorted(
            attacked, key=lambda i: -float(srows[i]["cross_delta_wer"])),
            start=1):
        srows[idx]["attack_rank"] = str(pos)
    for pos, idx in enumerate(sorted(
            attacked, key=lambda i: -float(srows[i]["stoi_mean"])),
            start=1):
        srows[idx]["human_rank"] = str(pos)
    for pos, idx in enumerate(attacked):
        srows[idx]["gap_rank"] = str(pos + 1)
        srows[idx]["quality_rank"] = str(pos + 1)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(srows)

    (run_dir / "dataset_manifest.json").write_text(json.dumps({
        "protocol": "synthetic fixture (CLI test)", "n_samples": n_utt,
        "corpus": "none", "utterances": [], "created": "2026-01-01",
    }), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "script": "test_fixture",
        "args": {"seed": 42, "engines": [
            "whisper_tiny", "whisper_small", "vosk_small_en",
            "wav2vec2_base",
        ]},
        "created": "2026-01-01",
    }), encoding="utf-8")
    return run_dir


def _make_shelf_run(run_dir: Path, n_pairs: int = 14) -> Path:
    import numpy as np

    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    ladder = [("whisper_tiny", 39.0), ("whisper_base", 74.0),
              ("whisper_small", 244.0)]
    conds = [
        ("original", "original", "baseline.identity", "-", ""),
        ("psychoacoustic.masked_noise#margin=0", "sweep",
         "psychoacoustic.masked_noise", "F", "0"),
        ("control.no_hearing_threshold", "control",
         "psychoacoustic.masked_noise", "F", ""),
    ]
    rows = []
    for cid, kind, key, fam, mb in conds:
        for p in range(n_pairs):
            for mk, params in ladder:
                if kind == "original":
                    w = 0.07 * (39.0 / params) ** 0.2
                elif kind == "sweep":
                    w = 40.0 * params ** (-1.1)
                else:
                    w = 1.2
                w = float(np.clip(w + 0.01 * rng.normal(), 0.01, 3.0))
                rows.append({
                    "condition_id": cid, "kind": kind, "transform_key": key,
                    "family": fam, "params_json": "{}", "margin_db": mb,
                    "utt_id": f"u{p:03d}", "spk": "spk1", "dur_s": "2.0",
                    "status": "ok", "error": "", "ref_text": "hello world",
                    "stoi_proxy": "0.90", "snr_db": "20",
                    "model_key": mk, "params_m": f"{params:g}",
                    "wer": f"{w:.3f}", "hyp": "hello world",
                })
    with (run_dir / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SHELF_ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "script": "test_fixture", "hsr_label": "illustrative (STOI-derived "
        "proxy; no human study conducted)",
    }), encoding="utf-8")
    return run_dir


def _run_script(name: str, *args: str, timeout: int = 240) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        env=env, timeout=timeout,
    )
    return proc


# --------------------------------------------------------------------------
# CLI tests
# --------------------------------------------------------------------------
def test_benchmark_cli_help() -> None:
    proc = _run_script("run_comparative_benchmark.py", "--help")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--transforms" in proc.stdout


def test_shelf_life_cli_help() -> None:
    proc = _run_script("run_shelf_life.py", "--help")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--margins" in proc.stdout and "--analysis-only" in proc.stdout


def test_make_figures_cli(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    shelf = _make_shelf_run(tmp_path / "shelf")
    out = tmp_path / "figs"
    proc = _run_script(
        "make_figures.py", "--run", str(run), "--out", str(out),
        "--shelf-run", str(shelf),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    expected = [
        "fig1_pareto_human_vs_asr", "fig2_lambda_sweep_mirror",
        "fig3_cross_family_transfer", "fig4_family_summary",
        "fig5_taxonomy_overview", "fig6_strength_and_gap_curves",
        "fig7_quality_vs_degradation", "fig8_parameter_model_heatmaps",
        "fig9_speaker_variability", "fig10_word_confusion",
        "fig12_masking_thresholds", "fig13_metric_correlation",
        "fig14_forest_effect_sizes", "fig15_rank_stability",
        "fig16_shelf_life_forecast",
    ]
    for name in expected:
        assert (out / f"{name}.png").exists(), f"missing {name}.png"
        assert (out / f"{name}.svg").exists(), f"missing {name}.svg"
    # fig11 (spectrograms) needs real audio in the manifest — fixture has none
    assert not (out / "fig11_spectrograms.png").exists()


def test_make_figures_missing_run_is_nonzero(tmp_path: Path) -> None:
    proc = _run_script("make_figures.py", "--run", str(tmp_path / "nope"),
                       "--out", str(tmp_path / "o"))
    assert proc.returncode == 1


def test_make_tables_cli(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run")
    out = tmp_path / "tables"
    proc = _run_script("make_tables.py", "--run", str(run), "--out", str(out))
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in ("ranking_full", "top10_attack", "top10_human",
                 "lambda_sweep", "family_summary"):
        assert (out / f"{name}.csv").exists(), f"missing {name}.csv"
        tex = (out / f"{name}.tex").read_text(encoding="utf-8")
        assert "\\begin{table*}" in tex and "illustrative" in tex
    # every registered engine's ΔWER column reaches the paper tables
    top_tex = (out / "top10_attack.tex").read_text(encoding="utf-8")
    for label in ("Whisper tiny", "Whisper small", "Vosk", "wav2vec2"):
        assert label in top_tex, f"{label} column missing from top10_attack.tex"
    sweep_tex = (out / "lambda_sweep.tex").read_text(encoding="utf-8")
    assert "Whisper small" in sweep_tex and "wav2vec2" in sweep_tex


def test_shelf_life_analysis_cli(tmp_path: Path) -> None:
    shelf = _make_shelf_run(tmp_path / "shelf")
    proc = _run_script(
        "run_shelf_life.py", "--analysis-only", "--out", str(shelf),
        "--n-boot", "200",
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (shelf / "shelf_life.json").exists()
    report = (shelf / "shelf_life.md").read_text(encoding="utf-8")
    assert "illustrative" in report
    assert "Scaling fits and forecasts" in report
    summary = json.loads((shelf / "shelf_life.json").read_text(encoding="utf-8"))
    assert len(summary["capacity_points"]) == 3
    assert summary["fits"]  # at least one fitted condition
