"""Tests for the comparative benchmark module (experiments/comparative.py).

Covers: condition construction (dedupe, λ-sweep grid, controls), the
power-matched 'None' control, dataset selection with base-paper protocol
parity, Pareto logic, and end-to-end aggregation on a synthetic rows.csv.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.experiments.comparative import (
    ASR_LABELS,
    CONTROL_ID,
    DEFAULT_ENGINES,
    ORIGINAL_ID,
    ROW_FIELDS,
    SWEEP_MARGINS_DB,
    SWEEP_TARGETS,
    _check_engine_set,
    _is_nondominated,
    _make_engine,
    _macro_family_delta,
    aggregate_results,
    apply_no_threshold_control,
    asr_family_groups,
    build_conditions,
    run_benchmark,
    select_samples,
    write_dataset_manifest,
)


class TestBuildConditions:
    def test_original_and_dedupe_with_sweep(self) -> None:
        conds = build_conditions(
            ["psychoacoustic.masked_noise"], sweep=True, seed=42
        )
        ids = [c.condition_id for c in conds]
        assert ids[0] == ORIGINAL_ID
        # default margin of masked_noise is 20 dB which equals a sweep grid
        # point -> de-duplicated: the plain default row survives, the
        # '#margin=20' sweep twin must NOT exist
        assert ids.count("psychoacoustic.masked_noise") == 1
        assert "psychoacoustic.masked_noise#margin=20" not in ids
        default_rows = [
            c for c in conds
            if c.kind == "transform" and c.transform_key
            == "psychoacoustic.masked_noise"
        ]
        assert len(default_rows) == 1
        assert default_rows[0].params["margin_db"] == 20.0
        # all 4 sweep targets x 7 margins, minus the one duplicate
        sweep_rows = [c for c in conds if c.kind == "sweep"]
        assert len(sweep_rows) == len(SWEEP_TARGETS) * len(SWEEP_MARGINS_DB) - 1
        # control present exactly once
        assert ids.count(CONTROL_ID) == 1
        # margins are the paper grid
        for c in sweep_rows:
            assert c.margin_db in SWEEP_MARGINS_DB

    def test_no_sweep_keeps_control_and_defaults(self) -> None:
        conds = build_conditions(
            ["noise.white", "temporal.jitter"], sweep=False
        )
        ids = [c.condition_id for c in conds]
        assert ids[0] == ORIGINAL_ID
        assert "noise.white" in ids and "temporal.jitter" in ids
        assert not any(c.kind == "sweep" for c in conds)
        # the paper's 'None' column is always reported
        assert ids.count(CONTROL_ID) == 1

    def test_explicit_control_key(self) -> None:
        conds = build_conditions([CONTROL_ID], sweep=False)
        assert [c.condition_id for c in conds].count(CONTROL_ID) == 1

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown transform key"):
            build_conditions(["noise.does_not_exist"])

    def test_seed_propagated_to_seeded_transforms(self) -> None:
        conds = build_conditions(
            ["psychoacoustic.masked_noise"], sweep=False, seed=7
        )
        ctrl = [c for c in conds if c.condition_id == CONTROL_ID][0]
        assert ctrl.params["seed"] == 7
        t = [c for c in conds if c.kind == "transform"][0]
        # masked_noise accepts seed -> fixed for reproducibility
        assert t.params.get("seed") == 7

    def test_conditions_are_json_serialisable(self) -> None:
        for c in build_conditions(["noise.white"], sweep=True):
            json.dumps(c.to_dict(), default=str)


class TestNoThresholdControl:
    def test_power_matches_lambda0_injection(self) -> None:
        rng = np.random.default_rng(0)
        t = np.arange(16000) / 16000
        sig = Signal(
            waveform=0.5 * np.sin(2 * np.pi * 300 * t)
            + 0.2 * np.sin(2 * np.pi * 900 * t)
            + 0.02 * rng.standard_normal(16000),
            sample_rate=16000,
        )
        out = apply_no_threshold_control(sig, seed=42)
        pert = out.waveform - sig.waveform
        p_control = float(np.sum(pert**2))

        from audiocaptcha_dsp.experiments.runner import resolve_transform

        ref = resolve_transform(
            "psychoacoustic.masked_noise",
            {"margin_db": 0.0, "seed": 42},
        )(sig)
        p_ref = float(np.sum((ref.waveform - sig.waveform) ** 2))

        assert p_ref > 0
        # power-matched within numerical tolerance
        assert math.isclose(p_control, p_ref, rel_tol=0.01)

    def test_shape_and_signal_preserved(self) -> None:
        sig = Signal(
            waveform=np.random.default_rng(1).standard_normal(8000) * 0.1,
            sample_rate=16000,
        )
        out = apply_no_threshold_control(sig, seed=1)
        assert out.waveform.shape == sig.waveform.shape
        assert out.sample_rate == sig.sample_rate
        # noise was actually added (control != identity)
        assert not np.allclose(out.waveform, sig.waveform)


class TestPareto:
    def test_nondominated_detection(self) -> None:
        import pandas as pd

        #   A (best human, weak attack)   B (balanced)   C (dominated by B)
        #   D (best attack)               E (dominated by A)
        df = pd.DataFrame({
            "stoi_mean":      [0.90, 0.80, 0.75, 0.70, 0.70],
            "cross_delta_wer": [0.10, 0.40, 0.20, 0.45, 0.10],
        })
        assert _is_nondominated(0.90, 0.10, df) is True   # A: best human
        assert _is_nondominated(0.80, 0.40, df) is True   # B
        assert _is_nondominated(0.75, 0.20, df) is False  # C dominated by B
        assert _is_nondominated(0.70, 0.45, df) is True   # D: best attack
        assert _is_nondominated(0.70, 0.10, df) is False  # E dominated by A

    def test_nonfinite_delta_not_pareto(self) -> None:
        import pandas as pd

        df = pd.DataFrame({
            "stoi_mean": [1.0],
            "cross_delta_wer": [np.nan],
        })
        assert _is_nondominated(1.0, np.nan, df) is False


@pytest.fixture()
def mini_librispeech(tmp_path: Path) -> Path:
    """10 speakers x 8 utterances of short wavs + LibriSpeech trans files."""
    import soundfile as sf

    rng = np.random.default_rng(42)
    sr = 16000
    n = int(sr * 0.8)
    for s in range(10):
        spk = f"19{s:03d}"
        ch = f"19{s:03d}01"
        d = tmp_path / spk / ch
        d.mkdir(parents=True)
        lines = []
        for u in range(8):
            utt = f"{spk}-{ch}-{u:04d}"
            # low amplitude: the adapter discards clipped files (max >= 0.999)
            wav = 0.05 * rng.standard_normal(n)
            sf.write(str(d / f"{utt}.wav"), wav, sr)
            lines.append(f"{utt} THE QUICK BROWN FOX JUMPS")
        (d / f"{spk}-{ch}.trans.txt").write_text("\n".join(lines))
    return tmp_path


class TestSelectSamples:
    def test_subset_a_protocol_geometry(self, mini_librispeech: Path) -> None:
        samples, info = select_samples(
            "librispeech", mini_librispeech, subset="A", seed=42
        )
        assert len(samples) == 70  # base-paper subset A
        spks = {s.speaker_id for s in samples}
        assert len(spks) == 10
        assert all(
            sum(1 for s in samples if s.speaker_id == spk) == 7
            for spk in spks
        )
        assert info["protocol"] == "base-paper-A"
        assert "stand-in" in info["corpus"]
        # workers need the source path
        assert all(s.metadata.get("source_path") for s in samples)

    def test_max_utterances_cap(self, mini_librispeech: Path) -> None:
        samples, _ = select_samples(
            "librispeech", mini_librispeech, subset="A", max_utterances=5
        )
        assert len(samples) == 5

    def test_invalid_subset_raises(self, mini_librispeech: Path) -> None:
        with pytest.raises(ValueError):
            select_samples("librispeech", mini_librispeech, subset="Z")

    def test_manifest_written_and_hashed(
        self, mini_librispeech: Path, tmp_path: Path
    ) -> None:
        samples, info = select_samples(
            "librispeech", mini_librispeech, subset="A"
        )
        out = tmp_path / "run"
        out.mkdir()
        path = write_dataset_manifest(out, samples, info)
        m = json.loads(path.read_text(encoding="utf-8"))
        assert m["n_samples"] == 70
        assert m["protocol"] == "base-paper-A"
        assert len(m["utterances"]) == 70
        assert all(u["source_path"] for u in m["utterances"])
        assert len(m["corpus_sha256"]) == 64
        # stable hash: same selection -> same hash
        m2 = json.loads(
            write_dataset_manifest(out, samples, info).read_text(
                encoding="utf-8"
            )
        )
        assert m["corpus_sha256"] == m2["corpus_sha256"]


def _write_rows(out_dir: Path, rows: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "rows.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ROW_FIELDS})
    return p


def _row(cid, kind, key, fam, utt, stoi, snr, w1, w2, margin="", params="{}",
         w_small="", w_wav2vec=""):
    return {
        "condition_id": cid, "kind": kind, "transform_key": key,
        "family": fam, "params_json": params, "margin_db": margin,
        "utt_id": utt, "spk": "s1", "dur_s": 2.0, "status": "ok",
        "error": "", "snr_db": snr, "stoi_proxy": stoi, "mbsd": 1.0,
        "si_sdr_db": snr, "rms_ratio": 1.0, "ref_text": "some words here",
        "wer_whisper_tiny": w1, "wer_vosk_small_en": w2,
        "wer_whisper_small": w_small, "wer_wav2vec2_base": w_wav2vec,
        "hyp_whisper_tiny": "", "hyp_vosk_small_en": "",
        "hyp_whisper_small": "", "hyp_wav2vec2_base": "",
    }


class TestEngineRegistry:
    """Four registered engines: Whisper (tiny + small), Vosk/Kaldi, wav2vec2."""

    def test_row_fields_cover_every_registered_engine(self) -> None:
        # headline default = the two-family paper matrix
        assert DEFAULT_ENGINES == ["whisper_tiny", "vosk_small_en"]
        # four engines remain registered for the validation run
        assert set(ASR_LABELS) == {
            "whisper_tiny", "whisper_small", "vosk_small_en", "wav2vec2_base",
        }
        for eid in ASR_LABELS:
            assert f"wer_{eid}" in ROW_FIELDS
            assert f"hyp_{eid}" in ROW_FIELDS

    def test_family_groups(self) -> None:
        groups = asr_family_groups()
        assert set(groups) == {"whisper", "kaldi", "ssl"}
        assert groups["whisper"] == ["whisper_tiny", "whisper_small"]
        assert groups["kaldi"] == ["vosk_small_en"]
        assert groups["ssl"] == ["wav2vec2_base"]
        # a partial engine list keeps only the requested engines
        assert asr_family_groups(["whisper_small"]) == {
            "whisper": ["whisper_small"],
        }

    def test_macro_family_delta(self) -> None:
        # one engine per family -> plain mean (back-compatible with 2 engines)
        two = {"wer_whisper_tiny": 0.5, "wer_vosk_small_en": 0.3}
        assert _macro_family_delta(two) == pytest.approx(0.4)
        # Whisper measured at two sizes must not outweigh the other families
        four = {"wer_whisper_tiny": 0.6, "wer_whisper_small": 0.2,
                "wer_vosk_small_en": 0.3, "wer_wav2vec2_base": 0.1}
        whisper_mean = (0.6 + 0.2) / 2
        assert _macro_family_delta(four) == pytest.approx(
            (whisper_mean + 0.3 + 0.1) / 3
        )
        assert math.isnan(_macro_family_delta({}))

    def test_make_engine_and_unknown_rejected(self) -> None:
        small = _make_engine("whisper_small")
        assert small.name == "whisper_small"
        assert small.model_size == "small"
        with pytest.raises(ValueError, match="unknown ASR engine"):
            _make_engine("gpt4_voice")

    def test_run_benchmark_rejects_unknown_engine(self, tmp_path: Path) -> None:
        # validation happens before any dataset/transform work
        with pytest.raises(ValueError, match="unknown ASR engine"):
            run_benchmark(out_dir=tmp_path, engines=["gpt4_voice"])

    def test_engine_set_mismatch_refused(self, tmp_path: Path) -> None:
        """Rows produced by one engine set may not be resumed with another."""
        # rows.csv from a two-engine run (small/wav2vec2 cells empty)
        _write_rows(tmp_path, [
            _row(ORIGINAL_ID, "original", "baseline.identity", "-",
                 "u0", 1.0, "", 0.1, 0.2),
        ])
        rows_path = tmp_path / "rows.csv"
        # same engine set -> resume allowed
        _check_engine_set(rows_path, ["whisper_tiny", "vosk_small_en"])
        # requesting the extended four-engine set -> explicit refusal,
        # not a silent mix
        with pytest.raises(RuntimeError, match="engine set"):
            _check_engine_set(rows_path, list(ASR_LABELS))
        # and run_benchmark stops before touching dataset or ASR
        with pytest.raises(RuntimeError, match="engine set"):
            run_benchmark(out_dir=tmp_path, engines=list(ASR_LABELS))
        # nothing written yet -> nothing to check
        _check_engine_set(tmp_path / "absent.csv", list(ASR_LABELS))


class TestAggregateFourEngines:
    """Headline cross-ΔWER is a macro-average over ASR families."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        rows = []
        for i in range(4):
            rows.append(_row(
                ORIGINAL_ID, "original", "baseline.identity", "-",
                f"u{i}", 1.0, "", 0.10, 0.20, w_small="0.05",
                w_wav2vec="0.02",
            ))
            rows.append(_row(
                "attack.x", "transform", "attack.x", "G",
                f"u{i}", 0.88, 8.0, 0.60, 0.50, w_small="0.15",
                w_wav2vec="0.12",
            ))
        _write_rows(tmp_path, rows)
        return tmp_path

    def test_cross_delta_is_family_macro(self, run_dir: Path) -> None:
        csv_path = aggregate_results(run_dir, log=lambda *a: None)
        import pandas as pd

        s = pd.read_csv(csv_path).set_index("condition_id")
        a = s.loc["attack.x"]
        # per-engine deltas: tiny .50, small .10, vosk .30, wav2vec2 .10
        assert a.wer_whisper_tiny_delta == pytest.approx(0.50)
        assert a.wer_whisper_small_delta == pytest.approx(0.10)
        assert a.wer_wav2vec2_base_delta == pytest.approx(0.10)
        whisper_family = (0.50 + 0.10) / 2
        assert a.cross_delta_wer == pytest.approx(
            (whisper_family + 0.30 + 0.10) / 3
        )
        # ranking.md names every engine and the three families
        md = (run_dir / "ranking.md").read_text(encoding="utf-8")
        assert "ΔWER whisper small" in md
        assert "3 independent families" in md


class TestAggregate:
    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        rows = []
        for i in range(4):
            rows.append(_row(
                ORIGINAL_ID, "original", "baseline.identity", "-",
                f"u{i}", 1.0, "", 0.10, 0.20,
            ))
            # strong attack, decent human score
            rows.append(_row(
                "attack.strong", "transform", "attack.strong", "G",
                f"u{i}", 0.88, 8.0, 0.60, 0.50,
            ))
            # transparent, weak attack
            rows.append(_row(
                "gentle.soft", "transform", "gentle.soft", "F",
                f"u{i}", 0.99, 25.0, 0.12, 0.22,
            ))
            # sweep rows for margin formatting coverage
            rows.append(_row(
                "psy.masked#margin=0", "sweep", "psy.masked", "F",
                f"u{i}", 0.85, 0.5, 0.55, 0.45, margin="0",
                params='{"margin_db": 0}',
            ))
        _write_rows(tmp_path, rows)
        return tmp_path

    def test_summary_and_ranks(self, run_dir: Path) -> None:
        csv_path = aggregate_results(run_dir, log=lambda *a: None)
        assert csv_path.exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "ranking.md").exists()

        import pandas as pd

        s = pd.read_csv(csv_path)
        by_id = s.set_index("condition_id")

        # baselines: whisper 0.10, vosk 0.20
        a = by_id.loc["attack.strong"]
        assert a.n_ok == 4
        assert a.wer_whisper_tiny_delta == pytest.approx(0.50)
        assert a.wer_vosk_small_en_delta == pytest.approx(0.30)
        assert a.cross_delta_wer == pytest.approx(0.40)
        assert a.human_rank == 2  # stoi 0.88: gentle.soft 0.99 ranks above
        assert bool(a.pareto) is True  # best attack, no one >= in both

        g = by_id.loc["gentle.soft"]
        assert bool(g.pareto) is True  # best human among transforms

        # original is a control row: no ranks
        o = by_id.loc[ORIGINAL_ID]
        assert pd.isna(o.attack_rank) or o.attack_rank > 3

        # HSR label travels with every row
        assert s.hsr_label.str.startswith("illustrative").all()

        # sweep row got a margin + appears in ranking.md sweep section
        sw = by_id.loc["psy.masked#margin=0"]
        assert sw.margin_db == 0
        md = (run_dir / "ranking.md").read_text(encoding="utf-8")
        assert "λ-sweep mirror" in md
        assert "attack.strong" in md

    def test_error_rows_counted(self, run_dir: Path) -> None:
        # add an all-error condition
        rows = list(csv.DictReader(
            (run_dir / "rows.csv").open(encoding="utf-8")
        ))
        bad = dict(rows[0])
        bad.update({
            "condition_id": "broken.tf", "kind": "transform",
            "transform_key": "broken.tf", "status": "error",
            "error": "RuntimeError: boom",
        })
        rows.append(bad)
        _write_rows(run_dir, rows)

        csv_path = aggregate_results(run_dir, log=lambda *a: None)
        import pandas as pd

        s = pd.read_csv(csv_path)
        assert "broken.tf" in set(s.condition_id)
        summary_json = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )
        assert summary_json["n_err"] == 1

    def test_missing_rows_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            aggregate_results(tmp_path, log=lambda *a: None)
