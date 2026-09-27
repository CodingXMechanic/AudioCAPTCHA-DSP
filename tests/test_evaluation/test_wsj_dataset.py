"""Tests for the WSJ (base-paper dataset) adapter.

WSJ itself is LDC-licensed and not redistributable, so these tests build
synthetic Kaldi-layout and raw-LDC-layout trees and exercise the adapter,
the base-paper subset protocol (A/B/C) and the phone-rate helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audiocaptcha_dsp.evaluation.dataset import (
    BASE_PAPER_MAX_PHONES_PER_S,
    BASE_PAPER_SUBSETS,
    WSJAdapter,
    WSJDataset,
    estimate_phoneme_count,
    phone_rate,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic trees
# ---------------------------------------------------------------------------

def _write_tone(path: Path, dur: float = 0.55, sr: int = 16000, amp: float = 0.4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(dur * sr)) / sr
    wav = amp * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    sf.write(str(path), wav, sr)


def _build_kaldi_tree(
    root: Path, n_speakers: int, utts_per_spk: int, name: str = "test_dev93"
) -> Path:
    """Kaldi layout: root/{name}/{wav.scp,text,spk2utt} + audio/*.wav."""
    data = root / name
    audio = data / "audio"
    scp, text, spk2utt = [], [], []
    for s in range(n_speakers):
        spk = f"11{s}a"
        utts = []
        for u in range(utts_per_spk):
            utt = f"{spk}{u:03d}"
            rel = f"audio/{utt}.wav"
            _write_tone(data / rel)
            scp.append(f"{utt} {rel}")
            text.append(f"{utt} THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG {u}")
            utts.append(utt)
        spk2utt.append(f"{spk} {' '.join(utts)}")
    (data / "wav.scp").write_text("\n".join(scp) + "\n", encoding="utf-8")
    (data / "text").write_text("\n".join(text) + "\n", encoding="utf-8")
    (data / "spk2utt").write_text("\n".join(spk2utt) + "\n", encoding="utf-8")
    return data


def _build_kaldi_tree_named(
    root: Path, name: str, n_speakers: int, utts_per_spk: int
) -> Path:
    return _build_kaldi_tree(root, n_speakers, utts_per_spk, name=name)


def _build_raw_tree(root: Path, n_files: int = 6) -> None:
    for i in range(n_files):
        audio = root / "11-13.1" / f"wsj0/11x{i:02d}.wv1".replace(".wv1", ".wav")
        _write_tone(audio)
        audio.with_suffix(".txt").write_text(
            f"SENTENCE NUMBER {i} OF THE RAW LAYOUT", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

class TestLayouts:
    def test_kaldi_layout_loads(self, tmp_path) -> None:
        _build_kaldi_tree(tmp_path, n_speakers=3, utts_per_spk=4)
        adapter = WSJAdapter(tmp_path, subset="test_dev93")
        samples = adapter.load()
        assert len(samples) == 12
        assert {s.speaker_id for s in samples} == {"110a", "111a", "112a"}
        assert all(s.transcript for s in samples)
        assert all(s.dataset_name == "wsj" for s in samples)
        assert all(s.metadata["layout"] == "kaldi" for s in samples)

    def test_kaldi_layout_auto_detects_root(self, tmp_path) -> None:
        data = _build_kaldi_tree(tmp_path, n_speakers=2, utts_per_spk=3)
        # Move index to root so no subset dir lookup is needed
        for name in ("wav.scp", "text", "spk2utt"):
            (data / name).rename(tmp_path / name)
        audio_dir = tmp_path / "audio"
        if not audio_dir.exists():
            (data / "audio").rename(audio_dir)
        samples = WSJAdapter(tmp_path).load()
        assert len(samples) == 6

    def test_wav_scp_pipe_entries(self) -> None:
        entry = "sph2pipe -f wav -c 1 audio/11a0110.wv1 |"
        assert WSJAdapter._wav_scp_path(entry) == "audio/11a0110.wv1"
        assert WSJAdapter._wav_scp_path("audio/x.wav") == "audio/x.wav"

    def test_raw_layout_with_sidecar_transcripts(self, tmp_path) -> None:
        _build_raw_tree(tmp_path, n_files=5)
        samples = WSJAdapter(tmp_path).load()
        assert len(samples) == 5
        assert all("raw layout" in s.transcript.lower() for s in samples)
        assert all(s.metadata["layout"] == "raw" for s in samples)
        # Speaker derived from the WSJ utterance-id prefix (11x00 → "11x0")
        assert all(s.speaker_id.startswith("11x") for s in samples)

    def test_missing_directory_returns_empty(self, tmp_path) -> None:
        samples = WSJAdapter(tmp_path / "does-not-exist").load()
        assert samples == []


# ---------------------------------------------------------------------------
# Dataset fidelity: the selection protocol must be corpus-independent so the
# public stand-in mirrors the base-paper subset geometry exactly.
# ---------------------------------------------------------------------------

class TestProtocolParity:
    @staticmethod
    def _pool(n_speakers: int, utts_per_spk: int) -> list:
        from audiocaptcha_dsp.evaluation.dataset import SpeechSample

        return [
            SpeechSample(
                waveform=np.zeros(100),
                sample_rate=16000,
                transcript=f"utterance {u}",
                speaker_id=f"spk_{s:02d}",
                utterance_id=f"spk{s:02d}_utt{u:03d}",
            )
            for s in range(n_speakers)
            for u in range(utts_per_spk)
        ]

    def test_balanced_mode_matches_base_paper_geometry(self) -> None:
        from audiocaptcha_dsp.evaluation.dataset import select_paper_subset

        pool = self._pool(40, 10)  # stand-in corpus, e.g. LibriSpeech
        selected = select_paper_subset(pool, total=70, n_speakers=10)
        assert len(selected) == 70  # base-paper subset A size
        speakers = {s.speaker_id for s in selected}
        assert len(speakers) == 10
        assert all(
            sum(1 for s in selected if s.speaker_id == spk) == 7
            for spk in speakers
        )
        # Deterministic ordering
        again = select_paper_subset(pool, total=70, n_speakers=10)
        assert [s.utterance_id for s in selected] == [
            s.utterance_id for s in again
        ]

    def test_random_mode_is_seeded(self) -> None:
        from audiocaptcha_dsp.evaluation.dataset import select_paper_subset

        pool = self._pool(10, 20)
        a = select_paper_subset(pool, total=150, seed=5, mode="random")
        b = select_paper_subset(pool, total=150, seed=5, mode="random")
        assert len(a) == 150 == len(b)
        assert [s.utterance_id for s in a] == [s.utterance_id for s in b]

    def test_invalid_mode_raises(self) -> None:
        from audiocaptcha_dsp.evaluation.dataset import select_paper_subset

        with pytest.raises(ValueError):
            select_paper_subset(self._pool(2, 2), mode="fancy")

    def test_kaldi_dir_prefers_standard_test_sets(self, tmp_path) -> None:
        # Both standard test sets present, no explicit subset → test_dev93
        _build_kaldi_tree(tmp_path, n_speakers=2, utts_per_spk=3)
        _build_kaldi_tree_named(tmp_path, "test_eval92", n_speakers=2, utts_per_spk=3)
        adapter = WSJAdapter(tmp_path)
        assert adapter._kaldi_dir() == tmp_path / "test_dev93"

    def test_explicit_subset_wins(self, tmp_path) -> None:
        _build_kaldi_tree(tmp_path, n_speakers=2, utts_per_spk=3)
        _build_kaldi_tree_named(tmp_path, "test_eval92", n_speakers=2, utts_per_spk=3)
        adapter = WSJAdapter(tmp_path, subset="test_eval92")
        assert adapter._kaldi_dir() == tmp_path / "test_eval92"


# ---------------------------------------------------------------------------
# Base-paper subset protocol
# ---------------------------------------------------------------------------

class TestBasePaperSubsets:
    def test_subset_a_is_70_utterances_10_speakers(self, tmp_path) -> None:
        _build_kaldi_tree(tmp_path, n_speakers=12, utts_per_spk=8)
        adapter = WSJAdapter(tmp_path, subset="test_dev93")
        samples = adapter.load_base_paper_subset("A")
        assert len(samples) == BASE_PAPER_SUBSETS["A"]["speech"] == 70
        speakers = {s.speaker_id for s in samples}
        assert len(speakers) == 10
        # Round-robin selection: 70 across 10 speakers → 7 each
        counts = [sum(1 for s in samples if s.speaker_id == spk) for spk in speakers]
        assert counts == [7] * 10
        assert all(s.split == "base_paper_A" for s in samples)

    def test_subset_b_speech_count(self, tmp_path) -> None:
        _build_kaldi_tree(tmp_path, n_speakers=12, utts_per_spk=8)
        adapter = WSJAdapter(tmp_path, subset="test_dev93")
        samples = adapter.load_base_paper_subset("B")
        assert len(samples) == BASE_PAPER_SUBSETS["B"]["speech"] == 72

    def test_subset_c_random_pool_deterministic(self, tmp_path) -> None:
        _build_kaldi_tree(tmp_path, n_speakers=4, utts_per_spk=45)  # 180 pool
        a = WSJAdapter(tmp_path, subset="test_dev93").load_base_paper_subset(
            "C", seed=7
        )
        b = WSJAdapter(tmp_path, subset="test_dev93").load_base_paper_subset(
            "C", seed=7
        )
        assert len(a) == BASE_PAPER_SUBSETS["C"]["speech"] == 150
        assert [s.utterance_id for s in a] == [s.utterance_id for s in b]
        assert all(s.split == "base_paper_C" for s in a)
        assert all("est_phonemes" in s.metadata for s in a)

    def test_music_attached_from_separate_dir(self, tmp_path) -> None:
        _build_kaldi_tree(tmp_path, n_speakers=12, utts_per_spk=8)
        music = tmp_path / "music"
        for i in range(75):
            _write_tone(music / f"track_{i:03d}.wav")
        adapter = WSJAdapter(tmp_path, subset="test_dev93")
        samples = adapter.load_base_paper_subset("B", music_dir=music)
        assert sum(1 for s in samples if s.dataset_name == "music") == (
            BASE_PAPER_SUBSETS["B"]["music"]
        )

    def test_music_warns_but_continues_without_dir(self, tmp_path) -> None:
        _build_kaldi_tree(tmp_path, n_speakers=12, utts_per_spk=8)
        adapter = WSJAdapter(tmp_path, subset="test_dev93")
        samples = adapter.load_base_paper_subset("B")
        assert sum(1 for s in samples if s.dataset_name == "music") == 0
        assert len(samples) == 72

    def test_unknown_subset_raises(self, tmp_path) -> None:
        adapter = WSJAdapter(tmp_path)
        with pytest.raises(ValueError):
            adapter.load_base_paper_subset("Z")

    def test_alias_and_constants(self) -> None:
        assert WSJDataset is WSJAdapter
        assert BASE_PAPER_MAX_PHONES_PER_S == 6.0
        assert set(BASE_PAPER_SUBSETS) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Phone-rate helpers (subset-C ≤ 6 phones/s constraint)
# ---------------------------------------------------------------------------

class TestPhoneRate:
    def test_estimate_scales_with_text_length(self) -> None:
        short = estimate_phoneme_count("YES")
        long = estimate_phoneme_count(
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND RUNS AWAY FAST"
        )
        assert 0 < short < long

    def test_empty_text_is_zero(self) -> None:
        assert estimate_phoneme_count("") == 0

    def test_phone_rate_threshold_semantics(self) -> None:
        # ~10 phones over 2 s → about 5 phones/s ≤ 6 constraint
        rate = phone_rate("THE QUICK BROWN FOX JUMPS", 2.0)
        assert rate == estimate_phoneme_count("THE QUICK BROWN FOX JUMPS") / 2.0
        assert phone_rate("X", 0.0) == float("inf")

    def test_subset_annotations_include_rate(self, tmp_path) -> None:
        _build_kaldi_tree(tmp_path, n_speakers=4, utts_per_spk=45)
        samples = WSJAdapter(tmp_path, subset="test_dev93").load_base_paper_subset(
            "C", seed=1
        )
        for s in samples[:10]:
            assert s.metadata["est_phone_rate"] > 0
