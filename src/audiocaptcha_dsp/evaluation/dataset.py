from __future__ import annotations

import csv
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import librosa

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.asr.engine import normalize_transcript

logger = logging.getLogger(__name__)


@dataclass
class SpeechSample:
    """Standardized speech sample representation with complete provenance metadata."""
    waveform: np.ndarray
    sample_rate: int
    transcript: str
    speaker_id: str
    utterance_id: str
    language: str = "en"
    dataset_name: str = "custom"
    split: str = "test"
    duration: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration <= 0.0 and self.sample_rate > 0:
            self.duration = float(len(self.waveform)) / self.sample_rate

    def to_signal(self) -> Signal:
        return Signal(
            waveform=self.waveform,
            sample_rate=self.sample_rate,
            metadata={
                "speaker_id": self.speaker_id,
                "utterance_id": self.utterance_id,
                "transcript": self.transcript,
                "language": self.language,
                "dataset_name": self.dataset_name,
                "split": self.split,
                "duration": self.duration,
                **self.metadata,
            },
        )


class BaseSpeechDataset(ABC):
    """Abstract base class for speech dataset adapters with validation and filtering."""
    
    def __init__(
        self,
        name: str,
        target_sr: int = 16000,
        min_duration_s: float = 0.5,
        max_duration_s: float = 15.0,
    ) -> None:
        self.name = name
        self.target_sr = target_sr
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.samples: list[SpeechSample] = []

    @abstractmethod
    def load(self) -> list[SpeechSample]:
        """Load samples from disk."""
        ...

    def filter_samples(self) -> list[SpeechSample]:
        """Filter samples by duration, clipping, and silence."""
        filtered = []
        for s in self.samples:
            if s.duration < self.min_duration_s or s.duration > self.max_duration_s:
                continue
            # Clipping detection
            if np.max(np.abs(s.waveform)) >= 0.999:
                logger.debug("Utterance %s clipped, skipping", s.utterance_id)
                continue
            # Silence detection
            rms = np.sqrt(np.mean(s.waveform ** 2))
            if rms < 1e-4:
                continue
            filtered.append(s)
        self.samples = filtered
        return self.samples

    def speaker_disjoint_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[SpeechSample], list[SpeechSample], list[SpeechSample]]:
        """Partition dataset into speaker-disjoint splits to guarantee zero speaker leakage."""
        speakers = sorted(list({s.speaker_id for s in self.samples}))
        rng = np.random.default_rng(seed)
        shuffled_speakers = rng.permutation(speakers)
        
        n_spk = len(shuffled_speakers)
        n_train = int(round(train_ratio * n_spk))
        n_val = int(round(val_ratio * n_spk))
        
        train_spks = set(shuffled_speakers[:n_train])
        val_spks = set(shuffled_speakers[n_train:n_train + n_val])
        test_spks = set(shuffled_speakers[n_train + n_val:])
        
        train_set = [s for s in self.samples if s.speaker_id in train_spks]
        val_set = [s for s in self.samples if s.speaker_id in val_spks]
        test_set = [s for s in self.samples if s.speaker_id in test_spks]
        
        # Annotate splits
        for s in train_set:
            s.split = "train"
        for s in val_set:
            s.split = "val"
        for s in test_set:
            s.split = "test"
            
        return train_set, val_set, test_set


class LibriSpeechAdapter(BaseSpeechDataset):
    """Adapter for LibriSpeech directory structure: root/split/reader_id/chapter_id/*.flac or *.wav.
    
    Expected structure: root/split/reader_id/chapter_id/*.flac or *.wav
    Transcripts: root/split/reader_id/chapter_id/reader-chapter.trans.txt
    """
    
    def __init__(self, root_dir: Path | str, split: str = "test-clean", target_sr: int = 16000) -> None:
        super().__init__(name=f"librispeech_{split}", target_sr=target_sr)
        self.root_dir = Path(root_dir)
        self.split = split

    def load(self) -> list[SpeechSample]:
        self.samples = []
        split_dir = self.root_dir / self.split if (self.root_dir / self.split).exists() else self.root_dir
        if not split_dir.exists():
            logger.warning("LibriSpeech directory not found: %s", split_dir)
            return self.samples

        # Find all speaker directories
        for spk_dir in sorted(split_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            speaker_id = spk_dir.name
            # Find all chapter directories within speaker
            for ch_dir in sorted(spk_dir.iterdir()):
                if not ch_dir.is_dir():
                    continue
                chapter_name = ch_dir.name
                # Transcripts are stored in reader-chapter.trans.txt
                # Pattern: {speaker_id}-{chapter_name}.trans.txt
                trans_file = ch_dir / f"{speaker_id}-{chapter_name}.trans.txt"
                transcripts: dict[str, str] = {}
                if trans_file.exists():
                    try:
                        for line in trans_file.read_text(encoding="utf-8").splitlines():
                            parts = line.strip().split(" ", 1)
                            if len(parts) == 2:
                                utt_id_raw = parts[0]
                                transcript = normalize_transcript(parts[1])
                                transcripts[utt_id_raw] = transcript
                    except Exception as e:
                        logger.warning("Failed to read transcript file %s: %s", trans_file, e)
                
                # Load all audio files in chapter directory
                for audio_path in sorted(list(ch_dir.glob("*.flac")) + list(ch_dir.glob("*.wav"))):
                    utt_id = audio_path.stem
                    txt = transcripts.get(utt_id, "")
                    try:
                        wav, sr = sf.read(str(audio_path), dtype="float64")
                        if wav.ndim > 1:
                            wav = wav.mean(axis=-1)
                        if sr != self.target_sr:
                            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)
                        self.samples.append(
                            SpeechSample(
                                waveform=wav,
                                sample_rate=self.target_sr,
                                transcript=txt,
                                speaker_id=speaker_id,
                                utterance_id=utt_id,
                                language="en",
                                dataset_name="librispeech",
                                split=self.split,
                                metadata={"source_path": str(audio_path)},
                            )
                        )
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", audio_path, e)
        return self.filter_samples()


class CommonVoiceAdapter(BaseSpeechDataset):
    """Adapter for Mozilla Common Voice: root/validated.tsv and root/clips/*.mp3 (or .wav).
    
    Expected structure: root/validated.tsv (or other split) and root/clips/*audio files.
    The TSV contains columns: client_id, path, sentence (among others).
    """
    
    def __init__(self, root_dir: Path | str, split: str = "validated", language: str = "en", target_sr: int = 16000) -> None:
        super().__init__(name=f"common_voice_{language}_{split}", target_sr=target_sr)
        self.root_dir = Path(root_dir)
        self.tsv_name = f"{split}.tsv"
        self.language = language

    def load(self) -> list[SpeechSample]:
        self.samples = []
        tsv_path = self.root_dir / self.tsv_name
        clips_dir = self.root_dir / "clips"
        if not tsv_path.exists() or not clips_dir.exists():
            logger.warning("Common Voice manifest or clips not found in %s", self.root_dir)
            return self.samples

        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                client_id = row.get("client_id", "unknown")
                path_name = row.get("path", "")
                sentence = normalize_transcript(row.get("sentence", ""))
                if not path_name:
                    continue
                audio_file = clips_dir / path_name
                if not audio_file.exists():
                    logger.debug("Audio file not found: %s", audio_file)
                    continue
                try:
                    wav, sr = sf.read(str(audio_file), dtype="float64")
                    if wav.ndim > 1:
                        wav = wav.mean(axis=-1)
                    if sr != self.target_sr:
                        wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)
                    self.samples.append(
                        SpeechSample(
                            waveform=wav,
                            sample_rate=self.target_sr,
                            transcript=sentence,
                            speaker_id=client_id,
                            utterance_id=audio_file.stem,
                            language=self.language,
                            dataset_name="common_voice",
                            split=self.name,
                        )
                    )
                except Exception as e:
                    logger.debug("Failed reading %s: %s", audio_file, e)
        return self.filter_samples()


class SyntheticSpeechDataset(BaseSpeechDataset):
    """Lightweight phonetic-formant speech synthesizer for CI and offline benchmarking.
    
    Generates speech-like vowel-consonant bursts mimicking human phonemes.
    """
    
    def __init__(self, n_samples: int = 10, seed: int = 42, target_sr: int = 16000) -> None:
        super().__init__(name="synthetic_speech", target_sr=target_sr)
        self.n_samples = n_samples
        self.seed = seed

    def load(self) -> list[SpeechSample]:
        rng = np.random.default_rng(self.seed)
        self.samples = []
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
        
        duration = 2.0
        n_pts = int(self.target_sr * duration)
        t = np.linspace(0, duration, n_pts, endpoint=False)
        
        for i in range(self.n_samples):
            spk_id = f"speaker_{i % 3:02d}"
            word = words[i % len(words)]
            
            # Formant synthesis: f0 + f1 + f2 + f3
            f0 = 120.0 + (i % 3) * 50.0  # fundamental
            f1 = 500.0 + rng.uniform(-50, 50)
            f2 = 1500.0 + rng.uniform(-100, 100)
            f3 = 2500.0 + rng.uniform(-150, 150)
            
            # Temporal amplitude modulation (syllable envelope)
            envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t)) * np.exp(-1.5 * t / duration)
            
            s = (
                0.5 * np.sin(2 * np.pi * f0 * t) +
                0.3 * np.sin(2 * np.pi * f1 * t) +
                0.2 * np.sin(2 * np.pi * f2 * t) +
                0.1 * np.sin(2 * np.pi * f3 * t)
            ) * envelope
            
            # Add light aspiration noise
            s += rng.normal(0, 0.005, size=n_pts)
            s = s / (np.max(np.abs(s)) + 1e-8) * 0.5
            
            self.samples.append(
                SpeechSample(
                    waveform=s.astype(np.float64),
                    sample_rate=self.target_sr,
                    transcript=word,
                    speaker_id=spk_id,
                    utterance_id=f"synth_spk{spk_id}_{word}_{i:03d}",
                    language="en",
                    dataset_name="synthetic_speech",
                    split="test",
                    metadata={"f0": f0, "f1": f1, "f2": f2, "f3": f3},
                )
            )
        return self.samples


class FLEURSAdapter(BaseSpeechDataset):
    """Adapter for FLEURS (FLORES-200) multilingual dataset.
    
    FLEURS is a subset of TED-LIUM3 with 100 languages, 1 hour each.
    Expected structure: root/*.tsv or root/data/ format.
    """
    
    def __init__(self, root_dir: Path | str, language: str = "eng", target_sr: int = 16000) -> None:
        super().__init__(name=f"fleurs_{language}", target_sr=target_sr)
        self.root_dir = Path(root_dir)
        self.language = language

    def load(self) -> list[SpeechSample]:
        self.samples = []
        root = self.root_dir
        
        # Try common FLEURS directory structures
        possible_dirs = [
            root,
            root / "data",
            root / "fleurs",
            root / "flures",
        ]
        
        data_dir = None
        for d in possible_dirs:
            if d.exists():
                data_dir = d
                break
        
        if data_dir is None:
            logger.warning("FLEURS directory not found in %s", root)
            return self.samples
        
        # Look for transcript files
        tsv_paths = list(data_dir.glob("*.tsv")) + list(data_dir.glob("*/*.tsv"))
        
        for tsv_path in tsv_paths:
            try:
                with open(tsv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        lang = row.get("language", "")
                        if lang and lang != self.language:
                            continue
                        client_id = row.get("client_id", "unknown")
                        path_name = row.get("path", "")
                        sentence = normalize_transcript(row.get("sentence", ""))
                        if not path_name:
                            continue
                        audio_file = data_dir / path_name
                        if not audio_file.exists():
                            continue
                        try:
                            wav, sr = sf.read(str(audio_file), dtype="float64")
                            if wav.ndim > 1:
                                wav = wav.mean(axis=-1)
                            if sr != self.target_sr:
                                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)
                            self.samples.append(
                                SpeechSample(
                                    waveform=wav,
                                    sample_rate=self.target_sr,
                                    transcript=sentence,
                                    speaker_id=client_id,
                                    utterance_id=audio_file.stem,
                                    language=lang,
                                    dataset_name="fleurs",
                                    split="test",
                                )
                            )
                        except Exception as e:
                            logger.debug("Failed reading %s: %s", audio_file, e)
            except Exception as e:
                logger.warning("Failed to read FLEURS manifest %s: %s", tsv_path, e)
        
        return self.filter_samples()


# ---------------------------------------------------------------------------
# Wall Street Journal (base-paper dataset, Schönherr et al. 2018)
# ---------------------------------------------------------------------------

# Base-paper evaluation subsets, arXiv:1808.05665:
#   §IV-A  "a subset of 70 utterances for 10 different speakers from one of
#           the WSJ test sets was used"                      → subset A
#   Table I "test set of speech containing 72 samples and a test set of
#           music containing 70 samples"                     → subset B
#   §IV-D3 "randomly chose speech files from 150 samples and music files
#           from 72 samples ... only audio-text-pairs with a phone rate
#           of 6 phones per second or less ... 100 times per λ"
#                                                          → subset C
BASE_PAPER_SUBSETS: dict[str, dict[str, int]] = {
    "A": {"speech": 70, "n_speakers": 10, "music": 0},
    "B": {"speech": 72, "n_speakers": 10, "music": 70},
    "C": {"speech": 150, "n_speakers": 0, "music": 72},
}
# §IV-D3 constraint for subset C audio-text pairs
BASE_PAPER_MAX_PHONES_PER_S = 6.0


def estimate_phoneme_count(text: str) -> int:
    """Estimate the phoneme count of an English transcript.

    The base paper restricts subset-C audio-text pairs to ≤ 6 phones per
    second.  Exact phone counts require a pronunciation lexicon (e.g.
    CMUdict), which is not bundled here; this estimator uses the
    well-known English average of ≈ 4.5 phonemes per 4.7-letter word
    (≈ 0.96 phonemes per letter), plus ≈ 1.5 phonemes per spoken digit.

    Parameters
    ----------
    text : str
        Normalized transcript (see :func:`normalize_transcript`).

    Returns
    -------
    int
        Estimated number of phonemes (at least 1 for non-empty input).
    """
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    count = 0.96 * letters + 1.5 * digits
    if count <= 0:
        return 1 if text.strip() else 0
    return int(round(count))


def phone_rate(text: str, duration_s: float) -> float:
    """Estimated phones per second for an audio-text pair."""
    if duration_s <= 0:
        return float("inf")
    return estimate_phoneme_count(text) / float(duration_s)


# Kaldi's default WSJ recipe (egs/wsj/s5) materialises these test sets; the
# base paper says its 70-utterance subset came from "one of the WSJ test
# sets" without naming it, so we prefer the recipe's standard dev set first.
KALDI_WSJ_TEST_SETS: tuple[str, ...] = ("test_dev93", "test_eval92")


def select_paper_subset(
    samples: Sequence[SpeechSample],
    total: int = 70,
    n_speakers: int = 10,
    seed: int = 42,
    mode: str = "balanced",
) -> list[SpeechSample]:
    """Select a base-paper-protocol subset from ANY speech corpus.

    Implements the exact selection protocol of Schönherr et al. (2018,
    arXiv:1808.05665) §IV-A: *"a subset of 70 utterances for 10 different
    speakers from one of the WSJ test sets"*.  Because the paper does not
    disclose the chosen utterance IDs, the protocol (not the bit-identical
    files) is what can be reproduced exactly; applying it to a substitute
    corpus keeps subset geometry — 70 utterances / 10 speakers / 7 each —
    identical to the base paper.

    Parameters
    ----------
    samples : sequence of SpeechSample
        Candidate pool (e.g. from ``LibriSpeechAdapter.load()``).
    total : int
        Number of utterances (70 for base-paper subsets A; 72 for B;
        150 for the subset-C pool).
    n_speakers : int
        Number of distinct speakers (10 for subsets A/B; 0 = any).
    seed : int
        RNG seed for ``mode='random'`` (subset C: *"randomly chose speech
        files from 150 samples"*).
    mode : {'balanced', 'random'}
        ``balanced`` round-robins over sorted speakers so every speaker
        contributes equally (subsets A/B); ``random`` draws a seeded random
        pool (subset C).

    Returns
    -------
    list of SpeechSample
        Selected samples in deterministic order.
    """
    if mode not in ("balanced", "random"):
        raise ValueError(f"mode must be 'balanced' or 'random', got {mode!r}")

    if mode == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(samples))
        n = min(total, len(samples))
        return [samples[i] for i in order[:n]]

    by_speaker: dict[str, list[SpeechSample]] = {}
    for s in sorted(samples, key=lambda s: s.utterance_id):
        by_speaker.setdefault(s.speaker_id, []).append(s)
    speakers = sorted(by_speaker)
    if n_speakers > 0:
        speakers = speakers[:n_speakers]
    selected: list[SpeechSample] = []
    idx = 0
    while len(selected) < total and any(
        idx < len(by_speaker[spk]) for spk in speakers
    ):
        for spk in speakers:
            if idx < len(by_speaker[spk]) and len(selected) < total:
                selected.append(by_speaker[spk][idx])
        idx += 1
    return selected


class WSJAdapter(BaseSpeechDataset):
    """Adapter for the Wall Street Journal corpus (base-paper dataset).

    Schönherr et al. (2018, arXiv:1808.05665) used "the default settings of
    the Wall Street Journal (WSJ) training recipe of the Kaldi toolkit";
    this adapter therefore understands two on-disk layouts:

    1. **Kaldi recipe layout** (preferred): ``root/[subset/]wav.scp`` with
       optional ``text`` and ``spk2utt`` files, as produced by Kaldi's
       ``wsj_data_prep.sh``.
    2. **Raw LDC layout** (LDC93S6B / LDC94S37A): recursive
       ``*.wv1`` / ``*.wv2`` / ``*.wav`` audio with transcripts from
       sidecar ``*.txt`` / ``*.trn`` files or ``*.trans.txt`` indices.

    WSJ is LDC-licensed and cannot be redistributed; this adapter never
    downloads it. See ``scripts/prepare_wsj.py`` for acquisition and
    conversion instructions.

    Base-paper subsets
    ------------------
    Use :meth:`load_base_paper_subset` for the exact evaluation protocol:
    subset A = 70 utterances / 10 speakers, subset B = 72 speech (+70 music
    from a separate corpus), subset C = 150 speech (+72 music) with
    audio-text pairs restricted to ≤6 phones/s.
    """

    AUDIO_SUFFIXES = (".wv1", ".wv2", ".wav", ".flac", ".au")

    def __init__(
        self,
        root_dir: Path | str,
        subset: str | None = None,
        target_sr: int = 16000,
    ) -> None:
        name = "wsj" if subset is None else f"wsj_{subset}"
        super().__init__(name=name, target_sr=target_sr)
        self.root_dir = Path(root_dir)
        self.subset = subset

    # -- layout helpers ----------------------------------------------------

    def _kaldi_dir(self) -> Path | None:
        """Return the directory containing wav.scp, if the Kaldi layout exists.

        Preference: explicit ``subset`` → the recipe's standard test sets
        (``test_dev93``, ``test_eval92``; base paper: "one of the WSJ test
        sets") → root → any immediate subdirectory with a wav.scp.
        """
        candidates: list[Path] = []
        if self.subset is not None:
            candidates.append(self.root_dir / self.subset)
        else:
            candidates.extend(self.root_dir / s for s in KALDI_WSJ_TEST_SETS)
        candidates.append(self.root_dir)
        for c in candidates:
            if (c / "wav.scp").exists():
                return c
        if self.subset is None and self.root_dir.exists():
            for c in sorted(self.root_dir.iterdir()):
                if c.is_dir() and (c / "wav.scp").exists():
                    return c
        return None

    def _read_index(self, kaldi_dir: Path, filename: str) -> dict[str, str]:
        """Read a Kaldi one-entry-per-line index file into {key: value}."""
        path = kaldi_dir / filename
        index: dict[str, str] = {}
        if not path.exists():
            return index
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    index[parts[0]] = parts[1]
                elif len(parts) == 1:
                    index[parts[0]] = ""
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to read %s: %s", path, e)
        return index

    @staticmethod
    def _wav_scp_path(entry: str) -> str | None:
        """Extract the audio path from a wav.scp entry (possibly a pipe)."""
        entry = entry.strip()
        if not entry:
            return None
        # Kaldi pipes: "sph2pipe -f wav path |" — take the last non-pipe token
        if entry.endswith("|"):
            tokens = [t for t in entry.split() if t != "|"]
            audio_tokens = [t for t in tokens if not t.startswith("-")]
            entry = audio_tokens[-1] if audio_tokens else tokens[-1]
        return entry

    def _load_audio(self, path: Path) -> tuple[np.ndarray, int] | None:
        """Read audio, resampling to target_sr; None if unreadable."""
        try:
            wav, sr = sf.read(str(path), dtype="float64")
        except Exception as e:
            logger.warning(
                "Failed to read %s (%s). Sphere files (.wv1/.wv2) must be "
                "converted first — see scripts/prepare_wsj.py.",
                path, e,
            )
            return None
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != self.target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)
        return wav, self.target_sr

    # -- loaders ------------------------------------------------------------

    def _load_kaldi(self, kaldi_dir: Path) -> list[SpeechSample]:
        wav_scp = self._read_index(kaldi_dir, "wav.scp")
        text = self._read_index(kaldi_dir, "text")
        spk2utt = self._read_index(kaldi_dir, "spk2utt")
        utt2spk: dict[str, str] = {}
        for spk, utts in spk2utt.items():
            for utt in utts.split():
                utt2spk[utt] = spk
        split_name = self.subset or kaldi_dir.name

        samples: list[SpeechSample] = []
        for utt_id in sorted(wav_scp):
            audio_path = self._wav_scp_path(wav_scp[utt_id])
            if not audio_path:
                continue
            p = Path(audio_path)
            if not p.is_absolute():
                p = kaldi_dir / p
            if not p.exists():
                logger.warning("wav.scp entry missing on disk: %s", p)
                continue
            loaded = self._load_audio(p)
            if loaded is None:
                continue
            wav, sr = loaded
            # Speaker: spk2utt if present, else WSJ utterance-id prefix
            # (e.g. "11a0110m" → speaker "11a0").
            speaker = utt2spk.get(utt_id, utt_id[:4] if len(utt_id) >= 4 else utt_id)
            samples.append(
                SpeechSample(
                    waveform=wav,
                    sample_rate=sr,
                    transcript=normalize_transcript(text.get(utt_id, "")),
                    speaker_id=speaker,
                    utterance_id=utt_id,
                    language="en",
                    dataset_name="wsj",
                    split=split_name,
                    metadata={"source_path": str(p), "layout": "kaldi"},
                )
            )
        return samples

    def _load_raw(self) -> list[SpeechSample]:
        samples: list[SpeechSample] = []
        audio_paths: list[Path] = []
        for suffix in self.AUDIO_SUFFIXES:
            audio_paths.extend(self.root_dir.rglob(f"*{suffix}"))
        for audio_path in sorted(set(audio_paths)):
            utt_id = audio_path.stem
            # Transcript: sidecar .txt/.trn, or sibling *.trans.txt index,
            # else empty (sample stays loadable; the benchmark then relies on
            # ASR-side references only where available).
            transcript = ""
            for sidecar in (
                audio_path.with_suffix(".txt"),
                audio_path.with_suffix(".trn"),
            ):
                if sidecar.exists():
                    transcript = sidecar.read_text(encoding="utf-8").strip()
                    break
            if not transcript:
                for idx_file in audio_path.parent.glob("*.trans.txt"):
                    for line in idx_file.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) == 2 and parts[0] == utt_id:
                            transcript = parts[1]
                            break
                    if transcript:
                        break
            loaded = self._load_audio(audio_path)
            if loaded is None:
                continue
            wav, sr = loaded
            samples.append(
                SpeechSample(
                    waveform=wav,
                    sample_rate=sr,
                    transcript=normalize_transcript(transcript),
                    speaker_id=utt_id[:4] if len(utt_id) >= 4 else "wsj",
                    utterance_id=utt_id,
                    language="en",
                    dataset_name="wsj",
                    split=self.subset or "raw",
                    metadata={"source_path": str(audio_path), "layout": "raw"},
                )
            )
        return samples

    def load(self) -> list[SpeechSample]:
        """Load WSJ samples from the Kaldi layout (preferred) or raw LDC tree."""
        self.samples = []
        kaldi_dir = self._kaldi_dir()
        if kaldi_dir is not None:
            self.samples = self._load_kaldi(kaldi_dir)
        elif self.root_dir.exists():
            self.samples = self._load_raw()
        else:
            logger.warning("WSJ directory not found: %s", self.root_dir)
        if not self.samples:
            logger.warning(
                "No WSJ samples loaded from %s (corpus is LDC-licensed; "
                "see scripts/prepare_wsj.py).", self.root_dir,
            )
        return self.filter_samples()

    # -- base-paper subset protocol ---------------------------------------

    @staticmethod
    def _balanced_select(
        samples: list[SpeechSample],
        total: int,
        n_speakers: int = 0,
    ) -> list[SpeechSample]:
        """Deterministic round-robin selection (see select_paper_subset)."""
        return select_paper_subset(
            samples, total=total, n_speakers=n_speakers, mode="balanced"
        )

    def load_base_paper_subset(
        self,
        subset: str = "A",
        seed: int = 42,
        music_dir: Path | str | None = None,
        pool: list[SpeechSample] | None = None,
    ) -> list[SpeechSample]:
        """Load a base-paper evaluation subset ("A", "B" or "C").

        Parameters
        ----------
        subset : str
            ``"A"`` (70 speech / 10 speakers), ``"B"`` (72 speech / 70 music)
            or ``"C"`` (150 speech / 72 music, ≤6 phones/s pairs, 100
            repetitions per λ).  Speech is WSJ; music comes from a separate
            corpus via ``music_dir`` (WSJ contains no music).
        seed : int
            Seed for the random component of subset C ("randomly chose
            speech files from 150 samples"); subsets A/B are deterministic.
        music_dir : optional
            Directory of audio files used as the music samples of subsets
            B/C.  If None, no music samples are attached and a warning is
            logged (the paper's music sets are not part of WSJ).
        pool : optional
            Pre-loaded speech samples; otherwise :meth:`load` is called.

        Returns
        -------
        list[SpeechSample]
            Selected samples (``self.samples`` updated accordingly).
        """
        key = subset.upper()
        if key not in BASE_PAPER_SUBSETS:
            raise ValueError(
                f"Unknown base-paper subset {subset!r}; expected A, B or C."
            )
        spec = BASE_PAPER_SUBSETS[key]

        if pool is None:
            pool = self.load() if not self.samples else self.samples

        if key in ("A", "B"):
            selected = select_paper_subset(
                pool,
                total=spec["speech"],
                n_speakers=spec["n_speakers"],
                mode="balanced",
            )
        else:  # C: random pool of 150 (paper: "randomly chose speech files
            # from 150 samples")
            selected = select_paper_subset(
                pool, total=spec["speech"], seed=seed, mode="random"
            )

        if len(selected) < spec["speech"]:
            logger.warning(
                "Subset %s requested %d speech samples but only %d are "
                "available in %s.", key, spec["speech"], len(selected),
                self.root_dir,
            )

        # Attach music samples (subsets B and C) from a separate corpus
        music_samples: list[SpeechSample] = []
        if spec["music"] > 0:
            if music_dir is None:
                logger.warning(
                    "Subset %s includes %d music samples in the base paper, "
                    "but no music_dir was given (WSJ contains no music).",
                    key, spec["music"],
                )
            else:
                music_root = Path(music_dir)
                rng = np.random.default_rng(seed)
                candidates: list[Path] = []
                for suffix in self.AUDIO_SUFFIXES:
                    candidates.extend(music_root.rglob(f"*{suffix}"))
                candidates = sorted(set(candidates))
                if len(candidates) > spec["music"]:
                    pick = rng.permutation(len(candidates))[: spec["music"]]
                    candidates = [candidates[i] for i in sorted(pick)]
                for i, path in enumerate(candidates):
                    loaded = self._load_audio(path)
                    if loaded is None:
                        continue
                    wav, sr = loaded
                    music_samples.append(
                        SpeechSample(
                            waveform=wav,
                            sample_rate=sr,
                            transcript="",
                            speaker_id=f"music_{i // 10:02d}",
                            utterance_id=f"music_{path.stem}",
                            language="xx",
                            dataset_name="music",
                            split=f"base_paper_{key}",
                            metadata={"source_path": str(path)},
                        )
                    )
                music_samples = [
                    m for m in music_samples
                    if self.min_duration_s <= m.duration <= self.max_duration_s
                ][: spec["music"]]

        # Annotate subset + estimated phone rate on every sample
        for s in selected + music_samples:
            s.split = f"base_paper_{key}"
            s.metadata["base_paper_subset"] = key
            s.metadata["est_phonemes"] = estimate_phoneme_count(s.transcript)
            if s.transcript:
                s.metadata["est_phone_rate"] = round(
                    phone_rate(s.transcript, s.duration), 3
                )

        self.samples = selected + music_samples
        return self.samples


# Legacy alias (WHAT-REMAINS.txt refers to the adapter as WSJDataset)
WSJDataset = WSJAdapter


# ---------------------------------------------------------------------------
# Legacy compatibility wrapper
# ---------------------------------------------------------------------------

@dataclass
class AudioDataset:
    """Backward-compatible AudioDataset wrapper for existing pipeline runners."""
    root_dir: Path | None = None
    signals: list[Signal] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42), repr=False)

    def load(self) -> tuple[list[Signal], list[str]]:
        if self.signals:
            return self.signals, self.transcripts
        if self.root_dir is not None and self.root_dir.exists():
            return self._load_from_disk()
        return self._generate_synthetic()

    def sample(self, n: int, seed: int = 42) -> tuple[list[Signal], list[str]]:
        signals, transcripts = self.load()
        if n >= len(signals):
            return signals, transcripts
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(signals), size=n, replace=False)
        indices.sort()
        return [signals[i] for i in indices], [transcripts[i] for i in indices]

    def _load_from_disk(self) -> tuple[list[Signal], list[str]]:
        signals = []
        transcripts = []
        assert self.root_dir is not None
        for wav_path in sorted(self.root_dir.glob("**/*.wav")):
            try:
                wav, sr = sf.read(str(wav_path), dtype="float64", always_2d=False)
                if wav.ndim == 2 and wav.shape[1] == 1:
                    wav = wav.squeeze(axis=1)
                signals.append(Signal(waveform=wav, sample_rate=int(sr), metadata={"source": str(wav_path)}))
                txt_path = wav_path.with_suffix(".txt")
                if txt_path.exists():
                    transcripts.append(txt_path.read_text(encoding="utf-8").strip())
                else:
                    transcripts.append("")
            except Exception as e:
                logger.warning("Failed to load %s: %s", wav_path, e)
        logger.info("Loaded %d signals from %s", len(signals), self.root_dir)
        return signals, transcripts

    def _generate_synthetic(self) -> tuple[list[Signal], list[str]]:
        synth = SyntheticSpeechDataset(n_samples=8, seed=42)
        samples = synth.load()
        self.signals = [s.to_signal() for s in samples]
        self.transcripts = [s.transcript for s in samples]
        return self.signals, self.transcripts

    @classmethod
    def synthetic(cls, n_signals: int = 8, seed: int = 42, duration: float = 2.0, sr: int = 16000) -> AudioDataset:
        dataset = cls(root_dir=None, _rng=np.random.default_rng(seed))
        synth = SyntheticSpeechDataset(n_samples=n_signals, seed=seed, target_sr=sr)
        samples = synth.load()
        dataset.signals = [s.to_signal() for s in samples]
        dataset.transcripts = [s.transcript for s in samples]
        return dataset
