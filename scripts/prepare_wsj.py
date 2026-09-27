#!/usr/bin/env python3
"""Prepare the Wall Street Journal (WSJ) corpus for AudioCAPTCHA-DSP.

WSJ is the **base-paper dataset**: Schönherr et al. (2018, arXiv:1808.05665)
used "the default settings of the Wall Street Journal (WSJ) training recipe
of the Kaldi toolkit" and evaluated on subsets of one WSJ test set
(70 utterances / 10 speakers; 72-speech/70-music; 150-speech/72-music with
audio-text pairs of at most 6 phones per second).

WSJ is **LDC-licensed and cannot be redistributed**.  This script never
downloads it; it only verifies, converts and indexes a copy you obtained
yourself:

  * LDC93S6B  (WSJ0)  and/or LDC94S37A  (WSJ1)  from https://catalog.ldc.upenn.edu
  * or a Kaldi installation that already ran the ``wsj`` recipe
    (kaldi/egs/wsj/s5), which produces ``data/train_si284``,
    ``data/test_dev93``, ``data/test_eval92`` … with ``wav.scp``/``text``.

Usage
-----
  python scripts/prepare_wsj.py --check   ROOT [--subset test_dev93]
  python scripts/prepare_wsj.py --convert ROOT --out OUT
  python scripts/prepare_wsj.py --build-kaldi ROOT --out OUT
  python scripts/prepare_wsj.py --subsets ROOT

``--check``        reports layout, speaker/utterance counts and problems.
``--convert``      converts Sphere ``.wv1/.wv2`` audio to ``.wav``
                   (soundfile → sph2pipe → ffmpeg, first that works).
``--build-kaldi``  writes a Kaldi-style ``wav.scp`` + ``text`` (+ ``spk2utt``)
                   index next to converted audio, which ``WSJAdapter``
                   loads natively.
``--subsets``      prints the base-paper subset requirements against the
                   available pool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AUDIO_SUFFIXES = (".wv1", ".wv2", ".wav", ".flac", ".au")


def _iter_audio(root: Path) -> list[Path]:
    found: list[Path] = []
    for suffix in AUDIO_SUFFIXES:
        found.extend(root.rglob(f"*{suffix}"))
    return sorted(set(found))


def _read_transcript_for(audio_path: Path) -> str:
    utt_id = audio_path.stem
    for sidecar in (audio_path.with_suffix(".txt"), audio_path.with_suffix(".trn")):
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()
    for idx_file in audio_path.parent.glob("*.trans.txt"):
        for line in idx_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0] == utt_id:
                return parts[1]
    return ""


def _speaker_of(utt_id: str) -> str:
    return utt_id[:4] if len(utt_id) >= 4 else "wsj"


def convert_audio(src: Path, dst: Path) -> str:
    """Convert one audio file to wav; returns the backend used or ''."""
    import numpy as np
    import soundfile as sf

    dst.parent.mkdir(parents=True, exist_ok=True)
    # 1) libsndfile may read the file directly (incl. some Sphere files)
    try:
        wav, sr = sf.read(str(src), dtype="float64")
        sf.write(str(dst), wav, sr)
        return "soundfile"
    except Exception:
        pass
    # 2) sph2pipe (Kaldi's Sphere decoder)
    import shutil
    import subprocess

    sph2pipe = shutil.which("sph2pipe")
    if sph2pipe:
        try:
            out = subprocess.run(
                [sph2pipe, "-f", "wav", str(src)],
                capture_output=True, check=True,
            )
            dst.write_bytes(out.stdout)
            return "sph2pipe"
        except Exception:
            pass
    # 3) ffmpeg (works for wav/flac/au inputs)
    import shutil as _shutil

    ffmpeg = _shutil.which("ffmpeg")
    if ffmpeg:
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(src), "-ar", "16000", str(dst)],
                capture_output=True, check=True,
            )
            return "ffmpeg"
        except Exception:
            pass
    return ""


def cmd_check(args: argparse.Namespace) -> int:
    from audiocaptcha_dsp.evaluation.dataset import WSJAdapter

    root = Path(args.root)
    adapter = WSJAdapter(root, subset=args.subset)
    kaldi_dir = adapter._kaldi_dir()
    if kaldi_dir is not None:
        print(f"layout      : kaldi ({kaldi_dir})")
        scp = adapter._read_index(kaldi_dir, "wav.scp")
        text = adapter._read_index(kaldi_dir, "text")
        spk2utt = adapter._read_index(kaldi_dir, "spk2utt")
        speakers = set()
        for spk, utts in spk2utt.items():
            speakers.update(utts.split())
        if not speakers:
            speakers = {u[:4] if len(u) >= 4 else u for u in scp}
        print(f"utterances  : {len(scp)}")
        print(f"transcripts : {len(text)}")
        print(f"speakers    : {len(speakers)}")
        missing = [
            u for u in scp
            if not (kaldi_dir / (adapter._wav_scp_path(scp[u]) or "")).exists()
        ]
        print(f"missing wav : {len(missing)}")
    else:
        audio = _iter_audio(root)
        print("layout      : raw LDC tree")
        print(f"audio files : {len(audio)}")
        sphere = [p for p in audio if p.suffix in (".wv1", ".wv2")]
        print(f"sphere files: {len(sphere)} (need conversion: --convert)")
        with_txt = [p for p in audio if _read_transcript_for(p)]
        print(f"transcripts : {len(with_txt)}/{len(audio)}")
        if not audio:
            print(
                "NOT FOUND — WSJ is LDC-licensed; see the module docstring "
                "for acquisition options."
            )
            return 1
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out = Path(args.out)
    files = [p for p in _iter_audio(root) if p.suffix in (".wv1", ".wv2")]
    if not files:
        print("no Sphere (.wv1/.wv2) files found — nothing to convert")
        return 0
    ok, failed = 0, 0
    for src in files:
        rel = src.relative_to(root)
        dst = out / rel.with_suffix(".wav")
        backend = convert_audio(src, dst)
        if backend:
            ok += 1
            if ok % 50 == 0:
                print(f"converted {ok}/{len(files)} …")
        else:
            failed += 1
            print(f"FAILED: {src} (need sph2pipe or ffmpeg)")
    print(f"converted {ok}/{len(files)} ({failed} failed) → {out}")
    return 0 if failed == 0 else 1


def cmd_build_kaldi(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audio = _iter_audio(root)
    if not audio:
        print(f"no audio found under {root}")
        return 1
    scp_lines, text_lines, spk_map: dict[str, list[str]] = {}, {}, {}
    for p in audio:
        utt = p.stem
        rel = p.relative_to(root) if p.is_relative_to(root) else p
        scp_lines[utt] = str(rel)
        transcript = _read_transcript_for(p)
        if transcript:
            text_lines[utt] = transcript
        spk_map.setdefault(_speaker_of(utt), []).append(utt)
    (out / "wav.scp").write_text(
        "".join(f"{u} {p}\n" for u, p in sorted(scp_lines.items())),
        encoding="utf-8",
    )
    if text_lines:
        (out / "text").write_text(
            "".join(f"{u} {t}\n" for u, t in sorted(text_lines.items())),
            encoding="utf-8",
        )
    (out / "spk2utt").write_text(
        "".join(
            f"{s} {' '.join(sorted(u))}\n" for s, u in sorted(spk_map.items())
        ),
        encoding="utf-8",
    )
    print(
        f"wrote Kaldi layout → {out} "
        f"({len(scp_lines)} utts, {len(text_lines)} transcripts, "
        f"{len(spk_map)} speakers)"
    )
    return 0


def cmd_subsets(args: argparse.Namespace) -> int:
    from audiocaptcha_dsp.evaluation.dataset import (
        BASE_PAPER_SUBSETS,
        WSJAdapter,
    )

    adapter = WSJAdapter(args.root, subset=args.subset)
    pool = adapter.load()
    n_spk = len({s.speaker_id for s in pool})
    print(f"available pool: {len(pool)} utterances, {n_spk} speakers")
    for key, spec in BASE_PAPER_SUBSETS.items():
        need_spk = spec["n_speakers"]
        ok_spk = n_spk >= need_spk if need_spk else True
        ok_n = len(pool) >= spec["speech"]
        status = "OK " if (ok_spk and ok_n) else "LOW"
        print(
            f"  subset {key}: {spec['speech']} speech"
            + (f" / {need_spk} speakers" if need_spk else "")
            + (f" / {spec['music']} music (separate corpus)" if spec["music"] else "")
            + f"  [{status}]"
        )
    print(
        "\nNote: WSJ contains no music; subsets B/C additionally need a music "
        "corpus (the base paper does not name its music source)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="WSJ root directory (or Kaldi recipe data dir)")
    parser.add_argument("--subset", default=None, help="Kaldi data dir name, e.g. test_dev93")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify layout and counts")
    mode.add_argument("--convert", action="store_true", help="convert .wv1/.wv2 to .wav")
    mode.add_argument("--build-kaldi", action="store_true", help="write wav.scp/text/spk2utt")
    mode.add_argument("--subsets", action="store_true", help="report base-paper subset needs")
    parser.add_argument("--out", default=None, help="output directory for --convert/--build-kaldi")
    args = parser.parse_args(argv)

    if args.convert or args.build_kaldi:
        if not args.out:
            parser.error("--convert/--build-kaldi require --out")
    if args.convert:
        return cmd_convert(args)
    if args.build_kaldi:
        return cmd_build_kaldi(args)
    if args.subsets:
        return cmd_subsets(args)
    return cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
