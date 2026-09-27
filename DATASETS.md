# Datasets

Dataset layer of AudioCAPTCHA-DSP. The fidelity contract (verbatim base-paper
facts, exact-vs-approximated table, manifest schema) lives in
[`docs/BASE_PAPER_DATASET.md`](docs/BASE_PAPER_DATASET.md); this file
describes the layer itself.

---

## 1. Corpora

| Corpus | Role | Status | Location |
|---|---|---|---|
| **WSJ** (LDC, Kaldi WSJ recipe layout or raw LDC) | Exact base-paper corpus | **Implemented (adapter); corpus not present (licence)** | `data/raw/wsj` expected by `--dataset wsj` |
| **LibriSpeech test-clean** | Labelled public stand-in for WSJ | **Implemented, present (2 620 files)** | `data/raw/LibriSpeech/test-clean` |
| **CommonVoice / FLEURS** | Additional adapters (WHAT-REMAINS §4) | **Implemented (adapters)**; corpora not downloaded | set `--data-root` |
| Music corpus (subsets B/C) | Paper used 70/72 music files, source unnamed | **Optional** — requires `--music-dir`; omitted-and-labelled otherwise | any flat audio dir |

## 2. Adapters (`evaluation/dataset.py`)

| Adapter | Layout handled | Notes |
|---|---|---|
| `WSJAdapter` | Kaldi `wsj/…/{test_dev93,test_eval92}` + raw LDC `wv1/trn/wrd` | subset preference: explicit → `test_dev93` → `test_eval92` → root; also loads music when `--music-dir` given |
| `LibriSpeechAdapter` | `root/split/reader/chapter/*.flac` + `.trans.txt` | normalized transcripts; records `source_path` metadata (workers read from disk) |
| `CommonVoiceAdapter` | CSV + audio clips | same `SpeechSample` contract |
| `FLEURSAdapter` | HF-style parquet/arrow or directory export | same `SpeechSample` contract |

All adapters return `SpeechSample` (waveform, sr, transcript, speaker_id,
utterance_id, duration auto-computed, metadata with `source_path`) and pass
`filter_samples` (duration 0.5–15 s, no clipping ≥ 0.999, RMS ≥ 1e-4).

## 3. Base-paper subset protocol

`BASE_PAPER_SUBSETS` + `select_paper_subset` + `WSJAdapter.load_base_paper_subset`
reproduce the paper's selection verbatim:

| Subset | Speech | Speakers | Mode | Extra constraint | Music |
|---|---|---|---|---|---|
| A | 70 utts | 10 | balanced (7 each) | — | — |
| B | 72 utts | 10 | balanced (7–8 each) | — | 70 files (paper) |
| C | 150 utts | — | random, seeded | ≤ 6 phones/s (`phone_rate`) | 72 files (paper) |

Phone estimation: `estimate_phoneme_count` (≈ 0.96 phonemes/letter,
1.5/spoken digit) — labelled *estimated*; the paper's phonemizer was not
published.

Standard test sets: `KALDI_WSJ_TEST_SETS = ("test_dev93", "test_eval92")`.

## 4. Provenance — manifest or it did not happen

Every benchmark run writes:

```
<run>/dataset_manifest.json
  corpus, base_paper, protocol, adapter, root,
  selection {mode, seed, total, n_speakers},
  speakers[], utterances[{id, speaker, duration, transcript, source_path}],
  n_samples, corpus_sha256, music, created
<run>/run_manifest.json
  args, conditions, ASR engine labels, HSR label,
  git_commit, python, platform, package versions
```

`corpus_sha256` hashes `(utterance_id | speaker | transcript)` over the
selection — identical selections produce identical hashes, so two runs can be
proven to have used the same data.

## 5. Stand-in policy (enforced, not aspirational)

- Headline runs on LibriSpeech carry
  `corpus = "librispeech test-clean (stand-in for WSJ: LDC-licensed)"` in
  **both** manifests and in every generated table caption.
- Conclusions that depend on the corpus being exactly WSJ are marked as such
  in `docs/BASE_PAPER_DATASET.md` §"re-run on WSJ".
- Subsets B/C on LibriSpeech omit music and say so
  (`"music": "omitted (no music corpus supplied)"`).

## 6. Commands

```powershell
# what do I have / what's missing / convert LDC / build Kaldi index / show subsets
python scripts/prepare_wsj.py --check
python scripts/prepare_wsj.py --convert   --wsj-root D:\corpora\wsj0
python scripts/prepare_wsj.py --build-kaldi --wsj-root D:\corpora\wsj0 --out data/raw/wsj
python scripts/prepare_wsj.py --subsets

# benchmark against either corpus with the identical protocol
python scripts/run_comparative_benchmark.py --dataset librispeech --subset A ...
python scripts/run_comparative_benchmark.py --dataset wsj         --subset A --data-root data/raw/wsj
```
