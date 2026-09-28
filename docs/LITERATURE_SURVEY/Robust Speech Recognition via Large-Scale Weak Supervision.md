# Robust Speech Recognition via Large-Scale Weak Supervision

## Full reference
- **Title:** Robust Speech Recognition via Large-Scale Weak Supervision
- **Authors (as printed):** Alec Radford *, Jong Wook Kim *, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (* equal contribution)
- **Affiliation:** OpenAI, San Francisco, CA 94110, USA
- **Venue / year:** Proceedings of the International Conference on Machine Learning 2022 (ICML 2022) — printed in the PDF's document metadata (`/Subject: Proceedings of the International Conference on Machine Learning 2022`); no PMLR page numbers, arXiv ID, or DOI are printed on the pages themselves.
- Code/models URL printed in the paper: https://github.com/openai/whisper (model name footnote: "WSPSR — Web-scale Supervised Pretraining for Speech Recognition" can be used as an acronym).

## One-line contribution
Show that simply supervising a single encoder–decoder Transformer on **680,000 hours** of weakly (web-)labelled multilingual, multitask audio — with *no* self-supervision, self-training, or dataset-specific fine-tuning — yields zero-shot ASR that matches or beats prior fully-supervised systems and approaches human accuracy *and* human-level robustness across distributions.

## Problem & motivation
- Unsupervised pre-training (wav2vec 2.0) scaled to 1,000,000 h of unlabelled speech but has no comparably strong decoder, so it still needs **dataset-specific fine-tuning** — a complex, expert-only process that rewards brittle, dataset-specific quirks rather than generalization (cited example: +9.2 % ImageNet accuracy with no gain on seven other image datasets).
- Supervised multi-domain mixing (SpeechStew: 7 datasets, 5,140 h) and weakly supervised sets (10,000 h / 30,000 h) exist but are far smaller than unlabelled data.
- Goal: one system that works "out of the box" across environments; measure it **zero-shot**, so evaluation measures out-of-distribution generalization (as with humans) rather than in-distribution fit.

## Method (precise but simple language, key equations)
1. **Data:** audio–transcript pairs scraped from the internet; minimal text normalization (predict raw transcript text; no inverse text normalization step). Automated filters remove machine-generated ("transcript-ese") captions, mismatched-language pairs (audio language detector fine-tuned on VoxLingua107 vs CLD2), and fuzzy-duplicate transcripts; a first-trained model's per-source error rates guide manual removal of misaligned/low-quality sources; transcript-level de-duplication against TED-LIUM 3. Audio is cut into **30-second segments** with the transcript text inside the segment; non-speech segments are kept (sub-sampled) as VAD training data.
2. **Model:** off-the-shelf **encoder–decoder Transformer**. Audio resampled to **16 kHz**, **80-channel log-magnitude Mel spectrogram**, **25 ms window / 10 ms stride**, inputs globally scaled to [−1, 1] with ≈ 0 mean. Encoder stem = 2 Conv1D layers (filter width 3, GELU, second stride 2) + sinusoidal position embeddings; pre-activation residual blocks; decoder uses learned position embeddings and tied input–output token embeddings; encoder and decoder have the same width and block count. GPT-2 byte-level BPE tokenizer (refit for multilingual).
3. **Multitask format:** all tasks/conditioning are a token sequence for the decoder: `<|startoftranscript|>` → language token (99 total) → `<|nospeech|>` / `<|transcribe|>` / `<|translate|>` → `<|notimestamps|>` or 20 ms-quantized timestamp tokens interleaved with text → `<|endoftranscript|>`. Prior transcript text is occasionally prepended as context (rate 50 %); loss is masked over that context.
4. **Training:** AdamW (β1 0.9, β2 0.98, ε 1e-6, weight decay 0.1), linear LR decay after 2048 warmup updates, batch 256 segments, **2^20 = 1,048,576 updates** (2–3 passes over the data), FP16, **no data augmentation or regularization** in V1 (Large V2 adds SpecAugment, Stochastic Depth, BPE Dropout, batch 1024, 655,360 updates, 2.5× more epochs).
5. **Evaluation discipline:** zero-shot on every benchmark (no training data from that dataset), with an extensively hand-built **text standardizer** before WER (12 English rules: strip bracketed/parenthetical phrases, drop filler words, expand contractions, strip symbols/diacritics, number/currency normalization, British→American spelling, …).

## Key quantitative results
- **Table 2 — effective robustness (WER %, after text normalizer), zero-shot Whisper Large V2 vs supervised wav2vec2-large-960h:**
  | Dataset | wav2vec2 Large (no LM) | Whisper Large V2 | Rel. error reduction |
  |---|---|---|---|
  | LibriSpeech Clean | 2.7 | 2.7 | 0.0 |
  | Artie | 24.5 | 6.2 | 74.7 |
  | Common Voice | 29.9 | 9.0 | 69.9 |
  | Fleurs En | 14.6 | 4.4 | 69.9 |
  | Tedlium | 10.5 | 4.0 | 61.9 |
  | CHiME6 | 65.8 | 25.5 | 61.2 |
  | VoxPopuli En | 17.9 | 7.3 | 59.2 |
  | CORAAL | 35.6 | 16.2 | 54.5 |
  | AMI IHM | 37.0 | 16.9 | 54.3 |
  | Switchboard | 28.3 | 13.8 | 51.2 |
  | CallHome | 34.8 | 17.6 | 49.4 |
  | WSJ | 7.7 | 3.9 | 49.4 |
  | AMI SDM1 | 67.6 | 36.4 | 46.2 |
  | LibriSpeech Other | 6.2 | 5.2 | 16.1 |
  | **Average** | **29.3** | **12.8** | **55.2** |
  → equal on the reference split (2.7 vs 2.7) but **55.2 % fewer errors on average** out of distribution.
- Best zero-shot model has "relatively unremarkable" **LibriSpeech test-clean WER = 2.5** (beam search + temperature fallback; greedy = 2.7; test-other 4.9 beam / 5.2 greedy); **smallest model (39 M params, tiny) = 6.7 WER on LibriSpeech test-clean**, yet roughly competitive with the best supervised LibriSpeech model on other datasets.
- Context numbers: Deep Speech 2 reported 5.3 % on LibriSpeech test-clean in 2015 with human-level 5.8 %; SOTA has since dropped **73 %** to **1.4 %** — yet supervised models remain far above human error out of distribution; supervised LibriSpeech models make **roughly twice as many errors as a human** on other datasets (Fig. 2), while the zero-shot Whisper robustness frontier **includes the 95 % CI for the human** tested (Alec).
- **Multilingual (Table 3):** MLS WER — Zero-shot Whisper **7.3** vs XLS-R (1B) 10.9, mSLAM-CTC (2B) 9.7; VoxPopuli — Whisper **13.6** vs Maestro 8.1, mSLAM 9.1, XLS-R 10.6, VP-10K+FT 15.3 (Whisper underperforms here).
- **Translation (Table 4, CoVoST2 X→en BLEU):** Whisper **29.1 All / 36.2 High / 32.6 Mid / 25.2 Low** vs XMEF-X 14.7, XLS-R 22.1, mSLAM 24.8, Maestro 25.2 → new zero-shot SOTA overall (+6.7 BLEU over mSLAM on low-resource), driven by 68,000 h of X→en data vs 861 h in CoVoST2; slight underperformance on high-resource.
- **Language ID (Table 5, Fleurs):** Whisper **64.5** vs mSLAM-CTC 77.7, w2v-bert-51 71.4 → **13.6 %** behind supervised SOTA; bounded at 80.4 % because 20 of 102 Fleurs languages have no Whisper training data; **80.3 %** on the 82 overlapping languages.
- **Scaling:** WER vs training hours per language on Fleurs — **r² = 0.83** (log–log), WER **halves every 16×** more data; translation r² = only **0.24** (Fig. 4, e.g. Welsh at 13 BLEU despite ~9,000 h because most "Welsh" data was mislabelled English).
- **Dataset scaling (Table 6, medium model):** hours → (English WER / multilingual WER / X→en BLEU): 3,405 → 30.5/92.4/0.2; 6,811 → 19.6/72.7/1.7; 13,621 → 14.4/56.6/7.9; 27,243 → 12.3/45.0/13.9; 54,486 → 10.9/36.4/19.2; 681,070 → **9.9/29.2/24.8** (diminishing returns past 54 k h).
- **Additive-noise robustness (§3.7, Fig. 5):** WER on LibriSpeech test-clean vs SNR from +40 dB to −10 dB for **14 LibriSpeech-trained models** (12 LibriSpeech-trained + 2 NVIDIA STT) under **white noise** and **pub noise** (Audio Degradation Toolbox). Many models beat zero-shot Whisper at low noise (40 dB SNR), but **all are worse than Whisper below SNR = 10 dB under pub noise**.
- **Long-form (Fig. 6 / Table 7 / Table 16):** 7 datasets, 4 commercial ASR services (queried **September 1, 2022**) + NVIDIA STT Conformer-CTC Large (best open source). Whisper beats NVIDIA STT on **all seven** datasets and most commercial ones; heuristic ablation average WER: greedy **11.0** → +beam 10.6 → +temperature fallback 10.6 → +VAD 10.2 → +previous-text 10.0 → +initial-timestamp constraint **10.0**.
- **Text normalization:** standardization cuts WER by **up to 50 percent** on several datasets (e.g. contraction/whitespace quirks); FairSpeech normalizer comparison (§4.4) shows the biggest Whisper-specific gains on WSJ, CallHome, Switchboard.
- **Training-data composition (Fig. 11):** 680,000 h total = **438,218 h (65 %) English ASR**, **125,739 h (18 %) translation**, **117,113 h (17 %) multilingual ASR**; 117,000 h cover **96 other languages**; speech-recognition training data for **75 languages** (e.g. Chinese 23,446 h, German 13,344 h, Spanish 11,100 h; Lao/Sundanese/Burmese 0.1 h).
- **Multitask transfer (Fig. 9):** only **65 % of compute** goes to English ASR in joint training; small models show *negative* transfer (English-only better at equal FLOPs), largest joint models outperform English-only.

## Datasets / corpora used
- **Training:** 680,000 h of weakly labelled internet audio (web transcripts), 30 s segments; language labels from a VoxLingua107-fine-tuned detector; de-duplicated against TED-LIUM 3.
- **Short-form English evaluation (Appendix A.1):** LibriSpeech test-clean/test-other, TED-LIUM 3 test, Common Voice 5.1 (English), Artie bias corpus, CallHome + Switchboard (LDC2002S09/LDC2002T43), WSJ (LDC93S6B, LDC94S13B, s5 recipe), CORAAL (231 interviews), CHiME-6, AMI-IHM and AMI-SDM1.
- **Long-form English (Appendix A.2):** TED-LIUM 3 (11 full talks), **Meanwhile** (64 Late Show segments), **Rev16** (16 of 30 Rev.AI podcast episodes), **Kincaid46** (46 files; 25 used for the human study), Earnings-21 / Earnings-22, CORAAL (231 interviews).
- **Multilingual:** Multilingual LibriSpeech, Fleurs (102 languages), VoxPopuli (16 languages), Common Voice 9, CoVoST2 (X→en).
- **Noise robustness:** Audio Degradation Toolbox — additive white noise and "pub noise".
- **Contamination control:** transcript-level de-duplication of the training set against TED-LIUM 3.

## Models / systems evaluated
- **Whisper model family (Table 1):** Tiny 4 layers/384 width/6 heads/**39M**; Base 6/512/8/**74M**; Small 12/768/12/**244M**; Medium 24/1024/16/**769M**; Large 32/1280/20/**1550M** — plus English-only `.en` variants and **Large V2**.
- **Baselines (Appendix B):** wav2vec2-base-100h / base-960h / large-960h / large-960h-lv60-self / large-robust-ft-libri-960h, HuBERT large/xlarge, fairseq s2t medium/large, UniSpeech-SAT base, SpeechBrain CRDNN+RNNLM and Transformer+TransformerLM, NVIDIA STT Conformer-CTC Large and Conformer-Transducer X-Large.
- **Multilingual/translation baselines:** XLS-R (1B/2B), mSLAM-CTC (2B), Maestro, XMEF-X, VP-10K+FT, w2v-bert-51 (0.6B).
- **Long-form:** 4 unnamed commercial ASR services (default English settings, queried 2022-09-01) + NVIDIA STT.

## Human-study details (if any: n participants, protocol, key numbers)
No perceptual listening study was run. The human comparison (§3.9, Fig. 7) used **25 recordings from Kincaid46** (scripted/unscripted broadcast, telephone, VoIP, meetings) transcribed by **5 professional services** — one computer-assisted and four fully human. The computer-assisted service had the lowest aggregate WER, **1.15 percentage points better than Whisper**, and pure-human performance was "**only a fraction of a percentage point** better than Whisper's" → "Whisper's English ASR performance is not perfect but very close to human-level accuracy." Separately, Fig. 2 compares zero-shot Whisper against a **single zero-shot human ("Alec")** on LibriSpeech dev-clean vs average WER on [Common Voice, CHiME-6, TED-LIUM]: supervised LibriSpeech models make roughly twice the human's errors out of distribution, while the zero-shot Whisper robustness frontier contains the human's 95 % confidence interval.

## Limitations acknowledged by authors
From §6 "Limitations and Future Work" (plus caveats elsewhere):
- **Decoding failure modes remain "non-human/perceptual":** repeat loops, skipping the first/last words of a segment, and complete **hallucination** of text unrelated to the audio — beam search/temperature fallback only mitigates them; fine-tuning or RL for decoding is proposed.
- **Low-resource languages are still poor** (most languages have < 1000 h; data-heavy languages dominate), worsened by an English-centric collection pipeline.
- **Only zero-shot was studied** — no fine-tuning, so no direct comparison to prior work in the common setting.
- **Encoder vs decoder contribution unclear** (robustness may come from the audio-conditional LM decoder); no unsupervised pre-training/self-training used, though adding auxiliary objectives might help.
- **Text normalizer risk:** developed jointly with Whisper, may be over-fitted to Whisper's transcription style (mitigated by comparison with FairSpeech's normalizer).
- **WER itself** penalizes innocuous style differences — an acknowledged metric problem for zero-shot systems.
- Possible **contamination of commercial ASR results** (commercial services may have trained on the public long-form datasets); translation data quality issues (language-ID errors, e.g. Welsh); dataset-scaling returns have flattened since 54 k h (under-training vs end-of-scaling unresolved); no claim of SOTA on MLS because a simple text standardizer is used.

## Relevance to our project (one specific paragraph)
Whisper is one of the two ASR backbones in AudioCAPTCHA-DSP, so this paper defines the *machine* side of our Human–ASR Gap and, importantly, supplies the evaluation philosophy. Three concrete transfers: (a) **zero-shot, out-of-distribution evaluation** is the point — the paper's core argument is that in-distribution numbers overstate machine ability, and a CAPTCHA is precisely an out-of-distribution stimulus for any ASR, so we should report Whisper WER on our distorted prompts *without* any fine-tuning and be explicit about decoding configuration (greedy vs beam + temperature fallback changed LibriSpeech test-clean from 2.7 to 2.5); (b) its **additive-noise robustness sweep (WER vs SNR from 40 to −10 dB with white and pub noise, Fig. 5)** is the closest published analogue to our distortion-strength sweeps — we can present our DSP transforms on the same WER-vs-severity axes and use their finding that models degrade fast below 10 dB SNR as a sanity reference for how much distortion is needed to move a strong ASR; (c) the **text-normalization caveat (up to 50 % WER reduction from formatting rules alone)** is a direct warning for our HAG metric: human-scored transcription and machine WER must be computed under the same normalization, or part of the measured gap is an artifact. Finally, their human-comparison protocol (professional transcription services on 25 fixed recordings, differences reported in *percentage points*) is a methodological model for anchoring our STOI-derived human-intelligibility proxy against real listener data, and their conclusion that modern ASR is near human accuracy on clean speech but brittle under shift is exactly the asymmetry our CAPTCHA design tries to exploit.

## Keywords
Whisper, weak supervision, large-scale supervised pre-training, zero-shot ASR, robustness, out-of-distribution generalization, word error rate, text normalization, encoder–decoder Transformer, multilingual speech recognition, speech translation, noise robustness, human-level accuracy, effective robustness, multitask learning, long-form transcription
