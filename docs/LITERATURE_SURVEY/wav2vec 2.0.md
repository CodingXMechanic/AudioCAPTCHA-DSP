# wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

## Full reference
Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (Facebook AI). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." As printed on the PDF: "Preprint. Under review." arXiv:2006.11477v3 [cs.CL], 22 Oct 2020. Code: https://github.com/pytorch/fairseq

## One-line contribution
A self-supervised framework that masks latent speech representations and solves a contrastive task over jointly learned discrete (quantized) speech units, showing that pre-training on unlabeled audio followed by CTC fine-tuning outperforms the best semi-supervised ASR methods while using orders of magnitude less labeled data.

## Problem & motivation
Labeled transcribed speech is expensive and unavailable for most of the ~7,000 languages spoken worldwide; current ASR needs thousands of hours of transcribed speech. Self-supervised learning offers a way to learn general speech representations from unlabeled audio and then fine-tune on small labeled sets, mirroring how infants learn by listening. Prior speech pre-training either used a two-step pipeline (quantize first, then contextualize) or reconstructed filter-bank features; the authors want a single end-to-end objective that learns both contextualized representations and discrete speech units.

## Method
- Multi-layer convolutional feature encoder f: raw waveform X → latent representations z1..zT (7 blocks, 512 channels, strides (5,2,2,2,2,2,2), kernels (10,3,3,3,3,2,2); output at 49 Hz, 20 ms stride, 25 ms receptive field); input waveform normalized to zero mean/unit variance.
- Context network g: Transformer over the continuous encoder output with a convolutional relative positional embedding (kernel 128, 16 groups) instead of absolute embeddings.
- Masking in latent space: sample p = 0.065 of time steps as span starts, mask next M = 10 steps → ~49% of all time steps masked, mean span 14.7 steps = 299 ms (median 10, max ~100 for a 15 s sample).
- Quantization module: product quantization with G = 2 codebooks × V = 320 entries (theoretical max 102.4k codewords), selected differentiably with Gumbel softmax (straight-through estimator); temperature τ annealed 2 → 0.5 (BASE) / 0.1 (LARGE).
- Objective: L = Lm + α·Ld (α = 0.1), with contrastive loss Lm = −log exp(sim(ct, qt)/κ) / Σ_q̃∼Qt exp(sim(ct, q̃)/κ) (cosine similarity, κ = 0.1, K = 100 distractors sampled from other masked steps of the same utterance) and a codebook diversity loss Ld maximizing entropy/perplexity of the averaged codebook distribution.
- Fine-tuning: linear projection on top of the Transformer + CTC loss (29 characters + word boundary for Librispeech), feature encoder frozen, modified SpecAugment masking on encoder outputs (time-steps and channels).
- Sizes: BASE = 12 Transformer blocks, dim 768, FFN 3,072, 8 heads, 95M params; LARGE = 24 blocks, dim 1,024, FFN 4,096, 16 heads, 317M params.

## Key quantitative results
(All WER/PER as printed; clean = test-clean, other = test-other of Librispeech unless stated.)
- Full 960 h labeled Librispeech (Table 2, Transformer LM): LARGE pre-trained on LV-60k → dev 1.6/3.0, **test 1.8/3.3 WER**; BASE LS-960 → test 2.1/4.8; LARGE LS-960 → test 2.0/4.1; LARGE trained from scratch → test 2.1/4.6. Comparators: ContextNet 1.9/4.1, Conformer 1.9/3.9, Noisy Student 1.7/3.4, Iterative pseudo-labeling (LV-60k) 2.10/4.01.
- 10 min labeled (Table 1): LARGE LV-60k + Transformer LM → dev 4.6/7.9, **test 4.8/8.2**; LARGE LS-960 → test 6.8/10.8; BASE LS-960 + 4-gram → test 9.1/15.6 vs Discrete BERT 16.3/25.2. 10 min = 48 recordings, avg 12.5 s. Without LM/lexicon (Table 9): LARGE LV-60k test 40.2/38.7 (BASE LS-960 46.9/50.9).
- 1 h labeled: LARGE LV-60k → test 2.9/5.8; LARGE LS-960 → test 3.9/7.6, which the text says improves 7%/12% over iterative self-training with two orders of magnitude less labeled data.
- 10 h labeled: LARGE LV-60k → test 2.6/4.9; LARGE LS-960 → test 3.2/6.1 (text: 24%/29% relative error reduction vs iterative self-training at 100 h).
- 100 h labeled (Table 1): LARGE LV-60k → test 2.0/4.0; LARGE LS-960 → test 2.3/5.0 vs Noisy Student 4.2/8.6 (text: **45%/42% relative WER reduction**); Discrete BERT 4.5/12.1.
- TIMIT phoneme recognition, no LM (Table 3): LARGE (LS-960) **dev PER 7.4 / test PER 8.3**, vs vq-wav2vec 9.6/11.6, wav2vec 12.9/14.7, Li-GRU+fMLLR 14.9 → text states 23%/29% relative PER reduction over next best.
- Ablation (Table 4, avg WER ± std on dev-clean+dev-other, 3 seeds, reduced setup): continuous inputs + quantized targets (baseline) **7.97 ± 0.02**; quantized inputs + quantized targets 12.18 ± 0.41; quantized inputs + continuous targets 11.18 ± 0.16; continuous inputs + continuous targets 8.58 ± 0.08. Training accuracy of picking the correct latent rises 62% → 78.0% when switching from quantized to continuous targets.
- Masking ablation (Table 5): baseline (p = 0.075) 7.97; M = 8 → 8.33, M = 15 → 8.43; non-overlapping length-10 masking → 9.15, length-15 → 9.43.
- Other ablations (Table 13): α = 0 → 8.48, α = 0.2 → 8.58; no Gumbel noise → 8.73; K = 200 negatives → 8.12; negatives from batch → 8.79; G = 4, V = 18 → 9.02; G = 8, V = 8 → 8.13.
- Discrete units vs phonemes (Appendix D): on TIMIT train (3,696 utterances, avg 13.6 s, 563k discrete latents), many latents specialize in specific phonemes; silence phoneme bcl = 22% of annotated speech.
- Training cost: BASE 64 V100 GPUs for 1.6 days (total batch 1.6 h); LARGE 128 V100 for 2.3 days (LS-960) / 5.2 days (LV-60k), total batch 2.7 h.

## Datasets / corpora used
- Unlabeled pre-training: Librispeech without transcriptions (LS-960, 960 h) and LibriVox/Libri-light audio (LV-60k, 53.2k h after preprocessing of Kahn et al. 2020).
- Labeled fine-tuning: Librispeech 960 h; Libri-light limited splits train-10min, train-1h, train-10h; Librispeech train-clean-100 (100 h); TIMIT (5 h, labels collapsed to 39 phone classes).
- Evaluation: standard Librispeech dev-clean/dev-other/test-clean/test-other.
- Language models: 4-gram and Transformer LM (20 blocks, dim 1,280, FFN 6,144, 16 heads) trained on the Librispeech LM corpus; weights tuned by Bayesian optimization (128 trials).

## Models / systems evaluated
Own: wav2vec 2.0 BASE (95M) and LARGE (317M). Compared against: Discrete BERT, vq-wav2vec, wav2vec, iterative pseudo-labeling, Noisy Student, Hybrid DNN/HMM (RWTH), TTS data augmentation, supervised CTC Transformer, seq2seq Transformer, Transformer Transducer, ContextNet, Conformer, PASE+, Li-GRU + fMLLR, CNN + TD-filterbanks.

## Human-study details
None — no human participants, listening tests, or intelligibility judgments are reported; all evaluation is automatic (WER, PER).

## Limitations acknowledged by authors
- Their architecture is a "weaker baseline": a simple Transformer with CTC underperforms seq2seq models; supervised from-scratch LARGE gets 2.1/4.6 vs ContextNet 1.9/4.1.
- Acoustic model outputs characters while the LM predicts words, delaying LM feedback — likely detrimental; most recent work uses word pieces for both.
- No data balancing (unlike Noisy Student); self-training is likely complementary to pre-training and the combination may do better.
- Authors expect gains from switching to a seq2seq architecture and a word-piece vocabulary.

## Relevance to our project
This paper defines the kind of modern ASR backbone whose robustness we are probing: Whisper/Vosk-class systems and SSL-pretrained encoders are trained to keep phonetic content recoverable from raw audio, and the ablation showing that continuous targets retain "speaker and background information" (making the pretext task easier but hurting generalization) is a reminder that these models deliberately discard nuisance acoustic detail — exactly the channel that psychoacoustic hiding (Schönherr et al.) exploits to bury adversarial energy without changing perceived speech. Two concrete methodological lessons transfer to our Human–ASR Gap metric: (1) decoding configuration swings WER enormously (10-min LV-60k test-clean is 40.2 WER without LM vs 4.8 with a Transformer LM), so ASR settings must be frozen across clean vs distorted CAPTCHA conditions or the gap is an artifact of decoding, not of distortion; (2) the paper's reporting style — WER paired with dataset, labeled-data budget, model size, and LM — is the level of experimental specification our distortion-condition results should carry. Its TIMIT discrete-latent/phoneme co-occurrence analysis also offers a template for a representation-level "phonetic content preserved" probe alongside our STOI-based human intelligibility proxy.

## Keywords
self-supervised learning, speech representations, contrastive learning, masked prediction, product quantization, Gumbel-softmax, CTC fine-tuning, low-resource ASR, word error rate, LibriSpeech, TIMIT, Transformer encoder, pre-training, phoneme recognition, language model fusion
