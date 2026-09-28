# HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

## Full reference
Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." As printed: arXiv:2106.07447v1 [cs.CL], 14 Jun 2021 (preprint). Index terms as printed: "Self-supervised learning, BERT." Code: https://github.com/pytorch/fairseq/tree/master/examples/hubert

## One-line contribution
HuBERT replaces wav2vec 2.0's carefully-tuned contrastive/quantization objective with an offline clustering step (k-means) that provides noisy frame-level targets for a BERT-like masked-prediction loss applied only on masked regions, matching or beating wav2vec 2.0 across every labeled-data regime from 10 minutes to 960 hours.

## Problem & motivation
Self-supervised speech learning faces three unique problems: (1) each utterance contains many different sound units (breaking the instance-classification assumption used in vision), (2) there is no lexicon of sound units during pre-training (unlike NLP word/piece targets), and (3) sound units have variable lengths with no explicit segmentation. Pseudo-labeling/self-training dominates but merely mimics a supervised teacher and is geared to a single downstream task. The authors hypothesize that the *consistency* of unsupervised targets, not their intrinsic correctness, is what matters, so even low-quality k-means labels can drive representation learning.

## Method
- Offline unit discovery: run k-means to get frame-level target labels z_t. Iteration 1: 100 clusters on 39-dim MFCCs (13 coefficients + 1st/2nd derivatives) over 960 h LibriSpeech; iteration 2: 500 clusters on 768-d features from the 6th Transformer layer of the iteration-1 model (fit on a random 10% of the data due to memory). MiniBatchKMeans (scikit-learn), mini-batch 10,000 frames, k-means++ with 20 random starts.
- Masked prediction: SpanBERT/wav2vec 2.0-style span masking (p% of frames start spans of length l); BERT Transformer consumes masked features and predicts cluster IDs.
- Loss: L = α·Lm + (1−α)·Lu where Lm = Σ_{t∈M} log p_f(z_t | X̃, t) over masked frames and Lu over unmasked frames. HuBERT sets **α = 1** (masked only), forcing the model to learn both an acoustic model of unmasked inputs and long-range temporal structure; α = 0 merely mimics the clusterer.
- Cluster ensembles (multiple k-means teachers, Eq. 2) and product k-means on spliced 117-d MFCCs (3 × 39-d subspaces, 100 entries each) as complementary targets; iterative refinement of targets from the model's own learned latents.
- Prediction head: p_f^(k)(c | X̃, t) = exp(sim(A^(k) o_t, e_c)/τ) / Σ_c' exp(sim(A^(k) o_t, e_c')/τ), cosine similarity, τ = 0.1, one projection matrix A^(k) per clustering model.
- Architecture follows wav2vec 2.0: CNN waveform encoder (7 × 512-channel layers, strides 5,2,2,2,2,2,2; kernels 10,3,3,3,3,2,2; 320× downsampling → 20 ms frame rate at 16 kHz) + BERT Transformer. Mask span l = 10, p = 8% of encoder frames. Fine-tuning: CTC over the whole network except the frozen CNN encoder, vocab = 26 English characters + space + apostrophe + CTC blank; freeze-step hyperparameter like wav2vec 2.0.
- Target-quality metrics: phone purity, cluster purity, and phone-normalized mutual information (PNMI = 1 − H(y|z)/H(y)) computed against forced-aligned frame-level phonetic transcripts from a hybrid ASR system.

## Key quantitative results
- Model sizes (Table I): BASE 95M (12 layers, dim 768, FFN 3,072, 8 heads, proj 256), LARGE 317M (24 layers, 1024, 4096, 16 heads, proj 768), X-LARGE 964M / described as "about 1 billion" / "1B parameter model" (48 layers, 1280, 5120, 16 heads, proj 1024).
- Low-resource (Table II, Libri-light 60k h unlabeled, Transformer LM): 10-min labels — HuBERT LARGE **4.7/7.6** test-clean/other vs wav2vec 2.0 LARGE 4.8/8.2; HuBERT X-LARGE **4.6/6.8**; HuBERT BASE (LS-960, 4-gram) 9.7/15.3 vs wav2vec 2.0 BASE 9.1/15.6. 1-hour — LARGE 2.9/5.4, X-LARGE 2.8/4.8. 10-hour — LARGE 2.4/4.6, X-LARGE 2.3/4.0. 100-hour — LARGE 2.1/3.9 (0.1 higher than wav2vec 2.0 LARGE's 2.0 on test-clean), X-LARGE 1.9/3.5.
- Abstract/intro claim: with the 1B-parameter model, **up to 19% (dev-other) and 13% (test-other) relative WER reduction** over LARGE models, pre-trained on Libri-light 60k h.
- High-resource, 960 h labeled (Table III): HuBERT LARGE dev 1.5/3.0, test **1.9/3.3**; HuBERT X-LARGE dev 1.5/2.5, test **1.8/2.9**; wav2vec 2.0 LARGE 1.6/3.0, 1.8/3.3; supervised Conformer L 1.9/3.9; Noisy Student 1.7/3.4; pre-trained Conformer XXL 1.5/3.1; wav2vec 2.0 + self-training 1.1/3.1 (test 1.5/3.1); pre-trained Conformer XXL + Noisy Student 1.3/2.6 → HuBERT lags behind pre-training + self-training combinations.
- K-means stability (Table IV, PNMI mean±std over 10 trials): MFCC K=100 → 0.251±0.001 / 0.253±0.001 / 0.253±0.001 for 1 h / 10 h / 100 h fitting data; MFCC K=500 → 0.283–0.287; BASE-it1-layer6 K=100 → 0.563±0.012 … 0.575±0.008; K=500 → 0.680±0.005 … 0.686±0.004 (max gain from 100× more fitting data = 0.012 PNMI).
- Loss-placement ablation (Table V, dev-other WER, 100k-step pre-training, 10 h fine-tune, n-gram LM): supervised chenone teacher (C = 8976, PNMI 0.809): α=1.0 → 10.38, α=0.5 → 9.16, α=0.0 → 9.79. K-means MFCC C=100 (PNMI 0.243): α=1.0 → **17.86**, α=0.5 → 29.57, α=0.0 → **96.37** (C=50: 18.68/31.07/94.60; C=500: 18.40/33.42/97.66). K-means BASE-it1-layer6 C=500 (PNMI 0.637): 11.91/13.47/23.29; BASE-it2-layer9 C=500 (PNMI 0.704): 10.75/11.59/13.79.
- Cluster ensembles (Table VI, dev-other WER): K-means{50,100} 17.81; {50,100,500} 17.56; Product K-means-0-100 19.26; -1-100 17.64; -2-100 18.46; Product K-means-{0,1,2}-100 **16.73** (best).
- Pre-training length (Table VII, dev-other WER, p = 6.5%): K-means 100 → 17.86 (100k) / 12.97 (250k) / 12.32 (400k) / **11.68** (800k steps); K-means 50 → 18.68 / 13.65 / 12.40 / 11.82; DiscreteBERT's 13.5k k-means units → 26.6.
- Clustering quality baselines: MFCC (cluster purity, phone purity, PNMI) = (0.099, 0.335, 0.255) at K=100 and (0.031, 0.356, 0.287) at K=500; HuBERT features are significantly better on all three, with BASE-it1 peaking around layer 6 and its last layers degrading dramatically, while BASE-it2 improves across layers.
- Ablations also show optimal mask fraction p = 8%, larger effective batch size (more GPUs) helps, longer training consistently helps.
- Compute: BASE pre-trained 2 iterations on 32 GPUs (250k then 400k steps; 100k steps ≈ 9.5 h); LARGE / X-LARGE on 128 / 256 GPUs for 400k steps on 60k h (labeled as effectively "third iteration" models, targets from layer 9 of the 2nd-iteration BASE). Peak LR 5e-4 / 1.5e-3 / 3e-3.

## Datasets / corpora used
- Pre-training: LibriSpeech 960 h audio (no transcripts) or Libri-light 60,000 h audio (both derived from LibriVox).
- Fine-tuning: Libri-light 10-minute / 1-hour / 10-hour splits and LibriSpeech train-clean-100 (100 h) and full 960 h (train-clean-100 + train-clean-360 + train-other-500).
- Evaluation: LibriSpeech dev-clean, dev-other, test-clean, test-other; decoding with n-gram and Transformer LMs trained on the official LibriSpeech language-modeling data (wav2letter++ beam search, hyper-parameters tuned with Ax Bayesian optimization).
- Analysis: 39-dim MFCCs; forced-aligned frame-level phonetic transcripts from a hybrid ASR (for purity/PNMI).

## Models / systems evaluated
HuBERT BASE / LARGE / X-LARGE (own); compared with DiscreteBERT, wav2vec 2.0 BASE / LARGE, DeCoAR 2.0, iterative pseudo-labeling (IPL), slimIPL, Noisy Student, supervised Conformer L, pre-trained Conformer XXL, wav2vec 2.0 + self-training, pre-trained Conformer XXL + Noisy Student; supervised "chenone" forced-alignment teacher as a target-quality top line.

## Human-study details
None — no human participants or listening tests; all results are ASR WER/PER or target-quality statistics (purity, PNMI).

## Limitations acknowledged by authors
- Trails methods that combine pre-training *with* self-training on the 960 h setup (though the authors expect combining HuBERT with self-training to close/beat the gap).
- HuBERT BASE is occasionally slightly worse than wav2vec 2.0 BASE (e.g., 10-min test-clean 9.7 vs 9.1; 100-h test-other 8.1 vs 8.0) and HuBERT LARGE is 0.1 WER higher on 100-h test-clean.
- Training requires multiple clustering/training iterations (and re-clustering); future work: a single-phase procedure.
- Memory limits forced fitting k-means on a random 10% of the data for second-generation targets (MiniBatchKMeans still has to load the dataset into memory first — noted in a footnote).
- Analysis/downstream coverage limited to ASR in this paper; other recognition and generation tasks left for future work.

## Relevance to our project
HuBERT supplies both a modeling hypothesis and an evaluation-style template for our CAPTCHA work. Its central insight — that target *consistency* rather than target correctness drives learning, and that masked-only prediction forces a model to internalize acoustic + temporal structure — implies that modern ASR encoders are optimized to recover phonetic units from context even when local acoustics are degraded, i.e., they are structurally robust to exactly the kind of band-limited psychoacoustic perturbations a CAPTCHA might add, which helps explain a large Human–ASR Gap in our favor of machines unless distortions corrupt phonetic content itself. Methodologically, the paper's PNMI/phone-purity/cluster-purity metrics quantify "how much phonetic content survives in the representation," which is a representation-space analogue of our STOI-based human intelligibility proxy; reporting both next to ASR WER would let us locate the gap in phonetic space rather than only at the transcript. The Table V result — that WER explodes to 96.37–97.66 when the loss includes unmasked frames with poor targets but falls to ~10–11 with good targets — is also a concrete warning that ASR robustness numbers are inseparable from training/decoding configuration, reinforcing our need to fix Whisper/Vosk settings across distortion conditions.

## Keywords
self-supervised learning, masked prediction, hidden units, k-means clustering, cluster ensembles, iterative refinement, BERT, phone-normalized mutual information, acoustic unit discovery, LibriSpeech, Libri-light, word error rate, CTC fine-tuning, representation learning, self-training
