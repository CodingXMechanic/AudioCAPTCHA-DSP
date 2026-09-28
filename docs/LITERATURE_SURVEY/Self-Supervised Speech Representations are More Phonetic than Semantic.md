# Self-Supervised Speech Representations are More Phonetic than Semantic

## Full reference
Kwanghee Choi¹, Ankita Pasad², Tomohiko Nakamura³, Satoru Fukayama³, Karen Livescu², Shinji Watanabe¹ — ¹Carnegie Mellon University, ²Toyota Technological Institute at Chicago, ³National Institute of Advanced Industrial Science and Technology (AIST). "Self-Supervised Speech Representations are More Phonetic than Semantic." As printed: arXiv:2406.08619v1 [cs.CL], 12 Jun 2024 (preprint). Index terms as printed: "self-supervised learning, model analysis, lexical semantics, phonological distance." Code: https://github.com/juice500ml/phonetic_semantic_probing

## One-line contribution
Using a curated dataset of near-homophone (phonetically similar) and synonym (semantically similar) word pairs, the authors show that self-supervised speech model (S3M) word representations are consistently and significantly more phonetic than semantic in every layer, and that a trivial bag-of-words baseline matches/beats S3Ms on intent classification — so high scores on those "semantic" benchmarks do not demonstrate semantic knowledge.

## Problem & motivation
Prior analyses of S3Ms either only study phonetics or merely show that various linguistic properties are *present*. It remains unknown whether S3M representations encode phonetics or semantics better, or equally well. Separately, intent classification (IC) datasets such as Fluent Speech Commands (FSC) show >99% frozen-S3M accuracy, making the task look solved; the authors question whether those scores reflect word *meaning* at all. Presence of information (usually shown with learnable probes) is not the same as how representations are *distanced*, so the authors probe raw geometry directly.

## Method
- Synonyms: WordNet synsets (all members of a synset treated as synonyms) via NLTK; crosslingual synonyms via Open Multilingual Wordnet v1.4 (OMW).
- Near homophones: phonemize words with the CMU pronouncing dictionary (English) / Epitran (other languages), compute Levenshtein distance normalized by the longer word (range 0–1); threshold set at d ≤ 0.4 because the top 0.1% of random LibriSpeech word pairs have phonetic distance ≤ 0.4. Example pairs (Table 1): [oysters, stirs (0.20)], [dripping, drilling (0.17)], [mind, mound (0.25)], [bread, braids (0.4)], [socks, saxon (0.5)], [caffeine, patio (0.6)], [morally, sausage (0.67)], [spring, constrain (0.75)], [euclid, jack (0.83)], [cherry, shrank (1.0)].
- Five word-pair conditions: Random (lower bound), Same word (upper bound), Same speaker, Synonym, Near homophone (crosslingual analysis drops Same speaker and restricts pairs to English + one non-English language).
- Representations: layer-wise features of wav2vec2.0-Base/-Large, HuBERT-Base/-Large, XLS-R-300M, WavLM-Large; similarity = cosine similarity (present in all these models' pre-training losses). Because raw cosine values sit near 1.0 due to anisotropy of Transformer embeddings, all reported curves subtract the Random baseline ("Norm.").
- Slicing: feature slicing (feed whole utterance, slice the word's time span) vs audio slicing (feed only the word segment, removing surrounding context). Pooling: mean (default), center (temporally middle frame), centroid (frame most similar to all others).
- Statistics: bootstrap — randomly choose 10K word utterances, repeat the experiment five times, report the average with 95% confidence interval.
- Intent classification: utterances reduced to a bag of words (word identity only, no meaning), classified with a decision tree; S3Ms classified with a single fully-connected layer on frozen features; original splits plus the challenging speaker-split (SPK) and utterance-split (UTT) splits of Arora et al.

## Key quantitative results
- Main finding (Figures 1, 3): ordering **near-homophones > synonyms > random** cosine similarity holds in all layers of all six S3Ms (excluding cases where confidence intervals overlap) — representations are more phonetic than semantic everywhere; same trend in crosslingual MSW pairs (Figure 4), including English-only WavLM-Large on cross-language pairs (suggesting "limited translation abilities").
- Feature slicing squashes representations (absolute similarities nearly identical across pair types; normalized differences on the order of the 0.00–0.04 range in Figure 1's axis), so all subsequent analysis uses audio slicing; raw similarities crowd near 1.0 (anisotropy), making the random-baseline subtraction necessary.
- Pooling: center pooling retains more same-word identity, while mean and centroid pooling retain more synonym (semantic) content; mean pooling used thereafter.
- Layer trends align with prior work: HuBERT-Large and WavLM-Large keep phonetic information until the last layer; wav2vec2.0-Large encodes most information in the middle layers. Speaker information diminishes in later layers.
- Intent classification: frozen-S3M accuracy on FSC reported in prior work [SUPERB, ref 4] is **over 99%**; the authors' bag-of-words decision-tree baseline **achieves 100% accuracy on the original and the speaker (SPK) split of FSC**, and almost always outperforms or matches S3Ms on both IC datasets (Figure 6). WavLM-Large (layer index 21) is the best-performing S3M among those tested.
- Dataset sizes used: FSC = 30K utterances, 31 unique intents; Snips Smartlights (SNIPS) = 1660 utterances, 6 intents.
- Sampling/protocol numbers: 10K word utterances × 5 bootstrap repeats (English); 2K word utterances per language × 7 languages = 14K utterances × 5 repeats (crosslingual: English, Chinese, Italian, Spanish, Indonesian, Polish, Swedish); homophone threshold 0.4 = top 0.1% of random LibriSpeech word pairs; single-speaker check uses the five speakers with most utterances, speaker IDs 5142, 2412, 6313, 1580, 2277 — results "nearly identical" to the full-data analysis with higher variance for random/synonym pairs.
- Note: most quantitative layer-wise outcomes are presented only as figures (no numeric tables), so exact per-layer similarity values are not printed in the text.

## Datasets / corpora used
LibriSpeech dev-clean and test-clean (English word-pair analysis) with existing Montreal Forced Aligner (MFA) word timestamps; Multilingual Spoken Words dataset (MSW, 1-second word slices from Common Voice) for crosslingual analysis; WordNet + Open Multilingual WordNet v1.4 (synonyms); CMU pronouncing dictionary and Epitran (phonemization); Fluent Speech Commands (FSC) and Snips Smartlights (SNIPS) for intent classification, including original/SPK/UTT splits.

## Models / systems evaluated
wav2vec2.0-Base, wav2vec2.0-Large, HuBERT-Base, HuBERT-Large, XLS-R-300M (multilingual), WavLM-Large — all frozen, layer-wise features. Baselines: bag-of-words decision tree (word identity only, no semantics) and a single fully-connected layer on top of each frozen S3M.

## Human-study details
None — no human participants, no listening or intelligibility protocol; all measurements are automatic (cosine-similarity probes and classification accuracy).

## Limitations acknowledged by authors
- Audio-sliced representations do not differentiate word *senses* (multiple meanings of a word), which require surrounding context; the analysis therefore excludes that case.
- Feature slicing "squashes" representations so information may be present but not linearly/ geometrically accessible ("existence of information is not equivalent to how representations are distanced"), meaning distance-based conclusions are method-dependent.
- Results depend on slicing/pooling choices (center vs mean vs centroid give different phonetic/semantic balances); mean pooling is chosen for the rest of the analysis.
- The crosslingual setup is constrained: only 7 languages covered by both OMW and Epitran, always English paired with one non-English language, no Same-speaker condition, and only 2K utterances per language.
- The conclusion about intent classification is phrased as "not necessarily indicative" of semantic capability — a benchmark-validity claim, not a claim that S3Ms lack semantics.

## Relevance to our project
This paper is a caution about baselines and about what our metrics actually measure — both central to our Human–ASR Gap. First, the phonetic-dominance result predicts the direction of our HAG: ASR systems built on wav2vec2/HuBERT-style backbones keep phonetic content close together in representation space while giving semantics comparatively little geometric weight, so CAPTCHA distortions that preserve phonetic cues (masking-threshold-limited, psychoacoustic hiding) should degrade human listeners far less than machines — or conversely, distortions aimed at phonetic corruption should hurt both, and our STOI-based human proxy (itself a phonetic-intelligibility measure) may track the ASR side more closely than expected; comparing STOI against ASR WER per distortion is exactly how we can test that alignment. Second, the paper's methodological lessons transfer directly to our evaluation framework: (a) always subtract a trivial baseline — raw similarity scores sat near 1.0 and looked "solved" until a random baseline was removed, just as a near-ceiling STOI or WER needs a floor/ceiling reference before interpretation; (b) a deliberately naive baseline (bag of words) beat sophisticated frozen representations on FSC, echoing the need in our HAG to compare Whisper/Vosk against simple reference systems so the reported gap is not an artifact of a weak or mis-specified comparator; (c) benchmark validity must be checked — 99%+ accuracy on FSC did not imply semantics, just as low WER on clean audio does not imply robustness to our DSP distortions.

## Keywords
self-supervised speech models, phonetic similarity, semantic similarity, near-homophones, synonyms, word representations, probing methodology, anisotropy, cosine similarity, intent classification, bag-of-words baseline, layer-wise analysis, LibriSpeech, multilingual probing, representation geometry
