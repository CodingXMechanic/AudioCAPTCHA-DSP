# Mathematical Analysis

**Depth document for the AudioCAPTCHA-DSP paper (professor-requested foundation).**
Every formula below is tied to a number we actually computed; every number was
re-verified against `results/comparative/main/summary.csv` and `rows.csv` on
**2026-10-06** (verification scripts in Appendix C). Companion documents:
`STATISTICAL_ANALYSIS.md` (machinery inventory), `docs/COMPARATIVE_ANALYSIS.md`
(findings), `REPORT.md` (claims).

**How to read this document.** Each section gives (1) the mathematics, (2) an
*in plain words* explanation, (3) the verified numbers from our runs.

**Honesty labels used throughout** (never drop them when quoting):

- **HSR / HAG human half** = *illustrative (STOI-derived proxy; no human
  study conducted)* — the label is printed in every results file.
- **Corpus** = LibriSpeech stand-in (WSJ adapter ready, LDC licence blocked).
- **Forecasts** = scenarios, not measured trends.

---

## Table of contents

1. Notation and system model
2. Machine axis: WER, ΔWER, cross-ΔWER
3. Human axis: STOI, HSR, asr_sr, and the HAG gap
4. Psychoacoustics: the admissible-noise set and the λ dial
5. The power-matched control (a counterfactual in one equation)
6. Correlation and consistency statistics
7. Inference: paired tests, effect sizes, multiple comparisons
8. Multi-objective structure: Pareto frontier and the design menu
9. Forecasting: scaling law, break capacity, shelf life
10. End-to-end worked example
11. Honesty constraints: what the mathematics does *not* prove
12. Viva formula card
13. Appendix A: file map · B: reproducibility snippets · C: verification log

---

## 1. Notation and system model

### 1.1 Signals and experiment space

| Symbol | Meaning |
|---|---|
| $x_i \in \mathbb{R}^{N_i}$ | clean utterance $i$ (one recorded spoken sentence), $i = 1..n$, $n = 70$ |
| $\mathcal{S}$ | the 10 speakers of the test subset (70 utterances / 10 speakers = base-paper protocol) |
| $\Theta$ | set of **conditions**: $\lvert\Theta\rvert = 147$ (118 transforms + sweep variants + control + clean original) |
| $\theta \in \Theta$ | one condition = one transform with one parameter setting |
| $T_\theta : \mathbb{R}^{N} \to \mathbb{R}^{N}$ | the **attack operator**: the signal-processing function that maps clean audio to distorted audio, $y_i^{(\theta)} = T_\theta(x_i)$ |
| $\mathcal{E}$ | engine set (headline: 2 recognizers; validation: 4) |
| $e \in \mathcal{E}$ | one ASR engine (Whisper tiny, Vosk small, …) |
| $W_e(y, x)$ | word error rate of engine $e$ on clip $y$ against reference transcript $x$ |
| $S(y, x)$ | STOI-type intelligibility score in $[0,1]$ (our proxy implementation) |
| id | the identity operator (clean original, $T_{\text{id}} = I$) |

**In plain words:** we have 70 sentences; 147 different ways to mangle each
one; two (later four) machines that transcribe them; and one formula that
estimates how well a *person* would understand each mangled version. The
mathematics below is about how to turn those pieces into honest numbers.

### 1.2 The experiment as a function evaluation

Every condition is evaluated on the **identical** utterance set (this pairing
is what makes almost all inference below *paired*):

$$
\text{condition } \theta \;\longmapsto\;
\Big(\underbrace{S_i(\theta),\; i{=}1..70}_{\text{human axis}},\;
\underbrace{W_{e,i}(\theta),\; e \in \mathcal{E},\; i{=}1..70}_{\text{machine axis}}\Big)
$$

The **row count check**: $147 \times 70 = 10{,}290$ (headline rows, all `ok`);
validation $11 \times 70 = 770$; shelf-life ladder $5 \times 70 \times 3 + \dots = 1{,}260$.
All three verified 2026-10-06.

### 1.3 The four formal questions

- **Q1 (damage).** How large is $W_e(\theta) - W_e(\text{id})$? → §2
- **Q2 (human cost).** How large is $S(\theta)$ vs $S(\text{id}) = 1$? → §3
- **Q3 (window).** How large is the *difference* between human and machine
  success? → §3.4 (HAG)
- **Q4 (structure).** Which conditions dominate which (Pareto), how do
  results correlate (§6), are they significant (§7), and when do they expire
  (§9)?

---

## 2. Machine axis: WER, ΔWER, cross-ΔWER

### 2.1 Word error rate as a normalized edit distance

Given reference transcript $R = (r_1,\dots,r_L)$ (length $L$ words) and
engine output (hypothesis) $H$, let $S, D, I$ be the numbers of substitutions,
deletions and insertions in the minimum-cost alignment of $H$ to $R$:

$$
\mathrm{WER} = \frac{S + D + I}{L}
$$

**In plain words:** count how many words the machine got wrong — wrong word,
missing word, or made-up word — and divide by how many words there should be.

**Lemma 2.1 (WER is unbounded above).** $\mathrm{WER} \in [0, \infty)$, and
values $> 1$ are possible.

*Proof.* The alignment cost of a hypothesis with $M$ insertions (engine
babble, reference empty overlap) is $I = M$ with $L$ fixed, so
$\mathrm{WER} = M/L \to \infty$ as $M \to \infty$. Nothing in the definition
caps the hypothesis length. $\square$

**Consequence for reading our tables:** ΔWER = 1.483 does *not* mean
"148% of the sentence destroyed" — it means **+148 percentage points of
extra errors relative to the baseline**. Our own Whisper baseline:
$\hat b_{\text{tiny}} = 0.0790$, $\hat b_{\text{vosk}} = 0.1178$
(verified from `summary.csv`: 1.1696938 − 1.0906564 = 0.0790374;
0.9657276 − 0.8478993 = 0.1178283).

### 2.2 The per-condition estimator

For condition $\theta$, engine $e$, over the $n = 70$ paired utterances:

$$
\widehat{\Delta}_e(\theta)
= \frac{1}{n}\sum_{i=1}^{n} \Big( W_{e,i}(\theta) - W_{e,i}(\text{id}) \Big)
$$

**Properties** (i.i.d. utterances, treating the 70 as a sample from the
speaker/sentence population):

- unbiased: $\mathbb{E}[\widehat{\Delta}_e] = \Delta_e$;
- standard error $\mathrm{SE} = s_d / \sqrt{n}$ where $s_d$ is the std-dev of
  the *per-utterance differences* (pairing removes utterance difficulty —
  this is why we report paired, not pooled, uncertainty);
- 95 % CI $\approx \widehat{\Delta} \pm 1.994 \cdot \mathrm{SE}$
  ($t_{0.975, 69} = 1.994$).

**In plain words:** we don't compare "average of attacked" to "average of
clean" as unrelated groups — we subtract *sentence by sentence*, which
cancels the fact that some sentences are simply harder. The noise in the
comparison shrinks by $\sqrt{70} \approx 8.4\times$.

### 2.3 cross-ΔWER: the macro-average over engines

$$
\text{cross-}\Delta\mathrm{WER}(\theta)
= \frac{1}{\lvert\mathcal{E}\rvert} \sum_{e \in \mathcal{E}} \widehat{\Delta}_e(\theta)
$$

**Worked check (row 1 of the headline table):**

$$
\text{cross-}\Delta\mathrm{WER}(\texttt{spectral.minimum\_phase})
= \frac{2.084 + 0.882}{2} = 1.483 \;\checkmark
$$

**Design-based, not population-based.** With $\lvert\mathcal{E}\rvert = 2$
(and 4 in validation) the average is over a *fixed, deliberately chosen*
engine set — an equal-weight macro-average (each engine 50 %, because we
assert no probability distribution over "engines in the wild"). It is
**not** an estimate of $\mathbb{E}[\Delta]$ over all ASRs ever built; that
claim would need engines sampled from a defined population. We state this
whenever cross-ΔWER is interpreted: *an attack ranks high only if it fools
both chosen machines*, and §6 quantifies how much the two agree.

### 2.4 Why subtract the baseline at all

Model an engine's observed error as $W_e(\theta) = b_e + \delta_e(\theta) +
\varepsilon$, where $b_e$ is its intrinsic error on clean speech (0.079 /
0.118 — *different by 49 % between engines*). Reporting $W_e(\theta)$ alone
would rank engines, not attacks; $\widehat{\Delta}$ removes $b_e$ so the
attack's effect is comparable across engines of unequal strength.

---

## 3. Human axis: STOI, HSR, asr_sr, and the HAG gap

### 3.1 STOI: envelope correlation as an intelligibility estimate

The published Short-Time Objective Intelligibility measure (Taal et al.,
2011) computes, per critical band $j$ and analysis frame $k$, the Pearson
correlation between the temporal envelopes of clean and processed speech:

$$
\rho_{j,k} = \frac{\big(\tilde{e}^{\,(x)}_{j,k} - \bar{e}^{\,(x)}_{j}\big)
\big(\tilde{e}^{\,(y)}_{j,k} - \bar{e}^{\,(y)}_{j}\big)}
{\sqrt{\big(\tilde{e}^{\,(x)}_{j,k} - \bar{e}^{\,(x)}_{j}\big)^2}\;
\sqrt{\big(\tilde{e}^{\,(y)}_{j,k} - \bar{e}^{\,(y)}_{j}\big)^2}},
\qquad
\mathrm{STOI} = \frac{1}{JK}\sum_{j=1}^{J}\sum_{k=1}^{K} \rho_{j,k}
$$

**In plain words:** chop the audio into frequency bands; in each band and
time window, ask "does the loudness pattern still have the same *shape* as
the original?" Correlation 1 = identical shape. Average over all bands and
windows.

**Honesty note:** our implementation is the short-time envelope-correlation
**proxy** consistent with that literature (see `METRICS.md`), reported
everywhere as `stoi_proxy` — a validated model of human intelligibility, not
a human measurement. External validation vs real listeners: §6.3.

### 3.2 HSR: an explicit, visible affine map

$$
\mathrm{HSR} = \mathrm{clip}\big(0.6 + 0.4 \cdot S,\; 0,\; 1\big)
\qquad\text{(code: `hag_metrics.py`; label: illustrative proxy)}
$$

Properties, all worth stating at a viva:

| Property | Math | Meaning |
|---|---|---|
| endpoints | $S{=}0 \Rightarrow \mathrm{HSR}{=}0.6$; $S{=}1 \Rightarrow \mathrm{HSR}{=}1$ | floor 0.6 = a determined human still passes even on degraded audio (conservative for security claims) |
| slope | $\partial \mathrm{HSR}/\partial S = 0.4$ | 1 point of STOI = 0.4 points of pass-rate |
| error propagation | $\sigma_{\mathrm{HSR}} = 0.4\,\sigma_S$ | STOI noise is *damped* 2.5× in HSR |
| range | $\mathrm{HSR} \in [0.6, 1]$ for $S \in [0,1]$ | never claims humans score below 0.6 |

**Worked check:** $S = 0.8290895714285713 \Rightarrow 0.6 + 0.4 \times
0.8290895714285713 = 0.931636 \to$ stored as **0.9316** ✓ (exact row value).

**Why this is legitimate and not "made up":** the map is one straight line,
its two anchor points are declared, its slope appears in the code, and it
is **never used as the subject of inferential statistics** (rule from
`STATISTICAL_ANALYSIS.md` §8). The day real listeners are run, their
measured pass-rates replace the right-hand side *without changing any other
formula*.

### 3.3 asr_sr: the machine pass-rate as a binomial proportion

$$
\mathrm{asr\_sr}(\theta) = \frac{1}{n\lvert\mathcal{E}\rvert}
\sum_{e \in \mathcal{E}} \sum_{i=1}^{n}
\mathbf{1}\!\left[ W_{e,i}(\theta) \le \tau \right],
\qquad \tau = 0.30
$$

**In plain words:** count how many clips the machine still "gets" (no more
than 30 % words wrong) out of all 140 trials (70 clips × 2 engines), and
divide. It is a coin-flip proportion, so it obeys binomial statistics:

$$
\mathbb{E}[\mathrm{asr\_sr}] = p, \qquad \mathrm{Var} = \frac{p(1-p)}{n|\mathcal{E}|}
$$

**Wilson score interval** (preferred over the normal approximation for small
$p$ or $p$ near 0/1):

$$
\mathrm{CI}_{1-\alpha} =
\frac{\hat p + \tfrac{z^2}{2N} \pm z \sqrt{\tfrac{\hat p(1-\hat p)}{N} + \tfrac{z^2}{4N^2}}}{1 + \tfrac{z^2}{N}},
\qquad N = n|\mathcal{E}|,\; z = 1.96
$$

**Verified values:**

- `novel.captcha_optimal`: $\hat p = 1/140 = 0.00714$ → Wilson 95 % CI
  **[0.0013, 0.0393]**. One success in 140 trials; even the upper bound
  says < 4 % machine success.
- illustrative $k = 49, n = 70$ ($\hat p \approx 0.70$, the ladder regime):
  Wilson 95 % CI **[0.585, 0.795]**.

**Threshold honesty:** $\tau = 0.30$ is a *declared convention* ("still
understands" = ≥ 70 % words right), not a theorem. Every HAG number inherits
$\tau$; sensitivity: lowering $\tau$ raises asr_sr, lowering HAG.

### 3.4 The HAG gap: definition, bounds, sensitivities

$$
\boxed{\;\mathrm{HAG}(\theta) = \mathrm{HSR}(\theta) - \mathrm{asr\_sr}(\theta)\;}
$$

**Theorem 3.1 (range).** $\mathrm{HAG} \in [-1, +1]$.

*Proof.* HSR ∈ [0,1] (clip), asr_sr ∈ [0,1] (mean of indicators). Difference
of two numbers in [0,1] lies in [−1, 1]. $\square$

**Reading the scale (three regimes):**

| HAG | Regime |
|---|---|
| → +1 | perfect CAPTCHA: humans pass, machines fail |
| ≈ 0 | useless: both succeed or both fail together |
| < 0 | inverted: machines do better than humans |

**Worked example (`novel.captcha_optimal`, exact row from `summary.csv`):**

$$
S = 0.82909 \Rightarrow \mathrm{HSR} = 0.9316,\quad
\mathrm{asr\_sr} = \tfrac{1}{140} = 0.0071,\quad
\mathrm{HAG} = 0.9316 - 0.0071 = \mathbf{0.9245}\;\checkmark
$$

**Sensitivities (the full differential):**

$$
d\,\mathrm{HAG} = 0.4 \cdot dS \;-\; d\,\mathrm{asr\_sr}
\qquad\Rightarrow\qquad
\frac{\partial\,\mathrm{HAG}}{\partial S} = 0.4,\quad
\frac{\partial\,\mathrm{HAG}}{\partial\,\mathrm{asr\_sr}} = -1
$$

- +0.01 STOI → **+0.004** HAG (human side is damped)
- +1 pp machine pass-rate → **−0.01** HAG (machine side is full-weight)
- therefore HAG is *more* sensitive to the machine axis by design: security
  is decided by whether the machine still passes.

**Uncertainty:** HAG combines a continuous estimate (STOI) and a binomial
proportion (asr_sr) on the *same* clips — the clean instrument is the
**paired bootstrap** (resample the 70 utterance indices, recompute both
halves, percentile interval; §7.3), not a naive quadrature that would
assume independence.

---

## 4. Psychoacoustics: the admissible-noise set and the λ dial

### 4.1 The masking budget in code = the math in the paper

The psychoacoustic constraint is implemented as a **per-band budget**:

```python
# src/audiocaptcha_dsp/psychoacoustics/masking_models.py:414
budget_db = masking_db - margin_db          # λ ≡ margin_db ≥ 0
# extensions.py:110
thresh_db  = spread_db - margin_db
# novel/phoneme_aware.py:60 — linear-domain equivalent
scale = 10.0 ** (-margin_db / 20.0)
```

Formally, with masking threshold $T_m(f)$ (spread excitation of the speech
masker, optionally combined with the absolute quiet threshold $T_q(f)$),
the noise $n$ is **admissible under λ** iff its band levels obey

$$
L_{n}(f_k) \;\le\; T_m(f_k) - \lambda \quad \forall k,
\qquad \lambda \ge 0 \text{ in dB}
$$

Define the **admissible set**

$$
\mathcal{A}_\lambda = \big\{\, n : L_n(f_k) \le T_m(f_k) - \lambda \;\;\forall k \,\big\}
$$

**In plain words:** hearing science computes, for every frequency, how loud
noise may be before a person notices it. λ is a *safety subtraction* in
decibels: λ = 0 means "up to the limit", λ = 50 dB means "50 dB quieter
than the limit". **Larger λ = stricter = quieter noise.**

### 4.2 Lemma (nestedness ⇒ monotone attack strength)

**Lemma 4.1.** If $\lambda_1 < \lambda_2$ then
$\mathcal{A}_{\lambda_2} \subseteq \mathcal{A}_{\lambda_1}$.

*Proof.* The right-hand side decreases by the constant λ at every band, so
every $n$ satisfying the tighter bound also satisfies the looser one.
$\square$

**Corollary (monotonicity of the sweep).** The attacker maximizes damage
over $n \in \mathcal{A}_\lambda$; the feasible set shrinks as λ grows, so
the optimum objective value is **non-increasing in λ** (a maximum over a
smaller set cannot grow).

**Empirical confirmation — our sweep (verified 2026-10-06):**

| λ (dB) | 0 | 5 | 10 | 30 | 40 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| cross-ΔWER | 0.3618 | 0.1047 | 0.0451 | 0.0085 | 0.0085 | 0.0018 |
| STOI | 0.8819 | 0.9509 | 0.9818 | 0.9998 | 1.0000 | 1.0000 |

Non-increasing damage, non-decreasing intelligibility — exactly the
corollary's prediction. (λ = 30 vs 40 is numerically equal to 3 dp: the
corollary promises *non-increasing*, not strictly decreasing ✓.)

### 4.3 The dB arithmetic (why margins multiply)

Decibels for power ratios: $L = 10 \log_{10}(P/P_0)$. Therefore a *subtraction*
of λ dB in the log domain is a *division* by $10^{\lambda/10}$ in the linear
domain — the two forms above (`budget_db - margin_db` vs
`10 ** (-margin_db/20)` on amplitudes, where amplitude ratio $10^{-\lambda/20}$
= power ratio $10^{-\lambda/10}$) are the same statement.

### 4.4 λ ≡ the base paper's dial

Their single knob is our column `margin_db`; the difference is **coverage**:
they swept λ for one method; the rows above sweep it for
`masked_noise`, `bark_perturbation`, `temporal_masking`,
`signal_threshold` (and the shelf-life run adds `margin=20`) — the dial's
semantics are inherited unchanged (Lemma 4.1 applies to every transform that
uses the budget).

---

## 5. The power-matched control (a counterfactual in one equation)

### 5.1 Signal-to-noise ratio and the inversion

$$
\mathrm{SNR_{dB}} = 10\log_{10}\frac{P_x}{P_n}
\qquad\Longleftrightarrow\qquad
\frac{P_n}{P_x} = 10^{-\mathrm{SNR_{dB}}/10}
$$

**Verified:** both control arms are built at $\mathrm{SNR} = 0.305$ dB, so

$$
P_n / P_x = 10^{-0.0305/10} = \mathbf{0.932}
$$

**In plain words:** the injected noise carries **93 % of the speech's own
power** — the two arms are equally "loud" in noise, so neither can claim the
effect comes from *more* noise.

### 5.2 The energy-matching constraint

$$
\lVert n_{\text{shaped}} \rVert_2^2 = \lVert n_{\text{white}} \rVert_2^2 = P
\quad\text{(equal total noise energy)}
$$

with the *only* difference being the spectral/temporal **arrangement** of that
energy (hearing-shaped vs unconstrained).

### 5.3 Why this identifies the "shaping effect" (the causal argument)

Let $M(n)$ be any metric (STOI or WER) and $n^{(s)}, n^{(w)}$ the two
constructions at fixed power $P$. The naive contrast the base paper's
"None" column used conflates two factors:

$$
\underbrace{M(n^{(w)}) - M(n^{(s)})}_{\text{observed}} \;\;
\text{with} \;\; P_w \ne P_s
\quad\text{(energy and shaping move together — confounded)}
$$

At fixed $P$ the potential confound is removed **by construction**, so the
difference is attributable to arrangement alone — the same logic as a
randomized trial holding dose constant. This is novelty **N4**, now stated
mathematically:

$$
\Delta_{\text{shaping}} := M\big(n^{(w)}, P\big) - M\big(n^{(s)}, P\big)
$$

### 5.4 The measured contrast with full inference (NEW, verified 2026-10-06)

Pairs = the same 70 utterances, both arms, same engines
(`psychoacoustic.masked_noise#margin=0` vs `control.no_hearing_threshold`):

| metric | shaped (λ=0) | control (same power) | difference | test | effect |
|---|---:|---:|---:|---|---:|
| STOI | 0.8819 | 0.6751 | **+0.2068** (sd 0.0746) | Wilcoxon paired $p = 3.6 \times 10^{-13}$ | Cohen's $d_z = 2.77$ |
| WER (Whisper) | 0.4027 | 0.5439 | **−0.1412** | $p = 6.6 \times 10^{-4}$ | $d_z = -0.43$ |
| WER (Vosk) | 0.5177 | 0.6482 | **−0.1304** | $p = 7.9 \times 10^{-5}$ | $d_z = -0.54$ |

**Reading:** at identical noise power, psychoacoustic shaping *buys* +0.207
human intelligibility (an effect of almost three standard deviations) while
giving up only ~0.13–0.14 of machine damage. Both claims are now backed by
$n$, a test and an effect size — the reporting standard in
`STATISTICAL_ANALYSIS.md` §8.

---

## 6. Correlation and consistency statistics

### 6.1 Spearman's ρ (rank correlation)

For paired samples with ranks $R_i, R'_i$ (ties handled by mid-ranks, then
Pearson correlation on the ranks):

$$
\rho = \frac{\sum_i (R_i - \bar R)(R'_i - \bar R')}
{\sqrt{\sum_i (R_i - \bar R)^2 \sum_i (R'_i - \bar R')^2}}
\qquad
\Big(\text{no ties: } \rho = 1 - \frac{6\sum_i d_i^2}{n(n^2-1)}\Big)
$$

**Significance (t-approximation):**

$$
t = \rho \sqrt{\frac{n-2}{1-\rho^2}} \;\sim\; t_{n-2}
$$

**Verified values (recomputed from `summary.csv`, 147 conditions):**

| pair | ρ | scipy p | t-approx |
|---|---:|---:|---:|
| Whisper ΔWER vs Vosk ΔWER (all 147) | **0.7821** | $1.4 \times 10^{-31}$ | t = 15.11 |
| STOI vs cross-ΔWER (all 147) | **−0.6442** | $1.3 \times 10^{-18}$ | t = −10.14 |
| 2-engine vs 4-engine ranking (11 conditions **excluding** `original`, n = 10) | **0.9636** | $7.3 \times 10^{-6}$ | — |

All three match the published figures (0.782, −0.644, 0.964 / p = 7.3e-6) —
the last confirms our validation claim was computed on the **10 attack
conditions** (the clean original excluded).

**Interpretation discipline:** ρ = 0.782 ⇒ the two architectures order the
attacks almost identically ⇒ attacks **transfer**; ρ = −0.644 (human score vs
machine damage) ⇒ related but far from perfect — $\rho^2 \approx 0.41$ as a
*heuristic* "≈40 % of rank variation shared" (ρ² is not a variance
decomposition for ranks; state it as a heuristic only).

### 6.2 Which correlation when

- **Spearman** for *rankings* (attack orderings — invariant to monotone
  rescaling; our engines have very different WER scales: 0.079 vs 0.118).
- **Pearson** for *interval-valued paired measurements* (proxy vs real
  listener scores below).

### 6.3 Fisher-z interval for the proxy validation r (NEW, verified)

For Pearson $r$ with $n$ pairs: $z = \operatorname{atanh}(r)$,
$\mathrm{SE}_z = 1/\sqrt{n-3}$, CI $= \tanh\big(z \pm 1.96\,\mathrm{SE}_z\big)$.

With **r = 0.336, n = 1,770** (TMHINT-QI real listeners):

$$
z = 0.3496,\quad \mathrm{SE}_z = 0.02379,\quad
r \in [\,0.294,\ 0.377\,]\ \text{(95 \%)},\quad
z_{\text{stat}} = 14.69,\ p = 7.0 \times 10^{-49}
$$

**Reading:** the interval excludes 0 comfortably — the proxy tracks real
human intelligibility at $r \approx 0.34$ **[0.29, 0.38]**; state the
interval, not just the point value (it converts "imperfect" into a precise,
defensible claim).

---

## 7. Inference: paired tests, effect sizes, multiple comparisons

### 7.1 The pairing argument

All 147 conditions run on the *same* 70 utterances ⇒ observations are
paired. Default tests (inventory in `STATISTICAL_ANALYSIS.md` §4):
**Wilcoxon signed-rank** (distribution-free), **paired permutation test**
(assumption-free), Welch $t$ only for unpaired comparisons; effect sizes
**Cohen's $d$** (paired: $d_z = \bar d / s_d$), **Hedges' $g$** for small $n$.

### 7.2 Multiple comparisons: the mathematics of 147 tests

Testing $m = 147$ conditions at level α against the original:

$$
\mathrm{FWER} = 1 - (1-\alpha)^m \;\overset{\alpha=0.05}{=}\; 1 - 0.95^{147} = \mathbf{0.9995}
$$

**In plain words:** if we naively declared "significant" at p < 0.05 for
each of 147 tests, the chance of **at least one false alarm is 99.95 %**
(computed under independence — the honest caveat; positive dependence
between conditions makes the *expected* false count similar while FWER
varies).

Corrections provided (code: `evaluation/stats.py`):

- **Bonferroni** (strict): reject only if $p < \alpha/m = 0.05/147 =
  3.40\times10^{-4}$.
- **Benjamini–Hochberg** (FDR, recommended here): sort
  $p_{(1)} \le \dots \le p_{(m)}$, take largest $k$ with
  $p_{(k)} \le (k/m)\,q$, reject the $k$ smallest, controlling the expected
  false-discovery *proportion* at $q$.

### 7.3 Paired bootstrap (general instrument)

Resample the 70 utterance indices with replacement, recompute the statistic,
repeat $B = 1000$ times (seed 42), take percentiles. Pairing is preserved by
resampling *indices* (the same drawn indices for every arm/engine). Used for
CIs on HAG components, rank stability (REPORT: 300 resamples, top-3 stable)
and the scaling fits (§9.4). Percentile + BCa variants available.

### 7.4 NEW demonstration: BH applied to our own matrix

Wilcoxon (condition vs `original`) on `stoi_proxy`, BH at $q = 0.05$,
$m = 146$ conditions (original excluded):

$$
\textbf{127 of 146 rejected};\quad \min p = 1.9\times10^{-13}
$$

**The 19 non-rejections are the sanity proof** — they are exactly the
conditions that *should* be invisible:

```
baseline.gain, baseline.limiter, baseline.loudness_normalize,
baseline.peak_normalize, baseline.rms_normalize,            (benign controls)
channel.resampling_chain, multirate.drift, spectral.warping, (gentle edits)
psychoacoustic.signal_threshold (bare + #10-#50; #0, #5 rejected),
psychoacoustic.bark_perturbation #30/40/50,
psychoacoustic.masked_noise #50,
psychoacoustic.temporal_masking #50                          (strict-λ attacks)
```

FDR control *agrees with physics*: gentle transforms and λ = 30–50 dB
attacks produce no detectable intelligibility change; the method does not
manufacture significance. (Report with $m$, $q$, and this list — never the
count alone.)

---

## 8. Multi-objective structure: Pareto frontier and the design menu

### 8.1 Dominance

Each condition has the objective vector
$v(\theta) = \big(S(\theta),\ \text{cross-}\Delta(\theta)\big)$, both
maximized (more intelligibility preserved, more machine damage):

$$
\theta' \succ \theta \iff S(\theta') \ge S(\theta) \ \wedge\
\text{cross-}\Delta(\theta') \ge \text{cross-}\Delta(\theta)
\ \wedge\ \text{at least one strict}
$$

Frontier $\mathcal{F} = \{\theta : \nexists\, \theta' \succ \theta\}$.

**Code semantics** (`comparative.py:881`, verified): a point is kept iff no
*other* row has both coordinates ≥ it — the weak-≥ scan, $O(|\Theta|^2)$.

**Verified:** $\lvert\mathcal{F}\rvert = \mathbf{13}$ ✓ (recomputed from
`summary.csv`, matches REPORT §3.4).

**In plain words:** a condition is on the frontier when nobody is better at
*both* things — the "no free lunch" line. Everyone else is beaten on both
axes by some frontier point and can be discarded.

### 8.2 Threshold-constrained selection (exact counts, NEW)

The documented practical rule (REPORT §3.4 / COMPARATIVE §6) thresholds the
human and machine axes. Exact sets from `summary.csv`:

$$
\mathcal{S}(h_0, \delta_0) = \{\theta \ne \text{original} : \mathrm{HSR} \ge h_0 \wedge
\text{cross-}\Delta \ge \delta_0\}
$$

| rule applied | count | members |
|---|---:|---|
| $h_0 = 0.8,\ \delta_0 = 0.3$ (literal) | **9** | captcha_optimal, noise.clicks, phase_randomization, multi_domain, control, defense_robust, codec_simulation, masked_noise#0, packet_jitter |
| … excluding the control (not a policy) | 8 | as above minus control |
| … plus $\mathrm{STOI} \ge 0.75$ (human-realism cut) | 5 | captcha_optimal, multi_domain, codec_simulation, masked_noise#0, packet_jitter |
| frontier $\cap$ ($h_0{=}0.8, \delta_0{=}0.3$) | 3 | captcha_optimal, multi_domain, codec_simulation |
| **published "defensible set" (REPORT §3.4)** | **4** | captcha_optimal, codec_simulation, masked_noise#0, packet_jitter |

**Honest reconciliation note (fix before submission).** The four published
conditions satisfy $h_0 = 0.8, \delta_0 = 0.3$ — but so do five others;
`novel.multi_domain` (HSR 0.951, Δ 0.653, on the frontier) meets every
stated threshold yet is not in the published four. Before the paper freezes,
**one canonical rule must be chosen in one place** and the other documents
pointed at it. Recommended wording: *"practical menu = threshold-set at
HSR ≥ 0.8, cross-ΔWER ≥ 0.3, STOI ≥ 0.75, excluding controls, selecting the
best-HAG representative per regime"* — or simply publish the 5-member set.
This is a documentation-precision item, not a results change: no number in
any table moves.

### 8.3 Threshold ⇔ STOI: the algebra behind "human pass ≳ 0.8"

Since $\mathrm{HSR} = 0.6 + 0.4S$:

$$
\mathrm{HSR} \ge 0.8 \iff 0.4S \ge 0.2 \iff S \ge 0.5
$$

so the headline threshold is *exactly* "keep at least half the
intelligibility" — worth one sentence in the paper (it makes the
illustrative threshold's meaning concrete and auditable).

### 8.4 Why keep the whole frontier instead of one combined score

A weighted score $g_w(\theta) = w\cdot\text{cross-}\Delta + (1-w)\cdot S$
recovers only *supported* points of a non-convex frontier — some frontier
conditions are optimal for no constant weight. Ranking columns
(`human_rank`, `attack_rank`, `gap_rank`) plus the explicit frontier avoid
collapsing the trade-off into one number the designer never asked for.

---

## 9. Forecasting: scaling law, break capacity, shelf life

### 9.1 The model

Attacked-audio WER against capacity $C$ (million parameters) follows the
standard decaying power law:

$$
W(C) = a \cdot C^{-b}, \qquad b > 0
$$

Linearized: $\log W = \log a - b \log C$ — fit by OLS on the three ladder
points (Whisper tiny 39 M, base 74 M, small 244 M; one architecture so only
size varies).

### 9.2 OLS on logs

With $x_i = \log C_i$, $y_i = \log W_i$:

$$
\hat b = -\frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2},
\qquad
\log \hat a = \bar y + \hat b\,\bar x,
\qquad
R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}
$$

**Degrees of freedom:** $n_{\text{fit}} = 3 \Rightarrow \mathrm{df} = 1$ —
stated in every table we publish; CIs come from the bootstrap (§9.4), not
from $t_1$.

### 9.3 Break capacity C\* (closed form)

Set $W(C^*) = W_{\text{break}}$ with the **declared** security threshold
$W_{\text{break}} = 0.30$:

$$
a\, C^{*-b} = W_{\text{break}}
\;\Longrightarrow\;
\boxed{\,C^* = \Big(\frac{a}{W_{\text{break}}}\Big)^{1/b}\,}
\qquad
\log C^* = \frac{\log a - \log W_{\text{break}}}{b}
$$

(the exact expression in `shelf_life.py:483`; `None` if $b \le 0$ or
$C^* > 10^{12}$ M — never silently coerced).

**Sensitivity (why CIs matter):**

$$
\frac{\partial \log C^*}{\partial b} = -\frac{\log a - \log W_{\text{break}}}{b^2}
= -\frac{\log C^*}{b}
$$

Small $b$ ⇒ the exponent divides a large log ⇒ C\* swings widely — visible
in the published intervals (λ=40: C\* = 2 [0–8]).

### 9.4 Paired bootstrap across the ladder

$B = 1000$ resamples, **same index drawn for every rung** (preserves the
utterance × condition pairing across models), refit each time, percentile
95 % CIs for $a, b, C^*$ + prediction band; non-positive exponents are kept
in the $b$ distribution and counted in `n_break_valid` (honest handling of
the df = 1 fragility).

### 9.5 Time translation (scenario arithmetic)

$$
\text{months-to-break} = \delta \cdot \log_2 \frac{C^*}{C_{\text{ref}}},
\qquad \delta \in \{6, 12, 24\} \text{ months per doubling}
$$

(one doubling every δ months; $C_{\text{ref}}$ = largest evaluated model,
244 M). **δ is an assumption, not an observation** — labelled *scenario* in
all outputs.

### 9.6 The verified ladder and forecasts (`shelf_life.md`)

| model | C (M) | baseline WER | attacked WER | asr_sr | gap |
|---|---:|---:|---:|---:|---:|
| tiny | 39 | 0.079 | 0.254 | 0.703 | 0.260 |
| base | 74 | 0.071 | 0.164 | 0.806 | 0.157 |
| small | 244 | 0.042 | 0.091 | 0.929 | 0.034 |

| policy | $b$ [95 % CI] | $R^2$ | $C^*$ (M) [95 % CI] | months @6/12/24 |
|---|---|---:|---|---|
| control | 0.53 [0.43, 0.63] | 0.987 | 112 [81–159] | 0 / 0 / 0 |
| masked_noise λ=0 | 0.64 [0.50, 0.83] | 0.999 | 60 [45–81] | 0 / 0 / 0 |
| masked_noise λ=10 | 0.55 [0.38, 0.75] | 0.965 | 8 [2–17] | 0 / 0 / 0 |
| masked_noise λ=20 | 0.50 [0.32, 0.70] | 0.996 | 4 [1–11] | 0 / 0 / 0 |
| masked_noise λ=40 | 0.41 [0.22, 0.61] | 1.000 | 2 [0–8] | 0 / 0 / 0 |

**Conclusion (all $C^* \le 112 < 244$):** every evaluated policy's break
point lies *behind* today's open-model sizes ⇒ 0 months in every scenario —
at $W_{\text{break}} = 0.30$ the shelf life **has already expired**; and
stronger policies retain 25–50× more capacity than gentle ones (monotone in
λ — an internal-consistency check on the whole construction).

---

## 10. End-to-end worked example (`novel.captcha_optimal`)

Exact row values, then the derived chain — this is the "one row, fully
explained" artifact for the appendix or a viva request:

| quantity | formula | value |
|---|---|---:|
| STOI (proxy) | measured | 0.82909 |
| HSR | $0.6 + 0.4 \times 0.82909$ | **0.9316** |
| asr_sr | $1/140$ (one passing trial) | **0.0071** |
| HAG | $0.9316 - 0.0071$ | **0.9245** |
| ΔWER Whisper | $1.16969 - 0.07904$ | 1.09066 |
| ΔWER Vosk | $0.96573 - 0.11783$ | 0.84790 |
| cross-ΔWER | $(1.09066 + 0.84790)/2$ | **0.96928** |
| Wilson CI (asr_sr) | §3.3 | [0.0013, 0.0393] |
| ranks | human 118, attack 2, **gap 1**, quality 143 | — |
| frontier | nondominated (§8.1) | **True** |

**Reading:** rank 2 on damage, rank 1 on the human–machine gap, frontier
member — and the machine half of the gap is a single success in 140 trials.

---

## 11. Honesty constraints: what the mathematics does *not* prove

Carried verbatim from `STATISTICAL_ANALYSIS.md` §8 and the analysis docs:

1. **No p-value without $n$ and an effect size.** Never report "significant"
   alone.
2. **The illustrative HSR is never the subject of inference.** Tests run on
   STOI and WER (measured), never on the proxy pass-rate.
3. **Corpus:** LibriSpeech stand-in; no cross-corpus numeric comparison with
   the base paper (no formula above involves their published numbers).
4. **Engine set is fixed:** cross-ΔWER is a macro-average over our chosen
   engines — a design-based summary, not a population estimate (§2.3).
5. **Scaling fits:** 3 points, df = 1, extrapolation is scenario-based
   (§9.5); W_break = 0.30 and δ ∈ {6,12,24} are declared conventions.
6. **No security proof:** HAG is a separation *score*; it is not a proof
   that answers cannot be extracted (THREAT_MODEL §8 non-claims).
7. **Proxy validation** is external (TMHINT-QI, different corpus/language):
   $r = 0.336$ [0.29, 0.377] is *agreement*, never "equivalence".
8. **Doc reconciliation item:** §8.2's threshold-set count (9/5/3 vs the
   published 4) must be resolved to one canonical rule before submission.

---

## 12. Viva formula card

1. $\mathrm{WER} = (S+D+I)/L$; unbounded (Lemma 2.1).
2. $\widehat{\Delta}_e = \frac{1}{n}\sum_i \big(W_{e,i}(\theta) - W_{e,i}(\mathrm{id})\big)$; SE $= s_d/\sqrt{70}$.
3. cross-$\Delta = \frac{1}{|\mathcal{E}|}\sum_e \widehat{\Delta}_e$ (macro-average; 2.084 & 0.882 → 1.483).
4. HSR $= \mathrm{clip}(0.6 + 0.4S)$; slope 0.4; $\sigma_{\mathrm{HSR}} = 0.4\sigma_S$.
5. asr_sr $= \frac{1}{140}\sum \mathbf{1}[W \le 0.30]$; Wilson CI.
6. HAG $= \mathrm{HSR} - \mathrm{asr\_sr} \in [-1, 1]$; 0.9316 − 0.0071 = 0.9245.
7. $\partial \mathrm{HAG}/\partial S = 0.4$, $\partial \mathrm{HAG}/\partial \mathrm{asr\_sr} = -1$.
8. Admissible noise: $L_n(f) \le T_m(f) - \lambda$; nested sets ⇒ monotone sweep.
9. $\mathrm{SNR_{dB}} = 10\log_{10}(P_x/P_n)$; 0.305 dB ⇒ $P_n/P_x = 0.932$.
10. Counterfactual: $\Delta_{\text{shaping}} = M(n^{(w)}, P) - M(n^{(s)}, P)$ at fixed $P$; +0.207 STOI, $d_z = 2.77$, $p = 3.6\times10^{-13}$.
11. Spearman $t = \rho\sqrt{(n-2)/(1-\rho^2)}$; ρ = 0.782 / −0.644 / 0.964.
12. Fisher: $z = \operatorname{atanh}(r)$, CI $\tanh(z \pm 1.96/\sqrt{n-3})$ → 0.336 ⇒ [0.294, 0.377].
13. FWER $= 1-(1-\alpha)^{147} = 0.9995$; Bonferroni $3.4\times10^{-4}$; BH → 127/146.
14. Pareto: $\theta' \succ \theta$ ⇔ ≥ on both, one strict; $|\mathcal{F}| = 13$; HSR ≥ 0.8 ⇔ STOI ≥ 0.5.
15. Scaling: $W = aC^{-b}$; $C^* = (a/W_{\text{break}})^{1/b}$; months $= \delta\log_2(C^*/C_{\text{ref}})$.

---

## 13. Appendices

### A. File map

| math lives in | evidence file |
|---|---|
| §2 rows & baselines | `results/comparative/main/rows.csv`, `summary.csv` |
| §3 HAG components | `summary.csv` columns `hsr, asr_sr, hag` + `hag_metrics.py` |
| §4 budget/margins | `psychoacoustics/masking_models.py`, `transform` metadata |
| §5 control arms | `control.no_hearing_threshold`, `psychoacoustic.masked_noise#margin=0` rows |
| §6 correlations | recomputed from `summary.csv` (Appendix B.1) |
| §7 tests & corrections | `evaluation/stats.py`; BH demo B.3 |
| §8 frontier | `comparative.py:_is_nondominated`; 13 flags in `summary.csv` |
| §9 fits/bootstrap | `experiments/shelf_life.py`; `results/shelf_life/shelf_life.json` |
| §12 card | every figure/table in `results/` |

### B. Reproducibility snippets

**B.1 — correlations (§6.1):**
```python
import pandas as pd; from scipy import stats as st
df = pd.read_csv("results/comparative/main/summary.csv")
st.spearmanr(df.wer_whisper_tiny_delta, df.wer_vosk_small_en_delta)   # 0.7821, p=1.4e-31
st.spearmanr(df.stoi_mean, df.cross_delta_wer)                        # -0.6442, p=1.3e-18
```

**B.2 — Fisher-z interval (§6.3):**
```python
import numpy as np; from scipy import stats as st
z = np.arctanh(0.336); se = 1/np.sqrt(1770-3)
np.tanh([z-1.96*se, z+1.96*se])          # [0.294, 0.377]
2*st.norm.sf(z/se)                       # 6.97e-49
```

**B.3 — BH-FDR over the matrix (§7.4):**
```python
rows = pd.read_csv("results/comparative/main/rows.csv"); rows = rows[rows.status == "ok"]
orig = rows[rows.condition_id == "original"].set_index("utt_id")["stoi_proxy"]
tests = []
for cid, g in rows[rows.condition_id != "original"].groupby("condition_id"):
    s = g.set_index("utt_id")["stoi_proxy"]; ix = orig.index.intersection(s.index)
    p = 1.0 if np.allclose(orig[ix], s[ix]) else st.wilcoxon(orig[ix], s[ix]).pvalue
    tests.append((cid, p))
tests.sort(key=lambda t: t[1]); m = len(tests); q = 0.05
k = max([i for i, (c, p) in enumerate(tests, 1) if p <= q*i/m] or [0])   # 127 / 146
```

**B.4 — control contrast (§5.4):**
```python
A = rows[rows.condition_id == "psychoacoustic.masked_noise#margin=0"].set_index("utt_id")
B = rows[rows.condition_id == "control.no_hearing_threshold"].set_index("utt_id")
ix = A.index.intersection(B.index); d = A.loc[ix, "stoi_proxy"] - B.loc[ix, "stoi_proxy"]
d.mean(), d.std(ddof=1), d.mean()/d.std(ddof=1), st.wilcoxon(A.loc[ix,"stoi_proxy"], B.loc[ix,"stoi_proxy"])
# 0.2068, 0.0746, 2.77, p = 3.6e-13
```

### C. Verification log (2026-10-06)

| claim | verified value | source |
|---|---|---|
| rows | 10,290 / 770 / 1,260 (+header) | line counts |
| baseline WER | 0.0790374 / 0.1178283 | `summary.csv` arithmetic |
| HAG row | 0.9316 − 0.0071 = 0.9245 | `summary.csv` |
| cross-ΔWER row | (1.09066 + 0.84790)/2 = 0.96928 | `summary.csv` |
| Pn/Px @ 0.305 dB | 0.93218 | formula |
| ρ(Whisper, Vosk) | 0.782119, p = 1.38e-31 | scipy on 147 |
| ρ(STOI, cross) | −0.644225, p = 1.33e-18 | scipy on 147 |
| ρ(2eng, 4eng) | 0.963636, p = 7.32e-6 (n = 10) | scipy, `engine_validation/summary.csv` |
| Fisher CI | [0.2940, 0.3767], p = 6.97e-49 | formula |
| FWER / Bonferroni | 0.999469 / 3.401e-4 | formula |
| control STOI contrast | +0.206752 (sd 0.0746), dz 2.77, p 3.56e-13 | Wilcoxon, 70 pairs |
| control WER contrasts | −0.1412 (p 6.6e-4), −0.1304 (p 7.9e-5) | Wilcoxon, 70 pairs |
| BH-FDR | 127/146 at q = 0.05; min p 1.88e-13 | §B.3 |
| Wilson (1/140) | [0.00126, 0.03935] | formula |
| frontier size | 13 | `_is_nondominated` scan |
| λ sweep | 0.3618 → 0.0018 non-increasing ✓ | `summary.csv` |
| threshold counts | 9 / 8 / 5 / 3 vs published 4 | §8.2 |
