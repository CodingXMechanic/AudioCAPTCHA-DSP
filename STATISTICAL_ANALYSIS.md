# Statistical Analysis

Statistical machinery available in `evaluation/stats.py` and how it is used
in benchmark reporting. WHAT-REMAINS §9 — *Implemented*.

---

## 1. Descriptive statistics

`summarize_condition(...) -> ConditionSummary`:

- n, mean, median, std (ddof options), min/max, quartiles, IQR, skew, kurtosis
  for any per-condition metric vector (STOI, WER, SNR, MBSD…).
- Used internally by `compare_conditions` and available standalone.

## 2. Confidence intervals

| Function | Method | Use |
|---|---|---|
| `bootstrap_ci` | percentile bootstrap | general CIs on means/medians |
| `bootstrap_ci_bca` | bias-corrected & accelerated (with scipy fallback) | skewed distributions (WER is bounded + zero-inflated) |
| figure CIs | t-based 95 % CI (1.96·SE) | `fig4_family_summary` family bars |

Baseline WER and all per-condition means are reported alongside n
(`n_ok`, `n_err` per condition) so any CI can be recomputed.

## 3. Effect sizes

- `cohens_d(group_a, group_b)` — standardized mean difference.
- `hedges_g(...)` — small-sample corrected version (used when n < 20,
  e.g. subset-A conditions).

## 4. Hypothesis tests

| Test | Function | Applies to |
|---|---|---|
| Welch *t*-test | `welch_ttest` | condition vs condition, unequal variance |
| Mann–Whitney *U* | `mann_whitney_u` | non-normal metric vectors |
| Wilcoxon signed-rank | `wilcoxon_signed_rank` | paired (same utterances across conditions — the default pairing in our design) |
| Paired permutation test | `paired_permutation_test` | assumption-free paired comparison |

**Design note:** every condition runs on the *identical* utterance set, so
paired tests are the natural choice; condition-vs-condition comparisons in
`compare_conditions` default accordingly.

## 5. Multiple comparisons

Ranking 100+ transforms means hundreds of pairwise tests.
`bonferroni_correction` (strict) and `benjamini_hochberg_correction`
(FDR — recommended for this many comparisons) are both provided;
`compare_conditions` accepts a correction parameter.

## 6. Multi-objective structure

- `compute_pareto_frontier(objectives, maximize=...)` — generic frontier
  extraction; the benchmark's `pareto` column applies it to
  (STOI, cross-ΔWER).
- `SecurityEvaluator`'s `is_pareto_efficient` provides an independent flag
  on its own metric set.

## 7. What the published outputs actually contain

| Output | Statistical content |
|---|---|
| `summary.csv` | means/stds, n_ok/n_err, ranks |
| `summary.json` | same + baselines + labels |
| `ranking.md` | means; per-utterance rows remain in `rows.csv` for re-analysis |
| `fig4_family_summary` | mean ± 95 % CI per family |
| `tables/*.csv` | aggregates |

**Not yet automatic:** per-condition significance tests against the
`original` baseline are *Available, not auto-emitted* — run
`compare_conditions` on the `rows.csv` groups (example below).

```python
import pandas as pd
from audiocaptcha_dsp.evaluation.stats import compare_conditions

rows = pd.read_csv("results/comparative/main/rows.csv")
orig = rows[(rows.condition_id == "original") & (rows.status == "ok")]
for cid, g in rows[rows.status == "ok"].groupby("condition_id"):
    if cid == "original":
        continue
    res = compare_conditions(
        metrics=["stoi_proxy"],
        group_a=orig["stoi_proxy"].to_numpy(),
        group_b=g["stoi_proxy"].to_numpy(),
        correction="benjamini_hochberg",
    )
    ...
```

## 8. Honesty constraints on statistics

- No p-value is published without its n and effect size (per WHAT-REMAINS
  §9 "avoid p-value-only reporting" pattern).
- CIs from n = 1 smoke runs are meaningless — smoke runs are labelled
  non-citable in `PROJECT_STATUS.md`.
- The illustrative HSR is never the subject of inferential statistics.

## 9. Scaling-law fit and paired bootstrap (shelf-life, novelty N11)

`experiments/shelf_life.py` adds a second inference layer on top of the
benchmark — forecasting, not testing:

- **Model.** Attacked-ASR WER against model capacity C (M parameters) is fit
  as a power law `WER = a · C^(−b)` by log–log OLS (3 ladder points:
  Whisper tiny/base/small — df = 1; `fit_power_law`).
- **Uncertainty.** A **paired bootstrap** (`paired_bootstrap_scaling`)
  resamples evaluation pairs *with the same index drawn across models*
  (preserving the pairing of utterance × condition across rungs), refits on
  each of B = 1000 resamples, and reports percentile 95 % CIs for a, b, the
  break capacity C\*, and a prediction band over a log-spaced capacity grid.
  Resamples that produce a non-positive exponent are kept in the b
  distribution and yield no break capacity (`n_break_valid` is published).
- **Break capacity.** C\* = (a / W_break)^(1/b), reported with its bootstrap
  CI; `None` when the exponent is non-positive or C\* exceeds 1e12 M params
  (`MAX_BREAK_PARAMS_M`) — never silently coerced to a number.
- **Scenario translation.** Months-to-break = `doubling_months ·
  log2(C*/C_ref)` for 6/12/24-month doubling scenarios. These are
  *assumptions stated as scenarios*; the fit does not observe any trend over
  time, and §8's honesty constraints apply verbatim to every published
  forecast.
