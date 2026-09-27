"""
Human-ASR Gap (HAG) Metrics
==============================
Implements the complete security and robustness metric suite for evaluating
audio CAPTCHA transformations in the Human-vs-ASR research paradigm.

Research Metrics Defined
-------------------------
- Human Success Rate (HSR): fraction of humans who correctly answer the CAPTCHA
- ASR Success Rate (ASR-SR): fraction of ASR attempts that correctly extract the answer
- Human-ASR Gap (HAG): HSR - ASR-SR (higher = better CAPTCHA security)
- Transformation Robustness Score (TRS): defense survival across preprocessing
- Defense Survival Rate (DSR): fraction of WER degradation that survives defense
- Cross-Model Transfer Rate (CMTR): fraction of models that fail on the same utterance
- Preprocessing Recovery Rate (PRR): fraction of WER recovered by defense pipeline
- Perceptual Acceptability Rate (PAR): fraction of samples with acceptable quality
- Forensic Detectability Rate (FDR): fraction detectable as manipulated (future work)
- Computational Cost Score (CCS): normalized runtime cost

References
----------
- Schönherr et al. (2018). arXiv:1808.05665 — psychoacoustic adversarial attacks.
- Cullen et al. (2025). arXiv:2606.27698 — certified ASR robustness.
- Yang et al. (1997). IEEE Workshop — MBSD perceptual quality.
- Paper 15 (arXiv:2307.04517) — objective/subjective correlation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Security Metrics
# ---------------------------------------------------------------------------

def compute_hsr(correct_responses: list[bool]) -> float:
    """Compute Human Success Rate (HSR).

    Parameters
    ----------
    correct_responses : list[bool]
        Per-trial correctness outcomes from human participants.

    Returns
    -------
    float
        HSR in [0, 1]. 1.0 = all humans succeeded.
    """
    if not correct_responses:
        return 0.0
    return float(np.mean(correct_responses))


def compute_asr_sr(
    wer_values: list[float],
    wer_threshold: float = 0.3,
) -> float:
    """Compute ASR Success Rate (ASR-SR).

    A transcription is considered 'successful' (i.e., the ASR correctly
    extracted the answer) if its Word Error Rate is below `wer_threshold`.

    Parameters
    ----------
    wer_values : list[float]
        Per-utterance WER values from an ASR model.
    wer_threshold : float, optional
        WER below which ASR is considered to have 'succeeded'. Default 0.3.

    Returns
    -------
    float
        ASR-SR in [0, 1]. 0.0 = ASR always fails (best for CAPTCHA security).
    """
    if not wer_values:
        return 0.0
    successes = [w < wer_threshold for w in wer_values]
    return float(np.mean(successes))


def compute_hag(
    hsr: float,
    asr_sr: float,
) -> float:
    """Compute Human-ASR Gap (HAG).

    HAG = HSR - ASR-SR

    Higher positive HAG values indicate superior CAPTCHA efficacy:
    - Humans easily understand and answer the challenge (high HSR)
    - Automated ASR systems fail to extract the answer (low ASR-SR)

    Parameters
    ----------
    hsr : float
        Human Success Rate in [0, 1].
    asr_sr : float
        ASR Success Rate in [0, 1].

    Returns
    -------
    float
        HAG in [-1, 1]. Target: HAG > 0.5 for a practical CAPTCHA.

    Notes
    -----
    HAG > 0.7 with STOI proxy > 0.7 constitutes an excellent CAPTCHA condition
    according to the security-quality trade-off framework (Section 6 analysis).
    """
    hsr = float(np.clip(hsr, 0.0, 1.0))
    asr_sr = float(np.clip(asr_sr, 0.0, 1.0))
    return hsr - asr_sr


def compute_defense_survival_rate(
    wer_original: float,
    wer_transformed: float,
    wer_post_defense: float,
    epsilon: float = 1e-6,
) -> float:
    """Compute Defense Survival Rate (DSR).

    DSR measures how much of the WER degradation caused by a transformation
    survives after a defense preprocessing pipeline is applied.

    DSR = (WER_post_defense - WER_original) / (WER_transformed - WER_original + ε)

    DSR = 1.0: defense recovers nothing, full ASR degradation survives.
    DSR = 0.0: defense fully recovers ASR performance.
    DSR > 1.0: defense makes things worse.

    Parameters
    ----------
    wer_original : float
        WER on original (untransformed) audio.
    wer_transformed : float
        WER on transformed audio (before defense).
    wer_post_defense : float
        WER on transformed audio after defense preprocessing.
    epsilon : float
        Numerical stability constant.

    Returns
    -------
    float
        DSR value. Values in [0, inf), but typically in [0, 1.5].
    """
    degradation = wer_transformed - wer_original
    survival = wer_post_defense - wer_original

    if abs(degradation) < epsilon:
        return 1.0 if abs(survival) < epsilon else 0.0

    return float(np.clip(survival / degradation, -0.5, 2.0))


def compute_preprocessing_recovery_rate(
    wer_transformed: float,
    wer_post_defense: float,
    wer_original: float,
    epsilon: float = 1e-6,
) -> float:
    """Compute Preprocessing Recovery Rate (PRR).

    PRR = 1 - DSR = fraction of WER degradation recovered by defense.

    PRR = 1.0: defense fully restores ASR to original performance.
    PRR = 0.0: defense recovers nothing.

    Parameters
    ----------
    wer_transformed, wer_post_defense, wer_original : float
        As in compute_defense_survival_rate.

    Returns
    -------
    float
        PRR in [0, 1] (clipped).
    """
    dsr = compute_defense_survival_rate(
        wer_original, wer_transformed, wer_post_defense, epsilon
    )
    return float(np.clip(1.0 - dsr, 0.0, 1.0))


def compute_cross_model_failure_rate(
    per_model_wer: dict[str, list[float]],
    wer_threshold: float = 0.5,
) -> float:
    """Compute Cross-Model Transfer Rate / Universal Failure Rate.

    Fraction of utterances where ALL provided ASR models fail (WER > threshold).
    A high rate indicates the transformation generalizes across model architectures.

    Parameters
    ----------
    per_model_wer : dict[str, list[float]]
        Mapping from model_name to list of per-utterance WER values.
        All lists must have the same length.
    wer_threshold : float, optional
        WER above which ASR is considered to have 'failed'. Default 0.5.

    Returns
    -------
    float
        Cross-model failure rate in [0, 1].

    Raises
    ------
    ValueError
        If per_model_wer is empty or lists have mismatched lengths.
    """
    if not per_model_wer:
        return 0.0

    model_names = list(per_model_wer.keys())
    n_utt = len(per_model_wer[model_names[0]])
    if n_utt == 0:
        return 0.0

    # Verify alignment
    for name, wers in per_model_wer.items():
        if len(wers) != n_utt:
            raise ValueError(
                f"WER list for model '{name}' has length {len(wers)}, "
                f"expected {n_utt}."
            )

    # Count utterances where ALL models fail
    universal_failures = 0
    for i in range(n_utt):
        all_fail = all(per_model_wer[m][i] > wer_threshold for m in model_names)
        if all_fail:
            universal_failures += 1

    return float(universal_failures / n_utt)


def compute_perceptual_acceptability_rate(
    stoi_proxy_values: list[float],
    mbsd_values: list[float],
    snr_values: list[float],
    stoi_threshold: float = 0.7,
    mbsd_threshold: float = 5.0,
    snr_threshold: float = 15.0,
) -> float:
    """Compute Perceptual Acceptability Rate (PAR).

    Fraction of transformed samples that meet minimum perceptual quality
    thresholds across all three criteria simultaneously:
    - STOI proxy >= stoi_threshold (intelligibility preserved)
    - MBSD <= mbsd_threshold (spectral quality acceptable)
    - SNR >= snr_threshold (signal fidelity acceptable)

    Parameters
    ----------
    stoi_proxy_values : list[float]
        STOI proxy intelligibility scores (0-1).
    mbsd_values : list[float]
        Modified Bark Spectral Distortion values (dB, lower is better).
    snr_values : list[float]
        SNR values (dB, higher is better). Infinite values are treated as
        passing (identity transform condition).
    stoi_threshold : float
        Minimum STOI proxy for acceptability.
    mbsd_threshold : float
        Maximum MBSD for acceptability (dB).
    snr_threshold : float
        Minimum SNR for acceptability (dB).

    Returns
    -------
    float
        PAR in [0, 1].
    """
    n = len(stoi_proxy_values)
    if n == 0:
        return 0.0
    if len(mbsd_values) != n or len(snr_values) != n:
        logger.warning("Metric array length mismatch in compute_par. Using min length.")
        n = min(n, len(mbsd_values), len(snr_values))

    acceptable = 0
    for i in range(n):
        stoi_ok = stoi_proxy_values[i] >= stoi_threshold
        mbsd_ok = mbsd_values[i] <= mbsd_threshold
        snr_ok = np.isposinf(snr_values[i]) or (snr_values[i] >= snr_threshold)
        if stoi_ok and mbsd_ok and snr_ok:
            acceptable += 1

    return float(acceptable / n)


def compute_transformation_robustness_score(
    dsr: float,
    cmfr: float,
    weight_dsr: float = 0.6,
    weight_cmfr: float = 0.4,
) -> float:
    """Compute Transformation Robustness Score (TRS).

    Weighted combination of Defense Survival Rate and Cross-Model Failure Rate.

    TRS = weight_dsr * DSR + weight_cmfr * CMFR

    TRS = 1.0 is the ideal: transformation both survives defenses and
    transfers across all ASR models.

    Parameters
    ----------
    dsr : float
        Defense Survival Rate (0=defeated by defense, 1=survives defense).
    cmfr : float
        Cross-Model Failure Rate (fraction of models that fail).
    weight_dsr, weight_cmfr : float
        Weighting factors (must sum to 1.0 approximately).

    Returns
    -------
    float
        TRS in [0, 1].
    """
    dsr = float(np.clip(dsr, 0.0, 1.0))
    cmfr = float(np.clip(cmfr, 0.0, 1.0))
    return float(weight_dsr * dsr + weight_cmfr * cmfr)


def compute_captcha_security_score(
    hag: float,
    trs: float,
    par: float,
    weight_hag: float = 0.5,
    weight_trs: float = 0.3,
    weight_par: float = 0.2,
) -> float:
    """Compute overall CAPTCHA Security Score (CSS).

    Weighted combination of HAG, TRS, and PAR for overall ranking.
    This is a user-defined preference score — NOT an objective ground truth.
    Always examine individual components separately for full analysis.

    Parameters
    ----------
    hag : float
        Human-ASR Gap (higher = more secure, in [-1, 1]).
    trs : float
        Transformation Robustness Score (in [0, 1]).
    par : float
        Perceptual Acceptability Rate (in [0, 1]).
    weight_hag, weight_trs, weight_par : float
        User-defined weights.

    Returns
    -------
    float
        CSS in approximately [0, 1].

    Notes
    -----
    This is a user-configurable ranking metric. The default weights prioritize
    security (HAG 50%) > robustness (TRS 30%) > quality (PAR 20%).
    Researchers should report all components separately alongside this score.
    """
    # Normalize HAG from [-1, 1] to [0, 1]
    hag_norm = float(np.clip((hag + 1.0) / 2.0, 0.0, 1.0))
    trs = float(np.clip(trs, 0.0, 1.0))
    par = float(np.clip(par, 0.0, 1.0))

    return float(
        weight_hag * hag_norm
        + weight_trs * trs
        + weight_par * par
    )


# ---------------------------------------------------------------------------
# Comprehensive Security Evaluation
# ---------------------------------------------------------------------------

@dataclass
class SecurityEvaluation:
    """Complete security evaluation result for a single transform condition.

    Attributes
    ----------
    transform_name : str
        Name of the transformation evaluated.
    condition_params : dict
        Transform parameter configuration.
    n_samples : int
        Number of audio samples evaluated.
    n_human_participants : int
        Number of human participants (0 if no human study).
    hsr : float
        Human Success Rate (from human study or simulation).
    asr_sr : float
        ASR Success Rate (from ASR evaluation).
    hag : float
        Human-ASR Gap = HSR - ASR-SR.
    dsr : float
        Defense Survival Rate (mean across defense conditions).
    prr : float
        Preprocessing Recovery Rate = 1 - DSR.
    cmfr : float
        Cross-Model Failure Rate (if multiple models evaluated).
    par : float
        Perceptual Acceptability Rate.
    trs : float
        Transformation Robustness Score.
    css : float
        CAPTCHA Security Score (user-weighted composite).
    mean_wer : float
        Mean WER across models and conditions.
    mean_stoi_proxy : float
        Mean STOI proxy across samples.
    mean_mbsd : float
        Mean MBSD across samples.
    mean_snr_db : float
        Mean SNR in dB.
    is_pareto_efficient : bool
        Whether this condition is Pareto-efficient.
    notes : list[str]
        Qualitative notes and warnings.
    """

    transform_name: str
    condition_params: dict = field(default_factory=dict)
    n_samples: int = 0
    n_human_participants: int = 0

    # Core security metrics
    hsr: float = 0.0
    asr_sr: float = 0.0
    hag: float = 0.0

    # Robustness
    dsr: float = 0.0
    prr: float = 0.0
    cmfr: float = 0.0
    trs: float = 0.0

    # Quality
    par: float = 0.0
    mean_stoi_proxy: float = 0.0
    mean_mbsd: float = 0.0
    mean_snr_db: float = 0.0

    # ASR performance
    mean_wer: float = 0.0
    mean_cer: float = 0.0

    # Composite
    css: float = 0.0

    # Analysis
    is_pareto_efficient: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_name": self.transform_name,
            "condition_params": self.condition_params,
            "n_samples": self.n_samples,
            "n_human_participants": self.n_human_participants,
            "hsr": round(self.hsr, 4),
            "asr_sr": round(self.asr_sr, 4),
            "hag": round(self.hag, 4),
            "dsr": round(self.dsr, 4),
            "prr": round(self.prr, 4),
            "cmfr": round(self.cmfr, 4),
            "trs": round(self.trs, 4),
            "par": round(self.par, 4),
            "mean_stoi_proxy": round(self.mean_stoi_proxy, 4),
            "mean_mbsd": round(self.mean_mbsd, 4),
            "mean_snr_db": round(self.mean_snr_db, 4),
            "mean_wer": round(self.mean_wer, 4),
            "mean_cer": round(self.mean_cer, 4),
            "css": round(self.css, 4),
            "is_pareto_efficient": self.is_pareto_efficient,
            "notes": self.notes,
        }

    @property
    def security_label(self) -> str:
        """Human-readable security label based on HAG."""
        if self.hag >= 0.7:
            return "EXCELLENT"
        elif self.hag >= 0.5:
            return "GOOD"
        elif self.hag >= 0.3:
            return "MODERATE"
        elif self.hag >= 0.1:
            return "WEAK"
        else:
            return "POOR"


class SecurityEvaluator:
    """Compute comprehensive security evaluations for all transforms.

    This integrates DSP metrics, ASR results, and (optionally) human study
    data into a unified security evaluation framework.

    Parameters
    ----------
    wer_success_threshold : float
        WER below which ASR is considered to have 'succeeded'. Default 0.3.
    stoi_acceptability_threshold : float
        Minimum STOI proxy for perceptual acceptability. Default 0.7.
    mbsd_acceptability_threshold : float
        Maximum MBSD for perceptual acceptability (dB). Default 5.0.
    snr_acceptability_threshold : float
        Minimum SNR for perceptual acceptability (dB). Default 15.0.
    """

    def __init__(
        self,
        wer_success_threshold: float = 0.3,
        stoi_acceptability_threshold: float = 0.7,
        mbsd_acceptability_threshold: float = 5.0,
        snr_acceptability_threshold: float = 15.0,
    ) -> None:
        self.wer_success_threshold = wer_success_threshold
        self.stoi_acceptability_threshold = stoi_acceptability_threshold
        self.mbsd_acceptability_threshold = mbsd_acceptability_threshold
        self.snr_acceptability_threshold = snr_acceptability_threshold

    def evaluate(
        self,
        transform_name: str,
        condition_params: dict,
        dsp_metrics: list[dict[str, float]],
        asr_results: dict[str, list[float]] | None = None,
        defense_asr_results: dict[str, dict[str, list[float]]] | None = None,
        human_responses: list[bool] | None = None,
        original_wer: float = 0.0,
    ) -> SecurityEvaluation:
        """Evaluate a single transform condition.

        Parameters
        ----------
        transform_name : str
            Transform identifier.
        condition_params : dict
            Parameter configuration for this condition.
        dsp_metrics : list[dict[str, float]]
            Per-sample DSP metric dicts (from compute_metrics).
        asr_results : dict[str, list[float]], optional
            Per-model WER lists, e.g. {'whisper_tiny': [0.1, 0.3, ...]}.
        defense_asr_results : dict[str, dict[str, list[float]]], optional
            Per-defense, per-model WER lists.
            e.g. {'defense.spectral_denoise': {'whisper_tiny': [...]}}
        human_responses : list[bool], optional
            Per-trial human correctness. None uses a simulated HSR of 0.85.
        original_wer : float
            WER on the original (untransformed) audio for defense comparison.

        Returns
        -------
        SecurityEvaluation
        """
        n = len(dsp_metrics)
        notes: list[str] = []

        # --- DSP metric aggregation ---
        stoi_vals = [m.get("stoi_proxy", 1.0) for m in dsp_metrics if np.isfinite(m.get("stoi_proxy", 1.0))]
        mbsd_vals = [m.get("mbsd", 0.0) for m in dsp_metrics if np.isfinite(m.get("mbsd", 0.0))]
        snr_vals = [m.get("snr_db", np.inf) for m in dsp_metrics]

        mean_stoi = float(np.mean(stoi_vals)) if stoi_vals else 1.0
        mean_mbsd = float(np.mean(mbsd_vals)) if mbsd_vals else 0.0
        # For SNR, replace inf with very large number for mean, but flag it
        finite_snr = [v for v in snr_vals if np.isfinite(v)]
        if len(finite_snr) < len(snr_vals):
            notes.append("Some SNR values are infinite (identity/near-identity transform).")
        mean_snr = float(np.mean(finite_snr)) if finite_snr else float("inf")

        # --- ASR evaluation ---
        mean_wer = 0.0
        mean_cer = 0.0
        asr_sr = 0.0

        if asr_results:
            all_wers = []
            for model_wers in asr_results.values():
                finite_w = [w for w in model_wers if np.isfinite(w)]
                all_wers.extend(finite_w)
            if all_wers:
                mean_wer = float(np.mean(all_wers))
                asr_sr = compute_asr_sr(all_wers, self.wer_success_threshold)
        else:
            notes.append("No ASR results provided. ASR-SR set to 0.0.")

        # --- Human Success Rate ---
        if human_responses is not None:
            hsr = compute_hsr(human_responses)
            n_human = len(human_responses)
        else:
            # Simulate based on perceptual quality if no human study
            # Conservative model: high STOI → high HSR
            hsr = float(np.clip(0.6 + 0.4 * mean_stoi, 0.0, 1.0))
            n_human = 0
            notes.append(
                "No human study data. HSR estimated from STOI proxy "
                f"({mean_stoi:.3f}). Treat as illustrative only."
            )

        hag = compute_hag(hsr, asr_sr)

        # --- Defense survival ---
        dsr_values: list[float] = []
        if defense_asr_results and asr_results:
            base_wer = mean_wer
            for defense_name, def_model_wers in defense_asr_results.items():
                for model_name, def_wers in def_model_wers.items():
                    if model_name in asr_results:
                        finite_dw = [w for w in def_wers if np.isfinite(w)]
                        if finite_dw:
                            def_wer = float(np.mean(finite_dw))
                            dsr_val = compute_defense_survival_rate(
                                original_wer, base_wer, def_wer
                            )
                            dsr_values.append(dsr_val)

        dsr = float(np.mean(dsr_values)) if dsr_values else 1.0
        prr = float(np.clip(1.0 - dsr, 0.0, 1.0))

        # --- Cross-model failure rate ---
        if asr_results and len(asr_results) > 1:
            cmfr = compute_cross_model_failure_rate(
                asr_results, self.wer_success_threshold
            )
        else:
            cmfr = 0.0
            if asr_results:
                notes.append("Only one ASR model. Cross-model failure rate not computable.")

        # --- Perceptual acceptability ---
        par = compute_perceptual_acceptability_rate(
            stoi_vals if stoi_vals else [mean_stoi] * n,
            mbsd_vals if mbsd_vals else [mean_mbsd] * n,
            snr_vals,
            stoi_threshold=self.stoi_acceptability_threshold,
            mbsd_threshold=self.mbsd_acceptability_threshold,
            snr_threshold=self.snr_acceptability_threshold,
        )

        # --- Composite scores ---
        trs = compute_transformation_robustness_score(dsr, cmfr)
        css = compute_captcha_security_score(hag, trs, par)

        return SecurityEvaluation(
            transform_name=transform_name,
            condition_params=condition_params,
            n_samples=n,
            n_human_participants=n_human,
            hsr=hsr,
            asr_sr=asr_sr,
            hag=hag,
            dsr=dsr,
            prr=prr,
            cmfr=cmfr,
            trs=trs,
            par=par,
            mean_stoi_proxy=mean_stoi,
            mean_mbsd=mean_mbsd,
            mean_snr_db=mean_snr,
            mean_wer=mean_wer,
            mean_cer=mean_cer,
            css=css,
            notes=notes,
        )

    def evaluate_batch(
        self,
        evaluations: list[dict[str, Any]],
    ) -> list[SecurityEvaluation]:
        """Evaluate a batch of transform conditions from a list of evaluation specs.

        Each spec dict must contain: transform_name, condition_params, dsp_metrics.
        Optional: asr_results, defense_asr_results, human_responses, original_wer.

        Returns sorted list by CSS descending.
        """
        results = []
        for spec in evaluations:
            try:
                ev = self.evaluate(
                    transform_name=spec.get("transform_name", "unknown"),
                    condition_params=spec.get("condition_params", {}),
                    dsp_metrics=spec.get("dsp_metrics", []),
                    asr_results=spec.get("asr_results"),
                    defense_asr_results=spec.get("defense_asr_results"),
                    human_responses=spec.get("human_responses"),
                    original_wer=spec.get("original_wer", 0.0),
                )
                results.append(ev)
            except Exception as e:
                logger.warning("Security evaluation failed for %s: %s",
                               spec.get("transform_name", "?"), e)

        return sorted(results, key=lambda x: x.css, reverse=True)

    def mark_pareto_efficient(
        self,
        evaluations: list[SecurityEvaluation],
    ) -> list[SecurityEvaluation]:
        """Mark Pareto-efficient transforms in a list of evaluations.

        A transform is Pareto-efficient if no other transform simultaneously
        dominates it across all objectives:
        - Higher HAG (better security)
        - Higher PAR (better perceptual quality)
        - Higher DSR (better defense robustness)
        - Higher CMFR (better cross-model transfer)

        Parameters
        ----------
        evaluations : list[SecurityEvaluation]
            List of evaluated transforms.

        Returns
        -------
        list[SecurityEvaluation]
            Same list with is_pareto_efficient field updated.
        """
        n = len(evaluations)
        if n == 0:
            return evaluations

        # Build objective matrix (higher = better for all)
        objectives = np.array([
            [e.hag, e.par, e.dsr, e.cmfr] for e in evaluations
        ], dtype=np.float64)

        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j >= i in all and j > i in at least one
                if (np.all(objectives[j] >= objectives[i]) and
                        np.any(objectives[j] > objectives[i])):
                    dominated[i] = True
                    break

        for i, ev in enumerate(evaluations):
            ev.is_pareto_efficient = bool(not dominated[i])

        return evaluations


# ---------------------------------------------------------------------------
# Ranking support
# ---------------------------------------------------------------------------

def rank_transforms_multiobjective(
    evaluations: list[SecurityEvaluation],
    ranking_mode: str = "hag",
) -> list[SecurityEvaluation]:
    """Rank transforms by a specified multi-objective mode.

    Parameters
    ----------
    evaluations : list[SecurityEvaluation]
        Evaluated transforms.
    ranking_mode : str
        One of:
        - 'hag': Sort by Human-ASR Gap (best CAPTCHA security)
        - 'par': Sort by Perceptual Acceptability Rate (best quality)
        - 'dsr': Sort by Defense Survival Rate (most robust)
        - 'trs': Sort by Transformation Robustness Score
        - 'css': Sort by CAPTCHA Security Score (composite)
        - 'wer': Sort by mean WER (highest ASR degradation)
        - 'stoi': Sort by STOI proxy (highest perceptual quality)
        - 'pareto': Return only Pareto-efficient transforms, then rest

    Returns
    -------
    list[SecurityEvaluation]
        Sorted list (descending, best first).
    """
    mode_map = {
        "hag": lambda e: e.hag,
        "par": lambda e: e.par,
        "dsr": lambda e: e.dsr,
        "trs": lambda e: e.trs,
        "css": lambda e: e.css,
        "wer": lambda e: e.mean_wer,
        "stoi": lambda e: e.mean_stoi_proxy,
        "cmfr": lambda e: e.cmfr,
    }

    if ranking_mode == "pareto":
        return sorted(
            evaluations,
            key=lambda e: (int(not e.is_pareto_efficient), -e.css),
        )

    key_fn = mode_map.get(ranking_mode, mode_map["css"])
    return sorted(evaluations, key=key_fn, reverse=True)
