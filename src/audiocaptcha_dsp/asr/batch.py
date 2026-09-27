"""
Batch ASR Transcription Pipeline
=================================
Multi-model batch transcription with defense pipeline integration,
WER/CER computation, and defense recovery analysis.

Supports:
- Single-model and multi-model batch evaluation
- Defense preprocessing integration
- Paired comparison (original vs transformed)
- Per-utterance error analysis
- Defense survival rate computation

References
----------
- Paper 8 (arXiv:2606.27698): WER detection and confidence calibration.
- Paper 9 (arXiv:2503.11627): Speech denoising robustness under adversarial noise.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.core.types import TranscriptionResult
from audiocaptcha_dsp.asr.engine import ASREngine, IndependentASREngine, normalize_transcript
from audiocaptcha_dsp.evaluation.metrics import compute_wer, compute_cer, compute_detailed_wer
from audiocaptcha_dsp.asr.defense import ASRDefense, IdentityDefense, DefensePipeline

logger = logging.getLogger(__name__)


@dataclass
class SingleTranscriptionRecord:
    """Record of a single ASR transcription with quality metrics.

    Attributes
    ----------
    utterance_id : str
        Unique identifier for the utterance.
    model_name : str
        Name of the ASR model used.
    defense_name : str
        Name of the defense preprocessing applied (or 'defense.raw').
    reference : str
        Normalized ground-truth transcript.
    hypothesis : str
        Normalized ASR hypothesis.
    wer : float
        Word Error Rate in [0, ∞).
    cer : float
        Character Error Rate in [0, ∞).
    substitutions : float
        Substitution rate (substitutions / total reference words).
    deletions : float
        Deletion rate.
    insertions : float
        Insertion rate.
    confidence : float
        ASR model confidence score if available, else 0.5.
    latency_s : float
        Wall-clock transcription latency in seconds.
    condition_index : int
        Experiment condition index.
    transform_name : str
        Name of the DSP transform applied.
    """

    utterance_id: str
    model_name: str
    defense_name: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    substitutions: float
    deletions: float
    insertions: float
    confidence: float
    latency_s: float
    condition_index: int = -1
    transform_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "model_name": self.model_name,
            "defense_name": self.defense_name,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "wer": self.wer,
            "cer": self.cer,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "confidence": self.confidence,
            "latency_s": self.latency_s,
            "condition_index": self.condition_index,
            "transform_name": self.transform_name,
        }


@dataclass
class BatchTranscriptionResult:
    """Aggregated result of batch transcription across multiple utterances.

    Attributes
    ----------
    model_name : str
        ASR model used.
    defense_name : str
        Defense preprocessing applied.
    n_utterances : int
        Number of utterances transcribed.
    records : list[SingleTranscriptionRecord]
        Per-utterance records.
    mean_wer : float
        Mean WER across utterances.
    mean_cer : float
        Mean CER across utterances.
    std_wer : float
        Standard deviation of WER.
    total_latency_s : float
        Total transcription wall time.
    """

    model_name: str
    defense_name: str
    n_utterances: int
    records: list[SingleTranscriptionRecord] = field(default_factory=list)
    mean_wer: float = 0.0
    mean_cer: float = 0.0
    std_wer: float = 0.0
    total_latency_s: float = 0.0

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "defense_name": self.defense_name,
            "n_utterances": self.n_utterances,
            "mean_wer": round(self.mean_wer, 4),
            "mean_cer": round(self.mean_cer, 4),
            "std_wer": round(self.std_wer, 4),
            "total_latency_s": round(self.total_latency_s, 4),
            "records": [r.to_dict() for r in self.records],
        }

    def summary_stats(self) -> dict[str, float]:
        """Return summary statistics dict."""
        wers = [r.wer for r in self.records if np.isfinite(r.wer)]
        cers = [r.cer for r in self.records if np.isfinite(r.cer)]
        confs = [r.confidence for r in self.records]
        lats = [r.latency_s for r in self.records]
        return {
            "mean_wer": float(np.mean(wers)) if wers else 0.0,
            "std_wer": float(np.std(wers)) if wers else 0.0,
            "median_wer": float(np.median(wers)) if wers else 0.0,
            "mean_cer": float(np.mean(cers)) if cers else 0.0,
            "std_cer": float(np.std(cers)) if cers else 0.0,
            "mean_confidence": float(np.mean(confs)) if confs else 0.0,
            "mean_latency_s": float(np.mean(lats)) if lats else 0.0,
            "total_latency_s": self.total_latency_s,
            "n_utterances": self.n_utterances,
        }


class BatchTranscriber:
    """Batch transcription engine integrating ASR and defense preprocessing.

    Parameters
    ----------
    engine : ASREngine
        Loaded or loadable ASR engine.
    defense : ASRDefense or None
        Defense preprocessing to apply before ASR. If None, uses IdentityDefense.
    """

    def __init__(
        self,
        engine: ASREngine,
        defense: ASRDefense | None = None,
    ) -> None:
        self.engine = engine
        self.defense = defense if defense is not None else IdentityDefense()
        self.latencies: list[float] = []
        # Ensure model is loaded
        try:
            self.engine.load()
        except Exception as e:
            logger.warning("Engine load warning (%s): %s", engine.name, e)

    def mean_latency(self) -> float:
        """Return the mean per-sample transcription latency in seconds."""
        return float(np.mean(self.latencies)) if self.latencies else 0.0

    def transcribe_batch(
        self,
        signals: list[Signal],
        references: list[str] | None = None,
        utterance_ids: list[str] | None = None,
        condition_index: int = -1,
        transform_name: str = "",
    ) -> BatchTranscriptionResult:
        """Transcribe a batch of signals.

        Applies defense preprocessing, runs ASR, computes WER/CER vs references.

        Parameters
        ----------
        signals : list[Signal]
            Input audio signals.
        references : list[str] or None
            Ground-truth transcripts. If None, read from signal metadata.
        utterance_ids : list[str] or None
            Utterance identifiers. Auto-generated if None.
        condition_index : int
            Experiment condition index for record tracking.
        transform_name : str
            Name of the transform applied to these signals.

        Returns
        -------
        BatchTranscriptionResult
        """
        if references is None:
            references = [sig.metadata.get("transcript", "") for sig in signals]

        if utterance_ids is None:
            utterance_ids = [f"utt_{i:04d}" for i in range(len(signals))]

        records: list[SingleTranscriptionRecord] = []
        total_start = time.perf_counter()

        for i, (sig, ref) in enumerate(zip(signals, references)):
            utt_id = utterance_ids[i] if i < len(utterance_ids) else f"utt_{i:04d}"
            ref_norm = normalize_transcript(ref)

            try:
                # Apply defense preprocessing
                defended_sig = self.defense(sig)
            except Exception as e:
                logger.warning("Defense failed for %s: %s", utt_id, e)
                defended_sig = sig

            try:
                start_t = time.perf_counter()
                result = self.engine.transcribe(defended_sig)
                latency = time.perf_counter() - start_t
            except Exception as e:
                logger.warning("ASR failed for %s: %s", utt_id, e)
                result = TranscriptionResult(text="", confidence=0.0, language="en")
                latency = 0.0

            self.latencies.append(latency)

            hyp_norm = normalize_transcript(result.text)

            try:
                detailed = compute_detailed_wer(ref_norm, hyp_norm)
                wer_val = detailed["wer"]
                cer_val = compute_cer(ref_norm, hyp_norm)
                subs = detailed["substitutions"]
                dels = detailed["deletions"]
                ins = detailed["insertions"]
            except Exception:
                wer_val = 1.0
                cer_val = 1.0
                subs = dels = ins = 0.0

            records.append(
                SingleTranscriptionRecord(
                    utterance_id=utt_id,
                    model_name=self.engine.name,
                    defense_name=self.defense.name,
                    reference=ref_norm,
                    hypothesis=hyp_norm,
                    wer=float(np.clip(wer_val, 0.0, 10.0)),
                    cer=float(np.clip(cer_val, 0.0, 10.0)),
                    substitutions=float(subs),
                    deletions=float(dels),
                    insertions=float(ins),
                    confidence=float(result.confidence or 0.5),
                    latency_s=float(latency),
                    condition_index=condition_index,
                    transform_name=transform_name,
                )
            )

        total_latency = time.perf_counter() - total_start
        wers = [r.wer for r in records if np.isfinite(r.wer)]
        cers = [r.cer for r in records if np.isfinite(r.cer)]

        return BatchTranscriptionResult(
            model_name=self.engine.name,
            defense_name=self.defense.name,
            n_utterances=len(records),
            records=records,
            mean_wer=float(np.mean(wers)) if wers else 0.0,
            mean_cer=float(np.mean(cers)) if cers else 0.0,
            std_wer=float(np.std(wers)) if wers else 0.0,
            total_latency_s=total_latency,
        )

    def transcribe_paired(
        self,
        originals: list[Signal],
        transformed: list[Signal],
        references: list[str],
        utterance_ids: list[str] | None = None,
        condition_index: int = -1,
        transform_name: str = "",
    ) -> dict[str, BatchTranscriptionResult]:
        """Transcribe original and transformed signals for paired comparison.

        Parameters
        ----------
        originals : list[Signal]
            Original (untransformed) audio signals.
        transformed : list[Signal]
            Transformed audio signals.
        references : list[str]
            Ground-truth transcripts.
        utterance_ids : list[str] or None
            Utterance identifiers.
        condition_index : int
            Experiment condition index.
        transform_name : str
            Transform name.

        Returns
        -------
        dict with keys 'original' and 'transformed', each a BatchTranscriptionResult.
        """
        return {
            "original": self.transcribe_batch(
                originals, references, utterance_ids=utterance_ids,
                condition_index=condition_index, transform_name="original",
            ),
            "transformed": self.transcribe_batch(
                transformed, references, utterance_ids=utterance_ids,
                condition_index=condition_index, transform_name=transform_name,
            ),
        }


class MultiModelEvaluator:
    """Evaluate transforms across multiple ASR models and defense conditions.

    Parameters
    ----------
    engines : list[ASREngine]
        List of ASR engines to evaluate. Must have distinct ``name`` attributes.
    """

    def __init__(self, engines: list[ASREngine]) -> None:
        if not engines:
            raise ValueError("At least one ASR engine is required.")
        self.engines = engines

    def evaluate_all_models(
        self,
        signals: list[Signal],
        references: list[str],
        utterance_ids: list[str] | None = None,
        defenses: list[ASRDefense] | None = None,
    ) -> dict[str, dict[str, BatchTranscriptionResult]]:
        """Evaluate all model × defense combinations.

        Parameters
        ----------
        signals : list[Signal]
            Input audio signals.
        references : list[str]
            Ground-truth transcripts.
        utterance_ids : list[str] or None
            Utterance identifiers.
        defenses : list[ASRDefense] or None
            Defenses to evaluate. Uses IdentityDefense only if None.

        Returns
        -------
        dict[model_name, dict[defense_name, BatchTranscriptionResult]]
        """
        if defenses is None:
            defenses = [IdentityDefense()]

        results: dict[str, dict[str, BatchTranscriptionResult]] = {}

        for engine in self.engines:
            results[engine.name] = {}
            for defense in defenses:
                logger.info(
                    "Evaluating model=%s defense=%s on %d signals",
                    engine.name, defense.name, len(signals),
                )
                try:
                    transcriber = BatchTranscriber(engine=engine, defense=defense)
                    batch_result = transcriber.transcribe_batch(
                        signals=signals,
                        references=references,
                        utterance_ids=utterance_ids,
                    )
                    results[engine.name][defense.name] = batch_result
                except Exception as e:
                    logger.error(
                        "Evaluation failed model=%s defense=%s: %s",
                        engine.name, defense.name, e,
                    )

        return results

    def compute_cross_model_failure_rate(
        self,
        results: dict[str, dict[str, BatchTranscriptionResult]],
        wer_threshold: float = 0.5,
        defense_name: str = "defense.raw",
    ) -> float:
        """Compute fraction of utterances where ALL models fail.

        A model 'fails' on an utterance when WER > wer_threshold.
        Cross-model failure rate = fraction of utterances where all models fail.

        Parameters
        ----------
        results : dict[model_name, dict[defense_name, BatchTranscriptionResult]]
        wer_threshold : float
            WER threshold above which ASR has 'failed'.
        defense_name : str
            Which defense condition to use for this computation.

        Returns
        -------
        float
            Cross-model failure rate in [0, 1].
        """
        # Collect per-utterance WER for each model under the given defense
        per_model_wers: dict[str, list[float]] = {}
        for model_name, def_results in results.items():
            if defense_name in def_results:
                per_model_wers[model_name] = [
                    r.wer for r in def_results[defense_name].records
                ]
            elif def_results:
                # Fall back to first available defense
                first_def = list(def_results.values())[0]
                per_model_wers[model_name] = [r.wer for r in first_def.records]

        if not per_model_wers:
            return 0.0

        n_utt = min(len(v) for v in per_model_wers.values())
        if n_utt == 0:
            return 0.0

        failures = 0
        for i in range(n_utt):
            all_fail = all(
                per_model_wers[m][i] > wer_threshold
                for m in per_model_wers
                if i < len(per_model_wers[m])
            )
            if all_fail:
                failures += 1

        return float(failures / n_utt)

    def compute_defense_survival_rate(
        self,
        original_wer: float,
        transformed_wer: float,
        post_defense_wer: float,
        epsilon: float = 1e-6,
    ) -> float:
        """Compute Defense Survival Rate (DSR).

        DSR = (WER_post_defense - WER_original) / (WER_transformed - WER_original + ε)

        Returns
        -------
        float
            DSR; 1.0 = defense recovers nothing, 0.0 = full recovery.
        """
        degradation = transformed_wer - original_wer
        survival = post_defense_wer - original_wer
        if abs(degradation) < epsilon:
            return 1.0 if abs(survival) < epsilon else 0.0
        return float(np.clip(survival / (degradation + epsilon), -0.5, 2.0))

    def compute_asr_success_rate(
        self,
        batch_result: BatchTranscriptionResult,
        wer_threshold: float = 0.3,
    ) -> float:
        """Compute ASR Success Rate.

        Fraction of utterances where ASR succeeds (WER < wer_threshold).

        Parameters
        ----------
        batch_result : BatchTranscriptionResult
        wer_threshold : float

        Returns
        -------
        float in [0, 1].
        """
        if not batch_result.records:
            return 0.0
        successes = sum(1 for r in batch_result.records if r.wer < wer_threshold)
        return float(successes / len(batch_result.records))

    def summary_table(
        self,
        results: dict[str, dict[str, BatchTranscriptionResult]],
    ) -> list[dict[str, Any]]:
        """Build a cross-model summary table.

        Returns
        -------
        list of dicts, one per model × defense combination.
        """
        rows = []
        for model_name, def_results in results.items():
            for defense_name, batch_result in def_results.items():
                stats = batch_result.summary_stats()
                rows.append({
                    "model_name": model_name,
                    "defense_name": defense_name,
                    **stats,
                })
        return rows


class ASRExperimentRunner:
    """Full ASR evaluation integrated into the experiment pipeline.

    Parameters
    ----------
    engines : list[ASREngine]
        ASR engines to evaluate.
    defenses : list[ASRDefense] or None
        Defenses to apply. Defaults to DefensePipeline.all_defenses().
    """

    def __init__(
        self,
        engines: list[ASREngine],
        defenses: list[ASRDefense] | None = None,
    ) -> None:
        self.engines = engines
        self.defenses = defenses if defenses is not None else DefensePipeline.all_defenses()
        self.evaluator = MultiModelEvaluator(engines=engines)

    def run_asr_evaluation(
        self,
        signals: list[Signal],
        transformed_signals: list[Signal],
        references: list[str],
        utterance_ids: list[str] | None = None,
        condition_index: int = -1,
        transform_name: str = "",
    ) -> dict[str, Any]:
        """Run full ASR evaluation across all models and defenses.

        Evaluates both original and transformed signals across all model ×
        defense combinations and computes cross-model failure rate, defense
        survival rate, and ASR-SR metrics.

        Parameters
        ----------
        signals : list[Signal]
            Original (untransformed) signals.
        transformed_signals : list[Signal]
            Transformed signals.
        references : list[str]
            Ground-truth transcripts.
        utterance_ids : list[str] or None
            Utterance identifiers.
        condition_index : int
            Experiment condition index.
        transform_name : str
            Name of the transform evaluated.

        Returns
        -------
        dict with keys:
            original_results, transformed_results (both model×defense dicts),
            cross_model_failure_rate, per_model_wer_original,
            per_model_wer_transformed, defense_survival_rates,
            asr_success_rates, summary_table.
        """
        logger.info(
            "ASR evaluation: transform=%s, n_signals=%d, n_models=%d, n_defenses=%d",
            transform_name, len(signals), len(self.engines), len(self.defenses),
        )

        # Evaluate originals
        original_results = self.evaluator.evaluate_all_models(
            signals=signals,
            references=references,
            utterance_ids=utterance_ids,
            defenses=[self.defenses[0]] if self.defenses else None,
        )

        # Evaluate transformed across all defenses
        transformed_results = self.evaluator.evaluate_all_models(
            signals=transformed_signals,
            references=references,
            utterance_ids=utterance_ids,
            defenses=self.defenses,
        )

        # Per-model original WER
        per_model_wer_orig: dict[str, float] = {}
        for model_name, def_res in original_results.items():
            first_def_res = list(def_res.values())[0] if def_res else None
            per_model_wer_orig[model_name] = first_def_res.mean_wer if first_def_res else 0.0

        # Per-model transformed WER (identity defense only)
        per_model_wer_trans: dict[str, float] = {}
        for model_name, def_res in transformed_results.items():
            # Use raw/identity defense
            identity_res = def_res.get(
                "defense.raw",
                list(def_res.values())[0] if def_res else None,
            )
            per_model_wer_trans[model_name] = identity_res.mean_wer if identity_res else 0.0

        # Defense survival rates per model × defense
        defense_survival_rates: dict[str, dict[str, float]] = {}
        for model_name in [e.name for e in self.engines]:
            defense_survival_rates[model_name] = {}
            orig_wer = per_model_wer_orig.get(model_name, 0.0)
            trans_wer = per_model_wer_trans.get(model_name, 0.0)
            if model_name in transformed_results:
                for def_name, batch_res in transformed_results[model_name].items():
                    dsr = self.evaluator.compute_defense_survival_rate(
                        orig_wer, trans_wer, batch_res.mean_wer
                    )
                    defense_survival_rates[model_name][def_name] = dsr

        # Cross-model failure rate
        cmfr = self.evaluator.compute_cross_model_failure_rate(
            transformed_results, defense_name="defense.raw"
        )

        # ASR success rates (identity defense on transformed)
        asr_success_rates: dict[str, float] = {}
        for model_name in [e.name for e in self.engines]:
            if model_name in transformed_results:
                raw_res = transformed_results[model_name].get(
                    "defense.raw",
                    list(transformed_results[model_name].values())[0]
                    if transformed_results[model_name]
                    else None,
                )
                if raw_res:
                    asr_success_rates[model_name] = self.evaluator.compute_asr_success_rate(raw_res)

        summary = self.evaluator.summary_table(transformed_results)

        return {
            "transform_name": transform_name,
            "condition_index": condition_index,
            "original_results": {
                m: {d: r.to_dict() for d, r in dr.items()}
                for m, dr in original_results.items()
            },
            "transformed_results": {
                m: {d: r.to_dict() for d, r in dr.items()}
                for m, dr in transformed_results.items()
            },
            "cross_model_failure_rate": cmfr,
            "per_model_wer_original": per_model_wer_orig,
            "per_model_wer_transformed": per_model_wer_trans,
            "defense_survival_rates": defense_survival_rates,
            "asr_success_rates": asr_success_rates,
            "summary_table": summary,
        }
