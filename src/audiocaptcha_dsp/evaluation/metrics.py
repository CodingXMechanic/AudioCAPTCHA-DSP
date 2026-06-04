from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from jiwer import wer, cer
from scipy.signal import stft

from audiocaptcha_dsp.core.signal import Signal


def compute_wer(reference: str, hypothesis: str) -> float:
    return float(wer(reference, hypothesis))


def compute_cer(reference: str, hypothesis: str) -> float:
    return float(cer(reference, hypothesis))


def compute_snr(original: np.ndarray, processed: np.ndarray) -> float:
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(original.shape[-1], processed.shape[-1])
    original = original[:min_len]
    processed = processed[:min_len]
    noise = original - processed
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-15:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def compute_rmse(original: np.ndarray, processed: np.ndarray) -> float:
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(original.shape[-1], processed.shape[-1])
    original = original[:min_len]
    processed = processed[:min_len]
    return float(np.sqrt(np.mean((original - processed) ** 2)))


def compute_spectral_convergence(original: np.ndarray, processed: np.ndarray) -> float:
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(original.shape[-1], processed.shape[-1])
    original = original[:min_len]
    processed = processed[:min_len]
    orig_mag = np.abs(np.fft.rfft(original))
    proc_mag = np.abs(np.fft.rfft(processed))
    numerator = np.linalg.norm(orig_mag - proc_mag)
    denominator = np.linalg.norm(orig_mag)
    if denominator < 1e-15:
        return 0.0
    return float(numerator / denominator)


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    rmse = compute_rmse(original, processed)
    if rmse < 1e-15:
        return float("inf")
    peak = np.max(np.abs(np.asarray(original, dtype=np.float64)))
    if peak < 1e-15:
        return 0.0
    return float(20.0 * np.log10(peak / rmse))


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(original.shape[-1], processed.shape[-1])
    original = original[:min_len]
    processed = processed[:min_len]
    return float(np.mean((original - processed) ** 2))


def compute_normalized_cross_correlation(original: np.ndarray, processed: np.ndarray) -> float:
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(original.shape[-1], processed.shape[-1])
    original = original[:min_len]
    processed = processed[:min_len]
    orig_centered = original - np.mean(original)
    proc_centered = processed - np.mean(processed)
    numerator = np.sum(orig_centered * proc_centered)
    denominator = np.sqrt(np.sum(orig_centered ** 2) * np.sum(proc_centered ** 2))
    if denominator < 1e-15:
        return 1.0 if np.allclose(original, processed) else 0.0
    return float(numerator / denominator)


def compute_metrics(original: Signal, processed: Signal) -> dict[str, float]:
    orig_mono = original.to_mono().waveform
    proc_mono = processed.to_mono().waveform
    return {
        "snr_db": compute_snr(orig_mono, proc_mono),
        "rmse": compute_rmse(orig_mono, proc_mono),
        "mse": compute_mse(orig_mono, proc_mono),
        "psnr_db": compute_psnr(orig_mono, proc_mono),
        "spectral_convergence": compute_spectral_convergence(orig_mono, proc_mono),
        "normalized_cross_correlation": compute_normalized_cross_correlation(orig_mono, proc_mono),
        "rms_original": float(np.sqrt(np.mean(orig_mono ** 2))),
        "rms_processed": float(np.sqrt(np.mean(proc_mono ** 2))),
        "rms_ratio": float(np.sqrt(np.mean(proc_mono ** 2)) / (np.sqrt(np.mean(orig_mono ** 2)) + 1e-12)),
    }
