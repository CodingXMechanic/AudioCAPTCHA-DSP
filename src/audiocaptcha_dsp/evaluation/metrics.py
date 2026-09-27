from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from jiwer import wer, cer, process_words
from scipy.signal import stft

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.psychoacoustics.bark_scale import hz_to_bark


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate between reference and hypothesis transcriptions."""
    return float(wer(reference, hypothesis))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate between reference and hypothesis transcriptions."""
    return float(cer(reference, hypothesis))


def compute_detailed_wer(reference: str, hypothesis: str) -> dict[str, float]:
    """Compute detailed WER metrics: WER, substitutions, deletions, insertions."""
    res = process_words(reference, hypothesis)
    total_words = max(1, len(res.references[0])) if res.references and res.references[0] else 1
    return {
        "wer": float(res.wer),
        "substitutions": float(res.substitutions / total_words),
        "deletions": float(res.deletions / total_words),
        "insertions": float(res.insertions / total_words),
        "hits": float(res.hits / total_words),
    }


def compute_snr(original: np.ndarray, processed: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB."""
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
    if signal_power < 1e-15:
        return 0.0
    return float(10.0 * np.log10(signal_power / noise_power))


def compute_si_sdr(original: np.ndarray, processed: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.
    
    Reference: Le Roux et al., SDR — half-baked or well done? ICASSP 2019.
    """
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(original.shape[-1], processed.shape[-1])
    s = original[:min_len]
    s_hat = processed[:min_len]
    
    # Zero-mean
    s = s - np.mean(s)
    s_hat = s_hat - np.mean(s_hat)
    
    s_norm_sq = np.sum(s ** 2)
    if s_norm_sq < 1e-15:
        return 0.0
        
    alpha = np.sum(s_hat * s) / s_norm_sq
    e_target = alpha * s
    e_res = s_hat - e_target
    
    target_power = np.sum(e_target ** 2)
    res_power = np.sum(e_res ** 2)
    
    if res_power < 1e-15:
        return float("inf")
    if target_power < 1e-15:
        return -100.0
        
    return float(10.0 * np.log10(target_power / res_power))


def compute_rmse(original: np.ndarray, processed: np.ndarray) -> float:
    """Root Mean Squared Error."""
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(original.shape[-1], processed.shape[-1])
    original = original[:min_len]
    processed = processed[:min_len]
    return float(np.sqrt(np.mean((original - processed) ** 2)))


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """Mean Squared Error."""
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(original.shape[-1], processed.shape[-1])
    original = original[:min_len]
    processed = processed[:min_len]
    return float(np.mean((original - processed) ** 2))


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    rmse = compute_rmse(original, processed)
    if rmse < 1e-15:
        return float("inf")
    peak = np.max(np.abs(np.asarray(original, dtype=np.float64)))
    if peak < 1e-15:
        return 0.0
    return float(20.0 * np.log10(peak / rmse))


def compute_spectral_convergence(original: np.ndarray, processed: np.ndarray) -> float:
    """Spectral Convergence metric."""
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


def compute_normalized_cross_correlation(original: np.ndarray, processed: np.ndarray) -> float:
    """Normalized Cross Correlation [-1, 1]."""
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


def compute_log_spectral_distance(original: np.ndarray, processed: np.ndarray, sr: int = 16000) -> float:
    """Log-Spectral Distance (LSD) in dB.
    
    Measures the average distance between logarithmic power spectra.
    """
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(len(original), len(processed))
    if min_len < 256:
        return 0.0
    s1 = original[:min_len]
    s2 = processed[:min_len]
    
    nperseg = min(512, min_len)
    _, _, zxx1 = stft(s1, fs=sr, nperseg=nperseg)
    _, _, zxx2 = stft(s2, fs=sr, nperseg=nperseg)
    
    p1 = np.maximum(np.abs(zxx1) ** 2, 1e-12)
    p2 = np.maximum(np.abs(zxx2) ** 2, 1e-12)
    
    log_diff = 10.0 * np.log10(p1 / p2)
    # Root mean squared across frequency bins per frame, then averaged across frames
    lsd_per_frame = np.sqrt(np.mean(log_diff ** 2, axis=0))
    return float(np.mean(lsd_per_frame))


def compute_mbsd(original: np.ndarray, processed: np.ndarray, sr: int = 16000) -> float:
    """Modified Bark Spectral Distortion (MBSD).
    
    Reference: Yang & Yantorno (1997), Performance of the Modified Bark Spectral
    Distortion as an Objective Speech Quality Measure, IEEE Workshop on Speech Coding.
    """
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(len(original), len(processed))
    if min_len < 256:
        return 0.0
    s1 = original[:min_len]
    s2 = processed[:min_len]
    
    nperseg = min(512, min_len)
    f, _, zxx1 = stft(s1, fs=sr, nperseg=nperseg)
    _, _, zxx2 = stft(s2, fs=sr, nperseg=nperseg)
    
    p1 = np.abs(zxx1) ** 2
    p2 = np.abs(zxx2) ** 2
    
    # Map frequencies to Bark
    bark_freqs = hz_to_bark(f)
    n_bark = 24
    bark_edges = np.linspace(0, 24, n_bark + 1)
    
    b1 = np.zeros((n_bark, p1.shape[1]), dtype=np.float64)
    b2 = np.zeros((n_bark, p2.shape[1]), dtype=np.float64)
    
    for i in range(n_bark):
        mask = (bark_freqs >= bark_edges[i]) & (bark_freqs < bark_edges[i + 1])
        if np.any(mask):
            b1[i] = np.sum(p1[mask, :], axis=0)
            b2[i] = np.sum(p2[mask, :], axis=0)
            
    # Convert to dB Bark spectra
    b1_db = 10.0 * np.log10(np.maximum(b1, 1e-10))
    b2_db = 10.0 * np.log10(np.maximum(b2, 1e-10))
    
    # Bark spectral distortion only on active frames
    frame_energy = np.sum(b1, axis=0)
    active_frames = frame_energy > 1e-4
    if not np.any(active_frames):
        return 0.0
        
    diff = np.abs(b1_db[:, active_frames] - b2_db[:, active_frames])
    return float(np.mean(diff))


def compute_stoi_proxy(original: np.ndarray, processed: np.ndarray, sr: int = 16000) -> float:
    """Lightweight correlation-based Short-Time Objective Intelligibility proxy (0.0 to 1.0).
    
    Computes intermediate subband envelope correlation following Taal et al. (2011).
    """
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    min_len = min(len(original), len(processed))
    if min_len < 512:
        return 1.0
        
    s1 = original[:min_len]
    s2 = processed[:min_len]
    
    nperseg = min(512, min_len)
    _, _, zxx1 = stft(s1, fs=sr, nperseg=nperseg)
    _, _, zxx2 = stft(s2, fs=sr, nperseg=nperseg)
    
    env1 = np.abs(zxx1)
    env2 = np.abs(zxx2)
    
    correlations = []
    for k in range(env1.shape[0]):
        x = env1[k] - np.mean(env1[k])
        y = env2[k] - np.mean(env2[k])
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x > 1e-8 and norm_y > 1e-8:
            r = np.dot(x, y) / (norm_x * norm_y)
            correlations.append(np.clip(r, -1.0, 1.0))
            
    if not correlations:
        return 1.0
    mean_corr = float(np.mean(correlations))
    # Map [-1, 1] to [0, 1]
    return max(0.0, min(1.0, (mean_corr + 1.0) / 2.0))


def compute_human_asr_gap(
    human_success_rate: float,
    asr_success_rate: float,
) -> float:
    """Compute the Human-ASR Gap (HAG).
    
    HAG = HSR - ASR-SR
    Higher positive values indicate superior CAPTCHA efficacy:
    humans easily succeed while automated ASR models fail.
    """
    return float(np.clip(human_success_rate, 0.0, 1.0) - np.clip(asr_success_rate, 0.0, 1.0))


def compute_metrics(original: Signal, processed: Signal) -> dict[str, float]:
    """Compute comprehensive research metric suite for audio transformation evaluation."""
    orig_mono = original.to_mono().waveform
    proc_mono = processed.to_mono().waveform
    sr = original.sample_rate

    orig_rms = float(np.sqrt(np.mean(orig_mono ** 2)))
    proc_rms = float(np.sqrt(np.mean(proc_mono ** 2)))

    return {
        "snr_db": compute_snr(orig_mono, proc_mono),
        "si_sdr_db": compute_si_sdr(orig_mono, proc_mono),
        "rmse": compute_rmse(orig_mono, proc_mono),
        "mse": compute_mse(orig_mono, proc_mono),
        "psnr_db": compute_psnr(orig_mono, proc_mono),
        "spectral_convergence": compute_spectral_convergence(orig_mono, proc_mono),
        "normalized_cross_correlation": compute_normalized_cross_correlation(orig_mono, proc_mono),
        "log_spectral_distance": compute_log_spectral_distance(orig_mono, proc_mono, sr=sr),
        "mbsd": compute_mbsd(orig_mono, proc_mono, sr=sr),
        "stoi_proxy": compute_stoi_proxy(orig_mono, proc_mono, sr=sr),
        "rms_original": orig_rms,
        "rms_processed": proc_rms,
        "rms_ratio": float(proc_rms / (orig_rms + 1e-12)),
    }
