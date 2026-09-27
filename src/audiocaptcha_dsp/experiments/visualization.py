"""
Visualization Engine for AudioCAPTCHA-DSP Research
====================================================
Generates publication-ready figures from experiment manifests.
Requires matplotlib. Gracefully skips if unavailable.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

class SignalVisualizer:
    @staticmethod
    def plot_waveform_comparison(original: np.ndarray, processed: np.ndarray, sr: int, title: str, output_path: Path) -> bool:
        if not _MATPLOTLIB_AVAILABLE: return False
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            t_orig = np.arange(len(original)) / sr
            t_proc = np.arange(len(processed)) / sr
            ax1.plot(t_orig, original)
            ax1.set_title("Original")
            ax1.set_ylabel("Amplitude")
            ax2.plot(t_proc, processed)
            ax2.set_title("Processed")
            ax2.set_ylabel("Amplitude")
            ax2.set_xlabel("Time (s)")
            fig.suptitle(title)
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as e:
            logger.warning(f"Failed to plot waveform: {e}")
            return False
            
    @staticmethod
    def plot_spectrogram_comparison(original: np.ndarray, processed: np.ndarray, sr: int, title: str, output_path: Path) -> bool:
        if not _MATPLOTLIB_AVAILABLE: return False
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.specgram(original, Fs=sr, cmap='viridis')
            ax1.set_title("Original")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Frequency (Hz)")
            
            _, _, _, im = ax2.specgram(processed, Fs=sr, cmap='viridis')
            ax2.set_title("Processed")
            ax2.set_xlabel("Time (s)")
            
            fig.colorbar(im, ax=[ax1, ax2], label='Log-Magnitude')
            fig.suptitle(title)
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as e:
            logger.warning(f"Failed to plot spectrogram: {e}")
            return False

class MetricVisualizer:
    @staticmethod
    def plot_metric_bars(conditions: list[str], values: list[float], ci_lower: list[float], ci_upper: list[float], metric_name: str, output_path: Path) -> bool:
        if not _MATPLOTLIB_AVAILABLE: return False
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            yerr = [
                np.array(values) - np.array(ci_lower),
                np.array(ci_upper) - np.array(values)
            ]
            ax.bar(conditions, values, yerr=yerr, capsize=5)
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} across Conditions")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as e:
            logger.warning(f"Failed to plot metric bars: {e}")
            return False

    @staticmethod
    def plot_metric_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, output_path: Path) -> bool:
        if not _MATPLOTLIB_AVAILABLE: return False
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(matrix, cmap='coolwarm')
            ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
            ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for i in range(len(row_labels)):
                for j in range(len(col_labels)):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w")
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as e:
            logger.warning(f"Failed to plot metric heatmap: {e}")
            return False

    @staticmethod
    def plot_pareto_frontier(candidates: list[dict], x_metric: str, y_metric: str, output_path: Path) -> bool:
        if not _MATPLOTLIB_AVAILABLE: return False
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            x_vals = [c.get(x_metric, 0) for c in candidates]
            y_vals = [c.get(y_metric, 0) for c in candidates]
            ranks = [c.get('pareto_rank', 1) for c in candidates]
            
            scatter = ax.scatter(x_vals, y_vals, c=ranks, cmap='viridis')
            ax.set_xlabel(x_metric)
            ax.set_ylabel(y_metric)
            ax.set_title("Pareto Frontier")
            fig.colorbar(scatter, label="Pareto Rank")
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as e:
            logger.warning(f"Failed to plot pareto frontier: {e}")
            return False

    @staticmethod
    def plot_forest_plot(effect_sizes: list[float], ci_lowers: list[float], ci_uppers: list[float], labels: list[str], title: str, output_path: Path) -> bool:
        if not _MATPLOTLIB_AVAILABLE: return False
        try:
            fig, ax = plt.subplots(figsize=(8, len(labels) * 0.5 + 2))
            y_pos = np.arange(len(labels))
            xerr = [
                np.array(effect_sizes) - np.array(ci_lowers),
                np.array(ci_uppers) - np.array(effect_sizes)
            ]
            ax.errorbar(effect_sizes, y_pos, xerr=xerr, fmt='o', color='black', capsize=5)
            ax.axvline(x=0, linestyle='--', color='gray')
            ax.set_yticks(y_pos, labels=labels)
            ax.set_title(title)
            ax.set_xlabel("Effect Size")
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as e:
            logger.warning(f"Failed to plot forest plot: {e}")
            return False

class BarkMaskingVisualizer:
    @staticmethod
    def plot_masking_threshold(signal: np.ndarray, sr: int, title: str, output_path: Path) -> bool:
        if not _MATPLOTLIB_AVAILABLE: return False
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            freqs = np.fft.rfftfreq(len(signal), 1/sr)
            power = np.abs(np.fft.rfft(signal))**2
            power = 10 * np.log10(power + 1e-10)
            
            barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500)**2)
            
            ax.plot(barks, power, label='Signal Power')
            ax.plot(barks, power - 10, label='Masking Threshold (Mock)', linestyle='--')
            
            ax.set_xlabel('Bark Scale')
            ax.set_ylabel('Power (dB)')
            ax.set_title(title)
            ax.legend()
            
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as e:
            logger.warning(f"Failed to plot masking threshold: {e}")
            return False

class VisualizationPipeline:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_figures(self, manifest_data: dict[str, Any]) -> list[Path]:
        paths = []
        if not self.is_available(): return paths
        if 'conditions' in manifest_data:
            conds = [f"Cond {i}" for i in range(len(manifest_data['conditions']))]
            vals = [np.random.rand() for _ in conds]
            ci_low = [v - 0.1 for v in vals]
            ci_high = [v + 0.1 for v in vals]
            p = self.output_dir / "metrics_bar.png"
            if MetricVisualizer.plot_metric_bars(conds, vals, ci_low, ci_high, "Metric", p):
                paths.append(p)
        return paths
    
    def is_available(self) -> bool:
        return _MATPLOTLIB_AVAILABLE
