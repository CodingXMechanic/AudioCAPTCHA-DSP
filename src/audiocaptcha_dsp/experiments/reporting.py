"""
Publication-Ready Report Generator
====================================
Generates paper-ready tables (Markdown, CSV, LaTeX, JSON),
visualization metadata, and reproducibility reports from experiment manifests.

Designed for the AudioCAPTCHA-DSP Human-vs-ASR research platform.
"""
from __future__ import annotations

import csv
import json
import logging
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from audiocaptcha_dsp.experiments.benchmark import ExperimentManifest

logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    results_dir: Path
    output_dir: Path
    experiment_ids: list[str]
    ci_level: float = 0.95
    include_latex: bool = True
    include_csv: bool = True
    include_json: bool = True
    top_n_transforms: int = 10

class TableFormatter:
    @staticmethod
    def to_markdown(headers: list[str], rows: list[list[Any]], title: str = '') -> str:
        lines = []
        if title:
            lines.append(f"### {title}")
            lines.append("")
        
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def to_csv_str(headers: list[str], rows: list[list[Any]]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
        return output.getvalue()
    
    @staticmethod
    def to_latex(headers: list[str], rows: list[list[Any]], caption: str = '', label: str = '') -> str:
        def escape_tex(text):
            return str(text).replace("_", "\\_").replace("%", "\\%")
            
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\begin{tabular}{" + "c" * len(headers) + "}",
            "\\toprule"
        ]
        lines.append(" & ".join(escape_tex(h) for h in headers) + " \\\\")
        lines.append("\\midrule")
        for row in rows:
            lines.append(" & ".join(escape_tex(c) for c in row) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{escape_tex(caption)}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")
        return "\n".join(lines) + "\n"

class MetricSummaryTable:
    @staticmethod
    def build(manifest: ExperimentManifest, metrics: list[str] | None = None) -> list[dict[str, Any]]:
        rows = []
        if not metrics:
            metrics = ['pesq', 'stoi', 'snr']
        for i, cond in enumerate(manifest.conditions):
            row = {'condition_index': i}
            if hasattr(cond, 'transform_chain') and cond.transform_chain:
                row['transforms'] = " + ".join([t['name'] for t in cond.transform_chain])
            else:
                row['transforms'] = "Baseline"
            for m in metrics:
                val = 0.0
                if hasattr(manifest, 'results') and str(i) in manifest.results:
                    val = manifest.results[str(i)].get(m, 0.0)
                row[m] = val
            rows.append(row)
        return rows
        
    @staticmethod
    def format_markdown(rows: list[dict]) -> str:
        if not rows: return ""
        headers = list(rows[0].keys())
        table_rows = [[r[h] for h in headers] for r in rows]
        return TableFormatter.to_markdown(headers, table_rows, "Metric Summary")
        
    @staticmethod
    def format_latex(rows: list[dict], caption: str) -> str:
        if not rows: return ""
        headers = list(rows[0].keys())
        table_rows = [[r[h] for h in headers] for r in rows]
        return TableFormatter.to_latex(headers, table_rows, caption=caption)
        
    @staticmethod
    def format_csv(rows: list[dict]) -> str:
        if not rows: return ""
        headers = list(rows[0].keys())
        table_rows = [[r[h] for h in headers] for r in rows]
        return TableFormatter.to_csv_str(headers, table_rows)

class RankingTable:
    @staticmethod
    def rank_by_metric(rows: list[dict], metric: str, ascending: bool = False) -> list[dict]:
        return sorted(rows, key=lambda x: x.get(metric, 0.0), reverse=not ascending)
        
    @staticmethod
    def pareto_rank(rows: list[dict], maximize_metrics: list[str], minimize_metrics: list[str]) -> list[dict]:
        ranked = []
        for i, row in enumerate(rows):
            dominated_count = 0
            for j, other in enumerate(rows):
                if i == j: continue
                
                strictly_worse = False
                worse_or_equal = True
                
                for m in maximize_metrics:
                    if row.get(m, 0) < other.get(m, 0): strictly_worse = True
                    if row.get(m, 0) > other.get(m, 0): worse_or_equal = False
                for m in minimize_metrics:
                    if row.get(m, 0) > other.get(m, 0): strictly_worse = True
                    if row.get(m, 0) < other.get(m, 0): worse_or_equal = False
                    
                if strictly_worse and worse_or_equal:
                    dominated_count += 1
            
            new_row = dict(row)
            new_row['pareto_rank'] = dominated_count
            ranked.append(new_row)
            
        return sorted(ranked, key=lambda x: x['pareto_rank'])
        
    @staticmethod
    def format_ranking_markdown(rows: list[dict], metrics: list[str]) -> str:
        if not rows: return ""
        headers = ["condition_index", "transforms"] + metrics + (["pareto_rank"] if "pareto_rank" in rows[0] else [])
        table_rows = []
        for r in rows:
            table_rows.append([r.get(h, "") for h in headers])
        return TableFormatter.to_markdown(headers, table_rows, "Ranking Table")

class ReproducibilityReport:
    @staticmethod
    def generate(manifest: ExperimentManifest) -> dict[str, Any]:
        report = {
            "experiment_id": manifest.experiment_id,
            "timestamp": manifest.timestamp,
            "seed": getattr(manifest, "seed", 42),
            "n_samples": getattr(manifest, "dataset_size", 0),
            "n_conditions": len(manifest.conditions),
            "transform_chains": [],
            "metric_names": getattr(manifest, "metrics", []),
            "software_note": "AudioCAPTCHA-DSP Research Platform"
        }
        for c in manifest.conditions:
            chain = [t['name'] for t in c.transform_chain] if hasattr(c, 'transform_chain') else []
            if chain not in report['transform_chains']:
                report['transform_chains'].append(chain)
        return report

    @staticmethod
    def to_markdown(report: dict) -> str:
        lines = ["### Reproducibility Appendix", ""]
        for k, v in report.items():
            lines.append(f"**{k}**: {v}")
        return "\n".join(lines) + "\n"

@dataclass
class ReportGenerator:
    config: ReportConfig
    
    def generate_all(self, manifest: ExperimentManifest) -> dict[str, Path]:
        tables = self.generate_tables(manifest)
        paths = self.save_tables(tables)
        
        md_app = self.generate_reproducibility_appendix(manifest)
        app_path = self.config.output_dir / "reproducibility.md"
        app_path.write_text(md_app)
        paths["reproducibility_md"] = app_path
        
        lim_rep = self.generate_limitation_report(manifest)
        lim_path = self.config.output_dir / "limitations.md"
        lim_path.write_text(lim_rep)
        paths["limitations_md"] = lim_path
        
        return paths
    
    def generate_tables(self, manifest: ExperimentManifest) -> dict[str, str]:
        metrics = getattr(manifest, "metrics", ["snr"])
        rows = MetricSummaryTable.build(manifest, metrics)
        
        tables = {}
        tables["markdown"] = MetricSummaryTable.format_markdown(rows)
        if self.config.include_csv:
            tables["csv"] = MetricSummaryTable.format_csv(rows)
        if self.config.include_latex:
            tables["latex"] = MetricSummaryTable.format_latex(rows, caption="Experiment Results")
        if self.config.include_json:
            tables["json"] = json.dumps(rows, indent=2)
            
        return tables
    
    def save_tables(self, tables: dict[str, str]) -> dict[str, Path]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for fmt, content in tables.items():
            ext = fmt if fmt != "markdown" else "md"
            ext = ext if ext != "latex" else "tex"
            p = self.config.output_dir / f"summary.{ext}"
            p.write_text(content)
            paths[fmt] = p
        return paths
    
    def generate_reproducibility_appendix(self, manifest: ExperimentManifest) -> str:
        rep = ReproducibilityReport.generate(manifest)
        return ReproducibilityReport.to_markdown(rep)
    
    def generate_limitation_report(self, manifest: ExperimentManifest) -> str:
        lines = [
            "### Limitation and Failure-Case Report",
            "",
            "This report details conditions where transformations resulted in degenerate audio or metric failures.",
            ""
        ]
        issues = 0
        for i, cond in enumerate(manifest.conditions):
            val = 0.0
            if hasattr(manifest, 'results') and str(i) in manifest.results:
                val = manifest.results[str(i)].get('snr', 0.0)
            if val < -10.0:
                lines.append(f"- Condition {i}: SNR below acceptable threshold ({val:.2f} dB).")
                issues += 1
        if issues == 0:
            lines.append("No significant limitations or failures observed.")
        return "\n".join(lines) + "\n"
