from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from audiocaptcha_dsp import __version__
from audiocaptcha_dsp.core.types import ExperimentConfig
from audiocaptcha_dsp.experiments.registry import ExperimentRegistry
from audiocaptcha_dsp.experiments.runner import ExperimentRunner

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiocaptcha",
        description="AudioCAPTCHA-DSP: Psychoacoustic multirate distortion framework",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("experiment", type=str, help="Experiment name (e.g., exp_05)")
    run_parser.add_argument("--config-dir", type=Path, default=Path("configs/experiments"))
    run_parser.add_argument("--save-wav", action="store_true", help="Save processed WAV files")

    list_parser = subparsers.add_parser("list", help="List available experiments")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "run":
        registry = ExperimentRegistry(experiments_dir=args.config_dir)
        experiments = registry.discover()
        if args.experiment not in experiments:
            logger.error("Experiment '%s' not found. Available: %s", args.experiment, list(experiments.keys()))
            return 1
        config_path = experiments[args.experiment]
        config = ExperimentConfig.from_yaml(config_path)
        logger.info("Loaded experiment: %s (%s)", config.name, config.experiment_id)
        runner = ExperimentRunner(config)
        benchmark = runner.run(save_wav=getattr(args, "save_wav", False))
        summary = benchmark.summary()
        print(f"\nExperiment: {summary['experiment_id']}")
        print(f"  Conditions: {summary['total_conditions']}")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Duration: {summary['total_duration_seconds']:.2f}s")
        if summary["per_condition_summaries"]:
            first = summary["per_condition_summaries"][0]
            if first.get("metric_summaries"):
                snr_stats = first["metric_summaries"].get("snr_db", {})
                if snr_stats:
                    print(f"  SNR (cond_0): mean={snr_stats.get('mean', 'N/A'):.2f} dB, std={snr_stats.get('std', 'N/A'):.2f}")
        return 0

    if args.command == "list":
        registry = ExperimentRegistry()
        experiments = registry.discover()
        if not experiments:
            print("No experiments found.")
            return 0
        for name, path in experiments.items():
            print(f"  {name}: {path}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
