from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReportGenerator:
    results_dir: Path = Path("results")
    output_dir: Path = Path("results")

    def generate_tables(self) -> None:
        raise NotImplementedError

    def generate_figures(self) -> None:
        raise NotImplementedError
