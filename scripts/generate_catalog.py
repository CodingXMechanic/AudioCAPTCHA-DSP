#!/usr/bin/env python
"""Generate TRANSFORM_CATALOG.md from the central transform registry.

WHAT-REMAINS §15: "Every transform must cite its scientific basis where
applicable." Citations, descriptions, families and default parameters are
read from ``transforms/registry.py`` — the registry is the single source of
truth, so this catalog can never drift from the code.

Usage
-----
    python scripts/generate_catalog.py [--out TRANSFORM_CATALOG.md]
"""
from __future__ import annotations

import argparse
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path

FAMILY_TITLES = {
    "A": "A — Baseline and identity conditions",
    "B": "B — Temporal transformations",
    "C": "C — Resampling and multirate transformations",
    "D": "D — Spectral transformations",
    "E": "E — Noise and interference transformations",
    "F": "F — Psychoacoustic transformations",
    "G": "G — Adversarial and representation-aware transformations",
    "H": "H — Channel and deployment transformations",
}
FAMILY_PURPOSE = {
    "A": "Controls: level, clipping, bit-depth and identity paths that define "
         "the comparison floor for every axis.",
    "B": "Time-axis edits: rate, pitch, alignment, segmentation, prosody.",
    "C": "Sample-rate and bandwidth changes: narrowband, aliasing, codec-rate "
         "artifacts, drift.",
    "D": "Frequency-axis edits: masks, formants, phase, filters, warping.",
    "E": "Additive and interfering signals: stationary/modulated noise, "
         "competing speech, tonal and echo interference.",
    "F": "Perturbations constrained by the human auditory model (masking "
         "thresholds, Bark/ERB budgets, loudness) — the base paper's threat "
         "model, generalized.",
    "G": "Search/gradient-style attacks against a surrogate recognizer; "
         "evaluated black-box (cross-family transfer).",
    "H": "Deployment/channel effects: codecs, packet loss, rooms, devices, "
         "and the defenses induced by them.",
}


def _fmt_params(sig: inspect.Signature) -> str:
    items = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.default is inspect.Parameter.empty:
            items.append(f"`{name}` (required)")
        else:
            items.append(f"`{name}={p.default!r}`")
    return ", ".join(items) if items else "_(defaults only)_"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", type=Path, default=Path("TRANSFORM_CATALOG.md"))
    args = p.parse_args(argv)

    from audiocaptcha_dsp.transforms.registry import get_registry

    reg = get_registry()
    by_family: dict[str, list[str]] = {}
    for key in sorted(reg):
        by_family.setdefault(reg[key].family, []).append(key)

    lines: list[str] = []
    a = lines.append
    a("# Transform Catalog")
    a("")
    a("Generated from `transforms/registry.py` — **do not hand-edit**; "
      "regenerate with `python scripts/generate_catalog.py`.")
    a("")
    a(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
      f"— {len(reg)} transforms in {len(by_family)} families._")
    a("")
    a("Status labels (WHAT-REMAINS §15): every transform below is "
      "**Implemented** (registered, default-parameter execution covered by "
      "`tests/test_transforms/test_registry.py`); family-F budget behavior is "
      "additionally pinned by the psychoacoustics test suites. "
      "Anything not listed is *Not supported*.")
    a("")

    # ---- summary table -------------------------------------------------
    a("## Summary")
    a("")
    a("| Family | Transforms |")
    a("|:-:|---:|")
    for fam in sorted(by_family):
        a(f"| {FAMILY_TITLES.get(fam, fam)} | {len(by_family[fam])} |")
    a(f"| **Total** | **{len(reg)}** |")
    a("")

    # ---- per-family ----------------------------------------------------
    for fam in sorted(by_family):
        a(f"## {FAMILY_TITLES.get(fam, fam)}")
        a("")
        a(FAMILY_PURPOSE.get(fam, ""))
        a("")
        a("| Key | Class | Scientific basis / description | Default parameters |")
        a("|---|---|---|---|")
        for key in by_family[fam]:
            spec = reg[key]
            desc = (spec.description or "").strip().replace("\n", " ")
            refs = ", ".join(spec.references) if spec.references else ""
            basis = desc + (f"  \n*Refs:* {refs}" if refs else "")
            cls_name = spec.cls.__name__
            try:
                sig = inspect.signature(spec.cls.__init__)
                params = _fmt_params(sig)
            except (TypeError, ValueError):
                params = "_(unknown)_"
            a(f"| `{key}` | `{cls_name}` | {basis} | {params} |")
        a("")

    a("---")
    a("")
    a("### Notes")
    a("")
    a("- **Registry params vs constructor defaults:** the benchmark passes "
      "`TransformSpec.params` plus `seed` (when accepted) through "
      "`resolve_transform`; full constructor defaults shown above apply for "
      "any parameter not listed there.")
    a("- **Seeded reproducibility:** transforms whose `__init__` accepts "
      "`seed` receive the run seed (default 42) in benchmark conditions.")
    a("- **Composite chains:** `experiments/runner.py` composes registered "
      "transforms for multi-transform conditions.")
    a("- **λ-sweep targets:** family-F transforms accepting `margin_db` are "
      "sweep-capable; four are swept in the headline grid "
      "(`psychoacoustic.masked_noise`, `bark_perturbation`, "
      "`temporal_masking`, `signal_threshold`).")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(reg)} transforms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
