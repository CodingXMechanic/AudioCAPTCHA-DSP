"""Registry integrity: completeness, catalog export, resolve_transform wiring."""
from __future__ import annotations

import numpy as np
import pytest

from audiocaptcha_dsp.core.signal import Signal
from audiocaptcha_dsp.transforms.base import BaseTransform
from audiocaptcha_dsp.transforms.registry import (
    FAMILIES,
    build_transform,
    catalog_rows,
    family_counts,
    get_registry,
    get_spec,
    list_transform_keys,
    register_transform,
)


@pytest.fixture(scope="module")
def registry():
    return get_registry()


@pytest.fixture
def signal() -> Signal:
    sr = 16000
    t = np.arange(int(0.5 * sr)) / sr
    wave = 0.5 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    return Signal(waveform=wave, sample_rate=sr, metadata={"transcript": "test"})


def test_registry_is_nonempty(registry):
    assert len(registry) >= 100


def test_every_family_has_coverage(registry):
    counts = family_counts()
    for family in FAMILIES:
        assert counts[family] > 0, f"taxonomy family {family} has no transforms"


def test_all_entries_are_base_transform_subclasses(registry):
    for key, spec in registry.items():
        assert isinstance(spec.cls, type)
        assert issubclass(spec.cls, BaseTransform), key


def test_all_entries_have_unique_keys_and_family(registry):
    keys = list(registry.keys())
    assert len(keys) == len(set(keys))
    for key, spec in registry.items():
        assert spec.family in FAMILIES, key
        assert spec.key == key
        assert spec.description, f"{key} missing description"


def test_catalog_rows_export(registry):
    rows = catalog_rows()
    assert len(rows) == len(registry)
    for row in rows:
        assert {"key", "family", "family_name", "class", "module",
                "description", "references", "default_params"} <= set(row)
    # Deterministic ordering
    assert [r["key"] for r in rows] == sorted(r["key"] for r in rows)


def test_build_transform_applies_params(registry):
    tr = build_transform("noise.white", {"snr_db": 7.0})
    assert tr.name == "noise.white" or type(tr).__name__ == "WhiteNoise"


def test_build_transform_filters_experiment_keys(registry):
    tr = build_transform("baseline.gain", {"gain_db": 3.0, "condition_index": 0,
                                           "sample_index": 2, "type": "baseline.gain"})
    assert tr.gain_db == 3.0


def test_unknown_key_raises(registry):
    with pytest.raises(KeyError):
        get_spec("does.not.exist")
    with pytest.raises(KeyError):
        build_transform("does.not.exist", {})


def test_list_transform_keys_by_family(registry):
    keys_b = list_transform_keys("B")
    assert keys_b
    for k in keys_b:
        assert registry[k].family == "B"


def test_register_transform_validation():
    with pytest.raises(ValueError):
        register_transform("bad.family", object, family="Z")  # unknown family
    # non-BaseTransform class rejected (valid family)
    with pytest.raises(TypeError):
        register_transform("registry.test_not_a_transform", dict, family="A")
    # duplicate key raises unless overwrite
    register_transform(
        "registry.test_probe", BaseTransform, family="A", overwrite=True
    )
    with pytest.raises(ValueError):
        register_transform("registry.test_probe", BaseTransform, family="A")


def test_runner_resolve_transform_reaches_new_families():
    """experiments.runner.resolve_transform must address all registry keys."""
    from audiocaptcha_dsp.experiments.runner import resolve_transform

    # A brand-new key that only exists in the central registry:
    tr = resolve_transform("multirate.narrowband", {})
    assert tr is not None
    tr2 = resolve_transform("channel.mp3", {"bitrate_kbps": 64})
    assert tr2.bitrate_kbps == 64
    # Legacy path still works
    tr3 = resolve_transform("noise.white", {"snr_db": 5.0})
    assert tr3.snr_db == 5.0


def test_runner_resolve_unknown_raises_with_hint():
    from audiocaptcha_dsp.experiments.runner import resolve_transform

    with pytest.raises(ValueError) as exc:
        resolve_transform("nope.nope", {})
    assert "nope.nope" in str(exc.value)


def test_registry_transforms_run_with_default_params(signal, registry):
    """Every registered transform must be constructible with its default params."""
    failures = []
    for key, spec in registry.items():
        try:
            out = spec.cls(**spec.params)(signal)
            assert len(np.asarray(out.waveform).ravel()) > 0
            assert np.all(np.isfinite(np.asarray(out.waveform)))
        except Exception as exc:  # noqa: BLE001 - collect and report
            failures.append((key, type(exc).__name__, str(exc)[:150]))
    assert not failures, f"transform failures: {failures}"


def test_every_transform_cites_scientific_basis(registry):
    """WHAT-REMAINS §15: every transform cites its scientific basis.

    The no-op reference condition (``baseline.identity``) is the one
    intentional exemption — a no-op has nothing to cite.
    """
    missing = [
        key for key, spec in registry.items()
        if key != "baseline.identity" and not spec.references
    ]
    assert not missing, f"transforms without scientific references: {missing}"


def test_legacy_runner_keys_live_in_central_registry(registry):
    """experiments.runner's legacy mirror must not hide transforms.

    The central registry is the single source of truth: any key the runner
    can resolve must also appear in the registry, catalog and
    ``--transforms all`` benchmark grids.
    """
    from audiocaptcha_dsp.experiments.runner import _TRANSFORM_REGISTRY

    hidden = [key for key in _TRANSFORM_REGISTRY if key not in registry]
    assert not hidden, f"runner-only transform keys not in registry: {hidden}"


def test_novel_transforms_registered_under_family_g(registry):
    """Original (project-novel) transforms are part of the taxonomy (§3G)."""
    for key in (
        "novel.phoneme_aware",
        "novel.phoneme_dropout",
        "novel.multi_domain",
        "novel.adaptive_formant",
        "novel.captcha_optimal",
        "novel.defense_robust",
    ):
        assert key in registry, key
        assert registry[key].family == "G"
