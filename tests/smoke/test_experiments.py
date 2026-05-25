"""Smoke tests: each experiment config in src/configs/experiment/ composes,
builds a pipeline, runs on a tiny synthetic dataset, and produces finite
metrics.

FM-based configs (CBraMod / LaBraM / REVE) require multi-GB HuggingFace weights
plus a GPU to be fast — they're marked @pytest.mark.fm and skipped by default
(see pyproject.toml `addopts`). Run them explicitly with:

    uv run pytest -m fm tests/smoke/

Each test invokes the same builders the runner uses (no subprocess, no hydra
@main), with the dataset loader monkeypatched to return the synthetic fixture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

# Side-effect imports to populate the registries (same as src.runner does).
import src.adapters  # noqa: F401
import src.backbones  # noqa: F401
import src.heads  # noqa: F401
import src.protocols  # noqa: F401
from omegaconf import OmegaConf

from src.adapters.base import NoopAdapter
from src.adapters.composite import AdapterStack
from src.core.pipeline import Pipeline
from src.core.registry import build_from_cfg
from src.data.way_eeg_gal import concat_trials
from tests.conftest import make_synth_dataset


REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "src" / "configs"
EXPERIMENT_DIR = CONFIG_DIR / "experiment"


# Experiments grouped by whether they need FM weights.
# Anything that mentions cbramod/labram/reve in its name needs the download +
# is heavy; we mark these `fm` so default `pytest` skips them.
FM_TOKENS = ("cbramod", "labram", "reve")


def _all_experiment_names() -> list[str]:
    return sorted(p.stem for p in EXPERIMENT_DIR.glob("*.yaml"))


def _is_fm(name: str) -> bool:
    return any(tok in name for tok in FM_TOKENS)


# Parametrize: every experiment name is its own test row.
EXPERIMENT_PARAMS = [
    pytest.param(name, marks=[pytest.mark.fm] if _is_fm(name) else [])
    for name in _all_experiment_names()
]


def _build_pipeline(cfg_dict: dict[str, Any]) -> Pipeline:
    backbone = build_from_cfg("backbone", cfg_dict["backbone"])
    adapter_cfgs = cfg_dict.get("adapters") or []
    adapters = []
    for ac in adapter_cfgs:
        if ac.get("name") == "none":
            adapters.append(NoopAdapter())
        else:
            adapters.append(build_from_cfg("adapter", ac))
    head_cfg = cfg_dict.get("head")
    head = build_from_cfg("head", head_cfg) if head_cfg else None
    return Pipeline(backbone, adapters, head)


def _shrink_for_smoke(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Aggressively shrink anything that would make the smoke take too long."""
    proto = cfg_dict.get("protocol", {})
    # KMin sweep: replace 6+ budgets with just (0.0,) so the per-fold cost is 1×, not 6×.
    if proto.get("name") == "kmin":
        proto["k_budgets_min"] = [0.0]
    # Trim train/finetune epochs for neural backbones if the field exists.
    bb = cfg_dict.get("backbone", {})
    for key in ("train", "finetune_train"):
        if isinstance(bb.get(key), dict):
            for ek in ("max_epochs", "epochs"):
                if ek in bb[key]:
                    bb[key][ek] = 1
    return cfg_dict


def _compose_experiment(name: str) -> dict[str, Any]:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(config_name="config", overrides=[f"+experiment={name}"])
    cfg_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return _shrink_for_smoke(cfg_dict)


@pytest.mark.parametrize("name", EXPERIMENT_PARAMS)
def test_experiment_runs_on_synthetic_data(name: str):
    """For each experiment config: build, fit, calibrate, predict on synthetic data.

    Pass criterion: the pipeline completes one full split end-to-end and emits
    finite predictions of the correct shape. We do NOT assert any accuracy
    metric — synthetic data has no signal — only that nothing explodes.
    """
    cfg_dict = _compose_experiment(name)

    # 2 subjects × 4 trials is the minimum for LOSO (need 1 held out + 1 train).
    # We use 3×4 so kmin sweeps have something to calibrate on too.
    data = make_synth_dataset(n_subjects=3, n_trials=4)

    protocol = build_from_cfg("protocol", cfg_dict["protocol"])
    splits = list(protocol.iter_splits(data))
    assert len(splits) > 0, f"protocol {cfg_dict['protocol']} produced no splits"

    # Run the cheapest single split end-to-end.
    split = splits[0]
    pipeline = _build_pipeline(cfg_dict)
    pipeline.fit(split.train)
    pipeline.calibrate(split.calib)
    preds = pipeline.predict_concat(split.eval)

    _, _, vel_true = concat_trials(split.eval)
    assert preds.shape == vel_true.shape, \
        f"shape mismatch on {name}: pred={preds.shape} truth={vel_true.shape}"
    assert np.all(np.isfinite(preds)), f"{name}: non-finite values in predictions"


def test_all_experiments_have_a_smoke_row():
    """Sanity: the parametrize list isn't empty (catches silent test-collection bugs)."""
    assert len(EXPERIMENT_PARAMS) > 0
    # And every yaml file in the directory should be represented.
    yaml_count = len(list(EXPERIMENT_DIR.glob("*.yaml")))
    assert len(EXPERIMENT_PARAMS) == yaml_count
