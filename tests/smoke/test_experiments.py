"""Smoke tests: each experiment config in src/configs/experiment/ composes,
builds a pipeline, runs on a tiny synthetic dataset, and produces finite
class-probability predictions.

FM-based configs (CBraMod / LaBraM / REVE) require multi-GB HuggingFace weights
plus a GPU to be fast — they're marked @pytest.mark.fm and skipped by default
(see pyproject.toml `addopts`). Run them explicitly with:

    uv run pytest -m fm tests/smoke/
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

# Side-effect imports to populate the registries (same as src.runner).
import src.adapters  # noqa: F401
import src.backbones  # noqa: F401
import src.heads  # noqa: F401
import src.protocols  # noqa: F401
from omegaconf import OmegaConf

from src.adapters.base import NoopAdapter
from src.core.pipeline import Pipeline
from src.core.registry import build_from_cfg
from tests.conftest import make_synth_dataset


REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "src" / "configs"
EXPERIMENT_DIR = CONFIG_DIR / "experiment"


FM_TOKENS = ("cbramod", "labram", "reve")


def _all_experiment_names() -> list[str]:
    return sorted(p.stem for p in EXPERIMENT_DIR.glob("*.yaml"))


def _is_fm(name: str) -> bool:
    return any(tok in name for tok in FM_TOKENS)


EXPERIMENT_PARAMS = [
    pytest.param(name, marks=[pytest.mark.fm] if _is_fm(name) else [])
    for name in _all_experiment_names()
]


def _build_pipeline(cfg_dict: dict[str, Any]) -> Pipeline:
    backbone = build_from_cfg("backbone", cfg_dict["backbone"])
    adapters = []
    for ac in cfg_dict.get("adapters") or []:
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
    if proto.get("name") in ("kmin_calibration", "ktrials_calibration"):
        proto["k_budgets_trials"] = [0]
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

    Pass criterion: finite (n_eval, K) probability matrix with rows that sum to ~1.
    """
    cfg_dict = _compose_experiment(name)
    n_classes = int(cfg_dict["dataset"].get("n_classes", 4))

    # 4 subjects × 24 trials × 4 classes: each of the 3 imagery sessions
    # gets 8 trials/subject = 2/class. Enough for LSO+LDA which needs
    # `train_size > n_classes`.
    data = make_synth_dataset(n_subjects=4, n_trials=24, n_classes=n_classes)

    protocol = build_from_cfg("protocol", cfg_dict["protocol"])
    splits = list(protocol.iter_splits(data))
    assert len(splits) > 0, f"protocol {cfg_dict['protocol']} produced no splits"

    split = splits[0]
    pipeline = _build_pipeline(cfg_dict)
    pipeline.fit(split.train)
    pipeline.calibrate(split.calib)
    proba = pipeline.predict_concat(split.eval)

    assert proba.shape == (len(split.eval), n_classes), \
        f"shape mismatch on {name}: got {proba.shape}"
    assert np.all(np.isfinite(proba)), f"{name}: non-finite values in proba"
    # Monolithic backbones return softmax probabilities, head path returns probabilities;
    # both should sum to ~1 per row.
    assert np.allclose(proba.sum(axis=-1), 1.0, atol=1e-3), \
        f"{name}: per-row probabilities should sum to 1"


def test_all_experiments_have_a_smoke_row():
    assert len(EXPERIMENT_PARAMS) > 0
    yaml_count = len(list(EXPERIMENT_DIR.glob("*.yaml")))
    assert len(EXPERIMENT_PARAMS) == yaml_count
