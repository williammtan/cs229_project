"""Hydra entry point. One loop. Every experiment goes through here.

  uv run python -m src.runner +experiment=baseline_eegnet_loso
  uv run python -m src.runner -m +experiment=baseline_eegnet_loso seed=0,1,2
  uv run python -m src.runner +experiment=cbramod_lora_kmin \\
      protocol.k_budgets_min=[0,1,5] dataset.subjects=[1,2,3]

The loop is deliberately tiny: build pipeline + protocol + logger, then for
each ``Split`` produced by the protocol run ``pipeline.fit → calibrate →
predict`` and ship metrics to the logger with split-specific tagging.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Side-effect imports populate the registries.
import src.backbones  # noqa: F401
import src.adapters  # noqa: F401
import src.heads  # noqa: F401
import src.protocols  # noqa: F401

from src.adapters.base import AdapterBase, NoopAdapter
from src.adapters.composite import AdapterStack
from src.core.pipeline import Pipeline
from src.core.registry import build_from_cfg
from src.data.way_eeg_gal import SubjectData, concat_trials, load_dataset
from src.eval.metrics import flatten_for_logging, summarize_evaluation
from src.loggers.base import LoggerBase, derive_run_meta
from src.loggers.offline import OfflineLogger
from src.loggers.wandb import WandbLogger


# ---- builders ---------------------------------------------------------------

def _build_backbone(cfg: dict[str, Any]):
    return build_from_cfg("backbone", cfg)


def _build_adapters(adapter_cfgs: list[dict[str, Any]] | None) -> list[AdapterBase]:
    if not adapter_cfgs:
        return []
    adapters: list[AdapterBase] = []
    for ac in adapter_cfgs:
        if ac.get("name") == "none":
            adapters.append(NoopAdapter())
        else:
            adapters.append(build_from_cfg("adapter", ac))
    return adapters


def _build_head(head_cfg: dict[str, Any] | None):
    if head_cfg is None:
        return None
    return build_from_cfg("head", head_cfg)


def _build_protocol(cfg: dict[str, Any]):
    return build_from_cfg("protocol", cfg)


def _build_logger(cfg: dict[str, Any]) -> LoggerBase:
    name = cfg.get("name", "offline")
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    if name == "wandb":
        return WandbLogger(**kwargs)
    if name == "offline":
        return OfflineLogger(**kwargs)
    raise KeyError(f"Unknown logger: {name}")


def _load_data(raw_dir: str, dataset_cfg: dict[str, Any]) -> dict[int, SubjectData]:
    name = dataset_cfg["name"]
    if name != "way_eeg_gal":
        raise NotImplementedError(f"dataset={name!r} not wired up yet.")
    return load_dataset(
        Path(raw_dir),
        subjects=dataset_cfg["subjects"],
        series=dataset_cfg["series"],
        dst_fs=dataset_cfg.get("target_fs", 100),
        per_trial_zscore=dataset_cfg.get("per_trial_zscore", True),
    )


# ---- main loop --------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    seed = int(cfg_dict.get("seed", 0))
    np.random.seed(seed)

    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)

    data = _load_data(cfg_dict["raw_dir"], cfg_dict["dataset"])
    for sid, sd in data.items():
        print(f"  loaded subject P{sid}: {len(sd.trials)} trials")

    protocol = _build_protocol(cfg_dict["protocol"])

    logger = _build_logger(cfg_dict["logger"])

    # Per-group summary aggregation across splits — populates run.summary so
    # the headline table can be assembled from W&B without re-running anything.
    per_split_r: list[float] = []
    per_split_r2: list[float] = []
    per_split_rmse: list[float] = []

    splits = list(protocol.iter_splits(data))
    print(f"  protocol={protocol.name} → {len(splits)} splits")
    if not splits:
        print("  no splits produced; nothing to do.")
        return

    # One W&B run per split (so per-fold rows show up cleanly). Group/job-type
    # are derived per-split; all runs in the same call share the same group.
    for i, split in enumerate(splits):
        run_meta = derive_run_meta(cfg_dict, split.meta)
        logger.init_run(run_meta)

        # Build a fresh pipeline per split so state never leaks.
        backbone = _build_backbone(cfg_dict["backbone"])
        adapters = _build_adapters(cfg_dict.get("adapters"))
        head = _build_head(cfg_dict.get("head"))
        pipeline = Pipeline(backbone, adapters, head)

        t0 = time.time()
        try:
            pipeline.fit(split.train)
            pipeline.calibrate(split.calib)
            preds = pipeline.predict_concat(split.eval)
            _, _, vel_true = concat_trials(split.eval)
            assert preds.shape == vel_true.shape, f"{preds.shape} != {vel_true.shape}"
            bundle = summarize_evaluation(vel_true, preds, seed=seed)
            wall = time.time() - t0

            flat = flatten_for_logging(bundle)
            flat["wall_sec"] = wall
            # K-min curves: log k_minutes as a step metric so eval/* plots vs K.
            if "k_minutes" in split.meta:
                flat["calib/k_minutes"] = float(split.meta["k_minutes"])
            logger.log_metrics(flat)
            logger.log_summary({
                **flat,
                "split_meta": split.meta,
            })

            r = bundle["metrics"]["pearson_r_mean"]
            r2 = bundle["metrics"]["r2_mean"]
            rmse = bundle["metrics"]["rmse_mean"]
            per_split_r.append(r)
            per_split_r2.append(r2)
            per_split_rmse.append(rmse)
            print(
                f"  [{i+1}/{len(splits)}] {run_meta.job_type:>24}  "
                f"r_mean={r:+.3f}  r2_mean={r2:+.3f}  rmse_mean={rmse:.3f}  ({wall:.1f}s)"
            )
        except Exception as e:
            print(f"  [{i+1}/{len(splits)}] {run_meta.job_type}: FAILED {type(e).__name__}: {e}")
            logger.log_summary({"error": f"{type(e).__name__}: {e}"})
        finally:
            logger.finish()

    if per_split_r:
        print()
        print(f"  aggregate r_mean    = {np.mean(per_split_r):+.3f} ± {np.std(per_split_r):.3f}")
        print(f"  aggregate r2_mean   = {np.mean(per_split_r2):+.3f} ± {np.std(per_split_r2):.3f}")
        print(f"  aggregate rmse_mean = {np.mean(per_split_rmse):.3f} ± {np.std(per_split_rmse):.3f}")


if __name__ == "__main__":
    main()
