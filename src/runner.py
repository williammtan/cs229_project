"""Hydra entry point for the EEGMMI classification pipeline.

One loop. Every experiment goes through here.

  uv run python -m src.runner +experiment=baseline_eegnet_loso
  uv run python -m src.runner -m +experiment=baseline_eegnet_loso seed=0,1,2
  uv run python -m src.runner +experiment=cbramod_frozen_linear_kmin \
      'protocol.k_budgets_trials=[0,1,5]' 'dataset.subjects=[1,2,3]'

For each ``Split`` produced by the protocol we run ``pipeline.fit → calibrate
→ predict`` and ship metrics to the logger with split-specific tagging.
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
from src.core.pipeline import Pipeline
from src.core.registry import build_from_cfg
from src.data.eegmmi import SubjectData, load_dataset
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
    if name != "eegmmi":
        raise NotImplementedError(f"dataset={name!r} not wired up yet.")
    return load_dataset(
        Path(raw_dir),
        subjects=dataset_cfg["subjects"],
        runs=dataset_cfg.get("runs"),
        dst_fs=dataset_cfg.get("target_fs", 100),
        tmin=float(dataset_cfg.get("tmin", 0.0)),
        tmax=float(dataset_cfg.get("tmax", 4.0)),
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
        print(f"  loaded subject S{sid:03d}: {len(sd.trials)} trials")
    if not data:
        print("  no data loaded; exiting.")
        return

    protocol = _build_protocol(cfg_dict["protocol"])
    logger = _build_logger(cfg_dict["logger"])

    n_classes = int(cfg_dict["dataset"].get("n_classes", 4))
    per_split_acc: list[float] = []
    per_split_kappa: list[float] = []

    splits = list(protocol.iter_splits(data))
    print(f"  protocol={protocol.name} → {len(splits)} splits")
    if not splits:
        print("  no splits produced; nothing to do.")
        return

    for i, split in enumerate(splits):
        run_meta = derive_run_meta(cfg_dict, split.meta)
        logger.init_run(run_meta)

        backbone = _build_backbone(cfg_dict["backbone"])
        adapters = _build_adapters(cfg_dict.get("adapters"))
        head = _build_head(cfg_dict.get("head"))
        pipeline = Pipeline(backbone, adapters, head)

        t0 = time.time()
        try:
            pipeline.fit(split.train)
            pipeline.calibrate(split.calib)
            y_proba = pipeline.predict_concat(split.eval)
            y_true = np.asarray([t.label for t in split.eval], dtype=np.int64)
            y_pred = y_proba.argmax(axis=1).astype(np.int64)

            bundle = summarize_evaluation(
                y_true, y_pred, y_proba=y_proba, n_classes=n_classes, seed=seed,
            )
            wall = time.time() - t0

            flat = flatten_for_logging(bundle)
            flat["wall_sec"] = wall
            if "k_trials_per_class" in split.meta:
                flat["calib/k_trials_per_class"] = float(split.meta["k_trials_per_class"])
            logger.log_metrics(flat)
            logger.log_summary({**flat, "split_meta": split.meta})

            acc = bundle["metrics"]["accuracy"]
            kappa = bundle["metrics"]["cohen_kappa"]
            per_split_acc.append(acc)
            per_split_kappa.append(kappa)
            print(
                f"  [{i+1}/{len(splits)}] {run_meta.job_type:>24}  "
                f"acc={acc:.3f}  kappa={kappa:+.3f}  ({wall:.1f}s)"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [{i+1}/{len(splits)}] {run_meta.job_type}: FAILED {type(e).__name__}: {e}")
            logger.log_summary({"error": f"{type(e).__name__}: {e}"})
        finally:
            logger.finish()

    if per_split_acc:
        print()
        print(f"  aggregate accuracy = {np.mean(per_split_acc):.3f} ± {np.std(per_split_acc):.3f}")
        print(f"  aggregate kappa    = {np.mean(per_split_kappa):+.3f} ± {np.std(per_split_kappa):.3f}")


if __name__ == "__main__":
    main()
