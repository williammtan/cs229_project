"""Logger protocol + tag/group derivation.

The convention lives in *one place* (this file). WandbLogger reads from
``derive_run_meta(cfg)``; OfflineLogger does too. Don't bypass these helpers
when writing new logger backends — the whole point is enforcement.

W&B scoping convention (see /Users/williamtan/.claude/plans/elegant-wishing-rocket.md):

  project   = "etm"
  group     = "{protocol}__{backbone}__{adapter_stack}"  e.g. "loso__cbramod-frozen__ra-ema+lora-r4"
  job_type  = the nuisance axis swept inside the group: "loso-P{n}", "fold-{i}",
              "kmin-{k}_loso-P{n}", "latency", "drift"
  tags      = ["backbone:<name>", "adapter:<each-name>", "head:<name>",
              "dataset:<name>", "channels:<n>", "seed:<n>"]
  name      = "{group}__{job_type}__seed{seed}__{git_short_sha}"
"""
from __future__ import annotations

import subprocess
from typing import Any

from src.core.types import RunMeta


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        sha = out.stdout.strip()
        return sha or "nogit"
    except Exception:
        return "nogit"


def _adapter_stack_id(adapters_cfg: list[dict[str, Any]] | None) -> str:
    if not adapters_cfg:
        return "none"
    parts = []
    for a in adapters_cfg:
        name = a.get("name", "?")
        rank = a.get("rank")
        suffix = f"-r{rank}" if rank is not None else ""
        parts.append(f"{name}{suffix}")
    return "+".join(parts)


def _group_from_cfg(cfg: dict[str, Any]) -> str:
    protocol = cfg["protocol"]["name"]
    backbone = cfg["backbone"]["name"]
    adapter = _adapter_stack_id(cfg.get("adapters"))
    return f"{protocol}__{backbone}__{adapter}"


def _job_type_from_meta(protocol_name: str, meta: dict[str, Any] | None) -> str:
    meta = meta or {}
    if "k_minutes" in meta and "held_out_subject" in meta:
        return f"kmin-{meta['k_minutes']}_loso-P{meta['held_out_subject']}"
    if "held_out_subject" in meta:
        return f"loso-P{meta['held_out_subject']}"
    if "fold" in meta and "subject" in meta:
        return f"fold-{meta['fold']}_P{meta['subject']}"
    if "fold" in meta:
        return f"fold-{meta['fold']}"
    return protocol_name


def _tags_from_cfg(cfg: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    tags.append(f"backbone:{cfg['backbone']['name']}")
    for a in cfg.get("adapters") or []:
        tags.append(f"adapter:{a.get('name', '?')}")
    if cfg.get("head") and cfg["head"].get("name"):
        tags.append(f"head:{cfg['head']['name']}")
    if cfg.get("dataset") and cfg["dataset"].get("name"):
        tags.append(f"dataset:{cfg['dataset']['name']}")
    if cfg.get("dataset") and cfg["dataset"].get("channels"):
        tags.append(f"channels:{cfg['dataset']['channels']}")
    tags.append(f"seed:{cfg.get('seed', 0)}")
    return tags


def derive_run_meta(cfg: dict[str, Any], split_meta: dict[str, Any] | None = None) -> RunMeta:
    """Compute (group, job_type, tags, name) from a Hydra config + optional Split meta."""
    group = _group_from_cfg(cfg)
    job_type = _job_type_from_meta(cfg["protocol"]["name"], split_meta)
    tags = _tags_from_cfg(cfg)
    seed = cfg.get("seed", 0)
    sha = _git_short_sha()
    name = f"{group}__{job_type}__seed{seed}__{sha}"
    return RunMeta(group=group, job_type=job_type, tags=tags, name=name, config_dict=cfg)


class LoggerBase:
    """Interface every logger backend implements."""

    def init_run(self, run_meta: RunMeta) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        raise NotImplementedError

    def log_summary(self, summary: dict[str, float]) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError
