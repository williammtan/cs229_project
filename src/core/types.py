"""Shared dataclasses passed between protocols, pipelines, and loggers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.data.eegmmi import Trial


@dataclass
class Split:
    """One iteration produced by a Protocol.

    train:  trials used to fit the pipeline (source / pooled subjects)
    calib:  trials used for per-subject calibration (K-trials budget). May be empty.
    eval:   trials used for evaluation. Disjoint from train and calib.
    meta:   anything the logger should attach (held_out_subject, fold, k_trials, seed, ...).
    """

    train: list["Trial"]
    calib: list["Trial"]
    eval: list["Trial"]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMeta:
    """Identifying info for one run, derived from config once at startup."""

    group: str
    job_type: str
    tags: list[str]
    name: str
    config_dict: dict[str, Any]
