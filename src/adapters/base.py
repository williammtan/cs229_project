"""Adapter protocol — input-space *and* feature-space hooks.

An adapter can intervene at three loci. Each is independent; subclasses
override only what they need (all defaults are no-ops).

**Input-space (Trial -> Trial)** — runs before backbone.encode_trial:

    fit_source_trials(trials)     # source-pool fit; per-subject grouping via trial.subject
    calibrate_trials(trials)      # per-target-subject calibration set
    update_trial(trial)           # online streaming update (one new trial)
    transform_trial(trial)        # apply transformation; returns a new Trial

**Feature-space ((N_win, D) -> (N_win, D))** — runs after backbone.encode_trial:

    fit_source(feats, targets)
    calibrate(feats, targets)
    update(feat, target)
    transform(feats)

**Use cases**

* Riemannian alignment (RED) — input-space only.
* LoRA (PINK) — modifies backbone weights; uses ``fit_source`` to fit the
  per-subject adapter, ``transform`` is identity (it's applied via the
  modified backbone, not here).
* Convex NN (YELLOW) — feature-space only.

Compose multiple via :class:`AdapterStack`; trial-space transforms run in
order before encoding, feature-space transforms in order after.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.registry import register

if TYPE_CHECKING:
    from src.data.eegmmi import Trial


class AdapterBase:
    # ---- trial-space (raw EEG, pre-backbone) -------------------------------
    def fit_source_trials(self, trials: list["Trial"]) -> None:
        return None

    def calibrate_trials(self, trials: list["Trial"]) -> None:
        return None

    def update_trial(self, trial: "Trial") -> None:
        return None

    def transform_trial(self, trial: "Trial") -> "Trial":
        return trial

    def transform_calibration_trial(self, trial: "Trial") -> "Trial":
        """Apply the current transform to a calibration trial without advancing
        any online state. Stateless adapters can leave this as the default;
        online adapters (e.g. EMA-RA) override it to skip self-updates so the
        calibration set is not folded into the target estimate twice."""
        return self.transform_trial(trial)

    # ---- feature-space (post-backbone) -------------------------------------
    def fit_source(self, feats: np.ndarray, targets: np.ndarray) -> None:
        return None

    def calibrate(self, feats: np.ndarray, targets: np.ndarray) -> None:
        return None

    def update(self, feat: np.ndarray, target: np.ndarray) -> None:
        return None

    def transform(self, feats: np.ndarray) -> np.ndarray:
        return feats


@register("adapter", "none")
class NoopAdapter(AdapterBase):
    """Explicit no-op so configs can declare ``adapter=none`` for symmetry."""
