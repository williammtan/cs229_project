"""Composable inference pipeline: trial-space adapters -> Backbone -> feature-space adapters -> Head.

Two flow modes:

* **Monolithic** — ``head=None`` and no adapters. ``Pipeline.predict_trial``
  delegates to ``backbone.predict_trial``. The four classical baselines flow
  this way. Trial-space adapters can still run on the monolithic path
  (e.g., Riemannian alignment in front of BradberrymLR).

* **Composed** — at least one feature-space adapter or a non-None head.
  ``backbone.encode_trial`` produces per-window features + targets,
  feature-space adapters transform features in order, the head maps features
  to per-window predictions, and the backbone upsamples back to the per-sample
  grid. Frozen FM + (adapter) + linear-probe pipelines flow this way.

Trial-space adapters (RA-style) ALWAYS run, in both modes, before any
backbone work.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.adapters.base import AdapterBase
    from src.backbones.base import BackboneBase
    from src.data.way_eeg_gal import Trial
    from src.heads.base import HeadBase


class Pipeline:
    def __init__(
        self,
        backbone: "BackboneBase",
        adapters: list["AdapterBase"] | None = None,
        head: "HeadBase | None" = None,
    ):
        self.backbone = backbone
        self.adapters = list(adapters or [])
        self.head = head

    @property
    def _monolithic(self) -> bool:
        return self.head is None and not self.adapters

    # ---- source-only training -----------------------------------------------
    def fit(self, train_trials: list["Trial"]) -> None:
        # Trial-space adapters fit on source pool (per-subject groupings live
        # inside each adapter via trial.subject).
        for a in self.adapters:
            a.fit_source_trials(train_trials)
        train_trials = [self._apply_trial_transforms(t) for t in train_trials]

        self.backbone.fit_source(train_trials)
        if self.head is None and not any(self._has_feature_path(a) for a in self.adapters):
            return

        feats, targets = self._encode_concat(train_trials)
        for a in self.adapters:
            a.fit_source(feats, targets)
            feats = a.transform(feats)
        if self.head is not None:
            self.head.fit(feats, targets)

    # ---- per-subject calibration --------------------------------------------
    def calibrate(self, calib_trials: list["Trial"]) -> None:
        if not calib_trials:
            return
        # Trial-space calibration first (e.g., RA computes target-subject mean cov).
        for a in self.adapters:
            a.calibrate_trials(calib_trials)
        calib_trials = [self._apply_trial_transforms(t) for t in calib_trials]

        if self.head is None and not any(self._has_feature_path(a) for a in self.adapters):
            return

        feats, targets = self._encode_concat(calib_trials)
        for a in self.adapters:
            a.calibrate(feats, targets)
            feats = a.transform(feats)
        if self.head is not None and hasattr(self.head, "calibrate"):
            self.head.calibrate(feats, targets)

    # ---- prediction ---------------------------------------------------------
    def predict_trial(self, trial: "Trial") -> np.ndarray:
        trial = self._apply_trial_transforms(trial)
        if self._monolithic:
            return self.backbone.predict_trial(trial)
        if self.head is None:
            # No head but trial-space adapters present (e.g., RA + monolithic
            # finetuned FM). Backbone owns prediction.
            return self.backbone.predict_trial(trial)
        feats, _ = self.backbone.encode_trial(trial)
        for a in self.adapters:
            feats = a.transform(feats)
        y_win = self.head.predict(feats)
        return self.backbone.upsample_to_per_sample(y_win, trial)

    def predict_concat(self, trials: list["Trial"]) -> np.ndarray:
        return np.concatenate([self.predict_trial(t) for t in trials], axis=0)

    # ---- internal -----------------------------------------------------------
    def _apply_trial_transforms(self, trial: "Trial") -> "Trial":
        for a in self.adapters:
            trial = a.transform_trial(trial)
        return trial

    def _has_feature_path(self, adapter: "AdapterBase") -> bool:
        """True if the adapter overrides any feature-space hook."""
        cls = type(adapter)
        from src.adapters.base import AdapterBase

        return any(
            getattr(cls, name, None) is not getattr(AdapterBase, name)
            for name in ("fit_source", "calibrate", "update", "transform")
        )

    def _encode_concat(self, trials: list["Trial"]) -> tuple[np.ndarray, np.ndarray]:
        feats_list, target_list = [], []
        for t in trials:
            f, y = self.backbone.encode_trial(t)
            if len(f) == 0:
                continue
            feats_list.append(f)
            target_list.append(y)
        if not feats_list:
            raise RuntimeError("No windows produced across training trials.")
        return np.concatenate(feats_list, axis=0), np.concatenate(target_list, axis=0)
