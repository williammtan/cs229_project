"""Composable inference pipeline: Backbone -> Adapter stack -> Head.

Two flow modes:

* **Monolithic** — ``head=None`` and no adapters. ``Pipeline.predict_trial``
  delegates to ``backbone.predict_trial``. The four current baselines flow
  this way.

* **Composed** — at least one adapter or a non-None head.
  ``backbone.encode_trial`` produces per-window features + targets, adapters
  transform features in order, the head maps features to per-window
  predictions, and the backbone upsamples back to the per-sample grid. Frozen
  FM + (adapter) + linear-probe pipelines flow this way.
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

    # ---- source-only training -------------------------------------------------
    def fit(self, train_trials: list["Trial"]) -> None:
        self.backbone.fit_source(train_trials)
        if self._monolithic:
            return
        feats, targets = self._encode_concat(train_trials)
        for a in self.adapters:
            a.fit_source(feats, targets)
            feats = a.transform(feats)
        if self.head is not None:
            self.head.fit(feats, targets)

    # ---- per-subject calibration (K-min) -------------------------------------
    def calibrate(self, calib_trials: list["Trial"]) -> None:
        if not calib_trials or self._monolithic:
            return
        feats, targets = self._encode_concat(calib_trials)
        for a in self.adapters:
            a.calibrate(feats, targets)
            feats = a.transform(feats)
        if self.head is not None and hasattr(self.head, "calibrate"):
            self.head.calibrate(feats, targets)

    # ---- prediction ----------------------------------------------------------
    def predict_trial(self, trial: "Trial") -> np.ndarray:
        if self._monolithic:
            return self.backbone.predict_trial(trial)
        feats, _ = self.backbone.encode_trial(trial)
        for a in self.adapters:
            feats = a.transform(feats)
        assert self.head is not None, "composed pipeline requires a head"
        y_win = self.head.predict(feats)
        return self.backbone.upsample_to_per_sample(y_win, trial)

    def predict_concat(self, trials: list["Trial"]) -> np.ndarray:
        return np.concatenate([self.predict_trial(t) for t in trials], axis=0)

    # ---- internal -----------------------------------------------------------
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
