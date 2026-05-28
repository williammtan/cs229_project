"""Composable inference pipeline for per-trial EEG classification.

Trial-space adapters (RA-style) → Backbone → feature-space adapters → Head.

Two flow modes:

* **Monolithic** — ``head=None`` and no feature-space adapters.
  ``Pipeline.predict_trial`` delegates to ``backbone.predict_trial``.
  EEGNet / ShallowConvNet / CSP+LDA / Riemannian+LR flow this way. Trial-space
  adapters (RA) can still run on the monolithic path in front of any backbone.

* **Composed** — at least one feature-space adapter or a non-None head.
  ``backbone.encode_trial`` returns one D-dim embedding per trial, feature-space
  adapters transform features in order, and the head maps features to per-trial
  class probabilities. Frozen FM + (adapter) + softmax-probe pipelines flow this
  way.

Trial-space adapters ALWAYS run, in both modes, before any backbone work.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.adapters.base import AdapterBase
    from src.backbones.base import BackboneBase
    from src.data.eegmmi import Trial
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
        return self.head is None and not any(
            self._has_feature_path(a) for a in self.adapters
        )

    # ---- source-only training -----------------------------------------------
    def fit(self, train_trials: list["Trial"]) -> None:
        for a in self.adapters:
            a.fit_source_trials(train_trials)
        train_trials = [self._apply_trial_transforms(t) for t in train_trials]

        self.backbone.fit_source(train_trials)
        if self._monolithic:
            return

        feats, labels = self._encode_concat(train_trials)
        for a in self.adapters:
            a.fit_source(feats, labels)
            feats = a.transform(feats)
        if self.head is not None:
            self.head.fit(feats, labels)

    # ---- per-subject calibration --------------------------------------------
    def calibrate(self, calib_trials: list["Trial"]) -> None:
        if not calib_trials:
            return
        for a in self.adapters:
            a.calibrate_trials(calib_trials)
        # Use the calibration-aware transform so online adapters (e.g. EMA-RA)
        # don't advance their state on trials that were already folded into the
        # calibration estimator above.
        calib_trials = [self._apply_calibration_transforms(t) for t in calib_trials]

        if self._monolithic:
            return

        feats, labels = self._encode_concat(calib_trials)
        for a in self.adapters:
            a.calibrate(feats, labels)
            feats = a.transform(feats)
        if self.head is not None and hasattr(self.head, "calibrate"):
            self.head.calibrate(feats, labels)

    # ---- prediction ---------------------------------------------------------
    def predict_trial(self, trial: "Trial") -> np.ndarray:
        """Return (n_classes,) class probabilities for one trial."""
        trial = self._apply_trial_transforms(trial)
        if self._monolithic:
            return np.asarray(self.backbone.predict_trial(trial), dtype=np.float32)
        feats = self.backbone.encode_trial(trial)  # (D,)
        feats = feats.reshape(1, -1)
        for a in self.adapters:
            feats = a.transform(feats)
        if self.head is None:
            # Adapters but no head: feature transform without classification is
            # meaningless on the classification branch — fall back to monolithic.
            return np.asarray(self.backbone.predict_trial(trial), dtype=np.float32)
        proba = self.head.predict_proba(feats)
        if proba is None:
            labels = self.head.predict(feats)
            n_classes = int(np.max(labels)) + 1
            one_hot = np.zeros((1, n_classes), dtype=np.float32)
            one_hot[0, int(labels[0])] = 1.0
            return one_hot[0]
        return np.asarray(proba[0], dtype=np.float32)

    def predict_concat(self, trials: list["Trial"]) -> np.ndarray:
        """Return (n_trials, n_classes) probabilities."""
        rows = [self.predict_trial(t) for t in trials]
        # Pad rows to a common K if monolithic backbones returned narrower probas
        # (shouldn't happen in practice, but defend against it).
        K = max(r.shape[0] for r in rows)
        out = np.zeros((len(rows), K), dtype=np.float32)
        for i, r in enumerate(rows):
            out[i, : r.shape[0]] = r
        return out

    # ---- internal -----------------------------------------------------------
    def _apply_trial_transforms(self, trial: "Trial") -> "Trial":
        for a in self.adapters:
            trial = a.transform_trial(trial)
        return trial

    def _apply_calibration_transforms(self, trial: "Trial") -> "Trial":
        for a in self.adapters:
            trial = a.transform_calibration_trial(trial)
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
        feats_list: list[np.ndarray] = []
        label_list: list[int] = []
        for t in trials:
            f = self.backbone.encode_trial(t)
            feats_list.append(np.asarray(f, dtype=np.float32).reshape(-1))
            label_list.append(int(t.label))
        if not feats_list:
            raise RuntimeError("No features produced across training trials.")
        return np.stack(feats_list, axis=0), np.asarray(label_list, dtype=np.int64)
