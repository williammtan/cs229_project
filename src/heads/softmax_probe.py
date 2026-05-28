"""Softmax linear classifier on backbone embeddings.

The canonical FM-head for classification: closed-form-ish (L-BFGS), fast, no
hyperparameter pain. Multinomial logistic regression on (N_trials, D) features
producing (N_trials,) integer class labels and (N_trials, K) probabilities.

StandardScaler keeps the L2 regularization scale-invariant across backbones.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.core.registry import register
from src.heads.base import HeadBase


@register("head", "softmax_probe")
@register("head", "linear_probe")  # alias for config compat
class SoftmaxProbeHead(HeadBase):
    def __init__(self, C: float = 1.0, max_iter: int = 1000, calibrate_on_calib: bool = True):
        self.C = C
        self.max_iter = max_iter
        self.calibrate_on_calib = calibrate_on_calib
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        self._fitted = False
        self._source_feats: np.ndarray | None = None
        self._source_targets: np.ndarray | None = None

    def _fit_impl(
        self,
        feats: np.ndarray,
        targets: np.ndarray,
        *,
        remember_source: bool,
    ) -> None:
        X = self.scaler.fit_transform(feats.astype(np.float32))
        y = targets.astype(np.int64)
        self.model.fit(X, y)
        self._fitted = True
        if remember_source:
            self._source_feats = feats.astype(np.float32, copy=True)
            self._source_targets = y.copy()

    def fit(self, feats: np.ndarray, targets: np.ndarray) -> None:
        self._fit_impl(feats, targets, remember_source=True)

    def calibrate(self, feats: np.ndarray, targets: np.ndarray) -> None:
        """Per-subject calibration: refit on source + calibration data.

        With ``calibrate_on_calib=False`` this is a no-op and keeps the
        source-only weights. The calibrated fit uses a fresh scaler/classifier
        over the combined supervised data, so target calibration can influence
        the decision boundary without discarding the source pool.
        """
        if not self.calibrate_on_calib or len(targets) < 2:
            return
        if self._source_feats is None or self._source_targets is None:
            if len(np.unique(targets)) < 2:
                return
            self.fit(feats, targets)
            return
        combined_feats = np.concatenate(
            [self._source_feats, feats.astype(np.float32)], axis=0
        )
        combined_targets = np.concatenate(
            [self._source_targets, targets.astype(np.int64)], axis=0
        )
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        self._fit_impl(combined_feats, combined_targets, remember_source=False)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        assert self._fitted, "fit first"
        X = self.scaler.transform(feats.astype(np.float32))
        return self.model.predict(X).astype(np.int64)

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        assert self._fitted, "fit first"
        X = self.scaler.transform(feats.astype(np.float32))
        return self.model.predict_proba(X).astype(np.float32)
