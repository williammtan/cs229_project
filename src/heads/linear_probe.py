"""Ridge linear probe on backbone features.

The canonical FM-head: closed-form, fast, no hyperparameter pain. Multi-output
ridge regression mapping (N_win, D) features to (N_win, 3) velocity windows.
StandardScaler keeps the ridge regularization scale-invariant across backbones.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.core.registry import register
from src.heads.base import HeadBase


@register("head", "linear_probe")
class LinearProbeHead(HeadBase):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None

    def fit(self, feats: np.ndarray, targets: np.ndarray) -> None:
        X = self.scaler.fit_transform(feats.astype(np.float32))
        # Target z-scoring lets ridge α stay in a stable range across datasets.
        self._y_mean = targets.mean(axis=0)
        self._y_std = targets.std(axis=0) + 1e-6
        y = (targets - self._y_mean) / self._y_std
        self.model.fit(X, y)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        assert self._y_mean is not None, "fit first"
        X = self.scaler.transform(feats.astype(np.float32))
        y = self.model.predict(X)
        return (y * self._y_std + self._y_mean).astype(np.float32)
