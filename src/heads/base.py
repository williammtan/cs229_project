"""Head protocol — per-trial features (N, D) -> per-trial class predictions.

For classification:
  ``fit(X, y)``         X: (N, D) float32, y: (N,) int64
  ``predict(X)``        -> (N,) int64 predicted labels
  ``predict_proba(X)``  -> (N, K) float32 class probabilities (optional)

Heads operate on **per-trial** features in this branch (one feature vector per
trial), not per-window features — the per-window aggregation lives inside the
backbone.
"""
from __future__ import annotations

import numpy as np


class HeadBase:
    def fit(self, feats: np.ndarray, targets: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, feats: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, feats: np.ndarray) -> np.ndarray | None:
        """Return (N, K) class probabilities, or ``None`` if not supported."""
        return None
