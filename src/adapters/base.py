"""Adapter protocol.

The four hooks correspond to the four moments at which an adapter might update
its state:

* ``fit_source(feats, targets)`` — source-only fit, with the labelled training
  pool. Riemannian alignment computes its source-domain reference here.
* ``calibrate(feats, targets)`` — per-subject K-min update. LoRA fits here.
* ``update(feat, target)`` — online streaming update of one window. RA-EMA
  hooks in here.
* ``transform(feats)`` — required; applied at every inference step.

Defaults are no-ops so subclasses only implement what they need.
"""
from __future__ import annotations

import numpy as np

from src.core.registry import register


class AdapterBase:
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
