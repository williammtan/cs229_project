"""Stack multiple adapters: each subsequent adapter sees the previous one's output.

Plan.md calls for combinations like ``RA + LoRA`` (align in feature space first,
then per-subject LoRA fine-tune). AdapterStack is what those compose into.
"""
from __future__ import annotations

import numpy as np

from src.adapters.base import AdapterBase


class AdapterStack(AdapterBase):
    def __init__(self, adapters: list[AdapterBase]):
        self.adapters = adapters

    def fit_source(self, feats: np.ndarray, targets: np.ndarray) -> None:
        for a in self.adapters:
            a.fit_source(feats, targets)
            feats = a.transform(feats)

    def calibrate(self, feats: np.ndarray, targets: np.ndarray) -> None:
        for a in self.adapters:
            a.calibrate(feats, targets)
            feats = a.transform(feats)

    def update(self, feat: np.ndarray, target: np.ndarray) -> None:
        for a in self.adapters:
            a.update(feat, target)
            feat = a.transform(feat[None])[0] if feat.ndim == 1 else a.transform(feat)

    def transform(self, feats: np.ndarray) -> np.ndarray:
        for a in self.adapters:
            feats = a.transform(feats)
        return feats
