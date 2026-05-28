"""Identity head — passes features straight through.

Useful when a backbone emits per-trial class logits/labels directly as
"features"; rare at runtime (those backbones are monolithic) but registered
for config symmetry.
"""
from __future__ import annotations

import numpy as np

from src.core.registry import register
from src.heads.base import HeadBase


@register("head", "identity")
class IdentityHead(HeadBase):
    def fit(self, feats: np.ndarray, targets: np.ndarray) -> None:
        return None

    def predict(self, feats: np.ndarray) -> np.ndarray:
        return feats
