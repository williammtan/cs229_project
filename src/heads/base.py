"""Head protocol — features (N_win, D) -> per-window predictions (N_win, n_out).

The backbone owns windowing and per-sample upsampling. Heads are window-only.
"""
from __future__ import annotations

import numpy as np


class HeadBase:
    def fit(self, feats: np.ndarray, targets: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, feats: np.ndarray) -> np.ndarray:
        raise NotImplementedError
