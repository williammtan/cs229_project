"""Sliding-window utilities used by every neural backbone.

Moved from src/train.py so adapters and FM backbones can reuse them without
pulling in the NeuralRegressor training loop.
"""
from __future__ import annotations

import numpy as np


def make_windows(
    eeg: np.ndarray, vel: np.ndarray, win: int, hop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """eeg: (T, C). vel: (T, 3). Returns (n_windows, C, win) and (n_windows, 3)
    (mean vel in window) and window centers."""
    T, C = eeg.shape
    Xs, ys, centers = [], [], []
    for start in range(0, T - win + 1, hop):
        end = start + win
        Xs.append(eeg[start:end].T)
        ys.append(vel[start:end].mean(axis=0))
        centers.append((start + end) // 2)
    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(centers, dtype=np.int64),
    )


def windows_to_per_sample(
    y_windows: np.ndarray, centers: np.ndarray, T: int, hop: int
) -> np.ndarray:
    """Map per-window predictions back to the per-sample grid by NN within ± hop/2."""
    out = np.zeros((T, y_windows.shape[1]), dtype=np.float32)
    if len(centers) == 0:
        return out
    for i, c in enumerate(centers):
        lo = max(0, c - hop // 2)
        hi = min(T, c + hop // 2)
        out[lo:hi] = y_windows[i]
    if centers[-1] + hop // 2 < T:
        out[centers[-1] + hop // 2:] = y_windows[-1]
    return out
