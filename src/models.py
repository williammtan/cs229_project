"""Baseline decoders for continuous EEG kinematic regression.

Four models, in order of capacity:
  1. BradberrymLR : linear regression on lag-augmented low-frequency potentials.
                    Replicates Bradberry et al. 2010 (J. Neurosci.).
  2. RidgeBandPower: ridge regression on per-channel mu/beta band-power features
                    in sliding windows. Replicates the Korik et al. 2018 BTS style.
  3. EEGNetReg    : EEGNet (Lawhern et al. 2018) with a linear regression head.
  4. ShallowConvNetReg: ShallowConvNet (Schirrmeister et al. 2017) with linear head.

All models share an sklearn-like fit(X, y)/predict(X) interface where:
  - For linear models, X has shape (N_samples, n_features) and y has shape (N_samples, 3).
  - For neural models, X is a list of trial arrays (32, T_i) and y is a list of
    (3, T_i) arrays; the wrapper converts to sliding windows internally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ utilities

def _hilbert_envelope_pow(x: np.ndarray) -> np.ndarray:
    """Mean squared amplitude in window — proxy for band power. x: (..., T)."""
    return (x ** 2).mean(axis=-1)


def make_lagged_features(
    eeg: np.ndarray, lags: Sequence[int]
) -> np.ndarray:
    """Bradberry-style lag-augmented features.
    eeg: (T, C). Output: (T, C * len(lags)). Negative lags are PAST EEG (causal).
    """
    T, C = eeg.shape
    cols = []
    for lag in lags:
        shifted = np.zeros_like(eeg)
        if lag >= 0:
            shifted[lag:] = eeg[: T - lag]
        else:
            shifted[: T + lag] = eeg[-lag:]
        cols.append(shifted)
    return np.concatenate(cols, axis=1)


def make_band_power_features(
    eeg: np.ndarray, fs: int, bands: Sequence[tuple[float, float]],
    win_samples: int, hop_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window log-band-power features.
    eeg: (T, C). Returns (n_windows, C * n_bands) features and window center indices.
    """
    T, C = eeg.shape
    n_bands = len(bands)
    feats = []
    centers = []
    band_filtered = []
    for lo, hi in bands:
        sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
        band_filtered.append(sosfiltfilt(sos, eeg, axis=0))
    for start in range(0, T - win_samples + 1, hop_samples):
        end = start + win_samples
        row = []
        for bf in band_filtered:
            row.append(np.log(np.mean(bf[start:end] ** 2, axis=0) + 1e-8))
        feats.append(np.concatenate(row))
        centers.append((start + end) // 2)
    return np.asarray(feats, dtype=np.float32), np.asarray(centers, dtype=np.int64)


# ------------------------------------------------------------------ linear

class BradberrymLR:
    """Multivariate linear regression on lag-augmented low-frequency EEG."""

    def __init__(self, lags: Sequence[int] = (0, 5, 10, 15, 20, 25),
                 alpha: float = 1.0):
        self.lags = list(lags)
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()

    def fit(self, eeg: np.ndarray, vel: np.ndarray):
        X = make_lagged_features(eeg, self.lags)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, vel)
        return self

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        X = make_lagged_features(eeg, self.lags)
        X = self.scaler.transform(X)
        return self.model.predict(X)


class RidgeBandPower:
    """Ridge regression on log-band-power features in sliding windows.
    Targets are velocity averaged within the same window. Predictions are upsampled
    back to the original time grid via nearest-neighbor for compatibility with the
    per-sample evaluation pipeline."""

    def __init__(self, fs: int = 100,
                 bands: Sequence[tuple[float, float]] = ((8, 13), (13, 30), (30, 45)),
                 win_ms: int = 500, hop_ms: int = 100,
                 alpha: float = 10.0):
        self.fs = fs
        self.bands = bands
        self.win_samples = int(fs * win_ms / 1000)
        self.hop_samples = int(fs * hop_ms / 1000)
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()

    def _windows(self, eeg: np.ndarray, vel: np.ndarray):
        feats, centers = make_band_power_features(
            eeg, self.fs, self.bands, self.win_samples, self.hop_samples
        )
        # window-averaged target
        targets = []
        for c in centers:
            lo = max(0, c - self.win_samples // 2)
            hi = min(vel.shape[0], c + self.win_samples // 2)
            targets.append(vel[lo:hi].mean(axis=0))
        return feats, np.asarray(targets, dtype=np.float32), centers

    def fit(self, eeg: np.ndarray, vel: np.ndarray):
        X, y, _ = self._windows(eeg, vel)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        T = eeg.shape[0]
        feats, centers = make_band_power_features(
            eeg, self.fs, self.bands, self.win_samples, self.hop_samples
        )
        X = self.scaler.transform(feats)
        y_windows = self.model.predict(X)  # (n_windows, 3)
        # nearest-neighbor upsample to per-sample grid
        out = np.zeros((T, y_windows.shape[1]), dtype=np.float32)
        if len(centers) == 0:
            return out
        for i, c in enumerate(centers):
            lo = max(0, c - self.hop_samples // 2)
            hi = min(T, c + self.hop_samples // 2)
            out[lo:hi] = y_windows[i]
        # fill any remaining tail with last window
        if centers[-1] + self.hop_samples // 2 < T:
            out[centers[-1] + self.hop_samples // 2 :] = y_windows[-1]
        return out


# ------------------------------------------------------------------ neural nets


class EEGNetReg(nn.Module):
    """EEGNet (Lawhern 2018) with linear regression head.
    Input: (B, 1, C, T). Output: (B, n_out)."""

    def __init__(self, n_channels: int = 32, n_samples: int = 100, n_out: int = 3,
                 f1: int = 8, d: int = 2, f2: int = 16, kernel_len: int = 50,
                 dropout: float = 0.25):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_len), padding=(0, kernel_len // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        # figure out output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.separable(self.depthwise(self.firstconv(dummy)))
            self.flat_dim = out.numel()
        self.head = nn.Linear(self.flat_dim, n_out)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.flatten(start_dim=1)
        return self.head(x)


class ShallowConvNetReg(nn.Module):
    """ShallowConvNet (Schirrmeister 2017) with linear regression head.
    Mirrors FBCSP: temporal conv -> spatial conv -> square -> avg pool -> log -> linear.
    Input: (B, 1, C, T). Output: (B, n_out)."""

    def __init__(self, n_channels: int = 32, n_samples: int = 100, n_out: int = 3,
                 n_filters: int = 40, temporal_kernel: int = 25,
                 pool_size: int = 75, pool_stride: int = 15,
                 dropout: float = 0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, (1, temporal_kernel), bias=False)
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            h = self.temporal(dummy)
            h = self.spatial(h)
            h = self.bn(h)
            h = h * h  # square
            h = self.pool(h)
            h = torch.log(torch.clamp(h, min=1e-6))
            self.flat_dim = h.numel()
        self.head = nn.Linear(self.flat_dim, n_out)

    def forward(self, x):
        h = self.temporal(x)
        h = self.spatial(h)
        h = self.bn(h)
        h = h * h
        h = self.pool(h)
        h = torch.log(torch.clamp(h, min=1e-6))
        h = self.dropout(h)
        h = h.flatten(start_dim=1)
        return self.head(h)
