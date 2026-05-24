"""Training loop for neural-net regressors (EEGNet, ShallowConvNet).

Trials are split into fixed-length sliding windows, each window predicting the
mean velocity in that window. Predictions are then upsampled back to the
sample grid via nearest-neighbor for compatibility with the eval framework.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_windows(eeg: np.ndarray, vel: np.ndarray, win: int, hop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """eeg: (T, C). vel: (T, 3). Returns (n_windows, C, win) and (n_windows, 3) (mean vel in window) and window centers."""
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
        out[centers[-1] + hop // 2 :] = y_windows[-1]
    return out


@dataclass
class TrainConfig:
    win_samples: int = 100   # 1 s @ 100 Hz
    hop_samples: int = 20    # 200 ms
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 5
    val_frac: float = 0.1
    seed: int = 0


class NeuralRegressor:
    """sklearn-style wrapper around an nn.Module for sliding-window regression."""

    def __init__(self, model_factory: Callable[[int, int, int], nn.Module],
                 cfg: TrainConfig | None = None, n_channels: int = 32, n_out: int = 3):
        self.model_factory = model_factory
        self.cfg = cfg or TrainConfig()
        self.n_channels = n_channels
        self.n_out = n_out
        self.device = get_device()
        self.model: nn.Module | None = None

    def fit(self, eeg: np.ndarray, vel: np.ndarray, verbose: bool = False):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        X, y, _ = make_windows(eeg, vel, cfg.win_samples, cfg.hop_samples)
        if len(X) < 10:
            raise RuntimeError(f"Not enough windows to train: {len(X)}")
        # Target z-scoring (computed on training data, applied at eval time)
        self._y_mean = y.mean(axis=0)
        self._y_std = y.std(axis=0) + 1e-6
        y = (y - self._y_mean) / self._y_std
        n = len(X)
        idx = rng.permutation(n)
        n_val = max(1, int(n * cfg.val_frac))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        Xtr = torch.from_numpy(X[tr_idx]).unsqueeze(1)
        ytr = torch.from_numpy(y[tr_idx])
        Xva = torch.from_numpy(X[val_idx]).unsqueeze(1).to(self.device)
        yva = torch.from_numpy(y[val_idx]).to(self.device)
        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size,
                            shuffle=True, drop_last=False)

        self.model = self.model_factory(self.n_channels, cfg.win_samples, self.n_out).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        crit = nn.MSELoss()

        best_val = float("inf")
        bad = 0
        best_state = None
        for epoch in range(cfg.epochs):
            self.model.train()
            total = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                pred = self.model(xb)
                loss = crit(pred, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
                total += loss.item() * len(xb)
            train_loss = total / len(tr_idx)
            self.model.eval()
            with torch.no_grad():
                vpred = self.model(Xva)
                val_loss = crit(vpred, yva).item()
            if verbose:
                print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                bad = 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, eeg: np.ndarray) -> np.ndarray:
        assert self.model is not None, "fit first"
        cfg = self.cfg
        self.model.eval()
        T = eeg.shape[0]
        # Pad eeg so windows cover the tail
        # Build all windows
        Xs, _, centers = make_windows(eeg, np.zeros((T, self.n_out)), cfg.win_samples, cfg.hop_samples)
        if len(Xs) == 0:
            # Single-window fallback for short trials
            pad = np.zeros((cfg.win_samples, eeg.shape[1]), dtype=np.float32)
            pad[: min(T, cfg.win_samples)] = eeg[: cfg.win_samples]
            Xs = pad.T[None, ...]
            centers = np.array([T // 2])
        X = torch.from_numpy(Xs).unsqueeze(1).to(self.device)
        y_windows = self.model(X).cpu().numpy()
        # Un-z-score predictions back to original velocity units
        y_windows = y_windows * self._y_std + self._y_mean
        return windows_to_per_sample(y_windows, centers, T, cfg.hop_samples)
