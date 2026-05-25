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


class _NaNTrainingError(RuntimeError):
    pass


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
        try:
            return self._fit_on_device(eeg, vel, verbose, self.device)
        except _NaNTrainingError as err:
            if self.device.type == "cpu":
                raise
            print(f"[NeuralRegressor.fit] {self.device} produced NaN ({err}); retraining on CPU")
            self.device = torch.device("cpu")
            return self._fit_on_device(eeg, vel, verbose, self.device)

    def _fit_on_device(self, eeg: np.ndarray, vel: np.ndarray, verbose: bool, device: torch.device):
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
        Xva = torch.from_numpy(X[val_idx]).unsqueeze(1).to(device)
        yva = torch.from_numpy(y[val_idx]).to(device)
        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size,
                            shuffle=True, drop_last=False)

        self.model = self.model_factory(self.n_channels, cfg.win_samples, self.n_out).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        crit = nn.MSELoss()

        best_val = float("inf")
        bad = 0
        best_state = None
        for epoch in range(cfg.epochs):
            self.model.train()
            total = 0.0
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = self.model(xb)
                loss = crit(pred, yb)
                if not torch.isfinite(loss):
                    raise _NaNTrainingError(f"non-finite loss at epoch {epoch}")
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
            if not np.isfinite(val_loss) or any(
                not torch.isfinite(t).all() for t in self.model.state_dict().values()
                if t.is_floating_point()
            ):
                raise _NaNTrainingError(f"non-finite val_loss or model state at epoch {epoch}")
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
        X_cpu = torch.from_numpy(Xs).unsqueeze(1)

        def _chunked_forward(model: nn.Module, x: torch.Tensor, device: torch.device, chunk: int = 1024) -> np.ndarray:
            outs = []
            for i in range(0, x.shape[0], chunk):
                xb = x[i : i + chunk].to(device, non_blocking=True)
                outs.append(model(xb).detach().cpu().numpy())
            return np.concatenate(outs, axis=0)

        y_windows = _chunked_forward(self.model, X_cpu, self.device)
        if np.isnan(y_windows).any():
            print("[NeuralRegressor.predict] MPS produced NaN; falling back to CPU")
            original_device = self.device
            cpu_device = torch.device("cpu")
            self.model.to(cpu_device)
            try:
                y_windows = _chunked_forward(self.model, X_cpu, cpu_device)
            finally:
                self.model.to(original_device)
        # Un-z-score predictions back to original velocity units
        y_windows = y_windows * self._y_std + self._y_mean
        return windows_to_per_sample(y_windows, centers, T, cfg.hop_samples)
