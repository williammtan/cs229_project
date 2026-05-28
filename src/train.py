"""Training loop for neural classifiers (EEGNet / ShallowConvNet) on EEGMMI.

One label per trial. Each trial enters the model as a single (1, C, T) input
and gets a single class logit vector out. No sliding-window decoding.
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


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 8
    val_frac: float = 0.1
    seed: int = 0


class _NaNTrainingError(RuntimeError):
    pass


class NeuralClassifier:
    """sklearn-style wrapper around an nn.Module for per-trial classification.

    Input shapes:
      ``X`` : (N, C, T) float32 — one trial per row.
      ``y`` : (N,) int64 — class labels in [0, n_classes).
    """

    def __init__(
        self,
        model_factory: Callable[[int, int, int], nn.Module],
        cfg: TrainConfig | None = None,
        n_channels: int = 64,
        n_classes: int = 4,
    ):
        self.model_factory = model_factory
        self.cfg = cfg or TrainConfig()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = get_device()
        self.model: nn.Module | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        try:
            return self._fit_on_device(X, y, verbose, self.device)
        except _NaNTrainingError as err:
            if self.device.type == "cpu":
                raise
            print(f"[NeuralClassifier.fit] {self.device} produced NaN ({err}); retraining on CPU")
            self.device = torch.device("cpu")
            return self._fit_on_device(X, y, verbose, self.device)

    def _fit_on_device(self, X: np.ndarray, y: np.ndarray, verbose: bool, device: torch.device):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)

        if X.ndim != 3:
            raise ValueError(f"X must be (N, C, T); got shape {X.shape}")
        N, C, T = X.shape
        if N < 4:
            raise RuntimeError(f"Not enough trials to train: {N}")

        n_val = max(1, int(N * cfg.val_frac))
        idx = rng.permutation(N)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        Xtr = torch.from_numpy(X[tr_idx]).unsqueeze(1)  # (Ntr, 1, C, T)
        ytr = torch.from_numpy(y[tr_idx].astype(np.int64))
        Xva = torch.from_numpy(X[val_idx]).unsqueeze(1).to(device)
        yva = torch.from_numpy(y[val_idx].astype(np.int64)).to(device)
        loader = DataLoader(
            TensorDataset(Xtr, ytr), batch_size=cfg.batch_size,
            shuffle=True, drop_last=False,
        )

        self.model = self.model_factory(C, T, self.n_classes).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        crit = nn.CrossEntropyLoss()

        best_val = float("inf")
        bad = 0
        best_state = None
        for epoch in range(cfg.epochs):
            self.model.train()
            running = 0.0
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = self.model(xb)
                loss = crit(logits, yb)
                if not torch.isfinite(loss):
                    raise _NaNTrainingError(f"non-finite loss at epoch {epoch}")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
                running += loss.item() * len(xb)
            train_loss = running / len(tr_idx)
            self.model.eval()
            with torch.no_grad():
                vlogits = self.model(Xva)
                val_loss = crit(vlogits, yva).item()
                val_acc = (vlogits.argmax(1) == yva).float().mean().item()
            if verbose:
                print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  val_acc {val_acc:.3f}")
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
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, n_classes) softmax probabilities."""
        assert self.model is not None, "fit first"
        self.model.eval()
        if X.ndim == 2:  # single trial (C, T)
            X = X[None, ...]
        Xt = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)  # (N, 1, C, T)

        def _chunked_forward(model, x, device, chunk=256):
            outs = []
            for i in range(0, x.shape[0], chunk):
                xb = x[i : i + chunk].to(device, non_blocking=True)
                logits = model(xb)
                outs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
            return np.concatenate(outs, axis=0)

        proba = _chunked_forward(self.model, Xt, self.device)
        if np.isnan(proba).any():
            print("[NeuralClassifier.predict_proba] MPS produced NaN; falling back to CPU")
            original = self.device
            cpu = torch.device("cpu")
            self.model.to(cpu)
            try:
                proba = _chunked_forward(self.model, Xt, cpu)
            finally:
                self.model.to(original)
        return proba.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=-1).astype(np.int64)
