"""Shared scaffolding for foundation-model backbones.

All three target FMs (CBraMod, LaBraM, REVE) share substantial machinery:

* Sliding-window iteration over a trial at the model's native sampling rate
* 100→200 Hz upsample of WAY-EEG-GAL preprocessed input
* Optional channel subsetting (32 → 16 demo subset)
* Frozen / finetune toggle
* Per-window target = mean velocity in that window
* Per-window mean-pool of (C, S, D) backbone output → D-dim feature vector
* Upsampling per-window predictions back to the per-sample grid

Concrete subclasses implement only:

    _build_model() -> nn.Module        # loads pretrained weights
    _forward_features(x) -> Tensor     # (B, C, win_samples) -> (B, D) pooled features

That keeps wrapper LOC small and concentrates the model-specific surface
(input shape conventions, scaling factors, electrode-name handling) in one
method per FM.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from src.backbones.base import BackboneBase
from src.data.channels import (
    ACTICAP_32,
    channel_indices,
    get_channel_names,
)
from src.data.resample import resample_eeg
from src.data.windows import windows_to_per_sample

if TYPE_CHECKING:
    from src.data.way_eeg_gal import Trial


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FMBackboneBase(BackboneBase):
    """Composed-path backbone for pretrained FMs.

    Args:
      n_channels:    32 or 16. Drives the channel subset projection.
      target_fs:     resample target (200 for all three FMs).
      win_seconds:   sliding-window length at target_fs.
      hop_seconds:   sliding-window hop at target_fs.
      freeze:        if True, no parameters get gradient updates.
      batch_size:    inference batch size over windows.
    """

    embed_dim: int = 0  # subclass must set

    def __init__(
        self,
        n_channels: int = 32,
        target_fs: int = 200,
        win_seconds: float = 1.0,
        hop_seconds: float = 0.2,
        freeze: bool = True,
        batch_size: int = 32,
    ):
        self.n_channels = n_channels
        self.target_fs = target_fs
        self.win_samples = int(round(win_seconds * target_fs))
        self.hop_samples = int(round(hop_seconds * target_fs))
        self.freeze = freeze
        self.batch_size = batch_size

        self.channel_names = get_channel_names(n_channels)
        self._channel_indices = channel_indices(self.channel_names, ACTICAP_32)

        self.device = _device()
        self.model: nn.Module = self._build_model().to(self.device)
        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    # ---- subclass hooks ------------------------------------------------------

    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, win_samples) float32 at self.target_fs.
        Return (B, D) pooled per-window features."""
        raise NotImplementedError

    # ---- standard machinery --------------------------------------------------

    def fit_source(self, trials):
        # Frozen-mode: no training. Finetune-mode wiring lands in a follow-up.
        if not self.freeze:
            raise NotImplementedError("Finetune mode not wired yet; use freeze=True.")
        return None

    def _trial_at_target_fs(self, trial: "Trial") -> np.ndarray:
        """Return (n_channels, T_target) for this trial after channel subset and resample."""
        eeg = trial.eeg[self._channel_indices]  # (C, T_src)
        return resample_eeg(eeg, src_fs=trial.fs, dst_fs=self.target_fs)

    def _windowize(
        self, eeg_target: np.ndarray, vel_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """eeg_target: (C, T). vel_target: (3, T). Returns ((N_win, C, win), (N_win, 3), centers)."""
        C, T = eeg_target.shape
        Xs, ys, centers = [], [], []
        for start in range(0, T - self.win_samples + 1, self.hop_samples):
            end = start + self.win_samples
            Xs.append(eeg_target[:, start:end])
            ys.append(vel_target[:, start:end].mean(axis=-1))
            centers.append((start + end) // 2)
        if not Xs:
            return (
                np.zeros((0, C, self.win_samples), dtype=np.float32),
                np.zeros((0, vel_target.shape[0]), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return (
            np.stack(Xs).astype(np.float32),
            np.stack(ys).astype(np.float32),
            np.asarray(centers, dtype=np.int64),
        )

    def encode_trial(self, trial: "Trial") -> tuple[np.ndarray, np.ndarray]:
        eeg_t = self._trial_at_target_fs(trial)
        vel_t = resample_eeg(trial.vel, src_fs=trial.fs, dst_fs=self.target_fs)
        X, y, centers = self._windowize(eeg_t, vel_t)
        if len(X) == 0:
            return X.reshape(0, self.embed_dim), y
        # Stash centers for upsampling; one Trial at a time.
        self._last_centers = centers
        self._last_T_at_target_fs = eeg_t.shape[-1]

        # Forward in batches.
        feats_list: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i : i + self.batch_size]).to(self.device)
                fb = self._forward_features(xb)
                feats_list.append(fb.detach().to("cpu", dtype=torch.float32).numpy())
        feats = np.concatenate(feats_list, axis=0)
        return feats, y

    def upsample_to_per_sample(self, y_windows: np.ndarray, trial: "Trial") -> np.ndarray:
        # Re-derive centers at the original trial's grid: nearest-neighbor map
        # via the same hop, in the trial's native sample rate.
        # Centers are in target_fs units; rescale to trial.fs grid.
        if not hasattr(self, "_last_centers"):
            # No encode_trial has been called for this trial (shouldn't happen
            # in normal flow); fall back to recomputing centers.
            T_target = int(round(trial.eeg.shape[-1] * self.target_fs / trial.fs))
            centers_target = np.arange(self.win_samples // 2, T_target, self.hop_samples)
        else:
            centers_target = self._last_centers
        scale = trial.fs / self.target_fs
        centers_src = np.round(centers_target * scale).astype(np.int64)
        hop_src = max(1, int(round(self.hop_samples * scale)))
        T_src = trial.eeg.shape[-1]
        return windows_to_per_sample(y_windows, centers_src, T_src, hop_src)
