"""Shared scaffolding for foundation-model backbones.

All three target FMs (CBraMod, LaBraM, REVE) share substantial machinery:

* Sliding-window iteration over a trial at the model's native sampling rate
* 100→200 Hz upsample of WAY-EEG-GAL preprocessed input
* Optional channel subsetting (32 → 16 demo subset)
* Frozen (linear-probe path) or full finetune (end-to-end training) toggle
* Per-window target = mean velocity in that window
* Per-window mean-pool of (C, S, D) backbone output → D-dim feature vector
* Upsampling per-window predictions back to the per-sample grid

Concrete subclasses implement only:

    _build_model() -> nn.Module        # loads pretrained weights
    _forward_features(x) -> Tensor     # (B, C, win_samples) -> (B, D) pooled features
    _forward_predict(x)  -> Tensor     # (B, C, win_samples) -> (B, n_out) for finetune
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.backbones.base import BackboneBase
from src.backbones.lora import LoRALinear, count_trainable, inject_lora
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


# Default finetune hyperparameters — used when finetune_train is None.
#
# LP-warmup + LoRA defaults are OFF here so the legacy finetune path is
# unchanged. Experiment configs that want the REVE-paper recipe (LP→FT + LoRA)
# set ``lp_warmup_epochs``, ``lora_rank``, and ``lora_target_patterns``.
_DEFAULT_FT_TRAIN = dict(
    lr=5.0e-5,
    weight_decay=1.0e-4,
    epochs=10,
    patience=3,
    batch_size=32,
    val_frac=0.1,
    seed=0,
    grad_clip=1.0,
    max_train_windows=None,  # cap total training windows; None = no cap
    # ---- REVE-paper LP→FT + LoRA recipe (Kumar et al. 2022; REVE §3.3) ----
    # epochs of phase 1, head-only with encoder frozen. 0 disables LP warmup.
    lp_warmup_epochs=0,
    # learning rate during phase 1; can differ from main `lr`.
    lp_lr=None,  # None -> use `lr`
    # LoRA rank for phase 2. 0 = full unfreeze (legacy behavior).
    lora_rank=0,
    lora_alpha=16.0,
    # qualified-name patterns to inject LoRA into. Matched fnmatch-style; bare
    # names (e.g. "to_qkv") match any module ending with that name.
    lora_target_patterns=("to_qkv", "to_out"),
)


class FMBackboneBase(BackboneBase):
    """Composed-path backbone for pretrained FMs.

    Args:
      n_channels:    32 or 16. Drives the channel subset projection.
      target_fs:     resample target (200 for all three FMs).
      win_seconds:   sliding-window length at target_fs.
      hop_seconds:   sliding-window hop at target_fs.
      freeze:        if True, no parameters get gradient updates. If False, the
                     backbone's built-in regression head is trained end-to-end
                     via ``fit_source`` and ``predict_trial`` flows through the
                     trained model (monolithic path).
      batch_size:    inference batch size over windows.
      finetune_train: dict of training hyperparams (used only when freeze=False).
                     See ``_DEFAULT_FT_TRAIN`` for defaults.
    """

    embed_dim: int = 0  # subclass must set
    n_out: int = 3

    def __init__(
        self,
        n_channels: int = 32,
        target_fs: int = 200,
        win_seconds: float = 1.0,
        hop_seconds: float = 0.2,
        freeze: bool = True,
        batch_size: int = 32,
        finetune_train: dict | None = None,
    ):
        self.n_channels = n_channels
        self.target_fs = target_fs
        self.win_samples = int(round(win_seconds * target_fs))
        self.hop_samples = int(round(hop_seconds * target_fs))
        self.freeze = freeze
        self.batch_size = batch_size

        self.train_cfg = {**_DEFAULT_FT_TRAIN, **(finetune_train or {})}

        self.channel_names = get_channel_names(n_channels)
        self._channel_indices = channel_indices(self.channel_names, ACTICAP_32)

        self.device = _device()
        self.model: nn.Module = self._build_model().to(self.device)
        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

        # Set by fit_source / encode_trial; used by predict / upsample.
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None
        self._last_centers: np.ndarray | None = None

    # ---- subclass hooks ------------------------------------------------------

    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, win_samples). Return (B, D) pooled per-window features."""
        raise NotImplementedError

    def _forward_predict(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, win_samples). Return (B, n_out) predictions (uses model's
        built-in regression head). Default implementation calls ``model(x)``;
        subclasses with non-trivial forward signatures override."""
        return self.model(x)

    # ---- standard machinery --------------------------------------------------

    def _trial_at_target_fs(self, trial: "Trial") -> np.ndarray:
        eeg = trial.eeg[self._channel_indices]
        return resample_eeg(eeg, src_fs=trial.fs, dst_fs=self.target_fs)

    def _windowize(
        self, eeg_target: np.ndarray, vel_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _concat_windows(
        self, trials: list["Trial"]
    ) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for t in trials:
            eeg_t = self._trial_at_target_fs(t)
            vel_t = resample_eeg(t.vel, src_fs=t.fs, dst_fs=self.target_fs)
            X, y, _ = self._windowize(eeg_t, vel_t)
            if len(X):
                X_list.append(X)
                y_list.append(y)
        if not X_list:
            raise RuntimeError("No windows produced across trials.")
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    # ---- fit / encode / predict ---------------------------------------------

    def fit_source(self, trials):
        if self.freeze:
            return None
        self._finetune(trials)

    # ---- head detection (override in subclass if needed) ---------------------

    def _head_module(self) -> nn.Module:
        """The regression head trained during the LP-warmup phase.

        Default: ``self.model.final_layer`` (REVE/CBraMod). LaBraM keeps its
        head elsewhere; override in those subclasses if you enable LP warmup.
        """
        if hasattr(self.model, "final_layer"):
            return self.model.final_layer
        raise AttributeError(
            f"{type(self).__name__}: no `final_layer` found; override `_head_module()` "
            f"to point at the regression head before enabling LP warmup."
        )

    # ---- training-mode helpers ----------------------------------------------

    def _freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _enable_head_only(self) -> None:
        """Phase 1 (LP warmup): encoder frozen, only the regression head trains."""
        self._freeze_all()
        for p in self._head_module().parameters():
            p.requires_grad_(True)

    def _enable_full_or_lora(self, lora_rank: int) -> None:
        """Phase 2 (FT): either unfreeze everything (legacy) or train LoRA + head."""
        if lora_rank > 0:
            self._freeze_all()
            for p in self._head_module().parameters():
                p.requires_grad_(True)
            for m in self.model.modules():
                if isinstance(m, LoRALinear):
                    m.lora_A.requires_grad_(True)
                    m.lora_B.requires_grad_(True)
        else:
            for p in self.model.parameters():
                p.requires_grad_(True)

    # ---- finetune loop -------------------------------------------------------

    def _finetune(self, trials: list["Trial"]) -> None:
        """End-to-end finetune.

        Implements the REVE-paper recipe (Kumar et al. 2022 LP→FT, plus LoRA on
        attention QKVO) when ``lp_warmup_epochs > 0`` and/or ``lora_rank > 0``.
        With both at 0 this is identical to the legacy full-unfreeze finetune.
        """
        cfg = self.train_cfg
        rng = np.random.default_rng(cfg["seed"])
        torch.manual_seed(cfg["seed"])

        # Optional LoRA injection BEFORE building any optimizer — once injected,
        # the parameter list of the model is stable for the whole training run.
        lora_rank = int(cfg.get("lora_rank", 0) or 0)
        if lora_rank > 0:
            n_inj = inject_lora(
                self.model,
                target_patterns=list(cfg["lora_target_patterns"]),
                r=lora_rank,
                alpha=float(cfg.get("lora_alpha", 16.0)),
            )
            self.model.to(self.device)
            print(f"  LoRA: injected r={lora_rank} into {n_inj} modules")
            if n_inj == 0:
                warnings_msg = (
                    f"LoRA target_patterns {cfg['lora_target_patterns']} matched 0 "
                    f"modules in {type(self.model).__name__}. Phase 2 will fall back "
                    f"to LP-only (head trains, encoder frozen)."
                )
                import warnings as _warnings
                _warnings.warn(warnings_msg, stacklevel=2)

        # ----- data prep ------------------------------------------------------
        X, y = self._concat_windows(trials)
        if cfg["max_train_windows"] and len(X) > cfg["max_train_windows"]:
            idx = rng.choice(len(X), cfg["max_train_windows"], replace=False)
            X, y = X[idx], y[idx]

        self._y_mean = y.mean(axis=0)
        self._y_std = y.std(axis=0) + 1e-6
        y_z = ((y - self._y_mean) / self._y_std).astype(np.float32)

        n = len(X)
        idx = rng.permutation(n)
        n_val = max(1, int(n * cfg["val_frac"]))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        Xtr = torch.from_numpy(X[tr_idx])
        ytr = torch.from_numpy(y_z[tr_idx])
        Xva = torch.from_numpy(X[val_idx]).to(self.device)
        yva = torch.from_numpy(y_z[val_idx]).to(self.device)
        loader = DataLoader(
            TensorDataset(Xtr, ytr),
            batch_size=cfg["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        crit = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        bad = 0

        lp_epochs = int(cfg.get("lp_warmup_epochs", 0) or 0)
        ft_epochs = max(0, int(cfg["epochs"]) - lp_epochs)
        lp_lr = float(cfg["lp_lr"]) if cfg.get("lp_lr") is not None else float(cfg["lr"])

        # ----- Phase 1: linear-probe warmup ----------------------------------
        if lp_epochs > 0:
            self._enable_head_only()
            opt = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=lp_lr,
                weight_decay=cfg["weight_decay"],
            )
            print(f"  LP-warmup: {lp_epochs} epochs, {count_trainable(self.model):,} trainable params")
            for epoch in range(lp_epochs):
                self.model.train()
                for xb, yb in loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    pred = self._forward_predict(xb)
                    loss = crit(pred, yb)
                    opt.zero_grad()
                    loss.backward()
                    if cfg["grad_clip"]:
                        nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            cfg["grad_clip"],
                        )
                    opt.step()
                self.model.eval()
                with torch.no_grad():
                    val_loss = crit(self._forward_predict(Xva), yva).item()
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    bad = 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break

        # ----- Phase 2: full FT or LoRA-only FT ------------------------------
        if ft_epochs > 0:
            self._enable_full_or_lora(lora_rank)
            opt = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
            )
            phase_label = "LoRA-FT" if lora_rank > 0 else "Full-FT"
            print(f"  {phase_label}: {ft_epochs} epochs, {count_trainable(self.model):,} trainable params")
            for epoch in range(ft_epochs):
                self.model.train()
                for xb, yb in loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    pred = self._forward_predict(xb)
                    loss = crit(pred, yb)
                    opt.zero_grad()
                    loss.backward()
                    if cfg["grad_clip"]:
                        nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            cfg["grad_clip"],
                        )
                    opt.step()
                self.model.eval()
                with torch.no_grad():
                    val_loss = crit(self._forward_predict(Xva), yva).item()
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    bad = 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def encode_trial(self, trial: "Trial") -> tuple[np.ndarray, np.ndarray]:
        eeg_t = self._trial_at_target_fs(trial)
        vel_t = resample_eeg(trial.vel, src_fs=trial.fs, dst_fs=self.target_fs)
        X, y, centers = self._windowize(eeg_t, vel_t)
        if len(X) == 0:
            return X.reshape(0, self.embed_dim), y
        self._last_centers = centers

        feats_list: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i : i + self.batch_size]).to(self.device)
                fb = self._forward_features(xb)
                feats_list.append(fb.detach().to("cpu", dtype=torch.float32).numpy())
        return np.concatenate(feats_list, axis=0), y

    def predict_trial(self, trial: "Trial") -> np.ndarray:
        """Monolithic prediction path: only valid after end-to-end finetune."""
        if self.freeze:
            raise NotImplementedError(
                f"{type(self).__name__} is frozen; compose with a head via the Pipeline."
            )
        eeg_t = self._trial_at_target_fs(trial)
        vel_t = resample_eeg(trial.vel, src_fs=trial.fs, dst_fs=self.target_fs)
        X, _, centers = self._windowize(eeg_t, vel_t)
        T_src = trial.eeg.shape[-1]
        if len(X) == 0:
            return np.zeros((T_src, self.n_out), dtype=np.float32)
        self._last_centers = centers

        self.model.eval()
        preds_list: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i : i + self.batch_size]).to(self.device)
                p = self._forward_predict(xb)
                preds_list.append(p.detach().to("cpu", dtype=torch.float32).numpy())
        y_win = np.concatenate(preds_list, axis=0)
        if self._y_mean is not None and self._y_std is not None:
            y_win = y_win * self._y_std + self._y_mean
        return self.upsample_to_per_sample(y_win, trial)

    def upsample_to_per_sample(self, y_windows: np.ndarray, trial: "Trial") -> np.ndarray:
        if self._last_centers is None:
            T_target = int(round(trial.eeg.shape[-1] * self.target_fs / trial.fs))
            centers_target = np.arange(self.win_samples // 2, T_target, self.hop_samples)
        else:
            centers_target = self._last_centers
        scale = trial.fs / self.target_fs
        centers_src = np.round(centers_target * scale).astype(np.int64)
        hop_src = max(1, int(round(self.hop_samples * scale)))
        T_src = trial.eeg.shape[-1]
        return windows_to_per_sample(y_windows, centers_src, T_src, hop_src)
