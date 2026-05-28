"""Shared scaffolding for foundation-model backbones (classification branch).

All three target FMs (CBraMod, LaBraM, REVE) share:

* Resample preprocessed EEG to the FM's native 200 Hz at the model input boundary
* Optional channel subsetting (64 → 16 demo subset)
* Frozen (softmax-probe path) or full / LoRA fine-tune (end-to-end) toggle
* One embedding per trial: the trial is the window. EEGMMI trials are 4 s,
  which at 200 Hz is 800 samples = 4 patches for CBraMod / LaBraM (patch=200)
  and a standard 4-second window for REVE — no per-trial windowing needed.

Concrete subclasses implement:

    _build_model() -> nn.Module        # loads pretrained weights
    _forward_features(x) -> Tensor     # (B, C, T) -> (B, D) pooled features
    _forward_predict(x)  -> Tensor     # (B, C, T) -> (B, n_classes) for finetune
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
    EEGMMI_64,
    channel_indices,
    get_channel_names,
)
from src.data.resample import resample_eeg

if TYPE_CHECKING:
    from src.data.eegmmi import Trial


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Default finetune hyperparameters — classification.
_DEFAULT_FT_TRAIN = dict(
    lr=5.0e-5,
    weight_decay=1.0e-4,
    epochs=15,
    patience=4,
    batch_size=32,
    val_frac=0.1,
    seed=0,
    grad_clip=1.0,
    max_train_trials=None,
    # REVE-paper LP→FT + LoRA recipe (Kumar et al. 2022; REVE §3.3).
    lp_warmup_epochs=0,
    lp_lr=None,
    lora_rank=0,
    lora_alpha=16.0,
    lora_target_patterns=("to_qkv", "to_out"),
)


def pool_spatiotemporal_tokens(feats: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool FM token grids without silently discarding the requested axes.

    ``feats`` is either ``(B, C, S, D)`` or ``(B, N, D)``.  The latter supports
    only global token pooling / flattening because the channel-patch grid has
    already been flattened by the model.
    """
    valid_modes = {"mean", "channel_mean", "patch_mean", "flatten"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown feature_pool={mode!r}; expected one of "
            "'mean', 'channel_mean', 'patch_mean', 'flatten'."
        )
    if mode == "mean":
        dims = tuple(range(1, feats.ndim - 1))
        return feats.mean(dim=dims)
    if mode == "flatten":
        return feats.flatten(start_dim=1)
    if feats.ndim != 4:
        raise ValueError(
            f"feature_pool={mode!r} requires a (B, C, S, D) token grid; got {tuple(feats.shape)}"
        )
    if mode == "channel_mean":
        # Keep electrode identity, average over temporal patches.
        return feats.mean(dim=2).flatten(start_dim=1)
    if mode == "patch_mean":
        # Keep temporal-patch identity, average over electrodes.
        return feats.mean(dim=1).flatten(start_dim=1)
    raise AssertionError(f"Unhandled feature_pool={mode!r}")


class FMBackboneBase(BackboneBase):
    """Composed-path backbone for pretrained FMs (classification).

    Args:
      n_channels:    64 or 16. Drives the channel subset projection.
      target_fs:     resample target (200 for all three FMs).
      trial_seconds: trial length at target_fs. EEGMMI default = 4.0 s.
      freeze:        if True, no parameters get gradient updates. If False,
                     the model's built-in classification head is trained
                     end-to-end via ``fit_source``; ``predict_trial`` then
                     flows through the trained model (monolithic path).
      batch_size:    inference batch size over trials.
      n_classes:     output dim for the finetune head (default 4).
      input_scale:   multiplicative factor applied after resampling/channel
                     selection. MNE returns EEG in volts; CBraMod/LaBraM
                     pretrained checkpoints expect microvolt-scale values.
      finetune_train: dict of training hyperparams (used when freeze=False).
                     See ``_DEFAULT_FT_TRAIN`` for defaults.
    """

    embed_dim: int = 0  # subclass must set
    n_classes: int = 4

    def __init__(
        self,
        n_channels: int = 64,
        target_fs: int = 200,
        trial_seconds: float = 4.0,
        freeze: bool = True,
        batch_size: int = 32,
        n_classes: int = 4,
        input_scale: float = 1.0,
        finetune_train: dict | None = None,
    ):
        self.n_channels = n_channels
        self.target_fs = target_fs
        self.trial_samples = int(round(trial_seconds * target_fs))
        self.freeze = freeze
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.input_scale = float(input_scale)

        self.train_cfg = {**_DEFAULT_FT_TRAIN, **(finetune_train or {})}

        self.channel_names = get_channel_names(n_channels)
        self._channel_indices = channel_indices(self.channel_names, EEGMMI_64)

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
        """x: (B, C, T). Return (B, D) pooled per-trial features."""
        raise NotImplementedError

    def _forward_predict(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T). Return (B, n_classes) logits. Default delegates to the model."""
        return self.model(x)

    # ---- standard machinery --------------------------------------------------

    def _trial_at_target_fs(self, trial: "Trial") -> np.ndarray:
        eeg = trial.eeg[self._channel_indices]
        eeg = resample_eeg(eeg, src_fs=trial.fs, dst_fs=self.target_fs)
        # Trim or zero-pad to exactly trial_samples so every trial has identical T.
        C, T = eeg.shape
        if T == self.trial_samples:
            return self._preprocess_eeg(eeg)
        if T > self.trial_samples:
            return self._preprocess_eeg(eeg[:, : self.trial_samples])
        out = np.zeros((C, self.trial_samples), dtype=np.float32)
        out[:, :T] = eeg
        return self._preprocess_eeg(out)

    def _preprocess_eeg(self, eeg: np.ndarray) -> np.ndarray:
        """Final model-input preprocessing hook after channeling/resampling."""
        if self.input_scale != 1.0:
            eeg = eeg * self.input_scale
        return eeg.astype(np.float32, copy=False)

    def _stack_trials(self, trials: list["Trial"]) -> tuple[np.ndarray, np.ndarray]:
        X = np.stack([self._trial_at_target_fs(t) for t in trials], axis=0)
        y = np.asarray([t.label for t in trials], dtype=np.int64)
        return X.astype(np.float32), y

    # ---- fit / encode / predict ---------------------------------------------

    def fit_source(self, trials):
        if self.freeze:
            return None
        self._finetune(trials)

    def _head_module(self) -> nn.Module:
        """The classification head trained during the LP-warmup phase.

        Default: ``self.model.final_layer`` (REVE/CBraMod). LaBraM keeps its
        head elsewhere; override in those subclasses if you enable LP warmup.
        """
        if hasattr(self.model, "final_layer"):
            return self.model.final_layer
        raise AttributeError(
            f"{type(self).__name__}: no `final_layer` found; override `_head_module()` "
            f"to point at the classification head before enabling LP warmup."
        )

    def _freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _enable_head_only(self) -> None:
        self._freeze_all()
        for p in self._head_module().parameters():
            p.requires_grad_(True)

    def _enable_full_or_lora(self, lora_rank: int) -> None:
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

    def _finetune(self, trials: list["Trial"]) -> None:
        cfg = self.train_cfg
        rng = np.random.default_rng(cfg["seed"])
        torch.manual_seed(cfg["seed"])

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
                import warnings as _warnings
                _warnings.warn(
                    f"LoRA target_patterns {cfg['lora_target_patterns']} matched 0 modules; "
                    f"Phase 2 will fall back to LP-only.",
                    stacklevel=2,
                )

        X, y = self._stack_trials(trials)
        if cfg["max_train_trials"] and len(X) > cfg["max_train_trials"]:
            idx = rng.choice(len(X), cfg["max_train_trials"], replace=False)
            X, y = X[idx], y[idx]

        n = len(X)
        idx = rng.permutation(n)
        n_val = max(1, int(n * cfg["val_frac"]))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        Xtr = torch.from_numpy(X[tr_idx])
        ytr = torch.from_numpy(y[tr_idx])
        Xva = torch.from_numpy(X[val_idx]).to(self.device)
        yva = torch.from_numpy(y[val_idx]).to(self.device)
        loader = DataLoader(
            TensorDataset(Xtr, ytr),
            batch_size=cfg["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        crit = nn.CrossEntropyLoss()

        best_val = float("inf")
        best_state = None
        bad = 0

        lp_epochs = int(cfg.get("lp_warmup_epochs", 0) or 0)
        ft_epochs = max(0, int(cfg["epochs"]) - lp_epochs)
        lp_lr = float(cfg["lp_lr"]) if cfg.get("lp_lr") is not None else float(cfg["lr"])

        def _run_phase(epochs, lr, label):
            nonlocal best_val, best_state, bad
            if epochs <= 0:
                return
            opt = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=cfg["weight_decay"],
            )
            print(f"  {label}: {epochs} epochs, {count_trainable(self.model):,} trainable params")
            for _ in range(epochs):
                self.model.train()
                for xb, yb in loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    logits = self._forward_predict(xb)
                    loss = crit(logits, yb)
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
                        return

        if lp_epochs > 0:
            self._enable_head_only()
            _run_phase(lp_epochs, lp_lr, "LP-warmup")
        if ft_epochs > 0:
            self._enable_full_or_lora(lora_rank)
            phase_label = "LoRA-FT" if lora_rank > 0 else "Full-FT"
            _run_phase(ft_epochs, float(cfg["lr"]), phase_label)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def encode_trial(self, trial: "Trial") -> np.ndarray:
        x = self._trial_at_target_fs(trial)
        xb = torch.from_numpy(x[None, ...]).to(self.device)
        with torch.no_grad():
            f = self._forward_features(xb)
        return f.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)

    def predict_trial(self, trial: "Trial") -> np.ndarray:
        """Monolithic prediction: returns (n_classes,) softmax probabilities.
        Only valid after end-to-end finetune."""
        if self.freeze:
            raise NotImplementedError(
                f"{type(self).__name__} is frozen; compose with a head via the Pipeline."
            )
        x = self._trial_at_target_fs(trial)
        xb = torch.from_numpy(x[None, ...]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self._forward_predict(xb)
            proba = torch.softmax(logits, dim=-1)[0]
        return proba.detach().to("cpu", dtype=torch.float32).numpy()
