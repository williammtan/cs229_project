"""REVE backbone (frozen feature extractor by default).

Wraps ``braindecode.models.REVE``. REVE is channel-agnostic via a 4D Fourier
positional encoding fed by electrode coordinates; we pass EEGMMI channel names
using the exact case-sensitive spelling expected by REVE's position bank.

Pretrained weights:
    HuggingFace ``brain-bzh/reve-base`` — *gated*. Requires
    ``huggingface-cli login`` + accepting the responsible-use agreement on the
    model page. If access is not granted the wrapper falls back to random init.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn

from src.backbones.cbramod import _load_hf_state_dict, _load_overlap
from src.backbones.fm_base import FMBackboneBase, pool_spatiotemporal_tokens
from src.core.registry import register


_REVE_EEGMMI_NAME_MAP: dict[str, str] = {
    # REVE's position bank is case-sensitive and uses a mixed 10-10 spelling.
    # Keep this mapping local to the REVE wrapper so the rest of the repo can
    # use the EEGMMI/MNE-style names from src.data.channels.
    "Fc5": "FC5",
    "Fc3": "FC3",
    "Fc1": "FC1",
    "Fcz": "FCz",
    "Fc2": "FC2",
    "Fc4": "FC4",
    "Fc6": "FC6",
    "Cp5": "CP5",
    "Cp3": "CP3",
    "Cp1": "CP1",
    "Cpz": "CPz",
    "Cp2": "CP2",
    "Cp4": "CP4",
    "Cp6": "CP6",
    "Fp1": "FP1",
    "Fp2": "FP2",
    "Af7": "AF7",
    "Af3": "AF3",
    "Afz": "AFz",
    "Af4": "AF4",
    "Af8": "AF8",
    "Ft7": "FT7",
    "Ft8": "FT8",
    "Tp7": "TP7",
    "Tp8": "TP8",
    "Po7": "PO7",
    "Po3": "PO3",
    "Poz": "POz",
    "Po4": "PO4",
    "Po8": "PO8",
}


def _reve_ch_name(name: str) -> str:
    return _REVE_EEGMMI_NAME_MAP.get(name, name)


def _build_chs_info(ch_names: list[str]) -> list[dict]:
    # Braindecode REVE only reads ch["ch_name"] here, then looks the names up
    # in its own position bank. Supplying MNE montage dicts is unnecessary and
    # can silently retain the wrong spelling for REVE's case-sensitive lookup.
    return [{"ch_name": _reve_ch_name(ch)} for ch in ch_names]


def _zscore_clip_eeg(eeg: np.ndarray, clip: float = 15.0) -> np.ndarray:
    """REVE pretraining-style per-channel z-score + clipping."""
    eeg = np.asarray(eeg, dtype=np.float32)
    mean = eeg.mean(axis=-1, keepdims=True)
    std = eeg.std(axis=-1, keepdims=True)
    std = np.maximum(std, 1.0e-6)
    return np.clip((eeg - mean) / std, -clip, clip).astype(np.float32)


@register("backbone", "reve_frozen")
@register("backbone", "reve_finetune")
class REVEBackbone(FMBackboneBase):
    """REVE with configurable per-trial token pooling."""

    embed_dim = 512
    PATCH_SIZE = 200

    def __init__(
        self,
        n_channels: int = 64,
        trial_seconds: float = 4.0,
        freeze: bool = True,
        batch_size: int = 16,  # 69M params; smaller batch
        n_classes: int = 4,
        pretrained_id: str | None = "brain-bzh/reve-base",
        feature_pool: str = "channel_mean",
        input_scale: float = 1.0,
        finetune_train: dict | None = None,
    ):
        self.pretrained_id = pretrained_id
        self.feature_pool = feature_pool
        super().__init__(
            n_channels=n_channels,
            target_fs=200,
            trial_seconds=trial_seconds,
            freeze=freeze,
            batch_size=batch_size,
            n_classes=n_classes,
            input_scale=input_scale,
            finetune_train=finetune_train,
        )

    def _build_model(self) -> nn.Module:
        from braindecode.models import REVE

        chs_info = _build_chs_info(list(self.channel_names))
        model = REVE(
            n_outputs=self.n_classes,
            n_chans=self.n_channels,
            n_times=self.trial_samples,
            sfreq=self.target_fs,
            input_window_seconds=self.trial_samples / self.target_fs,
            chs_info=chs_info,
        )
        if self.pretrained_id:
            try:
                state = _load_hf_state_dict(self.pretrained_id, "model.safetensors")
                n_loaded = _load_overlap(model, state)
                if n_loaded == 0:
                    raise RuntimeError("0 overlapping keys; check filename / architecture.")
                print(f"  REVE: loaded {n_loaded}/{len(state)} pretrained tensors from {self.pretrained_id}")
            except Exception as e:
                warnings.warn(
                    f"REVE: failed to load pretrained '{self.pretrained_id}': "
                    f"{type(e).__name__}: {e}. "
                    f"This repo is gated — run `huggingface-cli login` and accept "
                    f"the use agreement at https://huggingface.co/{self.pretrained_id}. "
                    f"Falling back to random init.",
                    stacklevel=2,
                )
        return model

    def _trial_at_target_fs(self, trial) -> np.ndarray:
        eeg = super()._trial_at_target_fs(trial)
        return _zscore_clip_eeg(eeg)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, return_features=True)
        # out['features']: (B, C, S, 512). Keep channel identity by default;
        # a global C/S mean is too destructive for motor-imagery lateralization.
        feats = out["features"]
        return pool_spatiotemporal_tokens(feats, self.feature_pool)
