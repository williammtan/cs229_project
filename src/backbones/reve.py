"""REVE backbone (frozen feature extractor by default).

Wraps ``braindecode.models.REVE``. REVE is channel-agnostic via a 4D Fourier
positional encoding fed by electrode coordinates; we hand it WAY-EEG-GAL's
ActiCap montage as an MNE ``info`` derived from ``standard_1020``.

Pretrained weights:
    HuggingFace ``braindecode/REVE-Pretrained`` — *gated*. Requires
    ``huggingface-cli login`` + accepting the responsible-use agreement on the
    model page. If unavailable the wrapper falls back to random init with a
    clear warning so the pipeline still runs end-to-end.
"""
from __future__ import annotations

import warnings

import mne
import torch
import torch.nn as nn

from src.backbones.cbramod import _load_hf_state_dict, _load_overlap
from src.backbones.fm_base import FMBackboneBase
from src.core.registry import register


def _build_chs_info(ch_names: list[str]) -> list[dict]:
    """Return MNE chs info entries with 10-20 standard positions filled in."""
    info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="warn")
    return info["chs"]


@register("backbone", "reve_frozen")
@register("backbone", "reve_finetune")
class REVEBackbone(FMBackboneBase):
    """REVE with mean-pooled per-window features."""

    embed_dim = 512
    PATCH_SIZE = 200

    def __init__(
        self,
        n_channels: int = 32,
        win_seconds: float = 2.0,
        hop_seconds: float = 0.2,
        freeze: bool = True,
        batch_size: int = 16,  # 69M params; smaller batch
        pretrained_id: str | None = "braindecode/REVE-Pretrained",
        finetune_train: dict | None = None,
    ):
        self.pretrained_id = pretrained_id
        super().__init__(
            n_channels=n_channels,
            target_fs=200,
            win_seconds=win_seconds,
            hop_seconds=hop_seconds,
            freeze=freeze,
            batch_size=batch_size,
            finetune_train=finetune_train,
        )

    def _build_model(self) -> nn.Module:
        from braindecode.models import REVE

        chs_info = _build_chs_info(list(self.channel_names))
        model = REVE(
            n_outputs=3,
            n_chans=self.n_channels,
            n_times=self.win_samples,
            sfreq=self.target_fs,
            input_window_seconds=self.win_samples / self.target_fs,
            chs_info=chs_info,
        )
        if self.pretrained_id:
            try:
                # File name unverified — gated repo. If this throws, fall back gracefully.
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

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, return_features=True)
        # features: (B, C, S, 512) — mean over (C, S) -> (B, 512)
        feats = out["features"]
        return feats.mean(dim=(1, 2))
