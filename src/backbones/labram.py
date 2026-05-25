"""LaBraM backbone (frozen feature extractor by default).

Wraps ``braindecode.models.Labram``. LaBraM keys electrode rows to canonical
10-20 names via ch_names. WAY-EEG-GAL's ActiCap-32 labels (verified) map
cleanly into LaBraM's canonical channel table.

Pretrained weights:
    HuggingFace ``braindecode/Labram-Braindecode`` (open).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.backbones.cbramod import _load_hf_state_dict, _load_overlap
from src.backbones.fm_base import FMBackboneBase
from src.core.registry import register


@register("backbone", "labram_frozen")
@register("backbone", "labram_finetune")
class LabramBackbone(FMBackboneBase):
    """LaBraM with mean-pooled per-window features."""

    embed_dim = 200
    PATCH_SIZE = 200

    def __init__(
        self,
        n_channels: int = 32,
        win_seconds: float = 2.0,
        hop_seconds: float = 0.2,
        freeze: bool = True,
        batch_size: int = 32,
        pretrained_id: str | None = "braindecode/Labram-Braindecode",
        use_cls_token: bool = True,
        finetune_train: dict | None = None,
    ):
        self.pretrained_id = pretrained_id
        self.use_cls_token = use_cls_token
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
        from braindecode.models import Labram

        n_times = self.win_samples
        if n_times % self.PATCH_SIZE != 0:
            raise ValueError(
                f"win_seconds * 200 must be a multiple of {self.PATCH_SIZE}; got {n_times}."
            )
        model = Labram(
            n_outputs=3,
            n_chans=self.n_channels,
            n_times=n_times,
            sfreq=self.target_fs,
            input_window_seconds=n_times / self.target_fs,
            use_mean_pooling=True,
        )
        if self.pretrained_id:
            try:
                state = _load_hf_state_dict(self.pretrained_id, "braindecode_labram_base.pt")
                n_loaded = _load_overlap(model, state)
                if n_loaded == 0:
                    raise RuntimeError("0 overlapping keys; check filename / architecture.")
                print(f"  Labram: loaded {n_loaded}/{len(state)} pretrained tensors from {self.pretrained_id}")
            except Exception as e:
                warnings.warn(
                    f"Labram: failed to load pretrained '{self.pretrained_id}': "
                    f"{type(e).__name__}: {e}. Falling back to random init.",
                    stacklevel=2,
                )
        return model

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # LaBraM's wrapper normalizes its own input scaling internally per
        # braindecode's reimplementation (see preprocessor in cls).
        out = self.model(x, ch_names=list(self.channel_names), return_features=True)
        # features: (B, n_tokens, D); cls_token: (B, D) or None
        if self.use_cls_token and out.get("cls_token") is not None:
            return out["cls_token"]
        return out["features"].mean(dim=1)

    def _forward_predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, ch_names=list(self.channel_names))
