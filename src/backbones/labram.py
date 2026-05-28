"""LaBraM backbone (frozen feature extractor by default).

Wraps ``braindecode.models.Labram``. LaBraM keys electrode rows to canonical
10-20 names via ch_names. EEGMMI's 64-channel BCI2000 montage maps cleanly
into LaBraM's canonical channel table once names are normalized
(``Fc5.`` -> ``Fc5``, etc.).

Pretrained weights:
    HuggingFace ``braindecode/Labram-Braindecode`` (open).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.backbones.cbramod import _load_hf_state_dict, _load_overlap
from src.backbones.fm_base import FMBackboneBase, pool_spatiotemporal_tokens
from src.core.registry import register


@register("backbone", "labram_frozen")
@register("backbone", "labram_finetune")
class LabramBackbone(FMBackboneBase):
    """LaBraM with configurable per-trial token pooling."""

    embed_dim = 200
    PATCH_SIZE = 200

    def __init__(
        self,
        n_channels: int = 64,
        trial_seconds: float = 4.0,
        freeze: bool = True,
        batch_size: int = 32,
        n_classes: int = 4,
        pretrained_id: str | None = "braindecode/Labram-Braindecode",
        input_scale: float = 1.0e6,
        feature_pool: str = "channel_mean",
        use_cls_token: bool = False,
        finetune_train: dict | None = None,
    ):
        self.pretrained_id = pretrained_id
        self.feature_pool = feature_pool
        self.use_cls_token = use_cls_token
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
        from braindecode.models import Labram

        n_times = self.trial_samples
        if n_times % self.PATCH_SIZE != 0:
            raise ValueError(
                f"trial_seconds * 200 must be a multiple of {self.PATCH_SIZE}; got {n_times}."
            )
        model = Labram(
            n_outputs=self.n_classes,
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
        out = self.model(x, ch_names=list(self.channel_names), return_features=True)
        if self.use_cls_token and out.get("cls_token") is not None:
            return out["cls_token"]
        feats = out["features"]
        if self.feature_pool in {"channel_mean", "patch_mean"}:
            B, N, D = feats.shape
            if N % self.n_channels != 0:
                raise ValueError(
                    f"LaBraM returned {N} tokens, not divisible by n_channels={self.n_channels}"
                )
            feats = feats.reshape(B, self.n_channels, N // self.n_channels, D)
        return pool_spatiotemporal_tokens(feats, self.feature_pool)

    def _forward_predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, ch_names=list(self.channel_names))
