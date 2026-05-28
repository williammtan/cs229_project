"""CBraMod backbone (frozen feature extractor by default).

Wraps ``braindecode.models.CBraMod``. CBraMod is channel-agnostic by
construction (no electrode-name table), so 16-channel demo subsets just work.

Pretrained weights:
    HuggingFace ``braindecode/CBraMod-Pretrained`` (open).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.backbones.fm_base import FMBackboneBase, pool_spatiotemporal_tokens
from src.core.registry import register


@register("backbone", "cbramod_frozen")
@register("backbone", "cbramod_finetune")
class CBraModBackbone(FMBackboneBase):
    """CBraMod with configurable per-trial token pooling."""

    embed_dim = 200
    PATCH_SIZE = 200  # samples @ 200 Hz = 1 s, hard-coded by CBraMod pretraining

    def __init__(
        self,
        n_channels: int = 64,
        trial_seconds: float = 4.0,  # >=2s recommended (LAtte short-trial flaw); 4s for EEGMMI
        freeze: bool = True,
        batch_size: int = 32,
        n_classes: int = 4,
        pretrained_id: str | None = "braindecode/CBraMod-Pretrained",
        input_scale: float = 1.0e6,
        feature_pool: str = "channel_mean",
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
        from braindecode.models import CBraMod

        n_times = self.trial_samples
        if n_times % self.PATCH_SIZE != 0:
            raise ValueError(
                f"trial_seconds * 200 must be a multiple of {self.PATCH_SIZE}; got {n_times}."
            )
        model = CBraMod(
            n_outputs=self.n_classes,
            n_chans=self.n_channels,
            n_times=n_times,
            sfreq=self.target_fs,
            input_window_seconds=n_times / self.target_fs,
        )
        if self.pretrained_id:
            try:
                state = _load_hf_state_dict(self.pretrained_id, "pytorch_model.bin")
                n_loaded = _load_overlap(model, state)
                if n_loaded == 0:
                    raise RuntimeError("0 overlapping keys; check filename / architecture.")
                print(f"  CBraMod: loaded {n_loaded}/{len(state)} pretrained tensors from {self.pretrained_id}")
            except Exception as e:
                warnings.warn(
                    f"CBraMod: failed to load pretrained '{self.pretrained_id}': "
                    f"{type(e).__name__}: {e}. Falling back to random init.",
                    stacklevel=2,
                )
        return model

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, return_features=True)
        # out['features']: (B, C, S, 200). The model's native classifier flattens
        # the full C x S grid; default to preserving channel identity while
        # averaging over temporal patches for a tractable frozen probe.
        feats = out["features"]
        return pool_spatiotemporal_tokens(feats, self.feature_pool)


def _load_overlap(target: nn.Module, source_state: dict) -> int:
    """Copy any source parameters that exist (by name + shape) in target."""
    tgt = target.state_dict()
    loaded = 0
    for k, v in source_state.items():
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k] = v
            loaded += 1
    target.load_state_dict(tgt, strict=False)
    return loaded


def _load_hf_state_dict(repo_id: str, filename: str) -> dict:
    """Download a state-dict blob from HF and load it. Handles .bin/.pt/.safetensors."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, filename)
    if filename.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    return obj
