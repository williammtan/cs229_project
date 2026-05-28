"""Unit tests for shared foundation-model input/feature plumbing."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.backbones.fm_base import FMBackboneBase, pool_spatiotemporal_tokens
from src.data.eegmmi import Trial


class DummyFMBackbone(FMBackboneBase):
    def __init__(self, input_scale: float = 1.0):
        super().__init__(
            n_channels=64,
            target_fs=200,
            trial_seconds=0.02,
            freeze=True,
            input_scale=input_scale,
        )

    def _build_model(self) -> nn.Module:
        return nn.Identity()

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)


def test_fm_input_scale_applies_after_resampling():
    eeg = np.ones((64, 4), dtype=np.float32) * 2.0e-5
    trial = Trial(eeg=eeg, label=0, fs=200, subject=1, run=4, trial_idx=0)
    backbone = DummyFMBackbone(input_scale=1.0e6)

    out = backbone._trial_at_target_fs(trial)

    assert out.dtype == np.float32
    assert out.shape == (64, 4)
    assert np.allclose(out, 20.0)


def test_pool_spatiotemporal_tokens_modes():
    feats = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)

    assert pool_spatiotemporal_tokens(feats, "mean").shape == (2, 5)
    assert torch.equal(
        pool_spatiotemporal_tokens(feats, "channel_mean"),
        feats.mean(dim=2).flatten(start_dim=1),
    )
    assert torch.equal(
        pool_spatiotemporal_tokens(feats, "patch_mean"),
        feats.mean(dim=1).flatten(start_dim=1),
    )
    assert torch.equal(
        pool_spatiotemporal_tokens(feats, "flatten"),
        feats.flatten(start_dim=1),
    )


def test_pool_spatiotemporal_tokens_rejects_grid_modes_without_grid():
    feats = torch.zeros(2, 12, 5)

    with pytest.raises(ValueError, match="requires a .* token grid"):
        pool_spatiotemporal_tokens(feats, "channel_mean")

    with pytest.raises(ValueError, match="Unknown feature_pool"):
        pool_spatiotemporal_tokens(feats, "wat")
