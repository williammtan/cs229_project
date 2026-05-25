"""Unit tests for src.backbones.lora — LoRA wrapper + injector."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.backbones.lora import (
    LoRALinear,
    count_trainable,
    inject_lora,
    trainable_parameters,
)


class TestLoRALinear:
    def test_invalid_rank_rejected(self):
        base = nn.Linear(8, 16)
        with pytest.raises(ValueError):
            LoRALinear(base, r=0)

    def test_init_output_identical_to_base(self):
        """At init, B=0 so the LoRA delta is identically 0 — the wrapped Linear
        must produce exactly the same output as the bare one."""
        torch.manual_seed(0)
        base = nn.Linear(8, 16)
        lora = LoRALinear(base, r=4, alpha=8.0)
        x = torch.randn(3, 8)
        y_base = base(x)
        y_lora = lora(x)
        assert torch.allclose(y_base, y_lora, atol=1e-6)

    def test_base_weights_frozen_after_wrap(self):
        base = nn.Linear(8, 16)
        lora = LoRALinear(base, r=4)
        assert lora.base.weight.requires_grad is False
        if lora.base.bias is not None:
            assert lora.base.bias.requires_grad is False

    def test_lora_params_have_grad(self):
        base = nn.Linear(8, 16)
        lora = LoRALinear(base, r=4)
        assert lora.lora_A.requires_grad is True
        assert lora.lora_B.requires_grad is True

    def test_lora_delta_changes_output_after_step(self):
        torch.manual_seed(0)
        lora = LoRALinear(nn.Linear(8, 16), r=4, alpha=8.0)
        opt = torch.optim.SGD([lora.lora_A, lora.lora_B], lr=0.1)
        x = torch.randn(3, 8)
        target = torch.randn(3, 16)
        # One step of SGD: B should move off zero.
        loss = ((lora(x) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        # B is no longer all zero.
        assert torch.abs(lora.lora_B).sum() > 0
        # Output now differs from base alone.
        y_base = lora.base(x)
        y_lora = lora(x)
        assert not torch.allclose(y_base, y_lora, atol=1e-6)

    def test_lora_param_count_is_small(self):
        in_dim, out_dim, r = 100, 200, 4
        lora = LoRALinear(nn.Linear(in_dim, out_dim, bias=False), r=r)
        # LoRA adds r*(in+out) params; base has in*out frozen.
        n_lora = sum(p.numel() for p in [lora.lora_A, lora.lora_B])
        assert n_lora == r * (in_dim + out_dim)
        n_trainable = count_trainable(lora)
        assert n_trainable == n_lora


class _TinyAttnBlock(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Linear(dim, dim)

    def forward(self, x):
        return self.ff(self.to_out(self.to_qkv(x).chunk(3, dim=-1)[0]))


class _TinyTransformer(nn.Module):
    def __init__(self, n_layers: int = 3, dim: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([_TinyAttnBlock(dim) for _ in range(n_layers)])
        self.final_layer = nn.Linear(dim, 3)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)


class TestInjectLora:
    def test_returns_zero_on_empty_patterns(self):
        m = _TinyTransformer()
        n = inject_lora(m, target_patterns=[], r=8)
        assert n == 0

    def test_returns_zero_on_rank_zero(self):
        m = _TinyTransformer()
        n = inject_lora(m, target_patterns=["to_qkv"], r=0)
        assert n == 0

    def test_matches_by_bare_name(self):
        m = _TinyTransformer(n_layers=3)
        n = inject_lora(m, target_patterns=["to_qkv"], r=4)
        assert n == 3  # one per layer
        for layer in m.layers:
            assert isinstance(layer.to_qkv, LoRALinear)
            assert not isinstance(layer.to_out, LoRALinear)
            assert not isinstance(layer.ff, LoRALinear)

    def test_matches_multiple_patterns(self):
        m = _TinyTransformer(n_layers=3)
        n = inject_lora(m, target_patterns=["to_qkv", "to_out"], r=4)
        assert n == 6
        for layer in m.layers:
            assert isinstance(layer.to_qkv, LoRALinear)
            assert isinstance(layer.to_out, LoRALinear)

    def test_does_not_change_output_at_init(self):
        torch.manual_seed(0)
        m = _TinyTransformer(n_layers=2, dim=8)
        x = torch.randn(2, 5, 8)
        y_before = m(x).detach().clone()
        inject_lora(m, target_patterns=["to_qkv", "to_out"], r=4)
        y_after = m(x)
        assert torch.allclose(y_before, y_after, atol=1e-6)

    def test_post_injection_only_lora_and_head_trainable(self):
        """LoRA injection alone doesn't change which params train; that's the
        job of fm_base._enable_full_or_lora. But the LoRA params themselves
        must default to requires_grad=True."""
        m = _TinyTransformer(n_layers=2, dim=8)
        inject_lora(m, target_patterns=["to_qkv", "to_out"], r=4)
        lora_params = [p for n, p in m.named_parameters() if "lora_" in n]
        assert len(lora_params) > 0
        for p in lora_params:
            assert p.requires_grad is True
