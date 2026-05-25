"""Minimal LoRA (Hu et al. 2021) for fine-tuning frozen FM backbones.

Wraps an ``nn.Linear(in, out)`` with a trainable rank-r delta:

    y = W·x + (alpha/r) · (B·A)·x        with A ∈ R^{r×in}, B ∈ R^{out×r}

A is initialised Kaiming-uniform, B is zero — so the model output at injection
time is identical to the frozen baseline. The original Linear weight stays
frozen and only A, B receive gradients.

We follow the REVE paper's recipe (Section 3.3): inject LoRA into the Q, K, V
projections (which in braindecode's REVE are fused into one ``to_qkv`` Linear)
and the output ``to_out`` projection of every transformer block.
"""
from __future__ import annotations

import fnmatch
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen Linear + trainable low-rank delta. Drop-in replacement."""

    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0")
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        # Keep the original Linear as a frozen child module.
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = base.in_features
        out_features = base.out_features
        # A: (r, in), B: (out, r). y_delta = scale * x @ A^T @ B^T.
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # x: (..., in); A^T: (in, r); B^T: (r, out)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return out + self.scale * delta


def _matches(name: str, patterns: list[str]) -> bool:
    """fnmatch-style match — patterns like 'to_qkv' or '*to_out' or 'transformer.layers.*.to_qkv'."""
    return any(fnmatch.fnmatchcase(name, p) or name.endswith("." + p) or name == p for p in patterns)


def inject_lora(
    model: nn.Module,
    target_patterns: list[str],
    r: int = 8,
    alpha: float = 16.0,
) -> int:
    """Recursively replace nn.Linear modules whose qualified name matches any
    pattern with LoRALinear wrappers. Returns the number of modules replaced.

    Patterns are matched against the dotted ``model.named_modules()`` paths
    using fnmatch semantics; plain names (e.g. ``"to_qkv"``) match any module
    whose qualified name ends with ``.to_qkv``.
    """
    if r <= 0:
        return 0
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    for qname, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{qname}.{child_name}" if qname else child_name
            if isinstance(child, nn.Linear) and _matches(full_name, target_patterns):
                targets.append((module, child_name, child))

    for parent, attr, linear in targets:
        wrapped = LoRALinear(linear, r=r, alpha=alpha)
        setattr(parent, attr, wrapped)

    return len(targets)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in trainable_parameters(model))
