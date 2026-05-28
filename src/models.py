"""Baseline decoders for 4-class motor-imagery EEG classification (EEGMMI).

Two compact convnets, both producing per-trial class logits:

  1. EEGNetClf       : EEGNet (Lawhern et al. 2018) with a softmax head.
  2. ShallowConvNetClf: ShallowConvNet (Schirrmeister et al. 2017) with a softmax head.

Input convention for the convnets:
  X has shape (B, 1, C, T) where C is channels and T is samples in the trial.

Output: (B, n_classes) logits (apply softmax / CrossEntropyLoss downstream).
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ------------------------------------------------------------------ EEGNet --

class EEGNetClf(nn.Module):
    """EEGNet (Lawhern 2018) with a softmax classification head.
    Input: (B, 1, C, T). Output: (B, n_classes) logits."""

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 400,
        n_classes: int = 4,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_len: int = 64,  # ~half the sampling rate, per the EEGNet paper
        dropout: float = 0.5,
    ):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_len), padding=(0, kernel_len // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.separable(self.depthwise(self.firstconv(dummy)))
            self.flat_dim = out.numel()
        self.head = nn.Linear(self.flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.flatten(start_dim=1)
        return self.head(x)


# ------------------------------------------------------------ ShallowConvNet --

class ShallowConvNetClf(nn.Module):
    """ShallowConvNet (Schirrmeister 2017) with a softmax head.
    Mirrors FBCSP: temporal conv -> spatial conv -> square -> avg pool -> log -> linear.
    Input: (B, 1, C, T). Output: (B, n_classes) logits."""

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 400,
        n_classes: int = 4,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, (1, temporal_kernel), bias=False)
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            h = self.temporal(dummy)
            h = self.spatial(h)
            h = self.bn(h)
            h = h * h
            h = self.pool(h)
            h = torch.log(torch.clamp(h, min=1e-6))
            self.flat_dim = h.numel()
        self.head = nn.Linear(self.flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x)
        h = self.spatial(h)
        h = self.bn(h)
        h = h * h
        h = self.pool(h)
        h = torch.log(torch.clamp(h, min=1e-6))
        h = self.dropout(h)
        h = h.flatten(start_dim=1)
        return self.head(h)
