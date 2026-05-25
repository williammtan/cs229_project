"""Heads: per-window features -> per-window predictions.

The backbone owns windowing and per-sample upsampling. Heads are window-only.
"""
from src.heads.base import HeadBase
from src.heads.identity import IdentityHead
from src.heads.linear_probe import LinearProbeHead

__all__ = ["HeadBase", "IdentityHead", "LinearProbeHead"]
