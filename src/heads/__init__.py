"""Heads: per-trial features -> per-trial class predictions."""
from src.heads.base import HeadBase
from src.heads.identity import IdentityHead
from src.heads.softmax_probe import SoftmaxProbeHead
from src.heads.convex_nn import ConvexNNHead

__all__ = ["HeadBase", "IdentityHead", "SoftmaxProbeHead", "ConvexNNHead"]
