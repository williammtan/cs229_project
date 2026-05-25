"""Per-subject state memory accounting.

Stubs only — fill in when adapters land. Idea: sum nbytes of every numpy
array / torch tensor an adapter persists per-subject.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def state_size_bytes(obj: Any) -> int:
    """Best-effort sum of bytes held by ndarrays / tensors hanging off ``obj``."""
    total = 0
    for v in getattr(obj, "__dict__", {}).values():
        if isinstance(v, np.ndarray):
            total += v.nbytes
        elif hasattr(v, "element_size") and hasattr(v, "numel"):  # torch.Tensor duck-type
            total += v.element_size() * v.numel()
    return total
