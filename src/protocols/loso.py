"""Leave-one-subject-out cross-subject evaluation (zero-shot)."""
from __future__ import annotations

from typing import Iterator

from src.core.registry import register
from src.core.types import Split
from src.data.eegmmi import SubjectData
from src.data.splits import LOSO
from src.protocols.base import ProtocolBase


@register("protocol", "loso")
class LOSOProtocol(ProtocolBase):
    """Leave-one-subject-out, optionally subsampled to ``n_held_out`` folds.

    With ``n_held_out=None`` this is full N-fold LOSO. With ``n_held_out=20``
    on 104 subjects it samples 20 held-out subjects (deterministic from
    ``seed``), each trained on all 103 others — the standard efficiency trick
    when full N-fold LOSO is too expensive.
    """

    name = "loso"

    def __init__(self, seed: int = 0, n_held_out: int | None = None):
        self.n_held_out = n_held_out
        self._impl = LOSO(seed=seed, n_held_out=n_held_out)

    def iter_splits(self, data: dict[int, SubjectData]) -> Iterator[Split]:
        yield from self._impl.iter_splits(data)
