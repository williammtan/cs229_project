"""Leave-one-subject-out cross-subject evaluation (zero-shot)."""
from __future__ import annotations

from typing import Iterator

from src.core.registry import register
from src.core.types import Split
from src.data.splits import LOSO
from src.data.way_eeg_gal import SubjectData
from src.protocols.base import ProtocolBase


@register("protocol", "loso")
class LOSOProtocol(ProtocolBase):
    name = "loso"

    def __init__(self, seed: int = 0):
        self._impl = LOSO(seed=seed)

    def iter_splits(self, data: dict[int, SubjectData]) -> Iterator[Split]:
        yield from self._impl.iter_splits(data)
