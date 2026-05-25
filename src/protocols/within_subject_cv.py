"""Within-subject k-fold CV. Same trial set for train and eval, different folds."""
from __future__ import annotations

from typing import Iterator

from src.core.registry import register
from src.core.types import Split
from src.data.splits import WithinSubjectKFold
from src.data.way_eeg_gal import SubjectData
from src.protocols.base import ProtocolBase


@register("protocol", "within_subject_cv")
class WithinSubjectCVProtocol(ProtocolBase):
    name = "within_cv"

    def __init__(self, k: int = 5, seed: int = 0):
        self._impl = WithinSubjectKFold(k=k, seed=seed)

    def iter_splits(self, data: dict[int, SubjectData]) -> Iterator[Split]:
        yield from self._impl.iter_splits(data)
