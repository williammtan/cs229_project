"""Protocol protocol: iter_splits(data) -> Iterator[Split].

The runner is one generic loop — every protocol is just a Split source.
"""
from __future__ import annotations

from typing import Iterator

from src.core.types import Split
from src.data.way_eeg_gal import SubjectData


class ProtocolBase:
    name: str = "base"

    def iter_splits(self, data: dict[int, SubjectData]) -> Iterator[Split]:
        raise NotImplementedError
