"""K-minutes per-subject calibration sweep on top of LOSO.

For each held-out subject and each K in ``k_budgets_min``, yields:

  train = pool of N-1 subjects
  calib = first K minutes of the held-out subject (per ``sample_k_minutes``)
  eval  = remainder of the held-out subject

This is the headline K-min calibration-efficiency protocol in docs/plan.md §6.
"""
from __future__ import annotations

from typing import Iterator, Sequence

from src.core.registry import register
from src.core.types import Split
from src.data.splits import LOSO, sample_k_minutes
from src.data.way_eeg_gal import SubjectData
from src.protocols.base import ProtocolBase


@register("protocol", "kmin_calibration")
class KMinCalibrationProtocol(ProtocolBase):
    name = "kmin"

    def __init__(
        self,
        k_budgets_min: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0),
        seed: int = 0,
    ):
        self.k_budgets_min = tuple(float(k) for k in k_budgets_min)
        self.seed = seed
        self._loso = LOSO(seed=seed)

    def iter_splits(self, data: dict[int, SubjectData]) -> Iterator[Split]:
        for base in self._loso.iter_splits(data):
            held_out_id = base.meta["held_out_subject"]
            held_out_sd = data[held_out_id]
            for k in self.k_budgets_min:
                calib, eval_trials = sample_k_minutes(held_out_sd, k, seed=self.seed)
                yield Split(
                    train=base.train,
                    calib=calib,
                    eval=eval_trials,
                    meta={
                        **base.meta,
                        "k_minutes": k,
                        "n_calib_trials": len(calib),
                        "n_eval_trials": len(eval_trials),
                    },
                )
