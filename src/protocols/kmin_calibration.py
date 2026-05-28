"""K-trials-per-class per-subject calibration sweep on top of LOSO.

For each held-out subject and each K in ``k_budgets_trials``, yields:

  train = pool of N-1 subjects
  calib = first K trials per class from the held-out subject
  eval  = remaining trials of the held-out subject

This is the headline K-trials calibration-efficiency protocol in
docs/plan_clf.md §6. The protocol name is kept as ``kmin_calibration`` for
config-tree symmetry, but the budget unit is now *trials per class*.
"""
from __future__ import annotations

from typing import Iterator, Sequence

from src.core.registry import register
from src.core.types import Split
from src.data.eegmmi import SubjectData
from src.data.splits import LOSO, sample_k_trials_per_class
from src.protocols.base import ProtocolBase


@register("protocol", "kmin_calibration")
@register("protocol", "ktrials_calibration")
class KTrialsCalibrationProtocol(ProtocolBase):
    name = "ktrials"

    def __init__(
        self,
        k_budgets_trials: Sequence[int] = (0, 1, 2, 5, 10, 20),
        seed: int = 0,
        n_held_out: int | None = None,
    ):
        self.k_budgets_trials = tuple(int(k) for k in k_budgets_trials)
        self.seed = seed
        self.n_held_out = n_held_out
        self._loso = LOSO(seed=seed, n_held_out=n_held_out)

    def iter_splits(self, data: dict[int, SubjectData]) -> Iterator[Split]:
        for base in self._loso.iter_splits(data):
            held_out_id = base.meta["held_out_subject"]
            held_out_sd = data[held_out_id]
            for k in self.k_budgets_trials:
                calib, eval_trials = sample_k_trials_per_class(held_out_sd, k, seed=self.seed)
                yield Split(
                    train=base.train,
                    calib=calib,
                    eval=eval_trials,
                    meta={
                        **base.meta,
                        "k_trials_per_class": k,
                        "n_calib_trials": len(calib),
                        "n_eval_trials": len(eval_trials),
                    },
                )
