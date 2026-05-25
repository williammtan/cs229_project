"""Cross-validation and calibration-budget splits.

Splits operate on a ``{subject_id: SubjectData}`` mapping and yield
``Split(train, calib, eval, meta)``. The runner doesn't care which split it
got — every protocol uses the same loop.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from src.core.types import Split
from src.data.way_eeg_gal import SubjectData, Trial


class WithinSubjectKFold:
    """k-fold across trials within each subject, preserving temporal order.

    Yields one Split per (subject, fold). ``calib`` is always empty for
    within-subject CV — calibration is irrelevant when train and eval are the
    same subject.
    """

    def __init__(self, k: int = 5, seed: int = 0):
        self.k = k
        self.seed = seed

    def iter_splits(self, data: dict[int, SubjectData]) -> Iterator[Split]:
        for subject_id, sd in data.items():
            folds = _kfold_indices(len(sd.trials), self.k)
            for fold_i, eval_idx in enumerate(folds):
                eval_set = set(eval_idx.tolist())
                train_trials = [t for i, t in enumerate(sd.trials) if i not in eval_set]
                eval_trials = [t for i, t in enumerate(sd.trials) if i in eval_set]
                yield Split(
                    train=train_trials,
                    calib=[],
                    eval=eval_trials,
                    meta={
                        "subject": subject_id,
                        "fold": fold_i,
                        "n_folds": self.k,
                    },
                )


class LOSO:
    """Leave-one-subject-out. Train pooled on N-1, evaluate on held-out subject.

    ``calib`` is empty here. If you want LOSO + K-min calibration, use
    ``KMinCalibration(base=LOSO())``.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def iter_splits(self, data: dict[int, SubjectData]) -> Iterator[Split]:
        subjects = sorted(data.keys())
        for held_out in subjects:
            train_trials: list[Trial] = []
            for s in subjects:
                if s == held_out:
                    continue
                train_trials.extend(data[s].trials)
            yield Split(
                train=train_trials,
                calib=[],
                eval=list(data[held_out].trials),
                meta={
                    "held_out_subject": held_out,
                    "train_subjects": [s for s in subjects if s != held_out],
                },
            )


def _kfold_indices(n_items: int, k: int) -> list[np.ndarray]:
    """k contiguous-block folds. Preserves trial temporal order across the recording."""
    return np.array_split(np.arange(n_items), k)


def sample_k_minutes(
    sd: SubjectData,
    k_minutes: float,
    seed: int = 0,
) -> tuple[list[Trial], list[Trial]]:
    """Sample the first ``k_minutes`` of recording from a subject as the calibration
    set; everything after becomes the evaluation set. Uses trial wall-clock duration
    (computed from ``trial.fs``).

    Returns (calib_trials, eval_trials). If k_minutes <= 0, calib is empty.
    """
    if k_minutes <= 0:
        return [], list(sd.trials)
    budget_sec = k_minutes * 60.0
    calib: list[Trial] = []
    used_sec = 0.0
    eval_start = 0
    for i, t in enumerate(sd.trials):
        dur = t.eeg.shape[-1] / float(t.fs)
        if used_sec + dur <= budget_sec:
            calib.append(t)
            used_sec += dur
            eval_start = i + 1
        else:
            break
    eval_trials = list(sd.trials[eval_start:])
    return calib, eval_trials
