"""Cross-validation and calibration-budget splits for EEGMMI classification.

Splits operate on a ``{subject_id: SubjectData}`` mapping and yield
``Split(train, calib, eval, meta)``. The runner doesn't care which split it
got — every protocol uses the same loop.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from src.core.types import Split
from src.data.eegmmi import SubjectData, Trial


class WithinSubjectKFold:
    """k-fold across trials within each subject, preserving temporal order.

    Yields one Split per (subject, fold). ``calib`` is always empty here —
    calibration is irrelevant when train and eval are the same subject.
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

    When ``n_held_out`` is set and less than the number of subjects in ``data``,
    only that many subjects are sampled (deterministically from ``seed``) as
    held-out folds — each still trained on all *other* subjects. This is
    K-subject random-subsample LOSO, the standard efficiency trick when N is
    large (e.g. 104 EEGMMI subjects): 20 folds give a clean mean ± std and
    enough power for paired Wilcoxon comparisons, at 5× the speed.

    ``calib`` is empty here. If you want LOSO + K-trials calibration, use
    ``KTrialsCalibration(base=LOSO())``.
    """

    def __init__(self, seed: int = 0, n_held_out: int | None = None):
        self.seed = seed
        self.n_held_out = n_held_out

    def iter_splits(self, data: dict[int, SubjectData]) -> Iterator[Split]:
        subjects = sorted(data.keys())
        if self.n_held_out is not None and self.n_held_out < len(subjects):
            rng = np.random.default_rng(self.seed)
            held_out_set = sorted(rng.choice(subjects, size=self.n_held_out, replace=False).tolist())
        else:
            held_out_set = subjects
        for held_out in held_out_set:
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
                    "held_out_subject": int(held_out),
                    "train_subjects": [s for s in subjects if s != held_out],
                },
            )


def _kfold_indices(n_items: int, k: int) -> list[np.ndarray]:
    """k contiguous-block folds. Preserves trial temporal order across the recording."""
    return np.array_split(np.arange(n_items), k)


def sample_k_trials_per_class(
    sd: SubjectData,
    k_per_class: int,
    seed: int = 0,
) -> tuple[list[Trial], list[Trial]]:
    """Sample ``k_per_class`` trials per class as calibration, rest as eval.

    Class balance is enforced. Sampling is the *first* ``k_per_class`` trials of
    each class in temporal order, which mimics "the subject completed the first
    K cues of each type as a calibration block." If a class has fewer than
    ``k_per_class`` trials available, all of them go to calibration and the
    eval set still excludes them.

    Returns (calib, eval). If ``k_per_class <= 0``, calib is empty.
    """
    if k_per_class <= 0:
        return [], list(sd.trials)
    by_class: dict[int, list[int]] = {}
    for i, t in enumerate(sd.trials):
        by_class.setdefault(t.label, []).append(i)
    calib_idx: set[int] = set()
    for cls_indices in by_class.values():
        for i in cls_indices[:k_per_class]:
            calib_idx.add(i)
    calib = [t for i, t in enumerate(sd.trials) if i in calib_idx]
    eval_trials = [t for i, t in enumerate(sd.trials) if i not in calib_idx]
    # Determinism doesn't depend on seed yet because the first-N selector is
    # already deterministic, but keep the parameter for future shuffled variants.
    _ = seed
    return calib, eval_trials
