"""Static Riemannian alignment (calibration-based).

* Source-side fit: groups training trials by ``trial.subject`` and stores one
  Frechet-mean covariance per source subject. Each training trial is whitened
  by its OWN subject's mean cov at ``transform_trial`` time.
* Target-side calibration: computes a single Frechet-mean cov from the
  calibration trials of the held-out target subject.
* If no calibration was provided (zero-shot LOSO with no calibration set), the
  adapter falls back to per-trial whitening — each test trial is whitened by
  its own covariance. This is Euclidean alignment / PEA, the weakest baseline,
  but it still always works.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.adapters.base import AdapterBase
from src.adapters.riemannian._common import (
    frechet_mean,
    trial_cov,
    whiten_eeg,
    whitener,
)
from src.core.registry import register

if TYPE_CHECKING:
    from src.data.way_eeg_gal import Trial


@register("adapter", "ra_static")
class StaticRA(AdapterBase):
    """Riemannian alignment via Frechet-mean covariance whitening.

    Args:
        regularize: ridge added to each SCM for invertibility.
        target_space: ``'identity'`` whitens to I (Zanini-style RA). Future
            extension: parallel-transport to a stored source mean.
        zero_shot_fallback: when target calibration has not been called and a
            new trial arrives, ``'per_trial'`` whitens it by its own
            covariance (a.k.a. Euclidean alignment); ``'none'`` returns the
            trial unchanged.
    """

    def __init__(
        self,
        regularize: float = 1.0e-5,
        target_space: str = "identity",
        zero_shot_fallback: str = "per_trial",
    ):
        if target_space != "identity":
            raise NotImplementedError(
                "Only target_space='identity' (whiten-to-I) is implemented."
            )
        if zero_shot_fallback not in ("per_trial", "none"):
            raise ValueError(zero_shot_fallback)
        self.regularize = regularize
        self.target_space = target_space
        self.zero_shot_fallback = zero_shot_fallback

        # subject_id (int or None for the held-out target) -> (C, C) whitener
        self._whiteners: dict[int | None, np.ndarray] = {}
        # The known source subjects, for "is this a source or target?" lookup.
        self._source_subjects: set[int] = set()
        self._target_calibrated: bool = False

    # ---- source-pool fit ----------------------------------------------------
    def fit_source_trials(self, trials: list["Trial"]) -> None:
        self._whiteners.clear()
        self._target_calibrated = False
        by_subject: dict[int, list[np.ndarray]] = {}
        for t in trials:
            by_subject.setdefault(t.subject, []).append(trial_cov(t.eeg, self.regularize))
        for sid, covs in by_subject.items():
            M = frechet_mean(np.stack(covs))
            self._whiteners[sid] = whitener(M)
        self._source_subjects = set(self._whiteners.keys())

    # ---- target-subject calibration -----------------------------------------
    def calibrate_trials(self, trials: list["Trial"]) -> None:
        if not trials:
            return
        covs = np.stack([trial_cov(t.eeg, self.regularize) for t in trials])
        M = frechet_mean(covs)
        # Per Trial.subject convention, calibration trials share one subject.
        target_id = trials[0].subject
        self._whiteners[target_id] = whitener(M)
        self._target_calibrated = True

    # ---- per-trial transform ------------------------------------------------
    def transform_trial(self, trial: "Trial") -> "Trial":
        W = self._whiteners.get(trial.subject)
        if W is None:
            if self.zero_shot_fallback == "per_trial":
                W = whitener(trial_cov(trial.eeg, self.regularize))
            else:
                return trial
        eeg_aligned = whiten_eeg(trial.eeg, W)
        from dataclasses import replace

        return replace(trial, eeg=eeg_aligned)
