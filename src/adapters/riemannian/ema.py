"""EMA Riemannian alignment — online manifold update.

The target subject's mean covariance evolves as a geodesic EMA on the SPD
manifold:

    M_t = geodesic_α(M_{t-1}, cov(x_t))

where α ∈ (0, 1] is the step size (small α = slow tracking, large α = fast
tracking). After each update we recompute the whitener M^{-1/2}.

State updates happen inside ``transform_trial``: apply the current whitener,
then absorb the new trial's covariance for next time. This matches the online
"see one trial, predict, then update" loop you'd run in a real BCI.

For source-pool fit, this class is identical to :class:`StaticRA` — Frechet
mean per source subject, no online updates (source data is fixed). Online
behaviour only kicks in for the held-out target subject.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyriemann.utils.geodesic import geodesic_riemann

from src.adapters.base import AdapterBase
from src.adapters.riemannian._common import (
    frechet_mean,
    trial_cov,
    whiten_eeg,
    whitener,
)
from src.core.registry import register

if TYPE_CHECKING:
    from src.data.eegmmi import Trial


@register("adapter", "ra_ema")
class EMARA(AdapterBase):
    """Geodesic EMA Riemannian alignment.

    Args:
        alpha: geodesic step size in (0, 1]. 0.05 is conservative (~20-trial
            effective window). 0.2 is aggressive.
        regularize: ridge added to per-trial SCM for invertibility.
        update_online: if True, ``transform_trial`` updates M_target on every
            trial it sees (true online behaviour). If False, M_target is only
            updated at ``calibrate_trials`` time (equivalent to StaticRA).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        regularize: float = 1.0e-5,
        update_online: bool = True,
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.regularize = regularize
        self.update_online = update_online

        self._whiteners: dict[int | None, np.ndarray] = {}
        self._means: dict[int | None, np.ndarray] = {}  # current M_t per subject
        self._source_subjects: set[int] = set()

    # ---- source-pool fit ----------------------------------------------------
    def fit_source_trials(self, trials: list["Trial"]) -> None:
        self._whiteners.clear()
        self._means.clear()
        by_subject: dict[int, list[np.ndarray]] = {}
        for t in trials:
            by_subject.setdefault(t.subject, []).append(trial_cov(t.eeg, self.regularize))
        for sid, covs in by_subject.items():
            M = frechet_mean(np.stack(covs))
            self._means[sid] = M
            self._whiteners[sid] = whitener(M)
        self._source_subjects = set(self._whiteners.keys())

    # ---- target calibration -------------------------------------------------
    def calibrate_trials(self, trials: list["Trial"]) -> None:
        if not trials:
            return
        target_id = trials[0].subject
        covs = np.stack([trial_cov(t.eeg, self.regularize) for t in trials])
        M = frechet_mean(covs)
        self._means[target_id] = M
        self._whiteners[target_id] = whitener(M)

    # ---- per-trial transform (with online update) --------------------------
    def transform_trial(self, trial: "Trial") -> "Trial":
        sid = trial.subject
        cov = trial_cov(trial.eeg, self.regularize)

        # Source subjects: use static per-subject whitener; no online update.
        if sid in self._source_subjects:
            W = self._whiteners[sid]
            from dataclasses import replace

            return replace(trial, eeg=whiten_eeg(trial.eeg, W))

        # Target subject path.
        if sid not in self._means:
            # First trial we've ever seen for this subject — seed M with it.
            self._means[sid] = cov
            self._whiteners[sid] = whitener(cov)
            W = self._whiteners[sid]
        else:
            W = self._whiteners[sid]

        eeg_aligned = whiten_eeg(trial.eeg, W)

        if self.update_online:
            # Geodesic step on the SPD manifold: M_new = γ_α(M, cov_t)
            M_new = geodesic_riemann(self._means[sid], cov, self.alpha)
            self._means[sid] = M_new
            self._whiteners[sid] = whitener(M_new)

        from dataclasses import replace

        return replace(trial, eeg=eeg_aligned)

    # ---- calibration transform (no online state advancement) ---------------
    def transform_calibration_trial(self, trial: "Trial") -> "Trial":
        """Whiten with the current per-subject estimator without advancing the
        online EMA. ``calibrate_trials`` has already folded these trials into
        the target's Frechet mean; running the geodesic update here too would
        double-count them and leave eval starting from an order-dependent state."""
        sid = trial.subject
        if sid in self._whiteners:
            W = self._whiteners[sid]
        else:
            # Defensive: calibrate_trials() should have populated this first.
            cov = trial_cov(trial.eeg, self.regularize)
            self._means[sid] = cov
            self._whiteners[sid] = whitener(cov)
            W = self._whiteners[sid]
        from dataclasses import replace

        return replace(trial, eeg=whiten_eeg(trial.eeg, W))
