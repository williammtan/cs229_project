"""Wrap the two linear baselines as Backbones (monolithic).

Both reuse the existing implementations in ``src.models`` so the math is
identical — this file only handles the trial-list ↔ concatenated-array
plumbing and registers the classes for Hydra.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.backbones.base import BackboneBase
from src.core.registry import register
from src.data.way_eeg_gal import Trial, concat_trials
from src.models import BradberrymLR, RidgeBandPower


@register("backbone", "bradberry_mlr")
class BradberryMLRBackbone(BackboneBase):
    def __init__(self, lags: Sequence[int] = (0, 5, 10, 15, 20, 25), alpha: float = 1.0):
        self.model = BradberrymLR(lags=tuple(lags), alpha=alpha)

    def fit_source(self, trials: list[Trial]) -> None:
        eeg, _, vel = concat_trials(trials)
        self.model.fit(eeg, vel)

    def predict_trial(self, trial: Trial) -> np.ndarray:
        eeg = trial.eeg.T.astype(np.float32)  # (T, 32)
        return self.model.predict(eeg).astype(np.float32)


@register("backbone", "ridge_bandpower")
class RidgeBandPowerBackbone(BackboneBase):
    def __init__(
        self,
        fs: int = 100,
        bands: Sequence[Sequence[float]] = ((8, 13), (13, 30), (30, 45)),
        win_ms: int = 500,
        hop_ms: int = 100,
        alpha: float = 10.0,
    ):
        self.model = RidgeBandPower(
            fs=fs,
            bands=tuple((float(lo), float(hi)) for lo, hi in bands),
            win_ms=win_ms,
            hop_ms=hop_ms,
            alpha=alpha,
        )

    def fit_source(self, trials: list[Trial]) -> None:
        eeg, _, vel = concat_trials(trials)
        self.model.fit(eeg, vel)

    def predict_trial(self, trial: Trial) -> np.ndarray:
        eeg = trial.eeg.T.astype(np.float32)
        return self.model.predict(eeg).astype(np.float32)
