"""Stack multiple adapters: each subsequent adapter sees the previous one's output.

Trial-space transforms run in list order before encoding. Feature-space
transforms run in list order after encoding. Both orderings are independent;
an adapter that's "input-only" is silent on the feature path and vice versa.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.adapters.base import AdapterBase

if TYPE_CHECKING:
    from src.data.eegmmi import Trial


class AdapterStack(AdapterBase):
    def __init__(self, adapters: list[AdapterBase]):
        self.adapters = adapters

    # ---- trial-space ---------------------------------------------------------
    def fit_source_trials(self, trials: list["Trial"]) -> None:
        for a in self.adapters:
            a.fit_source_trials(trials)
            trials = [a.transform_trial(t) for t in trials]

    def calibrate_trials(self, trials: list["Trial"]) -> None:
        for a in self.adapters:
            a.calibrate_trials(trials)
            # Thread post-calibration trials through without advancing online state.
            trials = [a.transform_calibration_trial(t) for t in trials]

    def update_trial(self, trial: "Trial") -> None:
        for a in self.adapters:
            a.update_trial(trial)
            trial = a.transform_trial(trial)

    def transform_trial(self, trial: "Trial") -> "Trial":
        for a in self.adapters:
            trial = a.transform_trial(trial)
        return trial

    def transform_calibration_trial(self, trial: "Trial") -> "Trial":
        for a in self.adapters:
            trial = a.transform_calibration_trial(trial)
        return trial

    # ---- feature-space -------------------------------------------------------
    def fit_source(self, feats: np.ndarray, targets: np.ndarray) -> None:
        for a in self.adapters:
            a.fit_source(feats, targets)
            feats = a.transform(feats)

    def calibrate(self, feats: np.ndarray, targets: np.ndarray) -> None:
        for a in self.adapters:
            a.calibrate(feats, targets)
            feats = a.transform(feats)

    def update(self, feat: np.ndarray, target: np.ndarray) -> None:
        for a in self.adapters:
            a.update(feat, target)
            feat = a.transform(feat[None])[0] if feat.ndim == 1 else a.transform(feat)

    def transform(self, feats: np.ndarray) -> np.ndarray:
        for a in self.adapters:
            feats = a.transform(feats)
        return feats
