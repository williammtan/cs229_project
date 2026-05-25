"""EEGNet (Lawhern 2018) backbone wrapper.

Reuses ``src.models.EEGNetReg`` and ``src.train.NeuralRegressor`` unchanged —
the wrapper handles trial-list ↔ concatenated-array plumbing only.
"""
from __future__ import annotations

import numpy as np

from src.backbones.base import BackboneBase
from src.core.registry import register
from src.data.way_eeg_gal import Trial, concat_trials
from src.models import EEGNetReg
from src.train import NeuralRegressor, TrainConfig


@register("backbone", "eegnet")
class EEGNetBackbone(BackboneBase):
    def __init__(
        self,
        n_channels: int = 32,
        n_out: int = 3,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_len: int = 50,
        dropout: float = 0.25,
        train: dict | None = None,
    ):
        self.n_channels = n_channels
        self.n_out = n_out
        self._model_kwargs = dict(
            f1=f1, d=d, f2=f2, kernel_len=kernel_len, dropout=dropout
        )
        self.cfg = TrainConfig(**(train or {}))

        def _factory(c: int, t: int, o: int):
            return EEGNetReg(n_channels=c, n_samples=t, n_out=o, **self._model_kwargs)

        self.regressor = NeuralRegressor(
            model_factory=_factory, cfg=self.cfg, n_channels=n_channels, n_out=n_out
        )

    def fit_source(self, trials: list[Trial]) -> None:
        eeg, _, vel = concat_trials(trials)
        self.regressor.fit(eeg, vel)

    def predict_trial(self, trial: Trial) -> np.ndarray:
        eeg = trial.eeg.T.astype(np.float32)
        return self.regressor.predict(eeg).astype(np.float32)
