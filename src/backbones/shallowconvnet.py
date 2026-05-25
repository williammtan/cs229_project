"""ShallowConvNet (Schirrmeister 2017) backbone wrapper."""
from __future__ import annotations

import numpy as np

from src.backbones.base import BackboneBase
from src.core.registry import register
from src.data.way_eeg_gal import Trial, concat_trials
from src.models import ShallowConvNetReg
from src.train import NeuralRegressor, TrainConfig


@register("backbone", "shallowconvnet")
class ShallowConvNetBackbone(BackboneBase):
    def __init__(
        self,
        n_channels: int = 32,
        n_out: int = 3,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
        train: dict | None = None,
    ):
        self.n_channels = n_channels
        self.n_out = n_out
        self._model_kwargs = dict(
            n_filters=n_filters,
            temporal_kernel=temporal_kernel,
            pool_size=pool_size,
            pool_stride=pool_stride,
            dropout=dropout,
        )
        self.cfg = TrainConfig(**(train or {}))

        def _factory(c: int, t: int, o: int):
            return ShallowConvNetReg(n_channels=c, n_samples=t, n_out=o, **self._model_kwargs)

        self.regressor = NeuralRegressor(
            model_factory=_factory, cfg=self.cfg, n_channels=n_channels, n_out=n_out
        )

    def fit_source(self, trials: list[Trial]) -> None:
        eeg, _, vel = concat_trials(trials)
        self.regressor.fit(eeg, vel)

    def predict_trial(self, trial: Trial) -> np.ndarray:
        eeg = trial.eeg.T.astype(np.float32)
        return self.regressor.predict(eeg).astype(np.float32)
