"""ShallowConvNet (Schirrmeister 2017) backbone wrapper for classification."""
from __future__ import annotations

import numpy as np

from src.backbones.base import BackboneBase
from src.core.registry import register
from src.data.eegmmi import Trial, stack_eeg_and_labels
from src.models import ShallowConvNetClf
from src.train import NeuralClassifier, TrainConfig


@register("backbone", "shallowconvnet")
class ShallowConvNetBackbone(BackboneBase):
    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
        train: dict | None = None,
    ):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self._model_kwargs = dict(
            n_filters=n_filters,
            temporal_kernel=temporal_kernel,
            pool_size=pool_size,
            pool_stride=pool_stride,
            dropout=dropout,
        )
        self.cfg = TrainConfig(**(train or {}))

        def _factory(c: int, t: int, k: int):
            return ShallowConvNetClf(n_channels=c, n_samples=t, n_classes=k, **self._model_kwargs)

        self.clf = NeuralClassifier(
            model_factory=_factory, cfg=self.cfg,
            n_channels=n_channels, n_classes=n_classes,
        )

    def fit_source(self, trials: list[Trial]) -> None:
        X, y = stack_eeg_and_labels(trials)
        self.clf.fit(X, y)

    def predict_trial(self, trial: Trial) -> np.ndarray:
        proba = self.clf.predict_proba(trial.eeg.astype(np.float32)[None, ...])[0]
        return proba.astype(np.float32)
