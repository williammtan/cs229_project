"""EEGNet (Lawhern 2018) backbone wrapper for per-trial classification.

Reuses ``src.models.EEGNetClf`` and ``src.train.NeuralClassifier``. The wrapper
handles trial-list ↔ stacked-tensor plumbing only.
"""
from __future__ import annotations

import numpy as np

from src.backbones.base import BackboneBase
from src.core.registry import register
from src.data.eegmmi import Trial, stack_eeg_and_labels
from src.models import EEGNetClf
from src.train import NeuralClassifier, TrainConfig


@register("backbone", "eegnet")
class EEGNetBackbone(BackboneBase):
    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_len: int = 64,
        dropout: float = 0.5,
        train: dict | None = None,
    ):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self._model_kwargs = dict(
            f1=f1, d=d, f2=f2, kernel_len=kernel_len, dropout=dropout
        )
        self.cfg = TrainConfig(**(train or {}))

        def _factory(c: int, t: int, k: int):
            return EEGNetClf(n_channels=c, n_samples=t, n_classes=k, **self._model_kwargs)

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
