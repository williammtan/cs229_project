"""Riemannian + tangent-space logistic regression baseline.

Standard pipeline (Barachant et al.):
  EEG (C, T) → covariance via OAS shrinkage → tangent-space projection at the
  Frechet mean of training covariances → multinomial logistic regression.

Tangent-space LR is the strongest classical baseline on MI tasks and is what
all the FM-promising-only-on-paper experiments need to beat.
"""
from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline

from src.backbones.base import BackboneBase
from src.core.registry import register
from src.data.eegmmi import Trial, stack_eeg_and_labels


@register("backbone", "riemann_lr")
class RiemannLRBackbone(BackboneBase):
    """Covariances → TangentSpace → multinomial LogisticRegression."""

    def __init__(
        self,
        n_classes: int = 4,
        estimator: str = "oas",
        C: float = 1.0,
        max_iter: int = 2000,
    ):
        self.n_classes = n_classes
        self.estimator = estimator
        self.C = C
        self.max_iter = max_iter
        self.pipeline: SkPipeline | None = None

    def _build_pipeline(self) -> SkPipeline:
        return SkPipeline([
            ("cov", Covariances(estimator=self.estimator)),
            ("ts", TangentSpace(metric="riemann")),
            ("lr", LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")),
        ])

    def fit_source(self, trials: list[Trial]) -> None:
        X, y = stack_eeg_and_labels(trials)
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X.astype(np.float64), y)

    def predict_trial(self, trial: Trial) -> np.ndarray:
        assert self.pipeline is not None, "fit first"
        X = trial.eeg.astype(np.float64)[None, ...]
        proba = self.pipeline.predict_proba(X)[0]
        return proba.astype(np.float32)
