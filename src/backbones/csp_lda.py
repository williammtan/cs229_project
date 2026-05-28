"""CSP + LDA classical motor-imagery baseline.

The reviewer-mandatory floor on 4-class MI: ``mne.decoding.CSP`` extracts
discriminative spatial filters, log-variance features feed an LDA classifier.
Standard hyperparameters; we add ``shrinkage="auto"`` to the LDA to keep it
stable when CSP feature dimensionality is close to the number of training trials.
"""
from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline as SkPipeline

from src.backbones.base import BackboneBase
from src.core.registry import register
from src.data.eegmmi import Trial, stack_eeg_and_labels


@register("backbone", "csp_lda")
class CSPLDABackbone(BackboneBase):
    """CSP spatial filters → log-variance → LDA classification."""

    def __init__(
        self,
        n_classes: int = 4,
        n_components: int = 8,
        reg: str | float | None = "ledoit_wolf",
        shrinkage: str | float = "auto",
    ):
        self.n_classes = n_classes
        self.n_components = n_components
        self.reg = reg
        self.shrinkage = shrinkage
        self.pipeline: SkPipeline | None = None

    def _build_pipeline(self) -> SkPipeline:
        csp = CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=True,
            norm_trace=False,
            transform_into="average_power",
        )
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.shrinkage)
        return SkPipeline([("csp", csp), ("lda", lda)])

    def fit_source(self, trials: list[Trial]) -> None:
        X, y = stack_eeg_and_labels(trials)
        self.pipeline = self._build_pipeline()
        # MNE's CSP wants float64.
        self.pipeline.fit(X.astype(np.float64), y)

    def predict_trial(self, trial: Trial) -> np.ndarray:
        assert self.pipeline is not None, "fit first"
        X = trial.eeg.astype(np.float64)[None, ...]
        proba = self.pipeline.predict_proba(X)[0]
        return proba.astype(np.float32)
