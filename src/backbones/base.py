"""Backbone protocol for per-trial classification.

Two flow modes:

* **Monolithic** backbones (EEGNet / ShallowConvNet / CSP+LDA / Riemannian+LR)
  implement ``predict_trial`` directly — raw EEG → per-class probabilities.
  They live on the ``Pipeline._monolithic`` short-circuit path.

* **Composed** backbones (foundation models) implement ``encode_trial`` →
  one D-dim embedding per trial. The Pipeline runs
  ``encode_trial → adapters.transform → head.predict_proba``.

There is no per-sample upsampling on the classification branch: one feature
vector and one prediction per trial. Window-pooling lives inside the backbone.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.data.eegmmi import Trial


class BackboneBase:
    """Subclass and override either ``predict_trial`` (monolithic) or
    ``encode_trial`` (composed)."""

    def fit_source(self, trials: list["Trial"]) -> None:
        raise NotImplementedError

    def predict_trial(self, trial: "Trial") -> np.ndarray:
        """Return (n_classes,) class probabilities. Monolithic path."""
        raise NotImplementedError(
            f"{type(self).__name__} is composed; add a head + use encode_trial."
        )

    def encode_trial(self, trial: "Trial") -> np.ndarray:
        """Return (D,) per-trial embedding. Composed path."""
        raise NotImplementedError(
            f"{type(self).__name__} is monolithic; use predict_trial instead."
        )
