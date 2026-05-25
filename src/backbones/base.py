"""Backbone protocol.

Two flow modes:

* **Monolithic** backbones (linear/ConvNet baselines) implement ``predict_trial``
  directly — raw EEG → per-sample velocity in one call. They live on the
  ``Pipeline._monolithic`` short-circuit path; no head or adapters needed.

* **Composed** backbones (foundation models) implement ``encode_trial`` →
  per-window features + per-window targets, plus ``upsample_to_per_sample`` that
  maps per-window head outputs back to the per-sample grid. The Pipeline runs
  ``encode → adapters.transform → head.predict → upsample``. Windowing knowledge
  stays with the backbone (it knows its own input resolution and patch size).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.data.way_eeg_gal import Trial


class BackboneBase:
    """Subclass and override either ``predict_trial`` (monolithic) or both
    ``encode_trial`` + ``upsample_to_per_sample`` (composed)."""

    def fit_source(self, trials: list["Trial"]) -> None:
        raise NotImplementedError

    def predict_trial(self, trial: "Trial") -> np.ndarray:
        """Per-sample velocity, shape (T, n_out). Monolithic path."""
        raise NotImplementedError(
            f"{type(self).__name__} is composed; add a head + use encode_trial."
        )

    def encode_trial(self, trial: "Trial") -> tuple[np.ndarray, np.ndarray]:
        """Return (features (N_win, D), targets (N_win, n_out)). Composed path."""
        raise NotImplementedError(
            f"{type(self).__name__} is monolithic; use predict_trial instead."
        )

    def upsample_to_per_sample(
        self, y_windows: np.ndarray, trial: "Trial"
    ) -> np.ndarray:
        """Map per-window predictions back to the trial's per-sample grid.
        Required for the composed path."""
        raise NotImplementedError
