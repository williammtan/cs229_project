"""Integration tests for the vendored CLD convex NN head."""
from __future__ import annotations

import numpy as np
import pytest

from src.heads.convex_nn import ConvexNNHead


def _cluster_data(
    n_per_class: int = 12,
    n_classes: int = 3,
    n_features: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    centers = np.eye(n_classes, n_features, dtype=np.float32) * 5.0
    X, y = [], []
    for cls in range(n_classes):
        X.append(centers[cls] + 0.1 * rng.standard_normal((n_per_class, n_features)))
        y.append(np.full(n_per_class, cls, dtype=np.int64))
    return np.vstack(X).astype(np.float32), np.concatenate(y)


def _small_head() -> ConvexNNHead:
    return ConvexNNHead(
        n_classes=3,
        n_neurons=6,
        beta=1.0e-3,
        rho=0.1,
        admm_iters=3,
        pcg_iters=8,
        rank=4,
        seed=0,
    )


def test_convex_nn_matches_cld_weight_shapes_and_predicts():
    X, y = _cluster_data()
    head = _small_head()

    head.fit(X, y)
    proba = head.predict_proba(X)
    pred = head.predict(X)

    assert head._model.theta1.shape == (3, X.shape[1], 2 * head.n_neurons)
    assert head._model.theta2.shape == (3, 2 * head.n_neurons)
    assert head._model.d_diags.shape == (len(X), head.n_neurons)
    assert proba.shape == (len(X), 3)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1.0e-5)
    assert float((pred == y).mean()) > 0.9


def test_convex_nn_calibration_keeps_source_classes():
    X, y = _cluster_data()
    head = _small_head()
    head.fit(X, y)

    calib_idx = np.concatenate([np.where(y == 0)[0][:3], np.where(y == 1)[0][:3]])
    head.calibrate(X[calib_idx], y[calib_idx])
    pred = head.predict(X)

    assert set(np.unique(pred).tolist()) == {0, 1, 2}
    assert float((pred == y).mean()) > 0.9


def test_convex_nn_rejects_labels_outside_class_range():
    X, y = _cluster_data(n_classes=3)
    head = ConvexNNHead(n_classes=2, n_neurons=2, rank=2)

    with pytest.raises(ValueError, match=r"targets must be in \[0, 2\)"):
        head.fit(X, y)
