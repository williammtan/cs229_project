"""Unit tests for src.heads.* — softmax classifier behaviour contracts."""
from __future__ import annotations

import numpy as np
import pytest

from src.heads.identity import IdentityHead
from src.heads.softmax_probe import SoftmaxProbeHead


@pytest.fixture
def synth_features():
    """Linearly separable per-class features."""
    rng = np.random.default_rng(0)
    n_per_class, d, k = 60, 16, 4
    X = []
    y = []
    centers = rng.standard_normal((k, d)).astype(np.float32) * 2.0
    for ci in range(k):
        X.append(centers[ci][None, :] + 0.3 * rng.standard_normal((n_per_class, d)).astype(np.float32))
        y.append(np.full(n_per_class, ci, dtype=np.int64))
    return np.concatenate(X), np.concatenate(y)


class TestSoftmaxProbeHead:
    def test_predict_before_fit_raises(self, synth_features):
        head = SoftmaxProbeHead()
        X, _ = synth_features
        with pytest.raises(AssertionError):
            head.predict(X)

    def test_fit_predict_shapes(self, synth_features):
        head = SoftmaxProbeHead()
        X, y = synth_features
        head.fit(X, y)
        labels = head.predict(X)
        assert labels.shape == y.shape
        assert labels.dtype == np.int64
        proba = head.predict_proba(X)
        assert proba.shape == (len(X), 4)
        assert np.allclose(proba.sum(axis=-1), 1.0, atol=1e-5)

    def test_high_accuracy_on_separable_clusters(self, synth_features):
        head = SoftmaxProbeHead(C=1.0)
        X, y = synth_features
        head.fit(X, y)
        labels = head.predict(X)
        acc = float((labels == y).mean())
        assert acc > 0.9, f"separable synthetic should be easy; got {acc:.3f}"

    def test_calibrate_refits_on_source_plus_calib(self, synth_features):
        """Calibration should not discard source classes."""
        head = SoftmaxProbeHead(calibrate_on_calib=True)
        X, y = synth_features
        head.fit(X, y)
        # Two-class calib subset (labels 0 and 1 only), drawn from both.
        idx = np.concatenate([np.where(y == 0)[0][:10], np.where(y == 1)[0][:10]])
        head.calibrate(X[idx], y[idx])
        new_labels = head.predict(X)
        assert set(np.unique(new_labels).tolist()) == {0, 1, 2, 3}
        assert float((new_labels == y).mean()) > 0.9

    def test_calibrate_noop_with_too_few_samples(self, synth_features):
        head = SoftmaxProbeHead(calibrate_on_calib=True)
        X, y = synth_features
        head.fit(X, y)
        pred_before = head.predict(X[:10])
        head.calibrate(X[:1], y[:1])  # too few to refit -> no-op
        pred_after = head.predict(X[:10])
        assert np.array_equal(pred_before, pred_after)


class TestIdentityHead:
    def test_predict_passes_through(self):
        head = IdentityHead()
        X = np.random.default_rng(0).standard_normal((10, 3))
        head.fit(X, X)
        out = head.predict(X)
        assert np.array_equal(out, X)
