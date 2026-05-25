"""Unit tests for src.heads.* — fit/predict shape and behaviour contracts."""
from __future__ import annotations

import numpy as np
import pytest

from src.heads.identity import IdentityHead
from src.heads.linear_probe import LinearProbeHead


@pytest.fixture
def synth_features():
    """Linear-y mappable features so the head can actually learn something."""
    rng = np.random.default_rng(0)
    n, d_in, d_out = 200, 16, 3
    X = rng.standard_normal((n, d_in)).astype(np.float32)
    W_true = rng.standard_normal((d_in, d_out)).astype(np.float32)
    y = X @ W_true + 0.01 * rng.standard_normal((n, d_out)).astype(np.float32)
    return X, y.astype(np.float32)


class TestLinearProbeHead:
    def test_predict_before_fit_raises(self, synth_features):
        head = LinearProbeHead()
        X, _ = synth_features
        with pytest.raises(AssertionError):
            head.predict(X)

    def test_fit_predict_shape(self, synth_features):
        head = LinearProbeHead()
        X, y = synth_features
        head.fit(X, y)
        out = head.predict(X)
        assert out.shape == y.shape
        assert out.dtype == np.float32

    def test_fits_linear_relationship_well(self, synth_features):
        head = LinearProbeHead(alpha=0.1)
        X, y = synth_features
        head.fit(X, y)
        pred = head.predict(X)
        # Per-axis Pearson r should be near 1.0 on near-noise-free training data.
        for k in range(3):
            r = np.corrcoef(pred[:, k], y[:, k])[0, 1]
            assert r > 0.95, f"axis {k}: r={r:.3f} too low for clean synthetic"

    def test_stores_target_normalization(self, synth_features):
        head = LinearProbeHead()
        X, y = synth_features
        head.fit(X, y)
        assert head._y_mean is not None and head._y_std is not None
        assert head._y_mean.shape == (3,)

    def test_constant_target_does_not_divide_by_zero(self):
        """Trivial edge case: when target std is exactly 0, the +1e-6 floor
        in the head must prevent a divide-by-zero in the StandardScaler step."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 8)).astype(np.float32)
        y = np.ones((50, 3), dtype=np.float32) * 7.5
        head = LinearProbeHead()
        head.fit(X, y)
        pred = head.predict(X)
        assert np.all(np.isfinite(pred))
        assert np.allclose(pred, 7.5, atol=1e-3)


class TestIdentityHead:
    def test_predict_passes_through(self):
        head = IdentityHead()
        X = np.random.default_rng(0).standard_normal((10, 3))
        head.fit(X, X)
        out = head.predict(X)
        assert np.array_equal(out, X)
