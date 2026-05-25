"""Unit tests for src.adapters.riemannian._common — the SPD math primitives."""
from __future__ import annotations

import numpy as np
import pytest

from src.adapters.riemannian._common import (
    frechet_mean,
    trial_cov,
    whiten_eeg,
    whitener,
)


def _is_symmetric(M: np.ndarray, atol: float = 1e-6) -> bool:
    return np.allclose(M, M.T, atol=atol)


def _is_positive_definite(M: np.ndarray, atol: float = 1e-10) -> bool:
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return False
    eigs = np.linalg.eigvalsh((M + M.T) / 2)
    return bool(np.all(eigs > -atol))


# ----------------------------------------------------------------------------
# trial_cov
# ----------------------------------------------------------------------------

class TestTrialCov:
    def test_shape(self):
        eeg = np.random.default_rng(0).standard_normal((32, 500)).astype(np.float32)
        cov = trial_cov(eeg)
        assert cov.shape == (32, 32)

    def test_is_spd_on_normal_eeg(self):
        eeg = np.random.default_rng(0).standard_normal((16, 500)).astype(np.float32)
        cov = trial_cov(eeg, regularize=1e-5)
        assert _is_symmetric(cov)
        assert _is_positive_definite(cov)

    def test_regularize_floor_when_eeg_is_zero(self):
        """A zero-input trial would otherwise produce a zero covariance.
        Regularize must keep it SPD."""
        eeg = np.zeros((8, 500), dtype=np.float32)
        cov = trial_cov(eeg, regularize=1e-3)
        assert _is_positive_definite(cov)
        # Eigenvalues should equal the regularization constant.
        eigs = np.linalg.eigvalsh(cov)
        assert np.allclose(eigs, 1e-3, atol=1e-6)

    def test_dead_channel_still_spd(self, ill_conditioned_eeg):
        """A near-zero-variance channel should still produce SPD with regularize."""
        cov = trial_cov(ill_conditioned_eeg, regularize=1e-5)
        assert _is_positive_definite(cov)


# ----------------------------------------------------------------------------
# whitener / whiten_eeg
# ----------------------------------------------------------------------------

class TestWhitener:
    def test_whitener_inverts_to_identity(self, spd_matrix):
        M = spd_matrix(n=8, cond=20.0, seed=1)
        W = whitener(M)
        # W M W^T should be I.
        WI = W @ M @ W.T
        assert np.allclose(WI, np.eye(8), atol=1e-4)

    def test_whitener_returns_float32(self, spd_matrix):
        M = spd_matrix(n=4)
        W = whitener(M)
        assert W.dtype == np.float32

    def test_whiten_eeg_shape_preserved(self, spd_matrix):
        eeg = np.random.default_rng(0).standard_normal((8, 200)).astype(np.float32)
        M = spd_matrix(n=8)
        W = whitener(M)
        out = whiten_eeg(eeg, W)
        assert out.shape == eeg.shape
        assert out.dtype == np.float32

    def test_whiten_eeg_decorrelates(self):
        """Whitening with the trial's own cov should make its cov approximately I."""
        eeg = np.random.default_rng(0).standard_normal((6, 1000)).astype(np.float32)
        cov = trial_cov(eeg, regularize=1e-5)
        W = whitener(cov)
        out = whiten_eeg(eeg, W)
        out_cov = trial_cov(out, regularize=0.0)
        # Reasonable tolerance: 1000 samples for 6 channels.
        assert np.allclose(out_cov, np.eye(6), atol=0.15)


# ----------------------------------------------------------------------------
# frechet_mean
# ----------------------------------------------------------------------------

class TestFrechetMean:
    def test_singleton_returns_input(self, spd_matrix):
        M = spd_matrix(n=4)
        out = frechet_mean(M[None])
        assert np.allclose(out, M)

    def test_two_identical_returns_same(self, spd_matrix):
        M = spd_matrix(n=4, seed=0)
        covs = np.stack([M, M])
        out = frechet_mean(covs)
        assert np.allclose(out, M, atol=1e-5)

    def test_mean_is_spd(self, spd_matrix):
        covs = np.stack([spd_matrix(n=6, cond=10.0, seed=i) for i in range(5)])
        out = frechet_mean(covs)
        assert _is_symmetric(out)
        assert _is_positive_definite(out)

    def test_mean_close_to_arithmetic_mean_for_close_inputs(self, spd_matrix):
        """When covariances are nearly identical, Frechet mean ≈ arithmetic mean."""
        base = spd_matrix(n=4, cond=2.0, seed=0)
        rng = np.random.default_rng(0)
        # Tiny perturbations preserve closeness.
        covs = []
        for i in range(4):
            delta = 1e-3 * rng.standard_normal((4, 4))
            delta = (delta + delta.T) / 2
            covs.append(base + delta)
        out = frechet_mean(np.stack(covs))
        arith = np.mean(covs, axis=0)
        assert np.allclose(out, arith, atol=1e-2)


# ----------------------------------------------------------------------------
# Convergence-warning regression
# ----------------------------------------------------------------------------

class TestConvergenceBehavior:
    def test_does_not_raise_on_widely_spread_covs(self, spd_matrix):
        """pyriemann emits a UserWarning when Karcher iteration hits maxiter, but
        the call must still return a valid SPD matrix. We exercise the path that
        previously surfaced 'Convergence not reached' warnings."""
        covs = np.stack([
            spd_matrix(n=8, cond=cond, seed=i)
            for i, cond in enumerate([1.0, 50.0, 200.0, 500.0])
        ])
        out = frechet_mean(covs, maxiter=10)  # deliberately low to provoke warning
        assert out.shape == (8, 8)
        assert _is_symmetric(out, atol=1e-4)
        assert _is_positive_definite(out)
