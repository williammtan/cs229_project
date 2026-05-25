"""Shared SPD utilities for the RA adapters."""
from __future__ import annotations

import numpy as np
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann


def trial_cov(eeg: np.ndarray, regularize: float = 1.0e-5) -> np.ndarray:
    """SCM with tiny ridge for numerical stability. eeg: (C, T) -> (C, C)."""
    C = eeg.shape[0]
    cov = (eeg @ eeg.T) / max(1, eeg.shape[-1] - 1)
    return cov + regularize * np.eye(C, dtype=cov.dtype)


def whitener(mean_cov: np.ndarray) -> np.ndarray:
    """Whitening operator W = M^{-1/2} so that W @ Σ @ W.T ≈ I when Σ ~ M."""
    return invsqrtm(mean_cov).astype(np.float32)


def whiten_eeg(eeg: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply (C, C) whitener to (C, T) EEG."""
    return (W @ eeg).astype(np.float32)


def frechet_mean(covs: np.ndarray, maxiter: int = 50, tol: float = 1.0e-6) -> np.ndarray:
    """Frechet (Karcher) mean on SPD manifold. covs: (N, C, C) -> (C, C)."""
    if len(covs) == 1:
        return covs[0]
    return mean_riemann(covs, tol=tol, maxiter=maxiter)
