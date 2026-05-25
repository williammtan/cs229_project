"""Shared pytest fixtures: synthetic Trials, SubjectData, SPD covariances.

Everything here is small and deterministic so smoke tests run in seconds and
unit tests don't depend on real .mat downloads.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.way_eeg_gal import SubjectData, Trial

# ----- knobs the synthetic data fixtures honour -----------------------------
N_CHANNELS = 32
TRIAL_FS = 100
TRIAL_SAMPLES = 600  # 6 seconds — comfortably longer than any backbone window
N_SUBJECTS_DEFAULT = 3
N_TRIALS_PER_SUBJECT_DEFAULT = 6


def _make_trial(
    subject: int,
    series: int,
    trial_idx: int,
    n_channels: int = N_CHANNELS,
    n_samples: int = TRIAL_SAMPLES,
    fs: int = TRIAL_FS,
    seed: int = 0,
) -> Trial:
    """A synthetic Trial with subject-specific covariance structure.

    To make Riemannian alignment have something to align, each subject gets a
    random orthogonal mixing matrix applied to white noise — subjects differ in
    channel covariance but share the same underlying source distribution.
    """
    rng = np.random.default_rng(seed)
    # subject-specific mixing matrix (orthogonal), deterministic per subject id
    sub_rng = np.random.default_rng(1000 + subject)
    Q, _ = np.linalg.qr(sub_rng.standard_normal((n_channels, n_channels)))
    sources = rng.standard_normal((n_channels, n_samples)).astype(np.float32)
    eeg = (Q @ sources).astype(np.float32)

    # Kinematic signal: smooth sinusoid + noise. Velocity is the derivative.
    t = np.arange(n_samples) / fs
    phase = trial_idx * 0.3
    kin = np.stack([
        np.sin(2 * np.pi * 0.5 * t + phase),
        np.cos(2 * np.pi * 0.5 * t + phase),
        np.sin(2 * np.pi * 1.0 * t + phase),
    ]).astype(np.float32) + 0.05 * rng.standard_normal((3, n_samples)).astype(np.float32)
    vel = np.gradient(kin, 1.0 / fs, axis=-1).astype(np.float32)
    return Trial(
        eeg=eeg, kin=kin, vel=vel, fs=fs,
        subject=subject, series=series, trial_idx=trial_idx,
    )


def make_synth_dataset(
    n_subjects: int = N_SUBJECTS_DEFAULT,
    n_trials: int = N_TRIALS_PER_SUBJECT_DEFAULT,
    n_channels: int = N_CHANNELS,
    n_samples: int = TRIAL_SAMPLES,
    fs: int = TRIAL_FS,
) -> dict[int, SubjectData]:
    out: dict[int, SubjectData] = {}
    for s in range(1, n_subjects + 1):
        trials = [
            _make_trial(
                subject=s, series=1, trial_idx=i,
                n_channels=n_channels, n_samples=n_samples, fs=fs,
                seed=s * 100 + i,
            )
            for i in range(n_trials)
        ]
        out[s] = SubjectData(subject=s, trials=trials)
    return out


# ===== fixtures =============================================================

@pytest.fixture
def make_trial():
    """Factory for one-off Trials inside a test."""
    return _make_trial


@pytest.fixture
def synth_dataset() -> dict[int, SubjectData]:
    """3 subjects × 6 trials. Enough for LOSO + kmin sweep + RA fitting."""
    return make_synth_dataset()


@pytest.fixture
def two_subject_dataset() -> dict[int, SubjectData]:
    """Tiny 2-subject set for the cheapest smoke pass."""
    return make_synth_dataset(n_subjects=2, n_trials=4)


@pytest.fixture
def spd_matrix():
    """Factory: spd_matrix(n=4, cond=10.0, seed=0) -> (n, n) SPD with given condition number."""
    def _make(n: int = 4, cond: float = 10.0, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        # Eigenvalues geometrically spaced between 1 and cond.
        eigs = np.geomspace(1.0, cond, n)
        return (Q @ np.diag(eigs) @ Q.T).astype(np.float64)
    return _make


@pytest.fixture
def ill_conditioned_eeg():
    """EEG with one near-zero-variance channel — the failure mode behind the
    'matrices must be positive definite' error from pyriemann."""
    rng = np.random.default_rng(42)
    eeg = rng.standard_normal((N_CHANNELS, TRIAL_SAMPLES)).astype(np.float32)
    eeg[0] *= 1e-8  # channel 0 is effectively dead
    return eeg
