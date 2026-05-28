"""Shared pytest fixtures: synthetic Trials, SubjectData, SPD covariances.

Everything here is small and deterministic so smoke tests run in seconds and
unit tests don't depend on real .edf downloads.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.eegmmi import SubjectData, Trial

# ----- knobs the synthetic data fixtures honour -----------------------------
N_CHANNELS = 64
TRIAL_FS = 100
TRIAL_SAMPLES = 400  # 4 s — matches EEGMMI default epoch
N_CLASSES = 4
N_SUBJECTS_DEFAULT = 4
N_TRIALS_PER_SUBJECT_DEFAULT = 8


def _make_trial(
    subject: int,
    run: int,
    trial_idx: int,
    label: int,
    n_channels: int = N_CHANNELS,
    n_samples: int = TRIAL_SAMPLES,
    fs: int = TRIAL_FS,
    seed: int = 0,
) -> Trial:
    """Synthetic Trial with subject-specific covariance and class-specific mean.

    Each subject gets a random orthogonal mixing matrix (so Riemannian alignment
    has something to align) and each class adds a small class-specific offset
    so a linear classifier can learn the labels above chance.
    """
    rng = np.random.default_rng(seed)
    sub_rng = np.random.default_rng(1000 + subject)
    Q, _ = np.linalg.qr(sub_rng.standard_normal((n_channels, n_channels)))
    sources = rng.standard_normal((n_channels, n_samples)).astype(np.float32)
    eeg = (Q @ sources).astype(np.float32)

    # Class-conditional bias on a small number of "motor" channels.
    cls_rng = np.random.default_rng(2000 + label)
    bias = 0.5 * cls_rng.standard_normal((n_channels,)).astype(np.float32)
    eeg = eeg + bias[:, None]

    return Trial(
        eeg=eeg, label=label, fs=fs,
        subject=subject, run=run, trial_idx=trial_idx,
    )


def make_synth_dataset(
    n_subjects: int = N_SUBJECTS_DEFAULT,
    n_trials: int = N_TRIALS_PER_SUBJECT_DEFAULT,
    n_classes: int = N_CLASSES,
    n_channels: int = N_CHANNELS,
    n_samples: int = TRIAL_SAMPLES,
    fs: int = TRIAL_FS,
) -> dict[int, SubjectData]:
    # Use real EEGMMI imagery run IDs so leave-one-session-out splits work on
    # synthetic data. Classes 0/1 belong to L/R runs, classes 2/3 to hands/feet.
    LR_RUNS = (4, 8, 12)
    HF_RUNS = (6, 10, 14)
    out: dict[int, SubjectData] = {}
    for s in range(1, n_subjects + 1):
        trials = []
        for i in range(n_trials):
            label = i % n_classes
            # Cycle through the 3 sessions so every session has every class.
            session_idx = (i // n_classes) % 3
            run = (LR_RUNS, HF_RUNS)[label // 2][session_idx]
            trials.append(_make_trial(
                subject=s, run=run, trial_idx=i, label=label,
                n_channels=n_channels, n_samples=n_samples, fs=fs,
                seed=s * 100 + i,
            ))
        out[s] = SubjectData(subject=s, trials=trials)
    return out


# ===== fixtures =============================================================

@pytest.fixture
def make_trial():
    """Factory for one-off Trials inside a test."""
    return _make_trial


@pytest.fixture
def synth_dataset() -> dict[int, SubjectData]:
    """4 subjects × 8 trials × 4 classes. Enough for LOSO + K-trials sweep + RA fitting."""
    return make_synth_dataset()


@pytest.fixture
def two_subject_dataset() -> dict[int, SubjectData]:
    """Tiny 2-subject set for the cheapest smoke pass."""
    return make_synth_dataset(n_subjects=2, n_trials=8)


@pytest.fixture
def spd_matrix():
    """Factory: spd_matrix(n=4, cond=10.0, seed=0) -> (n, n) SPD with given condition number."""
    def _make(n: int = 4, cond: float = 10.0, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        eigs = np.geomspace(1.0, cond, n)
        return (Q @ np.diag(eigs) @ Q.T).astype(np.float64)
    return _make


@pytest.fixture
def ill_conditioned_eeg():
    """EEG with one near-zero-variance channel — the failure mode behind the
    'matrices must be positive definite' error from pyriemann."""
    rng = np.random.default_rng(42)
    eeg = rng.standard_normal((N_CHANNELS, TRIAL_SAMPLES)).astype(np.float32)
    eeg[0] *= 1e-8
    return eeg
