"""Unit tests for adapter contracts: StaticRA, EMARA, NoopAdapter, AdapterStack."""
from __future__ import annotations

import numpy as np
import pytest

from src.adapters.base import AdapterBase, NoopAdapter
from src.adapters.composite import AdapterStack
from src.adapters.riemannian.ema import EMARA
from src.adapters.riemannian.static import StaticRA
from src.data.way_eeg_gal import Trial


def _trials_from(dataset, subject: int) -> list[Trial]:
    return dataset[subject].trials


# ============================================================================
# StaticRA
# ============================================================================

class TestStaticRA:
    def test_invalid_target_space_rejected(self):
        with pytest.raises(NotImplementedError):
            StaticRA(target_space="parallel_transport")

    def test_invalid_zero_shot_fallback_rejected(self):
        with pytest.raises(ValueError):
            StaticRA(zero_shot_fallback="bogus")

    def test_fit_source_records_per_subject_whitener(self, synth_dataset):
        ra = StaticRA()
        train_trials = _trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2)
        ra.fit_source_trials(train_trials)
        assert set(ra._whiteners.keys()) == {1, 2}
        for sid, W in ra._whiteners.items():
            assert W.shape == (32, 32)

    def test_calibrate_adds_target_whitener(self, synth_dataset):
        ra = StaticRA()
        ra.fit_source_trials(_trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2))
        ra.calibrate_trials(_trials_from(synth_dataset, 3))
        assert ra._target_calibrated is True
        assert 3 in ra._whiteners

    def test_transform_preserves_shape(self, synth_dataset):
        ra = StaticRA()
        ra.fit_source_trials(_trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2))
        out = ra.transform_trial(_trials_from(synth_dataset, 1)[0])
        assert out.eeg.shape == _trials_from(synth_dataset, 1)[0].eeg.shape
        assert out.subject == 1
        assert out.eeg.dtype == np.float32

    def test_zero_shot_fallback_per_trial(self, synth_dataset):
        """Unknown subject + per_trial fallback should still produce SPD-whitened EEG."""
        ra = StaticRA(zero_shot_fallback="per_trial")
        ra.fit_source_trials(_trials_from(synth_dataset, 1))
        unknown = _trials_from(synth_dataset, 2)[0]  # subject 2 was never fit
        out = ra.transform_trial(unknown)
        # Per-trial whitening reduces the trial's cov to approximately I.
        from src.adapters.riemannian._common import trial_cov

        out_cov = trial_cov(out.eeg, regularize=0.0)
        assert np.allclose(out_cov, np.eye(32), atol=0.5)

    def test_zero_shot_fallback_none_is_identity(self, synth_dataset):
        ra = StaticRA(zero_shot_fallback="none")
        ra.fit_source_trials(_trials_from(synth_dataset, 1))
        unknown = _trials_from(synth_dataset, 2)[0]
        out = ra.transform_trial(unknown)
        assert np.array_equal(out.eeg, unknown.eeg)


# ============================================================================
# EMARA — online Riemannian alignment
# ============================================================================

class TestEMARA:
    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValueError):
            EMARA(alpha=0.0)
        with pytest.raises(ValueError):
            EMARA(alpha=1.5)

    def test_fit_source_per_subject(self, synth_dataset):
        ema = EMARA()
        ema.fit_source_trials(_trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2))
        assert ema._source_subjects == {1, 2}

    def test_first_target_trial_seeds_mean(self, synth_dataset):
        ema = EMARA()
        ema.fit_source_trials(_trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2))
        # No explicit calibration — first target trial should seed M.
        target_trial = _trials_from(synth_dataset, 3)[0]
        out = ema.transform_trial(target_trial)
        assert out.eeg.shape == target_trial.eeg.shape
        assert 3 in ema._means

    def test_online_update_changes_whitener(self, synth_dataset):
        ema = EMARA(alpha=0.3, update_online=True)
        ema.fit_source_trials(_trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2))
        target_trials = _trials_from(synth_dataset, 3)
        ema.transform_trial(target_trials[0])
        W_after_1 = ema._whiteners[3].copy()
        ema.transform_trial(target_trials[1])
        W_after_2 = ema._whiteners[3].copy()
        # Online update means whitener must move.
        assert not np.allclose(W_after_1, W_after_2)

    def test_no_online_update_freezes_whitener(self, synth_dataset):
        ema = EMARA(alpha=0.3, update_online=False)
        ema.fit_source_trials(_trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2))
        ema.calibrate_trials(_trials_from(synth_dataset, 3))
        target_trials = _trials_from(synth_dataset, 3)
        ema.transform_trial(target_trials[0])
        W1 = ema._whiteners[3].copy()
        ema.transform_trial(target_trials[1])
        W2 = ema._whiteners[3].copy()
        assert np.allclose(W1, W2)

    def test_calibrate_overrides_seed(self, synth_dataset):
        ema = EMARA()
        ema.fit_source_trials(_trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2))
        ema.calibrate_trials(_trials_from(synth_dataset, 3)[:2])
        assert 3 in ema._means


# ============================================================================
# RA-EMA regression: dead-channel input used to surface the
# "Matrices must be positive definite" failure on loso-P1.
# ============================================================================

class TestEMARARegression:
    """Regression tests for the loso-P1 SPD failure.

    The original error chain:
      1. trial_cov of a near-rank-deficient EEG → barely-SPD covariance.
      2. EMA geodesic step on a barely-SPD M + new ill-conditioned cov →
         floating-point eigenvalue dips below zero.
      3. pyriemann's invsqrtm raises "Matrices must be positive definite".

    These tests assert that the adapter does NOT crash on degenerate inputs:
    one near-rank-deficient channel (the most common cause).
    """

    def _trial(self, subject, idx, eeg):
        n = eeg.shape[-1]
        kin = np.zeros((3, n), dtype=np.float32)
        return Trial(
            eeg=eeg.astype(np.float32), kin=kin, vel=kin, fs=100,
            subject=subject, series=1, trial_idx=idx,
        )

    def test_dead_channel_does_not_crash_static_ra(self, ill_conditioned_eeg):
        ra = StaticRA(regularize=1e-5)
        trials = [self._trial(1, i, ill_conditioned_eeg) for i in range(3)]
        ra.fit_source_trials(trials)
        out = ra.transform_trial(trials[0])
        assert np.all(np.isfinite(out.eeg))

    def test_dead_channel_does_not_crash_ema_ra(self, ill_conditioned_eeg):
        """Reproduces (or guards against) the loso-P1 ra_ema failure path."""
        ema = EMARA(alpha=0.1, regularize=1e-5, update_online=True)
        source_trials = [self._trial(1, i, ill_conditioned_eeg) for i in range(3)]
        ema.fit_source_trials(source_trials)
        target_trials = [self._trial(99, i, ill_conditioned_eeg) for i in range(3)]
        for t in target_trials:
            out = ema.transform_trial(t)
            assert np.all(np.isfinite(out.eeg)), \
                "EMA-RA produced NaN/inf on dead-channel input — root cause of loso-P1 SPD failure"

    def test_mixed_subjects_with_one_degenerate(self, ill_conditioned_eeg, make_trial):
        """One subject's trials have a dead channel; others are healthy."""
        ema = EMARA(alpha=0.1, regularize=1e-5)
        healthy = [make_trial(subject=2, series=1, trial_idx=i, seed=i) for i in range(3)]
        degenerate = [self._trial(1, i, ill_conditioned_eeg) for i in range(3)]
        ema.fit_source_trials(degenerate + healthy)
        target = make_trial(subject=99, series=1, trial_idx=0, seed=42)
        out = ema.transform_trial(target)
        assert np.all(np.isfinite(out.eeg))


# ============================================================================
# AdapterStack
# ============================================================================

class TestAdapterStack:
    def test_empty_stack_is_identity(self, synth_dataset):
        stack = AdapterStack([])
        trial = _trials_from(synth_dataset, 1)[0]
        out = stack.transform_trial(trial)
        assert np.array_equal(out.eeg, trial.eeg)

    def test_noop_stack_does_nothing(self, synth_dataset):
        stack = AdapterStack([NoopAdapter()])
        trial = _trials_from(synth_dataset, 1)[0]
        out = stack.transform_trial(trial)
        assert np.array_equal(out.eeg, trial.eeg)

    def test_stack_propagates_trial_transforms_in_order(self, synth_dataset):
        """Stack of [RA, noop] should equal RA-alone for a single trial path."""
        ra_only = StaticRA()
        stacked = AdapterStack([StaticRA(), NoopAdapter()])
        train = _trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2)
        ra_only.fit_source_trials(train)
        stacked.fit_source_trials(train)
        trial = _trials_from(synth_dataset, 1)[0]
        a = ra_only.transform_trial(trial)
        b = stacked.transform_trial(trial)
        assert np.allclose(a.eeg, b.eeg, atol=1e-6)

    def test_calibrate_dispatches_to_children(self, synth_dataset):
        class Counter(AdapterBase):
            def __init__(self):
                self.n_calibrate = 0
            def calibrate_trials(self, trials):
                self.n_calibrate += 1

        c1, c2 = Counter(), Counter()
        stack = AdapterStack([c1, c2])
        stack.calibrate_trials(_trials_from(synth_dataset, 1))
        assert c1.n_calibrate == 1
        assert c2.n_calibrate == 1


# ============================================================================
# NoopAdapter / AdapterBase defaults
# ============================================================================

class TestNoopAdapter:
    def test_transform_trial_returns_same(self, synth_dataset):
        a = NoopAdapter()
        t = _trials_from(synth_dataset, 1)[0]
        assert a.transform_trial(t) is t

    def test_feature_transform_identity(self):
        a = NoopAdapter()
        feats = np.random.default_rng(0).standard_normal((10, 4))
        assert np.array_equal(a.transform(feats), feats)
