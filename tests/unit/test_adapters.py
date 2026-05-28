"""Unit tests for adapter contracts: StaticRA, EMARA, NoopAdapter, AdapterStack."""
from __future__ import annotations

import numpy as np
import pytest

from src.adapters.base import AdapterBase, NoopAdapter
from src.adapters.composite import AdapterStack
from src.adapters.riemannian.ema import EMARA
from src.adapters.riemannian.static import StaticRA
from src.data.eegmmi import Trial


N_CH = 64


def _trials_from(dataset, subject: int) -> list[Trial]:
    return dataset[subject].trials


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
        for W in ra._whiteners.values():
            assert W.shape == (N_CH, N_CH)

    def test_calibrate_adds_target_whitener(self, synth_dataset):
        ra = StaticRA()
        ra.fit_source_trials(_trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2))
        ra.calibrate_trials(_trials_from(synth_dataset, 3))
        assert ra._target_calibrated is True
        assert 3 in ra._whiteners

    def test_transform_preserves_shape_and_label(self, synth_dataset):
        ra = StaticRA()
        ra.fit_source_trials(_trials_from(synth_dataset, 1) + _trials_from(synth_dataset, 2))
        src = _trials_from(synth_dataset, 1)[0]
        out = ra.transform_trial(src)
        assert out.eeg.shape == src.eeg.shape
        assert out.subject == 1
        assert out.eeg.dtype == np.float32
        assert out.label == src.label

    def test_zero_shot_fallback_none_is_identity(self, synth_dataset):
        ra = StaticRA(zero_shot_fallback="none")
        ra.fit_source_trials(_trials_from(synth_dataset, 1))
        unknown = _trials_from(synth_dataset, 2)[0]
        out = ra.transform_trial(unknown)
        assert np.array_equal(out.eeg, unknown.eeg)


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


class TestEMARARegression:
    """Regression: near-rank-deficient EEG must not crash the SPD math."""

    def _trial(self, subject, idx, eeg, label=0):
        return Trial(
            eeg=eeg.astype(np.float32), label=label, fs=100,
            subject=subject, run=4, trial_idx=idx,
        )

    def test_dead_channel_does_not_crash_static_ra(self, ill_conditioned_eeg):
        ra = StaticRA(regularize=1e-5)
        trials = [self._trial(1, i, ill_conditioned_eeg) for i in range(3)]
        ra.fit_source_trials(trials)
        out = ra.transform_trial(trials[0])
        assert np.all(np.isfinite(out.eeg))

    def test_dead_channel_does_not_crash_ema_ra(self, ill_conditioned_eeg):
        ema = EMARA(alpha=0.1, regularize=1e-5, update_online=True)
        source_trials = [self._trial(1, i, ill_conditioned_eeg) for i in range(3)]
        ema.fit_source_trials(source_trials)
        target_trials = [self._trial(99, i, ill_conditioned_eeg) for i in range(3)]
        for t in target_trials:
            out = ema.transform_trial(t)
            assert np.all(np.isfinite(out.eeg))


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


class TestNoopAdapter:
    def test_transform_trial_returns_same(self, synth_dataset):
        a = NoopAdapter()
        t = _trials_from(synth_dataset, 1)[0]
        assert a.transform_trial(t) is t

    def test_feature_transform_identity(self):
        a = NoopAdapter()
        feats = np.random.default_rng(0).standard_normal((10, 4))
        assert np.array_equal(a.transform(feats), feats)
