"""Unit tests for split protocols: LOSO and KMinCalibration."""
from __future__ import annotations

import pytest

from src.data.splits import sample_k_minutes
from src.protocols.kmin_calibration import KMinCalibrationProtocol
from src.protocols.loso import LOSOProtocol


class TestLOSOProtocol:
    def test_one_split_per_subject(self, synth_dataset):
        protocol = LOSOProtocol()
        splits = list(protocol.iter_splits(synth_dataset))
        assert len(splits) == len(synth_dataset)

    def test_held_out_subject_not_in_train(self, synth_dataset):
        protocol = LOSOProtocol()
        for split in protocol.iter_splits(synth_dataset):
            held_out = split.meta["held_out_subject"]
            train_subjects = {t.subject for t in split.train}
            eval_subjects = {t.subject for t in split.eval}
            assert held_out not in train_subjects, \
                f"subject {held_out} leaked into train"
            assert eval_subjects == {held_out}

    def test_calib_always_empty(self, synth_dataset):
        protocol = LOSOProtocol()
        for split in protocol.iter_splits(synth_dataset):
            assert split.calib == []

    def test_train_pools_n_minus_1_subjects(self, synth_dataset):
        protocol = LOSOProtocol()
        n = len(synth_dataset)
        for split in protocol.iter_splits(synth_dataset):
            assert len(set(t.subject for t in split.train)) == n - 1


class TestKMinCalibrationProtocol:
    def test_splits_per_subject_per_k(self, synth_dataset):
        budgets = (0.0, 1.0)
        protocol = KMinCalibrationProtocol(k_budgets_min=budgets)
        splits = list(protocol.iter_splits(synth_dataset))
        # n_subjects x len(budgets)
        assert len(splits) == len(synth_dataset) * len(budgets)

    def test_k0_yields_empty_calib(self, synth_dataset):
        protocol = KMinCalibrationProtocol(k_budgets_min=(0.0,))
        for split in protocol.iter_splits(synth_dataset):
            assert split.meta["k_minutes"] == 0.0
            assert split.calib == []
            # All held-out trials remain in eval.
            held_out = split.meta["held_out_subject"]
            assert len(split.eval) == len(synth_dataset[held_out].trials)

    def test_kpos_consumes_calib_then_eval_is_remainder(self, synth_dataset):
        # Each synthetic trial is 6 seconds (=0.1 minutes). Two trials fit in 0.2 min;
        # ask for exactly 0.2 to make the budget unambiguous.
        protocol = KMinCalibrationProtocol(k_budgets_min=(0.2,))
        for split in protocol.iter_splits(synth_dataset):
            held_out = split.meta["held_out_subject"]
            n_total = len(synth_dataset[held_out].trials)
            assert len(split.calib) + len(split.eval) == n_total
            assert len(split.calib) >= 1
            # Calib trials all share the held-out subject.
            assert {t.subject for t in split.calib} == {held_out}

    def test_train_set_disjoint_from_held_out(self, synth_dataset):
        protocol = KMinCalibrationProtocol(k_budgets_min=(0.5,))
        for split in protocol.iter_splits(synth_dataset):
            held_out = split.meta["held_out_subject"]
            assert held_out not in {t.subject for t in split.train}


class TestSampleKMinutes:
    def test_zero_budget_returns_empty_calib(self, synth_dataset):
        sd = list(synth_dataset.values())[0]
        calib, eval_trials = sample_k_minutes(sd, k_minutes=0.0)
        assert calib == []
        assert len(eval_trials) == len(sd.trials)

    def test_huge_budget_absorbs_all_trials(self, synth_dataset):
        sd = list(synth_dataset.values())[0]
        calib, eval_trials = sample_k_minutes(sd, k_minutes=10_000.0)
        # All synthetic trials fit (each is 6s = 0.1min, 6 trials = 0.6min).
        assert len(calib) == len(sd.trials)
        assert eval_trials == []

    def test_calib_trials_come_first_in_order(self, synth_dataset):
        sd = list(synth_dataset.values())[0]
        calib, eval_trials = sample_k_minutes(sd, k_minutes=0.2)
        calib_idx = [t.trial_idx for t in calib]
        eval_idx = [t.trial_idx for t in eval_trials]
        assert calib_idx == sorted(calib_idx)
        assert all(c < e for c in calib_idx for e in eval_idx)
