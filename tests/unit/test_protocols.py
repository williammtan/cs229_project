"""Unit tests for split protocols: LOSO and K-trials calibration."""
from __future__ import annotations

from src.data.splits import sample_k_trials_per_class
from src.protocols.kmin_calibration import KTrialsCalibrationProtocol
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
            assert held_out not in train_subjects
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


class TestKTrialsCalibrationProtocol:
    def test_splits_per_subject_per_k(self, synth_dataset):
        budgets = (0, 1)
        protocol = KTrialsCalibrationProtocol(k_budgets_trials=budgets)
        splits = list(protocol.iter_splits(synth_dataset))
        assert len(splits) == len(synth_dataset) * len(budgets)

    def test_k0_yields_empty_calib(self, synth_dataset):
        protocol = KTrialsCalibrationProtocol(k_budgets_trials=(0,))
        for split in protocol.iter_splits(synth_dataset):
            assert split.meta["k_trials_per_class"] == 0
            assert split.calib == []
            held_out = split.meta["held_out_subject"]
            assert len(split.eval) == len(synth_dataset[held_out].trials)

    def test_kpos_consumes_calib_then_eval_is_remainder(self, synth_dataset):
        protocol = KTrialsCalibrationProtocol(k_budgets_trials=(1,))
        for split in protocol.iter_splits(synth_dataset):
            held_out = split.meta["held_out_subject"]
            n_total = len(synth_dataset[held_out].trials)
            assert len(split.calib) + len(split.eval) == n_total
            assert len(split.calib) >= 1
            assert {t.subject for t in split.calib} == {held_out}

    def test_train_set_disjoint_from_held_out(self, synth_dataset):
        protocol = KTrialsCalibrationProtocol(k_budgets_trials=(1,))
        for split in protocol.iter_splits(synth_dataset):
            held_out = split.meta["held_out_subject"]
            assert held_out not in {t.subject for t in split.train}


class TestSampleKTrialsPerClass:
    def test_zero_budget_returns_empty_calib(self, synth_dataset):
        sd = list(synth_dataset.values())[0]
        calib, eval_trials = sample_k_trials_per_class(sd, k_per_class=0)
        assert calib == []
        assert len(eval_trials) == len(sd.trials)

    def test_balanced_calibration(self, synth_dataset):
        sd = list(synth_dataset.values())[0]
        # 8 trials × 4 classes = 2/class. Ask for 1/class → 4 calib trials, balanced.
        calib, eval_trials = sample_k_trials_per_class(sd, k_per_class=1)
        labels = sorted(t.label for t in calib)
        assert len(calib) == 4
        assert labels == [0, 1, 2, 3]
        assert len(eval_trials) == 4

    def test_full_budget_absorbs_all_trials(self, synth_dataset):
        sd = list(synth_dataset.values())[0]
        calib, eval_trials = sample_k_trials_per_class(sd, k_per_class=999)
        assert len(calib) == len(sd.trials)
        assert eval_trials == []
