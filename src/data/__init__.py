"""Datasets and split protocols.

Re-exports keep the historical ``from src.data import Trial, load_subject, ...``
import path working.
"""
from src.data.way_eeg_gal import (
    EEG_BAND,
    EEG_FS,
    KIN_HAND_POS,
    N_EEG_CHANNELS,
    PTS_BAND,
    TARGET_FS,
    SubjectData,
    Trial,
    concat_trials,
    load_dataset,
    load_subject,
    preprocess_trial,
    trial_lengths,
)
from src.data.splits import LOSO, WithinSubjectKFold, sample_k_minutes
from src.data.windows import make_windows, windows_to_per_sample

__all__ = [
    "EEG_BAND",
    "EEG_FS",
    "KIN_HAND_POS",
    "N_EEG_CHANNELS",
    "PTS_BAND",
    "TARGET_FS",
    "SubjectData",
    "Trial",
    "concat_trials",
    "load_dataset",
    "load_subject",
    "preprocess_trial",
    "trial_lengths",
    "LOSO",
    "WithinSubjectKFold",
    "sample_k_minutes",
    "make_windows",
    "windows_to_per_sample",
]
