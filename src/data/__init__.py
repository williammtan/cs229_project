"""Datasets and split protocols."""
from src.data.eegmmi import (
    CLASS_NAMES,
    EEG_BAND,
    EEG_FS,
    EXCLUDED_SUBJECTS,
    IMAGERY_RUNS,
    N_CLASSES,
    N_EEG_CHANNELS,
    RUN_LABEL_MAP,
    TARGET_FS,
    SubjectData,
    Trial,
    load_dataset,
    load_subject,
    stack_eeg_and_labels,
)
from src.data.splits import LOSO, WithinSubjectKFold, sample_k_trials_per_class

__all__ = [
    "CLASS_NAMES",
    "EEG_BAND",
    "EEG_FS",
    "EXCLUDED_SUBJECTS",
    "IMAGERY_RUNS",
    "LOSO",
    "N_CLASSES",
    "N_EEG_CHANNELS",
    "RUN_LABEL_MAP",
    "SubjectData",
    "TARGET_FS",
    "Trial",
    "WithinSubjectKFold",
    "load_dataset",
    "load_subject",
    "sample_k_trials_per_class",
    "stack_eeg_and_labels",
]
