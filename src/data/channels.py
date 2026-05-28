"""Electrode-name conventions for PhysioNet EEGMMI.

Centralizes:
  * the 64 BCI2000 10-10 channel names in the row order EEGMMI EDF files use
    (after our trailing-dot/casing normalization),
  * a 16-channel motor-cortex demo subset,
  * helpers to project a trial down to a subset.

The EEGMMI EDF channel order is fixed by BCI2000 and is the same for every
subject; we list it once here so downstream code can index by name.
"""
from __future__ import annotations

import numpy as np


# ---- canonical 64-ch BCI2000 EEGMMI order (after _normalize_ch_name) --------

EEGMMI_64: tuple[str, ...] = (
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6",
    "Fp1", "Fpz", "Fp2",
    "Af7", "Af3", "Afz", "Af4", "Af8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "Ft7", "Ft8",
    "T7", "T8", "T9", "T10",
    "Tp7", "Tp8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "Po7", "Po3", "Poz", "Po4", "Po8",
    "O1", "Oz", "O2",
    "Iz",
)


# ---- 16-channel motor-cortex demo subset ------------------------------------
# C-row + flanking FC/CP + Pz, both hemispheres around the central sulcus.
# Matches the §9 demo channel list in docs/plan_clf.md.

MOTOR_16_EEGMMI: tuple[str, ...] = (
    "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
    "C3", "C1", "Cz", "C2", "C4",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
    "Pz",
)


def channel_indices(subset: tuple[str, ...], reference: tuple[str, ...] = EEGMMI_64) -> np.ndarray:
    """Return the row indices of ``subset`` within ``reference``. Case-insensitive."""
    ref_map = {n.lower(): i for i, n in enumerate(reference)}
    out, missing = [], []
    for name in subset:
        i = ref_map.get(name.lower())
        if i is None:
            missing.append(name)
        else:
            out.append(i)
    if missing:
        raise KeyError(f"{missing} not in reference montage {reference}")
    return np.asarray(out, dtype=np.int64)


def subset_trial_eeg(eeg: np.ndarray, subset: tuple[str, ...] = MOTOR_16_EEGMMI) -> np.ndarray:
    """Project (C, T) EEG down to a channel subset, preserving row order of ``subset``."""
    idx = channel_indices(subset)
    return eeg[idx]


def get_channel_names(n_channels: int = 64) -> tuple[str, ...]:
    """Return the canonical channel-name list for a given subset size."""
    if n_channels == 64:
        return EEGMMI_64
    if n_channels == 16:
        return MOTOR_16_EEGMMI
    raise ValueError(f"No canonical channel subset for n_channels={n_channels}")
