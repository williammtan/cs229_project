"""Electrode-name conventions for WAY-EEG-GAL.

Centralizes:
  * the 32 ActiCap channel names in trial.eeg row order,
  * the 16-channel motor-area subset used for demo day,
  * helpers to project a Trial down to a channel subset,
  * normalized name list for foundation models that key on 10-20 names.

WAY-EEG-GAL records 32 channels per Luciw 2014 Section 2.2; the row order in
trial.eeg follows ws.names.eeg in the .mat files (verified by inspection on
2026-05-24).
"""
from __future__ import annotations

import numpy as np

# ---- canonical 32-ch ActiCap order (matches Trial.eeg rows) ----------------

ACTICAP_32: tuple[str, ...] = (
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
)

# ---- 16-channel motor subset (demo day) -------------------------------------
# Strategy: keep frontocentral, central, and centroparietal — the
# motor / sensorimotor strip — and drop temporal-most and occipital sensors.
# These 16 cover both hemispheres symmetrically around the central sulcus.

MOTOR_16: tuple[str, ...] = (
    "F3", "Fz", "F4",
    "FC5", "FC1", "FC2", "FC6",
    "C3", "Cz", "C4",
    "CP5", "CP1", "CP2", "CP6",
    "P3", "P4",
)


def channel_indices(subset: tuple[str, ...], reference: tuple[str, ...] = ACTICAP_32) -> np.ndarray:
    """Return the row indices of ``subset`` within ``reference``."""
    ref_map = {n.lower(): i for i, n in enumerate(reference)}
    out = []
    missing = []
    for name in subset:
        i = ref_map.get(name.lower())
        if i is None:
            missing.append(name)
        else:
            out.append(i)
    if missing:
        raise KeyError(
            f"{missing} not in reference montage {reference}"
        )
    return np.asarray(out, dtype=np.int64)


def subset_trial_eeg(eeg: np.ndarray, subset: tuple[str, ...] = MOTOR_16) -> np.ndarray:
    """Project (C, T) EEG down to a channel subset, preserving row order of ``subset``."""
    idx = channel_indices(subset)
    return eeg[idx]


def get_channel_names(n_channels: int = 32) -> tuple[str, ...]:
    """Return the canonical channel-name list for a given subset size.

    Resolves the ``dataset.channels`` config field to a name list.
    """
    if n_channels == 32:
        return ACTICAP_32
    if n_channels == 16:
        return MOTOR_16
    raise ValueError(f"No canonical channel subset for n_channels={n_channels}")
