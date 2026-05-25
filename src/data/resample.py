"""Resample helpers for FM-backbone input preparation.

Our preprocessing pipeline downsamples raw 500 Hz EEG to 100 Hz (TARGET_FS).
All three foundation backbones (CBraMod, LaBraM, REVE) were pretrained at
200 Hz, so each FM wrapper resamples ``trial.eeg`` up to 200 Hz on the fly.

We use scipy.signal.resample_poly which is a polyphase upsample with an
implicit anti-aliasing filter — no harmonic injection. For integer ratios
this is correct; non-integer ratios use a rational approximation.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def resample_eeg(eeg: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    """Resample ``eeg`` along the last axis from src_fs to dst_fs."""
    if src_fs == dst_fs:
        return eeg.astype(np.float32, copy=False)
    from math import gcd
    g = gcd(src_fs, dst_fs)
    up, down = dst_fs // g, src_fs // g
    out = resample_poly(eeg, up, down, axis=-1)
    return out.astype(np.float32)
