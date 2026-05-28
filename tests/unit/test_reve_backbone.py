"""Regression tests for REVE wrapper setup."""
from __future__ import annotations

import numpy as np

from src.backbones.reve import _build_chs_info, _zscore_clip_eeg
from src.data.channels import EEGMMI_64, MOTOR_16_EEGMMI


def test_reve_eegmmi_channel_names_use_position_bank_spelling():
    chs = _build_chs_info(list(EEGMMI_64))
    names = [ch["ch_name"] for ch in chs]

    assert len(names) == len(EEGMMI_64)
    assert len(set(names)) == len(names)
    assert "FC3" in names
    assert "FCz" in names
    assert "AFz" in names
    assert "POz" in names
    assert "Fc3" not in names
    assert "Fcz" not in names


def test_reve_motor_16_channel_names_use_position_bank_spelling():
    chs = _build_chs_info(list(MOTOR_16_EEGMMI))
    names = [ch["ch_name"] for ch in chs]

    assert len(names) == len(MOTOR_16_EEGMMI)
    assert "FC3" in names
    assert "CPz" in names


def test_reve_zscore_clip_eeg_normalizes_per_channel():
    eeg = np.asarray([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 10.0, 10.0, 10.0],
    ], dtype=np.float32)

    out = _zscore_clip_eeg(eeg, clip=2.0)

    assert out.dtype == np.float32
    assert np.allclose(out[0].mean(), 0.0, atol=1.0e-6)
    assert np.allclose(out[0].std(), 1.0, atol=1.0e-6)
    assert np.allclose(out[1], 0.0)
