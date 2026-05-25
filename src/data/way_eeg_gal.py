"""WAY-EEG-GAL dataset loading and preprocessing.

WAY-EEG-GAL (Luciw, Jarocka & Edin, Scientific Data 2014) provides 12 subjects
performing grasp-and-lift trials of an instrumented object whose weight (165,
330, 660 g) and surface (sandpaper, suede, silk) vary unpredictably.

Per-subject zips on Figshare expand to 9 WS_P{n}_S{m}.mat files (windowed
trial-aligned format) and 9 HS_P{n}_S{m}.mat files (continuous "hand state")
plus events files. We use the WS files: each contains a struct `ws` with
`win(i).eeg` (samples x 32 channels) and `win(i).kin` (samples x 24 kinematic
channels) co-sampled at 500 Hz. Polhemus 3D hand-sensor position lives in
kinematic columns 1-3 (Px, Py, Pz) per the Luciw 2014 data description.

The decoding target is hand velocity (dPx/dt, dPy/dt, dPz/dt) at the EEG
sample rate, which is the standard Bradberry/Müller-Putz target.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, sosfilt


# WAY-EEG-GAL kinematic column indices (0-indexed, per ws.names.kin):
#   18=Px1 (hand sensor X), 22=Py1 (hand sensor Y), 26=Pz1 (hand sensor Z).
# Sensor 1 is the hand sensor (Polhemus FASTRAK on the dorsum of the hand),
# sensor 2 is on the object. Units: mm.
KIN_HAND_POS = (18, 22, 26)

EEG_FS = 500  # Hz, native sampling rate
TARGET_FS = 100  # Hz, after downsampling
EEG_BAND = (0.1, 40.0)  # wide-band for neural-net features
PTS_BAND = (0.1, 3.0)  # low-frequency time-domain for Bradberry-style decoders

N_EEG_CHANNELS = 32


@dataclass
class Trial:
    eeg: np.ndarray  # (n_channels, n_samples) float32, downsampled to TARGET_FS
    kin: np.ndarray  # (3, n_samples) float32 — hand position, same fs as eeg
    vel: np.ndarray  # (3, n_samples) float32 — first derivative of kin
    fs: int
    subject: int
    series: int
    trial_idx: int


@dataclass
class SubjectData:
    subject: int
    trials: list[Trial]


def _butter_sos(low: float | None, high: float | None, fs: int, order: int = 4):
    if low is not None and high is not None:
        return butter(order, [low, high], btype="band", fs=fs, output="sos")
    if high is not None:
        return butter(order, high, btype="low", fs=fs, output="sos")
    if low is not None:
        return butter(order, low, btype="high", fs=fs, output="sos")
    raise ValueError("Need at least one of low/high")


def _band_filter(x: np.ndarray, low: float | None, high: float | None, fs: int,
                 causal: bool = False) -> np.ndarray:
    """Filter along last axis. Zero-phase if causal=False; causal IIR if True."""
    sos = _butter_sos(low, high, fs)
    if causal:
        return sosfilt(sos, x, axis=-1).astype(np.float32)
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def _downsample(x: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    """Polyphase downsample along last axis."""
    assert src_fs % dst_fs == 0, "non-integer ratio not supported here"
    factor = src_fs // dst_fs
    return x[..., ::factor]


def _load_ws_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (eeg_list, kin_list) lists of arrays per trial in this series file.

    WAY-EEG-GAL WS file structure (post scipy.io.loadmat with simplify_cells):
      mat['ws']['win'] is a list of trial dicts, each with keys including
      'eeg' (samples x 32) and 'kin' (samples x 24).
    """
    mat = sio.loadmat(str(path), simplify_cells=True)
    ws = mat["ws"]
    wins = ws["win"]
    if isinstance(wins, dict):  # single-trial edge case
        wins = [wins]
    eegs = []
    kins = []
    for w in wins:
        eeg = np.asarray(w["eeg"], dtype=np.float32).T  # -> (32, n_samples)
        kin = np.asarray(w["kin"], dtype=np.float32).T  # -> (24, n_samples)
        if eeg.shape[0] != N_EEG_CHANNELS:
            # Some files have an extra ref channel; trim to first 32
            eeg = eeg[:N_EEG_CHANNELS]
        eegs.append(eeg)
        kins.append(kin)
    return eegs, kins


def preprocess_trial(
    eeg_raw: np.ndarray,
    kin_raw: np.ndarray,
    src_fs: int = EEG_FS,
    dst_fs: int = TARGET_FS,
    eeg_band: tuple[float, float] = EEG_BAND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess one trial.

    Returns (eeg, kin, vel) all at dst_fs.
      eeg: (32, T) float32 — band-passed, downsampled, common-average-referenced, z-scored per channel
      kin: (3, T) float32 — hand position downsampled
      vel: (3, T) float32 — first-difference velocity * dst_fs (mm/s)
    """
    # 1) Common Average Reference (subtract mean across channels at each sample)
    eeg = eeg_raw - eeg_raw.mean(axis=0, keepdims=True)
    # 2) Band-pass
    eeg = _band_filter(eeg, eeg_band[0], eeg_band[1], src_fs, causal=False)
    # 3) Downsample
    eeg = _downsample(eeg, src_fs, dst_fs)
    # 4) Per-channel z-score (trial-wise; will be cross-validated later)
    eeg = (eeg - eeg.mean(axis=-1, keepdims=True)) / (eeg.std(axis=-1, keepdims=True) + 1e-6)
    eeg = eeg.astype(np.float32)

    # Kinematics: pick hand position cols, low-pass smooth at 5 Hz, downsample,
    # then differentiate to velocity.
    kin = kin_raw[list(KIN_HAND_POS)]
    kin = _band_filter(kin, None, 5.0, src_fs, causal=False)
    kin = _downsample(kin, src_fs, dst_fs)
    vel = np.gradient(kin, 1.0 / dst_fs, axis=-1).astype(np.float32)
    kin = kin.astype(np.float32)
    return eeg, kin, vel


def load_subject(
    raw_dir: Path,
    subject: int,
    series: Iterable[int] | None = None,
    src_fs: int = EEG_FS,
    dst_fs: int = TARGET_FS,
) -> SubjectData:
    """Load all trials for a subject. WAY-EEG-GAL has up to 9 series per subject."""
    raw_dir = Path(raw_dir)
    if series is None:
        series = range(1, 10)
    trials: list[Trial] = []
    for s in series:
        path = raw_dir / f"P{subject}" / f"WS_P{subject}_S{s}.mat"
        if not path.exists():
            # try flat layout
            path = raw_dir / f"WS_P{subject}_S{s}.mat"
        if not path.exists():
            continue
        eegs, kins = _load_ws_mat(path)
        for i, (eeg_raw, kin_raw) in enumerate(zip(eegs, kins)):
            try:
                eeg, kin, vel = preprocess_trial(eeg_raw, kin_raw, src_fs, dst_fs)
            except Exception:
                continue
            trials.append(Trial(
                eeg=eeg, kin=kin, vel=vel, fs=dst_fs,
                subject=subject, series=s, trial_idx=i,
            ))
    return SubjectData(subject=subject, trials=trials)


def concat_trials(trials: list[Trial]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate trials into (n_samples_total, n_channels) and (n_samples_total, 3)
    arrays, suitable for linear-regression-style decoders.

    Returns eeg_concat (T, 32), pos_concat (T, 3), vel_concat (T, 3).
    """
    eegs = np.concatenate([t.eeg for t in trials], axis=-1).T  # (T, 32)
    pos = np.concatenate([t.kin for t in trials], axis=-1).T   # (T, 3)
    vel = np.concatenate([t.vel for t in trials], axis=-1).T   # (T, 3)
    return eegs.astype(np.float32), pos.astype(np.float32), vel.astype(np.float32)


def trial_lengths(trials: list[Trial]) -> np.ndarray:
    return np.array([t.eeg.shape[-1] for t in trials])


def load_dataset(
    raw_dir: Path | str,
    subjects: Iterable[int],
    series: Iterable[int] | None = None,
    src_fs: int = EEG_FS,
    dst_fs: int = TARGET_FS,
) -> dict[int, SubjectData]:
    """Convenience: load multiple subjects into ``{subject_id: SubjectData}``."""
    out: dict[int, SubjectData] = {}
    for s in subjects:
        out[s] = load_subject(raw_dir, subject=s, series=series, src_fs=src_fs, dst_fs=dst_fs)
    return out
