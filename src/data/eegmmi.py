"""PhysioNet EEG Motor Movement/Imagery (EEGMMI) dataset.

Schalk et al. 2004 (BCI2000), distributed via PhysioNet (`eegmmidb`). 109 subjects,
each with 14 runs in EDF format. We use only the **imagery** runs for the
canonical 4-class motor-imagery (MI) decoding task:

    Runs 4, 8, 12: imagined open/close LEFT or RIGHT fist
        T1 -> LeftFist  (label 0)
        T2 -> RightFist (label 1)
    Runs 6, 10, 14: imagined open/close BOTH fists or BOTH feet
        T1 -> BothFists (label 2)
        T2 -> BothFeet  (label 3)
    T0 in every run is rest and is dropped from the 4-class task.

Subjects {88, 89, 92, 100, 104} are excluded everywhere (mismatched sampling
rate / annotation issues — universal in the EEGMMI literature).

Epoch: [0, 4] s post-cue. At native 160 Hz -> 640 samples per trial.

Each trial is preprocessed to ``Trial(eeg=(C, T), label, fs, subject, run, trial_idx)``,
i.e. one label per trial. The default target rate is 100 Hz to match the existing
preprocessing convention; FM backbones upsample to 200 Hz at their input.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

EEG_FS = 160  # Hz, native sampling rate
TARGET_FS = 100  # Hz, after downsampling
EEG_BAND = (0.5, 40.0)  # wide-band, identical conceptually to the WAY-EEG-GAL pipeline

N_EEG_CHANNELS = 64

# Per the BCI2000 EDF naming convention these have a trailing dot ("Fc5.", "C3.")
EXCLUDED_SUBJECTS: frozenset[int] = frozenset({88, 89, 92, 100, 104})

# 4-class MI label assignment indexed by (run_id, marker_letter).
# run_id 4/8/12 are LR-imagery; 6/10/14 are HandsFeet-imagery.
RUN_LABEL_MAP: dict[tuple[int, str], int] = {
    (4, "T1"): 0, (8, "T1"): 0, (12, "T1"): 0,    # LeftFist
    (4, "T2"): 1, (8, "T2"): 1, (12, "T2"): 1,    # RightFist
    (6, "T1"): 2, (10, "T1"): 2, (14, "T1"): 2,   # BothFists
    (6, "T2"): 3, (10, "T2"): 3, (14, "T2"): 3,   # BothFeet
}
IMAGERY_RUNS: tuple[int, ...] = (4, 6, 8, 10, 12, 14)
CLASS_NAMES: tuple[str, ...] = ("LeftFist", "RightFist", "BothFists", "BothFeet")
N_CLASSES = 4

# Trial epoch: [tmin, tmax] s relative to cue onset.
DEFAULT_TMIN = 0.0
DEFAULT_TMAX = 4.0


@dataclass
class Trial:
    eeg: np.ndarray  # (n_channels, n_samples) float32, downsampled to TARGET_FS
    label: int
    fs: int
    subject: int
    run: int
    trial_idx: int


@dataclass
class SubjectData:
    subject: int
    trials: list[Trial]


def _normalize_ch_name(name: str) -> str:
    """BCI2000 EDF channel names have trailing dots and odd casing.
    Normalize to plain 10-10 (e.g. ``Fc5.`` -> ``FC5``, ``Cz..`` -> ``Cz``)."""
    n = name.strip().rstrip(".").strip()
    if not n:
        return n
    # 10-10 names are typically <Letter(s)><Digit(s)?> with title-cased letters.
    # Capitalize alphabetic prefix; lowercase 'z' / 'p' that BCI2000 uppercases.
    return n[0].upper() + n[1:]


def _load_one_run(
    edf_path: Path,
    subject: int,
    run: int,
    tmin: float,
    tmax: float,
    dst_fs: int,
    eeg_band: tuple[float, float],
) -> list[Trial]:
    """Return a list of Trial for all annotated T1/T2 events in this run."""
    import mne

    # MNE is verbose by default — silence so loading 109 subjects doesn't spam.
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    # Some subjects record at 128 Hz; resample everything to dst_fs.
    if raw.info["sfreq"] != EEG_FS and raw.info["sfreq"] != 128.0:
        # Unknown rate — still resample to dst_fs.
        pass
    raw.filter(eeg_band[0], eeg_band[1], fir_design="firwin", verbose="ERROR")
    raw.resample(dst_fs, verbose="ERROR")

    # Standardize channel names (Fc5. -> FC5) for downstream montage matching.
    rename = {c: _normalize_ch_name(c) for c in raw.ch_names}
    raw.rename_channels(rename)

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    # event_id maps {'T0':1,'T1':2,'T2':3} or similar — keep only T1/T2 keys
    # that correspond to a class label in this run.
    keep_ids: dict[str, int] = {}
    for k, v in event_id.items():
        if k in ("T1", "T2") and (run, k) in RUN_LABEL_MAP:
            keep_ids[k] = v
    if not keep_ids:
        return []

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=keep_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        proj=False,
        verbose="ERROR",
    )
    X = epochs.get_data(copy=False).astype(np.float32)  # (n_trials, n_ch, n_samples)
    # mne.Epochs includes the sample at tmax, giving (dst_fs * (tmax - tmin)) + 1
    # samples. Trim the final sample so the trial length is exactly 4 s * dst_fs.
    expected = int(round((tmax - tmin) * dst_fs))
    if X.shape[-1] > expected:
        X = X[..., :expected]

    out: list[Trial] = []
    for i, evt in enumerate(epochs.events):
        marker = next(k for k, v in keep_ids.items() if v == evt[-1])
        label = RUN_LABEL_MAP[(run, marker)]
        out.append(Trial(
            eeg=X[i],
            label=label,
            fs=dst_fs,
            subject=subject,
            run=run,
            trial_idx=i,
        ))
    return out


def load_subject(
    raw_dir: Path | str,
    subject: int,
    runs: Iterable[int] | None = None,
    dst_fs: int = TARGET_FS,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
    eeg_band: tuple[float, float] = EEG_BAND,
) -> SubjectData:
    """Load all imagery-run trials for a subject.

    ``raw_dir`` is expected to contain the standard PhysioNet eegmmidb layout:

        raw_dir/S{NNN}/S{NNN}R{NN}.edf

    where NNN is the zero-padded subject id (1 -> 001) and NN is the run id.
    This matches what ``mne.datasets.eegbci.load_data`` produces.
    """
    if subject in EXCLUDED_SUBJECTS:
        return SubjectData(subject=subject, trials=[])
    if runs is None:
        runs = IMAGERY_RUNS

    raw_dir = Path(raw_dir)
    subj_dir = raw_dir / f"S{subject:03d}"
    trials: list[Trial] = []
    for run in runs:
        edf = subj_dir / f"S{subject:03d}R{run:02d}.edf"
        if not edf.exists():
            continue
        try:
            trials.extend(_load_one_run(
                edf, subject=subject, run=run,
                tmin=tmin, tmax=tmax, dst_fs=dst_fs, eeg_band=eeg_band,
            ))
        except Exception as e:  # noqa: BLE001
            print(f"  [eegmmi] skip S{subject:03d}R{run:02d}: {type(e).__name__}: {e}")
            continue
    return SubjectData(subject=subject, trials=trials)


def load_dataset(
    raw_dir: Path | str,
    subjects: Iterable[int],
    runs: Iterable[int] | None = None,
    dst_fs: int = TARGET_FS,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
) -> dict[int, SubjectData]:
    """Convenience: load multiple subjects into ``{subject_id: SubjectData}``.

    Subjects in ``EXCLUDED_SUBJECTS`` are silently skipped (empty trial list).
    """
    out: dict[int, SubjectData] = {}
    for s in subjects:
        if s in EXCLUDED_SUBJECTS:
            continue
        sd = load_subject(raw_dir, subject=s, runs=runs, dst_fs=dst_fs, tmin=tmin, tmax=tmax)
        if sd.trials:
            out[s] = sd
    return out


def stack_eeg_and_labels(trials: list[Trial]) -> tuple[np.ndarray, np.ndarray]:
    """Stack trials into (N, C, T) EEG and (N,) int label arrays.

    The classification analogue of WAY-EEG-GAL's ``concat_trials``; the key
    difference is that there is one label per trial rather than a per-sample target.
    """
    X = np.stack([t.eeg for t in trials], axis=0).astype(np.float32)
    y = np.asarray([t.label for t in trials], dtype=np.int64)
    return X, y
