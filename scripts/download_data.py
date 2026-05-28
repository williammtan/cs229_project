"""Download PhysioNet EEG Motor Movement/Imagery (EEGMMI) subjects.

Uses MNE's eegbci helper which fetches the EDF files from PhysioNet and caches
them under ``raw_dir/S{NNN}/S{NNN}R{NN}.edf`` — the layout our loader expects.

Subjects are downloaded in parallel (4 workers default) since MNE's per-subject
``load_data`` is serial across runs; one subject ≈ 6 sequential HTTP requests.
Four workers gives ~4× wall-clock speedup without overloading PhysioNet.

Usage:
    uv run python scripts/download_data.py                 # default: subjects 1, 2
    uv run python scripts/download_data.py 1 2 3 4 5       # custom subset
    uv run python scripts/download_data.py --all           # all 109 minus exclusions
    uv run python scripts/download_data.py --all -j 8      # 8 parallel workers
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make `src` importable when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne  # noqa: E402

from src.data.eegmmi import EXCLUDED_SUBJECTS, IMAGERY_RUNS  # noqa: E402


def download_subject(subject: int, raw_dir: Path) -> None:
    if subject in EXCLUDED_SUBJECTS:
        print(f"  S{subject:03d}: excluded (annotation/sfreq issue), skipping")
        return
    subj_dir = raw_dir / f"S{subject:03d}"
    expected = [subj_dir / f"S{subject:03d}R{r:02d}.edf" for r in IMAGERY_RUNS]
    if all(p.exists() for p in expected):
        print(f"  S{subject:03d}: already present, skipping")
        return
    # mne.datasets.eegbci.load_data downloads to its own cache; we move/symlink
    # into raw_dir/S{NNN}/ so all downstream code uses a single layout.
    paths = mne.datasets.eegbci.load_data(
        subject, runs=list(IMAGERY_RUNS), path=str(raw_dir.parent), update_path=False,
        verbose="ERROR",
    )
    subj_dir.mkdir(parents=True, exist_ok=True)
    for src in paths:
        src = Path(src)
        dst = subj_dir / src.name
        if dst.exists():
            continue
        # symlink keeps the MNE cache canonical; falls back to copy if cross-fs.
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            import shutil

            shutil.copy2(src, dst)
    print(f"  S{subject:03d}: downloaded {len(paths)} runs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subjects", nargs="*", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="Download all 109 minus exclusions")
    ap.add_argument(
        "--raw-dir", default=None,
        help="Destination dir (default: <repo>/data/raw/eegmmi)",
    )
    ap.add_argument(
        "-j", "--jobs", type=int, default=4,
        help="Parallel workers (default 4). Use 1 for serial.",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    raw = Path(args.raw_dir) if args.raw_dir else repo / "data" / "raw" / "eegmmi"
    raw.mkdir(parents=True, exist_ok=True)

    if args.all:
        subjects = [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS]
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = [1, 2]

    print(f"Downloading EEGMMI subjects {len(subjects)} subjects to {raw} (jobs={args.jobs})")
    if args.jobs <= 1:
        for s in subjects:
            download_subject(s, raw)
        return 0

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(download_subject, s, raw): s for s in subjects}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  S{s:03d}: FAILED {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
