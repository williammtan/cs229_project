"""Download WAY-EEG-GAL subjects from Figshare into data/raw/.

Usage:
    uv run python scripts/download_data.py            # default: subjects 1, 2, 3
    uv run python scripts/download_data.py 1 2 3 4 5  # custom subset
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests


# Figshare article IDs for each WAY-EEG-GAL participant (from collection 988376).
ARTICLE_IDS = {
    1: 1185502, 2: 1185505, 3: 1185507, 4: 1185509, 5: 1185511, 6: 1119392,
    7: 1119691, 8: 1119669, 9: 1119677, 10: 1119682, 11: 1119680, 12: 1119678,
}


def article_files(article_id: int) -> list[dict]:
    r = requests.get(f"https://api.figshare.com/v2/articles/{article_id}", timeout=30)
    r.raise_for_status()
    return r.json().get("files", [])


def download_subject(subject: int, raw_dir: Path) -> None:
    art = ARTICLE_IDS[subject]
    files = article_files(art)
    p_zip = next(f for f in files if f["name"].lower() == f"p{subject}.zip")
    out = raw_dir / p_zip["name"]
    if (raw_dir / f"WS_P{subject}_S1.mat").exists():
        print(f"  P{subject}: already extracted, skipping")
        return
    print(f"  P{subject}: {p_zip['size'] / 1e6:.1f} MB -> {out}")
    with requests.get(p_zip["download_url"], stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    with zipfile.ZipFile(out) as zf:
        zf.extractall(raw_dir)
    out.unlink()
    print(f"  P{subject}: done")


def main():
    if len(sys.argv) > 1:
        subjects = [int(s) for s in sys.argv[1:]]
    else:
        subjects = [1, 2, 3]
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"Downloading WAY-EEG-GAL subjects {subjects} to {raw}")
    for s in subjects:
        if s not in ARTICLE_IDS:
            print(f"  P{s}: unknown subject, skipping")
            continue
        download_subject(s, raw)


if __name__ == "__main__":
    main()
