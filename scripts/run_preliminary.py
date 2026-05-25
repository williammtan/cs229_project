"""Drive the new Hydra runner for the four preliminary baselines.

Replaces the inline orchestration that previously lived here. Each baseline
becomes one ``python -m src.runner +experiment=...`` invocation; results land
on the configured logger (offline JSON by default, W&B if you set
``logger=wandb``).

Usage:
    uv run python scripts/run_preliminary.py
    uv run python scripts/run_preliminary.py --logger wandb
    uv run python scripts/run_preliminary.py --subjects 1 2 3 --protocols loso within_cv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (experiment-name, protocol-shorthand) tuples that fill the milestone table.
BASELINES = [
    "bradberry",
    "ridge_bandpower",
    "eegnet",
    "shallowconvnet",
]
PROTOCOL_TO_EXPERIMENT_SUFFIX = {
    "loso": "loso",
    "within_cv": "within_cv",
}


def run_one(experiment: str, subjects: list[int], series: list[int], logger: str) -> int:
    cmd = [
        sys.executable, "-m", "src.runner",
        f"+experiment={experiment}",
        f"logger={logger}",
        f"dataset.subjects=[{','.join(str(s) for s in subjects)}]",
        f"dataset.series=[{','.join(str(s) for s in series)}]",
    ]
    print("\n" + "=" * 70)
    print("$", " ".join(cmd))
    print("=" * 70)
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--series", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--baselines", nargs="+", default=BASELINES, choices=BASELINES)
    ap.add_argument(
        "--protocols", nargs="+", default=list(PROTOCOL_TO_EXPERIMENT_SUFFIX),
        choices=list(PROTOCOL_TO_EXPERIMENT_SUFFIX),
    )
    ap.add_argument("--logger", default="offline", choices=["offline", "wandb"])
    args = ap.parse_args()

    nonzero = 0
    for proto in args.protocols:
        suffix = PROTOCOL_TO_EXPERIMENT_SUFFIX[proto]
        for baseline in args.baselines:
            # Within-CV experiment configs only exist for a subset of baselines
            # right now; skip silently if the preset is missing.
            preset = f"baseline_{baseline}_{suffix}"
            preset_path = REPO_ROOT / "src" / "configs" / "experiment" / f"{preset}.yaml"
            if not preset_path.exists():
                print(f"[skip] {preset} (no preset)")
                continue
            rc = run_one(preset, args.subjects, args.series, args.logger)
            if rc != 0:
                print(f"[fail] {preset} exited {rc}")
                nonzero += 1
    return 0 if nonzero == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
