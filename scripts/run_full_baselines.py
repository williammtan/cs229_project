"""Run the full baseline grid: classical baselines + FM frozen-linear + FM finetune.

Each experiment runs as a separate Hydra invocation; per-split metrics are
written to results/offline_runs/ via the OfflineLogger. After all runs finish,
the script aggregates everything into a single summary table.

Usage:
    uv run python scripts/run_full_baselines.py                 # all 12 LOSO folds
    uv run python scripts/run_full_baselines.py --subjects 1 2  # smoke (2 folds)
    uv run python scripts/run_full_baselines.py --skip reve_finetune_loso
    uv run python scripts/run_full_baselines.py --only cbramod_frozen_linear_loso
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OFFLINE_ROOT = REPO / "results" / "offline_runs"

# LOSO experiments (zero per-user calibration) in execution order — cheap -> expensive.
EXPERIMENTS = [
    "baseline_bradberry_loso",
    "baseline_ridge_bandpower_loso",
    "baseline_eegnet_loso",
    "baseline_shallowconvnet_loso",
    "cbramod_frozen_linear_loso",
    "labram_frozen_linear_loso",
    "reve_frozen_linear_loso",
    "cbramod_finetune_loso",
    "labram_finetune_loso",
    "reve_finetune_loso",
    # Riemannian-adapter LOSO variants
    "bradberry_ra_static_loso",
    "eegnet_ra_static_loso",
    "cbramod_frozen_linear_ra_static_loso",
    "cbramod_frozen_linear_ra_ema_loso",
]

# k-min calibration-budget sweeps (different protocol; aggregated separately).
# Each FM has a plain and an RA-static variant so we can compare calibration
# curves with/without Riemannian alignment.
KMIN_EXPERIMENTS = [
    "cbramod_frozen_linear_kmin",
    "labram_frozen_linear_kmin",
    "reve_frozen_linear_kmin",
    "cbramod_frozen_linear_ra_static_kmin",
    "labram_frozen_linear_ra_static_kmin",
    "reve_frozen_linear_ra_static_kmin",
]


def run_experiment(name: str, overrides: list[str]) -> tuple[int, float]:
    cmd = [
        sys.executable, "-W", "ignore", "-m", "src.runner",
        f"+experiment={name}",
        f"hydra.run.dir=results/_hydra/{name}",
        *overrides,
    ]
    print()
    print("=" * 76)
    print(f"[{time.strftime('%H:%M:%S')}] launching: {name}")
    print("  " + " ".join(cmd))
    print("=" * 76)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=REPO)
    return rc, time.time() - t0


def load_all_summaries() -> list[dict]:
    """Walk offline_runs once and return every (meta, summary) pair."""
    if not OFFLINE_ROOT.exists():
        return []
    rows = []
    for run_dir in sorted(OFFLINE_ROOT.iterdir()):
        meta_path = run_dir / "meta.json"
        summary_path = run_dir / "summary.json"
        if not (meta_path.exists() and summary_path.exists()):
            continue
        meta = json.loads(meta_path.read_text())
        summary = json.loads(summary_path.read_text())
        rows.append({
            "group": meta.get("group", ""),
            "job_type": meta.get("job_type"),
            "tags": meta.get("tags", []),
            "summary": summary,
            "run_dir": str(run_dir),
        })
    return rows


def aggregate(rows: list[dict]) -> dict:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)

    out = {}
    for group, group_rows in by_group.items():
        rs, r2s, rmses, walls = [], [], [], []
        errors = 0
        for r in group_rows:
            s = r["summary"]
            if "error" in s:
                errors += 1
                continue
            rs.append(s.get("eval/pearson_r/mean", float("nan")))
            r2s.append(s.get("eval/r2/mean", float("nan")))
            rmses.append(s.get("eval/rmse/mean", float("nan")))
            walls.append(s.get("wall_sec", float("nan")))
        rs = np.array(rs, dtype=float)
        r2s = np.array(r2s, dtype=float)
        rmses = np.array(rmses, dtype=float)
        walls = np.array(walls, dtype=float)
        out[group] = {
            "n": len(group_rows),
            "n_ok": int(np.isfinite(rs).sum()),
            "errors": errors,
            "r_mean": float(np.nanmean(rs)) if rs.size else float("nan"),
            "r_std": float(np.nanstd(rs)) if rs.size else float("nan"),
            "r2_mean": float(np.nanmean(r2s)) if r2s.size else float("nan"),
            "rmse_mean": float(np.nanmean(rmses)) if rmses.size else float("nan"),
            "wall_sec_per_fold_mean": float(np.nanmean(walls)) if walls.size else float("nan"),
        }
    return out


def print_table(agg: dict) -> None:
    header = f"{'GROUP':50s}  {'N':>3s} {'R(mean)':>9s} {'R(std)':>8s} {'R2':>8s} {'RMSE':>8s} {'sec/fold':>10s}"
    print()
    print("=" * 76)
    print("AGGREGATE SUMMARY")
    print("=" * 76)
    print(header)
    print("-" * len(header))
    rows = sorted(agg.items(), key=lambda kv: -kv[1]["r_mean"] if np.isfinite(kv[1]["r_mean"]) else 1)
    for g, v in rows:
        print(
            f"{g:50s}  {v['n_ok']:>3d} "
            f"{v['r_mean']:>+9.4f} {v['r_std']:>8.4f} "
            f"{v['r2_mean']:>+8.4f} {v['rmse_mean']:>8.4f} "
            f"{v['wall_sec_per_fold_mean']:>10.1f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, nargs="*", default=None,
                    help="Override dataset.subjects (default: use config)")
    ap.add_argument("--series", type=int, nargs="*", default=None,
                    help="Override dataset.series (default: use config)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these experiments (by name)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Skip these experiments")
    ap.add_argument("--no-clear", action="store_true",
                    help="Don't wipe results/offline_runs/ before launch")
    ap.add_argument("--logger", default="offline", choices=["offline", "wandb"],
                    help="Logger backend (W&B requires `wandb login` first)")
    args = ap.parse_args()

    if not args.no_clear and OFFLINE_ROOT.exists():
        import shutil
        shutil.rmtree(OFFLINE_ROOT)
        print(f"[clean] removed {OFFLINE_ROOT}")

    overrides = [f"logger={args.logger}"]
    if args.subjects is not None:
        overrides.append(f"dataset.subjects=[{','.join(str(s) for s in args.subjects)}]")
    if args.series is not None:
        overrides.append(f"dataset.series=[{','.join(str(s) for s in args.series)}]")

    to_run = args.only if args.only else EXPERIMENTS
    to_run = [n for n in to_run if n not in args.skip]

    timings: dict[str, float] = {}
    failures: list[str] = []
    for name in to_run:
        rc, secs = run_experiment(name, overrides)
        timings[name] = secs
        if rc != 0:
            failures.append(name)
            print(f"  -> exit code {rc} ({secs:.1f}s)")
        else:
            print(f"  -> done ({secs:.1f}s)")

    # Aggregate from offline_runs.
    agg = aggregate(load_all_summaries())
    print_table(agg)

    print()
    print("Wall time per experiment:")
    for n, s in timings.items():
        print(f"  {n:50s}  {s:>8.1f}s  ({s/60:>5.1f} min)")
    print(f"  TOTAL:  {sum(timings.values())/60:.1f} min")
    if failures:
        print(f"\nFAILED experiments: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
