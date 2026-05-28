"""Run the full baseline grid: classical baselines + FM frozen-linear + FM finetune.

Each experiment runs as a separate Hydra invocation; per-split metrics are
written to results/offline_runs/ via the OfflineLogger. After all runs finish,
the script aggregates everything into a single summary table.

Usage:
    uv run python scripts/run_full_baselines.py                       # all subjects
    uv run python scripts/run_full_baselines.py --subjects 1 2 3      # smoke
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

# LOSO experiments (zero per-user calibration), cheap -> expensive.
EXPERIMENTS = [
    # ---- classical baselines (Week 2) ----
    "baseline_csp_lda_loso",
    "baseline_riemann_lr_loso",
    "baseline_eegnet_loso",
    "baseline_shallowconvnet_loso",
    # ---- frozen FM + softmax probe (Week 3) ----
    "cbramod_frozen_linear_loso",
    "labram_frozen_linear_loso",
    "reve_frozen_linear_loso",
    # ---- full FM finetune ----
    "cbramod_finetune_loso",
    "labram_finetune_loso",
    "reve_finetune_loso",
    # ---- RA in front of every backbone (RED) ----
    "csp_lda_ra_static_loso",
    "riemann_lr_ra_static_loso",
    "eegnet_ra_static_loso",
    "cbramod_frozen_linear_ra_static_loso",
    "cbramod_frozen_linear_ra_ema_loso",
    "labram_frozen_linear_ra_static_loso",
    "labram_frozen_linear_ra_ema_loso",
    "reve_frozen_linear_ra_static_loso",
    "reve_frozen_linear_ra_ema_loso",
    # ---- Convex 2-layer NN head on frozen FM embeddings (YELLOW) ----
    "cbramod_frozen_convexnn_loso",
    "labram_frozen_convexnn_loso",
    "reve_frozen_convexnn_loso",
    "cbramod_frozen_convexnn_ra_static_loso",  # RED + YELLOW combo
]

# K-trials calibration-budget sweeps.
KMIN_EXPERIMENTS = [
    "cbramod_frozen_linear_kmin",
    "labram_frozen_linear_kmin",
    "reve_frozen_linear_kmin",
    "cbramod_frozen_linear_ra_static_kmin",
    "labram_frozen_linear_ra_static_kmin",
    "reve_frozen_linear_ra_static_kmin",
    "cbramod_frozen_linear_ra_ema_kmin",
    "labram_frozen_linear_ra_ema_kmin",
    "reve_frozen_linear_ra_ema_kmin",
    "cbramod_frozen_convexnn_kmin",  # YELLOW K-trials
    # RA-static on traditional (non-FM) backbones — calibration-budget curves
    # to compare against the FM-frozen rows above.
    "csp_lda_ra_static_kmin",
    "riemann_lr_ra_static_kmin",
    "eegnet_ra_static_kmin",
    "shallowconvnet_ra_static_kmin",
]

# Within-subject leave-one-session-out (LOSesionO) — the canonical EEGMMI
# within-subject metric. 3 folds per subject; train on 2 imagery session pairs,
# test on the held-out one. See src/protocols/leave_one_session_out.py.
WITHIN_SUBJECT_EXPERIMENTS = [
    "baseline_csp_lda_lso",
    "baseline_riemann_lr_lso",
    "baseline_eegnet_lso",
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
        accs, kappas, walls = [], [], []
        errors = 0
        for r in group_rows:
            s = r["summary"]
            if "error" in s:
                errors += 1
                continue
            accs.append(s.get("eval/accuracy", float("nan")))
            kappas.append(s.get("eval/cohen_kappa", float("nan")))
            walls.append(s.get("wall_sec", float("nan")))
        accs = np.array(accs, dtype=float)
        kappas = np.array(kappas, dtype=float)
        walls = np.array(walls, dtype=float)
        out[group] = {
            "n": len(group_rows),
            "n_ok": int(np.isfinite(accs).sum()),
            "errors": errors,
            "acc_mean": float(np.nanmean(accs)) if accs.size else float("nan"),
            "acc_std": float(np.nanstd(accs)) if accs.size else float("nan"),
            "kappa_mean": float(np.nanmean(kappas)) if kappas.size else float("nan"),
            "wall_sec_per_fold_mean": float(np.nanmean(walls)) if walls.size else float("nan"),
        }
    return out


def print_table(agg: dict) -> None:
    header = f"{'GROUP':50s}  {'N':>3s} {'Acc(mean)':>10s} {'Acc(std)':>9s} {'kappa':>8s} {'sec/fold':>10s}"
    print()
    print("=" * 76)
    print("AGGREGATE SUMMARY")
    print("=" * 76)
    print(header)
    print("-" * len(header))
    rows = sorted(agg.items(), key=lambda kv: -kv[1]["acc_mean"] if np.isfinite(kv[1]["acc_mean"]) else 1)
    for g, v in rows:
        print(
            f"{g:50s}  {v['n_ok']:>3d} "
            f"{v['acc_mean']:>10.4f} {v['acc_std']:>9.4f} "
            f"{v['kappa_mean']:>+8.3f} "
            f"{v['wall_sec_per_fold_mean']:>10.1f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--runs", type=int, nargs="*", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--no-clear", action="store_true")
    ap.add_argument("--logger", default="offline", choices=["offline", "wandb"])
    ap.add_argument(
        "--suite", default="loso",
        choices=["loso", "kmin", "within_subject", "all"],
        help="Which experiment list to run.",
    )
    args = ap.parse_args()

    if not args.no_clear and OFFLINE_ROOT.exists():
        import shutil
        shutil.rmtree(OFFLINE_ROOT)
        print(f"[clean] removed {OFFLINE_ROOT}")

    overrides = [f"logger={args.logger}"]
    if args.subjects is not None:
        overrides.append(f"dataset.subjects=[{','.join(str(s) for s in args.subjects)}]")
    if args.runs is not None:
        overrides.append(f"dataset.runs=[{','.join(str(r) for r in args.runs)}]")

    if args.only:
        to_run = list(args.only)
    elif args.suite == "loso":
        to_run = list(EXPERIMENTS)
    elif args.suite == "kmin":
        to_run = list(KMIN_EXPERIMENTS)
    elif args.suite == "within_subject":
        to_run = list(WITHIN_SUBJECT_EXPERIMENTS)
    else:  # all
        to_run = list(EXPERIMENTS) + list(KMIN_EXPERIMENTS) + list(WITHIN_SUBJECT_EXPERIMENTS)
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
