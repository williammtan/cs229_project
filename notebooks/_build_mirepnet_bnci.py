"""Build mirepnet_kmin_bnci2014001.ipynb: MIRepNet on BNCI2014001 (BCI IV-2a),
full 9-subject leave-one-subject-out + K-min calibration, comparing a linear
probe vs. the 2-stage convex head on frozen MIRepNet features.

Dataset is loaded with MOABB (the repo's dataset names ARE MOABB classes).
Channel formatting follows MIRepNet exactly: EA on the native 22-ch montage, then
inverse-distance interpolation to the 45-ch template (pad_missing_channels_diff).
The convex ADMM head cell is copied verbatim from reve_kmin_convexnn_v3.ipynb.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent
v3 = json.loads((NB / "reve_kmin_convexnn_v3.ipynb").read_text())
CONVEX_HEAD_SRC = "".join(v3["cells"][8]["source"])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


C_TITLE = r'''# MIRepNet on BNCI2014001 (BCI IV-2a) - full LOSO + K-min calibration

Same protocol idea as `mirepnet_kmin.ipynb`, but on **BNCI2014001** (BCI
Competition IV dataset 2a) instead of EEGMMI, and with a **full
leave-one-subject-out** sweep: the dataset has only 9 subjects, so each subject
is held out once (source = the other 8) and gets the K-trial calibration sweep on
its own fixed eval set. Two heads on the frozen 256-d MIRepNet feature: a linear
probe and the 2-stage convex-NN head.

Unlike EEGMMI, **BNCI2014001 is *not* MIRepNet's `PHYSIONETMI` pretraining
source**, so this is a more honest cross-dataset evaluation (it is also the
"<30 trials/class" few-shot regime MIRepNet targets - hence K up to 30 here).

## Loader: MOABB

The repo's dataset names (`BNCI2014001`, `BNCI2014004`, `AlexMI`, `Lee`, ...) are
exactly MOABB dataset classes, so we load via MOABB's `MotorImagery` paradigm -
it handles download, epoching, band-pass and resampling uniformly.

## BNCI2014001 -> MIRepNet input format (verified against the data)

| field | value |
|---|---|
| subjects / classes | 9 subjects, 4 classes (left_hand, right_hand, feet, tongue), **144 trials/class/subject** (2 sessions x 288) |
| native montage | **22 EEG channels** = `BNCI2014001_chn_names` (Fz, FC3..FC4, C5..C6, CP3..CP4, P1/Pz/P2, POz) |
| band-pass / resample | **8-30 Hz**, **250 Hz** (`MotorImagery(fmin=8, fmax=30, resample=250)`) |
| window | dataset interval **[2, 6] s** -> 1001 samples, trimmed to **1000** (4 s) |
| distribution alignment | **Euclidean Alignment** on the native 22 ch, per subject |
| channel template | EA'd 22 ch -> **45-ch template** via inverse-distance interpolation (`pad_missing_channels_diff`); 21 of 45 are present, 24 are interpolated |
| model / feature | MIRepNet (emb 256, depth 6); **256-d** mean-pooled `pooled` feature |

The EA-then-interpolate order matches MIRepNet's `process_and_replace_loader`.
'''

C_CONFIG = r'''from __future__ import annotations

import sys, os, json, time, warnings
from pathlib import Path

import numpy as np
import torch
warnings.filterwarnings("ignore")

REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name == "notebooks":
    REPO_ROOT = REPO_ROOT.parent
CLD_DIR = REPO_ROOT / "vendor" / "CLD"
MIREPNET_VENDOR = REPO_ROOT / "vendor" / "MIRepNet"
FEAT_CACHE_DIR = REPO_ROOT / "data" / "cache" / "mirepnet_bnci2014001_feats"
assert CLD_DIR.exists(), f"missing CLD vendor dir: {CLD_DIR}"
assert (MIREPNET_VENDOR / "mirepnet_model.py").exists() and (MIREPNET_VENDOR / "mirepnet_align.py").exists()

# --- protocol ---
# Full LOSO: every subject is held out once. Per held-out subject, a K-per-class
# calibration sweep on a fixed eval set (constant across K, drawn disjoint from
# the calibration pool). BNCI2014001 has ~144 trials/class, so we reserve a
# generous eval set and probe K up to 30 (MIRepNet's "<30 trials/class" regime).
K_BUDGETS      = (0, 1, 2, 5, 10, 20, 30)   # trials per class; K=0 = pure LOSO
EVAL_PER_CLASS = 40                          # fixed held-out eval trials per class
SEED           = 0

# --- MIRepNet preprocessing spec ---
N_CLASSES    = 4
TARGET_FS    = 250
BANDPASS     = (8.0, 30.0)
TRIAL_SAMPLES = 1000                         # 4 s at 250 Hz (interval [2,6] -> trim 1001 to 1000)
FEATURE_DIM  = 256
N_CHANNELS   = 45                            # MIRepNet template (after interpolation)
LABEL_MAP = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")

MIREPNET_DEVICE = "cpu"   # 5.2M params; CPU is fine and avoids accelerator contention

# --- ConvexNN head hparams (same as the REVE/EEGMMI notebooks) ---
CVX_N_NEURONS  = 16
CVX_BETA       = 1.0e-3
CVX_RHO        = 0.1
CVX_ADMM_ITERS = 8
CVX_PCG_ITERS  = 32
CVX_RANK       = 20
STAGE2_ADMM_ITERS = 4
STAGE2_TARGET_MASS = 0.35
STAGE2_WARM_START_DUAL = False

# --- Linear probe hparams ---
LINPROBE_C = 1.0
LINPROBE_MAX_ITER = 2000
LINPROBE_TARGET_MASS = STAGE2_TARGET_MASS

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"torch device={DEVICE}  mirepnet_device={MIREPNET_DEVICE}  |  repo={REPO_ROOT}")
'''

C_LOAD = r'''import moabb
from moabb.paradigms import MotorImagery
try:
    from moabb.datasets import BNCI2014_001 as _BNCI
except Exception:
    from moabb.datasets import BNCI2014001 as _BNCI

if str(MIREPNET_VENDOR) not in sys.path:
    sys.path.insert(0, str(MIREPNET_VENDOR))
from mirepnet_align import use_channels_names, BNCI2014001_chn_names, channel_positions, EA, pad_missing_channels_diff

DATASET = _BNCI()
PARADIGM = MotorImagery(n_classes=N_CLASSES, fmin=BANDPASS[0], fmax=BANDPASS[1], resample=float(TARGET_FS))
SUBJECTS = list(DATASET.subject_list)
print(f"dataset={DATASET.code}  subjects={SUBJECTS}  interval={DATASET.interval}")
print(f"template channels needing interpolation: "
      f"{sum(1 for c in use_channels_names if c not in BNCI2014001_chn_names)}/45")

def get_subject_raw(subj: int):
    """Return (X_native (n,22,1000), y_int (n,), native_ch_names) for one subject."""
    epochs, y, _meta = PARADIGM.get_data(dataset=DATASET, subjects=[subj], return_epochs=True)
    native = [c.upper() for c in epochs.ch_names]
    X = epochs.get_data(copy=False).astype(np.float32)[:, :, :TRIAL_SAMPLES]
    yi = np.asarray([LABEL_MAP[str(v)] for v in y], dtype=np.int64)
    return X, yi, native
'''

C_EXTRACT = r'''from mirepnet_model import mlm_mask
from huggingface_hub import hf_hub_download

def build_mirepnet():
    model = mlm_mask(emb_size=256, depth=6, n_classes=N_CLASSES, pretrainmode=False)
    try:
        ckpt = hf_hub_download(repo_id="starself/MIRepNet", filename="MIRepNet.pth")
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        msd = model.state_dict()
        overlap = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        res = model.load_state_dict(overlap, strict=False)
        print(f"MIRepNet: loaded {len(overlap)}/{len(sd)} pretrained tensors (missing={res.missing_keys})")
    except Exception as e:
        warnings.warn(f"MIRepNet pretrained load failed ({type(e).__name__}: {e}); using random init.")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(MIREPNET_DEVICE)

@torch.no_grad()
def encode_subject(model, X_native: np.ndarray, native: list[str], batch_size: int = 64) -> np.ndarray:
    """EA the native 22-ch trials (per subject), interpolate to the 45-ch template,
    then take MIRepNet's 256-d pooled feature."""
    Xea = EA(X_native.astype(np.float64))                                  # (n, 22, 1000)
    Xt = pad_missing_channels_diff(Xea, use_channels_names, native).astype(np.float32)  # (n, 45, 1000)
    feats = []
    for i in range(0, len(Xt), batch_size):
        xb = torch.from_numpy(Xt[i:i + batch_size]).to(MIREPNET_DEVICE)
        pooled, _ = model(xb)
        feats.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, FEATURE_DIM), np.float32)
'''

C_ENCODE = r'''FEAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_FEAT_CFG = f"mirepnet_bnci2014001_C{N_CHANNELS}_T{TRIAL_SAMPLES}_fs{TARGET_FS}_bp{BANDPASS[0]:g}-{BANDPASS[1]:g}_EA_interp"

model = build_mirepnet()
feature_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

def get_features(subj: int) -> tuple[np.ndarray, np.ndarray]:
    if subj in feature_cache:
        return feature_cache[subj]
    cache_path = FEAT_CACHE_DIR / f"S{subj:02d}.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=False)
        if "cfg" in d and str(d["cfg"]) == _FEAT_CFG:
            feature_cache[subj] = (d["X"].astype(np.float32), d["y"].astype(np.int64))
            return feature_cache[subj]
    t0 = time.time()
    X_native, y, native = get_subject_raw(subj)
    X = encode_subject(model, X_native, native)
    np.savez(cache_path, X=X, y=y, cfg=_FEAT_CFG)
    feature_cache[subj] = (X, y)
    print(f"  S{subj:02d}: {len(y):3d} trials, feats={X.shape}, class counts={np.bincount(y, minlength=N_CLASSES).tolist()}, encode={time.time()-t0:.1f}s")
    return X, y

for s in SUBJECTS:
    get_features(s)
print("encoded all subjects.")
'''

C_SPLIT = r'''def make_subject_split(y: np.ndarray, n_classes: int = N_CLASSES,
                       n_eval_per_class: int = EVAL_PER_CLASS, seed: int = 0):
    """Fixed eval set + ordered calibration pool for one subject."""
    rng = np.random.default_rng(seed)
    eval_idx: list[int] = []
    pool_by_class: list[np.ndarray] = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        perm = rng.permutation(idx)
        if len(perm) <= n_eval_per_class:
            raise ValueError(f"class {c}: only {len(perm)} trials, need > {n_eval_per_class}")
        eval_idx.extend(perm[:n_eval_per_class].tolist())
        pool_by_class.append(perm[n_eval_per_class:])
    return np.asarray(sorted(eval_idx), dtype=np.int64), pool_by_class

def take_calib(pool_by_class: list[np.ndarray], k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    calib: list[int] = []
    for pool_c in pool_by_class:
        calib.extend(pool_c[:k].tolist())
    return np.asarray(sorted(calib), dtype=np.int64)

K_MAX = max(K_BUDGETS)
subject_splits: dict[int, dict] = {}
for s in SUBJECTS:
    _, ys = feature_cache[s]
    eval_idx, pool_by_class = make_subject_split(ys, seed=SEED + s)
    pool_min = min(len(p) for p in pool_by_class)
    if pool_min < K_MAX:
        warnings.warn(f"S{s:02d}: calib pool min={pool_min} < K_MAX={K_MAX}")
    subject_splits[s] = dict(eval_idx=eval_idx, pool_by_class=pool_by_class)
print(f"eval set = {EVAL_PER_CLASS*N_CLASSES} trials/subject; calib pool/class ~ "
      f"{min(len(p) for p in subject_splits[SUBJECTS[0]]['pool_by_class'])}; K_MAX={K_MAX}")
'''

C_SWEEP = r'''import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fit_logreg(Xs, y, sample_weight=None):
    clf = LogisticRegression(C=LINPROBE_C, max_iter=LINPROBE_MAX_ITER)
    clf.fit(Xs, y, sample_weight=sample_weight)
    return clf

lin_rows, cvx_rows = [], []
for ho in SUBJECTS:
    src_ids = [s for s in SUBJECTS if s != ho]
    X_src = np.concatenate([feature_cache[s][0] for s in src_ids], axis=0)
    y_src = np.concatenate([feature_cache[s][1] for s in src_ids], axis=0)

    # --- fit both source models for this fold ---
    lp_scaler = StandardScaler().fit(X_src)
    Xs_src = lp_scaler.transform(X_src).astype(np.float32)
    src_probe = fit_logreg(Xs_src, y_src)
    src_model, src_scaler = fit_stage1_source(X_src, y_src, n_classes=N_CLASSES)

    Xh, yh = feature_cache[ho]
    sp = subject_splits[ho]
    e_idx = sp["eval_idx"]
    Xe, ye = Xh[e_idx], yh[e_idx]
    Xe_lin = lp_scaler.transform(Xe).astype(np.float32)

    for k in K_BUDGETS:
        c_idx = take_calib(sp["pool_by_class"], k)
        yc = yh[c_idx]
        n_calib = int(len(c_idx))
        # linear probe
        if k == 0 or len(np.unique(yc)) < 2:
            lin_clf, lin_tag = src_probe, "src(K=0)"
        else:
            Xc = lp_scaler.transform(Xh[c_idx]).astype(np.float32)
            m = float(np.clip(LINPROBE_TARGET_MASS, 1.0e-3, 0.95))
            w = np.concatenate([np.full(len(Xs_src), (1.0 - m) / len(Xs_src)),
                                np.full(len(Xc), m / len(Xc))]).astype(np.float64)
            lin_clf = fit_logreg(np.concatenate([Xs_src, Xc], 0),
                                 np.concatenate([y_src, yc], 0), sample_weight=w)
            lin_tag = f"anchored,m={m:.2f}"
        lin_acc = float((lin_clf.predict(Xe_lin) == ye).mean())
        lin_rows.append(dict(head="linear", held_out=ho, k=k, n_calib=n_calib,
                             n_eval=int(len(ye)), acc=lin_acc, tag=lin_tag))
        # convex head
        if k == 0 or len(np.unique(yc)) < 2:
            yhat = convex_nn_predict(src_model, src_scaler, Xe); cvx_tag = "src(K=0)"
        else:
            sm = fit_stage2_source_anchored(X_src, y_src, Xh[c_idx], yc, src_model, src_scaler, n_classes=N_CLASSES)
            yhat = convex_nn_predict(sm, src_scaler, Xe); cvx_tag = f"anchored,r={sm.calib_repeat}"
        cvx_acc = float((yhat == ye).mean())
        cvx_rows.append(dict(head="convex", held_out=ho, k=k, n_calib=n_calib,
                             n_eval=int(len(ye)), acc=cvx_acc, tag=cvx_tag))
        print(f"  S{ho:02d} K={k:>3d} n_eval={len(ye):>3d}  linear={lin_acc:.3f}  convex={cvx_acc:.3f}")

lin_df = pd.DataFrame(lin_rows)
df = pd.DataFrame(cvx_rows)

def _summ(d):
    s = (d.groupby("k")["acc"].agg(["mean", "std", "count"])
         .rename(columns={"mean": "acc_mean", "std": "acc_std", "count": "n_subjects"}).reset_index())
    s["sem"] = s["acc_std"] / np.sqrt(s["n_subjects"].clip(lower=1))
    return s

lin_summary = _summ(lin_df)
summary = _summ(df)
print(f"\nLOSO (K=0): linear={lin_summary.loc[lin_summary.k==0,'acc_mean'].iat[0]:.3f}  "
      f"convex={summary.loc[summary.k==0,'acc_mean'].iat[0]:.3f}  (chance={1/N_CLASSES:.3f})")
summary
'''

C_PLOT = r'''import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4.4))
ax.errorbar(lin_summary["k"], lin_summary["acc_mean"], yerr=lin_summary["sem"],
            marker="o", capsize=3, label="MIRepNet + linear probe")
ax.errorbar(summary["k"], summary["acc_mean"], yerr=summary["sem"],
            marker="s", capsize=3, label="MIRepNet + ConvexNN (2-step)")
ax.axhline(1.0 / N_CLASSES, ls="--", color="grey", label=f"chance ({N_CLASSES}-class)")
ax.set_xlabel("K calibration trials per class")
ax.set_ylabel(f"held-out accuracy (fixed {EVAL_PER_CLASS}/class eval set)")
ax.set_title(f"BNCI2014001 (BCI IV-2a) - MIRepNet full LOSO + K-min\n"
             f"({len(SUBJECTS)}-fold leave-one-subject-out)")
ax.set_xticks(list(K_BUDGETS))
ax.legend()
fig.tight_layout()
fig.show()

comparison = lin_summary[["k", "acc_mean", "sem", "n_subjects"]].rename(columns={"acc_mean": "linear_acc", "sem": "linear_sem"})
comparison = comparison.merge(
    summary[["k", "acc_mean", "sem"]].rename(columns={"acc_mean": "convex_acc", "sem": "convex_sem"}), on="k")
comparison
'''

C_SAVE = r'''out_dir = REPO_ROOT / "results" / "mirepnet_bnci2014001_kmin_nb"
out_dir.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
lin_df.to_csv(out_dir / f"linear_per_split_{stamp}.csv", index=False)
lin_summary.to_csv(out_dir / f"linear_summary_{stamp}.csv", index=False)
df.to_csv(out_dir / f"convex_per_split_{stamp}.csv", index=False)
summary.to_csv(out_dir / f"convex_summary_{stamp}.csv", index=False)
comparison.to_csv(out_dir / f"comparison_{stamp}.csv", index=False)
print("wrote:", out_dir)
comparison
'''

cells = [
    md(C_TITLE),
    md("## 0. Config"),
    code(C_CONFIG),
    md("## 1. Load BNCI2014001 via MOABB + MIRepNet channel template"),
    code(C_LOAD),
    md("## 2. MIRepNet feature extractor (EA on 22 ch -> interpolate to 45 -> 256-d)"),
    code(C_EXTRACT),
    md("## 3. Convex two-layer ReLU MLP head with warm-startable ADMM (backbone-agnostic)"),
    code(CONVEX_HEAD_SRC),
    md("## 4. Encode all 9 subjects once (features disk-cached)"),
    code(C_ENCODE),
    md("## 5. Fixed per-subject eval split"),
    code(C_SPLIT),
    md("## 6. Full LOSO + K-min sweep (linear probe + convex head, 9 folds)"),
    code(C_SWEEP),
    md("## 7. Aggregate + comparison plot"),
    code(C_PLOT),
    md("## 8. Save artifacts"),
    code(C_SAVE),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "etm_clf", "language": "python", "name": "etm_clf"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = NB / "mirepnet_kmin_bnci2014001.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
