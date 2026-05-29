"""Build reve_kmin_convexnn_v4.ipynb from v3, swapping in the v4 cells.

v4 changes vs v3:
  * Fixed per-subject eval set, shared by both methods, constant across K.
  * Naive single-stage EEGNet: train ONCE on concatenated [source + K calib],
    no pretrain->finetune, no best-checkpoint, no calibration up-weighting.
  * K_BUDGETS topped at 10 (fixed eval set leaves <=11 trials/class to calibrate).
  * REVE feature disk cache so reruns are cheap.
Unchanged v3 cells are copied verbatim (avoids re-quoting their docstrings).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent
v3 = json.loads((NB / "reve_kmin_convexnn_v3.ipynb").read_text())
v3_cells = v3["cells"]


def vsrc(i):
    return "".join(v3_cells[i]["source"])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


# ---------------------------------------------------------------- new sources --
C_TITLE = r'''# REVE + Convex-NN head vs. naive EEGNet - K-trial calibration sweep (v4, fixed eval set)

Head-to-head on the EEGMMI 4-class motor-imagery task:

1. **REVE + source-anchored 2-step convex-NN head.** Frozen REVE features ->
   attention pooling -> a convex two-layer ReLU MLP fit with warm-started ADMM.
   Stage 1 solves on the source pool; stage 2 re-solves on `source + weighted
   K-per-class calibration`, warm-starting the primal weights (same as v3).
2. **Naive EEGNet baseline.** A single EEGNet trained *once* from a fresh random
   init on the **concatenated** dataset `[ all source trials + K-per-class
   calibration ]`. No source-pretrain -> per-subject fine-tune two-step, no
   best-checkpoint selection, no early stopping, no calibration up-weighting -
   just "pool everything and fit once" for a fixed number of epochs.

**What is new in v4 (the point of this notebook):**
- **One fixed eval set per held-out subject, shared by both methods and held
  constant across every K.** v3 used `first-K = calibration, rest = eval`, so the
  eval set shrank as K grew (K=0 -> 90 eval trials, K=20 -> 10), which made the
  high-K accuracies incomparable. v4 reserves `EVAL_PER_CLASS` trials/class as a
  fixed eval set and draws calibration only from the remaining pool. That caps
  the sweep at K=10 (dropping v3's degenerate K=20 cell).
- `K = 0` is plain LOSO (evaluate the source model on a brand-new subject).
- `N_HELD_OUT = 8` subjects chosen at random (SEED=0).
'''

# config cell (cell index 2 in v3)
C_CONFIG = r'''from __future__ import annotations

import sys, os, json, time, warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# --- paths ---
REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name == "notebooks":
    REPO_ROOT = REPO_ROOT.parent
EEGMMI_DIR = REPO_ROOT / "data" / "raw" / "eegmmi"
CLD_DIR = REPO_ROOT / "vendor" / "CLD"
FEAT_CACHE_DIR = REPO_ROOT / "data" / "cache" / "reve_feats"
assert EEGMMI_DIR.exists(), f"missing dataset dir: {EEGMMI_DIR}"
assert CLD_DIR.exists(),    f"missing CLD vendor dir: {CLD_DIR}"

# --- sweep config ---
# K = trials *per class* used for per-subject calibration. K=0 is plain LOSO
# (no calibration; evaluate the source model on a brand-new subject).
#
# IMPORTANT (vs. v3): every K is now scored on the SAME fixed held-out eval set
# per subject. We reserve EVAL_PER_CLASS trials/class for that eval set and draw
# calibration only from the remaining "pool". With ~21 trials in the smallest
# class, reserving 11 for eval leaves >=10 for calibration, so the sweep tops out
# at K=10 (v3's K=20 cell is dropped: it left almost nothing to evaluate on,
# which is exactly the confound we are removing).
K_BUDGETS         = (0, 1, 2, 5, 10)       # trials per class used for calibration
EVAL_PER_CLASS    = 11                      # fixed held-out eval trials per class (shared by both methods)
N_HELD_OUT        = 8                        # subjects held out and calibrated on
SEED              = 0
TRIAL_TMIN        = 0.0
TRIAL_TMAX        = 4.0
TARGET_FS_REVE    = 200
HIGHPASS_HZ       = 0.3
NOTCH_HZ          = 60.0
SCALE_FACTOR      = 100.0
IMAGERY_RUNS      = (4, 6, 8, 10, 12, 14)
EXCLUDED_SUBJECTS = frozenset({88, 89, 92, 100, 104})

# --- ConvexNN head hparams ---
CVX_N_NEURONS  = 16
CVX_BETA       = 1.0e-3
CVX_RHO        = 0.1
CVX_ADMM_ITERS = 8        # iterations for stage-1 (source) solve
CVX_PCG_ITERS  = 32
CVX_RANK       = 20

# Stage 2 is a source-anchored adaptation problem, not a target-only refit.
STAGE2_ADMM_ITERS = 4
STAGE2_TARGET_MASS = 0.35     # target calibration share in the stage-2 weighted loss
STAGE2_WARM_START_DUAL = False  # stale ADMM duals are diagnostic-only for target-shifted problems

# --- ConvexNN HP grid (off by default; the head-to-head comparison is the focus) ---
RUN_CVX_HP_GRID = False
CVX_BETA_GRID = (3.0e-4, 1.0e-3, 3.0e-3)
STAGE2_TARGET_MASS_GRID = (0.15, 0.35, 0.55)
HP_GRID_K_BUDGETS = tuple(k for k in K_BUDGETS if k > 0)

# --- EEGNet baseline hparams (NAIVE single-stage) ---
# Deliberately NOT a two-step (source-pretrain -> per-subject fine-tune) recipe.
# For each (held-out subject, K) we build ONE concatenated dataset
# [ all source trials  +  K-per-class calibration trials ] and train EEGNet once
# from a fresh random init for a fixed number of epochs. No validation split, no
# best-checkpoint selection, no early stopping, no calibration up-weighting - the
# plain "pool everything and fit once" baseline. (K=0 trains on source alone and
# that single source model is reused for every held-out subject.)
RUN_EEGNET_BASELINE = True
EEGNET_BATCH_SIZE = 64
EEGNET_EPOCHS = 40          # fixed; final-epoch weights are used as-is
EEGNET_LR = 1.0e-3
EEGNET_WEIGHT_DECAY = 1.0e-4
EEGNET_DROPOUT = 0.5
# EEGNet branch runs at 100 Hz (downsampled from REVE's 200 Hz) with kernel_len=100
# to match the receptive field implied by the EEGNet paper at fs=200 / src LOSO setup.
EEGNET_TARGET_FS = 100
EEGNET_KERNEL_LEN = 100
EEGNET_TRIAL_SAMPLES = int(round((TRIAL_TMAX - TRIAL_TMIN) * EEGNET_TARGET_FS))

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device={DEVICE}  |  repo={REPO_ROOT}")
'''

# md for section 4 (held-out + encode)
MD_HELDOUT = r'''## 4. Choose held-out subjects, encode everything once (REVE features disk-cached)'''

# encode + disk cache (replaces v3 cell 11)
C_ENCODE = r'''FEAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Cache key encodes the preprocessing config so a stale cache is never reused silently.
_FEAT_CFG = f"reve_base_C{N_CHANNELS}_T{TRIAL_SAMPLES}_fs{TARGET_FS_REVE}_sc{SCALE_FACTOR:g}"

trial_cache: dict[int, list[Trial]] = {}
feature_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

def get_trials(subj: int) -> list[Trial]:
    if subj not in trial_cache:
        trial_cache[subj] = load_subject(subj)
    return trial_cache[subj]

def get_features(subj: int) -> tuple[np.ndarray, np.ndarray]:
    if subj in feature_cache:
        return feature_cache[subj]
    cache_path = FEAT_CACHE_DIR / f"S{subj:03d}.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=False)
        if "cfg" in d and str(d["cfg"]) == _FEAT_CFG:
            feature_cache[subj] = (d["X"].astype(np.float32), d["y"].astype(np.int64))
            return feature_cache[subj]
    t0 = time.time()
    trials = get_trials(subj)
    if not trials:
        feature_cache[subj] = (np.zeros((0, 512), dtype=np.float32), np.zeros((0,), dtype=np.int64))
        return feature_cache[subj]
    X, y = encode_trials(model, trials)
    np.savez(cache_path, X=X, y=y, cfg=_FEAT_CFG)
    feature_cache[subj] = (X, y)
    print(f"  S{subj:03d}: {len(trials):3d} trials, feats={X.shape}, encode={time.time()-t0:.1f}s")
    return X, y

for s in source_subjects + held_out_subjects:
    get_features(s)

Xs_list, ys_list = [], []
for s in source_subjects:
    X, y = feature_cache[s]
    if len(X):
        Xs_list.append(X); ys_list.append(y)
X_src = np.concatenate(Xs_list, axis=0)
y_src = np.concatenate(ys_list, axis=0)
source_trials = [t for s in source_subjects for t in get_trials(s)]
print(f"source pool raw trials: n={len(source_trials)}")
print(f"source pool features: X={X_src.shape}  y={y_src.shape}  class counts={np.bincount(y_src, minlength=4)}")
'''

# md for fixed eval split section
MD_SPLIT = r'''## 5. Fixed per-subject eval split

For each held-out subject we shuffle each class once (seeded by subject) and
reserve the first `EVAL_PER_CLASS` trials/class as the **eval set**. The rest
form the **calibration pool**. The eval set does not depend on K, so every K and
both methods are scored on identical trials; calibration budgets are nested
prefixes of the pool (`K=1` calib is a subset of `K=2` calib, ...). The convex
features (`feature_cache[ho]`) and the raw EEG (`get_trials(ho)`) share trial
order, so the same indices select the same trials in both branches.'''

C_SPLIT = r'''def make_subject_split(y: np.ndarray, n_classes: int = 4,
                       n_eval_per_class: int = EVAL_PER_CLASS, seed: int = 0):
    """Fixed eval set + ordered calibration pool for one subject."""
    rng = np.random.default_rng(seed)
    eval_idx: list[int] = []
    pool_by_class: list[np.ndarray] = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        perm = rng.permutation(idx)
        if len(perm) <= n_eval_per_class:
            raise ValueError(
                f"class {c}: only {len(perm)} trials, need > {n_eval_per_class} for eval+calib"
            )
        eval_idx.extend(perm[:n_eval_per_class].tolist())
        pool_by_class.append(perm[n_eval_per_class:])
    return np.asarray(sorted(eval_idx), dtype=np.int64), pool_by_class

def take_calib(pool_by_class: list[np.ndarray], k: int) -> np.ndarray:
    """First-k-per-class calibration indices drawn from a subject's pool."""
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    calib: list[int] = []
    for pool_c in pool_by_class:
        calib.extend(pool_c[:k].tolist())
    return np.asarray(sorted(calib), dtype=np.int64)

# Build the (fixed) split for every held-out subject once, keyed by subject.
K_MAX = max(K_BUDGETS)
subject_splits: dict[int, dict] = {}
for ho in held_out_subjects:
    _, yh = feature_cache[ho]
    if len(yh) == 0:
        print(f"  S{ho:03d}: no trials; skipping")
        continue
    eval_idx, pool_by_class = make_subject_split(yh, seed=SEED + ho)
    pool_min = min(len(p) for p in pool_by_class)
    if pool_min < K_MAX:
        warnings.warn(f"S{ho:03d}: calib pool min={pool_min} < K_MAX={K_MAX}; "
                      "large-K cells reuse the whole pool")
    subject_splits[ho] = dict(eval_idx=eval_idx, pool_by_class=pool_by_class)
    print(f"  S{ho:03d}: eval={len(eval_idx)} trials  calib pool/class={[len(p) for p in pool_by_class]}")
'''

# md for convex sweep
MD_SWEEP = r'''## 6. Source-anchored 2-step convex sweep (fixed eval set)

Stage 1 is fit once. For every (held-out subject, K > 0) we run a second convex
solve on `source + weighted calibration`, warm-starting the primal variables
from stage 1. The K = 0 cells evaluate the stage-1 model directly. Every cell is
scored on the subject's fixed eval set, so `n_eval` is constant across K.'''

C_SWEEP = r'''print("Stage 1: fitting source convex NN on the full source pool...")
t0 = time.time()
src_model, src_scaler = fit_stage1_source(X_src, y_src)
print(f"  stage-1 solve: {time.time() - t0:.1f}s  (u/v/lam saved for warm-start)")

results = []  # one row per (held_out, k) for the source-anchored two-stage model
src_baseline_accs = []
for ho in held_out_subjects:
    if ho not in subject_splits:
        continue
    Xh, yh = feature_cache[ho]
    sp = subject_splits[ho]
    e_idx = sp["eval_idx"]
    Xe, ye = Xh[e_idx], yh[e_idx]
    for k in K_BUDGETS:
        c_idx = take_calib(sp["pool_by_class"], k)
        Xc, yc = Xh[c_idx], yh[c_idx]
        calib_repeat = 0
        target_mass = 0.0
        if k == 0 or len(np.unique(yc)) < 2:
            yhat = convex_nn_predict(src_model, src_scaler, Xe)
            tag = "src(K=0)"
            fit_t = 0.0
        else:
            t0 = time.time()
            subj_model = fit_stage2_source_anchored(X_src, y_src, Xc, yc, src_model, src_scaler)
            fit_t = time.time() - t0
            yhat = convex_nn_predict(subj_model, src_scaler, Xe)
            tag = f"anchored,r={subj_model.calib_repeat}"
            calib_repeat = int(subj_model.calib_repeat)
            target_mass = float(subj_model.stage2_target_mass)
        acc = float((yhat == ye).mean())
        if k == 0:
            src_baseline_accs.append(acc)
        results.append(dict(held_out=ho, k=k, n_calib=int(len(Xc)),
                            n_eval=int(len(Xe)), acc=acc, tag=tag,
                            calib_repeat=calib_repeat, target_mass=target_mass))
        print(f"  S{ho:03d}  K={k:>3d}  n_calib={len(Xc):>3d}  n_eval={len(Xe):>3d}  "
              f"acc={acc:.3f}  [{tag}, {fit_t:.1f}s]")
print(f"\n  K=0 source-only mean acc: {np.mean(src_baseline_accs):.3f}  (n={len(src_baseline_accs)})")
'''

# md for EEGNet
MD_EEGNET = r'''## 7. Naive EEGNet baseline (single-stage, concatenated training)

For each (held-out subject, K) we build one dataset `[ all source trials +
K-per-class calibration ]` and train EEGNet **once** from a fresh random init for
`EEGNET_EPOCHS` epochs - no source-pretrain -> fine-tune two-step, no
best-checkpoint selection, no calibration up-weighting. K = 0 trains on the
source pool alone and reuses that one model for every subject.

The per-channel scaler is fit on the source pool only (no leakage from
calibration or eval). EEGNet convnets are known to *intermittently* collapse to
chance on this machine's MPS backend (a numerical-instability training collapse,
not a data bug), so training keeps a NaN-loss / NaN-weight / suspicious-train-acc
guard that falls back to CPU for that fit. That guard is robustness, not model
selection - the final-epoch weights are still used as-is.'''

C_EEGNET = r'''import pandas as pd
import sys
sys.path.insert(0, str(REPO_ROOT))

import torch.nn as nn
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, TensorDataset
from src.models import EEGNetClf

_EEGNET_DOWN_FACTOR = TARGET_FS_REVE // EEGNET_TARGET_FS
assert TARGET_FS_REVE == EEGNET_TARGET_FS * _EEGNET_DOWN_FACTOR, (
    f"EEGNet downsample requires integer ratio; got {TARGET_FS_REVE}/{EEGNET_TARGET_FS}"
)

def _downsample_to_eegnet_fs(X: np.ndarray) -> np.ndarray:
    if _EEGNET_DOWN_FACTOR == 1:
        return X.astype(np.float32, copy=False)
    Xd = resample_poly(X, up=1, down=_EEGNET_DOWN_FACTOR, axis=-1).astype(np.float32)
    if Xd.shape[-1] > EEGNET_TRIAL_SAMPLES:
        Xd = Xd[..., :EEGNET_TRIAL_SAMPLES]
    elif Xd.shape[-1] < EEGNET_TRIAL_SAMPLES:
        pad = np.zeros(Xd.shape[:-1] + (EEGNET_TRIAL_SAMPLES,), dtype=Xd.dtype)
        pad[..., -Xd.shape[-1]:] = Xd
        Xd = pad
    return Xd

def trials_to_eeg_xy(trials: list[Trial]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([t.eeg for t in trials], axis=0).astype(np.float32)
    X = _downsample_to_eegnet_fs(X)
    y = np.asarray([t.label for t in trials], dtype=np.int64)
    return X, y

def fit_eeg_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
    return mean, np.maximum(std, 1.0e-4)

def transform_eeg(X: np.ndarray, scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = scaler
    return ((X.astype(np.float32) - mean) / std).astype(np.float32)

def make_eegnet_model() -> torch.nn.Module:
    return EEGNetClf(
        n_channels=N_CHANNELS,
        n_samples=EEGNET_TRIAL_SAMPLES,
        n_classes=4,
        kernel_len=EEGNET_KERNEL_LEN,
        dropout=EEGNET_DROPOUT,
    )

# --- robustness: detect intermittent MPS training collapse -> retry on CPU -----
class _EEGNetNaN(RuntimeError):
    pass

def _model_has_nonfinite(model: torch.nn.Module) -> bool:
    return any(
        not torch.isfinite(t).all()
        for t in model.state_dict().values()
        if t.is_floating_point()
    )

@torch.no_grad()
def _chunked_logits(model: torch.nn.Module, X: np.ndarray, device: torch.device,
                    chunk: int = 256) -> torch.Tensor:
    """Forward in chunks and return CPU logits (never one giant MPS forward)."""
    outs = []
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(X[i:i + chunk]).unsqueeze(1).to(device, non_blocking=True)
        outs.append(model(xb).detach().float().cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty(0, 4)

def _train_once_on_device(model, X, y, device, *, epochs, lr, weight_decay, seed, verbose):
    """One single-stage training run on a fixed device. Final-epoch weights, no
    checkpoint selection. Raises _EEGNetNaN on a detected device collapse."""
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y)),
        batch_size=EEGNET_BATCH_SIZE, shuffle=True, drop_last=False,
    )
    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                raise _EEGNetNaN(f"non-finite train loss at epoch {epoch}")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += float(loss.item()) * len(xb)
        if verbose and (epoch % 10 == 0 or epoch == int(epochs) - 1):
            print(f"    epoch {epoch:03d} train_loss={running / max(1, len(X)):.4f}")
    if _model_has_nonfinite(model):
        raise _EEGNetNaN("non-finite weights after training")
    logits = _chunked_logits(model, X, device)
    if not torch.isfinite(logits).all():
        raise _EEGNetNaN("non-finite train logits")
    train_acc = float((logits.argmax(1) == torch.from_numpy(y)).float().mean().item())
    # Documented MPS collapse yields ~chance train acc even with finite losses.
    if train_acc <= 0.27:
        raise _EEGNetNaN(f"suspicious train acc={train_acc:.3f} (<=chance); treating as device collapse")
    return model, train_acc

def train_eegnet_naive(X: np.ndarray, y: np.ndarray, *, epochs: int = EEGNET_EPOCHS,
                       lr: float = EEGNET_LR, weight_decay: float = EEGNET_WEIGHT_DECAY,
                       seed: int = SEED, verbose: bool = False) -> torch.nn.Module:
    """Train ONE EEGNet from scratch on the given (already standardized) concat set."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.int64)
    model = make_eegnet_model()
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    try:
        model, tr = _train_once_on_device(model, X, y, torch.device(DEVICE),
                                          epochs=epochs, lr=lr, weight_decay=weight_decay,
                                          seed=seed, verbose=verbose)
        if verbose:
            print(f"    [train acc={tr:.3f} on {DEVICE}]")
        return model
    except _EEGNetNaN as err:
        if torch.device(DEVICE).type == "cpu":
            raise
        print(f"  [EEGNet] {DEVICE} collapsed ({err}); retraining on CPU")
        model = make_eegnet_model()
        model.load_state_dict(init_state)
        model, tr = _train_once_on_device(model, X, y, torch.device("cpu"),
                                          epochs=epochs, lr=lr, weight_decay=weight_decay,
                                          seed=seed, verbose=verbose)
        print(f"    [CPU retry train acc={tr:.3f}]")
        return model

@torch.no_grad()
def predict_eegnet(model: torch.nn.Module, X_raw: np.ndarray,
                   scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    model.eval()
    Xs = transform_eeg(X_raw, scaler)
    devices = [torch.device(DEVICE)] + ([torch.device("cpu")] if DEVICE != "cpu" else [])
    logits = None
    for device in devices:
        model = model.to(device)
        logits = _chunked_logits(model, Xs, device)
        if torch.isfinite(logits).all():
            return logits.argmax(1).numpy().astype(np.int64)
        if device.type != "cpu":
            print(f"  [EEGNet] predict produced NaN on {device}; falling back to CPU")
    return logits.argmax(1).numpy().astype(np.int64)

eegnet_df = pd.DataFrame()
eegnet_summary = pd.DataFrame()
if RUN_EEGNET_BASELINE:
    print("EEGNet: preparing raw EEG tensors (downsampled to %d Hz)..." % EEGNET_TARGET_FS)
    X_src_eeg, y_src_eeg = trials_to_eeg_xy(source_trials)
    eeg_scaler = fit_eeg_scaler(X_src_eeg)                 # scaler fit on SOURCE only
    X_src_eeg_s = transform_eeg(X_src_eeg, eeg_scaler)
    del X_src_eeg
    print(f"  source EEG: X_s={X_src_eeg_s.shape}  class counts={np.bincount(y_src_eeg, minlength=4).tolist()}")

    ho_eeg = {}
    for ho in held_out_subjects:
        if ho not in subject_splits:
            continue
        ho_eeg[ho] = trials_to_eeg_xy(get_trials(ho))

    print(f"\nEEGNet K=0: training one source-only model ({EEGNET_EPOCHS} epochs)...")
    t0 = time.time()
    src_eeg_model = train_eegnet_naive(X_src_eeg_s, y_src_eeg, seed=SEED, verbose=True)
    print(f"  source fit: {time.time() - t0:.1f}s")

    eegnet_rows = []
    for k in K_BUDGETS:
        for ho in held_out_subjects:
            if ho not in subject_splits:
                continue
            Xh, yh = ho_eeg[ho]
            sp = subject_splits[ho]
            e_idx = sp["eval_idx"]
            c_idx = take_calib(sp["pool_by_class"], k)
            if k == 0 or len(c_idx) == 0:
                model_k = src_eeg_model            # reuse the single source model
                tag = "source(K=0)"
                fit_t = 0.0
                n_calib = 0
            else:
                # Concatenate ALL source + this subject's K-per-class calib, fit once.
                Xc_s = transform_eeg(Xh[c_idx], eeg_scaler)
                X_cat = np.concatenate([X_src_eeg_s, Xc_s], axis=0)
                y_cat = np.concatenate([y_src_eeg, yh[c_idx]], axis=0)
                t0 = time.time()
                model_k = train_eegnet_naive(X_cat, y_cat, seed=SEED + ho, verbose=False)
                fit_t = time.time() - t0
                tag = "concat"
                n_calib = int(len(c_idx))
            yhat = predict_eegnet(model_k, Xh[e_idx], eeg_scaler)
            acc = float((yhat == yh[e_idx]).mean())
            eegnet_rows.append(dict(held_out=ho, k=k, n_calib=n_calib,
                                    n_eval=int(len(e_idx)), acc=acc, tag=tag))
            print(f"  EEGNet S{ho:03d} K={k:>3d} n_calib={n_calib:>3d} n_eval={len(e_idx):>3d} "
                  f"acc={acc:.3f} [{tag}, {fit_t:.1f}s]")
    eegnet_df = pd.DataFrame(eegnet_rows)
    eegnet_summary = (
        eegnet_df.groupby("k")["acc"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "acc_mean", "std": "acc_std", "count": "n_subjects"})
        .reset_index()
    )
    eegnet_summary["sem"] = eegnet_summary["acc_std"] / np.sqrt(eegnet_summary["n_subjects"].clip(lower=1))
eegnet_summary
'''

# compare plot
C_COMPARE = r'''fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.errorbar(summary["k"], summary["acc_mean"], yerr=summary["sem"],
            marker="o", capsize=3, label="REVE + ConvexNN (2-step)")
if len(eegnet_summary):
    ax.errorbar(eegnet_summary["k"], eegnet_summary["acc_mean"], yerr=eegnet_summary["sem"],
                marker="s", capsize=3, label="EEGNet (naive concat)")
ax.axhline(0.25, ls="--", color="grey", label="chance (4-class)")
ax.set_xlabel("K calibration trials per class")
ax.set_ylabel(f"held-out accuracy (fixed {EVAL_PER_CLASS}/class eval set)")
ax.set_title(f"K-trial calibration: REVE+ConvexNN vs. naive EEGNet\n"
             f"({len(subject_splits)} held-out subj, {len(source_subjects)} source subj)")
ax.set_xticks(list(K_BUDGETS))
ax.legend()
fig.tight_layout()
fig.show()
'''

# HP grid (uses split helpers)
C_HPGRID = r'''def run_convex_hp_grid(
    beta_grid: tuple[float, ...] = CVX_BETA_GRID,
    target_mass_grid: tuple[float, ...] = STAGE2_TARGET_MASS_GRID,
    k_budgets: tuple[int, ...] = HP_GRID_K_BUDGETS,
) -> pd.DataFrame:
    rows = []
    for beta in beta_grid:
        print(f"Grid beta={beta:g}: fitting source model...")
        t0 = time.time()
        grid_src_model, grid_src_scaler = fit_stage1_source(X_src, y_src, beta=beta)
        print(f"  source fit {time.time() - t0:.1f}s")
        for target_mass in target_mass_grid:
            print(f"  target_mass={target_mass:.2f}")
            for ho in held_out_subjects:
                if ho not in subject_splits:
                    continue
                Xh, yh = feature_cache[ho]
                sp = subject_splits[ho]
                Xe, ye = Xh[sp["eval_idx"]], yh[sp["eval_idx"]]
                for k in k_budgets:
                    c_idx = take_calib(sp["pool_by_class"], k)
                    if len(c_idx) == 0:
                        continue
                    Xc, yc = Xh[c_idx], yh[c_idx]
                    t_fit = time.time()
                    m = fit_stage2_source_anchored(
                        X_src, y_src, Xc, yc, grid_src_model, grid_src_scaler,
                        beta=beta, target_mass=target_mass,
                    )
                    yhat = convex_nn_predict(m, grid_src_scaler, Xe)
                    rows.append(dict(
                        cvx_beta=float(beta), stage2_target_mass=float(target_mass),
                        held_out=ho, k=k, n_calib=int(len(Xc)), n_eval=int(len(Xe)),
                        acc=float((yhat == ye).mean()), calib_repeat=int(m.calib_repeat),
                        actual_target_mass=float(m.stage2_target_mass), fit_s=time.time() - t_fit,
                    ))
    return pd.DataFrame(rows)

grid_results = pd.DataFrame()
grid_summary = pd.DataFrame()
grid_overall = pd.DataFrame()
if RUN_CVX_HP_GRID:
    grid_results = run_convex_hp_grid()
    grid_summary = (
        grid_results.groupby(["cvx_beta", "stage2_target_mass", "k"])["acc"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "acc_mean", "std": "acc_std", "count": "n_subjects"})
        .reset_index()
    )
    grid_summary["sem"] = grid_summary["acc_std"] / np.sqrt(grid_summary["n_subjects"].clip(lower=1))
    grid_overall = (
        grid_results.groupby(["cvx_beta", "stage2_target_mass"])["acc"]
        .mean()
        .reset_index(name="acc_mean")
        .sort_values("acc_mean", ascending=False)
    )
grid_overall.head(10)
'''

# save artifacts to v4 dir
C_SAVE = r'''out_dir = REPO_ROOT / "results" / "reve_kmin_convexnn_v4_nb"
out_dir.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
df.to_csv(out_dir / f"convex_per_split_{stamp}.csv", index=False)
summary.to_csv(out_dir / f"convex_summary_{stamp}.csv", index=False)
if "eegnet_df" in globals() and len(eegnet_df):
    eegnet_df.to_csv(out_dir / f"eegnet_per_split_{stamp}.csv", index=False)
    eegnet_summary.to_csv(out_dir / f"eegnet_summary_{stamp}.csv", index=False)
# tidy combined comparison table
combined = summary[["k", "acc_mean", "sem", "n_subjects"]].copy()
combined.columns = ["k", "convex_acc", "convex_sem", "n_subjects"]
if "eegnet_summary" in globals() and len(eegnet_summary):
    e = eegnet_summary[["k", "acc_mean", "sem"]].rename(
        columns={"acc_mean": "eegnet_acc", "sem": "eegnet_sem"})
    combined = combined.merge(e, on="k", how="outer")
combined.to_csv(out_dir / f"comparison_{stamp}.csv", index=False)
if "grid_results" in globals() and len(grid_results):
    grid_results.to_csv(out_dir / f"convex_hp_grid_per_split_{stamp}.csv", index=False)
    grid_summary.to_csv(out_dir / f"convex_hp_grid_summary_{stamp}.csv", index=False)
    grid_overall.to_csv(out_dir / f"convex_hp_grid_overall_{stamp}.csv", index=False)
print("wrote:", out_dir)
combined
'''

# ----------------------------------------------------------------- assemble ----
cells = [
    md(C_TITLE),
    md("## 0. Config"),
    code(C_CONFIG),
    md(vsrc(3)),                 # 1. Load EEGMMI
    code(vsrc(4)),               # data loader
    md(vsrc(5)),                 # 2. REVE feature extractor
    code(vsrc(6)),               # REVE extractor
    md(vsrc(7)),                 # 3. Convex two-layer ReLU MLP
    code(vsrc(8)),               # ADMM convex head
    md(MD_HELDOUT),              # 4. choose held-out + encode
    code(vsrc(10)),              # choose held-out + build_reve
    code(C_ENCODE),              # encode + disk cache
    md(MD_SPLIT),                # 5. fixed eval split
    code(C_SPLIT),               # split helpers + build subject_splits
    md(MD_SWEEP),                # 6. convex sweep
    code(C_SWEEP),               # convex sweep
    md("## 6b. Aggregate + plot (convex)"),
    code(vsrc(15)),              # convex aggregate
    code(vsrc(16)),              # convex plot
    md(MD_EEGNET),               # 7. naive EEGNet
    code(C_EEGNET),              # EEGNet
    md("## 8. Comparison plot"),
    code(C_COMPARE),             # compare plot
    md(vsrc(20)),                # 9. convex HP grid (md)
    code(C_HPGRID),              # HP grid (fixed eval)
    code(vsrc(22)),              # HP grid plot
    md("## 10. Save artifacts"),
    code(C_SAVE),                # save
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

out = NB / "reve_kmin_convexnn_v4.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
