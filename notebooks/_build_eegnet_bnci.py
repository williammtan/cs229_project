"""Build eegnet_kmin_bnci2014001.ipynb: EEGNet (Lawhern 2018) on BNCI2014001
(BCI IV-2a), full 9-subject leave-one-subject-out, comparing THREE
calibration-only adaptation methods on top of a source-trained EEGNet.

Same protocol shell as neurogpt_kmin_bnci2014001.ipynb (MOABB loader, full LOSO,
fixed eval set + K-per-class calibration sweep), but EEGNet is a *trainable*
convnet rather than a frozen FM: per LOSO fold we train EEGNet end-to-end on the
8 source subjects, then *adapt to the held-out subject using only its K
calibration trials* via:

  1. fine-tune  - continue training all EEGNet weights on the K calib trials.
  2. LoRA       - inject low-rank deltas (Conv2d + head Linear), train only those.
  3. convex NN  - "traditional" Pilanci-Ergen 2-layer ReLU MLP (single admm()
                  solve) on the frozen source-EEGNet penultimate features.

K=0 (or <2 calib classes) = no adaptation = the source EEGNet itself.

NOTE on asymmetry: fine-tune & LoRA start from the source-trained weights, so
they carry source knowledge into the calibration-only adaptation. The convex
head is trained from scratch on calibration features (no source anchoring - that
is what "traditional, just admm()" means), so it is expected to lag at low K and
only become competitive as K grows.

The EEGNet training loop ports the MPS-collapse guard from
reve_kmin_convexnn_v3.ipynb (NaN / near-chance source fit -> deterministic CPU
retry). The convex cell uses the plain library admm() (cld.optimizers.admm),
matching src/heads/convex_nn.py - NOT the 2-stage warm-start variant.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


C_TITLE = r'''# EEGNet on BNCI2014001 (BCI IV-2a) - source LOSO + calibration-only adaptation

A trainable-convnet counterpart to the frozen-FM notebooks
(`neurogpt_kmin_bnci2014001`, `mirepnet_kmin_bnci2014001`). Same MOABB loader and
the same **full 9-subject leave-one-subject-out + K-per-class calibration**
protocol, but here **EEGNet** (Lawhern et al. 2018) is *trained* on the source
group rather than used as a frozen feature extractor.

**Per LOSO fold** (held-out subject `ho`, source = the other 8 subjects):
1. Train EEGNet end-to-end on the source group.
2. Evaluate on `ho`'s fixed eval set with **no adaptation** (the `source` curve,
   K-independent - this is the LOSO baseline).
3. Adapt to `ho` using **only its K calibration trials per class** with three methods:

| method | what it does | starts from |
|---|---|---|
| **fine-tune** (transfer learning) | continue training *all* EEGNet weights on the K calib trials | source EEGNet |
| **LoRA** | inject low-rank deltas into the Conv2d layers + head Linear; train only those (base frozen) | source EEGNet |
| **convex NN** | "traditional" Pilanci-Ergen 2-layer ReLU MLP, a single `admm()` solve, on the **frozen source-EEGNet penultimate features** | scratch (calib features only) |

`K=0` (or a calib set with <2 classes) means *no adaptation* - all three methods
fall back to the source EEGNet prediction, so the four curves coincide at K=0.

**Read the convex curve with the asymmetry in mind.** fine-tune and LoRA inherit
the source-trained weights, so they carry cross-subject knowledge into the
calibration-only step. The convex head is fit *from scratch* on the held-out
subject's K calibration features (no source data, no source anchoring - that is
exactly "traditional, just `admm()`"), so it should trail at low K and close the
gap only as K grows. The standardizer for the convex features is fit on the
source features (feature normalization, not classifier training); the convex
*classifier* sees calibration data only.

## BNCI2014001 -> EEGNet input format

| field | value |
|---|---|
| loader | MOABB `MotorImagery` (the repo's dataset names ARE MOABB classes) |
| subjects / classes | 9 subjects, 4 classes (left_hand, right_hand, feet, tongue), 144 trials/class/subject |
| channels | first **22 EEG channels** |
| band-pass / resample | **4-40 Hz**, **250 Hz** (standard EEGNet MI band on BCI IV-2a) |
| window | dataset interval **[2, 6] s** -> 1001 samples, trimmed to **1000** (4 s) |
| standardization | per-channel z-score over time, stats fit on the source group |
| model | EEGNet `f1=8, d=2, f2=16, kernel_len=125 (~fs/2), dropout=0.5`; penultimate feature ~ `f2 * (T/32)` dims |
'''

C_CONFIG = r'''from __future__ import annotations

import sys, os, json, time, math, copy, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")

REPO_ROOT = Path.cwd().resolve()
if REPO_ROOT.name == "notebooks":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))     # for `from src.models import EEGNetClf`
CLD_DIR = REPO_ROOT / "vendor" / "CLD"
RAW_CACHE_DIR = REPO_ROOT / "data" / "cache" / "bnci2014001_raw"
assert CLD_DIR.exists(), f"missing CLD vendor dir: {CLD_DIR}"

# --- protocol (same shell as the NeuroGPT/MIRepNet BNCI notebooks) ---
K_BUDGETS      = (0, 1, 2, 5, 10, 20, 30)   # trials per class; K=0 = no adaptation (source EEGNet)
EVAL_PER_CLASS = 40                          # fixed held-out eval trials per class
SEED           = 0

# --- dataset / EEGNet preprocessing spec ---
N_CLASSES     = 4
N_CHANS       = 22
TARGET_FS     = 250
BANDPASS      = (4.0, 40.0)                  # standard EEGNet MI band on BCI IV-2a
TRIAL_SAMPLES = 1000                         # 4 s at 250 Hz (interval [2,6] -> trim 1001 to 1000)
LABEL_MAP = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}

# --- EEGNet architecture (Lawhern 2018, fs=250) ---
EEGNET_F1         = 8
EEGNET_D          = 2
EEGNET_F2         = 16
EEGNET_KERNEL_LEN = 125                      # ~ fs/2
EEGNET_DROPOUT    = 0.5

# --- EEGNet source training ---
EEGNET_BATCH_SIZE      = 64
EEGNET_SOURCE_EPOCHS   = 100
EEGNET_PATIENCE        = 15
EEGNET_LR              = 1.0e-3
EEGNET_WEIGHT_DECAY    = 1.0e-4
EEGNET_SOURCE_VAL_FRAC = 0.1
# 4-class chance = 0.25; a near-chance source fit means the accelerator collapsed.
EEGNET_MIN_SOURCE_VAL_ACC = 0.32

# --- adaptation 1: full fine-tune (transfer learning), calibration-only ---
FT_EPOCHS    = 50
FT_LR        = 5.0e-4
FT_GRAD_CLIP = 1.0

# --- adaptation 2: LoRA fine-tune, calibration-only ---
LORA_RANK      = 8
LORA_ALPHA     = 16.0
LORA_EPOCHS    = 50
LORA_LR        = 1.0e-3
LORA_GRAD_CLIP = 1.0

# --- adaptation 3: convex two-layer ReLU MLP head (traditional ADMM), calibration-only ---
CVX_N_NEURONS  = 16
CVX_BETA       = 1.0e-3
CVX_RHO        = 0.1
CVX_ADMM_ITERS = 8
CVX_PCG_ITERS  = 32
CVX_RANK       = 20

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"torch device={DEVICE}  |  repo={REPO_ROOT}")
'''

C_LOAD = r'''import moabb
from moabb.paradigms import MotorImagery
try:
    from moabb.datasets import BNCI2014_001 as _BNCI
except Exception:
    from moabb.datasets import BNCI2014001 as _BNCI

DATASET = _BNCI()
PARADIGM = MotorImagery(n_classes=N_CLASSES, fmin=BANDPASS[0], fmax=BANDPASS[1], resample=float(TARGET_FS))
SUBJECTS = list(DATASET.subject_list)
print(f"dataset={DATASET.code}  subjects={SUBJECTS}  interval={DATASET.interval}")

def get_subject_raw(subj: int):
    """Return (X (n, 22, 1000) float32, y_int (n,)) for one subject (4-40 Hz, 250 Hz)."""
    epochs, y, _meta = PARADIGM.get_data(dataset=DATASET, subjects=[subj], return_epochs=True)
    X = epochs.get_data(copy=False).astype(np.float32)[:, :N_CHANS, :TRIAL_SAMPLES]
    yi = np.asarray([LABEL_MAP[str(v)] for v in y], dtype=np.int64)
    return X, yi
'''

C_EEGNET = r'''from src.models import EEGNetClf
from torch.utils.data import DataLoader, TensorDataset

def make_eegnet() -> nn.Module:
    return EEGNetClf(n_channels=N_CHANS, n_samples=TRIAL_SAMPLES, n_classes=N_CLASSES,
                     f1=EEGNET_F1, d=EEGNET_D, f2=EEGNET_F2,
                     kernel_len=EEGNET_KERNEL_LEN, dropout=EEGNET_DROPOUT)

def clone_eegnet(model: nn.Module) -> nn.Module:
    c = make_eegnet()
    c.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    return c

# --- per-channel standardization (stats fit on the source group) ---
def fit_eeg_scaler(X: np.ndarray):
    mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std  = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
    return mean, np.maximum(std, 1.0e-4)

def transform_eeg(X: np.ndarray, scaler) -> np.ndarray:
    mean, std = scaler
    return ((X.astype(np.float32) - mean) / std).astype(np.float32)

# --- robust training (NaN / near-chance source fit -> deterministic CPU retry) ---
# Ported from reve_kmin_convexnn_v3.ipynb: training EEGNet on Apple MPS can
# intermittently collapse to a degenerate near-chance model with no NaN. Because
# the source model is the base for every K, one bad source fit would pin the
# whole curve at chance. CPU fits are stable, so we fall back on either failure.
class _EEGNetNaN(RuntimeError):
    pass

def _model_has_nonfinite(model: nn.Module) -> bool:
    return any(not torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point())

@torch.no_grad()
def _chunked_logits(model: nn.Module, X: np.ndarray, device, chunk: int = 256) -> torch.Tensor:
    outs = []
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(X[i:i + chunk]).unsqueeze(1).to(device, non_blocking=True)
        outs.append(model(xb).detach().float().cpu())
    return torch.cat(outs, 0) if outs else torch.empty(0, N_CLASSES)

def _fit_loop(model, X, y, params, device, *, epochs, lr, wd, batch, tr_idx, val_idx, patience, grad_clip):
    model = model.to(device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X[tr_idx]).unsqueeze(1), torch.from_numpy(y[tr_idx])),
        batch_size=batch, shuffle=True, drop_last=False,
    )
    yva = torch.from_numpy(y[val_idx]) if len(val_idx) else None
    best_metric, best_state, bad = float("inf"), None, 0
    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                raise _EEGNetNaN(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip); opt.step()
            running += float(loss.item()) * len(xb)
        metric = running / max(1, len(tr_idx))
        if len(val_idx):
            model.eval()
            metric = float(crit(_chunked_logits(model, X[val_idx], device), yva).item())
        if not np.isfinite(metric) or _model_has_nonfinite(model):
            raise _EEGNetNaN(f"non-finite metric/state at epoch {epoch}")
        if metric < best_metric - 1.0e-5:
            best_metric, bad = metric, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if len(val_idx) and bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    quality = None
    if len(val_idx):
        model.eval()
        quality = float((_chunked_logits(model, X[val_idx], device).argmax(1) == yva).float().mean().item())
    return model, quality

def train_eegnet(model, X, y, params=None, *, epochs, lr, wd=EEGNET_WEIGHT_DECAY,
                 batch=EEGNET_BATCH_SIZE, val_frac=0.0, patience=EEGNET_PATIENCE,
                 grad_clip=1.0, seed=SEED, guard_collapse=False):
    """Train `model`; `params=None` trains all parameters, else only the given list
    (used for LoRA). Robust to MPS NaN / near-chance collapse via CPU retry."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    X = X.astype(np.float32); y = y.astype(np.int64)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_frac)) if (val_frac > 0 and len(X) >= 10) else 0
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    if len(tr_idx) == 0:
        tr_idx, val_idx = idx, np.asarray([], dtype=np.int64)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    dev = torch.device(DEVICE)
    try:
        model, quality = _fit_loop(model, X, y, params if params is not None else list(model.parameters()),
                                   dev, epochs=epochs, lr=lr, wd=wd, batch=batch,
                                   tr_idx=tr_idx, val_idx=val_idx, patience=patience, grad_clip=grad_clip)
    except _EEGNetNaN as err:
        if dev.type == "cpu":
            raise
        print(f"[EEGNet] {DEVICE} produced NaN ({err}); retraining on CPU")
        model.load_state_dict(init_state)
        model, _ = _fit_loop(model, X, y, params if params is not None else list(model.parameters()),
                             torch.device("cpu"), epochs=epochs, lr=lr, wd=wd, batch=batch,
                             tr_idx=tr_idx, val_idx=val_idx, patience=patience, grad_clip=grad_clip)
        return model
    if guard_collapse and quality is not None and quality <= EEGNET_MIN_SOURCE_VAL_ACC and dev.type != "cpu":
        print(f"[EEGNet] {DEVICE} degenerate source fit (val_acc={quality:.3f} <= "
              f"{EEGNET_MIN_SOURCE_VAL_ACC}); retraining on CPU")
        model.load_state_dict(init_state)
        model, _ = _fit_loop(model, X, y, list(model.parameters()), torch.device("cpu"),
                             epochs=epochs, lr=lr, wd=wd, batch=batch, tr_idx=tr_idx,
                             val_idx=val_idx, patience=patience, grad_clip=grad_clip)
    return model

@torch.no_grad()
def predict_eegnet(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    devs = [torch.device(DEVICE)] + ([torch.device("cpu")] if DEVICE != "cpu" else [])
    logits = None
    for dev in devs:
        model = model.to(dev)
        logits = _chunked_logits(model, X.astype(np.float32), dev)
        if torch.isfinite(logits).all():
            return logits.argmax(1).numpy().astype(np.int64)
        print(f"[EEGNet] predict NaN on {dev}; falling back to CPU")
    return logits.argmax(1).numpy().astype(np.int64)

@torch.no_grad()
def eegnet_features(model: nn.Module, X: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Penultimate (pre-head) flattened EEGNet features: (N, flat_dim). Runs the
    conv stack directly (firstconv -> depthwise -> separable -> flatten), so it
    works through LoRA-wrapped layers too."""
    model.eval()
    devs = [torch.device(DEVICE)] + ([torch.device("cpu")] if DEVICE != "cpu" else [])
    feats = None
    for dev in devs:
        model = model.to(dev)
        outs, ok = [], True
        for i in range(0, len(X), chunk):
            xb = torch.from_numpy(X[i:i + chunk].astype(np.float32)).unsqueeze(1).to(dev)
            h = model.separable(model.depthwise(model.firstconv(xb))).flatten(1).detach().float().cpu()
            if not torch.isfinite(h).all():
                ok = False; break
            outs.append(h)
        feats = torch.cat(outs, 0) if outs else torch.empty(0)
        if ok:
            return feats.numpy().astype(np.float32)
        print(f"[EEGNet] features NaN on {dev}; falling back to CPU")
    return feats.numpy().astype(np.float32)
'''

C_CONVEX = r'''if str(CLD_DIR) not in sys.path:
    sys.path.insert(0, str(CLD_DIR))
import jax
import jax.numpy as jnp
from cld.models.cvx_relu_mlp import CVX_ReLU_MLP
from cld.optimizers.admm import admm
from sklearn.preprocessing import StandardScaler

def fit_convex_nn(X: np.ndarray, y: np.ndarray, scaler: StandardScaler,
                  n_classes: int = N_CLASSES, seed: int = SEED):
    """Traditional Pilanci-Ergen convex two-layer ReLU MLP: build the convex
    model, init, run a single ADMM solve. `scaler` is a source-fit StandardScaler;
    the *classifier* is trained on `X` (calibration features) only. Mirrors
    src/heads/convex_nn.py - no 2-stage warm start, no source anchoring."""
    Xs = scaler.transform(X.astype(np.float32)).astype(np.float32)
    m = CVX_ReLU_MLP(jnp.asarray(Xs), jnp.asarray(y.astype(np.int32)),
                     n_classes=n_classes, P_S=CVX_N_NEURONS, beta=CVX_BETA, rho=CVX_RHO,
                     seed=jax.random.PRNGKey(seed))
    m.init_model()
    admm(m, dict(rank=CVX_RANK, beta=CVX_BETA, gamma_ratio=1.0,
                 admm_iters=CVX_ADMM_ITERS, pcg_iters=CVX_PCG_ITERS, check_opt=False))
    return m

def convex_predict(model, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X.astype(np.float32)).astype(np.float32)
    logits = np.asarray(model.stacked_predict(jnp.asarray(Xs), model.theta1, model.theta2))
    return logits.argmax(-1).astype(np.int64)
'''

C_LORA = r'''class LoRALinear(nn.Module):
    """Frozen Linear + trainable rank-r delta. B init 0 -> identity at injection."""
    def __init__(self, base: nn.Linear, r: int, alpha: float = LORA_ALPHA):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.lora_A.T) @ self.lora_B.T)

class LoRAConv2d(nn.Module):
    """Frozen Conv2d + trainable low-rank delta on the weight tensor:

        W_eff = W + scale * (B @ A).view_as(W)

    with A (r, fan_in), B (out_ch, r) init 0, fan_in = (in_ch/groups)*kH*kW. The
    delta is materialized on the weight, so it works for grouped/depthwise convs
    too. At injection B=0 -> W_eff == W (output identical to the frozen conv)."""
    def __init__(self, base: nn.Conv2d, r: int, alpha: float = LORA_ALPHA):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        fan_in = (base.in_channels // base.groups) * base.kernel_size[0] * base.kernel_size[1]
        self.lora_A = nn.Parameter(torch.empty(r, fan_in))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        delta = (self.lora_B @ self.lora_A).view_as(self.base.weight)
        w = self.base.weight + self.scale * delta
        return F.conv2d(x, w, self.base.bias, self.base.stride, self.base.padding,
                        self.base.dilation, self.base.groups)

def inject_lora_eegnet(model: nn.Module, r: int = LORA_RANK, alpha: float = LORA_ALPHA) -> int:
    """Wrap every Conv2d with LoRAConv2d and the Linear head with LoRALinear.
    Collect targets first, then replace, so we don't iterate over new wrappers."""
    targets = []
    for module in model.modules():
        for cname, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                targets.append((module, cname, "conv"))
            elif isinstance(child, nn.Linear):
                targets.append((module, cname, "lin"))
    for parent, attr, kind in targets:
        base = getattr(parent, attr)
        setattr(parent, attr, LoRAConv2d(base, r, alpha) if kind == "conv" else LoRALinear(base, r, alpha))
    return len(targets)

def set_lora_trainable(model: nn.Module):
    """Freeze everything except the LoRA delta tensors; return them for the optimizer."""
    for p in model.parameters():
        p.requires_grad_(False)
    for mod in model.modules():
        if isinstance(mod, (LoRALinear, LoRAConv2d)):
            mod.lora_A.requires_grad_(True)
            mod.lora_B.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]

# Smoke test: inject on a fresh EEGNet, confirm the output is unchanged at init.
_probe = make_eegnet().eval()
_x = torch.randn(2, 1, N_CHANS, TRIAL_SAMPLES)
with torch.no_grad():
    _y0 = _probe(_x)
_n = inject_lora_eegnet(_probe)
_params = set_lora_trainable(_probe)
with torch.no_grad():
    _y1 = _probe(_x)
print(f"LoRA: injected {_n} adapters; trainable params={sum(p.numel() for p in _params):,}; "
      f"init output match={torch.allclose(_y0, _y1, atol=1e-5)}")
del _probe, _x, _y0, _y1, _n, _params
'''

C_RAWCACHE = r'''RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_RAW_CFG = f"bnci2014001_C{N_CHANS}_T{TRIAL_SAMPLES}_fs{TARGET_FS}_bp{BANDPASS[0]:g}-{BANDPASS[1]:g}"
raw_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

def get_raw(subj: int) -> tuple[np.ndarray, np.ndarray]:
    if subj in raw_cache:
        return raw_cache[subj]
    cp = RAW_CACHE_DIR / f"S{subj:02d}.npz"
    if cp.exists():
        d = np.load(cp, allow_pickle=False)
        if "cfg" in d and str(d["cfg"]) == _RAW_CFG:
            raw_cache[subj] = (d["X"].astype(np.float32), d["y"].astype(np.int64))
            return raw_cache[subj]
    t0 = time.time()
    X, y = get_subject_raw(subj)
    np.savez(cp, X=X, y=y, cfg=_RAW_CFG)
    raw_cache[subj] = (X, y)
    print(f"  S{subj:02d}: {len(y):3d} trials, X={X.shape}, "
          f"classes={np.bincount(y, minlength=N_CLASSES).tolist()}, load={time.time()-t0:.1f}s")
    return raw_cache[subj]

for s in SUBJECTS:
    get_raw(s)
print("loaded all subjects.")
'''

C_SPLIT = r'''def make_subject_split(y: np.ndarray, n_classes: int = N_CLASSES,
                       n_eval_per_class: int = EVAL_PER_CLASS, seed: int = 0):
    """Fixed eval set + ordered calibration pool for one subject (eval disjoint from pool)."""
    rng = np.random.default_rng(seed)
    eval_idx, pool_by_class = [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        perm = rng.permutation(idx)
        if len(perm) <= n_eval_per_class:
            raise ValueError(f"class {c}: only {len(perm)} trials, need > {n_eval_per_class}")
        eval_idx.extend(perm[:n_eval_per_class].tolist())
        pool_by_class.append(perm[n_eval_per_class:])
    return np.asarray(sorted(eval_idx), dtype=np.int64), pool_by_class

def take_calib(pool_by_class, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    calib = []
    for pool_c in pool_by_class:
        calib.extend(pool_c[:k].tolist())
    return np.asarray(sorted(calib), dtype=np.int64)

K_MAX = max(K_BUDGETS)
subject_splits: dict[int, dict] = {}
for s in SUBJECTS:
    _, ys = raw_cache[s]
    eval_idx, pool_by_class = make_subject_split(ys, seed=SEED + s)
    if min(len(p) for p in pool_by_class) < K_MAX:
        warnings.warn(f"S{s:02d}: calib pool < K_MAX={K_MAX}")
    subject_splits[s] = dict(eval_idx=eval_idx, pool_by_class=pool_by_class)
print(f"eval set = {EVAL_PER_CLASS*N_CLASSES} trials/subject; "
      f"calib pool/class ~ {min(len(p) for p in subject_splits[SUBJECTS[0]]['pool_by_class'])}; K_MAX={K_MAX}")
'''

C_SWEEP = r'''import pandas as pd

rows = []
for ho in SUBJECTS:
    src_ids = [s for s in SUBJECTS if s != ho]
    X_src = np.concatenate([raw_cache[s][0] for s in src_ids], axis=0)
    y_src = np.concatenate([raw_cache[s][1] for s in src_ids], axis=0)

    eeg_scaler = fit_eeg_scaler(X_src)
    Xs_src = transform_eeg(X_src, eeg_scaler)

    t0 = time.time()
    src_model = train_eegnet(make_eegnet(), Xs_src, y_src,
                             epochs=EEGNET_SOURCE_EPOCHS, lr=EEGNET_LR,
                             val_frac=EEGNET_SOURCE_VAL_FRAC, patience=EEGNET_PATIENCE,
                             seed=SEED + ho, guard_collapse=True)
    feat_scaler = StandardScaler().fit(eegnet_features(src_model, Xs_src))   # source feature stats

    Xh, yh = raw_cache[ho]
    sp = subject_splits[ho]
    e_idx = sp["eval_idx"]
    Xe, ye = transform_eeg(Xh[e_idx], eeg_scaler), yh[e_idx]
    Fe = eegnet_features(src_model, Xe)                       # frozen-source eval features (for convex)

    base_acc = float((predict_eegnet(src_model, Xe) == ye).mean())
    print(f"S{ho:02d}: source EEGNet fit {time.time()-t0:.1f}s  LOSO(K=0)={base_acc:.3f}")

    for k in K_BUDGETS:
        c_idx = take_calib(sp["pool_by_class"], k)
        yc = yh[c_idx]
        n_calib = int(len(c_idx))
        adaptable = not (k == 0 or len(np.unique(yc)) < 2)
        Xc = transform_eeg(Xh[c_idx], eeg_scaler) if adaptable else None

        # 0) source (no adaptation) reference - K-independent
        rows.append(dict(method="source", held_out=ho, k=k, n_calib=n_calib, n_eval=int(len(ye)), acc=base_acc))

        # 1) full fine-tune (transfer learning) - calibration only
        if not adaptable:
            ft_acc = base_acc
        else:
            ft_model = train_eegnet(clone_eegnet(src_model), Xc, yc, epochs=FT_EPOCHS, lr=FT_LR,
                                    val_frac=0.0, grad_clip=FT_GRAD_CLIP, seed=SEED + 100 * ho + k)
            ft_acc = float((predict_eegnet(ft_model, Xe) == ye).mean())
        rows.append(dict(method="finetune", held_out=ho, k=k, n_calib=n_calib, n_eval=int(len(ye)), acc=ft_acc))

        # 2) LoRA - calibration only
        if not adaptable:
            lora_acc = base_acc
        else:
            lora_model = clone_eegnet(src_model)
            inject_lora_eegnet(lora_model)
            lora_params = set_lora_trainable(lora_model)
            lora_model = train_eegnet(lora_model, Xc, yc, params=lora_params, epochs=LORA_EPOCHS,
                                      lr=LORA_LR, val_frac=0.0, grad_clip=LORA_GRAD_CLIP,
                                      seed=SEED + 200 * ho + k)
            lora_acc = float((predict_eegnet(lora_model, Xe) == ye).mean())
        rows.append(dict(method="lora", held_out=ho, k=k, n_calib=n_calib, n_eval=int(len(ye)), acc=lora_acc))

        # 3) convex NN (traditional admm) on frozen source-EEGNet features - calibration only
        if not adaptable:
            cvx_acc = base_acc
        else:
            try:
                cvx_model = fit_convex_nn(eegnet_features(src_model, Xc), yc, feat_scaler, seed=SEED + ho)
                cvx_acc = float((convex_predict(cvx_model, feat_scaler, Fe) == ye).mean())
            except Exception as e:
                warnings.warn(f"convex fit failed S{ho:02d} K={k}: {type(e).__name__}: {e}")
                cvx_acc = base_acc
        rows.append(dict(method="convex", held_out=ho, k=k, n_calib=n_calib, n_eval=int(len(ye)), acc=cvx_acc))

        print(f"  S{ho:02d} K={k:>3d} n_calib={n_calib:>3d}  source={base_acc:.3f}  "
              f"finetune={ft_acc:.3f}  lora={lora_acc:.3f}  convex={cvx_acc:.3f}")

sweep_df = pd.DataFrame(rows)
print(f"\ndone: {len(sweep_df)} rows across {len(SUBJECTS)} folds x {len(K_BUDGETS)} K x 4 methods")
sweep_df.head()
'''

C_PLOT = r'''import matplotlib.pyplot as plt

def summarize(df):
    out = {}
    for m, g in df.groupby("method"):
        s = (g.groupby("k")["acc"].agg(["mean", "std", "count"])
             .rename(columns={"mean": "acc_mean", "std": "acc_std", "count": "n_subjects"}).reset_index())
        s["sem"] = s["acc_std"] / np.sqrt(s["n_subjects"].clip(lower=1))
        out[m] = s
    return out

summ = summarize(sweep_df)
STYLES = [("source",   "source EEGNet (no adapt)",     "x", "--"),
          ("finetune", "fine-tune (calib only)",       "o", "-"),
          ("lora",     "LoRA (calib only)",            "^", "-"),
          ("convex",   "ConvexNN admm (calib only)",   "s", "-")]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for m, lab, mk, ls in STYLES:
    if m not in summ:
        continue
    s = summ[m]
    ax.errorbar(s["k"], s["acc_mean"], yerr=s["sem"], marker=mk, ls=ls, capsize=3, label=lab)
ax.axhline(1.0 / N_CLASSES, ls=":", color="grey", label=f"chance ({N_CLASSES}-class)")
ax.set_xlabel("K calibration trials per class")
ax.set_ylabel(f"held-out accuracy (fixed {EVAL_PER_CLASS}/class eval set)")
ax.set_title(f"BNCI2014001 (BCI IV-2a) - EEGNet source LOSO + calibration-only adaptation\n"
             f"({len(SUBJECTS)}-fold leave-one-subject-out)")
ax.set_xticks(list(K_BUDGETS))
ax.legend()
fig.tight_layout()
fig.show()

comparison = None
for m, *_ in STYLES:
    if m not in summ:
        continue
    s = summ[m][["k", "acc_mean", "sem"]].rename(columns={"acc_mean": f"{m}_acc", "sem": f"{m}_sem"})
    comparison = s if comparison is None else comparison.merge(s, on="k")
comparison
'''

C_SAVE = r'''out_dir = REPO_ROOT / "results" / "eegnet_bnci2014001_kmin_nb"
out_dir.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
sweep_df.to_csv(out_dir / f"per_split_{stamp}.csv", index=False)
for m, s in summ.items():
    s.to_csv(out_dir / f"{m}_summary_{stamp}.csv", index=False)
comparison.to_csv(out_dir / f"comparison_{stamp}.csv", index=False)
print("wrote:", out_dir)
comparison
'''

cells = [
    md(C_TITLE),
    md("## 0. Config"),
    code(C_CONFIG),
    md("## 1. Load BNCI2014001 via MOABB (22 ch, 250 Hz, 4-40 Hz, interval [2,6])"),
    code(C_LOAD),
    md("## 2. EEGNet model, robust training, feature extraction, prediction"),
    code(C_EEGNET),
    md("## 3. Convex two-layer ReLU MLP head (traditional ADMM, calibration-only)"),
    code(C_CONVEX),
    md("## 4. LoRA adapters for EEGNet (Conv2d + head Linear low-rank deltas)"),
    code(C_LORA),
    md("## 5. Load all 9 subjects' raw trials once (disk-cached)"),
    code(C_RAWCACHE),
    md("## 6. Fixed per-subject eval split"),
    code(C_SPLIT),
    md("## 7. Full LOSO + calibration-only adaptation sweep (source / fine-tune / LoRA / convex)"),
    code(C_SWEEP),
    md("## 8. Aggregate + comparison plot"),
    code(C_PLOT),
    md("## 9. Save artifacts"),
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

out = NB / "eegnet_kmin_bnci2014001.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
