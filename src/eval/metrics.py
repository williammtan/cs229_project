"""Classification evaluation for motor-imagery decoding on PhysioNet EEGMMI.

All metrics operate on:
  y_true:  (N,) int labels
  y_pred:  (N,) int labels  (predicted class)
  y_proba: (N, K) optional, per-class probabilities (needed for AUC-OVR and CE).

The same module is used for baseline runs and for foundation-model evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    accuracy: float
    balanced_accuracy: float
    cohen_kappa: float
    macro_f1: float
    auc_ovr: float | None
    n_samples: int
    n_classes: int
    confusion: list[list[int]]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_classes: int | None = None,
) -> Metrics:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    labels = list(range(n_classes))
    acc = float(accuracy_score(y_true, y_pred))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    auc: float | None = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", labels=labels, average="macro",
            ))
        except ValueError:
            auc = None

    return Metrics(
        accuracy=acc,
        balanced_accuracy=bal,
        cohen_kappa=kappa,
        macro_f1=f1,
        auc_ovr=auc,
        n_samples=int(len(y_true)),
        n_classes=int(n_classes),
        confusion=cm,
    )


def shuffled_null_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perms: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Permutation null for accuracy: shuffle true labels and recompute. Tests
    whether the observed accuracy is meaningfully above chance."""
    if rng is None:
        rng = np.random.default_rng(0)
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    observed = float(accuracy_score(y_true, y_pred))
    null = np.zeros(n_perms, dtype=np.float64)
    yt = y_true.copy()
    for p in range(n_perms):
        rng.shuffle(yt)
        null[p] = accuracy_score(yt, y_pred)
    p_value = (np.sum(null >= observed) + 1) / (n_perms + 1)
    return {
        "observed_acc": observed,
        "null_acc_mean": float(null.mean()),
        "null_acc_p95": float(np.quantile(null, 0.95)),
        "null_acc_p99": float(np.quantile(null, 0.99)),
        "p_value": float(p_value),
        "n_perms": n_perms,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "accuracy",
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Bootstrap CI for accuracy or Cohen's κ over trial resamples."""
    if rng is None:
        rng = np.random.default_rng(0)
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    n = len(y_true)
    fn = accuracy_score if metric == "accuracy" else cohen_kappa_score
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = fn(y_true[idx], y_pred[idx])
    return {
        "metric": metric,
        "ci_low": float(np.quantile(stats, alpha / 2)),
        "ci_high": float(np.quantile(stats, 1 - alpha / 2)),
        "n_boot": n_boot,
        "alpha": alpha,
    }


def summarize_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_classes: int | None = None,
    n_perms: int = 1000,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    """Full eval bundle: metrics + permutation null + bootstrap CIs (acc + κ).
    Canonical evaluation entry point used by every model on this branch."""
    rng = np.random.default_rng(seed)
    m = compute_metrics(y_true, y_pred, y_proba=y_proba, n_classes=n_classes)
    null = shuffled_null_accuracy(y_true, y_pred, n_perms=n_perms, rng=rng)
    ci_acc = bootstrap_ci(y_true, y_pred, metric="accuracy", n_boot=n_boot, rng=rng)
    ci_kappa = bootstrap_ci(y_true, y_pred, metric="kappa", n_boot=n_boot, rng=rng)
    return {
        "metrics": m.to_dict(),
        "null": null,
        "ci_accuracy": ci_acc,
        "ci_kappa": ci_kappa,
    }


def flatten_for_logging(eval_bundle: dict, prefix: str = "eval") -> dict[str, float]:
    """Flatten ``summarize_evaluation`` output to dotted W&B-style keys.

    Produces ``eval/accuracy``, ``eval/balanced_accuracy``, ``eval/cohen_kappa``,
    ``eval/macro_f1``, ``eval/auc_ovr``, ``eval/null/p_value``,
    ``eval/ci/accuracy/{lower,upper}``, ``eval/ci/kappa/{lower,upper}``,
    ``eval/n_samples``.
    """
    out: dict[str, float] = {}
    m = eval_bundle["metrics"]
    out[f"{prefix}/accuracy"] = float(m["accuracy"])
    out[f"{prefix}/balanced_accuracy"] = float(m["balanced_accuracy"])
    out[f"{prefix}/cohen_kappa"] = float(m["cohen_kappa"])
    out[f"{prefix}/macro_f1"] = float(m["macro_f1"])
    if m.get("auc_ovr") is not None:
        out[f"{prefix}/auc_ovr"] = float(m["auc_ovr"])
    out[f"{prefix}/n_samples"] = int(m["n_samples"])

    null = eval_bundle.get("null", {})
    if null:
        out[f"{prefix}/null/p_value"] = float(null["p_value"])
        out[f"{prefix}/null/acc_p95"] = float(null["null_acc_p95"])

    ci_a = eval_bundle.get("ci_accuracy", {})
    if ci_a:
        out[f"{prefix}/ci/accuracy/lower"] = float(ci_a["ci_low"])
        out[f"{prefix}/ci/accuracy/upper"] = float(ci_a["ci_high"])
    ci_k = eval_bundle.get("ci_kappa", {})
    if ci_k:
        out[f"{prefix}/ci/kappa/lower"] = float(ci_k["ci_low"])
        out[f"{prefix}/ci/kappa/upper"] = float(ci_k["ci_high"])

    return out
