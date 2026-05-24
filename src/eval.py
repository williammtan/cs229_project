"""Evaluation framework for continuous EEG kinematic decoding.

All metrics operate on numpy arrays of shape (N, D) where N is timesteps (or
trials × timesteps flattened) and D is the kinematic output dimension
(e.g., 3 for x, y, z velocity). The same module is used for baseline runs
and will be reused for foundation-model SOTA evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class Metrics:
    pearson_r_per_axis: list[float]
    pearson_r_mean: float
    r2_per_axis: list[float]
    r2_mean: float
    rmse_per_axis: list[float]
    rmse_mean: float
    n_samples: int
    n_axes: int

    def to_dict(self) -> dict:
        return asdict(self)


def pearson_per_axis(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pearson r per output axis. y_true, y_pred shape (N, D)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, f"{y_true.shape} != {y_pred.shape}"
    n_axes = y_true.shape[1]
    rs = np.zeros(n_axes)
    for i in range(n_axes):
        a = y_true[:, i]
        b = y_pred[:, i]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            rs[i] = 0.0
        else:
            rs[i] = stats.pearsonr(a, b)[0]
    return rs


def r2_per_axis(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """R^2 (coefficient of determination) per axis. Can be negative."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def rmse_per_axis(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    r = pearson_per_axis(y_true, y_pred)
    r2 = r2_per_axis(y_true, y_pred)
    rm = rmse_per_axis(y_true, y_pred)
    return Metrics(
        pearson_r_per_axis=r.tolist(),
        pearson_r_mean=float(r.mean()),
        r2_per_axis=r2.tolist(),
        r2_mean=float(r2.mean()),
        rmse_per_axis=rm.tolist(),
        rmse_mean=float(rm.mean()),
        n_samples=int(y_true.shape[0]),
        n_axes=int(y_true.shape[1]),
    )


def shuffled_null_r(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perms: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Permutation null for Pearson r: circularly shift the predictions in time
    relative to the targets to destroy temporal alignment while preserving each
    signal's autocorrelation. Returns the null distribution per axis plus the
    observed r and a one-sided p-value (P(null >= observed))."""
    if rng is None:
        rng = np.random.default_rng(0)
    n, d = y_true.shape
    observed = pearson_per_axis(y_true, y_pred)
    null = np.zeros((n_perms, d))
    for p in range(n_perms):
        shift = int(rng.integers(1, n))
        null[p] = pearson_per_axis(y_true, np.roll(y_pred, shift, axis=0))
    p_values = (np.sum(null >= observed[None, :], axis=0) + 1) / (n_perms + 1)
    return {
        "observed_r": observed.tolist(),
        "null_r_mean": null.mean(axis=0).tolist(),
        "null_r_p95": np.quantile(null, 0.95, axis=0).tolist(),
        "null_r_p99": np.quantile(null, 0.99, axis=0).tolist(),
        "p_values": p_values.tolist(),
        "n_perms": n_perms,
    }


def bootstrap_ci_r(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Block-bootstrap 95% CI for Pearson r per axis."""
    if rng is None:
        rng = np.random.default_rng(0)
    n, d = y_true.shape
    block = max(1, n // 50)  # ~50 blocks
    n_blocks = n // block
    rs = np.zeros((n_boot, d))
    for b in range(n_boot):
        idx_blocks = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([np.arange(i * block, (i + 1) * block) for i in idx_blocks])
        rs[b] = pearson_per_axis(y_true[idx], y_pred[idx])
    lo = np.quantile(rs, alpha / 2, axis=0)
    hi = np.quantile(rs, 1 - alpha / 2, axis=0)
    return {
        "ci_low": lo.tolist(),
        "ci_high": hi.tolist(),
        "n_boot": n_boot,
        "alpha": alpha,
    }


def fraction_above_threshold(
    per_subject_r: Sequence[float], threshold: float = 0.2
) -> float:
    """Operational threshold: fraction of held-out subjects clearing r > threshold."""
    arr = np.asarray(per_subject_r)
    return float(np.mean(arr > threshold))


def summarize_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perms: int = 200,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    """Full eval bundle: metrics + permutation null + bootstrap CI.
    This is the canonical evaluation entry point used by every model in the project."""
    rng = np.random.default_rng(seed)
    m = compute_metrics(y_true, y_pred)
    null = shuffled_null_r(y_true, y_pred, n_perms=n_perms, rng=rng)
    ci = bootstrap_ci_r(y_true, y_pred, n_boot=n_boot, rng=rng)
    return {"metrics": m.to_dict(), "null": null, "ci": ci}
