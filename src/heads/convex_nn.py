"""Convex two-layer ReLU MLP head on top of backbone embeddings.

Wraps ``cld.models.cvx_relu_mlp.CVX_ReLU_MLP`` (Pilanci-Ergen 2020, ADMM-trained
via Cronos/CLD) as a HeadBase. The "YELLOW" track of docs/plan_clf.md §8.

The flow on the composed Pipeline path:

    backbone.encode_trial(trial)  ->  (D,) embedding per trial
    pool over trials              ->  (N, D) matrix
    ConvexNNHead.fit(X, y)        ->  ADMM on the convex reformulation
    head.predict_proba(X)         ->  (N, K) softmax-ed logits

Properties (Pilanci-Ergen 2020):
  * Provably converges to the global optimum of a 2-layer ReLU MLP.
  * No local-minimum / hyperparameter pain — only beta (L2 reg) and the
    number of neurons / hyperplane samples ``P_S`` matter for the model;
    ADMM hyperparameters (rho, rank, pcg_iters, admm_iters) only control
    convergence speed.
  * Fast on CPU for small N (≤ few thousand trials) and modest D (≤ 512).

Vendored package is at vendor/CLD (kept at HEAD of github.com/pilancilab/CLD).
We add it to sys.path lazily — no global side effects at import time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.core.registry import register
from src.heads.base import HeadBase

_VENDOR_PATH = Path(__file__).resolve().parents[2] / "vendor" / "CLD"


def _import_cld():
    """Lazy import: add vendor/CLD to sys.path and import jax + CLD modules."""
    if str(_VENDOR_PATH) not in sys.path:
        sys.path.insert(0, str(_VENDOR_PATH))
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from cld.models.cvx_relu_mlp import CVX_ReLU_MLP  # noqa: E402
    from cld.optimizers.admm import admm  # noqa: E402
    return jax, jnp, CVX_ReLU_MLP, admm


@register("head", "convex_nn")
class ConvexNNHead(HeadBase):
    """Convex 2-layer ReLU MLP classifier on per-trial features.

    Args:
        n_classes:    output dim. Defaults to 4 (EEGMMI 4-class MI).
        n_neurons:    P_S in Pilanci-Ergen. Number of sampled hyperplane gates.
                      Effective hidden width = 2 * n_neurons after the convex-to-
                      non-convex transform.
        beta:         L2 weight regularization on (v, w).
        rho:          ADMM penalty (convergence-speed knob, not a true hp).
        admm_iters:   outer ADMM sweeps.
        pcg_iters:    inner PCG iterations per ADMM step.
        rank:         Nyström preconditioner rank.
        seed:         JAX PRNGKey seed.
        calibrate_on_calib: if True, ``calibrate(X, y)`` refits on source
                      plus calibration data (mirrors SoftmaxProbeHead).
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_neurons: int = 64,
        beta: float = 1.0e-3,
        rho: float = 0.1,
        admm_iters: int = 8,
        pcg_iters: int = 32,
        rank: int = 20,
        seed: int = 0,
        calibrate_on_calib: bool = True,
    ):
        self.n_classes = int(n_classes)
        self.n_neurons = int(n_neurons)
        self.beta = float(beta)
        self.rho = float(rho)
        self.admm_iters = int(admm_iters)
        self.pcg_iters = int(pcg_iters)
        self.rank = int(rank)
        self.seed = int(seed)
        self.calibrate_on_calib = calibrate_on_calib

        self.scaler = StandardScaler()
        self._model = None  # CVX_ReLU_MLP after fit
        self._fitted = False
        self._source_feats: np.ndarray | None = None
        self._source_targets: np.ndarray | None = None

    def _validate_inputs(
        self,
        feats: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(feats, dtype=np.float32)
        y_raw = np.asarray(targets)
        if X.ndim != 2:
            raise ValueError(f"feats must be a 2D (N, D) matrix; got {X.shape}")
        if y_raw.ndim != 1:
            raise ValueError(f"targets must be a 1D class-label vector; got {y_raw.shape}")
        if len(X) != len(y_raw):
            raise ValueError(f"feats/targets length mismatch: {len(X)} != {len(y_raw)}")
        if len(X) == 0:
            raise ValueError("Cannot fit ConvexNNHead on an empty dataset.")
        y = y_raw.astype(np.int64)
        if not np.allclose(y_raw, y):
            raise ValueError("targets must contain integer class labels.")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(
                f"targets must be in [0, {self.n_classes}); got range "
                f"[{int(y.min())}, {int(y.max())}]"
            )
        return X, y

    def _fit_impl(
        self,
        feats: np.ndarray,
        targets: np.ndarray,
        *,
        remember_source: bool,
    ) -> None:
        feats, targets = self._validate_inputs(feats, targets)
        jax, jnp, CVX_ReLU_MLP, admm = _import_cld()

        X = self.scaler.fit_transform(feats)
        y = targets.astype(np.int32)

        model = CVX_ReLU_MLP(
            jnp.asarray(X), jnp.asarray(y),
            n_classes=self.n_classes, P_S=self.n_neurons,
            beta=self.beta, rho=self.rho,
            seed=jax.random.PRNGKey(self.seed),
        )
        model.init_model()
        admm_params = dict(
            rank=self.rank, beta=self.beta, gamma_ratio=1.0,
            admm_iters=self.admm_iters, pcg_iters=self.pcg_iters,
            check_opt=False,
        )
        # admm() mutates model.theta1 / model.theta2 in place.
        admm(model, admm_params)
        self._model = model
        self._fitted = True
        if remember_source:
            self._source_feats = feats.copy()
            self._source_targets = targets.copy()

    def fit(self, feats: np.ndarray, targets: np.ndarray) -> None:
        self._fit_impl(feats, targets, remember_source=True)

    def calibrate(self, feats: np.ndarray, targets: np.ndarray) -> None:
        """Per-subject calibration: refit on source + calibration data."""
        if not self.calibrate_on_calib or len(targets) < 2:
            return
        if self._source_feats is None or self._source_targets is None:
            if len(np.unique(targets)) < 2:
                return
            self.fit(feats, targets)
            return
        combined_feats = np.concatenate(
            [self._source_feats, feats.astype(np.float32)], axis=0
        )
        combined_targets = np.concatenate(
            [self._source_targets, targets.astype(np.int64)], axis=0
        )
        self.scaler = StandardScaler()
        self._fit_impl(combined_feats, combined_targets, remember_source=False)

    def _logits(self, feats: np.ndarray) -> np.ndarray:
        assert self._fitted, "fit first"
        _, jnp, _, _ = _import_cld()
        X = self.scaler.transform(feats.astype(np.float32))
        m = self._model
        logits = m.stacked_predict(jnp.asarray(X), m.theta1, m.theta2)
        return np.asarray(logits)

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        logits = self._logits(feats)
        # Numerically-stable softmax in numpy.
        logits = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits)
        proba = exp / exp.sum(axis=-1, keepdims=True)
        return proba.astype(np.float32)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        return self._logits(feats).argmax(axis=-1).astype(np.int64)
