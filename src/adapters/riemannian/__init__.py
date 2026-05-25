"""Riemannian / Euclidean Alignment (the RED branch from docs/plan.md).

Whitens each subject's EEG by its Frechet-mean covariance on the SPD manifold,
mapping per-subject distributions toward a common reference (identity). This
is the cheapest possible cross-subject alignment — closed-form, label-free,
sub-200ms — and the baseline every more sophisticated adapter must beat.

Two variants implemented:

* :class:`StaticRA` — Frechet mean over a fixed calibration set.
* :class:`EMARA` — exponential moving average on the SPD manifold (online).

Sliding-window and Riemannian-Kalman variants are next-up.
"""
from src.adapters.riemannian.static import StaticRA
from src.adapters.riemannian.ema import EMARA

__all__ = ["StaticRA", "EMARA"]
