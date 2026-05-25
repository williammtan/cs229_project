"""Paired statistical tests across LOSO folds (Wilcoxon signed-rank).

Stub. Will be wired in once two methods have completed runs to compare.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided Wilcoxon signed-rank on paired per-subject metrics."""
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return {"statistic": float("nan"), "p_value": float("nan"), "n": int(len(a))}
    res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "n": int(len(a)),
    }
