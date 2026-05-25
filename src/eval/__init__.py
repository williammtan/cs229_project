"""Evaluation: metrics, latency/memory profiling, paired tests, aggregation.

Re-exports the historical ``src.eval`` API so existing imports keep working.
"""
from src.eval.metrics import (
    Metrics,
    bootstrap_ci_r,
    compute_metrics,
    flatten_for_logging,
    fraction_above_threshold,
    pearson_per_axis,
    r2_per_axis,
    rmse_per_axis,
    shuffled_null_r,
    summarize_evaluation,
)

__all__ = [
    "Metrics",
    "bootstrap_ci_r",
    "compute_metrics",
    "flatten_for_logging",
    "fraction_above_threshold",
    "pearson_per_axis",
    "r2_per_axis",
    "rmse_per_axis",
    "shuffled_null_r",
    "summarize_evaluation",
]
