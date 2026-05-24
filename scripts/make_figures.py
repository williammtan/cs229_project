"""Generate figures from results/preliminary.json for the milestone document."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "results" / "preliminary.json"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["Bradberry-mLR", "Ridge-BandPower", "EEGNet", "ShallowConvNet"]
MODEL_COLORS = {"Bradberry-mLR": "#888888", "Ridge-BandPower": "#444444",
                "EEGNet": "#1f77b4", "ShallowConvNet": "#d62728"}
AXIS_LABELS = ["X", "Y", "Z"]


def load_results():
    return json.loads(RESULTS_JSON.read_text())


def _collect_within(res, model):
    """Return per-subject mean r across folds (filtered to valid folds)."""
    out = {}
    for subj_key, per_model in res["within_subject"].items():
        folds = per_model.get(model, [])
        rs = [f["metrics"]["pearson_r_mean"] for f in folds if "metrics" in f]
        if rs:
            out[subj_key] = np.array(rs)
    return out


def _collect_loso(res, model):
    out = {}
    for subj_key, per_model in res["loso"].items():
        v = per_model.get(model, {})
        if "metrics" in v:
            out[subj_key] = v["metrics"]["pearson_r_mean"]
    return out


def figure_within_vs_loso(res):
    fig, ax = plt.subplots(figsize=(8, 5))
    subjects = sorted(res["within_subject"].keys())
    x = np.arange(len(subjects))
    width = 0.10
    for i, model in enumerate(MODELS):
        wi = _collect_within(res, model)
        within_means = [np.nanmean(wi[s]) if s in wi else np.nan for s in subjects]
        within_stds = [np.nanstd(wi[s]) if s in wi else 0.0 for s in subjects]
        loso = _collect_loso(res, model)
        loso_vals = [loso.get(s, np.nan) for s in subjects]
        off = (i - 1.5) * width * 2
        ax.bar(x + off, within_means, width=width, color=MODEL_COLORS[model],
               yerr=within_stds, capsize=2, label=f"{model} (within)")
        ax.bar(x + off + width, loso_vals, width=width, color=MODEL_COLORS[model],
               alpha=0.45, hatch="///", label=f"{model} (LOSO)")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.axhline(0.2, color="red", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel("Pearson r (mean across X, Y, Z)")
    ax.set_xlabel("Subject")
    ax.set_title("Within-subject 5-fold CV vs LOSO cross-subject — velocity decoding (WAY-EEG-GAL P1-P3)")
    ax.legend(ncol=2, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "within_vs_loso_r.png", dpi=160)
    plt.close(fig)


def figure_perm_null(res):
    """For each model, plot observed r vs null distribution per subject (within-subject fold 0)."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4), sharey=True)
    subjects = sorted(res["within_subject"].keys())
    for ax, model in zip(axes, MODELS):
        for j, s in enumerate(subjects):
            folds = res["within_subject"][s].get(model, [])
            for f in folds[:1]:  # just fold 0
                if "null" not in f:
                    continue
                obs = f["metrics"]["pearson_r_mean"]
                null_mean = np.nanmean(f["null"]["null_r_mean"])
                null_p95 = np.nanmean(f["null"]["null_r_p95"])
                ax.plot([j - 0.2, j + 0.2], [null_p95, null_p95], color="red", lw=1)
                ax.plot([j - 0.2, j + 0.2], [null_mean, null_mean], color="gray", lw=1)
                ax.scatter([j], [obs], color=MODEL_COLORS[model], s=60, zorder=3)
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects)
        ax.set_title(model)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Subject")
    axes[0].set_ylabel("Pearson r (observed = colored dot,\nnull mean = gray, null p95 = red)")
    fig.suptitle("Observed r vs shuffled-target permutation null (within-subject fold 0)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "perm_null.png", dpi=160)
    plt.close(fig)


def figure_per_axis(res):
    """Per-axis r for within-subject, averaged across folds and subjects."""
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.18
    x = np.arange(3)
    for i, model in enumerate(MODELS):
        per_axis = []
        for axis_i in range(3):
            rs = []
            for s, per_model in res["within_subject"].items():
                for f in per_model.get(model, []):
                    if "metrics" in f:
                        rs.append(f["metrics"]["pearson_r_per_axis"][axis_i])
            per_axis.append(np.nanmean(rs) if rs else np.nan)
        off = (i - 1.5) * width
        ax.bar(x + off, per_axis, width=width, color=MODEL_COLORS[model], label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(AXIS_LABELS)
    ax.set_xlabel("Velocity axis")
    ax.set_ylabel("Pearson r (mean across subjects & folds)")
    ax.set_title("Per-axis decoding correlation (within-subject)")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_axis_r.png", dpi=160)
    plt.close(fig)


def write_summary_table(res):
    """Tabular summary, written as Markdown for inclusion in the milestone doc."""
    lines = ["| Model | Within r (mean ± std across subj) | LOSO r (mean across subj) | Within R² | Within RMSE (mm/s) | Wall-clock |",
             "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        within_r_subj = []
        within_r2_subj = []
        within_rmse_subj = []
        wall = []
        for s, per_model in res["within_subject"].items():
            folds = per_model.get(model, [])
            valid = [f for f in folds if "metrics" in f]
            if valid:
                within_r_subj.append(np.nanmean([f["metrics"]["pearson_r_mean"] for f in valid]))
                within_r2_subj.append(np.nanmean([f["metrics"]["r2_mean"] for f in valid]))
                within_rmse_subj.append(np.nanmean([f["metrics"]["rmse_mean"] for f in valid]))
                wall.extend([f.get("wall_sec", 0) for f in valid])
        loso_r = []
        for s, per_model in res["loso"].items():
            v = per_model.get(model, {})
            if "metrics" in v:
                loso_r.append(v["metrics"]["pearson_r_mean"])
        if within_r_subj:
            r_mean = np.nanmean(within_r_subj)
            r_std = np.nanstd(within_r_subj)
            r2 = np.nanmean(within_r2_subj)
            rmse = np.nanmean(within_rmse_subj)
            lo = np.nanmean(loso_r) if loso_r else float("nan")
            w = np.nanmean(wall) if wall else 0
            lines.append(
                f"| {model} | {r_mean:+.3f} ± {r_std:.3f} | {lo:+.3f} | {r2:+.3f} | {rmse:.2f} | {w:.1f}s/fold |"
            )
    table_path = ROOT / "results" / "summary_table.md"
    table_path.write_text("\n".join(lines))
    print(f"Wrote {table_path}")
    return "\n".join(lines)


if __name__ == "__main__":
    res = load_results()
    figure_within_vs_loso(res)
    figure_perm_null(res)
    figure_per_axis(res)
    table = write_summary_table(res)
    print()
    print(table)
    print()
    print(f"Figures saved to {FIG_DIR}")
