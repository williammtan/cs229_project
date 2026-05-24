"""Run the preliminary baseline experiments on WAY-EEG-GAL P1-P3.

Two evaluations per model, designed to match the SOTA evaluation we'll use later:

  (a) Within-subject 5-fold CV (trial-stratified). Reports per-subject
      Pearson r, R^2, RMSE, plus shuffled-target null.
  (b) Leave-one-subject-out cross-subject. Train pooled on N-1 subjects, evaluate
      on the held-out subject without any target-side data (zero-shot LOSO).

Outputs results/preliminary.json plus per-subject figures.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_subject, concat_trials, SubjectData  # noqa: E402
from src.eval import summarize_evaluation  # noqa: E402
from src.models import BradberrymLR, RidgeBandPower, EEGNetReg, ShallowConvNetReg  # noqa: E402
from src.train import NeuralRegressor, TrainConfig  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

SUBJECTS = [1, 2, 3]
SERIES = list(range(1, 10))


def load_all_subjects(subjects, series, target_fs=100):
    out: dict[int, SubjectData] = {}
    for s in subjects:
        t0 = time.time()
        sd = load_subject(RAW_DIR, subject=s, series=series)
        elapsed = time.time() - t0
        print(f"  loaded subject {s}: {len(sd.trials)} trials in {elapsed:.1f}s")
        out[s] = sd
    return out


def kfold_split_trials(n_trials: int, k: int = 5, seed: int = 0):
    """Group trials into k folds (contiguous blocks) — preserves temporal structure."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_trials)
    folds = np.array_split(idx, k)
    return folds


def fit_predict_linear(model_factory, train_eeg, train_vel, eval_eeg):
    m = model_factory()
    m.fit(train_eeg, train_vel)
    return m.predict(eval_eeg)


def fit_predict_neural(model_factory, train_eeg, train_vel, eval_eeg, cfg):
    nr = NeuralRegressor(model_factory=model_factory, cfg=cfg)
    nr.fit(train_eeg, train_vel, verbose=False)
    return nr.predict(eval_eeg)


def within_subject_cv(subject_data: SubjectData, model_specs: list[dict], k: int = 5) -> dict:
    """Per-model: 5-fold CV across trials within this subject."""
    trials = subject_data.trials
    folds = kfold_split_trials(len(trials), k=k)
    per_model: dict[str, list[dict]] = {spec["name"]: [] for spec in model_specs}

    for fold_i, val_idx in enumerate(folds):
        val_set = set(val_idx.tolist())
        train_trials = [t for i, t in enumerate(trials) if i not in val_set]
        val_trials = [t for i, t in enumerate(trials) if i in val_set]
        train_eeg, _, train_vel = concat_trials(train_trials)
        val_eeg, _, val_vel = concat_trials(val_trials)

        for spec in model_specs:
            name = spec["name"]
            t0 = time.time()
            try:
                if spec["kind"] == "linear":
                    pred = fit_predict_linear(spec["factory"], train_eeg, train_vel, val_eeg)
                elif spec["kind"] == "neural":
                    pred = fit_predict_neural(spec["factory"], train_eeg, train_vel, val_eeg, spec["cfg"])
                else:
                    raise ValueError(spec["kind"])
                ev = summarize_evaluation(val_vel, pred)
                ev["fold"] = fold_i
                ev["wall_sec"] = time.time() - t0
                per_model[name].append(ev)
                print(f"    [{name}] fold {fold_i}: r_mean={ev['metrics']['pearson_r_mean']:+.3f} "
                      f"(per-axis {['%+.3f' % x for x in ev['metrics']['pearson_r_per_axis']]}) "
                      f"in {ev['wall_sec']:.1f}s")
            except Exception as e:
                print(f"    [{name}] fold {fold_i}: FAILED {e}")
                per_model[name].append({"error": str(e), "fold": fold_i})
    return per_model


def loso(subject_pool: dict[int, SubjectData], target: int, model_specs: list[dict]) -> dict:
    """Leave-one-subject-out: train on subject_pool minus target, eval on target."""
    train_subjects = [s for s in subject_pool if s != target]
    train_trials = []
    for s in train_subjects:
        train_trials.extend(subject_pool[s].trials)
    train_eeg, _, train_vel = concat_trials(train_trials)
    val_eeg, _, val_vel = concat_trials(subject_pool[target].trials)

    out: dict[str, dict] = {}
    for spec in model_specs:
        name = spec["name"]
        t0 = time.time()
        try:
            if spec["kind"] == "linear":
                pred = fit_predict_linear(spec["factory"], train_eeg, train_vel, val_eeg)
            elif spec["kind"] == "neural":
                pred = fit_predict_neural(spec["factory"], train_eeg, train_vel, val_eeg, spec["cfg"])
            ev = summarize_evaluation(val_vel, pred)
            ev["wall_sec"] = time.time() - t0
            out[name] = ev
            print(f"    [LOSO -> P{target}] [{name}]: r_mean={ev['metrics']['pearson_r_mean']:+.3f} "
                  f"in {ev['wall_sec']:.1f}s")
        except Exception as e:
            out[name] = {"error": str(e)}
            print(f"    [LOSO -> P{target}] [{name}]: FAILED {e}")
    return out


def main():
    print("=" * 70)
    print("Loading subjects:", SUBJECTS)
    print("=" * 70)
    data = load_all_subjects(SUBJECTS, SERIES)

    cfg = TrainConfig(epochs=25, patience=4, batch_size=128, lr=1e-3)

    model_specs = [
        {"name": "Bradberry-mLR", "kind": "linear",
         "factory": lambda: BradberrymLR(lags=(0, 5, 10, 15, 20, 25), alpha=1.0)},
        {"name": "Ridge-BandPower", "kind": "linear",
         "factory": lambda: RidgeBandPower(alpha=10.0)},
        {"name": "EEGNet", "kind": "neural",
         "factory": lambda c, t, o: EEGNetReg(n_channels=c, n_samples=t, n_out=o),
         "cfg": cfg},
        {"name": "ShallowConvNet", "kind": "neural",
         "factory": lambda c, t, o: ShallowConvNetReg(n_channels=c, n_samples=t, n_out=o),
         "cfg": cfg},
    ]

    results = {"within_subject": {}, "loso": {}, "config": {
        "subjects": SUBJECTS, "series": SERIES,
        "target_fs": 100, "win_samples": cfg.win_samples, "hop_samples": cfg.hop_samples,
    }}

    print()
    print("=" * 70)
    print("Within-subject 5-fold CV")
    print("=" * 70)
    for s in SUBJECTS:
        print(f"\n  Subject P{s}: {len(data[s].trials)} trials")
        results["within_subject"][f"P{s}"] = within_subject_cv(data[s], model_specs, k=5)
        # checkpoint after each subject
        (RESULTS_DIR / "preliminary.json").write_text(json.dumps(results, indent=2))

    print()
    print("=" * 70)
    print("Leave-one-subject-out (zero-shot cross-subject)")
    print("=" * 70)
    for s in SUBJECTS:
        print(f"\n  Held out: P{s}")
        results["loso"][f"P{s}"] = loso(data, target=s, model_specs=model_specs)
        (RESULTS_DIR / "preliminary.json").write_text(json.dumps(results, indent=2))

    print()
    print(f"Results written to {RESULTS_DIR/'preliminary.json'}")


if __name__ == "__main__":
    main()
