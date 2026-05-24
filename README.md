# etm — Cross-Subject Calibration Efficiency for Continuous EEG Hand-Kinematic Decoding

CS229 final project. Decoding continuous 3D hand velocity from non-invasive scalp EEG with a focus on how little per-user calibration data is needed to bring a new user close to within-subject performance.

## Documents

- [`literature_review.md`](literature_review.md) — full prior-work survey across datasets, architectures, cross-subject transfer, recent top-venue papers, and open methodological problems.
- [`milestone.md`](milestone.md) — Motivation, Methods, Preliminary Experiments & Results, Next Steps.

## Quickstart

```bash
# 1. Install deps
uv sync

# 2. Download WAY-EEG-GAL subjects 1-3 (~2.4 GB) into data/raw/
#    (download URLs hard-coded; expand as needed for full 12-subject pool)
uv run python scripts/download_data.py    # TODO: separate from inline download

# 3. Run preliminary baselines (within-subject 5-fold CV + LOSO cross-subject)
uv run python scripts/run_preliminary.py

# 4. Generate figures and summary table for milestone.md
uv run python scripts/make_figures.py
```

## Layout

```
src/         data, eval, models, train  — all importable, sklearn-style APIs
scripts/     entry points
data/raw/    downloaded WAY-EEG-GAL .mat files
results/     preliminary.json, figures/, summary_table.md
```

## Status

- [x] Lit review
- [x] Project skeleton
- [x] Preprocessing pipeline (WAY-EEG-GAL)
- [x] Eval framework (Pearson r, R², RMSE, shuffled-target null, bootstrap CI)
- [x] Four baseline decoders (Bradberry mLR, Ridge BandPower, EEGNet, ShallowConvNet)
- [x] Preliminary results on P1-P3
- [ ] Extend to all 12 WAY-EEG-GAL subjects
- [ ] Add Jeong 2020 + Müller-Putz Graz datasets
- [ ] Causal-filter ablation
- [ ] Foundation-model backbone (EEGPT / CBraMod) + adaptation methods
- [ ] Calibration dose-response curve (headline figure)
