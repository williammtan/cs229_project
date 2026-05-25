# etm — Cross-Subject Calibration Efficiency for Continuous EEG Hand-Kinematic Decoding

CS229 final project. Decoding continuous 3D hand velocity from non-invasive scalp EEG with a focus on how little per-user calibration data is needed to bring a new user close to within-subject performance.

## Documents

- [`literature_review.md`](docs/literature_review.md) — full prior-work survey across datasets, architectures, cross-subject transfer, recent top-venue papers, and open methodological problems.
- [`milestone.md`](docs/milestone.md) — Motivation, Methods, Preliminary Experiments & Results, Next Steps.

## Quickstart

```bash
# 1. Install deps
uv sync

# 2. Download WAY-EEG-GAL subjects 1-3 (~2.4 GB) into data/raw/
uv run python scripts/download_data.py

# 3a. Run one experiment via Hydra (offline JSON logger by default)
uv run python -m src.runner +experiment=baseline_eegnet_loso

# 3b. Or run the full preliminary baseline grid (4 baselines × {LOSO, within-CV})
uv run python scripts/run_preliminary.py

# 3c. Log to W&B instead of disk
uv run python -m src.runner +experiment=baseline_eegnet_loso logger=wandb

# 3d. Hydra multirun: sweep K-min calibration budgets and seeds
uv run python -m src.runner -m +experiment=baseline_eegnet_loso \
    protocol=kmin_calibration 'protocol.k_budgets_min=[0,1,5]' seed=0,1,2

# 3e. Frozen foundation-model backbones + ridge linear probe (LOSO)
#     CBraMod and Labram weights download automatically from HuggingFace.
#     REVE is gated; run `huggingface-cli login` and accept the use agreement
#     at https://huggingface.co/braindecode/REVE-Pretrained — otherwise the
#     wrapper falls back to random init with a clear warning.
uv run python -m src.runner +experiment=cbramod_frozen_linear_loso
uv run python -m src.runner +experiment=labram_frozen_linear_loso
uv run python -m src.runner +experiment=reve_frozen_linear_loso

# 3f. 16-channel "demo day" subset (clinical motor-area subset of the 32 channels)
uv run python -m src.runner +experiment=cbramod_frozen_linear_16ch_loso

# 4. Generate figures and summary table
uv run python scripts/make_figures.py

# 5. (Optional) Rebuild the milestone PDF
cd docs && pdflatex milestone.tex && pdflatex milestone.tex
```

## Layout

```
docs/                 literature_review.md, milestone.{md,tex,pdf}, plan.md, cs229.sty
src/
  core/               Registry, Pipeline composition, shared dataclasses
  data/               WAY-EEG-GAL loader, splits (LOSO, WithinCV, K-min), windowing
  backbones/          Wrappers around src.models exposing a uniform Backbone API
  adapters/           Per-subject adapters (none today; LoRA / RA / convex-NN to come)
  heads/              Optional head modules (Identity today)
  protocols/          Evaluation protocols yielding Splits (LOSO, within-CV, K-min)
  eval/               Metrics, latency/memory profiling, paired stats
  loggers/            W&B + offline JSON; centralized tag/group convention
  configs/            Hydra config groups (dataset/backbone/adapter/head/protocol/...)
  runner.py           @hydra.main entry point; one generic train→calibrate→eval loop
  models.py, train.py original baseline implementations (still the source of truth)
  eval.py removed     (contents moved to src/eval/metrics.py; old import path preserved)
scripts/              run_preliminary.py (now a thin Hydra driver), download_data.py, make_figures.py
data/raw/             downloaded WAY-EEG-GAL .mat files
results/              offline_runs/, figures/, summary_table.md
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
