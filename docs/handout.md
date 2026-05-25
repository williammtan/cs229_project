# ETM Project Handout

**Last updated:** 2026-05-24
**Status:** Composable Hydra/W&B scaffolding built; all three FM backbones (CBraMod, Labram, REVE) confirmed loading real pretrained weights and producing finite predictions end-to-end on WAY-EEG-GAL. Ready for the full 12-subject LOSO sweep.

---

## What was built

### 1. Composable experiment scaffolding (commit `16f25d9`)

Replaced inline orchestration with a Hydra-driven runner that composes **Backbone + Adapter-stack + Head** behind one generic train → calibrate → predict loop.

```
src/
  core/         Pipeline, Registry, Split/RunMeta types
  data/         WAY-EEG-GAL loader, LOSO/WithinCV/K-min splits,
                ActiCap-32 names + MOTOR_16 demo subset, 100→200Hz resample
  backbones/    BradberrymLR, RidgeBandPower, EEGNet, ShallowConvNet
                (registered wrappers around the original implementations)
                + FMBackboneBase, CBraMod, Labram, REVE (via braindecode)
  heads/        LinearProbeHead (ridge), IdentityHead
  adapters/     base + NoopAdapter + AdapterStack (no real adapters yet)
  protocols/    LOSO, WithinSubjectCV, KMinCalibration — all yield Splits
  eval/         metrics (Pearson/R²/RMSE/null/CI) + flatten_for_logging
  loggers/      WandbLogger + OfflineLogger; centralized derive_run_meta
  configs/      Hydra tree (dataset/backbone/adapter/head/protocol/...)
  runner.py     @hydra.main entry point
```

### 2. Foundation-model backbones (FM frozen + linear probe, FM finetune)

All three use `braindecode>=1.3.1` wrappers — no vendoring of external code.

| Backbone | Pretrained weights | Status |
|----------|--------------------|--------|
| CBraMod  | `braindecode/CBraMod-Pretrained` (open) | 211/211 tensors loaded |
| Labram   | `braindecode/Labram-Braindecode` (open) | 219/225 tensors loaded |
| REVE     | `brain-bzh/reve-base` (gated, access granted) | 139/140 tensors loaded |

Both **frozen + linear-probe** (Pipeline composes backbone + LinearProbeHead) and **full finetune** (backbone trained end-to-end via `freeze=False`) modes work. Each FM is registered under both `<name>_frozen` and `<name>_finetune` so W&B groups distinguish the two regimes.

### 3. Full-baseline launcher — `scripts/run_full_baselines.py`

Runs all 10 experiment presets in sequence, then aggregates per-group Pearson r / R² / RMSE / wall-time from `results/offline_runs/` (or W&B if `--logger wandb`).

```bash
uv run python scripts/run_full_baselines.py                                  # full 12 subjects
uv run python scripts/run_full_baselines.py --subjects 1 2 --series 1        # smoke
uv run python scripts/run_full_baselines.py --logger wandb --skip reve_finetune_loso
uv run python scripts/run_full_baselines.py --only cbramod_frozen_linear_loso
```

### 4. W&B logging convention (centralized in `src/loggers/base.py`)

- **project**: `etm`
- **group**: `{protocol}__{backbone}__{adapter_stack}` — one row per headline-table cell
- **job_type**: `loso-P{n}`, `kmin-{k}_loso-P{n}`, `fold-{i}` — the nuisance axis swept inside the group
- **tags**: `backbone:<name>`, `adapter:<name>`, `head:<name>`, `dataset:<name>`, `channels:<n>`, `seed:<n>`
- **metric keys**: `eval/pearson_r/{mean,x,y,z}`, `eval/r2/mean`, `eval/rmse/mean`, `eval/null/p_value/mean`, `eval/ci/{lower,upper}/mean`, `calib/k_minutes`, `wall_sec`
- K-min curves plot automatically via `wandb.define_metric("eval/*", step_metric="calib/k_minutes")`

---

## Verified smoke results (2 subjects × 1 series — not publishable)

| Method                          | Pearson r (mean) | sec/fold |
|---------------------------------|------------------|----------|
| REVE frozen + linear probe      | +0.126           | 22.0     |
| Labram finetune                 | +0.132           | 43.2     |
| CBraMod finetune                | +0.158 (1 fold)  | 34.9     |
| EEGNet                          | +0.103           |  5.0     |
| CBraMod frozen + linear probe   | +0.103           |  2.9     |
| BradberrymLR                    | +0.102           |  0.9     |
| REVE finetune                   | +0.089           | 105.7    |
| ShallowConvNet                  | +0.065           |  6.1     |
| Labram frozen + linear probe    | +0.046           |  6.4     |
| RidgeBandPower                  | +0.050           |  0.9     |

These show the pipeline is wired up correctly. The full 12-subject LOSO sweep gives the real numbers.

---

## Next steps

### Immediate (this week)

1. **Run the full 12-subject LOSO sweep.** Estimated ~5h skipping `reve_finetune_loso`, ~10h with it. Log to W&B.
   ```bash
   uv run python scripts/run_full_baselines.py --logger wandb --skip reve_finetune_loso
   ```

2. **Build the W&B dashboard.** Pin these views per the convention above:
   - Headline: `eval/pearson_r/mean` grouped by `group`, x-axis = `job_type`
   - Per-axis: `eval/pearson_r/{x,y,z}`
   - Honesty checks: `eval/null/p_value/mean`, `eval/ci/{lower,upper}/mean`
   - Cost: `wall_sec`

3. **Populate `docs/plan.md` §7 headline table** from W&B summary metrics once the sweep finishes.

### Per `docs/plan.md` migration order (steps 6–10)

4. **Adapters** — `src/adapters/` has stubs only. Fill in:
   - **PINK** (per-subject LoRA on frozen CBraMod): rank sweep `r ∈ {1, 2, 4, 8, 16}`, layer-placement ablation. Hook into CBraMod's `nn.MultiheadAttention.in_proj_weight` / `out_proj` — note these are *not* the standard `q_proj`/`k_proj`/`v_proj` modules PEFT auto-targets, so the LoRA wrapper needs a custom mapping.
   - **YELLOW** (convex two-layer NN on frozen FM embeddings): Pilanci-Ergen via `cvxpy`. Should beat the linear probe if frozen embeddings need any nonlinear reshaping per subject.
   - **RED** (online Riemannian alignment): sliding-window, EMA, Kalman, recursive Karcher variants. Closed-form, label-free, sub-200ms baseline.

5. **K-min calibration sweep** — the headline plot in `docs/plan.md` §7. Will become meaningful once LoRA/ConvexNN actually use the calibration set (the current `LinearProbeHead.calibrate` is a no-op, which is why K>0 budgets currently match K=0).

6. **Latency + memory profiling** — `src/eval/latency.py` and `src/eval/memory.py` are stubs; wire into the runner's final summary. Hard constraint: <200 ms p95 on consumer CPU.

7. **16-channel demo** — `cbramod_frozen_linear_16ch_loso.yaml` already works. Need parallel configs for Labram (verify `MOTOR_16` names all map into `standard_1020`) and REVE (verify position bank handles the subset).

8. **Drift + channel-dropout protocols** — `src/protocols/drift_within_day.py` and `channel_dropout.py` are not yet written.

9. **`make_figures.py` rewrite** — query the W&B API to auto-populate `results/summary_table.md` from `run.summary` aggregates.

---

## Known issues / gotchas

- **CBraMod's `n_outputs=None` and Labram's `n_times` defaults break braindecode's `from_pretrained`** — we bypass with direct `hf_hub_download` + `_load_overlap`. Don't "fix" this by switching back to `from_pretrained`.
- **REVE weights come from `brain-bzh/reve-base`** (not `braindecode/REVE-Pretrained`, which 404s). The state-dict keys map directly (139/140) into braindecode's REVE class. Filename is `model.safetensors` — loader dispatches by extension.
- **WAY-EEG-GAL is preprocessed at 100 Hz** but FMs need 200 Hz — `FMBackboneBase` upsamples per-window at inference. A cleaner long-term move would be a second preprocessing path that goes from raw 500 Hz directly to 200 Hz (preserves more spectral content).
- **`config.yaml` defaults set `subjects: [1..12]`, `series: [1..9]`** — every Hydra invocation hits the full data unless explicitly overridden.
- **FM finetune `max_train_windows` cap** is set in the configs (4000 for CBraMod/Labram, 2000 for REVE) to bound runtime. Bump if you want to actually exhaust the training pool.
- **`LinearProbeHead.calibrate` is a no-op** — K-min sweeps currently produce a flat curve. The first real K-min comparison needs LoRA or ConvexNN, which both implement `.calibrate`.

---

## Useful commands

```bash
# Single experiment, offline JSON logger
uv run python -m src.runner +experiment=cbramod_frozen_linear_loso

# Single experiment, log to W&B
uv run python -m src.runner +experiment=cbramod_frozen_linear_loso logger=wandb

# Hydra multirun sweep
uv run python -m src.runner -m +experiment=cbramod_frozen_linear_loso \
    protocol=kmin_calibration 'protocol.k_budgets_min=[0,1,5]' seed=0,1,2

# Override data scope on the fly
uv run python -m src.runner +experiment=baseline_eegnet_loso \
    'dataset.subjects=[1,2,3]' 'dataset.series=[1]'

# Full baseline sweep, offline
uv run python scripts/run_full_baselines.py --skip reve_finetune_loso

# Smoke (every method on 2 subjects, 1 series)
uv run python scripts/run_full_baselines.py --subjects 1 2 --series 1
```
