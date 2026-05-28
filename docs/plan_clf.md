# EEG Foundation Models for Motor Imagery Classification (clf variant)

**Working document. Branch: `class`. Last updated: 2026-05-25.**
**Variant of `plan.md` recast for discrete-class motor imagery decoding on PhysioNet EEGMMI. Continuous-regression infrastructure (WAY-EEG-GAL, force/velocity heads, Pearson/R²/RMSE metrics, Bradberry baseline) is dropped — this branch rewrites the data + head + eval stack rather than carrying the old one.**

---

## 1. Problem Statement

Same diagnosis as the regression plan, restated for classification:

Non-invasive EEG foundation models (LaBraM, CBraMod, REVE, NeuroLM, EEGPT, …) fail in real deployment due to:
- **Within-session drift.** Yesterday's calibration ≈ stranger's calibration on you.
- **Long calibration sequences.** 20–60-minute supervised sessions; BCI-illiterate users.
- **Cross-subject zero-shot is poor.** Frozen FM representations do not transfer cleanly.

The classification framing keeps the *same* failure modes but adds the most common BCI deployment pattern: **N-class motor-imagery command decoding** (left/right/feet/hand selection for control). This is the regime where ITR, balanced accuracy, and Cohen's κ matter, and where most of the public benchmarking lives.

Evidence: Liu et al. (2601.17883), Cross-Domain EEG Survey (Aug 2025), NeuroAdapt-Bench (2604.16926), NeurIPS 2025 EEG Foundation Challenge, BCI-IV-2a leaderboards.

---

## 2. Refined Scope

**Goal:** a calibration-light, online-adaptive *4-class motor-imagery classifier* with sub-200 ms CPU inference + adaptation.

**Hard constraints:**
- **Task type: 4-class motor-imagery classification.** Primary: {LeftFist, RightFist, BothFists, BothFeet}. Optional 5-class with Rest.
- **Online adaptation.** Updates streaming (per-window or per-trial), not batched.
- **Latency: sub-200 ms total inference + adaptation on consumer CPU.** No GPU at deploy.
- **Small / low-resource.** Per-subject state tiny enough to fit on a wearable.
- **Fast convergence.** Adapter should reach acceptable accuracy in seconds to a few minutes, not an hour.
- **Unsupervised preferred, supervised fallback.** Label-free calibration is the win; supervised k-trial calibration is the safety net.

**Demo day constraint:** 16-electrode cap at inference; PhysioNet EEGMMI is recorded with 64 channels (10-10). Pick a clinically motivated motor-cortex 16-channel subset (C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4, FC3, FC1, FCz, FC2, FC4, Pz — to be finalized). Decide the train-64 → deploy-16 strategy early.

**N-of-trials regime, not N-of-minutes.** With ~21 trials/class/subject, the relevant calibration budget is *number of labeled trials per class*, not minutes. Sweep `K_trials ∈ {0, 1, 2, 5, 10, 20}`.

---

## 3. Dataset: PhysioNet EEG Motor Movement/Imagery (EEGMMI)

Schalk et al. 2004 (BCI2000), distributed via PhysioNet (`eegmmidb`). 109 subjects, 14 EDF runs each.

- **109 subjects**, **64 channels** at **160 Hz** (handful at 128 Hz → resample). 10-10 montage, BCI2000 channel names.
- **Standard exclusions:** subjects {88, 89, 92, 100, 104} (sampling-rate / annotation issues — universal in the literature).
- **Effective N:** ~104 subjects.
- **Per-subject runs:**
  - 1: baseline eyes open (60 s, unlabeled)
  - 2: baseline eyes closed (60 s, unlabeled)
  - 3, 7, 11: executed open/close left or right fist
  - 4, 8, 12: imagined open/close left or right fist
  - 5, 9, 13: executed open/close both fists or both feet
  - 6, 10, 14: imagined open/close both fists or both feet
- **Annotations per run:** T0 = rest, T1 = "left fist" or "both fists" (run-dependent), T2 = "right fist" or "both feet" (run-dependent).
- **Trial epoch:** primary `[0, 4]` s post-cue (160 samples/s × 4 s = 640 samples per trial). Secondary `[0.5, 2.5]` s ablation later.
- **Class labels (primary 4-class MI from imagery runs only):**
  - 0 = LeftFist  (runs {4,8,12}, T1)
  - 1 = RightFist (runs {4,8,12}, T2)
  - 2 = BothFists (runs {6,10,14}, T1)
  - 3 = BothFeet  (runs {6,10,14}, T2)
- **Approximate trial counts per subject:** ~21 per class, ~84 per subject, ~8.7K total post-exclusion. Class balance is approximately uniform.
- **Loading:** `mne.datasets.eegbci.load_data(subject, runs)` → EDF paths; `mne.io.read_raw_edf` + `mne.events_from_annotations` → standard epochs.

**Why this dataset:**
- 109 subjects → real LOSO with meaningful variance estimates (vs 12 for WAY-EEG-GAL). Meta-learning becomes feasible.
- Discrete 4-class MI is *the* BCI workhorse; lets us report all the conventional classification metrics (accuracy, balanced accuracy, F1, AUC, Cohen's κ, ITR) honestly.
- 64-ch recording cleanly subsets to a 16-ch demo motor subset.
- Open, free, well-benchmarked (EEGNet/ATCNet/CTNet leaderboards on this exact dataset).
- Trial-aligned, fixed-length — pipeline is simpler than continuous decoding.

**Trade-offs vs WAY-EEG-GAL:**
- Tiny per-subject trial count (~84) → small-K_trials regime is the natural domain.
- Class label rather than continuous target → easier to game with shortcut features; mitigate with permutation null and per-class confusion reporting.
- Two known annotation bugs in the source EDF files → bake-in the standard exclusion list at the loader level.

---

## 4. Foundation Models

Same minimum set as `plan.md`: **CBraMod (primary) + REVE (channel-agnostic comparison) + LaBraM-Base (canonical baseline).** No change here — three backbones is enough to claim FM-agnostic method.

EEGMMI-specific notes:
- All three FMs were pretrained at 200 Hz patch length 200. EEGMMI at 160 Hz → upsample to 200 Hz at the FM-input boundary (same as the existing 100→200 Hz path for WAY-EEG-GAL).
- 4-second trials at 200 Hz = 800 samples = 4 patches. Comfortably above LAtte's flagged short-trial regime (≤2 patches). No risk here.
- LaBraM channel-embedding subsetting is now easier: 64 → 16 is a strict subset of the 10-10 names LaBraM was pretrained on.

**Verification step:** for each FM, confirm that the EEGMMI 64-channel name set (`Fc5., Fc3., …`) maps cleanly into the model's expected montage. The trailing-dot BCI2000 convention is easy to miss.

---

## 5. Baselines

The classification-flavored counterpart of `plan.md` §5:

- **CSP + LDA.** Filter-bank CSP optional. The reviewer-mandatory floor for 4-class MI.
- **Riemannian + MDM / Riemannian + LR.** Tangent-space logistic regression. Lower bound — if FMs can't beat this, the FM premise is wrong for this regime.
- **EEGNet.** Standard small-CNN baseline (~2K params). Reports on this exact dataset in the original EEGNet paper.
- **ShallowConvNet / DeepConvNet.** Classical EEG DL baselines.
- **EEGNet + per-subject fine-tuning.** Supervised baseline for K-trial protocol.
- **Frozen CBraMod + linear classifier (softmax probe).** "FM doesn't actually help" null hypothesis.
- **Full fine-tuning of CBraMod.** Upper bound for supervised adaptation.

Drop: Bradberry mLR, Ridge bandpower regression (regression-only baselines).

---

## 6. Experimental Setup

**Two evaluation protocols:**

1. **Zero-shot LOSO.** Train on 103 subjects, evaluate on the held-out subject with *no* per-subject data. Tests cross-subject generalization. The pure foundation-model claim. With 104 subjects this gives proper variance estimates.

2. **K-trials fine-tuning.** Same LOSO but allow `K` labeled trials *per class* from the held-out subject for calibration. Sweep `K_per_class ∈ {0, 1, 2, 5, 10, 20}` (20 ≈ all available). Report accuracy curve vs K. The deployment-relevant metric.

**Within-session drift evaluation:** EEGMMI runs are alternating; trials early/mid/late in the same imagery run can be used as a coarse drift proxy. Better drift handling will come from synthetic perturbations injected mid-stream (see §8 RED).

**Optional cross-paradigm transfer:** train on imagery runs {4,8,12,6,10,14}, evaluate on execution runs {3,7,11,5,9,13}. Tests whether the FM-aligned representation transfers across the imagery/execution boundary. Stretch goal.

---

## 7. Metrics

### Primary: 4-class motor imagery classification

- **Top-1 accuracy** (macro-averaged over classes). Headline number.
- **Balanced accuracy** (macro-recall). Robust to small class imbalances introduced by trial-dropping.
- **Cohen's κ.** Chance-corrected; standard in MI literature, directly comparable to BCI-IV-2a numbers.
- **Macro F1.**
- **AUC-OVR** (one-vs-rest, macro-averaged). Threshold-free model-quality signal.
- **Per-class confusion matrix.** Reveals which class pairs the model conflates (typically L/R vs hands/feet split).

### Calibration efficiency (deployment metric)

- **K-trial curve.** X: labeled trials per class. Y: accuracy / κ. Every method gets a curve. **Headline plot.**
- **Trials-to-target-accuracy.** Number of labeled trials to reach 80% of asymptotic per-subject accuracy.
- **Data efficiency ratio.** `K_trials(FM + method) / K_trials(EEGNet from scratch)` at matched accuracy. <1 → FM helps; =1 → FM is overhead; >1 → FM is harmful.

### Cross-subject generalization (main protocol)

- **LOSO mean ± std across 104 subjects** for every primary metric.
- **Per-subject scatter plot** of accuracy. Reveals whether the method helps everyone or only on average.
- **Wilcoxon signed-rank for paired method comparisons** (not t-test).
- **Negative-transfer rate.** Fraction of subjects where FM + method does worse than EEGNet-from-scratch.

### Latency (hard constraint)

- **End-to-end inference latency on consumer CPU.** Median + p95 over ≥1000 windows. Specify CPU (M-series single thread).
- **Adaptation update latency.** Per-window cost of updating per-subject state.
- **Memory footprint of per-subject state** in bytes.
- **Sustained throughput** (samples/sec) with adaptation running.

### Drift handling

- **Pre/post synthetic perturbation accuracy.** Inject channel-gain perturbation or 1/f noise mid-session; measure accuracy recovery time.
- **Early-vs-late-trial accuracy** within an imagery run, with/without online adaptation.

### Robustness (demo story)

- **16-channel vs 64-channel accuracy.** Direct demo-scenario measurement.
- **Random channel dropout robustness.** Train with channel dropout; report accuracy at varying dropout rates.

### Statistical reporting

- Mean ± std across LOSO folds, never just mean.
- Bootstrap 95 % CIs on accuracy and κ (10K samples).
- Permutation null on accuracy (shuffle held-out labels, 1K perms) — distinguishes "barely above chance" from "actually decoding".
- Wilcoxon signed-rank for paired comparisons; exact p-values; no asterisk theater.
- Per-subject results in appendix.

### What NOT to report

- Pearson r / R² / RMSE — those belong to the regression plan.
- ITR for systems we haven't actually deployed online with a stop condition. Compute it if it matters for headline storytelling, but do not invent a delivery time.
- Single-subject "best fold" numbers as headlines.

### Headline summary table (paper's main figure)

| Method | Accuracy (LOSO) | κ | K_trials to 80 % | CPU latency (ms) | Memory (KB) | 16ch acc. |
|---|---|---|---|---|---|---|
| EEGNet | | | | | | |
| ShallowConvNet | | | | | | |
| CSP + LDA | | | | | | |
| Riemannian + LR | | | | | | |
| CBraMod frozen + softmax probe | | | | | | |
| CBraMod + RA (RED) | | | | | | |
| CBraMod + LoRA (PINK) | | | | | | |
| CBraMod + ConvexNN (YELLOW) | | | | | | |
| CBraMod + RA + LoRA (full system) | | | | | | |

Every experiment exists to fill a cell in this table.

---

## 8. Methods (Triaged)

Method triage is unchanged from `plan.md`; only the per-method details that depend on task type change.

### 🔴 RED — Online Riemannian / Euclidean Alignment

Identical motivation and update-rule menu as the regression plan (sliding-window, EMA, Kalman, recursive Karcher). One classification-specific note: **per-class covariance pooling vs global pooling** is now a live design question. Classical RA computes a single reference covariance per session; for MI it is often more effective to align using only rest-class covariances (T0 segments) since T0 trials are interspersed throughout the recording and give a label-free reference. Decide between:
- Global (all trials, no labels) — fully unsupervised.
- Rest-only — label-free if we trust rest annotations; otherwise unsupervised heuristic (e.g. low-power windows).
- Per-class supervised — needs labels.

The unsupervised online story prefers the first two.

### 🟢 GREEN — AlphaEvolve-discovered alignment

Same moonshot as regression plan. Fitness function changes: cross-subject classification accuracy on EEGMMI LOSO (cheap to evaluate) instead of Pearson r.

### 🟡 PINK — Per-subject LoRA on CBraMod

Same design. Classification head is a softmax linear on the FM CLS / pooled embedding instead of a linear regression head. LoRA still applied to attention projections. Rank sweep `r ∈ {1, 2, 4, 8, 16}`; layer-placement ablation.

**Open question (unchanged):** can the LoRA adapter be fit unsupervised via the FM's reconstruction objective? On EEGMMI, an additional unsupervised target is available: predicting the imagery class from a self-supervised pretext (e.g. trial-pair contrastive on same-subject windows).

### 🟡 YELLOW — Convex NN on frozen FM embeddings

Pilanci-Ergen convex two-layer NN; classification variant uses cross-entropy loss (convex in the parameters of the *equivalent convex program*, per the standard reformulation). Closed-form / CVXPY-solver-based, per-subject fit on `K_trials × n_classes` labeled examples. Still the method most distinctively ours.

### 🟡 YELLOW — Continued pre-training

Same as regression plan: domain-adaptive pretraining on EEGMMI baseline runs (1, 2) which give 60 s × 2 × 109 ≈ 3.6 hours of unlabeled in-domain EEG. Modest scale, modest expectations.

### 🟡 YELLOW (demoted → re-promoted?) — MAML / Reptile

With 104 meta-training subjects, MAML / Reptile is actually viable here (unlike the 12-subject regression regime). **Promote to a real comparison.** Reptile first; true MAML if Reptile plateaus.

---

## 9. Demo Day Strategy

64 → 16 channel deployment. Three options (identical to regression plan, but channel ordering changes):

1. **Channel-agnostic backbone (REVE).** Handles arbitrary configurations natively.
2. **Random channel dropout during fine-tuning.** Train CBraMod with random subsets masked → handles 16-channel input.
3. **Train two models.** Doubles compute.

Recommended demo subset (motor cortex, 10-10): {C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4, FC3, FC1, FCz, FC2, FC4, Pz}. Final list pending availability in EEGMMI channel set (verify against BCI2000 names with trailing dots).

---

## 10. Strategic Sequencing

**Week 1: Data pipeline rewrite.**
- New `src/data/eegmmi.py` (MNE-based loader, exclusions, epoch extraction, 160→200 Hz resample at FM boundary).
- Replace `src/data/way_eeg_gal.py` as the default dataset (keep the file for the regression branch but not wired into configs).
- Update `src/core/types.py` `Trial` to carry a `label: int` field instead of `kin/vel` (or add it as optional and keep the regression fields for the regression branch).
- New protocols: `loso.py` and `kmin_calibration.py` parameterized by trials-per-class instead of minutes.
- New `Head`: `softmax_probe` (logistic regression / linear softmax) and `softmax_finetune`.
- New eval bundle: accuracy / balanced acc / κ / F1 / AUC-OVR / confusion / per-class CI / permutation null.

**Week 2: Baselines + EEGNet sanity check.**
- CSP+LDA, Riemannian+LR (pyriemann), EEGNet, ShallowConvNet.
- Sanity-check EEGNet against published EEGMMI accuracy (~65 % for 4-class LOSO).

**Week 3: Frozen-FM linear probes + RED branch.**
- CBraMod / LaBraM / REVE frozen + softmax probe (no calibration).
- Static and online RA variants in front of the probe.
- Latency benchmarks at 64 and 16 channels.

**Week 4: PINK branch (per-subject LoRA on CBraMod).**
- LoRA adapter, supervised K-trial calibration.
- Rank + placement ablation.

**Week 5: YELLOW branch (convex NN on embeddings) + Reptile.**
- Convex two-layer NN classification.
- Reptile meta-learning across 103 source subjects.

**Week 6: Combination experiments + 16-channel demo.**
- RED + PINK, RED + ConvexNN.
- 16-channel evaluation across all methods.

**Week 7: Drift experiments + paper figures.**

**Slack: continued pre-training, AlphaEvolve, cross-paradigm execution-vs-imagery transfer.**

---

## 11. Open Questions / Decisions

These are the items that require a human judgment call before or during implementation. See §13 for the **analysis vs implementation** breakdown.

- [ ] **Primary class set:** 4-class MI {L, R, BothFists, BothFeet} (recommended) vs 2-class {L, R} (simpler, more comparable to BCI-IV-2a) vs 5-class adding Rest.
- [ ] **Trial epoch window:** `[0, 4]` s (more samples, more drift) vs `[0.5, 2.5]` s (the "clean" MI window).
- [ ] **Pre-cue baseline:** subtract / not subtract / per-trial z-score.
- [ ] **Exclude execution runs entirely from training?** Or use them as additional source data with a paradigm token.
- [ ] **Demo 16-channel set:** confirm final list against EEGMMI montage availability.
- [ ] **RA reference distribution:** global vs rest-only vs per-class.
- [ ] **K_trials sweep range:** `{0, 1, 2, 5, 10, 20}` per class — confirm 20 isn't already all data after splits.
- [ ] **Meta-learning method:** Reptile first vs Reptile + MAML.
- [ ] **Drift protocol:** within-run early/late split vs synthetic perturbation only.
- [ ] **CBraMod softmax head architecture:** linear-on-CLS vs linear-on-mean-pool vs 1-layer MLP.
- [ ] **Reference CPU for latency:** M-series single thread or Intel x86 reference.

---

## 12. Key References (curated)

**EEGMMI-specific:**
- Schalk et al. 2004 — BCI2000 / EEGMMI source.
- Lawhern et al. 2018 — EEGNet (reports on this exact dataset).
- Zhang et al. 2024 — survey of methods on PhysioNet MMI.

(EEG-FM landscape, adapter / TTA, Riemannian methods, convex NNs: same references as `plan.md` §12.)

---

## 13. Analysis vs Implementation

> **The user asked: "lmk what is analysis."**
> This section makes the distinction explicit. "Analysis" = decisions, research questions, judgment calls, experimental design — things that require thinking, not coding. "Implementation" = code that someone could be told to write without further design input.

### 13.1 Analysis (decisions / experimental design — must be made by us before / during the work)

These map 1-to-1 to the open questions in §11. Roughly in the order they need to be answered:

1. **Class set & label mapping** — choose 4-class (recommended), 2-class, or 5-class. Determines the head's output dim, the eval table, and how comparable our numbers are to public benchmarks.
2. **Trial epoch window** — `[0, 4]` s vs `[0.5, 2.5]` s. Affects every downstream metric and FM patch count.
3. **Pre-processing pipeline** — referencing scheme (CAR vs no), band-pass range, per-trial z-score yes/no (still breaks RA — same gotcha as the regression branch). Decide whether to maintain two preprocessing paths (RA-friendly + FM-friendly).
4. **Source-subject pool** — use all 104 imagery-run subjects, or also pool execution runs as source data.
5. **K-trials protocol** — what exact subset of held-out trials counts as "calibration" — balanced per class? First N trials? Random per seed?
6. **RA reference distribution** — global vs rest-only vs per-class (label-free vs supervised).
7. **Demo 16-channel set** — clinical motivation + verify availability.
8. **Headline metric** — accuracy vs κ. Recommend κ as the chance-corrected statistic but report both.
9. **What the FM's class signal is being read off of** — CLS token vs mean-pool vs last-patch — this changes the linear probe.
10. **Stop conditions and definitions of "convergence"** for the K-trials-to-target metric.
11. **Statistical-comparison protocol** — paired Wilcoxon across 104 subjects; CI method; multiple-comparison correction across the headline table.

These are the items where we should *think before typing*. The right move at the start of the week is to fix items 1–4 explicitly, push 5–11 into a "design log" that gets updated as data arrives.

### 13.2 Implementation (code to write — no further design input needed once §13.1 is set)

These are largely mechanical given the analysis decisions above. Most can be parallelized.

**Data layer (Week 1):**
- `src/data/eegmmi.py` — MNE loader, exclusion list, epoch extraction at chosen window, 160→200 Hz resample, channel-name normalization, per-trial label assignment.
- `src/data/channels.py` — extend with EEGMMI montage and the demo 16-channel subset.
- `src/data/splits.py` — extend with `n_trials_per_class` K-budget logic.
- `src/core/types.py` — add `label: int` to `Trial`; mark `kin/vel` as optional (or drop entirely on this branch since the user OK'd that).
- `src/configs/dataset/eegmmi.yaml`.

**Protocols (Week 1):**
- `src/protocols/loso.py` — parameterize by class set; trivial change.
- `src/protocols/kmin_calibration.py` → `ktrials_calibration.py` — replace minute-budget with trials-per-class-budget.
- New config files under `src/configs/protocol/`.

**Heads (Week 1):**
- `src/heads/softmax_probe.py` — sklearn `LogisticRegression` (or a tiny torch linear) on the pooled FM embedding; implements `.fit` and `.predict_proba`.
- `src/heads/softmax_finetune.py` — trainable linear head used in FM finetune mode.
- Update `Head` base + configs.

**Eval (Week 1):**
- `src/eval/metrics_clf.py` — accuracy, balanced accuracy, κ, F1, AUC-OVR, per-class confusion, bootstrap CIs, label-shuffle permutation null.
- `src/eval/flatten_for_logging` — add classification keys; remove regression keys.

**Backbones (Week 2-3):**
- No new backbones. CBraMod/LaBraM/REVE wrappers already accept arbitrary heads. Verify 160→200 Hz upsample path and 64-channel name mapping per FM.

**Baselines (Week 2):**
- `src/backbones/csp_lda.py` (mne.decoding.CSP + sklearn LDA pipeline).
- `src/backbones/riemann_lr.py` (pyriemann tangent-space LR).
- Drop `bradberry_mlr.py`, `ridge_bandpower.py` from configs.

**Adapters (Week 3-5):**
- Reuse `src/adapters/riemannian/{static,ema}.py` — they are label-free and class-agnostic. Add the rest-only-reference option.
- Implement `src/adapters/lora.py` (CBraMod attention modules — same custom mapping note as the regression plan).
- Implement `src/adapters/convex_nn.py` (CVXPY two-layer for classification).
- Implement `src/adapters/reptile.py` (meta-learning across source subjects).

**Latency / memory (any time):**
- Wire `src/eval/latency.py` and `src/eval/memory.py` into the runner summary. Same hooks as the regression branch.

**Scripts:**
- `scripts/download_eegmmi.py` — MNE auto-fetch + exclusion list logging.
- `scripts/run_full_baselines_clf.py` — analogous to existing runner, dispatches the headline-table experiments.
- `scripts/make_figures_clf.py` — populate `results/summary_table_clf.md` from W&B (or offline JSON).

**Tests:**
- Smoke test on 2 subjects × 1 imagery run that EEGNet hits > chance.
- Unit tests for epoch extraction (labels match run convention).
- Unit test for K-trials split balance.

### 13.3 What we are *deleting* (because the user said the branch doesn't need to stay backward-compatible)

- `src/data/way_eeg_gal.py` (loader), `src/configs/dataset/way_eeg_gal.yaml` — keep only if we want a regression sub-branch later; otherwise drop.
- `src/backbones/bradberry_mlr.py`, `src/backbones/linear_features.py` (Bradberry mLR is regression-only), `src/configs/backbone/{bradberry_mlr, ridge_bandpower}.yaml`.
- `src/eval/metrics.py` regression-only fields — replace wholesale.
- Existing `src/configs/experiment/*` files — rewrite around the new headline table cells.
- Regression-specific tests under `tests/`.

---

## 14. Notes / Scratch

(continue here)
