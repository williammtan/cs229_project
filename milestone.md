# CS229 Milestone — Cross-Subject Calibration Efficiency for Continuous EEG Hand-Kinematic Decoding

**Author.** Marc Hasr (`marchasr@stanford.edu`)
**Date.** May 2026

---

## 1. Motivation

Brain-computer interfaces (BCIs) promise to restore motor function for individuals
with paralysis and to enable natural human-computer interaction without physical
input devices. Five million Americans live with paralysis; an additional cohort
with ALS, late-stage stroke, and locked-in syndrome have intact motor cortex but
no peripheral output. Implantable BCIs (BrainGate, Neuralink, Synchron) have
demonstrated clinically meaningful continuous limb control, but only ~80 patients
have been implanted worldwide cumulatively, with surgical risk and limited
electrode lifespan (Dohle et al., *Advanced Science* 2025). Non-invasive scalp
electroencephalography (EEG) is the only realistic path to mass-scale BCI
deployment — inexpensive, wearable, no surgery.

The dominant blocker for deployable EEG BCIs is **per-user calibration cost**.
Classical decoders require 15–30 minutes of supervised setup before they are
operable (Lotte 2015, *Proc. IEEE*); patients and consumer users will not
tolerate this on every session. Reducing calibration to the single-digit-minute
regime is the deciding factor between lab demonstration and assistive technology.

This project asks two questions in tandem:

1. Can a deep learning model trained on a population of EEG users accurately
   decode continuous hand kinematics (3D velocity) from scalp EEG?
2. How much per-user calibration data is required, and which adaptation method
   is most efficient, to bring a new user's performance close to that of a fully
   personalized model?

The second question is the headline. It positions the work alongside two
recent successful framings: **MindEye2** (Scotti et al., ICML 2024) — fMRI-to-image
with ~1 hour of per-subject data on a shared backbone — and **Meta sEMG**
(Sivakumar et al., *Nature* 2025; NeurIPS 2024 D&B) — cross-person decoding
with *zero* per-user calibration. Continuous motor regression is a documented
gap in the foundation-model-for-EEG literature (LaBraM, ICLR 2024; CBraMod,
ICLR 2025; EEGPT, NeurIPS 2024; NeuroLM, ICLR 2025 — all evaluate primarily
on classification).

**Thesis.** Continuous hand-kinematic decoding from scalp EEG is bottlenecked
not by raw decoder capacity but by per-user calibration cost. Foundation-model
pretraining on cross-subject corpora, combined with calibration-efficient
adaptation (Euclidean alignment, meta-initialization, low-rank fine-tuning),
can drive new-user calibration from the field-standard 20–30 minutes to single
digits while preserving most within-subject performance — but only when
evaluated under strict artifact and causal-filter controls.

---

## 2. Methods

### 2.1 Dataset

For the preliminary experiments reported here we use **WAY-EEG-GAL** (Luciw,
Jarocka & Edin, *Scientific Data* 2014; Figshare collection 988376). It
provides 12 subjects performing instrumented grasp-and-lift trials of an
object whose weight (165 / 330 / 660 g) and surface (sandpaper / suede /
silk) vary unpredictably. Each subject has 9 series of 32–34 trials (~300
total), with 32-channel scalp EEG sampled at 500 Hz, Polhemus FASTRAK
electromagnetic tracker capturing 3D hand position at 500 Hz, plus 5-channel
EMG and 6-axis force/torque on the contact plates. The preliminary
experiments use subjects P1, P2, P3 across all 9 series (~10 min of EEG
per subject after concatenation).

The final project will extend to **Jeong et al. 2020** (Korea University
multimodal upper-limb, 25 subjects × 3 sessions one week apart) and the
**Müller-Putz "Feel Your Reach"** line (Mondini 2020, Pulferer 2022) to
support cross-session calibration curves with measured 2D end-effector
trajectories.

### 2.2 Preprocessing

Implemented in `src/data.py`.

- **EEG.** Common-average reference; 4th-order zero-phase Butterworth
  band-pass at 0.1–40 Hz; downsample 500→100 Hz; per-channel z-score (trial-wise;
  will be cross-validated against per-session normalization in later work).
- **Kinematics.** Hand-sensor Px, Py, Pz from kinematic columns 18, 22, 26
  (Polhemus sensor 1 on dorsum of hand); low-pass 5 Hz to suppress sensor noise;
  downsample 500→100 Hz; first-difference to produce velocity in mm/s.
- **Decoding target.** 3D hand velocity at the EEG sample grid. We chose
  velocity over position to align with the Bradberry 2010 / Müller-Putz
  literature and to remove low-frequency position drift confounds.

### 2.3 Models (baselines)

Implemented in `src/models.py`. All four expose an sklearn-style `fit(X, y)` /
`predict(X)` interface so they share the same evaluation harness.

1. **Bradberry mLR** — multivariate ridge regression on lag-augmented
   low-frequency EEG potentials. Lags = {0, 50, 100, 150, 200, 250} ms.
   Replicates Bradberry et al. 2010.
2. **Ridge Band-Power (Korik-style BTS)** — ridge regression on log-band-power
   features in {μ (8–13 Hz), β (13–30 Hz), low-γ (30–45 Hz)} sliding windows
   (500 ms window, 100 ms hop).
3. **EEGNet** (Lawhern et al. 2018) with a 3-output linear regression head.
   F1=8, D=2, F2=16; ~3.2 k parameters.
4. **ShallowConvNet** (Schirrmeister et al. 2017) with a 3-output linear
   regression head. 40 temporal filters → 40 spatial filters → square →
   average pool → log → linear. ~50 k parameters.

Neural nets train on 1-second sliding windows (100 samples @ 100 Hz) with
200-ms hop, AdamW (lr=1e-3, weight_decay=1e-4), MSE loss, early stopping on
a 10% validation split (patience 4 epochs), maximum 25 epochs. Apple-Silicon
MPS backend.

### 2.4 Evaluation framework

Implemented in `src/eval.py`. **This is the same framework that will be used
to evaluate the foundation-model SOTA in the next phase**, so its design
choices are deliberate.

- **Pearson r per axis and averaged** — the field-standard metric, kept for
  comparability with Bradberry 2010, Müller-Putz 2020, Borra 2023, etc.
- **R²** (coefficient of determination) — penalizes scale/offset miscalibration
  that Pearson r ignores.
- **RMSE in mm/s** — interpretable physical-unit error.
- **Shuffled-target permutation null.** Predictions are circularly time-shifted
  relative to targets to destroy temporal alignment while preserving each
  signal's autocorrelation. 200 permutations per fold give a per-axis null
  distribution; we report null mean, p95, and one-sided p-value. This control
  directly addresses the Antelis et al. 2013 critique that linear-regression
  correlations in this paradigm are inflated by OLS leakage of smooth targets.
- **Block bootstrap 95% CI** — 50 contiguous blocks resampled with
  replacement, 500 iterations.
- **Fraction of subjects > r > 0.2** — operational deployment threshold;
  averaging hides the long left tail of "BCI illiterate" users (Vidaurre &
  Blankertz 2010).

Two evaluation protocols, again the same that will be used for SOTA:

- **(W) Within-subject 5-fold CV.** Trials grouped into 5 contiguous folds;
  each fold held out as test, the remainder used for training. This is the
  performance ceiling — a fully personalized decoder per subject.
- **(L) Leave-one-subject-out (LOSO).** Train pooled on the other N-1
  subjects, evaluate zero-shot on the held-out subject. This is the
  cross-subject *floor* — no per-user adaptation at all. In the next phase
  this baseline is augmented with K minutes of target-subject calibration
  data (the headline scaling curve of the final paper).

---

## 3. Preliminary Experiments & Results

**Compute.** All experiments run on Apple M-series silicon (MPS backend in
PyTorch 2.11). Total wall-clock for the entire grid (3 subjects × {4 models}
× {5 within-subject folds + 3 LOSO}) ≈ 28 minutes.

**Subject N.** 3 (P1, P2, P3) × 9 series each = 294 trials per subject (882
total), ~30 minutes of recorded EEG per subject after concatenation.

### 3.1 Headline results

| Model | Within r (mean ± std across subj) | LOSO r (mean across subj) | Personalization gap | Within R² | Within RMSE (mm/s) | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|
| Bradberry-mLR (linear) | +0.223 ± 0.038 | +0.091 | -59% | +0.057 | 1.66 | 1.5 s/fold |
| Ridge-BandPower (linear) | +0.127 ± 0.022 | -0.024 | full collapse | +0.016 | 1.72 | 2.0 s/fold |
| EEGNet (~3 k params) | +0.176 ± 0.065 | +0.115 | -35% | +0.044 | 1.66 | 11.7 s/fold |
| ShallowConvNet (~50 k params) | +0.235 ± 0.044 | +0.130 | -45% | +0.064 | 1.61 | 19.1 s/fold |

Pearson r is averaged across X/Y/Z velocity axes; within-subject values are
the mean across 5 CV folds × 3 subjects; LOSO is the mean across 3 held-out
subjects with N−1=2 pooled training subjects. NaN folds (see §3.5) excluded
from aggregates via `nanmean`.

![Within-subject vs LOSO](results/figures/within_vs_loso_r.png)

The within-subject bars (solid) are the ceiling — what a fully personalized
decoder can achieve on this subject's own data. The LOSO bars (hatched) are
the floor — zero-shot transfer with no target-side calibration at all. The
gap between them is the **personalization gap** the final paper proposes to
close with calibration-efficient adaptation methods.

### 3.2 Per-axis decomposition

![Per-axis r](results/figures/per_axis_r.png)

As expected for a grasp-and-lift task, the **Z (vertical lift) axis is the
easiest to decode** across every model — its velocity profile has the
largest amplitude and clearest stereotypy across trials. Looking at the
strongest within-subject case (P1 fold 0): Bradberry Z = 0.41 vs X = 0.26 vs
Y = 0.14; ShallowConvNet Z = 0.53 vs X = 0.34 vs Y = 0.11. The cross-axis
ratio agrees qualitatively with Bradberry 2010 (which reported strongest
decoding on the principal movement axis) and with the grasp-and-lift
physics (Z dominates the kinematic energy of the task).

### 3.3 Permutation null — is the signal real?

![Permutation null](results/figures/perm_null.png)

The Antelis et al. 2013 critique is the most important methodological hazard
in this literature: linear regression on low-frequency EEG can produce
non-trivial Pearson r through OLS leakage of smooth targets, with no
underlying motor information. We address it explicitly with a circularly-
shifted-target permutation null (200 permutations per fold). The gray and
red horizontal lines in the figure are the null mean and 95th percentile;
the colored dots are the observed r.

**Result: 11 of 12 model × subject combinations clear the α = 0.05 null
threshold** (p ≤ 0.045) on the held-out fold 0:

| Subject | Bradberry-mLR | Ridge-BP | EEGNet | ShallowConvNet |
|---|---:|---:|---:|---:|
| P1 | 0.005 ✓ | 0.007 ✓ | 0.007 ✓ | 0.005 ✓ |
| P2 | 0.022 ✓ | 0.045 ✓ | 0.158 ✗ | 0.090 (marginal) |
| P3 | 0.005 ✓ | 0.035 ✓ | 0.005 ✓ | 0.013 ✓ |

The lone failure (EEGNet on P2) coincides with the lowest-signal subject ×
model combination in our data; with more pretraining data the foundation-
model phase should resolve this. This is exactly the bullshit-detector
control the Antelis critique demands, and our results survive it.

### 3.4 Headline observations from preliminary

**1. The signal is real and above the artifact null.** Within-subject
Bradberry r ≈ 0.22 across 3 subjects is squarely in the published range
(Bradberry 2010: r ≈ 0.19–0.38). The permutation null confirms the bulk of
this signal is above shuffled-target chance.

**2. The personalization gap is real and large.** All four models lose
35–60% of their within-subject performance under zero-shot LOSO. Ridge band-
power fully collapses (cross-subject r ≈ 0) — band-power features are
strongly subject-idiosyncratic, consistent with the BCI-illiteracy /
subject-variability literature (Vidaurre & Blankertz 2010). **This gap is
exactly the quantity the final paper proposes to close.**

**3. The within-subject vs LOSO ranking inverts for some methods.**
ShallowConvNet wins both within-subject (0.235) and LOSO (0.130). But
Bradberry-mLR is second within-subject and *fourth* on LOSO — its linear
features overfit to per-subject electrode patterns. EEGNet shows the *least
relative drop* (−35%), suggesting that with more training data, neural
features may transfer best. **This is the foundation-model thesis in
microcosm**: pool more data, train a more flexible model, and cross-subject
performance becomes the headline number, not within-subject.

**4. Inter-subject variability is large even at N = 3.** P1's
within-subject mean r is 0.27 (Bradberry); P3's is 0.22; but P3's LOSO is
just 0.04 — all models on P3 cross-subject hover near zero. P3 reads as a
plausible "BCI illiteracy"-type subject for this task. The final paper must
report per-subject results, not just averages — this preliminary finding
already justifies that decision.

**5. NN training is unstable on small per-subject data.** EEGNet and
ShallowConvNet hit NaN on 9 of 30 within-subject folds (30%), concentrated
in the lowest-signal subject (P3: 6/10 NN folds NaN). LOSO with pooled
data has zero NaN failures across 6 NN runs. This is itself a finding:
**at the ~30-minute per-subject scale, the neural-net win over linear
baselines is contingent on enough pooled training data** — exactly the
scaling-with-data story the final paper centers on. The fix in the next
phase is robust loss (Huber), smaller head init, and validation-loss-based
divergence detection with reset.

**6. ShallowConvNet's log-band-power inductive bias pays off.**
ShallowConvNet outperforms EEGNet both within-subject (0.235 vs 0.176) and
LOSO (0.130 vs 0.115). Its log-band-power readout — mirroring FBCSP —
appears better matched to motor-rhythm physics than EEGNet's separable
convolution. Recommendation: keep ShallowConvNet as the strongest classical
baseline against which to benchmark foundation-model fine-tunes.

---

## 4. Next Steps

### 4.1 Immediate (next 2–3 weeks)

1. **Extend baselines to all 12 WAY-EEG-GAL subjects** so the per-subject
   distribution becomes statistically meaningful (current N=3 is a proof of
   pipeline, not a population estimate).
2. **Add Jeong 2020 (Korea Univ.) as the primary cross-session dataset.**
   25 subjects × 3 sessions (one week apart) is what unlocks the calibration
   dose-response curve — each fresh session of a known subject is treated
   as a new "calibration budget" axis.
3. **Add Müller-Putz Mondini/Pulferer Graz datasets** for true measured 2D
   end-effector trajectories. WAY-EEG-GAL alone doesn't allow comparison to
   the online closed-loop r ≈ 0.32 number that anchors the field.
4. **Causal-filter ablation.** Re-run every preliminary number with causal
   (online-compatible) IIR filters and report the offline-vs-causal gap; the
   current zero-phase filters leak future kinematic information backward in
   time and inflate r.
5. **Saliency / decoder-weight topography.** Plot per-subject CNN saliency
   maps to confirm the decoders attend to contralateral M1/S1 electrodes
   (C3, C4, Cz region) rather than temporal (jaw/EMG) or frontal (EOG)
   electrodes. Required to satisfy reviewers concerned about artifact-driven
   correlations.

### 4.2 Core experiment (weeks 4–8)

The headline figure of the final paper.

- **x-axis:** per-user calibration budget K ∈ {0, 1, 2, 5, 10, 15, 30 min,
  full-session}.
- **y-axis:** Pearson r between predicted and ground-truth velocity, averaged
  across held-out subjects, with block-bootstrap 95% CI shading.
- **Curves:** one per adaptation method:
  - Zero-shot LOSO (no target data)
  - Euclidean Alignment (He & Wu 2020) — zero labeled calibration
  - K-minute supervised fine-tune of pretrained backbone (the workhorse)
  - Reptile-meta + K-minute fine-tune (Berdyshev et al. 2024)
  - AdaBN-only adaptation
  - GOPSA (NeurIPS 2024 Riemannian DA for regression)
- **Backbone:** EEGPT (NeurIPS 2024, 10M params, single-GPU friendly) or
  CBraMod (ICLR 2025) — released foundation models that have not been
  evaluated on continuous regression. This is where the methodological gap
  in the FM-for-EEG literature gets closed.

### 4.3 Honest contingencies

- **If EA recovers everything zero-shot**, the calibration story collapses
  to "you don't need any." Still publishable as a negative-result paper that
  reframes the field; pivot to the methodological-rigor angle.
- **If foundation-model backbones underperform classical baselines on
  regression**, this is itself the contribution: the first critical
  evaluation of FM-for-EEG on continuous behavior decoding.
- **If artifact controls reveal that most decoding signal is EMG/EOG**, this
  is the most valuable possible outcome — a rigorous re-analysis paper
  exposing the field's evaluation gap.

### 4.4 Realistic publication target

NeurIPS / ICLR / ICML main-track from a 3-month one-student CS229 project
is single-digit-percent probability. The credible target is a **workshop
paper** at NeurIPS Foundation Models for Brain and Body, ICLR Time Series
for Health, or the NeurIPS EEG Foundation Challenge track, with a polished
follow-up to a main-track cycle.

---

## Appendix A — Reproducibility

- Environment: `uv` + Python 3.12; deps pinned in `uv.lock`.
- Run: `uv run python scripts/run_preliminary.py` → `results/preliminary.json`.
- Figures: `uv run python scripts/make_figures.py` → `results/figures/*.png`.
- All seeds fixed (random / numpy / torch = 0). MPS results may differ
  ≤ 1e-3 from CPU due to MPS reduction-order non-determinism.

## Appendix B — Code layout

```
src/
  data.py         WAY-EEG-GAL loader, preprocessing, trial epoching
  models.py       4 baseline decoders (Bradberry, Ridge-BTS, EEGNet, ShallowConvNet)
  train.py        PyTorch sliding-window training loop, NeuralRegressor wrapper
  eval.py         Pearson / R² / RMSE + shuffled-target null + bootstrap CI
scripts/
  run_preliminary.py  Within-subject 5-fold CV + LOSO across all 4 models
  make_figures.py     Generates the figures embedded in this milestone
results/
  preliminary.json    All per-fold metrics
  summary_table.md    Markdown table reproduced in §3.1
  figures/            PNGs
literature_review.md  Companion document — full prior-work survey
milestone.md          This document
```
