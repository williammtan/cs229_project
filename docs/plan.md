# EEG Foundation Models: Closing the Deployment Gap (v2)

**Working document. Last updated: May 24, 2026.**
**Supersedes v1. Scope tightened: continuous decoding, online + low-resource, sub-200ms CPU latency.**

---

## 1. Problem Statement (unchanged)

Non-invasive EEG foundation models (LaBraM, CBraMod, REVE, NeuroLM, EEGPT, etc.) fail in real deployment due to:
- **Within-session drift.** Yesterday's calibration ≈ stranger's calibration on you.
- **Long calibration sequences.** 20-60 minute supervised sessions, BCI-illiterate users.
- **Cross-subject zero-shot is poor.** Frozen FM representations don't transfer cleanly.

Evidence: Liu et al. (arxiv 2601.17883), Cross-Domain EEG Survey (Aug 2025), NeuroAdapt-Bench (arxiv 2604.16926), NeurIPS 2025 EEG Foundation Challenge.

Diagnosis: current SSL objectives (masked reconstruction) optimize for reconstruction, not subject-invariance.

---

## 2. Refined Scope

**Goal:** a calibration-light, online-adaptive EEG decoder for continuous BCI control.

**Hard constraints:**
- **Task type: continuous regression.** Not classification. (Force / kinematic prediction in WAY-EEG-GAL framing.)
- **Online adaptation.** Updates streaming, not batched.
- **Latency: sub-200ms total inference + adaptation on consumer CPU.** No GPU at deploy.
- **Small / low-resource.** Per-subject state must be tiny enough to fit on a wearable.
- **Fast convergence.** Adapter should reach acceptable accuracy in seconds to a few minutes, not an hour.
- **Unsupervised preferred, supervised fallback.** Label-free calibration is the win; supervised k-min calibration is the safety net.

**Demo day constraint:** 16-electrode cap. Models must work with 16 channels at inference, even if trained on 32. Decide channel-handling strategy early.

---

## 3. Dataset: WAY-EEG-GAL

Luciw et al. 2014. Grasp-and-lift task.
- 12 subjects, ~290 trials each
- 32-channel EEG @ 500 Hz
- Paired kinematics (hand position, finger flexion, grip + load force) and EMG
- Supports two framings:
  - **Event detection:** 6 phases per trial (HandStart, FirstDigitTouch, BothStartLoadPhase, LiftOff, Replace, BothReleased). Classification-flavored.
  - **Continuous kinematic regression:** predict grip force / load force continuously. Truly continuous, harder, matches brief.

**Primary framing: continuous force regression.** Secondary metric: event detection AUC. Force is genuinely continuous and can't be gamed by classifier-in-disguise.

**Why this dataset:**
- Small enough for fast iteration (12 subjects, ~6 hours total recording)
- Continuous task with kinematic ground truth
- 32 channels supports 16-channel demo subset
- Published Kaggle baselines for sanity-checking
- Open-license, available on PhysioNet

---

## 4. Foundation Models

**Minimum set: CBraMod (primary) + REVE (channel-agnostic comparison) + LaBraM-Base (canonical baseline).**

Three backbones is enough to claim FM-agnostic method; more than three dilutes the story.

| Model | Size | Why include | Concerns | Verdict |
|---|---|---|---|---|
| **CBraMod** | ~5M (Base) | Strong motor imagery performance, ACPE handles variable channels better than LaBraM, small/fast, pretrained on TUEG (~26K hours), open weights | None major | **Primary backbone** |
| **REVE** | ~5M | Channel-agnostic via fixed 4D positional encoding — solves 16/32 demo natively, included in EEG-FM-Bench | Less established, fewer downstream evaluations | **Strong secondary; promote to primary if 16/32 handling is painful in CBraMod** |
| **LaBraM-Base** | 5.8M | ICLR 2024 spotlight, canonical reviewer-expected baseline, open weights | Fixed patch length 200; LAtte paper flagged underperformance on short trials (WAY-EEG-GAL @ ~10s × 500Hz = 25 patches, should be fine but verify); channel embeddings tied to electrode positions complicate 16/32 mismatch | **Required comparison baseline** |
| **EEGPT** | 10-25M | Dual-SSL objective; useful "does SSL design matter" ablation | Larger, latency risk on CPU | **Include if time** |
| **BIOT** | ~3M | Native arbitrary-montage tokenization, very small | Older, less competitive on recent benchmarks | **Skip unless small-FM story matters** |
| **TFM-Tokenizer** | ~10M | NeuroAdapt-Bench showed it's more robust to TTA than continuous-embedding FMs — possibly relevant to unsupervised online story | Discrete tokenization changes adapter design | **One comparison run if findings hold** |
| **LaBraM-Large / Huge** | 46M / 369M | Scale comparison | Will NOT hit sub-200ms on CPU | **Skip** |
| **DeeperBrain** | Unknown | Neuro-grounded priors | New, integration cost high | **Skip for v1, consider follow-up** |

**Verification step early:** confirm LaBraM channel embedding can be subset to 32 then 16 channels without breaking pretraining-time assumptions. Check `ch_names` handling in the official repo.

**Decision point:** if REVE's channel-agnostic property works cleanly out of the box, promote it to primary. Saves the channel-handling headache for the demo.

---

## 5. Baselines

- **Traditional ML.** Riemannian + LDA / Riemannian + linear regression (MDM-style). Lower bound — if FMs can't beat this, the FM premise is wrong for this regime.
- **EEGNet.** Standard small-CNN baseline. ~3K params.
- **ShallowConvNet / DeepConvNet.** Classical EEG DL baselines.
- **EEGNet + per-subject fine-tuning.** Supervised baseline for K-min protocol.
- **Frozen CBraMod + linear probe.** "FM doesn't actually help" null hypothesis.
- **Full fine-tuning of CBraMod.** Upper bound for supervised adaptation.

---

## 6. Experimental Setup

**Two evaluation protocols:**

1. **Zero-shot LOSO.** Train on 11 subjects, evaluate on held-out 12th with *no* per-subject data. Tests cross-subject generalization. The pure foundation-model claim.

2. **K-minutes fine-tuning.** Same LOSO but allow K minutes of held-out subject data for calibration. Sweep K ∈ {0, 0.5, 1, 2, 5, 10}. Report accuracy curve vs K. This is the deployment-relevant metric — how quickly does the system converge.

**Within-day drift evaluation (proposed addition):** if recordings are long enough, evaluate at start/mid/end of held-out subject's session. Measures whether online updates track drift.

---

## 7. Metrics

### Primary: continuous force regression on WAY-EEG-GAL

- **Pearson correlation coefficient (r)** between predicted and actual grip force, per-trial averaged. *Headline number.* Standard in continuous BCI literature.
- **R² (coefficient of determination).** Complementary view; penalizes scale errors that Pearson is invariant to.
- **RMSE in Newtons.** Interpretable, deployment-relevant.
- **Per-phase decoding.** Pearson r broken into rest / grasp / lift / hold / replace. Forces honesty about when the method actually works.

### Secondary: event detection (sanity check)

- **AUC-ROC per event**, averaged across 6 grasp-and-lift events. Matches original WAY-EEG-GAL Kaggle metric → direct comparison to existing baselines.

### Cross-subject generalization (main protocol)

- **LOSO mean ± std across 12 subjects** for every primary metric
- **Per-subject scatter plot.** Reveals whether the method helps everyone or only on average
- **Wilcoxon signed-rank for paired method comparisons** (not t-test; doesn't assume normality)
- **Negative transfer rate.** Fraction of subjects where FM + method does worse than EEGNet-from-scratch. Deployment-honesty metric papers usually hide.

### Calibration efficiency (deployment metric)

- **K-min curve.** X: minutes of supervised calibration. Y: Pearson r. Every method gets a curve. Single most important plot.
- **Time-to-target-accuracy.** Minutes to reach 80% of asymptotic performance.
- **Data efficiency ratio.** `K_min(FM + method) / K_min(EEGNet from scratch)` to reach matched accuracy. Direct FM-payoff measure. <1: FM helps. =1: FM is overhead. >1: FM is harmful.

### Latency (hard constraint)

- **End-to-end inference latency on consumer CPU.** Median + 95th percentile over ≥1000 windows. Specify CPU model (e.g., M-series Mac single thread, or Intel reference).
- **Adaptation update latency.** Per-window cost of updating the per-subject component, reported separately from inference.
- **Memory footprint of per-subject state** in bytes. Matters for wearable deployment.
- **Sustained throughput** in samples/sec with adaptation running continuously.

### Drift handling

- **Start vs end accuracy without adaptation.** Pure measurement of drift severity.
- **Start vs end accuracy with online adaptation.** Does the method recover?
- **Drift recovery time after synthetic perturbation.** Inject channel-gain perturbation or 1/f noise mid-session; measure Pearson r recovery time. Cleanest controlled drift experiment.

### Robustness (demo story)

- **16-channel vs 32-channel Pearson r.** Direct demo-scenario measurement.
- **Random channel dropout robustness.** Train with channel dropout; report accuracy at varying dropout rates.
- **Cross-dataset transfer (stretch).** Train on WAY-EEG-GAL, evaluate on another grasp dataset. Tests method generality.

### Statistical reporting

- Mean ± std across LOSO folds, never just mean
- Bootstrap 95% CIs on Pearson r (10K samples)
- Wilcoxon signed-rank for paired comparisons, exact p-values, no significance-asterisk theater
- Per-subject results in appendix

### What NOT to report

- Multi-class classification metrics on WAY-EEG-GAL. Committed to continuous regression — stay there.
- ITR (information transfer rate). Designed for discrete-command BCIs.
- Cohen's κ. Classification-only.

### Headline summary table (paper's main figure)

| Method | Pearson r (LOSO) | AUC (events) | K to 80% | CPU latency (ms) | Memory (KB) | 16ch Pearson r |
|---|---|---|---|---|---|---|
| EEGNet | | | | | | |
| ShallowConvNet | | | | | | |
| Riemannian + LDA | | | | | | |
| CBraMod frozen + linear probe | | | | | | |
| CBraMod + RA (RED) | | | | | | |
| CBraMod + LoRA (PINK) | | | | | | |
| CBraMod + ConvexNN (YELLOW) | | | | | | |
| CBraMod + RA + LoRA (full system) | | | | | | |

Every experiment exists to fill a cell in this table.

---

## 8. Methods (Triaged)

### 🔴 RED — Highest priority: Riemannian / Euclidean Alignment

**Why first:** closed-form, no training, sub-200ms by orders of magnitude, no hyperparameters, label-free, provides the baseline everything else builds on. If this doesn't work, nothing more sophisticated will.

**The actual research question:** *how fast can we update Riemannian alignment online?*

Classical RA computes the SPD covariance mean over a *static* calibration set. Online RA needs a rolling estimator. Concrete formulations:

- **Sliding-window mean.** Covariance over last *W* seconds, parallel-transport to source manifold. Simple, well-understood, latency dominated by Cholesky on a 32×32 matrix (~10μs).
- **Exponential moving average on the manifold.** Geodesic EMA of covariance estimates. Tighter latency, smoother updates. Harder to characterize.
- **Riemannian Kalman filter.** Track the alignment matrix as a state evolving on the SPD manifold. Higher complexity, but principled handling of measurement noise.
- **Recursive Karcher mean update.** Online estimation of the Frechet mean of covariances using stochastic gradient on the manifold.

**Research deliverables:**
- Latency benchmark of each variant on CPU at 32 and 16 channels
- Decoding accuracy vs update rate (Hz)
- Drift-tracking experiment: synthetic distribution shift (channel gain perturbation, additive 1/f noise) injected mid-stream, measure how fast each variant recovers
- Comparison to static (offline-computed) RA as a function of session length

**Why this is the right red item:** it directly answers whether *any* online unsupervised method can keep up with drift at the latency budget. If RA can be updated at 10Hz and tracks drift, you have a deployable baseline before doing anything ML-flavored.

**Risks:** RA assumes covariance shift is the dominant variability mode. If subject differences include non-covariance components (e.g., spectral peaks, nonlinear effects), RA hits a ceiling. Mitigation: that's exactly when LoRA helps.

---

### 🟢 GREEN — Stretch goal: AlphaEvolve-discovered alignment

**Concept:** use AlphaEvolve (or similar program-synthesis systems) to discover a better alignment update rule than hand-derived Riemannian methods. Search space: bounded operators on SPD matrices. Fitness: decoding accuracy on held-out subjects in WAY-EEG-GAL.

**Why this is interesting:** the standard RA recipe is ~15 years old (Barachant et al. era). Nobody has tried to *search* for better update rules. The objective is small, well-defined, easy to evaluate — exactly the kind of target where program synthesis can plausibly beat human-designed algorithms.

**Why this is risky:**
- AlphaEvolve not publicly accessible at scale yet
- Search budget could be large
- Discovered algorithms may not generalize beyond WAY-EEG-GAL
- Hard to publish if the discovered rule is uninterpretable

**Recommendation:** moonshot, pursue only if RED branch hits a ceiling and the ceiling is clearly in the update rule (not in the fundamental approach). Otherwise it's an interesting follow-up paper after the main work lands.

---

### 🟡 PINK — Second priority: Per-subject LoRA

**Why second:** supervised fallback with best impact-to-complexity ratio. SuLoRA (Klein et al., arxiv 2510.08059) introduced subject-specific LoRA on brain signals but did NOT apply it to LaBraM/CBraMod — that combination is genuinely unfilled.

**Concrete design:**
- Frozen CBraMod backbone (or LaBraM if 16-channel handling works)
- Rank-r LoRA on input projection layer initially; ablate placement (input only, middle, last few, distributed)
- Per-subject adapter fit during K-min supervised calibration phase
- Adapter rank as primary hyperparameter; sweep r ∈ {1, 2, 4, 8, 16}

**Research deliverables:**
- Rank-vs-accuracy curve at multiple K-min budgets
- Layer-placement ablation
- Comparison to task-level LoRA (EEG-FM-Bench setting) and full fine-tuning
- Comparison to SuLoRA on small models (their setting)

**Open question:** can the LoRA adapter be fit *unsupervised* online via reconstruction loss from the FM's pretraining objective? If yes, collapses into the RED branch as a learned addition to Riemannian alignment.

**Risks:** rank-1 might be enough (LoRA overkill); rank-16 might be needed (FM is the wrong abstraction). Both findings publishable.

---

### 🟡 YELLOW — Third priority: Convex NN on frozen FM embeddings

**Why high priority despite yellow color:** this is the method most distinctively yours. Convex formulations of two-layer NNs (Pilanci & Ergen 2020, lots of follow-up) give globally-optimal training in polynomial time. Per-subject fit becomes a small convex program. Properties:
- Provably converges to global optimum
- No local minima, no hyperparameter pain
- Closed-form or fast solver-based, not iterative SGD
- Easy to fit per-subject from a small amount of supervised data
- Trivially fast on CPU

**Concrete design:**
- Frozen CBraMod → embeddings (e.g., 200-dim per patch)
- Per-subject convex two-layer NN with ReLU, trained on K-min supervised data
- Decoder targets: continuous force / position
- Compare to linear probe (the convex special case with no hidden layer)

**Why it should beat linear probe:** convex two-layer NN can represent piecewise-linear functions; linear probe cannot. If frozen FM embeddings need any nonlinear reshaping per subject, the convex NN captures it while linear probe doesn't.

**Why it should beat full fine-tuning:** much smaller compute, no catastrophic forgetting, globally optimal.

**Research deliverables:**
- Convex NN vs linear probe vs LoRA at matched K-min budgets
- Convex NN training-time benchmark (target: <1s on CPU for 32 hidden units, 10K samples)
- Bound analysis from convex NN literature applied to this setting

**Risks:** convex NNs scale poorly in width × samples. For very small calibration budgets (K = 1 min) this is fine; for larger budgets it may be cost-prohibitive on CPU. Mitigation: sketching or sampling tricks from convex NN literature.

**Strategic value:** if this works as well as LoRA at a fraction of complexity, it's a quietly important result with a clean theoretical story.

---

### 🟡 YELLOW — Fourth priority: Continued pre-training

**Why fourth:** expensive, slow. Parallel "what if we had more compute" arm. Domain-adaptive pretraining on WAY-EEG-GAL recordings before downstream fine-tuning.

**Worth doing if:** frozen FM embeddings are obviously the bottleneck (revealed by RED + PINK + first YELLOW giving ceiling behavior).

**Skip if:** time pressure or compute pressure.

---

### 🟡 YELLOW (demoted) — MAML

**Why demoted:** WAY-EEG-GAL has 12 subjects. MAML with so few meta-training subjects is unstable and rarely outperforms simpler alternatives. Compute-intensive. Hard to characterize cleanly.

**Alternative framing:** Reptile (first-order MAML) instead of true MAML — much cheaper, often competitive, fewer pathologies with small meta-training sets.

**Recommendation:** drop unless other methods plateau and explicit meta-learning becomes necessary. Even then, Reptile > MAML.

---

## 9. Demo Day Strategy

16-electrode cap at demo, models trained on 32-channel WAY-EEG-GAL data. Three options:

1. **Channel-agnostic backbone (REVE).** Handles arbitrary configurations natively. Cleanest but requires switching from CBraMod.
2. **Random channel dropout during pretraining.** Train CBraMod with random subsets of channels masked → model handles 16-channel input zero-shot. Cheapest if it works.
3. **Train two models.** 32-channel for offline eval, 16-channel for demo. Avoids the question but doubles compute.

**Decision needed early.** Probably option 2 unless channel-agnostic backbone is easy to swap in.

**Additional demo consideration:** the 16 channels should be a *standardized clinical subset* (e.g., 10-20 system motor area channels) so the demo is reproducible and the trained-on-32 → deployed-on-16 result is a fair comparison.

---

## 10. Strategic Sequencing

**Week 1-2: Infrastructure + RED branch baseline.**
- WAY-EEG-GAL data pipeline, force-regression evaluation harness
- Implement static + online Riemannian alignment variants
- Latency benchmarks on CPU at 16 and 32 channels
- Decoding-accuracy-vs-update-rate curves
- Sanity-check against Kaggle baselines

**Week 3-4: PINK branch (per-subject LoRA on CBraMod).**
- Frozen CBraMod inference path
- LoRA adapter, supervised K-min calibration
- Rank + placement ablation
- Result: head-to-head with RED branch at multiple K budgets

**Week 5-6: First YELLOW branch (convex NN on embeddings).**
- Convex two-layer NN on CBraMod embeddings
- K-min calibration sweep
- Comparison with PINK and RED

**Week 7: Combination experiments.**
- RED + PINK (Riemannian-aligned input + LoRA adapter)
- RED + convex NN
- Best combination becomes the headline system

**Week 8: Demo prep, 16-channel evaluation, drift experiments.**

**Slack: continued pre-training and AlphaEvolve as parallel/stretch.**

---

## 11. Open Questions / Decisions

- [ ] Force regression vs event detection as primary metric (recommend: force)
- [ ] CBraMod vs LaBraM vs REVE as primary backbone (recommend: CBraMod, REVE if channel handling is easy)
- [ ] Verify LaBraM channel embedding subsetting for 32 → 16
- [ ] Channel-handling strategy for 16/32 demo (recommend: random channel dropout in pretraining)
- [ ] Online RA update rule: sliding window vs EMA vs Kalman vs recursive Karcher (benchmark all four)
- [ ] LoRA placement: input only vs distributed
- [ ] LoRA online unsupervised updates: feasible via FM reconstruction loss?
- [ ] Drift detector for gating online updates (probably reconstruction loss change-point)
- [ ] Convex NN architecture: width, regularization, sketching for scale
- [ ] Within-day drift evaluation protocol on WAY-EEG-GAL (recordings long enough?)
- [ ] Reference CPU for latency benchmark (M-series Mac? Intel reference?)

---

## 12. Key References (curated)

**EEG-FM landscape:**
- Liu et al., *EEG Foundation Models: Progresses, Benchmarking, and Open Problems* (arxiv 2601.17883)
- EEG-FM-Bench (arxiv 2508.17742)
- NeuroAdapt-Bench (arxiv 2604.16926)
- LAtte (arxiv 2603.10881) — flags CBraMod / LaBraM patch-length issue on small datasets

**Adapter / TTA / Calibration:**
- SuLoRA / Klein et al. (arxiv 2510.08059) — per-subject LoRA on small models
- NeuroTTT (arxiv 2509.26301) — TTA on LaBraM / CBraMod, BN-only
- EDAPT (arxiv 2508.10474) — supervised continual fine-tuning
- Calibration-free OTTA (Wimpff 2023, arxiv 2311.18520)

**Backbones:**
- CBraMod (arxiv 2412.07236)
- LaBraM (Jiang 2024 ICLR spotlight)
- REVE (channel-agnostic)

**Riemannian methods:**
- Barachant et al. — classical Riemannian BCI
- Kobler et al. — Riemannian batch norm
- Yair et al. — parallel transport on SPD

**Convex NNs:**
- Pilanci & Ergen 2020 — convex two-layer NN training
- Follow-up theory

**Dataset:**
- Luciw et al. 2014 — WAY-EEG-GAL

---

## 13. Notes / Scratch

(continue here)