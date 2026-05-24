# Continuous Hand-Kinematic Decoding from Scalp EEG: A Literature Review for a CS229 / NeurIPS-ICLR-ICML Project

*Compiled May 2026. Five parallel literature passes were run covering (1) datasets, (2) architectures, (3) cross-subject transfer & calibration, (4) recent top-venue ML papers, and (5) neurophysiology + open problems + skeptical/negative results.*

---

## 0. Executive Summary

**Problem.** Decode continuous hand kinematics (3D position/velocity, optionally finger flexion or grasp force) from non-invasive scalp EEG. Then ask: how little per-user calibration data is needed, and which adaptation method is most efficient, to bring a new user close to within-subject performance?

**State of the field, honestly.**

1. The *signal* is real but small. Bradberry-style mLR decoders reliably hit Pearson r ≈ 0.2–0.4 per kinematic axis within-subject; the best online closed-loop number (Mondini/Müller-Putz 2020) is r ≈ 0.32. Anything reported above r ≈ 0.7 without explicit surrogate/shuffle controls and topographic plausibility checks is a yellow flag — likely contaminated by EMG, EOG, or motion artifact.
2. Cross-subject decoding is meaningfully worse than within-subject (typically 30–60% relative drop in r), and there are no clean, published "calibration minutes vs. performance" dose-response curves for continuous EEG kinematics. **This is a gap.**
3. EEG foundation models (LaBraM, ICLR 2024; CBraMod, ICLR 2025; EEGPT, NeurIPS 2024; REVE, NeurIPS 2025; NeuroLM, ICLR 2025) have been evaluated almost exclusively on **classification** tasks. Continuous-behavior regression for motor BCIs is essentially absent from their evaluation suites. **This is the second gap.**
4. Industrial precedents the field cares about right now: **Meta sEMG** (Nature 2025 / NeurIPS 2024 D&B) showed cross-person sEMG decoding with *zero* per-user calibration, and **MindEye2** (ICML 2024) demonstrated fMRI-to-image with ~1 hour of new-subject data via a shared backbone. The "shared backbone + minutes-of-calibration" framing is the winning narrative in 2025–2026.

**The strongest contribution a CS229 project can plausibly make** is to characterize, with proper methodological rigor, how continuous hand-kinematic decoding accuracy scales with (a) the number of pretraining subjects, (b) per-user calibration minutes, and (c) the choice of adaptation method, using publicly released EEG foundation-model backbones (EEGPT / CBraMod) plus classical baselines, across 3+ public continuous-kinematic datasets. The headline figure is "minutes of calibration vs. Pearson r" curves with within-subject and zero-shot horizontal asymptotes.

---

## 1. Why this problem matters (motivation)

- ~5 million Americans live with paralysis; ALS, SCI, stroke, and locked-in syndromes leave intact motor cortex with no peripheral output. Restoring continuous, natural limb control is the long-standing goal of BCIs (Wolpaw & Birbaumer 2002; Hochberg et al. 2012 *Nature*; BrainGate trials).
- Implantable systems (Neuralink, Synchron, Blackrock, BrainGate) achieve clinically meaningful continuous control but are limited to ~80 patients worldwide cumulatively, with surgical risk and ~5–10 year electrode lifespans (Dohle et al. 2025 *Advanced Science*).
- Non-invasive scalp EEG is the only realistic path to mass-scale BCI deployment — inexpensive, no surgery, wearable form factors emerging — but suffers from low SNR, non-stationarity, and large inter-subject variability.
- The *practical* deployment barrier is calibration: users will not tolerate 30+ minutes of supervised setup each session. Reducing this cost is what would actually move scalp EEG BCIs from lab demos to deployable assistive technology.

This motivation frame is **defensible and well-aligned with how reviewers think in 2026** because it mirrors the framings that worked in MindEye2 and Meta sEMG.

---

## 2. Public datasets for continuous EEG kinematics

(See agent report 1 for exhaustive coverage; this is the short list.)

| Dataset | Subjects × sessions | EEG | Kinematic ground truth | Why it matters |
|---|---|---|---|---|
| **Müller-Putz "Feel Your Reach" line** — Mondini 2020 *JNE*; Pulferer 2022 *JNE*; Pulferer 2024 *JNE* | 10 × 1–3 (multiple releases) | 60–64 ch, 200 Hz, g.tec | Measured 2D end-effector position via robotic arm / pen tablet; some online closed-loop | The **only** public corpus with measured continuous 2D trajectories and online benchmarks; supports cross-session calibration. Available via BNCI Horizon 2020. |
| **Jeong et al. 2020 Korea Univ. multimodal** *GigaScience* | 25 × 3 | 60 ch + 7 EMG + 4 EOG, 2.5 kHz | Cued 6-direction reaching, 3 grasps, wrist twists (cue-derived, not mocap) | 25 subjects with 3 sessions one week apart — the strongest dataset for **cross-subject pretraining + cross-session calibration**. Open via GigaDB. |
| **WAY-EEG-GAL** (Luciw et al. 2014 *Sci Data*) | 12 × ~10 series, 3936 trials | 32 ch, 500 Hz, BrainAmp | Polhemus 3D hand position + 6-axis force + EMG | The only public dataset with **measured 3D hand position and grasp force**. CC0 on Figshare. |
| **Schwarz/Ofner 2017 reach-and-grasp** *JNE* | 15 × 1 (+ 45 in 2020 dry-EEG follow-up) | 32–64 ch, 256 Hz | Wrist accelerometer + EMG (coarse) | Good for reach-and-grasp classification; coarse kinematics. |
| **Crell & Müller-Putz 2024 handwriting EEG** | 20 × 1 | 60 EEG + 4 EOG | Pen-tablet 2D trajectory | 2D pen kinematics — good supplementary 2D regression target. |

**Datasets to flag and skip** for continuous kinematics work: BCI Competition IV-2a/2b, PhysioNet EEGMMI, OpenBMI/Lee2019, Cho2017 (all motor-imagery classification, no kinematics); BCI Competition IV-4 (ECoG, not EEG); Forenzo 2024 continuous-pursuit (cursor is the kinematic signal, not the hand — MI-driven, not movement-driven).

**Recommended dataset triple for the project:** (i) **Jeong 2020** as the primary multi-subject multi-session corpus for pretraining and calibration-curve experiments; (ii) **Mondini/Pulferer Graz** for true measured 2D end-effector trajectories with online benchmarks; (iii) **WAY-EEG-GAL** for measured 3D hand position + grasp force as a third independent test bed.

Optional pretraining pool to scale the population: combine Jeong + Stieger 2021 (62 subjects × 7–11 sessions, MI-cursor) + Schwarz 2017/2020 + PhysioNet EEGMMI for ~100 unique subjects of motor-task EEG.

---

## 3. Architectures: a baseline ladder

(See agent report 2 for full detail.)

Recommended four-rung ladder, from weakest to strongest, each more expressive than the last. Stop wherever the curve flattens — but report all rungs.

1. **Linear baselines** with proper controls.
   - Bradberry-style mLR on 0.1–3 Hz potentials (PTS).
   - Korik-style mLR on band-power features (mu, beta, low-gamma BTS).
   - Ridge regression on Riemannian tangent-space covariance features (Yger et al. 2017).
   - *Every one must include a permuted/shuffled-target chance baseline.*
2. **Compact CNN regressors.** EEGNet (Lawhern 2018) and ShallowConvNet (Schirrmeister 2017) with regression heads. ~3k–40k params. Canonical baselines every reviewer expects.
3. **Mid-scale kinematic-specialized models.** **Borra ICNN** (*Comput. Biol. Med.* 2023) — the most directly relevant published baseline for continuous hand-kinematic regression. **EEG Conformer** (Song et al. 2022) with regression head. **Pancholi CNN-LSTM** on WAY-EEG-GAL. ~100k–800k params.
4. **Foundation-model fine-tune.** **EEGPT** (NeurIPS 2024, 10M params, single-GPU friendly, designed for linear probing) or **LaBraM-Base** / **CBraMod** with a regression head. Compare full fine-tune vs. linear probe vs. LoRA adapter vs. parameter-efficient (EEG-GraphAdapter) variants.

Optional fifth rung for a novelty / "frontier" angle: **EEGMamba** or **MI-Mamba** — state-space models for long causal windows. Defensible if you want a methodological-novelty story rather than a scaling-laws story.

---

## 4. Cross-subject transfer and calibration efficiency: methods

(See agent report 3.)

Four adaptation methods to compare in the headline calibration-vs-performance figure:

1. **Zero-shot LOSO** — train on N−1 subjects, test on held-out subject, no adaptation. The floor.
2. **Euclidean Alignment + LOSO** (He & Wu 2020) — unsupervised target-side whitening using only resting/unlabeled target trials. The **zero-labeled-calibration** ceiling. Costs almost nothing.
3. **Fine-tuning on K minutes of target data**, varied K ∈ {0, 1, 2, 5, 10, 15, 30 min, full}. Pretrained backbone + small regression head. Optionally combined with AdaBN. This is the workhorse experiment.
4. **Meta-learned initialization (Reptile/MAML) + K-min fine-tune** — tests whether meta-pretraining produces a better starting point per calibration minute than naïve pooling. Berdyshev et al. 2024 EEG-Reptile is the right reference implementation.

**Auxiliary methods worth one ablation column each:** Riemannian recentering (Zanini 2018), Riemannian Procrustes Analysis (Rodrigues 2019), DANN-style adversarial subject-invariant features (Özdenizci 2020), GOPSA (NeurIPS 2024 — explicitly designed for cross-site EEG **regression** under shift, and underused).

**Cutting-edge angle (optional):** add a **test-time adaptation** column — NeuroTTT (2025) or a simple entropy-min TTA on the regression head — to test whether unsupervised online adaptation closes the calibration gap further.

---

## 5. Recent top-venue ML context

(See agent report 4.)

The 5 papers to cite as direct anchors:

1. **LaBraM** (Jiang et al., **ICLR 2024 spotlight**, arXiv:2405.18765) — establishes EEG foundation-model pretraining works; we extend to continuous regression.
2. **CBraMod** (Wang et al., **ICLR 2025**) or **REVE** (NeurIPS 2025, 25k subjects / 60k hours) — current EEG-FM SOTA; cite as a backbone candidate and as evidence that the heterogeneous-channel problem is largely solved.
3. **POYO** (Azabou et al., **NeurIPS 2023**) and **NDT2** (Ye et al., **NeurIPS 2023**) — the "neural population foundation model" template. We are doing the non-invasive analog.
4. **MindEye2** (Scotti et al., **ICML 2024**) — the structural analog: shared backbone, ~1 hour of target-subject data, continuous behavior reconstruction. Best precedent for our headline framing.
5. **Meta sEMG** (CTRL-Labs / Reality Labs, *Nature* 2025; NeurIPS 2024 D&B datasets) — the industrial proof-point that cross-person continuous decoding without calibration is taken seriously by the field. Different modality (sEMG), same framing as ours.

Honorable mention: **GOPSA** (NeurIPS 2024) — the only NeurIPS EEG paper with explicit cross-site **regression** plus domain adaptation. Must-cite baseline if our story is "regression + cross-subject."

The NeurIPS 2025 EEG Foundation Challenge (eeg2025.github.io) is also worth tracking — current best public benchmarks for cross-subject EEG decoding live there.

---

## 6. Open problems, artifact pitfalls, and what makes a paper publishable

(See agent report 5 — read this in full; it is the most important pass.)

**Artifact-confound problem (the existential threat to this kind of paper).** Antelis et al. 2013 *PLOS ONE* is the key critical paper. They argue that linear regression from low-frequency EEG to low-frequency kinematics can produce non-trivial Pearson r through OLS leakage, with no underlying motor information. Their permuted-target controls show several published r values are not statistically distinguishable from chance.

Required controls in any rigorous EEG kinematic decoding paper:
- **Shuffled / circularly shifted target** chance baseline at every reported number.
- **Observed-movement** condition (subject watches the same trajectory; same EOG and saccades, no motor command).
- **Topographic plausibility** — decoder weights / saliency should localize over contralateral M1/S1, not over temporal/jaw (EMG) or frontal/eye (EOG) electrodes.
- **ICA-based EOG/EMG removal**, with reported performance pre- and post-ICA.
- **Causal-filter ablation** — non-causal Butterworth filters leak future kinematics; report performance with causal filters only, and with vs. without the last 200 ms of "future" EEG.

**Offline-vs-online gap.** Mondini 2020 reports offline r ≈ 0.45 dropping to online r ≈ 0.32 on the same subjects — a 25–30% relative drop. Reviewers in 2026 will ask about this. Either evaluate online (hard with public datasets) or carefully separate causal and non-causal numbers.

**Subject heterogeneity / BCI illiteracy.** Vidaurre & Blankertz 2010: ~15–30% of users can't drive a typical MI BCI. For executed-movement kinematics the failure rate is lower but not zero. **Report per-subject results, not just averages, and report the fraction of users who clear an operational threshold (e.g., r > 0.2) at each calibration budget.** This is what reviewers want for a calibration-efficiency story.

**What clears the NeurIPS/ICLR/ICML bar in 2026:**
- ✗ "New CNN/Transformer, +3% on BCI-IV-2a." This is JNE/Frontiers material.
- ✗ Single-dataset evaluation. Reviewers expect 3+ public datasets.
- ✓ Foundation-model adaptation method evaluated on multiple public datasets with proper per-subject reporting, ablations, and artifact controls.
- ✓ Calibration-efficiency / scaling-laws characterization that doesn't currently exist for continuous EEG regression.
- ✓ Methodologically rigorous re-analysis demonstrating *how much* of published EEG kinematic decoding is artifact-driven.

**Honest probability assessment:** Direct main-track accept at NeurIPS/ICLR/ICML from a one-student, ~3-month, public-data-only project is single-digit percent. The much more achievable and credible target is **a strong workshop paper** at NeurIPS Foundation Models for Brain/Body, NeurIPS AI4Science, ICLR Time Series for Health, or the NeurIPS 2025/2026 EEG Foundation Challenge, with a polished follow-up submission to a main-track cycle.

---

## 7. Recommended thesis and contribution

**Thesis.** Continuous hand-kinematic decoding from scalp EEG is bottlenecked not by raw decoder capacity but by *per-user calibration cost*. Foundation-model pretraining on cross-subject EEG corpora, combined with calibration-efficient adaptation (Euclidean alignment / meta-initialization / low-rank fine-tune), can drive new-user calibration from the field-standard 20–30 minutes down to the single-digit-minute regime while preserving most of the within-subject performance — but only when evaluated under strict artifact controls and causal-filter constraints.

**Headline contribution.** First systematic characterization of the **minutes-of-calibration vs. continuous-regression-accuracy** Pareto frontier for non-invasive EEG hand-kinematic decoding, across multiple public datasets and adaptation methods, evaluated with the artifact and offline/online controls the field has historically neglected.

**Subsidiary contributions (one each, depending on time):**
- A practical recipe for adapting EEG foundation models (EEGPT / CBraMod) to continuous regression — currently undocumented in the FM-for-EEG literature.
- An honest analysis of how much of measured "kinematic decoding" is signal vs. artifact across published datasets, with a standardized evaluation protocol the community can adopt.
- A LoRA-style **subject-specific adapter** scheme that scales sublinearly in per-subject parameter cost while approaching full-FT performance.

---

## 8. Concrete experimental protocol

**Datasets:** Jeong 2020 (primary, 25 subj × 3 sessions); Mondini/Pulferer Graz (secondary, 2D trajectories with online numbers); WAY-EEG-GAL (tertiary, 3D hand position + force). Optionally pretrain on a pooled corpus including Stieger 2021, Schwarz 2017/2020, PhysioNet EEGMMI.

**Decoders (rows of every table):**
1. Bradberry mLR + Korik BTS-mLR (linear)
2. EEGNet-regression and ShallowConvNet-regression (compact CNN)
3. Borra ICNN and EEG Conformer-regression (mid-scale)
4. EEGPT or CBraMod backbone with: (a) linear probe, (b) LoRA adapter, (c) full fine-tune

**Adaptation methods (columns):**
- Zero-shot LOSO
- Euclidean Alignment (zero labeled-calibration)
- Riemannian recentering / RPA
- AdaBN
- K-minute supervised fine-tune (the workhorse)
- Reptile-meta + K-min fine-tune
- GOPSA (NeurIPS 2024 cross-site Riemannian DA)
- (Stretch) NeuroTTT-style test-time adaptation

**Calibration sweep:** K ∈ {0, 1, 2, 5, 10, 15, 30 min, full-session} held-out target subject. Per-subject minutes are converted to trial counts based on dataset trial length. At least 12 held-out subjects per dataset, repeated LOSO.

**Metrics:**
- Pearson r per axis and averaged (the field-standard metric, kept for comparability).
- R² / fraction of variance explained.
- RMSE in physical units (cm, cm/s).
- **Shuffled-target permutation null** at every reported r — report p-value or % above null.
- Fraction of held-out subjects clearing r > 0.2 (operational threshold).
- (Mondini-style) closed-loop or simulated closed-loop time-to-target where possible.

**Controls and ablations:**
- Causal vs. non-causal filter comparison.
- Pre- and post-ICA artifact removal comparison.
- Decoder-weight topography (saliency map per subject; should localize over contralateral M1/S1).
- Observed-movement or motor-imagery control where the dataset provides it.

**Headline figure:** x-axis = calibration minutes (log-spaced), y-axis = Pearson r (with shuffled-target null shaded), one curve per adaptation method, horizontal asymptote at within-subject upper bound. Bootstrap 95% CI shading. Each dataset gets its own panel. The one-sentence claim the figure tells: *"Method M reaches 90% of within-subject performance with only K minutes of calibration."*

---

## 9. Risks and contingencies

1. **No new-subject gain over EA baseline.** If Euclidean Alignment recovers everything for free, the "minutes" story collapses to "you don't need any." This is still publishable — it's a useful negative result, frames the field, and shifts the contribution toward the foundation-model regression-evaluation angle (subsidiary contribution 1).
2. **Foundation-model backbone underperforms classical baselines on regression.** Highly plausible — the FMs were not pretrained for regression. This is also publishable as a critical evaluation paper (subsidiary contribution 2). Reviewers reward honest negative results when the analysis is airtight.
3. **Artifact confound dominates.** Saliency maps and shuffle tests reveal that most decoding signal is EMG/EOG. This is the most valuable possible negative result — re-frame as a methodological-rigor paper exposing the field's evaluation problem.
4. **Compute constraints.** EEGPT (10M params) fits on one consumer GPU. LaBraM-Base and CBraMod are comparable. REVE and NeuroLM-XL are out of reach — fine. Pretraining can be skipped entirely by using released checkpoints.
5. **Public dataset access friction.** BNCI Horizon 2020 hosts most Graz data; GigaDB hosts Jeong; Figshare hosts WAY-EEG-GAL. All CC-BY or CC0. Download budget ≈ 100–300 GB.

---

## 10. Canonical reading list (in priority order for week 1)

1. Bradberry, Gentili & Contreras-Vidal 2010 *J. Neurosci.* — the founding paper.
2. **Antelis et al. 2013 *PLOS ONE*** — the critique you must internalize.
3. Mondini, Kobler, Sburlea & Müller-Putz 2020 *J. Neural Eng.* — the cleanest online closed-loop number.
4. Borra et al. 2023 *Comput. Biol. Med.* — the most directly relevant deep-learning baseline.
5. He & Wu 2020 *IEEE TBME* — Euclidean Alignment, the trick everyone uses.
6. Jiang et al., LaBraM, **ICLR 2024** — foundation-model EEG.
7. Wang et al., EEGPT, **NeurIPS 2024** — best feasible backbone for this project.
8. Wang et al., CBraMod, **ICLR 2025** — current SOTA EEG FM with channel flexibility.
9. Scotti et al., MindEye2, **ICML 2024** — structural analog for the framing.
10. Sivakumar et al. (Meta), sEMG, *Nature* 2025 / NeurIPS 2024 D&B — industrial precedent.
11. Vidaurre & Blankertz 2010 *Brain Topogr.* — BCI illiteracy.
12. Schirrmeister et al. 2017 *HBM* (DeepConvNet/ShallowConvNet) and Lawhern et al. 2018 *J. Neural Eng.* (EEGNet) — baseline architectures.

---

## Appendix: agent reports

The five raw research reports underlying this document are preserved in the conversation transcript:
1. Datasets survey — ~2400 words covering 12+ public datasets.
2. Architectures survey — ~2300 words, full baseline ladder with parameter counts.
3. Cross-subject transfer & calibration — ~2300 words, 9 sections.
4. Top-venue ML survey 2022–2026 — ~2700 words, 25 papers in 6 clusters.
5. Open problems / neurophysiology / publishability — ~2900 words, the skeptical pass.

Re-read agent 5 (skeptical pass) before drafting motivation; re-read agent 3 (calibration) before drafting methods.
