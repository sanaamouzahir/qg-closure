# [QG][BRIEF][SGS-CLOSURE] ML SPEC 01 — Π_FF closure model: build + test instructions

Global supervisor: commit this verbatim as `docs/briefs/ML_SPEC_01_piff_closure.md`. This amends the §8 authorship matrix for this branch only: the ML training code specified here is authored by YOU (Fable 5), exactly to this spec. Design authority remains with Sanaa; any deviation, substitution, or "improvement" requires her approval via a B-item in the morning digest before implementation. Where this spec is silent, ask; do not improvise.

## 0. Scope and prerequisites

Offline (a priori) training and evaluation of the spatial SGS closure: a FiLM-conditioned CNN feature extractor with a sparse variational GP (SVGP) head predicting Π_FF from filtered fields. Cylinder geometry only. Online/a posteriori deployment is OUT of scope. Prerequisite data: at least the FPC-const production run with its `DNS_LES_s4.npz` (scale 4 is the primary scale) and `DATASET_MANIFEST.md`. Do not block on the full ensemble; the pipeline is developed and smoke-tested on FPC-const, then retrained on the ensemble as runs land.

Precision policy for THIS code: float32 throughout (data is stored f32; targets are O(1)). This differs from the temporal-closure branch and is intentional.

## 1. Module 1 — `dataset_piff.py` (loader)

1. Reads paths/shapes/conventions from each run's `DATASET_MANIFEST.md`; hard-fails with a clear message on any mismatch. Never hard-code shapes.
2. Sample = random crop of 64x64 coarse-grid points from a snapshot in the usable window (t >= 30). Channels, in order:
   - ω̄* = ω̄ · D/U(tₙ)
   - ū* = ū/U(tₙ), v̄* = v̄/U(tₙ)
   - SDF: signed distance to the obstacle boundary, clipped to ±2D, divided by 2D. Computed once from the mask, cached.
   U(tₙ) comes from the run's `U_of_t.npz` at the snapshot's step index — never from per-snapshot field statistics. NO per-sample standardization anywhere.
3. Target: Π* = Π_FF · D²/U(tₙ)², same crop.
4. Conditioning scalar per sample: ζ = (Re(tₙ) − 3900)/1700.
5. Loss/eval mask per crop: exclude sponge-region pixels and pixels with SDF < 0 (inside body). Masking happens in the loss, not by zeroing inputs.
6. Wake-biased sampling: 80% of crops centered in x ∈ [x_c − 1D, x_c + 12D], |y − y_c| ≤ 3D; 20% uniform over the domain. Both ratios are config values.
7. Split by TIME, not by random shuffle: train t ∈ [30, 100], validation t ∈ [100, 120] within each run (temporal-decorrelation argument in `Supervisor_simulation.md` §4). When multiple runs exist, additionally support leave-one-modulation-out splits via config.
8. Deterministic given a seed; seed logged in every artifact.

## 2. Module 2 — `model_piff.py`

### 2.1 CNN feature extractor
- 3 conv layers, kernel 5x5, periodic padding in both directions (domain is periodic; the mask/SDF channel carries the obstacle), channels 4 → 32 → 32 → F with F = 16 feature dims. RF = 13 coarse points, parameter count O(3·10⁴). Justification: measured Π–ω̄ locality ~7 points (Srinivasan et al. 2024); do NOT deepen without a B-item.
- Activation: swish.
- FiLM conditioning: after EVERY conv block, h_c ← γ_c(ζ)·h_c + β_c(ζ). One shared MLP: ζ → 16 (swish) → 2·(32+32+16). Identity init: output layer zero-initialized, γ = 1 + δγ. Config flag to freeze FiLM (γ≡1, β≡0) for the Re-blind ablation.

### 2.2 SVGP head (pointwise; this is the Phase-1 design decision)
- Each output pixel is one GP evaluation: input = the F-dim CNN feature vector at that pixel, PLUS ζ appended as one extra kernel input dimension (F+1 total).
- Kernel: RBF with ARD over all F+1 dims. Report the learned ARD lengthscale of the ζ dimension in every eval summary — it measures regime dependence not absorbed by FiLM.
- M = 512 inducing points in the (F+1)-dim feature space, initialized by k-means on features of 10k random training pixels from the initial (untrained-FiLM) network.
- Likelihood: Gaussian, learned homoscedastic noise. Before the first full training run, produce the Π* residual PDF (target minus CNN-mean after warmup epoch) — if excess kurtosis > 5, raise a B-item proposing a heteroscedastic or Student-t likelihood; do not switch silently.
- Whitened variational parameterization; natural-gradient updates for the variational distribution are optional, standard Adam is acceptable for Phase 1.
- Implementation: GPyTorch (pip-install into the branch venv clone — never the shared venv). If GPyTorch is unavailable on the cluster nodes, raise a FLAG; do not hand-roll SVGP.

## 3. Module 3 — `train_piff.py`

1. Objective: SVGP ELBO, minibatched over crops; each crop contributes its masked pixels as GP data points; ELBO data-count = number of masked pixels in the epoch (document the exact scaling in the code).
2. Optimizer: Adam. LR schedule: cosine annealing with warm restarts, T₀ = 5 epochs. Hyperparameter grid: lr ∈ {1e-4, 3e-4, 1e-3} × weight decay ∈ {1e-5, 1e-4} on CNN weights only (never on GP hypers). 6 runs; one GPU job each, `-q ibgpu.q -l gpu=1`.
3. Model selection: lowest validation NLL (not MSE) over the whole schedule; checkpoint every epoch, keep best + last.
4. PLAN B (pre-authorized, use only if joint training diverges or feature-collapses — symptom: validation NLL improving while validation RMSE worsens AND feature-space pairwise distances shrinking > 10x): two-stage. Stage 1: CNN + linear head, MSE loss, same schedule. Stage 2: freeze CNN, fit SVGP on frozen features. Report which path was taken, always.
5. Every run logs per epoch: train/val ELBO, NLL, RMSE, R², mean predictive σ, FiLM γ/β norms, ζ ARD lengthscale. Machine-readable (npz/yaml) + PNG curves.

## 4. Module 4 — `eval_piff.py` (a priori evaluation; run on best checkpoint)

1. Pointwise metrics on validation: R², RMSE, NLL — global and binned by ζ decile.
2. Calibration: reliability diagram (empirical coverage of ±1σ, ±2σ, ±3σ intervals vs nominal) and spread–skill plot (binned predictive σ vs empirical |error|), global and per ζ-bin.
3. Field visualizations: for 6 validation snapshots spanning the Re range: truth Π*, predictive mean, predictive σ, |error| — 4-panel figures, plus the Re_inlet(t) trace marking the snapshot times.
4. Scale check (once s ∈ {2,8} datasets exist): retrain per scale (config change only) and plot mean predictive σ vs filter scale Δ — the stochastic-closure framing predicts σ grows with Δ; report whether it does.
5. OOD probe (once ≥ 2 modulation runs exist): train on one modulation class, evaluate on another; the required qualitative result for the paper is that predictive σ INCREASES on the held-out class. Report the σ ratio.

## 5. Tests (gate: all pass before any grid search is submitted)

T1. Loader regression: fixed seed → identical batch tensors across two runs (hash check).
T2. Normalization identity: feeding the same physical field at two different U(t) values yields identical normalized channels up to float tolerance.
T3. FiLM identity init: at init, output with FiLM on vs frozen differs by exactly 0.
T4. Periodic padding: translating the input crop cyclically translates the CNN output cyclically (equivariance check, tolerance 1e-5).
T5. SVGP sanity: on a synthetic 1D-feature problem with known noise, recovered noise within 20% and coverage of ±2σ in [90, 99]%.
T6. Overfit smoke: 500 crops from FPC-const, model reaches R² > 0.95 train within 50 epochs (capacity check).
T7. One full pipeline smoke on GPU: 2 epochs end-to-end on FPC-const s=4, produces all logs/plots, < 30 min walltime.
All tests run as batch/qlogin jobs — never on the frontend.

## 6. Choreography

- CP-ML-1: after reading this spec, reply with your implementation plan (file layout, GPyTorch version, ELBO scaling formula you will use, test schedule). WAIT for Sanaa's approval.
- Build + run T1–T7. Email `[QG][GATE-ML][SGS-CLOSURE]` with test results and the residual-PDF kurtosis number. WAIT for approval.
- Submit the 6-run grid on FPC-const s=4. `[QG][SUBMIT]` email per convention.
- On completion: `[QG][MILESTONE][SGS-CLOSURE] A priori closure v1` with the §4 evaluation package for the best model, hyperparameter table, and which training path (joint/plan B) was used.
- Retraining on additional ensemble members/scales as they land is pre-authorized with the SAME spec; any spec change is a B-item.
- All standing rules hold: no frontend .py, ibgpu.q only, digest reporting, never touch DNS_FR files.
