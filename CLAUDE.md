# CLAUDE.md — QG-Closure

## What this project is

ML-based closure modeling for coarse-grid 2D quasi-geostrophic (QG) turbulence. Pseudo-spectral GPU solver (PyTorch) with AB2CN2 IMEX time stepping. Two closure tracks:

1. **Temporal closure (δR)** — primary. A cheap physics-structured network (learned discretization, Bar-Sinai/Kochkov lineage — NOT an FNO/DeepONet operator learner) supplies the N-time-derivatives that assemble the AB2CN2↔fine-step closure. Multi-objective: cost (memory + walltime), stability region, order/accuracy. Being generalized to match any generic Linear Multistep Method to its better-resolved self.
2. **Spatial closure (Π)** — CNN+SVGP sub-grid-scale forcing; multi-β generalization via `Pi_spatial × β_NN(features)` factorization with β as explicit input.

Flow scenarios: decaying turbulence (DEC), forced turbulence (FRC, β-plane), flow past cylinder, flow past cape. All domains are ISOTROPIC squares (FRC 4π, cylinder/cape 8π) — the anisotropic (Lx≠Ly) case is guarded against, not supported.

## Repo layout (agent orientation)

```
external/qg-simple/  QG solver — SUBMODULE (fork of akhilsadam/qg-simple, `closure` branch
                     off package-stable). Install: uv pip install -e external/qg-simple.
                     Solver bug reports/fixes go to the fork, not this repo.
solver_patches/      0.2.1-era local solver modifications awaiting port onto the fork
                     (PORTING.md). Not imported. Deleted once ported.
training/            v2 derivative-loss pipeline — ACTIVE, now MULTIGRID (see lineage).
diagnostics/         Debugging probes + RESULTS_*.md session logs. The 2026-07-03 log
                     documents the quiescent-window investigation end to end — read it
                     before re-debugging anything that smells like "huge val error".
analysis/            Convergence / truncation / stability / rollout / error-prop figures.
spatial_closure/     Π_FF pipeline.
scripts/sge/         Cluster submission (submit_X.sh → X_job.sh).
docs/, paper/        Theory notes + audit; manuscript .tex (THEORETICAL_GUARANTEES).
legacy/              Superseded code; legacy/snapshots/ = read-only locks.
```
Subdirectories carry their own CLAUDE.md with local rules — read the one for the dir you work in.

## Canonical math

PDE convention: `∂_t ω̄ = Lω̄ + N`, `N = −J(ψ̄, ω̄) + F` (F time-independent ⇒ Ḟ = F̈ = 0), `ψ̄ = ∇⁻²ω̄`, `L̂(k) = −ν|k|² − μ + iβk_x/|k|²`.

N-derivative chain rule (general): `N^(m) = −Σ_{j=0}^{m} C(m,j) J(ψ^(m−j), ω^(j))` with `ω^(k) = Lω^(k−1) + N^(k−1)`, `ψ^(k) = ∇⁻²ω^(k)`. Explicitly:
```
ω̇̄ = Lω̄ + N;              ψ̇̄ = ∇⁻²ω̇̄
Ṅ  = −J(ψ̇̄, ω̄) − J(ψ̄, ω̇̄);  ω̈̄ = Lω̇̄ + Ṅ;  ψ̈̄ = ∇⁻²ω̈̄
N̈  = −J(ψ̈̄, ω̄) − 2J(ψ̇̄, ω̇̄) − J(ψ̄, ω̈̄)
N⃛  = −J(ψ⃛,ω̄) − 3J(ψ̈̄,ω̇̄) − 3J(ψ̇̄,ω̈̄) − J(ψ̄,ω⃛)
```
Forcing is load-bearing for N̈/N⃛ (enters via ω̇̄) even though Ḟ=0 — always rebuild F exactly from the manifest (FRC: `A cos(Bx) + D cos(Ey)`; DEC: F=0). Training itself needs NO forcing input: targets are m≥1 (F drops) and the snapshots already evolved under F.

δR closure structure:
```
δR = (ΔT²/12)·[L³ω̄ + L²N + L·Ṅ − 5·N̈]
```
The "−5" is AB2's structural blind spot to N̈ — the NN piece is structurally necessary, not optional. Runtime: ~3× base AB2 step + NN forward.

Truncation operators (validated in `analysis/validate_ab2cn2_vs_truth.py`; assembled scheme-specifically in `training/closure_operators.py`):
```
τ = −(h³/12)R₃ − (h⁴/24)R₄ − (h⁵/240)R₅ − (h⁶/1440)R₆ + O(h⁷)
R₃ = L³ω + L²N + LṄ − 5N̈    R₄ = 2L⁴ω + 2L³N + 2L²Ṅ − 4LN̈ + N⃛   (R₅,R₆: see script)
```

**Benign amplification — now verified across the WHOLE ΔT envelope** (`closure_error_propagation.py` + `run_error_prop_3dt.sh`, eps=1 geometry pass, Re25k at ΔT=1.5e-2): the L⁰N̈ term is 100% of ‖δ‖; L²Ṅ ≈ 0.00%, L¹N̈ ≈ 0.2%, L⁰N⃛ ≈ 2.6%. Closure error ≈ ε_N̈ **1:1** at 5e-3 AND 1.5e-2. Consequences: (a) equal-weight per-order rel-L2 loss is the right objective across the envelope; (b) **N̈ val rel-L2 sets the rollout floor** — the number to watch; (c) the amplification-weighted loss variant is unnecessary.

Convergence radius: modified-equation series converges only for ΔT < ΔT★ = C·τ_eddy, C = 2.08±0.15 universal; radii Re25k 0.066 / combo 0.139 / kf4 0.199. Inner wall from the finite-lag time stencil is hit first. **At S=7, dt=1.5e-2 the stencil SPAN (6·dt = 0.09) exceeds Re25k's ΔT★ = 0.066** — those samples are unlearnable by any network (field decorrelates inside the stencil). This is in the current training pool by design (see current run) and explains the pooled-val gap vs the old single-(member,dt) run.

Inference decomposition (fixD/f_NN_target track):
```
f(ω̄) = R⁻¹(ΔT)·ΔT³(1−1/K²)·(1/12)·L³ω̄     [linear, IMPLICIT, folded into IMEX]
      + S⁻¹(ΔT)·ΔT³(1−1/K²)·(1/12)·L²N(ω̄)  [nonlinear via J, EXPLICIT analytical]
      + f_NN(ω̄)                              [NN, EXPLICIT]
S⁻¹(ΔT) = (1 − 0.5·ΔT·L̂)/ΔT
```

## Error theory of the current model (derived; Wiener section = active iPad work)

The model computes `D_θ FD_k ω`. Two error operators: `FD_k ω = ω^(k) + ε_k` (time truncation, ε_k = C_k ΔT^(S−k) ω^(S) + h.o.t., ε_0 = 0, ε^ψ_k = ∇⁻² ε^ω_k because ∇⁻² commutes with the FD stencil) and `D_θ = ∂_x + δ_θ` (learned stencil gap). Per-Jacobian error:
```
ΔJ_ij = J(ε^ψ_i, ω^(j)) + J(ψ^(i), ε^ω_j)   [time part: field errors through EXACT Jacobians]
      + δJ_ij                                 [stencil part: operator error on true fields]
```
Leading ΔN^(m) ~ C_m ΔT^(S−m) [J(∇⁻²ω^(S), ω) + J(ψ, ω^(S))] — all orders driven by the SAME field ω^(S). Training solves a Wiener problem: the stencil learns δ★ ∝ −ik·C_m·(ΔT·σ(k;Re,β,μ))^(S−m) to pre-cancel the time error. **This derives the physics conditioning**: δ★ varies by (ΔT·σ)^(S−m) — factor 3^5 ≈ 243 across the dt sweep for N̈ — so one unconditioned stencil fits only the pooled mean; its loss floor = the pooled variance of δ★. Next model: `δ_θ(k) = ΔT^(S−m) [analytic] × g_θ(k; Re,β,μ) [learned]`. Only the nonlinear-cascade remainder of ε (not expressible as a local filter of ω^(m)) is uncancellable = the true inner wall.
**[BOOKMARK: the full Wiener filter derivation is being worked on iPad — revisit and formalize before the conditioned model is built.]**

## Model & trainer lineage (the versioning story — respect it)

```
fixD (bilinear, predicts f_NN_target)             model_fixD.py + train_v2_annealing.py
 └→ cheap_deriv, derivative-loss, 1 trajectory    legacy/snapshots/Shallow_NN_one_test_case
     │   (S=4, dt=1e-3, grad_kernel 15, lr 5e-5 → val N_dot .026 N̈ .031 N⃛ .043 —
     │    the reference numbers; at dt=1e-3 the time-FD error is invisible, so this
     │    floor is PURELY the spatial gap. NOT comparable to the S=7 sweep runs.)
     └→ ensemble, SINGLE-GRID ("pre-6.1.2")       v2 lock: legacy/snapshots/..._fixedGrid_v2
     └→ MULTIGRID, S=7, dt-sweep                  ← CURRENT ACTIVE, lives in training/
          (per-sample dx,dy rescale; per-shape dealias; order-clip fix; by-window splits)
 parallel: δ-pivot (empirical delta + corrector)  legacy/snapshots/..._fixedGrid_v1, train_delta.py
```
- **Current status (2026-07-03):** the first multigrid S=7 run (`deriv7_equalw_R3R4`, lr 2e-4) was **poisoned by quiescent spin-up windows** (see diagnostics/RESULTS_2026-07-03.md): ~9–33% of each member's windows are quasi-zonal early flow with J(ψ,ω)≈0, so their N-derivative targets are ~1e4–1e5× smaller than developed-flow windows; the per-sample relative loss explodes on them and the optimizer's cheapest move is shrinking all predictions toward zero (destroyed mix, N3dot≈0.99, trained medians WORSE than init). Fix: `filter_quiescent_windows.py` (drops whole windows by target-norm + stack-roughness thresholds; splits backed up to `split_prefilter.npz`), then retrain **from physics init** at lr 5e-5 with ALL members (Re25k's apparent catastrophe was this same artifact — do not cut it). Healthy-window init medians: Ndot≈0.19, Nddot≈0.26, N3dot≈0.33 (= width-15 spatial gap; the [spec] probe puts the pure time-FD floor at 0.0003/0.008/0.10 at 5e-3).
- **Active run to launch/monitor:** `deriv7_filtered_lr5e-5` — 18 roots, filtered splits, S=7, grad_kernel 15, lr 5e-5, equal weights, f64. Expect ep0 val ≈ 0.2–0.4 and val_N3dot < 1 and falling; watch val_Nddot (rollout ceiling).
- **Model (unchanged):** cheap_deriv, ~3,700 params, predicts LOCAL [Ṅ, N̈, N⃛]; L^k weightings (incl. the nonlocal β term) applied **analytically at inference, never learned**. Pipeline: TimeFD → spatial grads → Jacobian features → 1×1 mix (physics-init to chain-rule binomials). Corrector OFF (hidden=0). Shapes {256², 512²}, dT sweep {5e-3, 1e-2, 1.5e-2}.
- **After the filtered retrain:** per-root eval + trained-vs-init medians (diagnostics/), then the physics-conditioned model per the Wiener theory above.
- **δ-pivot track:** target `δ = Φ_ref − Φ_AB2CN2`, references `exact` / `rk4` / `both` (1-bit FiLM); `δ_exact − δ_rk4 = τ_RK4`. Follow-ons per `legacy/snapshots/Shallow_NN_enssemble_fixedGrid_v1/README_SNAPSHOT_v1.txt`.
- The snapshot READMEs are the authoritative lineage docs. Never edit files inside `legacy/snapshots/` — they are locks.

## Locked-in model facts (do not "fix" these)

- **ORDER CLIP (critical, 2026-07 fix):** the model emits ONLY time-orders 0..out_orders. Do not revert. With S=7 and out_orders=3, emitting orders 4–6 (scaled 1/dt⁴⁻⁶ ≈ 1e10–1e13) creates ~1e18-magnitude Jacobian features; physics-init zeros their mix weights but ONE Adam step puts lr-sized weights on them → output explodes ~1e14 (observed). The clip keeps ALL S snapshots in each emitted row (the order-3 row of a 7-node stencil is 4th-order accurate — the point of the deep stencil); only the unused noisy ORDER OUTPUTS are dropped. Features: (out_orders+1)² = 16, not S² = 49.
- **TimeFD is frozen-exact:** across a dT sweep the model uses the per-sample `W_unit/dt^k` path, NOT the learnable `self.weight` — an inert parameter (no gradient, 49 dead params in the count). Do not report it as trained capacity. FD accuracy per output at S=7: Ṅ O(dt⁶), N̈ O(dt⁵), N⃛ O(dt⁴).
- **Spatial stencils ARE learnable** (width-15 = 14th-order at init, FD-init, refine toward spectral). Their job is DOUBLE: approximate ∂x,∂y AND absorb the differentiated time-FD error (the Wiener mechanism above) — which is why they will be physics-conditioned next, and why width was bumped 7→15.
- **Factored SpatialGrad:** the Parameter is the DIMENSIONLESS unit-spacing stencil; 1/dx, 1/dy applied per-sample at forward (`conv(x,S)/dx == conv(x,S/dx)` exactly). One stencil serves all grids; Adam never sees the 1/dx amplification. Bonus: inverting the well-conditioned A(1) beats inverting A(dx) by ~30× on the high-order coefficients.
- **The multigrid trap:** `GridHomogeneousBatchSampler` groups by SHAPE, which does NOT imply equal dx (512²/4π vs 512²/8π). The dx rescale must be **per-sample, never per-batch**. Sampler needs `set_epoch(ep)` called each epoch or it never reshuffles (fixed in trainer; guarded no-op on single-grid path).
- Dealias projections are per-SHAPE, keyed `(Ny,Nx)` — valid ONLY because all domains are isotropic (mask is mode-index based, L-independent). Trainer has a hard guard that refuses Lx≠Ly members.
- Model is exactly quadratic in the input field; β, ν, μ are not model inputs TODAY (N^(m), m≥1 is forcing-free; regime enters via the snapshots). The Wiener theory shows the OPTIMAL stencil is nonetheless (ΔT,Re,β,μ)-dependent through the time-error absorption — hence the conditioning roadmap. Regime vector is `[dT, β, ν, μ, dx, dy]` (6-dim; indices 0/4/5 feed the model as dt/dx/dy).
- No FFTs inside the model: ∂_t commutes with FFT exactly but NOT with products (product rule = the binomial structure); the pointwise product is the sole irreducible nonlinearity (convolution in spectral, O(N⁴) direct vs O(N²logN) via FFT). Time-FD from stored ψ-history replaces ∇⁻² at inference (∇⁻² commutes with time-FD; the builder inverted once, offline) — the design, not an approximation to remove. D⁻¹ is the ONLY spatially-nonlocal letter; with ψ carried, even the β term is local (β∂ₓ∇⁻²ω = −βv).
- **Dealias placement is a non-issue in the model:** everything between product and output is linear, so end-projection == per-product dealiasing exactly (mask commutes with ik and distributes over sums); end-projection is ~100× cheaper. The residual train/solver inconsistency is the Jacobian FORM (model advective, solver/target conservative flux) — small, partly absorbed by training, unfixable without FFTs; a deliberate deferred item for the conditioned/recursion model.
- Storage vs compute precision: training compute is FULL float64; sliced `inputs.npy` is float32 ON DISK (upcast at load). This can floor N⃛ (deepest cancellation) but not Ṅ/N̈; harmless for rollout (N⃛ = 2.6% of closure). Re-slice float64 only if a pristine N⃛ number is needed for the doc.
- Reference config (v2 lock, S=4 single-grid): inputs `omega_0..m3 psi_0..m3`, targets `N_dot/N_ddot/N_3dot_0_anal`, grad_kernel 15, rel_l2 per-sample/per-channel, dealias-pred on, normalize off, bs 4, 200 ep, lr 5e-5, wd 1e-4, cosine, f64, seed 0.

## Environment

- Python ≥ 3.10, `uv`-managed venv. Solver = `quasigeostrophic-flow` (upstream 0.2.3), installed editable from the submodule: `uv pip install -e external/qg-simple` (clone with `--recurse-submodules`). Deps (unpinned): numpy, torch, lightning, einops, hydra-core, omegaconf, wandb, mura ≥ 0.3.0, jpcm, matplotlib, ffmpeg-python, netcdf4, pyyaml, pytest, tqdm, gitpython.
- Cluster: SGE. `QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure`, `QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg`, venv `$QG_ROOT/qg-env/`, training dir `$QG_DIR/training/`.
- Data: per-sample `.npz` (HDF5 abandoned — do not reintroduce) for the fixD track; packed contiguous memmap `.npy` (`inputs.npy`, `deriv_anal_f64.npy`, `delta_*_f64.npy` under `sweep_dT_*/packed/`) for the ensemble track. Current S=7 ensemble: `data/ensemble_N5_7lag/<MEMBER>/` holds BOTH the deep 28-mark builds (`forced_turbulence_dT_*`, n_snapshots_per_sample=28) AND the sliced training sweeps (`sweep_dT_{5em3,1em2,1p5em2}/`, S=7, max_anchors=3, targets [Ṅ,N̈,N⃛]) — glob `sweep_dT_*` for training, never the deep dirs. Members: FRC-{b0,b05,b075,b1,b2,b25,kf4,Re25k,combo}, FRC-256, DEC-*.

## Hard rules — an agent must NEVER

1. **SGE:** never `-q ibamd.q`; never `-l h_vmem=...G`. GPU jobs use exactly `-q ibgpu.q -l gpu=1`. Match `submit_qg.sh` conventions.
2. **Run location:** all `training/` scripts run FROM `training/` — flat sibling imports. Never restructure `training/` into subpackages; never run from inside a snapshot dir.
3. **Precision:** float64 mandatory throughout the closure-data pipeline and training compute. float32 causes catastrophic cancellation (target O(ΔT³) ≈ 1e-9 < float32 eps). Disk float32 for inputs is the one sanctioned exception (see locked facts).
4. **YAML:** never `5e-3` (PyYAML parses as string) — always `5.0e-3`; explicit `float()` casts on read.
5. **Convergence sweeps:** restart every dt-run from a shared *developed-flow* snapshot, never t=0.
6. **Multigrid:** never convert the per-sample dx,dy rescale to per-batch. Never assume equal shape ⇒ equal dx. Never remove the anisotropy guard or the set_epoch call.
7. **Splits:** never chronological per-sample splits (covariate shift) AND never per-sample random splits on anchored slices (within-window leakage: adjacent anchors are ~dt_fine-apart near-duplicates). Use `resplit_by_window.py` — whole windows to one split, windows shuffled. It backs up the old split to `split_persample.npz`.
8. **Snapshots:** never edit `legacy/snapshots/` (locks).
9. **matplotlib mathtext:** `\tfrac` unsupported (use `\frac`); `\left(` + `\frac{...}` combinations can fail to parse.
10. **Plots:** `cmap='seismic'`; aspect-preserving centered fit — never stretch. Never regenerate slide decks unless explicitly asked.
11. Read the actual code before speculating about bugs; assume commands run from the correct working directory. Give commands assuming the user is already in the right cwd.
12. Resolution: 512² is under-resolved for cylinder at Re ≥ 600 — use 1024².
13. Numerics context: AB2CN2 stability from CN-treated viscosity; imaginary-axis eigenvalues favor AB3/AB4, off-axis favor AB2.
14. **Never emit unused time-orders** (see ORDER CLIP). Never let the mix or stencils receive gradients through 1/dt^k-scaled features that no output needs.
15. **Spin-up is β-dependent — never use one t-start for all members.** The zonal/quiescent phase (J(ψ,ω)≈0, tiny N-derivative targets) lasts longer at higher β (observed: combo 9% of windows still zonal at t-start=15, FRC-256 16%, b2 27%, b25 33%). For NEW deep builds (incl. the upcoming decaying-turbulence members): set t-start per member — either from a developed-flow criterion (e.g. start when the window-median ‖N⃛‖ reaches ~its long-run median, equivalently when stack roughness ‖Δ²ω‖/‖ω‖ ≳ 1e-4) or conservatively t-start ∝ (1+β)·t₀. Regardless, ALWAYS run `filter_quiescent_windows.py` after slicing — it is the safety net (drops windows with target-norm < 1e-2× member median or frozen stacks), and it must precede any training.
16. **Never train or report on unfiltered per-sample relative metrics.** Relative errors explode on near-zero-denominator (quiescent) samples and poison the optimizer (prediction-shrinking collapse: destroyed physics-init mix, trained medians worse than init). Report MEDIANS alongside means; if trained-worse-than-init appears, suspect the pool, not the model, and run `diagnostics/diagnose_error_distribution.py` first.
17. Communication: short sentences. No long walls of prose. Terse, math-forward, corrective.

## Main pipelines

**A. Simulation:** `python run_qg.py +scenario=<name> qg.grid.Nx=... hydra.run.dir=outputs/<name>`; cluster `./submit_qg.sh <job> [--gpu] -- <args>`. Postprocess: `prepare_npz_for_mmap.py`; restart ICs via `extract_restart_ic.py`.

**B. Ensemble dataset (current S=7):** deep 28-mark builds via `build_training_data_mmap.py` (`--Delta-T 5.0e-3 --n-marks 28`, argparse last-wins passthrough in `scripts/sge/build_ensemble_mmap.sh`; **t-start per member, β-scaled — rule 15**) → slice `slice_deriv_from_deep.py --n-snapshots 7 --target-dts 5e-3 1e-2 1.5e-2 --max-anchors 3` (targets computed once per window, dt-independent, per-member forcing from manifest) → **`resplit_by_window.py`** (mandatory — rule 7) → **`filter_quiescent_windows.py`** (mandatory — rule 15). Per-shape dealias masks are member-grid-derived (256² and 512² differ even at equal L).

**C. Train — derivative-loss (current production, multigrid S=7):**
```bash
cd $QG_DIR/training
qsub -q ibgpu.q -l gpu=1 -N deriv7_R3R4 train_deriv_job.sh \
  --sweep-roots data/ensemble_N5_7lag/FRC-*/sweep_dT_* \
  --n-snapshots 7 --out-orders 3 --grad-kernel 15 \
  --epochs 200 --lr 2e-4 --batch-size 4 --compute-dtype float64 \
  --run-name <name>
```
Startup checks: `params=3,700` (not 1,571 — else old un-clipped model), `MULTIGRID ... shapes=[(256,256),(512,512)]`, `2 per-shape projection(s)`, `dT_sweep=[0.005, 0.01, 0.015]`. Epoch-0 val should be O(1), NOT 1e14 (that was the pre-clip leak). Old-vs-new comparability warning: the S=4 lock's 2.6–4% was at dt=1e-3 (no time error); pooled S=7 sweep val includes near/past-wall samples and is expected O(0.1–1) — read `eval_deriv_by_root.py` per-(member,dt), not the pooled number.

**D. Train — δ-pivot:** `train_delta.py --reference both ...` via `train_delta_job.sh`.

**E. Evaluate:** `eval_deriv_by_root.py` (per-member × dt × order rel-L2 of a ckpt — the breakdown the pooled val hides; writes CSV next to ckpt) → `rollout_perfect_closure.py` (analytic ceiling) → `rollout_timed_pareto.py` → `rollout_multistep_comparison.py`; error budget `closure_error_propagation.py` + `run_error_prop_3dt.sh` (two-pass: eps=1 geometry first, real eps after training; rebuild forcing per member via `build_forcing_npy.py`); FD floor `temporal_fd_floor_deep.py`. **When a run looks wrong**, the debug ladder that works (diagnostics/): per-root eval → init-ckpt eval → `diagnose_mark_noise.py` (deep smoothness) → `diagnose_sliced_inputs.py` (byte-compare) → `diagnose_one_sample.py` (stage audit: spectral-floor / fdgrad / model on one sample) → `diagnose_error_distribution.py` (median vs mean + worst offenders).

**F. Numerics validation:** step1–3, `validate_ab2cn2_vs_truth.py` (R₃–R₆), `measure_truncation_magnitudes.py` (ΔT★).

**G. Π_FF (spatial):** `spatial_closure/compute_pi_ff.py`; sweeps `submit_pi_ff_sweep.sh` / `submit_beta_sweep.sh`.

## Known open items

- **Launch `deriv7_filtered_lr5e-5`** (filtered splits, physics init, lr 5e-5, all 18 roots). Then: trained-vs-init medians per root → physics-conditioned model.
- **Upcoming ensemble extension:** decaying-turbulence members to be deep-built — apply rule 15 (per-member t-start; DEC members have no forcing so the zonal-trap differs, but the developed-flow criterion still applies) + the filter.
- **Wiener filter theory** — iPad derivation in progress (see Error theory section). Formalize before building the conditioned model. Open sub-items: explicit C_k from the 7-node Vandermonde remainder; the diagonal-vs-cascade split of ω^(S) that bounds what conditioning can recover.
- **Weight-tied recursion cell** (order-p generalization: one cell unrolled p times gives any N^(p); the recursion is the exact ∂_t — no inner wall, but needs D⁻¹ per order): parked as the next amazing step after conditioning. Decision pending: exact-FFT D⁻¹ (~3 transforms/order) vs learned-local.
- **Flux-vs-advective Jacobian form** in the model: deferred to the conditioned/recursion model design (it is the same decision as the FFT/D⁻¹ question).
- Port `solver_patches/` onto the fork's `closure` branch per `solver_patches/PORTING.md` (diff, don't overwrite — upstream 0.2.3 refactored BCs/integrator/operator splitting).
- Manuscript style: Charous & Lermusiaux (2023, SIAM) — run-in bold headings, notation table, Derivation/Analysis/Illustration pattern, Prop/Cor/Remark environments. Key refs: Suresh Babu, Sadam & Lermusiaux (2025, arXiv 2508.06678); Gupta & Lermusiaux (2021); Ascher–Ruuth–Wetton (1995); Frank–Hundsdorfer–Verwer (1997). δR positioning: modified-equation analysis / non-iterative defect correction / "amortized parareal" (strongest framing); δR vs DC: DC upgrades the scheme at fixed h, δR emulates the SAME scheme at h/K — gain bounded by K (settable), not by reference order.

## EXECUTION MODEL — charter v1.4/v1.5 (adopted 2026-07-15; full text: OPERATING_CHARTER.md I21–I27)

- **DAY MODE (default):** agents run on Sanaa's LOCAL station only. The **I21c ssh sequence is the ONLY day-mode submission path**: `ssh mseas "<cd branch> && git pull --rebase && qsub <job> && qsub <monitor -hold_jid>"` → report job ids. No agent process on the cluster front end in day mode (RED-tier incident).
- **NIGHT MODE** (21:00–07:00 weekdays + weekends): cluster-resident sessions allowed ONLY after the load veto (`ssh mseas "uptime; who | wc -l"`; load > 2.0 or users > 4 ⇒ DAY regardless of hour), `nice -n 19 ionice -c3`, inside tmux, ≤ 3 sessions, hard 07:00 curfew with the I27 yield ritual (push everything, BRANCH_LOG + DECISIONS, exit — never detach-and-leave).
- **PATH PARTITION (I22):** cluster jobs commit ONLY `reports/` and `logs/` with explicit `git add reports/<run>/` — never `-A`. Local agents write everything else. Pushes use the pull-rebase-retry loop (3×, 10 s).
- **LOGS (I23):** raw SGE .o/.e → `<branch>/logs/` (git-ignored, cluster-only, never pasted into a context). Every monitored job pushes a digest to `reports/<run-name>/` (progress.csv + status.md ≤ 20 lines + summary.md on completion) via `diagnostics/digest_writer.py`.
- **REFLEXES (I24):** monitors carry the X1–X6 ladder (`diagnostics/monitor_training.py` v3, canonical on main — adopt on this branch's NEXT submission and confirm adoption in the next digest). X1 explode / X2 inversion×3 / X3 stall ⇒ qdel + X4 auto-diagnose; X5 whitelist-only auto-resubmit, once; X6 heartbeat. Scientific failures wait for a session.
- **SESSION OPEN (I25):** `git pull` → read `reports/*/status.md` → `ssh mseas "qstat -u sanaamz"` → reconcile — BEFORE anything Sanaa asks for.
