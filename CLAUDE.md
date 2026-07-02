# CLAUDE.md — QG-Closure

## What this project is

ML-based closure modeling for coarse-grid 2D quasi-geostrophic (QG) turbulence (MIT MSEAS, Lermusiaux group). Pseudo-spectral GPU solver (PyTorch) with AB2CN2 IMEX time stepping. Two closure tracks:

1. **Temporal closure (δR)** — primary. Train a small NN so AB2CN2 at coarse ΔT matches AB2CN2 at ΔT/K: a K-fold *effective step-size reduction* at K-fold lower cost. This is **not an order bump** — the scheme stays 2nd-order; effective resolution is what improves. Error floor: `C·ΔT²/K² + ε_NN·T`. Paper framings: "amortized parareal" (fine propagator paid at training, reused at inference); vs. defect correction: DC matches a higher-order scheme at the same h, δR matches the *same* scheme at h = ΔT/K; gain settable via K; K=∞ targets exact flow (no DC analog).
2. **Spatial closure (Π)** — CNN+SVGP sub-grid-scale forcing; goal is multi-β generalization via `Pi_spatial × β_NN(features)` factorization with β as explicit input.

Flow scenarios: decaying turbulence (DEC), forced turbulence (FRC, β-plane), flow past cylinder, flow past cape.

## Repo layout (agent orientation)

```
external/qg-simple/  QG solver — SUBMODULE (fork of akhilsadam/qg-simple, `closure` branch
                     off package-stable). Install: uv pip install -e external/qg-simple.
                     Solver bug reports/fixes go to the fork, not this repo.
solver_patches/      0.2.1-era local solver modifications awaiting port onto the fork
                     (PORTING.md). Not imported. Deleted once ported.
training/            v2 derivative-loss pipeline — ACTIVE WIP. Flat; run from here.
staging/multigrid/   Next step — parallel deposit area (editable, never auto-merged).
analysis/            Convergence / truncation / stability / rollout figures.
spatial_closure/     Π_FF pipeline.
scripts/sge/         Cluster submission (submit_X.sh → X_job.sh).
docs/, paper/        Theory notes + audit; manuscript .tex.
legacy/              Superseded code; legacy/snapshots/ = read-only locks.
```
Subdirectories carry their own CLAUDE.md with local rules — read the one for the dir you work in.

## Canonical math (do not deviate)

PDE convention: `∂_t ω̄ = Lω̄ + N`, `N = −J(ψ̄, ω̄) + F` (F time-independent ⇒ Ḟ = F̈ = 0), `ψ̄ = ∇⁻²ω̄`, `L̂(k) = −ν|k|² − μ + iβk_x/|k|²`.

N-derivative chain rule:
```
ω̇̄ = Lω̄ + N;              ψ̇̄ = ∇⁻²ω̇̄
Ṅ  = −J(ψ̇̄, ω̄) − J(ψ̄, ω̇̄);  ω̈̄ = Lω̇̄ + Ṅ;  ψ̈̄ = ∇⁻²ω̈̄
N̈  = −J(ψ̈̄, ω̄) − 2J(ψ̇̄, ω̇̄) − J(ψ̄, ω̈̄)
```
Forcing is load-bearing for N̈/N⃛ (enters via ω̇̄) even though Ḟ=0 — always rebuild F exactly from the manifest (FRC: `A cos(Bx) + D cos(Ey)`; DEC: F=0).

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
Error propagation (`closure_error_propagation.py`): at operating ΔT the closure error ≈ ε_N̈ **1:1** — no L-amplification. Consequence: the per-derivative loss IS the closure objective, and **N̈ val rel-L2 sets the rollout floor** — it is the number to watch.

Convergence radius: modified-equation series converges only for ΔT < ΔT★ ≈ 2.08·τ_eddy (≈2e-2 here); inner wall ΔT_inner(k) from the finite-lag time stencil is hit first (`temporal_fd_floor_deep.py`, `docs/THEORETICAL_GUARANTEES.md`). No finite-order closure helps past ΔT★.

Inference decomposition (fixD/f_NN_target track):
```
f(ω̄) = R⁻¹(ΔT)·ΔT³(1−1/K²)·(1/12)·L³ω̄     [linear, IMPLICIT, folded into IMEX]
      + S⁻¹(ΔT)·ΔT³(1−1/K²)·(1/12)·L²N(ω̄)  [nonlinear via J, EXPLICIT analytical]
      + f_NN(ω̄)                              [NN, EXPLICIT]
S⁻¹(ΔT) = (1 − 0.5·ΔT·L̂)/ΔT
```

## Model & trainer lineage (the versioning story — respect it)

```
fixD (bilinear, predicts f_NN_target)             model_fixD.py + train_v2_annealing.py
 └→ cheap_deriv, derivative-loss, 1 trajectory    legacy/snapshots/Shallow_NN_one_test_case
     └→ ensemble, SINGLE-GRID ("pre-6.1.2")       ← CURRENT ACTIVE WIP, lives in training/
     │    (v2 lock preserved in legacy/snapshots/..._fixedGrid_v2; the factored
     │     dx-independent model + 6-dim regime are already part of this line)
     └→ [NEXT STEP] multigrid                     staging/multigrid/ — parallel deposit dir,
          (per-sample dx,dy rescale)              updated whenever a multigrid consideration
                                                  surfaces mid-v2; merged only after v2 is done
 parallel: δ-pivot (empirical delta + corrector)  legacy/snapshots/..._fixedGrid_v1, train_delta.py
```
- **Derivative-loss track (current — v2, single-grid/same-domain, active WIP in `training/`):** `cheap_deriv` predicts LOCAL [Ṅ, N̈, N⃛] directly; L^k weightings (incl. the nonlocal β term) applied **analytically at inference, never learned**. Pipeline: 4 stages — TimeFD → spatial grads → Jacobian features → 1×1 mix (physics-init to chain-rule binomials). Corrector OFF (hidden=0).
- **Multigrid is the staged next step, not the present:** `staging/multigrid/` accumulates changes and must-not-forget considerations in parallel while v2 work proceeds. Before starting any multigrid task, read `staging/multigrid/README_SNAPSHOT.txt` (DIFF 1–4) AND check the staging dir for deposits newer than the README. Promotion after v2 completes: merge staged trainer changes, run the single-grid equivalence check (one member must reproduce the v2 lock to ~machine precision).
- **δ-pivot track:** target `δ = Φ_ref − Φ_AB2CN2`, references `exact` (fine RK4) / `rk4` / `both` (1-bit FiLM flag); `δ_exact − δ_rk4 = τ_RK4`. Three-run experiment: `--pure-empirical` / hybrid_unfrozen / `--freeze-physics`; follow-ons Option A (pointwise kernel-1 corrector on L̂-basis channels, R5-coeff init (13,13,13,−17,8,−7)/240) vs Option B (`--no-corrector`, delta-loss only) are specified in `legacy/snapshots/Shallow_NN_enssemble_fixedGrid_v1/README_SNAPSHOT_v1.txt`.
- The snapshot READMEs are the authoritative lineage docs. Never edit files inside `legacy/snapshots/` — they are locks.

## Locked-in model facts (from the snapshot locks — do not "fix" these)

- **TimeFD is frozen-exact:** across a dT sweep the model uses the per-sample `W_unit/dt^k` path, NOT the learnable `self.weight` — which is an inert parameter (no gradient) that just pads the param count. Do not report it as trained capacity.
- **Spatial stencils ARE learnable** (width-15, FD-init, refine toward spectral); this tightens the high-k N̈ gap.
- **Factored SpatialGrad:** the Parameter is the DIMENSIONLESS unit-spacing stencil; 1/dx, 1/dy applied per-sample at forward (`conv(x,S)/dx == conv(x,S/dx)` exactly). One stencil serves all grids; Adam never sees the 1/dx amplification.
- **The multigrid trap:** `GridHomogeneousBatchSampler` groups by SHAPE, which does NOT imply equal dx (512²/2π vs 512²/4π). The dx rescale must be **per-sample, never per-batch** — per-batch silently corrupts mixed-domain batches.
- Dealias projections are per-SHAPE (mode-index based, independent of L) — build a dict keyed by `(Ny,Nx)`.
- Model is exactly quadratic in the input field; β, ν, μ are not model inputs (N^(m), m≥1 is forcing-free and regime-independent by construction). Regime vector `[dT, β, ν, μ, dx, dy]` conditions only the corrector/assembly, not the derivative map.
- No FFTs inside the model: ∂_t commutes with FFT exactly but NOT with products (product rule generates the binomial structure) — time-FD from stored ψ-history is the design, not an approximation to remove.
- float32-spatial mixed precision in the model is an INFERENCE optimization only; training is full float64.
- Reference config that produced the reference rollout (v2 lock): inputs `omega_0..m3 psi_0..m3` (n_time=4), targets `N_dot/N_ddot/N_3dot_0_anal`, grad_kernel 15, loss rel_l2 per-sample/per-channel, dealias-pred on, normalize off, bs 4, 200 epochs, lr 5e-5, wd 1e-4, cosine, f64, seed 0. Current run `deriv7_equalw_R3R4`: S=7 stencil, lr 1e-4, pooling 256²+512² FRC members.

## Environment

- Python ≥ 3.10, `uv`-managed venv. Solver = `quasigeostrophic-flow` (upstream now 0.2.3), installed editable from the submodule: `uv pip install -e external/qg-simple` (clone with `--recurse-submodules`). Deps (unpinned): numpy, torch, lightning, einops, hydra-core, omegaconf, wandb, mura ≥ 0.3.0, jpcm, matplotlib, ffmpeg-python, netcdf4, pyyaml, pytest, tqdm, gitpython.
- Cluster: SGE. `QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure`, `QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg`, venv `$QG_ROOT/qg-env/`, training dir `$QG_DIR/training/`.
- Data: per-sample `.npz` (HDF5 abandoned — do not reintroduce) for the fixD track; packed contiguous memmap `.npy` (`inputs.npy`, `deriv_anal_f64.npy`, `delta_*_f64.npy` under `sweep_dT_*/packed/`) for the ensemble track. Ensemble layout: `data/ensemble_N5/<MEMBER>/sweep_dT_<tag>/{manifest.json, split.npz, packed/}`. Members: FRC-{b0,b05,b075,b1,b2,b25,kf4,Re25k,combo}, FRC-256, DEC-*.

## Hard rules — an agent must NEVER

1. **SGE:** never `-q ibamd.q`; never `-l h_vmem=...G`. GPU jobs use exactly `-q ibgpu.q -l gpu=1`. Match `submit_qg.sh` conventions.
2. **Run location:** all `training/` scripts run FROM `training/` — they do flat sibling imports (`concat_dataset`, `build_training_data_fixD_v2`, installed `qg`). Never restructure `training/` into subpackages; never run from inside a snapshot dir (ModuleNotFoundError by design — snapshots are archival locks).
3. **Precision:** float64 mandatory throughout the closure-data pipeline and training. float32 causes catastrophic cancellation in `e_total` at K=100 (target O(ΔT³) ≈ 1e-9 < float32 eps).
4. **YAML:** never `5e-3` (PyYAML parses as string) — always `5.0e-3`; explicit `float()` casts on read.
5. **Convergence sweeps:** restart every dt-run from a shared *developed-flow* snapshot, never t=0 (chaos amplifies spinup differences — this is why sweep v1 scripts were superseded).
6. **Multigrid:** never convert the per-sample dx,dy rescale to per-batch (rule above). Never assume equal shape ⇒ equal dx.
7. **Splits:** never use chronological per-sample splits on chaotic trajectories (covariate shift) — use `reshuffle_splits.py` block-shuffle or `resplit_by_window.py` (window-level, prevents within-window leakage).
8. **Snapshots:** never edit `legacy/snapshots/` (locks); flatten a snapshot into `training/` only deliberately, documenting it. `staging/multigrid/` is the one snapshot-style dir that IS editable — it's the forward deposit area, and edits there must never be silently merged into `training/` before v2 is done.
9. **matplotlib mathtext:** `\tfrac` unsupported (use `\frac`); `\left(` + `\frac{...}` combinations can fail to parse.
10. **Plots:** `cmap='seismic'`; aspect-preserving centered fit — never stretch. Never regenerate slide decks unless explicitly asked.
11. Read the actual code before speculating about bugs; assume commands run from the correct working directory.
12. Resolution: 512² is under-resolved for cylinder at Re ≥ 600 — use 1024².
13. Numerics context: AB2CN2 stability from CN-treated viscosity (νk² ≳ (3/8)U⁴k⁴Δt³ — see `analysis/vn_stability.py`); imaginary-axis eigenvalues favor AB3/AB4, off-axis favor AB2 (`analysis/ab_stability_regions.py`).

## Main pipelines

**A. Simulation:** `python run_qg.py +scenario=<name> qg.grid.Nx=... hydra.run.dir=outputs/<name>`; cluster `./submit_qg.sh <job> [--gpu] -- <args>`. Postprocess: `prepare_npz_for_mmap.py` / `extract_omega_from_dns_npy.py`; restart ICs via `extract_restart_ic.py`.

**B. Ensemble dataset (current):** `build_training_data_mmap.py` (S=7 stencil, N^(1)..N^(5) targets, mmap-direct) over every member via `scripts/sge/build_ensemble_mmap.sh` → slice (`slice_delta_sweep.py` / `slice_deriv_from_deep.py`) → fix splits (`resplit_by_window.py`) → add targets once: `python add_deriv_targets.py data/ensemble_N5 --device cuda` (idempotent; `--overwrite` to rebuild) and/or `add_delta_target.py`.

**C. Train — derivative-loss (production):**
```bash
cd $QG_DIR/training
qsub -N deriv_mg -q ibgpu.q -l gpu=1 train_deriv_job.sh \
  --sweep-roots data/ensemble_N5/*/sweep_dT_* \
  --n-snapshots 4 --out-orders 3 --grad-kernel 15 \
  --epochs 200 --lr 5e-5 --weight-decay 1e-4 --batch-size 4 \
  --num-workers 8 --compute-dtype float64 --seed 0
```
Multigrid smoke test first (FRC-b05 + FRC-256, 5 epochs): confirm startup shows both shapes and epoch-0 N̈ < 1 (physics-init survives rescale). Single-grid equivalence check: one member must match the v2 lock to ~machine precision.

**D. Train — δ-pivot:** `train_delta.py --reference both ...` via `train_delta_job.sh`; three-way pure / hybrid_unfrozen / `--freeze-physics`; then Options A/B per the v1 snapshot README.

**E. Evaluate:** `rollout_perfect_closure.py` (analytic ceiling) → `rollout_timed_pareto.py` (timed truth/bare/closure + Pareto front) → `rollout_multistep_comparison.py` / `replot_rollout_multistep.py`; error budget via `closure_error_propagation.py` (+ `run_error_prop_3dt.sh`), FD floor via `temporal_fd_floor_deep.py`.

**F. Numerics validation:** step1–3 (convergence, fields, Ṅ/N̈ vs finite-diff `(Nⁿ−Nⁿ⁻¹)/dt_fine`), `validate_ab2cn2_vs_truth.py` (R₃–R₆), `measure_truncation_magnitudes.py` (ΔT★).

**G. Π_FF (spatial):** `spatial_closure/compute_pi_ff.py`; sweeps `submit_pi_ff_sweep.sh` / `submit_beta_sweep.sh`.

## Known open items

- Solver completeness (incl. `qg/solver/util.py`) is now provided by the `external/qg-simple` submodule — the previously-missing modules live upstream. Open task instead: port `solver_patches/` onto the fork's `closure` branch per `solver_patches/PORTING.md` (local files are 0.2.1-era; upstream 0.2.3 refactored BCs/integrator/operator splitting — diff, don't overwrite).
- v2 (single-grid) is in progress; multigrid promotion pending v2 completion (see lineage section). Do not treat `staging/multigrid/` contents as canonical until then.
- Manuscript style: Charous & Lermusiaux (2023, SIAM) — run-in bold headings, notation table, Derivation/Analysis/Illustration pattern, Prop/Cor/Remark environments. Key refs: Suresh Babu, Sadam & Lermusiaux (2025, arXiv 2508.06678); Gupta & Lermusiaux (2021); Ascher–Ruuth–Wetton (1995); Frank–Hundsdorfer–Verwer (1997).
