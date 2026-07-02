# QG-Closure Repo Audit v2 — Project files + uploaded training/ zips
*(2026-07-02. Sources: Project knowledge snapshot (solver/analysis/SGE) + 5 uploaded zips: `General_files` = live `training/` dump; `Shallow_NN_*` = archival snapshot locks. Categories: (a) source to version-control, (b) config, (c) docs, (d) data/output — NOT for GitHub, (e) obsolete/superseded → `legacy/`. Version claims below are md5/diff-verified, not filename-guessed.)*

---

## 0. What the uploads are

| Zip | Role |
|---|---|
| `General_files/` | Dump of the live cluster `training/` dir (68 files). Contains the **newest** copies of the shared modules, plus history (.bak, superseded fix-generations, output PNGs). |
| `Shallow_NN_one_test_case/` | Lock: single-trajectory forced-turbulence `cheap_deriv` pipeline (train.py with `--model cheap_deriv`, FT builder, timed-Pareto + perfect-closure rollouts). |
| `Shallow_NN_enssemble_fixedGrid_v1/` | Lock: the **δ-pivot** — `train_delta.py` + FiLM corrector, empirical `delta = Φ_ref − Φ_AB2CN2`, `--reference {exact,rk4,both}`. |
| `Shallow_NN_enssemble_fixedGrid_v2/` | Lock: the **derivative-loss** single-grid model (pre-6.1.2): `cheap_deriv` predicts [Ṅ, N̈, N⃛] directly, L^k applied analytically at inference. No corrector. |
| `Shallow_NN_enssemble_MultiGrid/` | **Live staging area, NOT a lock**: the *next step* after v2 — v2 generalized to pool mixed grids/domains (per-sample dx,dy rescale; bit-identical to v2 on a single grid). Receives parallel updates whenever a must-not-forget multigrid consideration comes up during v2 work. Promoted only once v2 is finished. |

**Snapshot handling recommendation:** commit the three true locks (`one_test_case`, `fixedGrid_v1`, `fixedGrid_v2`) verbatim under `legacy/snapshots/` — self-documenting, README'd, and referenced by the pending A/B experiment plan in `README_SNAPSHOT_v1.txt`. `MultiGrid` is *active* work-in-waiting, not history: track it at `staging/multigrid/` (top level, clearly not legacy). Once on git, the cleaner long-term home for this workflow is a `multigrid` **branch** — "remembered a multigrid consideration" becomes a commit there instead of a parallel folder edit — but the folder works fine until then. Canonical v2 working copies go in `training/`.

---

## 1. Canonical-version resolution (hash/diff-verified)

| Module | Copies | Canonical | Evidence |
|---|---|---|---|
| `model_deriv_closure.py` | 3 distinct versions | **`General_files/`** (19,527 B) | General = *factored dx-independent* SpatialGrad (dimensionless unit-spacing stencil, per-sample 1/dx at forward — mirrors TimeFD's W_unit/dt^k). Supersedes MultiGrid's physical-kernel variant, which supersedes v1==v2==one_test_case (single-grid). |
| `deriv_dataset.py` | 3 distinct | **`General_files/`** | 6-dim regime `[dT, β, ν, μ, dx, dy]`, refactored to match the factored model. |
| `train_deriv.py` | 2 distinct | **`General_files/`** (== v2, *by design*) | The v2 single-grid trainer IS the current WIP — General_files matches it because v2 work happens in the main training dir. The MultiGrid copy (no grid restriction, per-shape dealias dict) is the staged next step, not the canonical present. |
| `train_delta.py` | 2 distinct | **`General_files/`** | v1 snapshot copy is the older lock. |
| `delta_dataset.py` | 2 distinct | **`General_files/`** | v1 copy older. |
| `build_training_data_fixD_v2.py` | 2 distinct | **`General_files/`** (50,966 B vs Project 47,363 B) | General is a later revision — the Project copy from the first audit is stale. |
| `rollout_perfect_closure.py` | 2 distinct | **`one_test_case/`** (41,893 B, FT + cheap_deriv edition) | Project copy (31,474 B) is the older decay edition → legacy. |
| `dataset.py` (training) | identical in General/v1/one_test_case | any | Note: unrelated to the *solver's* `dataset.py` (FR extraction) despite the name. |
| `add_deriv_targets.py`, `concat_dataset.py`, `closure_operators.py`, `build_training_data_mmap.py`, `slice_delta_sweep.py`, `split_ensemble.py`, `resplit_by_window.py`, `rollout_timed_pareto.py` | identical everywhere they appear | any | Single version. |

**Workflow note (resolves the apparent trainer/model mismatch):** General_files carries the factored model + 6-dim-regime dataset alongside the v2 single-grid trainer because **v2 is the active WIP in the main training dir** — the factored SpatialGrad is grid-agnostic and runs single-grid at rescale ×1, so it's already part of the v2 line. The MultiGrid dir is the parallel staging area for the step above (updated whenever a multigrid consideration surfaces mid-v2). Promotion path when v2 is done: merge the staged trainer changes (DIFF 1–4 in the MultiGrid README) into `training/train_deriv.py`, run the single-grid equivalence check (one member must match the v2 lock to ~machine precision), then re-snapshot.

---

## 2. Inventory — uploaded files (Project-file inventory unchanged from audit v1, §4 below carries it forward)

### `General_files/` → `training/` keepers (a)
| File | What it does |
|---|---|
| `model_deriv_closure.py` | **cheap_deriv** — CANONICAL model. TimeFD (frozen-exact W_unit/dt^k) → learnable spatial stencils (grad_kernel 15, FD-init) → Jacobian features → 1×1 mix (binomial physics-init) → [Ṅ, N̈, N⃛]; optional FiLM corrector tail. Factored dx-independent. |
| `model.py` / `model_fixD.py` | Earlier architectures (PeriodicUNet/CNN; bilinear ~150K) — keep as baselines. |
| `closure_operators.py` | Scheme-specific R_p assembly (AB2CN2 / AB4CN2 verified coefficients) from N-derivative channels. |
| `train_deriv.py` | Derivative-loss trainer (rel-L2 per-sample/channel, cosine, dealias-pred, f64) — commit MultiGrid variant pending cluster pull (see §1). |
| `train_delta.py` | δ-pivot trainer (`--reference exact/rk4/both`, `--pure-empirical`, `--freeze-physics`). |
| `dataset.py` | Per-sample .npz Dataset for the fixD track (manifest.json + split.npz + samples/). |
| `concat_dataset.py` | Ensemble concat + `GridHomogeneousBatchSampler` + `snapshot_input_fields` (imported by deriv/delta datasets — load-bearing). |
| `deriv_dataset.py` | Pooled derivative-loss dataset; regime `[dT,β,ν,μ,dx,dy]`. |
| `delta_dataset.py` | Pooled δ-pivot dataset (delta_exact/delta_rk4 targets). |
| `build_training_data_fixD_v2.py` | fixD v2 builder (also imported by `add_deriv_targets.py` for `build_L_hat`, `J_phys`, `compute_n_{dot,ddot}_analytical` — keep even as the mmap builder takes over). |
| `build_training_data_mmap.py` | **Current** builder: mmap-direct, SEVEN-snapshot stencil (S=7 ⇒ up to N⁽⁶⁾ representable), N⁽¹⁾..N⁽⁵⁾ targets. |
| `add_deriv_targets.py` | Builds `packed/deriv_anal_f64.npy` = analytic [Ṅ,N̈,N⃛] per sweep dir (forcing rebuilt exactly from manifest). Run once before deriv training. |
| `add_delta_target.py` | Adds empirical δ targets to a packed ensemble. |
| `build_forcing_npy.py` | Rebuilds FRC static forcing byte-matching builder/slicer (for error-propagation). |
| `slice_delta_sweep.py` | Slices δ sweeps from packed fine trajectories (no new RK4). |
| `slice_deriv_from_deep.py` | Builds n-lag deriv sweep dirs by slicing the deep survivors (Re25k, combo, kf4). |
| `split_ensemble.py` | Manifest-driven (not folder-name) train/test ensemble split. |
| `reshuffle_splits.py` | Leakage-free block-shuffle split (fixes chronological covariate shift). |
| `resplit_by_window.py` | Split by deep window (fixes within-window temporal leakage). |
| `pack_dataset_mmap.py` | One-time repack of per-sample .npz → contiguous memmap arrays. |
| `extract_omega_from_dns_npy.py` | DNS.npy (B,T,4,Ny,Nx) → omega/times .npy. |
| `closure_error_propagation.py` | ε_Ndot/ε_Nddot/ε_N3dot → δ-error propagation through the analytic assembly (source of the "closure error ≈ ε_N̈ 1:1" result). |
| `temporal_fd_floor_deep.py` | Temporal-FD floor of a depth-n stencil with perfect spatial operators (sets ΔT_inner(k)). |
| `rollout_timed_pareto.py` | Timed truth/bare/closure rollout + cost-vs-accuracy Pareto front. |
| `rollout_perfect_closure.py` *(one_test_case copy)* | Closure-ceiling rollout, FT + cheap_deriv edition. |
| `diagnose_training_plateau.py`, `diagnose_target_distribution.py`, `inspect_fixD_v2_data.py`, `verify_target_consistency_1.py`, `visualize_rollout_correction.py` | Diagnostics (same as audit v1). |
| `benchmark_walltime.py` | Wall-time vs dt (General copy 7,481 B — slightly newer than Project's 7,228 B; keep General's). |

### `General_files/` → `analysis/` keepers (a)
| File | What it does |
|---|---|
| `ab_stability_regions.py` | AB1–AB4 + RK4 absolute-stability regions (the AB2-vs-AB3/4 eigenvalue argument). |
| `vn_stability.py` | Von Neumann frozen-coefficient stability of the AB2CN2 IMEX as implemented (νk² ≳ (3/8)U⁴k⁴Δt³ result). |
| `compute_vs_dt.py` | Theoretical cost-to-solve curves vs ΔT. |

### `General_files/` → SGE `scripts/sge/` keepers (a)
`build_ensemble_mmap.sh`, `build_training_data_mmap_ft.sh` (header is a stale copy of build_ensemble_mmap — cosmetic), `submit_build_fixD_v2_K100.sh`, `submit_build_fixD_v2_K1000.sh`, `train_deriv_job.sh` (header comment stale — says train_delta), `train_delta_job.sh`, `train_decay_fixD_v2.sh`, `train_decay_fixD_v2_annealing.sh`, `train.sh`, `train_v2.sh`, `train_job.sh`, `rollout_perfect_closure.sh`, `rollout_timed_pareto.sh`, `run_error_prop_3dt.sh`, `build_training_data_job.sh`, `build_training_data_decay_bytime.sh`, `build_training_data_ft.sh` *(from one_test_case)*, `train_ft_cheap_deriv.sh` *(from one_test_case)*, `train_v2_job.sh` *(from one_test_case)*.

### `General_files/` → legacy (e)
| File | Why |
|---|---|
| `build_training_data.py.bak`, `dataset.py.bak` | .bak files — never commit at top level. |
| `build_training_data.py`, `build_training_data_fixD.py`, `build_training_data.sh`, `submit_build_fixD.sh` | Pre-fixD_v2 builder generations. |
| `dataset_fixB.py`, `dataset_fixD.py` | Superseded by the unified `dataset.py`. |
| `train_v2.py` | Strict subset of `train_v2_annealing.py` (verified in audit v1). |
| `train_decay_fixB.sh`, `train_decay_fixC.sh`, `build_training_data_decay_richN.sh`, `train_decay_bytime_richinput.sh` | fixB/fixC/richN experiment generations. |

### `General_files/` → NOT for GitHub (d)
`ab_stability_regions.png`, `compute_vs_dt_2.png` (regenerable figures), `walltime.json` (measurement output — regenerable by `benchmark_walltime.py`; commit only if you want it as a record, in which case put it under `docs/results/`).

### Snapshot dirs (a — commit verbatim under `legacy/snapshots/`)
All files in `Shallow_NN_one_test_case/`, `Shallow_NN_enssemble_fixedGrid_v1/`, `Shallow_NN_enssemble_fixedGrid_v2/`, `Shallow_NN_enssemble_MultiGrid/` including the three `README_SNAPSHOT*.txt` (category c — these are the best model-lineage documentation in the project).

---

## 3. Updated repo hierarchy

```
qg-closure/
├── README.md, CLAUDE.md, .gitignore, pyproject.toml
│
├── src/qg/                          # UNCHANGED from audit v1:
│   ├── __init__.py, config.py, train.py, test.py, run_qg.py
│   ├── conf/{config.yaml, scenario/*.yaml}
│   ├── solver/{qg.py, util.py*, dataset.py, integrator/imex.py,
│   │           grid/cartesian.py, opt/{basis,derivative,filter}.py,
│   │           opt/operator/{__init__,jacobian,obstacle}.py}
│   ├── _input/sources/{bc,ic,forcing,mask,submarine_mask}.py
│   └── viz/draw.py
│   # *util.py (_Math/_Cache) STILL missing — remains a cluster pull.
│
├── training/                        # FLAT — mirrors cluster $QG_DIR/training/.
│   │                                # Do NOT introduce subpackages: deriv_dataset,
│   │                                # add_deriv_targets etc. do flat sibling imports
│   │                                # (concat_dataset, build_training_data_fixD_v2)
│   │                                # and MUST run from this dir.
│   ├── model_deriv_closure.py       # ← General_files (factored) — CANONICAL cheap_deriv
│   ├── model.py, model_fixD.py
│   ├── closure_operators.py
│   ├── train_deriv.py               # ← General_files (== v2 single-grid; the active WIP)
│   ├── train_delta.py               # ← General_files
│   ├── train_v2_annealing.py        # fixD/f_NN_target track trainer
│   ├── train.py                     # ← one_test_case (single-trajectory cheap_deriv entry)
│   ├── dataset.py, concat_dataset.py, deriv_dataset.py, delta_dataset.py
│   ├── build_training_data_mmap.py  # current S=7 builder
│   ├── build_training_data_fixD_v2.py   # ← General_files copy (newer than Project's)
│   ├── add_deriv_targets.py, add_delta_target.py, build_forcing_npy.py
│   ├── slice_delta_sweep.py, slice_deriv_from_deep.py
│   ├── split_ensemble.py, reshuffle_splits.py, resplit_by_window.py
│   ├── pack_dataset_mmap.py, prepare_npz_for_mmap.py
│   ├── extract_omega_from_dns_npy.py, extract_restart_ic.py
│   ├── closure_error_propagation.py, temporal_fd_floor_deep.py
│   ├── rollout_timed_pareto.py
│   ├── rollout_perfect_closure.py   # ← one_test_case (FT/cheap_deriv edition)
│   ├── benchmark_walltime.py        # ← General_files copy
│   └── diagnostics: verify_target_consistency_1.py, inspect_fixD_v2_data.py,
│                    diagnose_training_plateau.py, diagnose_target_distribution.py,
│                    visualize_rollout_correction.py
│
├── analysis/                        # audit-v1 set, plus:
│   ├── ab_stability_regions.py, vn_stability.py, compute_vs_dt.py
│   └── (step1/2/3, validate_ab2cn2_vs_truth, measure_truncation_magnitudes,
│        convergence_plot_bulk, rollout_multistep_comparison + replot,
│        rollout_load_truth_compare, rerender_videos)
│
├── spatial_closure/compute_pi_ff.py
│
├── scripts/sge/                     # audit-v1 set, plus:
│   ├── build_ensemble_mmap.sh, build_training_data_mmap_ft.sh
│   ├── submit_build_fixD_v2_K100.sh, submit_build_fixD_v2_K1000.sh
│   ├── train_deriv_job.sh, train_delta_job.sh
│   ├── train_decay_fixD_v2.sh, train_decay_fixD_v2_annealing.sh
│   ├── train_ft_cheap_deriv.sh, train_v2_job.sh, train_v2.sh
│   ├── build_training_data_ft.sh
│   ├── rollout_perfect_closure.sh, rollout_timed_pareto.sh, run_error_prop_3dt.sh
│   └── (qg_job/submit_qg, steps 1–3, decay/cyl/ft sweep v-latest, pi_ff, β-sweep)
│
├── staging/
│   └── multigrid/                   # ← Shallow_NN_enssemble_MultiGrid — ACTIVE, not legacy.
│                                    #   Parallel deposit of multigrid changes/notes while v2
│                                    #   is in progress; merge into training/ when v2 is done.
│                                    #   (Long-term: replace with a `multigrid` git branch.)
├── docs/
│   ├── THEORETICAL_GUARANTEES.md
│   └── slides/build_backup_slides.js
├── paper/qg_closure_arxiv.tex
│
└── legacy/
    ├── snapshots/                   # COMMIT VERBATIM — self-documenting locks
    │   ├── Shallow_NN_one_test_case/           # single-trajectory FT lock
    │   ├── Shallow_NN_enssemble_fixedGrid_v1/  # δ-pivot lock (train_delta + corrector)
    │   └── Shallow_NN_enssemble_fixedGrid_v2/  # derivative-loss single-grid lock (pre-6.1.2)
    ├── build_training_data.py, build_training_data_fixD.py, *.py.bak
    ├── dataset_fixB.py, dataset_fixD.py, train_v2.py
    ├── rollout_perfect_closure_decay.py        # old Project copy, renamed
    ├── train_decay_fixB.sh, train_decay_fixC.sh,
    │   build_training_data_decay_richN.sh, train_decay_bytime_richinput.sh
    └── (audit-v1 legacy set: __init__qg.py, util/util_output, sweep v1–v3
        scripts, verbatim colleague reproductions, step3_v1, etc.)
```

**Model/trainer lineage (keep straight — this is the versioning story):**
```
fixD (bilinear, f_NN_target)                    → model_fixD.py + train_v2_annealing.py
  └→ cheap_deriv, derivative-loss, 1 trajectory → one_test_case lock (train.py)
      └→ ensemble, single-grid ("pre-6.1.2")    → fixedGrid_v2 lock; ACTIVE WIP in training/
      │                                           (factored dx-independent model + 6-dim regime
      │                                            already merged into this line)
      └→ [NEXT] multigrid (per-sample dx,dy)    → staging/multigrid — parallel deposit,
                                                  promoted after v2 completes
  and, in parallel:
      δ-pivot (empirical delta + corrector)     → fixedGrid_v1 lock (train_delta.py)
```

## 4. Project-file inventory (audit v1) — deltas only
Unchanged except: `build_training_data_fixD_v2.py`, `rollout_perfect_closure.py`, `benchmark_walltime.py` in the Project are **stale copies** of the General_files/one_test_case versions → superseded (e). The `cheap_deriv` gap flagged in v1 is now **closed** (`model_deriv_closure.py`). Remaining cluster pull: `qg/solver/util.py` only.
