# [QG][BRIEF][SGS-CLOSURE] Branch charter: SGS spatial closure — case setup, diagnostics prep, a priori Π_FF

You are the supervising agent for a new experiment branch. This brief is binding. Where it conflicts with your own judgment, follow the brief and flag the conflict in your report. Where the brief is silent, ask before acting.

## 0. Mission summary

Goal of this branch (Phase 1 of the SGS spatial-closure track): prepare non-stationary flow-past-obstacle cases with time-varying inlet velocity, run the fine-resolution reference ensemble, perform grid/dt convergence verification, and compute the a priori SGS forcing Π_FF at multiple filter scales. Online closure (CNN+SVGP) comes later and is out of scope for this brief. Comprehensive CFD diagnostics code (forces, St, Cp, wake statistics) will be delivered by Sanaa separately; your job is to have the data ready for it and to integrate/run it when it lands.

Framing constraint (must appear in every report): the solver is 2D. "Truth" is defined as the fine-grid penalized-obstacle 2D solution. No claim of agreement with 3D Re=3900 experiments or LES benchmarks is ever made. Literature values (Cd ≈ 0.99, St ≈ 0.21) may be cited only as context, never as validation targets.

## 1. Branch and environment setup (Phase 0)

1. Create branch `exp/sgs-closure` as a git worktree, following the existing worktree pattern of the other `exp/*` branches. Base it on the current main of the closure repo.
2. All work happens in this worktree. The PreToolUse guard hook applies: no forbidden SGE flags, no direct pushes to main, no float32 in closure commands.
3. Solver precision: float64 for all runs on this branch. State the dtype explicitly in every job script; do not rely on defaults.
4. Config audit (do this before any run):
   - `flow_past_cylinder_sponge.yaml`: `nu: 5e-3` and `tol: 1e-3` are PyYAML string-parse traps. On this branch, rewrite ALL scientific-notation numerics in every YAML you touch to the explicit form (`5.0e-3`, `1.0e-3`, `2.5e-4`, ...). Additionally audit that every consumer of these values applies `float()`; report any bare lookup you find, do not silently fix code outside the scope below.
   - Confirm `mask.r = 0.628318530717959` (D = 2r = 1.256637) in the FPC config; all Reynolds/shedding numbers below assume this D. Do NOT copy timescale tables from any prior modulation document — those were derived for a different geometry. Recompute everything from the config as specified here.
5. Storage check: the production ensemble will produce roughly 100 GB of `DNS_FR.npz` under `$QG_ROOT` (estimate below). Verify quota headroom on /gdata before submitting Phase B and report the number.

## 2. Physical regime (fixed, non-negotiable)

- Geometries: (i) flow past cylinder (FPC), `flow_past_cylinder_sponge.yaml` lineage; (ii) flow past cape, `flow_past_cape.yaml` lineage, with `B: 0.0` (no beta-plane on this branch).
- Reynolds regime (FPC, based on D): Re ∈ [2200, 5600], Re_mid = 3900, Re_amp = 1700.
- Implementation: Reynolds is modulated through the inlet velocity ONLY. Viscosity is fixed:
  - nu = U_mid * D / Re_mid = 2.0 * 1.256637 / 3900 = 6.4443e-4. In YAML, write it with an explicit decimal mantissa (`6.4443e-4`) per the PyYAML rule in §1.4.
  - U(t) = nu * Re(t) / D = 5.1282e-4 * Re(t). Check: U(3900) = 2.0000. Range: U ∈ [1.1282, 2.8718].
  - Never modulate nu. Never change nu between cases.
- Cape uses the SAME nu and the SAME U(t) signals. Its Reynolds number based on L_cape = 1 is then Re_cape(t) = U(t)/nu ∈ [1751, 4456], mid 3103. Report cape Re with its own length scale; do not call it "Re 3900".
- Strouhal reference (context only): St ≈ 0.21 assumed approximately constant across [2200, 5600] (upper subcritical regime).
- Shedding timescale used for ALL modulation parameters:
  - T_shed_mid = D / (St * U_mid) = 1.256637 / (0.21 * 2.0) = 2.9920.

## 3. modulation.py — you author this (Phase A)

Create `modulation.py` in the training/tools area of the branch (colocate with the other pipeline scripts; mirror their CLI style). Spec:

### 3.1 Signal definitions

All signals: Re(t) = Re_mid for 0 <= t < T_wait; for t >= T_wait define tau = t - T_wait and apply the modulation. All signals except telegraph must be continuous at t = T_wait (they all start from Re_mid; sine starts at phase 0, OU is initialized at Re_mid, ramp starts at Re_mid). Telegraph jumps by construction; that is accepted.

First-wave cases (exactly these five; parameters derived from T_shed_mid = 2.9920):

| ID | Signal | Parameters |
|----|--------|------------|
| MOD-const | Constant | Re(t) = 3900 for all t |
| MOD-sine | Sinusoidal | Re = 3900 + 1700*sin(2*pi*tau/P), P = 5.0*T_shed_mid = 14.960 |
| MOD-ramp | Single ramp | Re: 3900 -> 5600 linearly over tau in [0, 15.0], then hold at 5600 |
| MOD-ou | Ornstein-Uhlenbeck | Discrete update Re_{n+1} = 3900 + rho*(Re_n - 3900) + sqrt(1-rho^2)*sigma*z_n, z_n ~ N(0,1), rho = exp(-dt/tau_OU), tau_OU = 5.0*T_shed_mid = 14.960, sigma = 0.2*Re_amp = 340, hard-clipped to [2200, 5600], seed = 20260707 |
| MOD-tel | Random telegraph | Re in {2200, 5600}, exponentially distributed dwell times, mean dwell tau_dwell = 4.0*T_shed_mid = 11.968, initial state Re_max = 5600, seed = 20260707 |

### 3.2 Interface and outputs

- CLI: `python modulation.py --signal {const,sine,ramp,ou,telegraph} --dt DT --T T --t-wait TWAIT --out U_of_t.npz [--seed N]`.
- Output npz contains: `t` (shape [N+1], t_n = n*dt from 0 to T inclusive), `Re` (same shape), `U` (= 5.1282e-4 * Re), plus a `meta` dict (all parameters, seed, git SHA).
- The table must be generated at the SOLVER dt of the run that consumes it (no interpolation at runtime; direct index lookup U[n] at step n). If a convergence run uses a different dt, regenerate the table at that dt with the same seed — the OU update above is dt-consistent by construction (rho and sqrt(1-rho^2) scaling), and telegraph dwell draws depend only on the seed sequence; verify by plotting the two tables overlaid and include the plot in the Gate 1 report.
- Also emit a per-case PNG: Re(t) and T_shed(t) = D/(St*U(t)) on twin axes, with T_wait marked.

### 3.3 Solver hook

Minimal-diff change to the BC path: `Flow.const_x_flow` (bc.py) currently sets `state.uh[...,0,0] = inlet_velocity` with a static scalar from config. Add an optional config key `bc.inlet_table: <path to U_of_t.npz>`; when present, load once at startup, and at step n set the inlet to `U[n]`. When absent, behavior is bit-identical to current (this is mandatory — verify with a short regression run diffed against an unmodified checkout, report max|Δω| which must be exactly 0.0).

Keep the diff surgical: no refactors, no renames, no drive-by cleanups.

### 3.4 Smoke tests (Gate 1)

Before any production submission, run on 512^2, T = 15, dt = 2.5e-4, FPC geometry, float64:
1. MOD-const via inlet_table (constant table) vs. legacy static inlet: max|Δω| = 0.0 required.
2. MOD-sine and MOD-ou: run to completion, no NaN, plot energy and enstrophy traces and the realized inlet history read back from the run.
3. dt-consistency check of §3.2 (OU/telegraph tables at dt = 2.5e-4 vs 1.25e-4, same seed, overlay plot).

Email the Gate 1 report ([QG][GATE1][SGS-CLOSURE]) with all plots and the bc.py diff. DO NOT proceed to Phase B before Sanaa's explicit approval.

## 4. Production ensemble (Phase B — after Gate 1 approval only)

### 4.1 Case matrix

10 runs = 5 modulations x 2 geometries.

| | FPC | Cape |
|---|---|---|
| Grid (fine/"FR") | 2048 x 2048 | 1024 x 1024 |
| Lx = Ly | 8*pi (unchanged) | 8*pi (unchanged) |
| dt | 2.5e-4 | 2.5e-4 (override the YAML's 1.0e-4; verify stability in a T=15 smoke first) |
| T | 120 | 120 |
| T_wait (= transient discard) | 30 | 30 |
| nu | 6.4443e-4 | 6.4443e-4 |
| B (beta) | n/a | 0.0 |
| Brinkman penalty, sponge | keep the YAML's eta = factor*dt convention for production runs; record the resulting PHYSICAL eta values in the run summary | same |
| save_rate | 1000 steps (snapshot every 0.25 t.u. -> 481 snapshots) | 1000 |
| IC | YAML default (randn, energy 0 / paper settings) | YAML default |
| dtype | float64 | float64 |

Timescale sanity (report these in the submission email): usable window = 90 t.u.; sine periods = 6.0; OU N_eff = 90/14.96 = 6.0; telegraph N_eff = 2*90/11.968 = 15.0; T_wait = 10.0 shedding periods at Re_mid.

### 4.2 Submission mechanics

- One qsub per case, `-q ibgpu.q -l gpu=1` ONLY. NEVER `-q ibamd.q`, NEVER `-l h_vmem=...`.
- `hydra.run.dir=outputs/SGS_closure_ensemble/<GEOM>-<MODID>` with GEOM in {FPC, CAPE}, MODID in {const, sine, ramp, ou, tel}.
- Job names: `sgs_<geom>_<modid>`.
- Follow the structure of `submit_ensemble.sh` / `submit_pi_ff_sweep.sh` (existence checks, skip logic, monitoring hints in the echo footer).
- Each run's directory must contain: `DNS_FR.npz`, the consumed `U_of_t.npz`, the resolved config, and the job log.
- Storage estimate to verify against quota beforehand: FPC omega float64 = 2048^2*8 B = 33.6 MB/snapshot x 481 ≈ 16 GB/run x 5 = 81 GB; cape ≈ 4 GB/run x 5 = 20 GB. Total ≈ 100 GB plus Π_FF outputs.

### 4.3 Per-run report ([QG][RUN][SGS-CLOSURE], one email per completed run)

- Walltime, steps/s, device.
- max CFL number encountered (advective; compute u_max per save and report the max over the run).
- Energy and enstrophy time series plot; realized inlet U(t) read back from the state (not from the table) overlaid on the prescribed table.
- Snapshot count, file sizes, any warnings, NaN check (must be none).
- Two vorticity snapshot images: t = 30 (end of transient) and t = 120.

## 5. Convergence tier (Phase C — runs in parallel with Phase B)

Purpose: put a number on the self-consistency of the FR "truth". FPC geometry, MOD-const (Re = 3900) only.

### 5.1 Shared IC (mandatory protocol)

1. From the FPC MOD-const production run (or a dedicated 2048^2 run to t = 30 if you need to start earlier), extract a restart IC at t = 30 using the scenario's `extract_restart_ic.py`.
2. Transfer to other grids SPECTRALLY: 2048 -> 1024 by spectral truncation, 2048 -> 4096 by spectral zero-padding. If no such utility exists in the repo, author a minimal `spectral_regrid.py` (rfft layout, verify round-trip 2048 -> 4096 -> 2048 recovers the field to machine precision; include that number in the report).
3. ALL convergence runs start from this shared developed-flow IC. Never from t = 0.

### 5.2 Fixed-physics rule (mandatory)

For ALL convergence runs, the Brinkman penalty eta and the sponge eta are held at FIXED PHYSICAL VALUES: compute them once from the baseline (dt = 2.5e-4) YAML convention and override explicitly in every convergence job. They must NOT scale with dt or grid.

### 5.3 Run matrix

All: T = 60 from the shared IC (i.e., t in [30, 90] in production time), float64, MOD-const.

- Grid study at fixed dt = 1.25e-4: N in {1024, 2048, 4096}.
- dt study at fixed N = 2048: dt in {5.0e-4, 2.5e-4, 1.25e-4}.

(2048, 1.25e-4) is shared between the two studies; 7 runs total. The 4096 run is the walltime risk — report its projected finish time within 12 h of submission.

### 5.4 Analysis and report ([QG][CONV][SGS-CLOSURE])

The flow is chaotic; trajectories diverge. Therefore two comparison horizons:
1. Short-horizon field error: relative L2 and Linf of omega vs. the finest member of the study, at t - 30 in {1, 3, 6} (i.e., up to ~2 shedding periods). Table: error vs N (grid study) and error vs dt (dt study).
2. Long-horizon statistics: time-averaged energy and enstrophy over t in [60, 90], and energy spectra E(k) averaged over the same window, overlaid across members.

Do NOT compute observed convergence orders from the chaotic long-horizon fields; orders may be estimated from the short-horizon table only, and only if the errors are monotone. State clearly if they are not.

## 6. A priori Π_FF sweep (Phase D — after each production run lands; do not wait for all 10)

1. If snapshot loading is memory-limited, first run `prepare_npz_for_mmap.py` on each `DNS_FR.npz` (follow its existing job-script pattern).
2. For every completed production run, compute Π_FF with the EXISTING `compute_pi_ff.py` (do not modify it) at scale in {2, 4, 8}, alpha = 1.5, GPU:
   - FPC: 2048 -> {1024, 512, 256}. Cape: 1024 -> {512, 256, 128}.
   - Clone the `submit_pi_ff_sweep.sh` pattern: one qsub per (case, scale), `-q ibgpu.q -l gpu=1`, existence checks, skip-if-done.
   - Name outputs so scales coexist in one run dir: `DNS_LES_s2.npz`, `DNS_LES_s4.npz`, `DNS_LES_s8.npz` — pass through whatever output-naming mechanism `compute_pi_ff.py` exposes; if it hard-codes `{name}_LES.npz`, use distinct `--name` values or per-scale subdirectories rather than editing the script, and document the choice.
3. Per-case report ([QG][PIFF][SGS-CLOSURE]): the summary YAMLs, Π_FF magnitude statistics per scale, and the diagnostic videos for scale 2. Note: the EXTENDED a priori analysis (Pearson correlations vs. resolved-field predictors, PDFs, Π_FF spectra, time-resolved correlation along the non-stationary trajectory) uses code that Sanaa will deliver; your deliverable here is the complete, correctly named set of `DNS_LES_s{2,4,8}.npz` files, ready for it.

## 7. Diagnostics integration (Phase E — code arrives from Sanaa)

`diagnostics_wake.py` (Cd(t), Cl(t) from the Brinkman force integral, Strouhal via spectral analysis of Cl, pressure and Cp via the pressure Poisson solve, wake profiles, recirculation length, spectra) is being authored by Sanaa and will be handed to you. When it arrives:
- Run it on every production run and every convergence member as instructed in its header.
- Do not modify it. Report bugs; do not patch.
- Final deliverable of this branch is the group-meeting data package: one directory per case containing diagnostics outputs, Π_FF outputs at all scales, and the per-run plots from §4.3.

## 8. Standing rules (apply to every phase)

- You are the sole code author on this branch except for files explicitly marked as delivered by Sanaa. Runner/summarizer subagents execute and report; they never edit code.
- Email subject convention: `[QG][GATE1|RUN|CONV|PIFF|ISSUE|WEEKLY][SGS-CLOSURE] ...`.
- Any deviation from this brief, any instability, any NaN, any job kill: stop the affected phase and send [QG][ISSUE][SGS-CLOSURE] immediately with logs. Do not self-remediate by changing physics parameters.
- Never delete or overwrite `DNS_FR.npz` files. Π_FF recomputes are cheap; FR runs are not.
- Gates requiring Sanaa's explicit approval before proceeding: Gate 1 (§3.4) before Phase B; nothing else blocks, but the first completed production run's report must be sent before the remaining 8 are submitted (submit FPC-const and CAPE-const first, then the rest after those two reports).
