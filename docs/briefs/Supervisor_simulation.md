# [QG][THEORY][SGS-CLOSURE] Supervisor_simulation.md — sampling, decorrelation, and resolution theory for the SGS-closure ensemble

Status: binding theoretical reference for exp/sgs-closure. All audit code must compute the
empirical counterparts of the quantities below and compare against these estimates,
reporting (theory, measured, ratio) tables. Derivations were done and approved by Sanaa;
do not re-derive, do implement faithfully. FPC (cylinder) scope only for now.

## 1. Regime recap

Re(t) in [2200, 5600] via inlet velocity, Re_mid = 3900, Re_amp = 1700.
nu = 6.4443e-4 fixed; U(t) = 5.1282e-4 * Re(t), U_mid = 2.0; D = 1.256637; St_ref ~ 0.21.
Fine grid 2048^2, Lx = 8*pi, dx_FR = 0.012272; dt = 2.5e-4; T = 120; T_wait = 30; usable window T_u = 90.
Shedding: T_sh(Re) = D^2/(St nu Re): T_sh(2200) = 5.304, T_sh(3900) = 2.992, T_sh(5600) = 2.084. f_sh in [0.189, 0.480].

## 2. Timescale inventory

Eulerian decorrelation at a fixed wake point, sweeping argument: a structure of correlation
length l advected at U_c ~ 0.85 U decorrelates in tau_E ~ l / U_c.

- Coherent/wake scale l = D: tau_E_wake = D/(0.85 U) in [0.52, 1.31]; 0.74 at U_mid.
- Filter scale l = Delta_s = s * dx_FR: Delta_{2,4,8} = {0.0245, 0.0491, 0.0982};
  tau_E_s = Delta_s/(0.85 U_mid) = {0.014, 0.029, 0.058}.
- Forcing: tau_OU = 14.96; telegraph tau_corr = tau_dwell/2 = 5.98; sine P = 14.96.
- Domain flush Lx/U in [8.8, 22].

Hierarchy: tau_E_s << T_sh <~ few * tau_E_wake << tau_forcing << T_u.

## 3. Snapshot rate: two criteria, one binding

### 3.1 Nyquist (frequency identification)

f_Nyq = 1/(2 dt_save). At dt_save = 0.25: f_Nyq = 2.0 vs. lift line f_sh <= 0.480, drag
line 2 f_sh <= 0.960, third harmonic 1.44. All clear with margin; no aliasing of shedding
lines. Nyquist is NOT the binding constraint — and St is measured from the dense scalar
series anyway (n_pp ~ 830 there).

### 3.2 Phase-binning (waveform along the cycle) — binding

Phase-resolved fields (omega, Pi_FF vs. shedding phase) require n_pp = T_sh/dt_save >=
2*pi/delta_phi phase bins. Chosen resolution delta_phi = 45 deg (8 bins: the coarsest
binning at which the canonical shedding stages — upper roll-up, detachment, crossover,
lower roll-up — remain separable; standard floor in the phase-averaging literature). Hence

    dt_save <= T_sh_min / 8 = 2.084/8 = 0.260  =>  dt_save = 0.25 t.u.
    (save_rate = 1000 at dt = 2.5e-4). 0.5 fails (n_pp = 4.2).

The 8 is a CHOICE (45-deg bins), not physics; 16 bins would need dt_save = 0.13 and 2x
storage, and is not required because the phase-resolved analysis is diagnostic, not the
training target.

## 4. ML sample independence (why finer saving buys nothing)

Variance of a T-mean of a stationary signal with integral timescale tau_int scales as
2 tau_int / T, hence

    N_eff = T_u / (2 tau_int)  if dt_save << 2 tau_int   (set by T_u, not by rate)
    N_eff = T_u / dt_save      if dt_save >= 2 tau_int   (every sample independent).

At dt_save = 0.25 over T_u = 90:

| scale        | 2 tau_int | regime            | N_eff                |
|--------------|-----------|-------------------|----------------------|
| wake (l = D) | 1.48      | oversampled 5.9x  | 61 (rate-independent)|
| s = 8        | 0.116     | independent       | 360                  |
| s = 4        | 0.058     | independent       | 360                  |
| s = 2        | 0.029     | independent       | 360                  |

Consecutive-snapshot correlation exp(-0.25/tau): 0.71 at wake scale, 0.013 at s = 8. The
closure input is LOCAL (small receptive field), so the learning-relevant decorrelation is
the fast, filter-scale one: snapshots 0.25 apart are statistically fresh where it matters.
Halving dt_save adds correlated copies of the slow coherent structures (which the model
must not memorize) while doubling storage.

Spatial independence: Pi-active area A_wake ~ 3D x 0.7 Lx ~ 66; independent patches per
snapshot ~ A_wake/l^2 in {66/Delta_4^2, 66/D^2} = {2.7e4, 42}. Per run: ~1e7 patch-samples
at filter scale, ~2.6e3 at coherent scale; x5 runs (cylinder ensemble). SVGP inducing
budgets saturate far below either.

Regime diversity (the axis storage cannot buy): N_regimes ~ T_u/tau_OU = 6 per OU run;
increased only by more/longer runs. Accepted bottleneck for Phase 1.

## 5. Commensurability at MOD-const

T_sh(3900)/0.25 = 11.97, within 0.3% of integer: phase advance ~0.25% of a cycle per
period, so over ~30 usable periods snapshots cluster in 12 nearly frozen phase bins
(evenly spaced — phase-AVERAGED statistics unaffected, continuous phase coverage lost).
Modulated runs sweep T_sh(t) and break commensurability automatically. Fix, MOD-const
runs only: dt_save = 0.27 (save_rate = 1080): 2.992/0.27 = 11.08, ~29 deg/period phase
advance, full wheel in ~12 periods. Zero cost.

## 6. Scalar recorder rate

Scalars every 10 steps = 2.5e-3 t.u.: ~830 samples per FASTEST shedding period; PSD and
Hilbert instantaneous-frequency estimates sampling-noise-free; < 8 MB/run. If the hook
costs > 2% walltime, propose rate 20 via FLAG before changing anything.

## 7. Boundary-layer and penalty-layer resolution (analysis, not a fix)

The wall layer is under-resolved by design; the deliverable is quantification.

### 7.1 A priori laminar estimate delta ~ D Re^{-1/2}

| Re   | delta  | pts/delta @1024^2 | @2048^2 | @4096^2 |
|------|--------|-------------------|---------|---------|
| 2200 | 0.0268 | 1.09              | 2.18    | 4.37    |
| 3900 | 0.0201 | 0.82              | 1.64    | 3.28    |
| 5600 | 0.0168 | 0.69              | 1.37    | 2.74    |

Classical wall resolution wants ~10 pts/delta; production grid sits at ~1.6 at Re_mid.
Stated openly in all reports.

### 7.2 Brinkman penalty layer

The penalization introduces its own layer delta_eta = sqrt(nu/eta_phys) and an
O(sqrt(nu/eta_phys)) model error (Angot et al. 1999). eta_phys MUST be derived from the
discrete update as implemented (the YAML's eta = factor*dt convention is not the physical
rate); document the derivation. Then classify: delta_eta <~ dx (sharp body) vs > dx
(mushy body), at each grid of the convergence tier.

### 7.3 Empirical audit (rides the convergence tier; no new runs)

On grids {1024, 2048, 4096}, MOD-const, from the shared-IC convergence runs: wall-normal
tangential-velocity profiles at theta = 60/90/120 deg; surface-vorticity distribution and
separation angle theta_sep (vorticity sign change on the ring r = R + 2 dx,
sensitivity-checked at R + {1,2,3} dx); effective diameter D_eff (radius where mean
tangential velocity crosses zero through the smeared mask). Deliverable claim template:
"wall layer under-resolved by ~Nx at 2048^2; theta_sep and shedding change by X% from
2048 to 4096; D_eff = D_nominal + Y dx" with N, X, Y measured. FLAG (not stop) if
|theta_sep(4096) - theta_sep(2048)| > 2 deg; ruling pre-committed: accept-and-report,
truth remains defined at the fine grid.

## 8. Audit specifications (empirical vs. theory; all thresholds pre-committed)

### Audit A — decorrelation and sampling (runs at the FPC-const gate, before the 4 modulated runs are submitted)

From FPC-const scalars + snapshots:

1. Autocorrelations rho_hat(tau) and integral timescales tau_int (trapezoid to first zero
   crossing) of: C_L(t); probe v(t) at x_c + 1D; Pi_FF(t) at 5 fixed wake points, per
   filter scale s in {2,4,8}. Compare to §2's tau_E_wake and tau_E_s: report
   (theory, measured, ratio).
2. Convection speed U_c: lag of peak cross-correlation between probes at x_c + 1D and
   x_c + 2D; compare to 0.85 U.
3. Spatial ACF of Pi_FF -> measured l_corr(s); replaces the [Delta_s, D] bracket in the
   §4 patch counts.
4. Phase-coverage histogram of snapshot times mod T_sh (validates the 0.27 fix on
   MOD-const).

DECISION RULE (pre-committed): if measured 2 tau_int(s=4) > 0.5, the 4 modulated runs
relax to dt_save = 0.5 with the phase-binning criterion re-checked against measured T_sh;
else 0.25 stands. Either way, report the recomputed N_eff table (§4) with measured values.

### Audit B — resolution (rides the convergence tier)

Exactly §7.3, plus the delta_eta classification of §7.2 per grid.

## 9. Precision policy

Solver runs in float64 (charter rule; guard hook enforces). STORAGE of all output
quantities (field snapshots, scalars, Pi_FF products) is float32: cast at write, never
compute in float32. Justification: spatial-closure targets and all diagnostics here are
O(1)–O(10) quantities, far above float32 epsilon; this halves the ensemble footprint
(~8 GB/run FPC). This is branch-specific and does NOT transfer to the temporal-closure
branch, whose O(dT^3) targets require float64 storage end to end.
