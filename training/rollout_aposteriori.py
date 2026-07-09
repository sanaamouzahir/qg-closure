"""
rollout_aposteriori.py -- the unified a-posteriori driver for the trained closure.

One driver, four arms from a SHARED developed-flow IC:

  truth    : RK4 at h = Delta_T / K  (fine reference -- rollout_timed_pareto's
             rollout_fine, IMPORTED: single-step RK4, no stencil bootstrap,
             F_phys PHYSICAL in the RK4 rhs, warmup excluded from timing,
             non-finite guard at checkpoints)
  bare     : AB2CN2 at Delta_T, no closure
  r3only   : AB2CN2 at Delta_T + ANALYTIC R3 pieces only (L^3 w implicit,
             L^2 N explicit) -- no NN. NB for low-nu members (kf4/b2 etc.,
             nu~1e-4, mu=0.02) these terms are ~1e-4 of the bracket (N-ddot
             carries it; verified vs the error-prop budget AND independently
             at IC kf4/820: predicted 4-step effect 3.2e-7 == observed
             bare-vs-r3only gap 3.0e-7) -- r3only ~= bare is PHYSICAL there,
             not a bug.
  r3anal   : AB2CN2 at Delta_T + FULL analytic R3 (+R4 with --r4): exact
             chain-rule Ndot/Nddot each step (rollout_perfect_closure's
             analytic_n_derivs_hat) through the IDENTICAL assembly as
             'closure' -- the analytic-LTE ceiling inside this driver and
             the blowup discriminator (r3anal stable + closure blown up =>
             NN-injected instability, not the scheme's)
  closure  : AB2CN2 at Delta_T + trained closure per the inference decomposition:
                 implicit : the -(coef/12) L^3 w term is folded into the IMEX
                            solve (evaluated at w_{n+1}; denominator gains
                            + (coef/12) L_hat^3),
                 explicit : analytic -(coef/12) L^2 N(w_n),
                 explicit : NN  -coef * f_NN,  f_NN = (1/12)(L Ndot - 5 Nddot)
                            from the trained [Ndot, Nddot, N3dot] heads
             with coef = Delta_T^3: the RK4 truth is the exact flow, so the
             target is the FULL Taylor defect -- NO (1 - 1/K^2) factor (that
             factor belongs to an AB2CN2-at-h/K truth only). Applied
             identically in the r3only arm; --r4 uses coef4 = Delta_T^4.
             K keeps its role as the truth refinement factor ONLY.
             Reference implementation for both the scheme and the
             coefficient: rollout_timed_pareto.py.

K RULE (accuracy runs): the truth must stand in for the ANALYTIC flow --
we deliberately do NOT model the RK4 LTE in the closure (the T5 formula
needs a much deeper network), so the RK4 truth error must be driven below
the closure residual by brute force: set K so h_fine = Delta_T/K <= ~1e-5
(smoke convention; e.g. K=1500 at Delta_T=1.5e-2). K=O(20) leaves the
truth's own LTE in the comparison and is NOT acceptable for accuracy
tables; small K remains fine for --no-truth stability runs where the
truth arm is skipped anyway.

FFT budget per coarse step (ported from rollout_timed_pareto): bare/r3only = 5
(minimal N eval, N kept SPECTRAL, no round trips); closure = 8 = the same 5
+ 1 psi iFFT (INPUT-STACK INFRASTRUCTURE -- bare only avoids it by never
needing psi as a field) + 2 NN-output FFTs (Ndot, Nddot -> spectral correction
assembly; the only genuine NN overhead). The cond_local NN itself is conv-only
(ZERO transforms): its sigma-hat context is computed from the stepper's OWN
spectral qh_curr/qh_minus (shell reduction only, zero extra FFTs) and passed
via cond_feats. E/Z scalar series are Parseval reductions in spectral space
(zero FFTs; cross-checked against the physical formulas at the IC).

Adapted from rollout_multistep_comparison.py (error-accumulation framing),
rollout_load_truth_compare.py (shared-truth reuse) and rollout_perfect_closure.py
(bracket assembly); physics primitives are IMPORTED from rollout_timed_pareto so
the operators are bit-identical to the validated legs.

IC options (mutually exclusive):
  --root-dir + --ic-index : history stack read from a packed sweep sample
                            (7 lags at the sweep's Delta_T -- rollout Delta_T
                            must equal the sweep's for the stencil to be valid).
  --restart-ic <npy>      : a developed-flow omega field from
                            extract_restart_ic.py; the S-deep history is built
                            by ULTRAFINE RK4 forward integration, recording a
                            mark every Delta_T (rule 5: never start from t=0 --
                            pass a developed-flow restart).

Horizon: --horizon-turnovers (default 10) eddy turnovers, tau_eddy = 1/omega_rms
of the IC unless --tau-eddy overrides. --n-steps overrides both.

Outputs (all under --out-dir, tagged):
  rollout_apost_<tag>.npz  : checkpointed FIELDS per arm (for spectra), times,
                             per-step E/Z scalar series, CFL log
  rollout_apost_<tag>.json : config + rel-L2 tables + blowup verdicts
  rollout_apost_<tag>.csv  : t, relL2_bare, relL2_r3only, relL2_closure
  sigma_hat_<tag>_<arm>.csv: sigma-hat(kappa) at closure-arm checkpoints
                             (--log-sigma, default on; zero extra FFTs)
  lte_<tag>_<arm>.csv      : --track-lte; full analytic LTE per-term rms at
                             each arm's own state per checkpoint + (closure
                             arms) NN-vs-analytic rel-L2 per head and the
                             injected error coef*rms(f_NN - f_anal)
  apost_refs_<tag>.npz     : truth stack for reuse (--save-refs/--load-refs)
  ..._pareto.png           : bare-dt-sweep cost/accuracy front (--pareto)

A/B and comparison flags: --freeze-sigma (cond_local static-recalibration
leg), --ckpt2 (second checkpoint as arm 'closure2' through the identical
code path -- the cond-vs-control comparison), --diag (per-term RMS at IC),
--profile-step (cuda-event per-block breakdown, pareto port).

Brinkman/sponge note: this driver covers the periodic FRC/DEC scenarios (no
mask). When the flow-past-obstacle scenario is added, the penalty/sponge eta
must be passed as FIXED PHYSICAL values across dt (charter 5.2 pattern); the
driver refuses masked configs rather than silently mis-scaling eta.

Usage (tomorrow, after deriv7_cond_local lands):
  python rollout_aposteriori.py \
      --root-dir data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
      --ckpt data/ensemble_N5_7lag/training_runs/deriv7_cond_local/best.pt \
      --ic-index 0 --K 100 --horizon-turnovers 10 --device cuda \
      --out-dir diagnostics/Results/apost_b2_5em3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _find_training_dir():
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        if (ancestor / 'dataset.py').exists():
            return ancestor
    return here


sys.path.insert(0, str(_find_training_dir()))

# physics primitives -- bit-identical to the validated pareto/perfect legs
from rollout_timed_pareto import (                                  # noqa: E402
    N_spectral, N_spectral_fields, _dealias_mul, build_L_hat, rk4_step,
    rollout_fine, build_forcing, psi_from_omega, assemble_inputs, _sync,
    _StepProfiler, _NOPROF)
from model_deriv_closure import build_model                         # noqa: E402
from cond_grad import sigma_hat_spec                                # noqa: E402
from rollout_perfect_closure import analytic_n_derivs_hat           # noqa: E402


# --------------------------------------------------------------------------- #
# model loading (deriv family: cheap_deriv / cond_local / cond_deriv)          #
# --------------------------------------------------------------------------- #

def load_deriv_model(ckpt_path: Path, manifest, dt_rollout, device,
                     nn_float64=True):
    """Load a train_deriv.py checkpoint (best.pt: {'model': sd, 'config': ...}).

    The deriv trainers use the frozen W_unit/dt^k TimeFD path with dt passed
    per-forward, so dt portability needs NO weight rescaling here -- the driver
    passes explicit dt/dx/dy tensors at every forward (exactly the training
    call signature). Construction-time dt/dx/dy only seed the fallbacks.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get('model', state.get('model_state', state.get('state_dict', state)))
    cfg = state.get('config', {})
    name = cfg.get('model', 'cheap_deriv')
    n_snap = int(cfg.get('n_snapshots', 7))
    Nx, Ny = int(manifest['Nx']), int(manifest['Ny'])
    dx = float(manifest['Lx']) / Nx
    dy = float(manifest['Ly']) / Ny
    gk = int(cfg.get('grad_kernel', 15))
    for k, v in sd.items():
        if k.endswith('grad.wx'):
            gk = int(v.shape[-1])
    model = build_model(name, in_channels=2 * n_snap,
                        out_orders=int(cfg.get('out_orders', 3)),
                        n_time=n_snap, grad_kernel=gk,
                        dt=dt_rollout, dx=dx, dy=dy,
                        physics_init=not cfg.get('no_physics_init', False))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise SystemExit(f'[load] UNEXPECTED ckpt keys (wrong model?): {unexpected}')
    if missing:
        raise SystemExit(f'[load] MISSING model keys (ckpt/model mismatch): {missing}')
    mdtype = torch.float64 if nn_float64 else torch.float32
    model.to(device=device, dtype=mdtype).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f'[load] model={name}  n_snapshots={n_snap}  grad_kernel={gk}  '
          f'params={n_par}  nn dtype={mdtype}  ckpt={ckpt_path}')
    return model, name, n_snap


# --------------------------------------------------------------------------- #
# scalar diagnostics                                                           #
# --------------------------------------------------------------------------- #

def _parseval_weight(qh, Nx):
    """Hermitian double-count weights for the rfft half-plane: interior kx
    columns stand for +-kx pairs (weight 2); kx=0 and (even Nx) Nyquist are
    self-conjugate (weight 1). With the solver's norm='forward' convention,
    <f^2> == sum(w |fh|^2)."""
    w = torch.full((qh.shape[-1],), 2.0, dtype=torch.float64, device=qh.device)
    w[0] = 1.0
    if Nx % 2 == 0:
        w[-1] = 1.0
    return w


def scalars_from_qh(qh, derivative, w_par):
    """(E, Z) domain means from spectral omega -- Parseval reductions, ZERO
    FFTs (was 2 iFFTs/step). Z = 0.5<w^2> = 0.5 sum w_par |qh|^2;
    E = 0.5<|grad psi|^2> = -0.5<psi w> = 0.5 sum w_par (-inv_lap) |qh|^2.
    Cross-checked against the physical-space formulas at the IC in main()."""
    il = derivative.inv_laplacian
    if torch.is_complex(il):
        il = il.real
    e = qh.real ** 2 + qh.imag ** 2
    Z = 0.5 * float((e * w_par).sum())
    E = 0.5 * float((e * (-il) * w_par).sum())
    return E, Z


def cfl_from_qh(qh, derivative, dt, dx, dy):
    from qg.solver.opt.basis import to_physical
    psih = derivative.inv_laplacian * qh
    u = to_physical(-1 * derivative.dy * psih)
    v = to_physical(+1 * derivative.dx * psih)
    return float(u.abs().max()) * dt / dx + float(v.abs().max()) * dt / dy


def rel_l2(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-30))


# --------------------------------------------------------------------------- #
# rollout arms                                                                 #
# --------------------------------------------------------------------------- #

def run_arm(arm, omega_stack, psi_stack, Delta_T, n_steps, cp_steps,
            derivative, L_hat, F_hat, device, model=None, input_fields=None,
            dealias_nn=True, include_r4=False, blowup_factor=10.0,
            scalars_every=1, freeze_sigma=False, sigma_log=None,
            lte_log=None, profile_step=0,
            nn_kcut=None, nn_gamma=1.0, nn_clip=None, drop_nddot=False,
            nn_project_radius=None):
    """One arm of the comparison. arm in {'bare','r3only','r3anal',
    'closure','closure2'} ('closure2' = a second checkpoint through the
    identical code path, e.g. the control vs the conditioned model;
    'r3anal' = full analytic R3 [+R4 with --r4] via the exact chain-rule
    Ndot/Nddot each step, NO NN, through the SAME assembly as 'closure' --
    the analytic-LTE ceiling and the blowup discriminator: if r3anal is
    stable where closure blows up, the instability is NN-injected, not
    the scheme's).

    Returns dict: fields {step: ndarray}, scalars (t, E, Z arrays),
    cfl_max, blowup (None or step), walltime (3-step warmup EXCLUDED).

    FFT budget per step: bare/r3only 5 (N_spectral, N kept spectral),
    closure 8 (N_spectral_fields hands back omega/psi physical for the
    stencil + checkpoint dump, + 2 NN-output FFTs). E/Z are Parseval
    reductions (zero FFTs); the cond_local sigma-hat context is a shell
    reduction of the stepper's own spectral states (zero FFTs).

    freeze_sigma : cond_local only -- compute the sigma-hat context ONCE at
        t=0 and reuse it every step (the static-recalibration A/B leg).
    sigma_log : optional list; at every checkpoint step the FULL sigma-hat
        shell vector from the stepper's spectral states is appended as
        (step, t, sig ndarray) -- zero extra FFTs (r2-drift measurement).
        NOTE: the shell reduction is non-zero COMPUTE inside the timed loop
        (at checkpoint steps only, ~sub-percent of the arm walltime);
        headline benchmarks (3b) run with sigma_log=None.
    nn_kcut : float or None -- remediation R1 (anomaly-playbook ladder): keep
        f_NN only for |k| <= nn_kcut * (2/3-dealias cut). The NN's high-k
        tail is its noisiest band and seeds the history-contamination loop.
    nn_gamma : float -- remediation R2: under-relax, rhs -= coef*gamma*f_NN
        (bounded accuracy factor traded for feedback margin). Default 1.0.
    nn_clip : float or None -- remediation R3: pointwise-clip the PHYSICAL
        f_NN at nn_clip * rms(f_NN) (kills outlier pixels that seed CFL
        spikes; +2 FFTs/step), then re-dealias.
        All three act on the R3 f_NN of closure arms ONLY (r3anal/r3only
        untouched; --r4 NN terms untouched). The --track-lte rms_inj stays
        the RAW (un-remediated) NN error by design.
    drop_nddot : ablation arm -- assemble the R3 correction WITHOUT the
        -5*Nddot contribution: f = (1/12) L*Ndot only (all other terms --
        implicit L^3 w, explicit L^2 N, L*Ndot -- kept). Applied to BOTH the
        NN (closure/closure2) and the analytic (r3anal) sources so the arms
        stay comparable; --r4 terms (when enabled) keep their own Nddot use.
    nn_project_radius : float or None -- Sanaa 2026-07-09 ruling (solver mask
        untouched, remediate on the correction alone): project the R3
        correction onto the TRUE alias-safe radius |k| <= factor *
        min(kx_max, ky_max) (factor 2/3 -> mode radius ~170.67 at 512^2).
        The solver's own dealias mask is RADIAL at sqrt(2)*(2/3)*k_max
        (mode ~241.4), leaving the 170.7-241.4 annulus alias-contaminated;
        this keeps the injected correction out of that annulus. Applies to
        f_NN of closure/closure2 AND, when the r3anal arm runs with it, to
        f_anal (the annulus-isolation reference). Distinct from nn_kcut
        (remediation ladder, closure arms only, alpha x N//3 convention).
        at the arm's CURRENT state is computed (chain-rule Ndot/Nddot,
        ~50 FFTs/checkpoint) and its per-term rms appended; closure arms
        additionally log the NN-vs-analytic rel-L2 per head and the
        injected per-step correction error rms(coef*(f_NN - f_anal)) --
        the noise-feedback track. NON-ZERO compute inside the timed loop:
        headline benchmarks run with lte_log=None.
    profile_step : cuda only; after the clean timed loop, step this many
        more times WITH cuda-event marks and print the per-block breakdown
        (ported from rollout_timed_pareto._StepProfiler).
    """
    from qg.solver.opt.basis import to_spectral, to_physical
    # truth = RK4 (exact flow) -> the target is the FULL Taylor defect:
    # coef = DT^3, coef4 = DT^4, NO (1-1/K^n) factors (those belong to an
    # AB2CN2-at-h/K truth). Applied identically to r3only and closure.
    coef = Delta_T ** 3
    coef4 = Delta_T ** 4
    c12 = coef / 12.0
    denom_bare = 1.0 - 0.5 * Delta_T * L_hat
    # precompute L_hat powers once (step-invariant; pure hoist)
    L2 = L_hat ** 2
    L3 = L2 * L_hat
    L4 = L2 * L2
    # implicit fold: the -(c12) L^3 w term evaluated at w_{n+1} moves to the LHS
    denom_clos = denom_bare + c12 * L3
    is_clos = arm.startswith('closure')
    is_anal = arm == 'r3anal'
    with_closure = is_clos or is_anal or arm == 'r3only'
    denom = denom_clos if with_closure else denom_bare
    Nyg, Nxg = omega_stack[0].shape[-2], omega_stack[0].shape[-1]
    is_cond = is_clos and hasattr(model, 'context_feats_from_spectral')
    kcut_mask = None
    if is_clos and nn_kcut is not None:
        # mode-index shells (isotropic square grids): dealias keeps
        # |k_index| <= N//3; R1 keeps kappa <= alpha * that cut.
        kx2 = (-(derivative.dx ** 2)).real
        ky2 = (-(derivative.dy ** 2)).real
        kmag = torch.sqrt(kx2 + ky2)
        dk = 2.0 * np.pi / _LX
        kcut_mask = (kmag <= float(nn_kcut) * (Nxg // 3) * dk).to(torch.float64)
    proj_mask = None
    if (is_clos or is_anal) and nn_project_radius is not None:
        kx2p = (-(derivative.dx ** 2)).real
        ky2p = (-(derivative.dy ** 2)).real
        kmagp = torch.sqrt(kx2p + ky2p)
        k_safe = float(nn_project_radius) * min(float(kx2p.max()) ** 0.5,
                                                float(ky2p.max()) ** 0.5)
        proj_mask = (kmagp <= k_safe).to(torch.float64)
    dt_v = torch.full((1,), Delta_T, device=device, dtype=torch.float64)
    dx_v = torch.full((1,), _DX, device=device, dtype=torch.float64)
    dy_v = torch.full((1,), _DY, device=device, dtype=torch.float64)
    frozen_feats = [None]        # filled at t=0 when freeze_sigma

    def one_step(qh_curr, qh_minus, Nh_curr, Nh_minus, om_hist, ps_hist,
                 prof=_NOPROF):
        prof.mark('bare')
        AB2_Nh = 1.5 * Nh_curr - 0.5 * Nh_minus
        rhs = qh_curr + Delta_T * (0.5 * L_hat * qh_curr + AB2_Nh)
        if with_closure:
            prof.mark('e_anal')
            rhs = rhs - c12 * (L2 * Nh_curr)                 # explicit analytic L^2 N
        if is_anal:
            # full analytic R3 (+R4 with --r4): exact chain-rule Ndot/Nddot
            # at w_n, assembled EXACTLY like the NN arm (same implicit fold,
            # same coef, same dealias placement) -- only the [Ndot, Nddot]
            # source differs. rollout_perfect_closure.analytic_n_derivs_hat
            # is the validated chain-rule builder. NB: mirrors THIS driver's
            # closure arm, which folds L^3 implicitly and divides the whole
            # corrected rhs by denom_clos; rollout_perfect_closure keeps L^3
            # explicit at w_n -- the two differ at O(h^4) BY DESIGN, so an
            # r3anal-vs-rollout_perfect numeric gap at that order is not a bug.
            prof.mark('anal_derivs')
            Nd = analytic_n_derivs_hat(qh_curr, derivative, L_hat, F_hat,
                                       max_order=3 if include_r4 else 2)
            f_anal = ((1.0 / 12.0) * (L_hat * Nd[1]) if drop_nddot
                      else (1.0 / 12.0) * (L_hat * Nd[1] - 5.0 * Nd[2]))
            prof.mark('dealias')
            if dealias_nn:
                f_anal = _dealias_mul(f_anal, derivative)
            if proj_mask is not None:                # alias-safe radius
                f_anal = f_anal * proj_mask
            prof.mark('combine')
            rhs = rhs - coef * f_anal
            if include_r4:
                e_r4 = -coef4 * (1.0 / 24.0) * (2.0 * L4 * qh_curr
                                                + 2.0 * L3 * Nh_curr
                                                + 2.0 * L2 * Nd[1]
                                                - 4.0 * L_hat * Nd[2]
                                                + Nd[3])
                if dealias_nn:
                    e_r4 = _dealias_mul(e_r4, derivative)
                rhs = rhs + e_r4
        if is_clos:
            # feed the stack at float64 so the TimeFD differencing is
            # cancellation-clean (the model handles its own mixed precision);
            # dt/dx/dy passed explicitly = the exact training call signature.
            prof.mark('inputs')
            x = assemble_inputs(input_fields, om_hist, ps_hist,
                                torch.float64, device)
            kw = {}
            if is_cond:
                # sigma-hat context from the stepper's OWN spectral states --
                # shell reduction only, ZERO extra FFTs (the solver's
                # norm='forward' scaling cancels in the per-shell ratio).
                prof.mark('sigma')
                if freeze_sigma:
                    if frozen_feats[0] is None:
                        frozen_feats[0] = model.context_feats_from_spectral(
                            qh_curr, qh_curr - qh_minus, dt_v, _LX, _LY,
                            Nyg, Nxg)
                    kw['cond_feats'] = frozen_feats[0]
                else:
                    kw['cond_feats'] = model.context_feats_from_spectral(
                        qh_curr, qh_curr - qh_minus, dt_v, _LX, _LY, Nyg, Nxg)
            prof.mark('nn_conv')
            with torch.no_grad():
                yhat = model(x, dt=dt_v, dx=dx_v, dy=dy_v,
                             **kw).to(torch.float64)
            prof.mark('nn_fft')
            Ndot_h = to_spectral(yhat[:, 0:1][0])            # 1 FFT
            Nddot_h = to_spectral(yhat[:, 1:2][0])           # 1 FFT
            prof.mark('f_NN')
            f_nn = ((1.0 / 12.0) * (L_hat * Ndot_h) if drop_nddot
                    else (1.0 / 12.0) * (L_hat * Ndot_h - 5.0 * Nddot_h))
            if nn_clip is not None:                          # R3: +2 FFTs
                fp = to_physical(f_nn)
                s = float(nn_clip) * torch.sqrt((fp ** 2).mean())
                f_nn = to_spectral(fp.clamp(-s, s))
            prof.mark('dealias')
            if dealias_nn:
                f_nn = _dealias_mul(f_nn, derivative)
            if kcut_mask is not None:                        # R1
                f_nn = f_nn * kcut_mask
            if proj_mask is not None:                        # alias-safe radius
                f_nn = f_nn * proj_mask
            prof.mark('combine')
            rhs = rhs - coef * float(nn_gamma) * f_nn        # R2: gamma=1 default
            if include_r4:
                prof.mark('r4')
                N3dot_h = to_spectral(yhat[:, 2:3][0])       # +1 FFT (--r4 only)
                e_r4 = -coef4 * (1.0 / 24.0) * (2.0 * L4 * qh_curr
                                                + 2.0 * L3 * Nh_curr
                                                + 2.0 * L2 * Ndot_h
                                                - 4.0 * L_hat * Nddot_h
                                                + N3dot_h)
                prof.mark('dealias')
                if dealias_nn:
                    e_r4 = _dealias_mul(e_r4, derivative)
                prof.mark('combine')
                rhs = rhs + e_r4
        prof.mark('combine')
        qh_new = rhs / denom
        if is_clos:
            # the new N eval hands back omega/psi PHYSICAL -> reused for the
            # stencil and the checkpoint dump; neither is iFFT'd a second time.
            Nh_new, om_new, ps_new = N_spectral_fields(qh_new, derivative,
                                                       F_hat, prof=prof)
            return qh_new, Nh_new, om_new, ps_new
        return qh_new, N_spectral(qh_new, derivative, F_hat), None, None

    qh_curr = to_spectral(omega_stack[0])
    qh_minus = to_spectral(omega_stack[1])
    Nh_curr = N_spectral(qh_curr, derivative, F_hat)
    Nh_minus = N_spectral(qh_minus, derivative, F_hat)
    om = [s.clone() for s in omega_stack]
    ps = [s.clone() for s in psi_stack]
    w_par = _parseval_weight(qh_curr, Nxg)

    def log_sigma(step, qc, qm):
        if sigma_log is None or not (is_clos or is_anal):
            return
        sig, _ = sigma_hat_spec(qc, qc - qm, dt_v, _LX, _LY, Nyg, Nxg)
        sigma_log.append((step, step * Delta_T,
                          sig[0].detach().cpu().numpy().copy()))

    def log_lte(step, qc, qm, om_hist, ps_hist):
        """--track-lte: the FULL analytic LTE at the arm's CURRENT state.
        Every arm logs the R3 term-rms budget (the tau the closure should be
        removing); closure arms additionally log NN-vs-analytic rel-L2 per
        head + the injected per-step correction error coef*rms(f_NN-f_anal)
        (the noise-feedback track)."""
        if lte_log is None:
            return
        if not (torch.isfinite(qc.real).all() and torch.isfinite(qc.imag).all()):
            return
        from qg.solver.opt.basis import to_spectral, to_physical

        def _rms(spec):
            return float(torch.sqrt((to_physical(spec) ** 2).mean()))

        Nd = analytic_n_derivs_hat(qc, derivative, L_hat, F_hat, max_order=2)
        tau_h = -c12 * (L3 * qc + L2 * Nd[0] + L_hat * Nd[1] - 5.0 * Nd[2])
        row = dict(step=step, t=step * Delta_T,
                   rms_tau_full=_rms(tau_h),
                   rms_L3w=c12 * _rms(L3 * qc),
                   rms_L2N=c12 * _rms(L2 * Nd[0]),
                   rms_LNdot=c12 * _rms(L_hat * Nd[1]),
                   rms_5Nddot=c12 * 5.0 * _rms(Nd[2]))
        if is_clos:
            x = assemble_inputs(input_fields, om_hist, ps_hist,
                                torch.float64, device)
            kw = {}
            if is_cond:
                kw['cond_feats'] = (frozen_feats[0]
                                    if (freeze_sigma and frozen_feats[0]
                                        is not None)
                                    else model.context_feats_from_spectral(
                                        qc, qc - qm, dt_v, _LX, _LY,
                                        Nyg, Nxg))
            with torch.no_grad():
                yhat = model(x, dt=dt_v, dx=dx_v, dy=dy_v,
                             **kw).to(torch.float64)
            Ndh = to_spectral(yhat[:, 0:1][0])
            Nddh = to_spectral(yhat[:, 1:2][0])
            f_nn = (1.0 / 12.0) * (L_hat * Ndh - 5.0 * Nddh)
            f_an = (1.0 / 12.0) * (L_hat * Nd[1] - 5.0 * Nd[2])
            if dealias_nn:
                f_nn = _dealias_mul(f_nn, derivative)
                f_an = _dealias_mul(f_an, derivative)
            row['rel_Ndot_nn'] = _rms(Ndh - Nd[1]) / max(_rms(Nd[1]), 1e-30)
            row['rel_Nddot_nn'] = _rms(Nddh - Nd[2]) / max(_rms(Nd[2]), 1e-30)
            row['rel_fnn'] = _rms(f_nn - f_an) / max(_rms(f_an), 1e-30)
            row['rms_inj'] = coef * _rms(f_nn - f_an)
        lte_log.append(row)

    # warmup (untimed) -- exercises the same code path on cloned state,
    # then discards it (lazy-init / cuFFT-plan costs stay out of walltime)
    qc, qm = qh_curr.clone(), qh_minus.clone()
    Nc, Nm = Nh_curr.clone(), Nh_minus.clone()
    om_w, ps_w = list(om), list(ps)
    for _ in range(3):
        qn, Nn, on, pn = one_step(qc, qm, Nc, Nm, om_w, ps_w)
        if is_clos:
            om_w = [on] + om_w[:-1]
            ps_w = [pn] + ps_w[:-1]
        qm, qc = qc, qn
        Nm, Nc = Nc, Nn
    # NOTE: freeze_sigma context is (re)pinned from the REAL IC states below,
    # not from the warmup clones.
    frozen_feats[0] = None

    fields = {}
    cps = set(cp_steps)
    ts, Es, Zs = [], [], []
    cfl_max = 0.0
    blowup = None
    E0, Z0 = scalars_from_qh(qh_curr, derivative, w_par)
    if 0 in cps:
        fields[0] = to_physical(qh_curr)[0].cpu().numpy()
    ts.append(0.0); Es.append(E0); Zs.append(Z0)
    log_sigma(0, qh_curr, qh_minus)
    log_lte(0, qh_curr, qh_minus, om, ps)

    _sync(device); t0 = time.time()
    for s in range(1, n_steps + 1):
        qh_new, Nh_new, om_new, ps_new = one_step(qh_curr, qh_minus,
                                                  Nh_curr, Nh_minus, om, ps)
        qh_minus, qh_curr = qh_curr, qh_new
        Nh_minus, Nh_curr = Nh_curr, Nh_new
        if is_clos:
            om = [om_new] + om[:-1]        # newest-first history for the stencil
            ps = [ps_new] + ps[:-1]
        if s % scalars_every == 0 or s in cps or s == n_steps:
            E, Z = scalars_from_qh(qh_curr, derivative, w_par)
            ts.append(s * Delta_T); Es.append(E); Zs.append(Z)
            if (not np.isfinite(Z)) or Z > blowup_factor * max(Z0, 1e-30):
                blowup = s
                if np.isfinite(Z):        # CFL AT the failing step (sanity flag)
                    cfl_max = max(cfl_max, cfl_from_qh(qh_curr, derivative,
                                                       Delta_T, _DX, _DY))
                print(f'      [{arm}] BLOWUP at step {s} '
                      f'(Z={Z:.3e} vs Z0={Z0:.3e}  cfl={cfl_max:.3f}) '
                      f'-- stopping arm.')
                break
        if s in cps:
            log_sigma(s, qh_curr, qh_minus)
            log_lte(s, qh_curr, qh_minus, om, ps)
            arr = (om_new if is_clos
                   else to_physical(qh_curr))[0].cpu().numpy()
            fields[s] = arr
            if not np.isfinite(np.sqrt(np.mean(arr ** 2))):   # non-finite guard
                blowup = blowup if blowup is not None else s
                print(f'      [{arm}] non-finite field at step {s} '
                      f'-- stopping arm.')
                break
            cfl_max = max(cfl_max, cfl_from_qh(qh_curr, derivative,
                                               Delta_T, _DX, _DY))
    _sync(device)
    wall = time.time() - t0

    # ---- optional per-block breakdown (separate short window; the headline
    # walltime above is from the un-instrumented loop) -- pareto port ------ #
    if profile_step and is_clos and str(device).startswith('cuda') \
            and blowup is None:
        prof = _StepProfiler(device)
        for _ in range(int(profile_step)):
            prof.step_begin()
            h0 = time.perf_counter()
            qh_new, Nh_new, om_new, ps_new = one_step(
                qh_curr, qh_minus, Nh_curr, Nh_minus, om, ps, prof=prof)
            prof.host_ms += (time.perf_counter() - h0) * 1e3   # enqueue, no sync
            prof.step_end()                                    # syncs + reads events
            qh_minus, qh_curr = qh_curr, qh_new
            Nh_minus, Nh_curr = Nh_curr, Nh_new
            om = [om_new] + om[:-1]
            ps = [ps_new] + ps[:-1]
        n_fft = 8 + (1 if include_r4 else 0)
        prof.report(clean_ms_per_step=wall / max(n_steps, 1) * 1e3,
                    n_fft=n_fft)

    return dict(fields=fields, t=np.asarray(ts), E=np.asarray(Es),
                Z=np.asarray(Zs), cfl_max=cfl_max, blowup=blowup,
                walltime=wall)


def diag_term_rms(model, omega_stack, psi_stack, Delta_T, derivative, L_hat,
                  F_hat, device, input_fields, include_r4=False):
    """--diag: per-term RMS of the closure correction AT THE IC (ported from
    rollout_timed_pareto --diag). Splits the bracket into analytic (L^3 w,
    L^2 N) vs NN-predicted (L Ndot, 5 Nddot; R4 terms with --r4) mass so a
    wrong closure arm is a one-look diagnosis of WHICH term is mis-scaled.
    The NN-predicted fraction x the per-derivative val rel-L2 is the
    NN-limited closure error floor. Returns {term: rms} (also saved to json).
    """
    from qg.solver.opt.basis import to_spectral, to_physical
    L2 = L_hat ** 2
    L3 = L2 * L_hat
    L4 = L2 * L2
    cf = 1.0 / 12.0
    qh0 = to_spectral(omega_stack[0])
    qh1 = to_spectral(omega_stack[1])
    Nh0 = N_spectral(qh0, derivative, F_hat)
    dt_v = torch.full((1,), Delta_T, device=device, dtype=torch.float64)
    dx_v = torch.full((1,), _DX, device=device, dtype=torch.float64)
    dy_v = torch.full((1,), _DY, device=device, dtype=torch.float64)
    x = assemble_inputs(input_fields, omega_stack, psi_stack,
                        torch.float64, device)
    kw = {}
    if hasattr(model, 'context_feats_from_spectral'):
        Nyg, Nxg = omega_stack[0].shape[-2], omega_stack[0].shape[-1]
        kw['cond_feats'] = model.context_feats_from_spectral(
            qh0, qh0 - qh1, dt_v, _LX, _LY, Nyg, Nxg)
    with torch.no_grad():
        yhat = model(x, dt=dt_v, dx=dx_v, dy=dy_v, **kw).to(torch.float64)
    Nd = to_spectral(yhat[:, 0:1][0])
    Ndd = to_spectral(yhat[:, 1:2][0])
    N3d = to_spectral(yhat[:, 2:3][0]) if yhat.shape[1] >= 3 else None

    def _rms(spec):
        return float(torch.sqrt((to_physical(spec) ** 2).mean()))

    terms = {
        'R3 L^3 w   (anal)': Delta_T ** 3 * cf * _rms(L3 * qh0),
        'R3 L^2 N   (anal)': Delta_T ** 3 * cf * _rms(L2 * Nh0),
        'R3 L*Ndot  (NN)':   Delta_T ** 3 * cf * _rms(L_hat * Nd),
        'R3 5*Nddot (NN)':   Delta_T ** 3 * cf * 5.0 * _rms(Ndd),
    }
    if include_r4:
        # driver assembly: -coef4 (1/24)(2L^4 w + 2L^3 N + 2L^2 Ndot
        #                            - 4L Nddot + N3dot), coef4 = DT^4
        c24 = Delta_T ** 4 / 24.0
        terms['R4 2L^4 w   (anal)'] = c24 * 2.0 * _rms(L4 * qh0)
        terms['R4 2L^3 N   (anal)'] = c24 * 2.0 * _rms(L3 * Nh0)
        terms['R4 2L^2 Ndot(NN)'] = c24 * 2.0 * _rms(L2 * Nd)
        terms['R4 4L*Nddot (NN)'] = c24 * 4.0 * _rms(L_hat * Ndd)
        if N3d is not None:
            terms['R4 N3dot    (NN)'] = c24 * _rms(N3d)
    tot = sum(terms.values()) or 1e-30
    nn_mass = sum(v for k, v in terms.items() if '(NN)' in k)
    print('\n============ CLOSURE TERM RMS AT IC (--diag) ============')
    for k, v in terms.items():
        print(f'  {k:<20} rms={v:.4e}  {100 * v / tot:5.1f}%')
    print(f'  {"-" * 46}')
    print(f'  total correction rms = {tot:.4e}')
    print(f'  NN-predicted fraction = {100 * nn_mass / tot:.1f}%')
    print('=' * 57)
    return terms


# --------------------------------------------------------------------------- #

_DX = _DY = None     # module-level grid spacings set in main (used by arms)
_LX = _LY = None     # domain lengths (sigma-hat shell cache key)


def main():
    global _DX, _DY, _LX, _LY
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-dir', type=Path, required=True,
                   help='sweep root with manifest.json (grid/physics/Delta_T)')
    p.add_argument('--ckpt', type=Path, required=True,
                   help="train_deriv checkpoint (best.pt) -- tomorrow's model drops in here")
    p.add_argument('--ic-index', type=int, default=None,
                   help='packed-sample row for the IC history stack')
    p.add_argument('--restart-ic', type=Path, default=None,
                   help='developed-flow omega .npy from extract_restart_ic.py '
                        '(history built by ultrafine RK4 forward)')
    p.add_argument('--Delta-T', type=float, default=None,
                   help='rollout coarse step (default: manifest Delta_T)')
    p.add_argument('--K', type=int, default=100,
                   help='fine RK4 substeps per coarse step -- truth refinement '
                        'factor ONLY (does NOT enter the closure coefficient). '
                        'Accuracy runs: pick K so h_fine=Delta_T/K <= ~1e-5 '
                        '(the truth stands in for the ANALYTIC flow; we do '
                        'not model the RK4 LTE). The driver warns above '
                        '2.5e-5.')
    p.add_argument('--horizon-turnovers', type=float, default=10.0,
                   help='horizon in eddy turnovers (tau_eddy = 1/omega_rms(IC))')
    p.add_argument('--tau-eddy', type=float, default=None,
                   help='override the tau_eddy estimate (physical time units)')
    p.add_argument('--n-steps', type=int, default=None,
                   help='explicit coarse-step horizon (overrides turnovers)')
    p.add_argument('--n-checkpoints', type=int, default=24,
                   help='number of checkpointed FIELD snapshots (for spectra)')
    p.add_argument('--arms', type=str, default='bare,r3only,closure',
                   help='comma list from {bare,r3only,r3anal,closure} '
                        '(r3anal = full analytic R3 via exact chain-rule '
                        'Ndot/Nddot, no NN -- the analytic-LTE ceiling / '
                        'blowup discriminator)')
    p.add_argument('--no-truth', action='store_true',
                   help='skip the fine-truth arm (long-horizon stability runs)')
    p.add_argument('--r4', action='store_true',
                   help='add the partial R4 bracket (uses the N3dot head)')
    p.add_argument('--dealias-nn', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--diag', action='store_true',
                   help='print the per-term RMS breakdown of the closure '
                        'correction at the IC (analytic L^k vs NN terms; '
                        'ported from rollout_timed_pareto)')
    p.add_argument('--ckpt2', type=Path, default=None,
                   help="second checkpoint run as arm 'closure2' through the "
                        'IDENTICAL code path (e.g. control vs conditioned; '
                        'appended to --arms automatically)')
    p.add_argument('--freeze-sigma', action='store_true',
                   help='cond_local: pin the sigma-hat context at its t=0 '
                        'value for the whole rollout (the static-'
                        'recalibration A/B leg vs live conditioning)')
    p.add_argument('--log-sigma', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='write sigma_hat(kappa,t) at closure-arm checkpoints '
                        'to sigma_hat_<tag>_<arm>.csv (zero extra FFTs; the '
                        'a-posteriori r2-drift measurement)')
    p.add_argument('--nn-kcut', type=float, default=None, metavar='ALPHA',
                   help='R1 remediation: keep f_NN only for |k| <= ALPHA x '
                        'the 2/3-dealias cut (closure arms only)')
    p.add_argument('--nn-gamma', type=float, default=1.0,
                   help='R2 remediation: under-relax the NN correction, '
                        'rhs -= coef*GAMMA*f_NN (default 1.0 = off)')
    p.add_argument('--nn-clip', type=float, default=None, metavar='C',
                   help='R3 remediation: pointwise-clip physical f_NN at '
                        'C x rms(f_NN), then re-dealias (+2 FFTs/step)')
    p.add_argument('--nn-project-radius', type=float, nargs='?',
                   const=2.0 / 3.0, default=None, metavar='FACTOR',
                   help='project the R3 correction onto the alias-safe '
                        'radius |k| <= FACTOR * min(kx_max, ky_max) (bare '
                        'flag -> FACTOR = 2/3, mode radius ~170.67 at '
                        '512^2). Applies to closure/closure2 f_NN and to '
                        'r3anal f_anal; solver dealias mask unchanged.')
    p.add_argument('--drop-nddot', action='store_true',
                   help='ablation: assemble the R3 correction WITHOUT the '
                        '-5*Nddot term (f = (1/12) L*Ndot only; implicit '
                        'L^3 w and explicit L^2 N kept). Applies to the NN '
                        '(closure/closure2) AND analytic (r3anal) sources.')
    p.add_argument('--track-lte', action='store_true',
                   help='at every checkpoint, compute the FULL ANALYTIC LTE '
                        'at each arm\'s current state (chain-rule Ndot/Nddot) '
                        'and write per-term rms + NN-vs-analytic rel-L2 '
                        '(closure arms) to lte_<tag>_<arm>.csv. Non-zero '
                        'compute inside the timed loop -- diagnostics runs '
                        'only, not headline timings.')
    p.add_argument('--profile-step', type=int, default=0, metavar='N',
                   help='after the clean closure timing, N instrumented '
                        'steps with per-block cuda-event timers + host-vs-'
                        'device verdict (pareto port; cuda only, 0=off)')
    p.add_argument('--save-refs', action='store_true',
                   help='save the RK4 truth checkpoint stack to '
                        'apost_refs_<tag>.npz for reuse via --load-refs '
                        '(the truth is the expensive leg). Stored float32: '
                        'reloaded rel-L2 matches the original to ~1e-7 '
                        '(fields are O(1); fine for errors >= 1e-5)')
    p.add_argument('--load-refs', type=Path, default=None,
                   help='reuse a saved truth stack; hard guards: identical '
                        'Delta_T, K, h_fine, checkpoint grid, and IC field')
    p.add_argument('--pareto', action='store_true',
                   help='bare at a dt sweep vs the SAME truth at the final '
                        'common time -- the cost-vs-accuracy front with the '
                        'closure point(s) overlaid; needs a truth')
    p.add_argument('--pareto-dt-factors', type=str,
                   default='1,2,4,8,16,40,100',
                   help='bare dt = Delta_T / factor (capped at h_fine)')
    p.add_argument('--scalars-every', type=int, default=1,
                   help='record E/Z every this many coarse steps')
    p.add_argument('--blowup-factor', type=float, default=10.0,
                   help='declare blowup when Z exceeds this multiple of Z(0)')
    p.add_argument('--world-mask-radius', type=float, default=None,
                   metavar='FACTOR',
                   help='ALIAS-CLEAN WORLD: rebuild the solver dealias mask '
                        'as radial |k| <= FACTOR * min(kx_max, ky_max) for '
                        'the ENTIRE harness (RK4 truth, bare, r3anal, NN '
                        'arms -- all read derivative.alias_mask), and '
                        'project the IC history stack onto the same ball. '
                        'FACTOR=0.6666666666666667 = strict alias-safe 2/3 '
                        '(quadratic-product folds land at |k| >= (2/3)kmax, '
                        'removed exactly by the product mask). sqrt2-world '
                        'refs FAIL the IC guard by design -- regenerate.')
    p.add_argument('--nn-float64', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='NN parameter dtype (float64 default: closure signal '
                        'sits at coef~1e-9; state math is float64 regardless)')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--tag', type=str, default='apost')
    p.add_argument('--out-dir', type=Path, default=None)
    args = p.parse_args()

    if (args.ic_index is None) == (args.restart_ic is None):
        sys.exit('pass exactly one of --ic-index / --restart-ic')
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    dtype = torch.float64
    out_dir = args.out_dir or args.root_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.root_dir / 'manifest.json').read_text())
    Nx, Ny = int(manifest['Nx']), int(manifest['Ny'])
    Lx, Ly = float(manifest['Lx']), float(manifest['Ly'])
    nu = float(manifest['nu']); mu = float(manifest.get('mu', 0.0))
    beta = float(manifest.get('beta', 0.0))
    Delta_T = args.Delta_T or float(manifest['Delta_T'])
    K = int(args.K)
    h_fine = Delta_T / K
    _DX, _DY = Lx / Nx, Ly / Ny
    _LX, _LY = Lx, Ly
    if manifest.get('mask') or manifest.get('scenario', '') in ('flow_past_cylinder',):
        sys.exit('masked/obstacle scenario: fixed-PHYSICAL-eta handling not '
                 'wired in this driver yet (charter 5.2); refusing.')
    print(f'[apost] grid {Ny}x{Nx} L={Lx:.4f}  nu={nu} mu={mu} beta={beta}')
    print(f'[apost] Delta_T={Delta_T}  K={K} (truth refinement only)  '
          f'h_fine={h_fine:.3e}  coef=DT^3={Delta_T**3:.6e} '
          f'(RK4 truth -> full Taylor defect, no (1-1/K^2))')
    if (args.nn_kcut is not None or args.nn_gamma != 1.0
            or args.nn_clip is not None):
        print(f'[apost] NN remediation active: kcut={args.nn_kcut}  '
              f'gamma={args.nn_gamma}  clip={args.nn_clip} '
              f'(closure arms, R3 f_NN only)')
    if args.drop_nddot:
        print('[apost] ABLATION: --drop-nddot -- R3 assembled WITHOUT the '
              '-5*Nddot term (closure/closure2/r3anal)')
    if args.nn_project_radius is not None:
        print(f'[apost] ALIAS-SAFE PROJECTION: R3 correction kept only for '
              f'|k| <= {args.nn_project_radius:.6f} * min(kx_max, ky_max) '
              f'(closure/closure2 f_NN + r3anal f_anal; solver mask '
              f'unchanged)')
    if args.world_mask_radius is not None:
        print(f'[apost] WORLD MASK OVERRIDE: radial |k| <= '
              f'{args.world_mask_radius:.6f} * min(kmax) (mode radius '
              f'{args.world_mask_radius * (Nx // 2):.1f} of {Nx // 2}) for '
              f'the ENTIRE harness incl. RK4 truth; IC stack projected onto '
              f'the same ball')
    if not args.no_truth and h_fine > 2.5e-5:
        print(f'[apost] WARNING: h_fine={h_fine:.3e} > 2.5e-5 -- the RK4 '
              f'truth is NOT a stand-in for the analytic flow at closure-'
              f'residual level (we do not model the RK4 LTE). Raise K to '
              f'>= {int(np.ceil(Delta_T / 1e-5))} for accuracy tables.')

    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral, to_physical
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device,
                         precision='float64')
    derivative = Derivative(grid)
    for attr in ('dx', 'dy', 'laplacian', 'inv_laplacian', 'alias_mask'):
        if hasattr(derivative, attr):
            setattr(derivative, attr, getattr(derivative, attr).to(device))
    if args.world_mask_radius is not None:
        # every dealias in the harness (J products in _N_core / J_phys /
        # the analytic chain, f_anal, f_NN end-projection -- in BOTH this
        # module and rollout_perfect_closure's own _dealias_mul) reads
        # derivative.alias_mask or the cached derivative._keep_mask, so
        # overriding the mask + clearing the cache re-worlds everything.
        kx2w = (-(derivative.dx ** 2)).real
        ky2w = (-(derivative.dy ** 2)).real
        kmagw = torch.sqrt(kx2w + ky2w)
        k_world = float(args.world_mask_radius) * min(
            float(kx2w.max()) ** 0.5, float(ky2w.max()) ** 0.5)
        derivative.alias_mask = (kmagw > k_world)
        if hasattr(derivative, '_keep_mask'):
            del derivative._keep_mask
    L_hat = build_L_hat(derivative, nu, mu, beta).to(device)
    fc = manifest.get('forcing') if manifest.get('has_forcing') else None
    F_phys = build_forcing(grid, fc, device, dtype)
    F_hat = to_spectral(F_phys) if F_phys is not None else None
    print(f'[apost] forcing: {"none" if F_phys is None else fc}')

    # ---- model ---- #
    model, model_name, n_snap = load_deriv_model(
        args.ckpt, manifest, Delta_T, device, nn_float64=args.nn_float64)
    input_fields = (['omega_0'] + [f'omega_m{k}' for k in range(1, n_snap)]
                    + ['psi_0'] + [f'psi_m{k}' for k in range(1, n_snap)])
    model2 = model2_name = None
    if args.ckpt2 is not None:
        model2, model2_name, n_snap2 = load_deriv_model(
            args.ckpt2, manifest, Delta_T, device, nn_float64=args.nn_float64)
        if n_snap2 != n_snap:
            sys.exit(f'[apost] --ckpt2 n_snapshots={n_snap2} != --ckpt '
                     f'n_snapshots={n_snap}; arms would need different '
                     f'history depths.')

    # ---- IC: S-deep history at Delta_T spacing, newest first ---- #
    if args.ic_index is not None:
        dt_sweep = float(manifest['Delta_T'])
        if abs(dt_sweep - Delta_T) > 1e-15:
            sys.exit(f'--ic-index history is at the sweep Delta_T={dt_sweep}; '
                     f'rollout Delta_T={Delta_T} differs. Use --restart-ic.')
        inp = np.load(args.root_dir / 'packed' / 'inputs.npy', mmap_mode='r')
        fields_idx = {f: c for c, f in enumerate(manifest['input_fields'])}
        omega_stack = [torch.tensor(np.asarray(inp[args.ic_index,
                                                   fields_idx[f'omega_0' if k == 0 else f'omega_m{k}']],
                                               dtype=np.float64),
                                    dtype=dtype, device=device)[None]
                       for k in range(n_snap)]
        if args.world_mask_radius is not None:
            omega_stack = [to_physical(_dealias_mul(to_spectral(o),
                                                    derivative))
                           for o in omega_stack]
            print('[apost] IC stack projected onto the world-mask ball')
        psi_stack = [psi_from_omega(o, derivative) for o in omega_stack]
        print(f'[apost] IC: packed row {args.ic_index} '
              f'(|omega_0|_rms={float(torch.sqrt((omega_stack[0]**2).mean())):.4e})')
    else:
        seed = np.load(args.restart_ic).astype(np.float64)
        if seed.ndim == 3:
            seed = seed[0]
        om_seed = torch.tensor(seed, dtype=dtype, device=device)[None]
        if args.world_mask_radius is not None:
            om_seed = to_physical(_dealias_mul(to_spectral(om_seed),
                                               derivative))
            print('[apost] restart IC projected onto the world-mask ball')
        h_uf = Delta_T / 200.0
        n_uf = int(round(Delta_T / h_uf))
        marks = [om_seed.clone()]
        cur = om_seed.clone()
        for m in range((n_snap - 1) * n_uf):
            cur = rk4_step(cur, h_uf, derivative, L_hat, F_phys)
            if (m + 1) % n_uf == 0:
                marks.append(cur.clone())
        omega_stack = marks[::-1]                    # newest first
        psi_stack = [psi_from_omega(o, derivative) for o in omega_stack]
        print(f'[apost] IC: restart {args.restart_ic} + {(n_snap-1)} Delta_T '
              f'ultrafine warmup marks (RK4 @ {h_uf:.2e})')

    # ---- one-time Parseval E/Z sanity vs the physical-space formulas ---- #
    qh0 = to_spectral(omega_stack[0])
    E_s, Z_s = scalars_from_qh(qh0, derivative, _parseval_weight(qh0, Nx))
    om0p = to_physical(qh0)
    ps0p = to_physical(derivative.inv_laplacian * qh0)
    E_p = 0.5 * float((-ps0p * om0p).mean())
    Z_p = 0.5 * float((om0p ** 2).mean())
    if (abs(Z_s - Z_p) > 1e-9 * max(abs(Z_p), 1e-30)
            or abs(E_s - E_p) > 1e-9 * max(abs(E_p), 1e-30)):
        sys.exit(f'[apost] Parseval E/Z mismatch: spectral ({E_s:.6e}, {Z_s:.6e}) '
                 f'vs physical ({E_p:.6e}, {Z_p:.6e}) -- norm convention changed?')
    print(f'[apost] E/Z Parseval check OK: E={E_s:.6e} Z={Z_s:.6e}')

    # ---- horizon ---- #
    om_rms = float(torch.sqrt((omega_stack[0] ** 2).mean()))
    tau_eddy = args.tau_eddy or 1.0 / om_rms
    if args.n_steps is not None:
        M = int(args.n_steps)
    else:
        M = max(int(round(args.horizon_turnovers * tau_eddy / Delta_T)), 10)
    cp = sorted(set(int(round(f * M))
                    for f in np.linspace(0, 1, args.n_checkpoints + 1)))
    print(f'[apost] tau_eddy={tau_eddy:.4f} (1/omega_rms={1.0/om_rms:.4f})  '
          f'horizon M={M} coarse steps = {M*Delta_T:.3f} t.u. '
          f'= {M*Delta_T/tau_eddy:.1f} turnovers; {len(cp)} field checkpoints')

    # ---- optional per-term diagnosis at the IC ---- #
    diag_terms = None
    if args.diag:
        diag_terms = diag_term_rms(model, omega_stack, psi_stack, Delta_T,
                                   derivative, L_hat, F_hat, device,
                                   input_fields, include_r4=args.r4)

    # ---- truth ---- #
    results = dict(config={k: str(v) for k, v in vars(args).items()},
                   Delta_T=Delta_T, K=K, h_fine=h_fine, M=M, tau_eddy=tau_eddy,
                   model=model_name, cp_steps=cp)
    if diag_terms is not None:
        results['diag_term_rms'] = diag_terms
    npz_payload = dict(cp_steps=np.asarray(cp, np.int64),
                       cp_times=np.asarray([s * Delta_T for s in cp]))
    truth_cp = None
    if args.load_refs is not None:
        rf = np.load(args.load_refs)
        for name, cur in (('Delta_T', Delta_T), ('K', float(K)),
                          ('h_fine', h_fine)):
            if abs(float(rf[name]) - cur) > 1e-12 * max(abs(cur), 1e-30):
                sys.exit(f'[apost] --load-refs {name} mismatch: refs '
                         f'{float(rf[name])} vs current {cur}')
        refs_cp = [int(s) for s in rf['cp_steps']]
        if refs_cp != cp:
            sys.exit(f'[apost] --load-refs checkpoint-grid mismatch: refs '
                     f'{refs_cp} vs current {cp} (match --n-steps / '
                     f'--n-checkpoints to the saving run)')
        ic_saved = np.asarray(rf['ic_omega_0'], np.float64)
        ic_cur = omega_stack[0][0].detach().cpu().numpy()
        rel = (np.sqrt(np.mean((ic_saved - ic_cur) ** 2))
               / max(np.sqrt(np.mean(ic_cur ** 2)), 1e-30))
        if rel > 1e-12:
            sys.exit(f'[apost] --load-refs IC mismatch (rel {rel:.2e}): the '
                     f'saved truth was integrated from a DIFFERENT state.')
        truth_cp = {s * K: np.asarray(rf['truth_stack'][i])
                    for i, s in enumerate(refs_cp)}
        t_truth = float(rf['t_truth'])
        results['t_truth'] = t_truth
        npz_payload['truth_stack'] = np.asarray(rf['truth_stack'])
        print(f'[apost] truth REUSED from {args.load_refs} '
              f'(skipped {M*K} RK4 steps; saved t_truth={t_truth:.1f}s)')
    elif not args.no_truth:
        cp_fine = [s * K for s in cp]
        print(f'[apost] truth: {M*K} RK4 steps @ h={h_fine:.3e} '
              f'(rollout_timed_pareto.rollout_fine; single-step, F_phys '
              f'physical, warmup untimed) ...')
        truth_cp, t_truth = rollout_fine(omega_stack[0], h_fine, M * K,
                                         cp_fine, derivative, L_hat,
                                         F_phys, device)
        results['t_truth'] = t_truth
        npz_payload['truth_stack'] = np.stack(
            [np.asarray(truth_cp[s * K], np.float32) for s in cp
             if s * K in truth_cp])
        print(f'[apost]   truth walltime {t_truth:.1f}s')
        if args.save_refs:
            avail_cp = [s for s in cp if s * K in truth_cp]
            refs_path = out_dir / f'apost_refs_{args.tag}.npz'
            np.savez(refs_path,
                     Delta_T=np.float64(Delta_T), K=np.int64(K),
                     h_fine=np.float64(h_fine),
                     cp_steps=np.asarray(avail_cp, np.int64),
                     ic_index=np.int64(-1 if args.ic_index is None
                                       else args.ic_index),
                     ic_omega_0=omega_stack[0][0].detach().cpu().numpy(),
                     t_truth=np.float64(t_truth),
                     truth_stack=np.stack(
                         [np.asarray(truth_cp[s * K], np.float32)
                          for s in avail_cp]))
            print(f'[apost] saved truth refs -> {refs_path} '
                  f'({len(avail_cp)} checkpoints, float32) -- reuse with '
                  f'--load-refs {refs_path}')

    # ---- arms ---- #
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    bad = [a for a in arms if a not in ('bare', 'r3only', 'r3anal',
                                        'closure', 'closure2')]
    if bad:
        sys.exit(f'[apost] unknown arm(s) {bad}; valid: '
                 f'bare, r3only, r3anal, closure, closure2')
    if 'closure2' in arms and model2 is None:
        sys.exit('[apost] arm closure2 requires --ckpt2 (fail fast: the '
                 'truth leg would otherwise run before the crash)')
    if model2 is not None and 'closure2' not in arms:
        arms.append('closure2')
    if model2 is not None:
        results['model2'] = model2_name
        results['ckpt2'] = str(args.ckpt2)
    arm_out = {}
    for arm in arms:
        mdl = model2 if arm == 'closure2' else model
        sig_rows = ([] if (args.log_sigma and (arm.startswith('closure')
                                               or arm == 'r3anal'))
                    else None)
        lte_rows = [] if args.track_lte else None
        print(f'[apost] arm={arm}: {M} coarse steps ...'
              + (f'  [ckpt2={args.ckpt2.parent.name}]'
                 if arm == 'closure2' else '')
              + ('  [sigma FROZEN at t=0]'
                 if (args.freeze_sigma and arm.startswith('closure'))
                 else ''))
        r = run_arm(arm, omega_stack, psi_stack, Delta_T, M, cp,
                    derivative, L_hat, F_hat, device,
                    model=mdl, input_fields=input_fields,
                    dealias_nn=args.dealias_nn, include_r4=args.r4,
                    blowup_factor=args.blowup_factor,
                    scalars_every=args.scalars_every,
                    freeze_sigma=args.freeze_sigma, sigma_log=sig_rows,
                    lte_log=lte_rows, profile_step=args.profile_step,
                    nn_kcut=args.nn_kcut, nn_gamma=args.nn_gamma,
                    nn_clip=args.nn_clip, drop_nddot=args.drop_nddot,
                    nn_project_radius=args.nn_project_radius)
        arm_out[arm] = r
        if lte_rows:
            lte_csv = out_dir / f'lte_{args.tag}_{arm}.csv'
            with open(lte_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(lte_rows[0].keys()))
                w.writeheader()
                w.writerows(lte_rows)
            print(f'[apost]   analytic-LTE track -> {lte_csv.name} '
                  f'({len(lte_rows)} checkpoints)')
        if sig_rows:
            sig_csv = out_dir / f'sigma_hat_{args.tag}_{arm}.csv'
            n_sh = len(sig_rows[0][2])
            with open(sig_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['step', 't'] + [f'kappa_{i}' for i in range(n_sh)])
                for st, tt, sg in sig_rows:
                    w.writerow([st, f'{tt:.6f}'] + [f'{v:.8e}' for v in sg])
            print(f'[apost]   sigma-hat drift -> {sig_csv.name} '
                  f'({len(sig_rows)} checkpoints x {n_sh} shells, zero FFTs)')
        print(f'[apost]   walltime {r["walltime"]:.1f}s  cfl_max={r["cfl_max"]:.3f}'
              f'  blowup={"none" if r["blowup"] is None else r["blowup"]}')
        npz_payload[f'{arm}_stack'] = np.stack(
            [np.asarray(r['fields'][s], np.float32) for s in cp
             if s in r['fields']])
        npz_payload[f'{arm}_cp_avail'] = np.asarray(
            [s for s in cp if s in r['fields']], np.int64)
        npz_payload[f'{arm}_t'] = r['t']
        npz_payload[f'{arm}_E'] = r['E']
        npz_payload[f'{arm}_Z'] = r['Z']
        results[f'{arm}_walltime'] = r['walltime']
        results[f'{arm}_cfl_max'] = r['cfl_max']
        results[f'{arm}_blowup_step'] = r['blowup']
        results[f'{arm}_verdict'] = ('UNSTABLE' if r['blowup'] is not None
                                     else 'STABLE')

    # ---- error tables vs truth ---- #
    if truth_cp is not None:
        rows = []
        for s in cp:
            if s * K not in truth_cp:
                continue
            row = {'t': s * Delta_T}
            for arm in arms:
                if s in arm_out[arm]['fields']:
                    row[f'relL2_{arm}'] = rel_l2(arm_out[arm]['fields'][s],
                                                 truth_cp[s * K])
            rows.append(row)
        hdr = ['t'] + [f'relL2_{a}' for a in arms]
        csv_path = out_dir / f'rollout_apost_{args.tag}.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
        print(f'\n{"t":>10}' + ''.join(f'{a:>14}' for a in arms))
        for row in rows:
            print(f'{row["t"]:>10.4f}' + ''.join(
                f'{row.get(f"relL2_{a}", float("nan")):>14.4e}' for a in arms))
        results['error_table'] = rows
        finals = {a: rows[-1].get(f'relL2_{a}') for a in arms if rows}
        results['final_relL2'] = finals
        if 'bare' in finals:
            imps = []
            for a in (x for x in arms if x.startswith('closure')):
                if finals.get(a):
                    results[f'improvement_x_{a}'] = (finals['bare']
                                                     / max(finals[a], 1e-30))
                    imps.append(f'{a}={results[f"improvement_x_{a}"]:.1f}x')
            if 'improvement_x_closure' in results:      # legacy key
                results['improvement_x'] = results['improvement_x_closure']
            if imps:
                print(f'\n[apost] final rel-L2: ' +
                      '  '.join(f'{a}={v:.4e}'
                                for a, v in finals.items() if v) +
                      '   improvement: ' + '  '.join(imps))

        # ---- pareto: bare at a dt sweep vs the SAME truth (headline
        # cost-vs-accuracy front, closure point(s) overlaid) ---- #
        if args.pareto and rows:
            common = [s for s in cp if s * K in truth_cp]
            s_end = common[-1]
            print('\n============ PARETO (bare dt sweep) ============')
            pts = []
            for fac in (float(x) for x in args.pareto_dt_factors.split(',')):
                dtb = Delta_T / fac
                if dtb < h_fine * 0.999:
                    continue
                # final time nst*dtb matches s_end*Delta_T to within dtb/2
                nst = int(round(s_end * Delta_T / dtb))
                # AB2 history must be dtb-spaced: seed omega(-dtb) with ONE
                # RK4 back-step (O(dtb^5)). Reusing the Delta_T-spaced
                # omega_m1 injects a dt^1 startup error that flattens the
                # small-dt tail of the front -- flattering the closure point
                # (reviewer catch; the same flaw exists in the original
                # rollout_timed_pareto sweep).
                om_m1_dtb = rk4_step(omega_stack[0], -dtb, derivative,
                                     L_hat, F_phys)
                rb = run_arm('bare', [omega_stack[0], om_m1_dtb],
                             psi_stack[:2], dtb, nst, [nst],
                             derivative, L_hat, F_hat, device,
                             scalars_every=10 ** 9)
                if nst not in rb['fields']:
                    print(f'  bare dt={dtb:.3e}  blew up -- skipped')
                    continue
                err = rel_l2(rb['fields'][nst], truth_cp[s_end * K])
                pts.append((dtb, rb['walltime'], err))
                print(f'  bare dt={dtb:.3e}  steps={nst}  '
                      f'wall={rb["walltime"]:.3f}s  rel-L2={err:.4e}')
            results['pareto'] = [list(x) for x in pts]
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6.2, 4.6))
                if pts:
                    _, wts_, errs_ = zip(*pts)
                    ax.loglog(wts_, errs_, 'o-', color='C0',
                              label=r'bare @ varying $\delta t$')
                for i, a in enumerate(x for x in arms
                                      if x.startswith('closure')):
                    fe = rows[-1].get(f'relL2_{a}')
                    if fe:
                        ax.loglog([results[f'{a}_walltime']], [fe], '*',
                                  ms=15, color=f'C{3 + i}', label=a)
                ax.set_xlabel('walltime (s)')
                ax.set_ylabel('final rel-L2 vs RK4 truth')
                ax.set_title(f'cost / accuracy front (K={K})')
                ax.grid(alpha=0.3, which='both')
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(out_dir / f'rollout_apost_{args.tag}_pareto.png',
                            dpi=130)
                print(f'[apost] wrote rollout_apost_{args.tag}_pareto.png')
            except Exception as e:              # noqa: BLE001 (fig never kills run)
                print(f'[apost] pareto figure skipped '
                      f'({type(e).__name__}: {e})')

    np.savez(out_dir / f'rollout_apost_{args.tag}.npz', **npz_payload)
    (out_dir / f'rollout_apost_{args.tag}.json').write_text(
        json.dumps(results, indent=2, default=float))
    print(f'[apost] wrote rollout_apost_{args.tag}.npz/.json'
          + ('' if truth_cp is None else '/.csv') + f' in {out_dir}')


if __name__ == '__main__':
    main()
