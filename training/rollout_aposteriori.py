"""
rollout_aposteriori.py -- the unified a-posteriori driver for the trained closure.

One driver, four arms from a SHARED developed-flow IC:

  truth    : AB2CN2 at h = Delta_T / K       (the scheme's better-resolved self --
             the deliverable's target per the "emulate the SAME scheme at h/K"
             framing; NOT RK4, so the (1 - 1/K^2) factor applies to the closure)
  bare     : AB2CN2 at Delta_T, no closure
  r3only   : AB2CN2 at Delta_T + ANALYTIC R3 pieces only (L^3 w implicit,
             L^2 N explicit) -- no NN
  closure  : AB2CN2 at Delta_T + trained closure per the inference decomposition:
                 implicit : the -(coef/12) L^3 w term is folded into the IMEX
                            solve (evaluated at w_{n+1}; denominator gains
                            + (coef/12) L_hat^3),
                 explicit : analytic -(coef/12) L^2 N(w_n),
                 explicit : NN  -coef * f_NN,  f_NN = (1/12)(L Ndot - 5 Nddot)
                            from the trained [Ndot, Nddot, N3dot] heads
             with coef = Delta_T^3 (1 - 1/K^2).  Optional partial R4 via --r4
             (uses the N3dot head; coefficients per rollout_timed_pareto).

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
    N_spectral, _dealias_mul, build_L_hat, rk4_step,
    build_forcing, psi_from_omega, assemble_inputs, _sync)
from model_deriv_closure import build_model                         # noqa: E402


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

def scalars_from_qh(qh, derivative):
    """(E, Z) domain means from spectral omega -- cheap reductions."""
    from qg.solver.opt.basis import to_physical
    om = to_physical(qh)
    psi = to_physical(derivative.inv_laplacian * qh)
    Z = 0.5 * float((om ** 2).mean())
    E = 0.5 * float((-psi * om).mean())      # E = 0.5 <|grad psi|^2> = -0.5 <psi omega>
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

def run_arm(arm, omega_stack, psi_stack, Delta_T, K, n_steps, cp_steps,
            derivative, L_hat, F_hat, device, model=None, input_fields=None,
            dealias_nn=True, include_r4=False, blowup_factor=10.0,
            scalars_every=1):
    """One arm of the comparison. arm in {'bare','r3only','closure'}.

    Returns dict: fields {step: ndarray}, scalars (t, E, Z arrays),
    cfl_max, blowup (None or step), walltime.
    """
    from qg.solver.opt.basis import to_spectral, to_physical
    coef = (Delta_T ** 3) * (1.0 - 1.0 / (K ** 2))   # AB2CN2 truth at h/K
    c12 = coef / 12.0
    denom_bare = 1.0 - 0.5 * Delta_T * L_hat
    # implicit fold: the -(c12) L^3 w term evaluated at w_{n+1} moves to the LHS
    denom_clos = denom_bare + c12 * (L_hat ** 3)
    L2 = L_hat ** 2
    L4 = L2 * L2
    with_closure = arm in ('r3only', 'closure')
    denom = denom_clos if with_closure else denom_bare

    qh_n = to_spectral(omega_stack[0])
    qh_m1 = to_spectral(omega_stack[1])
    Nh_n = N_spectral(qh_n, derivative, F_hat)
    Nh_m1 = N_spectral(qh_m1, derivative, F_hat)
    om = [s.clone() for s in omega_stack]
    ps = [s.clone() for s in psi_stack]

    def one_step(qh_curr, Nh_curr, Nh_minus, om_hist, ps_hist):
        AB2_Nh = 1.5 * Nh_curr - 0.5 * Nh_minus
        rhs = qh_curr + Delta_T * (0.5 * L_hat * qh_curr + AB2_Nh)
        if with_closure:
            rhs = rhs - c12 * (L2 * Nh_curr)                 # explicit analytic L^2 N
        if arm == 'closure':
            # feed the stack at float64 so the TimeFD differencing is
            # cancellation-clean (the model handles its own mixed precision);
            # dt/dx/dy passed explicitly = the exact training call signature.
            x = assemble_inputs(input_fields, om_hist, ps_hist,
                                torch.float64, device)
            dt_v = torch.full((1,), Delta_T, device=device, dtype=torch.float64)
            dx_v = torch.full((1,), _DX, device=device, dtype=torch.float64)
            dy_v = torch.full((1,), _DY, device=device, dtype=torch.float64)
            with torch.no_grad():
                yhat = model(x, dt=dt_v, dx=dx_v, dy=dy_v).to(torch.float64)
            Ndot_h = to_spectral(yhat[:, 0:1][0])
            Nddot_h = to_spectral(yhat[:, 1:2][0])
            f_nn = (1.0 / 12.0) * (L_hat * Ndot_h - 5.0 * Nddot_h)
            if dealias_nn:
                f_nn = _dealias_mul(f_nn, derivative)
            rhs = rhs - coef * f_nn
            if include_r4:
                N3dot_h = to_spectral(yhat[:, 2:3][0])
                coef4 = (Delta_T ** 4) * (1.0 - 1.0 / (K ** 3))
                e_r4 = -coef4 * (1.0 / 24.0) * (2.0 * L4 * qh_curr
                                                + 2.0 * (L_hat ** 3) * Nh_curr
                                                + 2.0 * L2 * Ndot_h
                                                - 4.0 * L_hat * Nddot_h
                                                + N3dot_h)
                if dealias_nn:
                    e_r4 = _dealias_mul(e_r4, derivative)
                rhs = rhs + e_r4
        qh_new = rhs / denom
        return qh_new

    fields = {}
    cps = set(cp_steps)
    ts, Es, Zs = [], [], []
    cfl_max = 0.0
    blowup = None
    E0, Z0 = scalars_from_qh(qh_n, derivative)
    if 0 in cps:
        fields[0] = to_physical(qh_n)[0].cpu().numpy()
    ts.append(0.0); Es.append(E0); Zs.append(Z0)

    _sync(device); t0 = time.time()
    for s in range(1, n_steps + 1):
        qh_new = one_step(qh_n, Nh_n, Nh_m1, om, ps)
        Nh_m1, Nh_n, qh_n = Nh_n, N_spectral(qh_new, derivative, F_hat), qh_new
        if arm == 'closure':
            om_new = to_physical(qh_n)
            ps_new = to_physical(derivative.inv_laplacian * qh_n)
            om = [om_new] + om[:-1]        # newest-first history for the stencil
            ps = [ps_new] + ps[:-1]
        if s % scalars_every == 0 or s in cps or s == n_steps:
            E, Z = scalars_from_qh(qh_n, derivative)
            ts.append(s * Delta_T); Es.append(E); Zs.append(Z)
            if (not np.isfinite(Z)) or Z > blowup_factor * max(Z0, 1e-30):
                blowup = s
                print(f'      [{arm}] BLOWUP at step {s} '
                      f'(Z={Z:.3e} vs Z0={Z0:.3e}) -- stopping arm.')
                break
        if s in cps:
            fields[s] = to_physical(qh_n)[0].cpu().numpy()
            cfl_max = max(cfl_max, cfl_from_qh(qh_n, derivative,
                                               Delta_T, _DX, _DY))
    _sync(device)
    return dict(fields=fields, t=np.asarray(ts), E=np.asarray(Es),
                Z=np.asarray(Zs), cfl_max=cfl_max, blowup=blowup,
                walltime=time.time() - t0)


def run_truth(omega_0, omega_m1, h, n_fine, cp_fine, derivative, L_hat, F_hat,
              device):
    """AB2CN2 at the fine step h = Delta_T/K -- the scheme's better-resolved self."""
    from qg.solver.opt.basis import to_spectral, to_physical
    qh_n = to_spectral(omega_0)
    qh_m1 = to_spectral(omega_m1)
    denom = 1.0 - 0.5 * h * L_hat
    Nh_n = N_spectral(qh_n, derivative, F_hat)
    Nh_m1 = N_spectral(qh_m1, derivative, F_hat)
    out = {}
    cps = set(cp_fine)
    if 0 in cps:
        out[0] = to_physical(qh_n)[0].cpu().numpy()
    _sync(device); t0 = time.time()
    for s in range(1, n_fine + 1):
        qh_new = (qh_n + h * (0.5 * L_hat * qh_n
                              + (1.5 * Nh_n - 0.5 * Nh_m1))) / denom
        Nh_m1, Nh_n, qh_n = Nh_n, N_spectral(qh_new, derivative, F_hat), qh_new
        if s in cps:
            arr = to_physical(qh_n)[0].cpu().numpy()
            out[s] = arr
            if not np.isfinite(np.sqrt(np.mean(arr ** 2))):
                print(f'      [truth] non-finite at fine step {s} -- stopping.')
                break
    _sync(device)
    return out, time.time() - t0


# --------------------------------------------------------------------------- #

_DX = _DY = None     # module-level grid spacings set in main (used by arms)


def main():
    global _DX, _DY
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
                   help='fine substeps per coarse step for the truth arm')
    p.add_argument('--horizon-turnovers', type=float, default=10.0,
                   help='horizon in eddy turnovers (tau_eddy = 1/omega_rms(IC))')
    p.add_argument('--tau-eddy', type=float, default=None,
                   help='override the tau_eddy estimate (physical time units)')
    p.add_argument('--n-steps', type=int, default=None,
                   help='explicit coarse-step horizon (overrides turnovers)')
    p.add_argument('--n-checkpoints', type=int, default=24,
                   help='number of checkpointed FIELD snapshots (for spectra)')
    p.add_argument('--arms', type=str, default='bare,r3only,closure',
                   help='comma list from {bare,r3only,closure}')
    p.add_argument('--no-truth', action='store_true',
                   help='skip the fine-truth arm (long-horizon stability runs)')
    p.add_argument('--r4', action='store_true',
                   help='add the partial R4 bracket (uses the N3dot head)')
    p.add_argument('--dealias-nn', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--scalars-every', type=int, default=1,
                   help='record E/Z every this many coarse steps')
    p.add_argument('--blowup-factor', type=float, default=10.0,
                   help='declare blowup when Z exceeds this multiple of Z(0)')
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
    if manifest.get('mask') or manifest.get('scenario', '') in ('flow_past_cylinder',):
        sys.exit('masked/obstacle scenario: fixed-PHYSICAL-eta handling not '
                 'wired in this driver yet (charter 5.2); refusing.')
    print(f'[apost] grid {Ny}x{Nx} L={Lx:.4f}  nu={nu} mu={mu} beta={beta}')
    print(f'[apost] Delta_T={Delta_T}  K={K}  h_fine={h_fine:.3e}  '
          f'coef=DT^3(1-1/K^2)={(Delta_T**3)*(1-1/K**2):.6e}')

    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral, to_physical
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device,
                         precision='float64')
    derivative = Derivative(grid)
    for attr in ('dx', 'dy', 'laplacian', 'inv_laplacian', 'alias_mask'):
        if hasattr(derivative, attr):
            setattr(derivative, attr, getattr(derivative, attr).to(device))
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
        psi_stack = [psi_from_omega(o, derivative) for o in omega_stack]
        print(f'[apost] IC: packed row {args.ic_index} '
              f'(|omega_0|_rms={float(torch.sqrt((omega_stack[0]**2).mean())):.4e})')
    else:
        seed = np.load(args.restart_ic).astype(np.float64)
        if seed.ndim == 3:
            seed = seed[0]
        om_seed = torch.tensor(seed, dtype=dtype, device=device)[None]
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

    # ---- truth ---- #
    results = dict(config={k: str(v) for k, v in vars(args).items()},
                   Delta_T=Delta_T, K=K, h_fine=h_fine, M=M, tau_eddy=tau_eddy,
                   model=model_name, cp_steps=cp)
    npz_payload = dict(cp_steps=np.asarray(cp, np.int64),
                       cp_times=np.asarray([s * Delta_T for s in cp]))
    truth_cp = None
    if not args.no_truth:
        cp_fine = [s * K for s in cp]
        print(f'[apost] truth: {M*K} AB2CN2 steps @ h={h_fine:.3e} ...')
        truth_cp, t_truth = run_truth(omega_stack[0], omega_stack[1], h_fine,
                                      M * K, cp_fine, derivative, L_hat,
                                      F_hat, device)
        results['t_truth'] = t_truth
        npz_payload['truth_stack'] = np.stack(
            [np.asarray(truth_cp[s * K], np.float32) for s in cp
             if s * K in truth_cp])
        print(f'[apost]   truth walltime {t_truth:.1f}s')

    # ---- arms ---- #
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    arm_out = {}
    for arm in arms:
        print(f'[apost] arm={arm}: {M} coarse steps ...')
        r = run_arm(arm, omega_stack, psi_stack, Delta_T, K, M, cp,
                    derivative, L_hat, F_hat, device,
                    model=model, input_fields=input_fields,
                    dealias_nn=args.dealias_nn, include_r4=args.r4,
                    blowup_factor=args.blowup_factor,
                    scalars_every=args.scalars_every)
        arm_out[arm] = r
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
        if 'bare' in finals and 'closure' in finals and finals.get('closure'):
            results['improvement_x'] = finals['bare'] / max(finals['closure'], 1e-30)
            print(f'\n[apost] final rel-L2: ' +
                  '  '.join(f'{a}={v:.4e}' for a, v in finals.items() if v) +
                  f'   closure improvement = {results["improvement_x"]:.1f}x')

    np.savez(out_dir / f'rollout_apost_{args.tag}.npz', **npz_payload)
    (out_dir / f'rollout_apost_{args.tag}.json').write_text(
        json.dumps(results, indent=2, default=float))
    print(f'[apost] wrote rollout_apost_{args.tag}.npz/.json'
          + ('' if truth_cp is None else '/.csv') + f' in {out_dir}')


if __name__ == '__main__':
    main()
