#!/usr/bin/env python
"""compare_arms_stepwise.py -- roll the r3anal and closure arms in LOCKSTEP
from the SAME initial condition and record, every step, exactly how far apart
the two states have drifted and how big each arm's applied correction was.

THE PUZZLE THIS EXISTS TO SETTLE. At the IC, measured through the driver's own
path, the NN closure and the exact analytic closure agree to ~1%:

    coef*(1/12)*L*Ndot   : analytic 1.1843e-07   NN 1.2040e-07   (1.7%)
    coef*(1/12)*5*Nddot  : analytic 1.9024e-05   NN 1.9247e-05   (1.2%)

and the whole correction is ~2e-5 relative to a state with |omega|_rms = 0.954.
Yet the r3anal arm runs 64 steps clean (CFL 0.298) while the closure arm
explodes at step 37 (CFL 3.76). A 1% difference inside a 2e-5 perturbation
cannot do that linearly, so the two arms must diverge AFTER step 0. This script
gives the divergence ONSET independent of the blowup: the first step at which
the two states differ by more than 1e-10 / 1e-6 / 1e-3 relative, and the radial
band in which the gap first opens.

WHAT IT LOGS, per coarse step s:
    rel           rms(w_clos - w_anal) / rms(w_anal)         [Parseval, 0 FFTs]
    d_<band>      rms(w_clos - w_anal) restricted to each radial band
    Cclos, Canal  rms of each arm's TOTAL applied closure correction that step
                  (analytic + NN + the implicitly folded L^3, summed as fields
                  -- run_arm's own instrumentation capture, not a re-derivation)
    cfl_clos, cfl_anal, Z_clos, Z_anal

BOTH arms are driven through run_arm(..., return_stepper=True), i.e. the EXACT
stepper the validated a-posteriori arms run -- this script owns no scheme code
of its own. The arms are stepped alternately from a shared IC so their states
stay aligned in time by construction.

Usage (from training/):
  python ../diagnostics/compare_arms_stepwise.py \
      --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_5em3 \
      --ckpt <best.pt> --ic-index 912 --Delta-T 5.0e-3 --n-steps 64 \
      --device cpu --out-dir <dir> --tag kf4_ic912_5em3

[fable-authored 2026-07-20]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))

import rollout_aposteriori as ra                                  # noqa: E402
from rollout_aposteriori import (                                 # noqa: E402
    _FNN_BAND_NAMES, _default_band_edges, _fnn_bands, _spec_rms,
    cfl_from_qh, run_arm,
    load_deriv_model, _parseval_weight, scalars_from_qh)
from rollout_timed_pareto import (                                # noqa: E402
    N_spectral, build_L_hat, build_forcing, psi_from_omega, rk4_step)

_THRESH = (1e-10, 1e-6, 1e-3)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-dir', type=Path, required=True)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--ic-index', type=int, default=None)
    p.add_argument('--restart-ic', type=Path, default=None)
    p.add_argument('--Delta-T', type=float, default=None)
    p.add_argument('--n-steps', type=int, default=64)
    p.add_argument('--dealias-nn', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--r4', action='store_true')
    p.add_argument('--nn-gamma', type=float, default=1.0)
    p.add_argument('--nn-kcut', type=float, default=None)
    p.add_argument('--nn-clip', type=float, default=None)
    p.add_argument('--nn-project-radius', type=float, default=None)
    p.add_argument('--nn-dissipative-proj', action='store_true')
    p.add_argument('--closure-apply', type=str, default='folded',
                   choices=('folded', 'postadd'))
    p.add_argument('--fnn-band-edges', type=str, default=None,
                   help='band edges as absolute mode indices; default is '
                        'DERIVED FROM THE GRID (core / (2/3)k_max alias-safe '
                        '/ sqrt(2)(2/3)k_max annulus / beyond).')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--tag', type=str, default='cmp')
    p.add_argument('--out-dir', type=Path, default=None)
    args = p.parse_args()

    if (args.ic_index is None) == (args.restart_ic is None):
        sys.exit('pass exactly one of --ic-index / --restart-ic')
    device = (args.device if (args.device == 'cpu' or torch.cuda.is_available())
              else 'cpu')
    dtype = torch.float64
    out_dir = args.out_dir or args.root_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.root_dir / 'manifest.json').read_text())
    Nx, Ny = int(manifest['Nx']), int(manifest['Ny'])
    Lx, Ly = float(manifest['Lx']), float(manifest['Ly'])
    nu = float(manifest['nu'])
    mu = float(manifest.get('mu', 0.0))
    beta = float(manifest.get('beta', 0.0))
    Delta_T = args.Delta_T or float(manifest['Delta_T'])
    if abs(Lx - Ly) > 1e-12:
        sys.exit('anisotropic domain: the radial band split assumes Lx == Ly')
    # the arms read these as module globals (grid spacings / domain lengths)
    ra._DX, ra._DY = Lx / Nx, Ly / Ny
    ra._LX, ra._LY = Lx, Ly

    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral
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

    model, model_name, n_snap = load_deriv_model(args.ckpt, manifest, Delta_T,
                                                 device, nn_float64=True)
    input_fields = (['omega_0'] + [f'omega_m{k}' for k in range(1, n_snap)]
                    + ['psi_0'] + [f'psi_m{k}' for k in range(1, n_snap)])

    # ---- IC: the S-deep history at Delta_T spacing, newest first ---- #
    if args.ic_index is not None:
        dt_sweep = float(manifest['Delta_T'])
        if abs(dt_sweep - Delta_T) > 1e-15:
            sys.exit(f'--ic-index history is at the sweep Delta_T={dt_sweep}; '
                     f'requested {Delta_T}. Use --restart-ic.')
        inp = np.load(args.root_dir / 'packed' / 'inputs.npy', mmap_mode='r')
        fidx = {f: c for c, f in enumerate(manifest['input_fields'])}
        omega_stack = [
            torch.tensor(np.asarray(
                inp[args.ic_index,
                    fidx['omega_0' if k == 0 else f'omega_m{k}']],
                dtype=np.float64), dtype=dtype, device=device)[None]
            for k in range(n_snap)]
    else:
        seed = np.load(args.restart_ic).astype(np.float64)
        if seed.ndim == 3:
            seed = seed[0]
        om_seed = torch.tensor(seed, dtype=dtype, device=device)[None]
        h_uf = Delta_T / 200.0
        n_uf = int(round(Delta_T / h_uf))
        marks, cur = [om_seed.clone()], om_seed.clone()
        for m in range((n_snap - 1) * n_uf):
            cur = rk4_step(cur, h_uf, derivative, L_hat, F_phys)
            if (m + 1) % n_uf == 0:
                marks.append(cur.clone())
        omega_stack = marks[::-1]
    psi_stack = [psi_from_omega(o, derivative) for o in omega_stack]
    om_rms = float(torch.sqrt((omega_stack[0] ** 2).mean()))
    print(f'[cmp] grid {Ny}x{Nx} L={Lx:.4f} nu={nu} mu={mu} beta={beta}')
    print(f'[cmp] Delta_T={Delta_T} steps={args.n_steps} model={model_name} '
          f'|omega_0|_rms={om_rms:.6e}')

    # ---- the two steppers: run_arm's OWN one_step, no scheme code here --- #
    common = dict(dealias_nn=args.dealias_nn, include_r4=args.r4,
                  nn_gamma=args.nn_gamma, nn_kcut=args.nn_kcut,
                  nn_clip=args.nn_clip,
                  nn_project_radius=args.nn_project_radius,
                  closure_apply=args.closure_apply,
                  return_stepper=True, return_capture=True)
    step_an, cap_an = run_arm('r3anal', omega_stack, psi_stack, Delta_T,
                              args.n_steps, [], derivative, L_hat, F_hat,
                              device, instrument={}, **common)
    step_cl, cap_cl = run_arm('closure', omega_stack, psi_stack, Delta_T,
                              args.n_steps, [], derivative, L_hat, F_hat,
                              device, model=model, input_fields=input_fields,
                              nn_dissipative_proj=args.nn_dissipative_proj,
                              instrument={}, **common)

    band_edges = (_default_band_edges(Nx) if args.fnn_band_edges is None
                  else tuple(int(x) for x in args.fnn_band_edges.split(',')))
    print(f'[cmp] radial bands (mode radius) {band_edges} '
          f'{"[grid-derived]" if args.fnn_band_edges is None else "[explicit]"}')
    bands, wb = _fnn_bands(Ny, Nx, device, band_edges)
    w_par = _parseval_weight(to_spectral(omega_stack[0]), Nx)

    # both arms start from the SAME IC and history
    q_an, qm_an = to_spectral(omega_stack[0]), to_spectral(omega_stack[1])
    q_cl, qm_cl = q_an.clone(), qm_an.clone()
    N_an, Nm_an = (N_spectral(q_an, derivative, F_hat),
                   N_spectral(qm_an, derivative, F_hat))
    N_cl, Nm_cl = N_an.clone(), Nm_an.clone()
    om_cl = [s.clone() for s in omega_stack]
    ps_cl = [s.clone() for s in psi_stack]

    def corr_rms(cap):
        """rms of the TOTAL applied closure correction of the step just taken
        (analytic + NN + folded L^3, summed as FIELDS -- the arm's own
        instrumentation capture)."""
        c = cap[0] or {}
        tot = None
        for k in ('Ca', 'Cn', 'Ci'):
            t = c.get(k)
            if t is not None:
                tot = t if tot is None else tot + t
        return 0.0 if tot is None else _spec_rms(tot, wb)

    rows = []
    onset = {t: None for t in _THRESH}
    print(f'\n{"step":>5}{"t":>9}{"rel":>12}' +
          ''.join(f'{"d_" + b:>12}' for b in _FNN_BAND_NAMES) +
          f'{"Cclos":>12}{"Canal":>12}{"cfl_cl":>9}{"cfl_an":>9}')
    for s in range(1, args.n_steps + 1):
        # r3anal is not a closure arm: one_step never reads om/ps for it
        # (guarded by `if is_clos:`), so pass None rather than the OTHER arm's
        # history -- correct today and safe if the analytic arm ever gains a
        # history consumer.
        qn_an, Nn_an, _, _ = step_an(q_an, qm_an, N_an, Nm_an, None, None)
        C_an = corr_rms(cap_an)
        qn_cl, Nn_cl, on_cl, pn_cl = step_cl(q_cl, qm_cl, N_cl, Nm_cl,
                                             om_cl, ps_cl)
        C_cl = corr_rms(cap_cl)
        qm_an, q_an, Nm_an, N_an = q_an, qn_an, N_an, Nn_an
        qm_cl, q_cl, Nm_cl, N_cl = q_cl, qn_cl, N_cl, Nn_cl
        om_cl = [on_cl] + om_cl[:-1]
        ps_cl = [pn_cl] + ps_cl[:-1]

        fin = bool(torch.isfinite(q_cl.real).all()
                   and torch.isfinite(q_cl.imag).all())
        fin_an = bool(torch.isfinite(q_an.real).all()
                      and torch.isfinite(q_an.imag).all())
        d = q_cl - q_an
        rms_an = _spec_rms(q_an, wb)
        row = dict(step=s, t=s * Delta_T,
                   rms_diff=_spec_rms(d, wb) if fin else float('nan'),
                   rms_anal=rms_an)
        row['rel'] = row['rms_diff'] / max(rms_an, 1e-300)
        for nm, m in zip(_FNN_BAND_NAMES, bands):
            row[f'd_{nm}'] = _spec_rms(d * m, wb) if fin else float('nan')
        row['Cclos'] = C_cl
        row['Canal'] = C_an
        E_cl, Z_cl = scalars_from_qh(q_cl, derivative, w_par)
        E_an, Z_an = scalars_from_qh(q_an, derivative, w_par)
        row['Z_clos'], row['Z_anal'] = Z_cl, Z_an
        row['cfl_clos'] = (cfl_from_qh(q_cl, derivative, Delta_T, ra._DX,
                                       ra._DY) if fin else float('nan'))
        row['cfl_anal'] = (cfl_from_qh(q_an, derivative, Delta_T, ra._DX,
                                       ra._DY) if fin_an else float('nan'))
        rows.append(row)
        for t in _THRESH:
            if onset[t] is None and np.isfinite(row['rel']) and row['rel'] > t:
                onset[t] = s
        print(f'{s:>5}{row["t"]:>9.4f}{row["rel"]:>12.4e}' +
              ''.join(f'{row["d_" + b]:>12.4e}' for b in _FNN_BAND_NAMES) +
              f'{C_cl:>12.4e}{C_an:>12.4e}'
              f'{row["cfl_clos"]:>9.3f}{row["cfl_anal"]:>9.3f}', flush=True)
        if not (fin and fin_an):
            print(f'[cmp] {"closure" if not fin else "r3anal"} arm non-finite '
                  f'at step {s} -- stopping.')
            break

    print('\n[cmp] DIVERGENCE ONSET (first step with '
          'rms(w_clos - w_anal)/rms(w_anal) above threshold):')
    for t in _THRESH:
        print(f'        > {t:.0e} : '
              + ('never in this horizon' if onset[t] is None
                 else f'step {onset[t]} (t={onset[t] * Delta_T:.4f})'))

    if not rows:
        sys.exit('[cmp] no steps recorded (--n-steps must be >= 1)')
    csv_path = out_dir / f'compare_arms_{args.tag}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(out_dir / f'compare_arms_{args.tag}.npz',
             Delta_T=np.float64(Delta_T), n_steps=np.int64(args.n_steps),
             bands=np.asarray(_FNN_BAND_NAMES),
             band_edges=np.asarray(band_edges, np.int64),
             onset_thresholds=np.asarray(_THRESH),
             onset_steps=np.asarray([-1 if onset[t] is None else onset[t]
                                     for t in _THRESH], np.int64),
             **{k: np.asarray([r[k] for r in rows], np.float64)
                for k in rows[0].keys()})
    print(f'[cmp] wrote {csv_path.name} + .npz in {out_dir}')


if __name__ == '__main__':
    main()
