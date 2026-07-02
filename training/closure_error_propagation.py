#!/usr/bin/env python
r"""
closure_error_propagation.py
============================
Given per-operator RELATIVE errors on the learned N-derivatives
(eps_Ndot, eps_Nddot, eps_N3dot), quantify how they propagate through the
analytic closure assembly to the final increment delta.

The closure (orders p, scheme s):
    delta = sum_p (Delta_T^p / D_p) * R_p,
    R_p   = sum_k a_{p,k} L^{p-k} * field_k,
            field_0 = omega, field_1 = N, field_{k>=2} = N^{(k-1)}.
Only k>=2 terms are LEARNED (carry error); k=0,1 are exact (built from omega,N).
The error in delta is therefore the L^{p-k}-WEIGHTED sum of the per-op errors:

    ||contribution_{p,k}|| = (Delta_T^p/D_p) |a_{p,k}| eps_{N^{(k-1)}} ||L^{p-k} N^{(k-1)}||

reported as a fraction of ||delta||. The point: a 3% error on N_ddot (k=3, L^0,
no amplification) and a 3% error on N_dot inside R4 (k=2, L^2, strong
amplification) contribute VERY differently. This is the Sense-A amplification
made quantitative -- the loss-on-derivatives (val rel/op ~ 0.03) is NOT the
closure error; the closure error is this operator-weighted combination.

Imports the VERIFIED builder functions so conventions match the training data
exactly. Run from .../training (so build_training_data_fixD_v2 + closure_operators
+ the qg package are importable).

Usage:
  python closure_error_propagation.py MEMBER_DIR
      [--eps 0.026 0.030 0.040] [--scheme ab2cn2] [--orders 3 4]
      [--dt DT] [--n-samples 32] [--forcing forcing.npy] [--device cuda]
  MEMBER_DIR : a sliced sweep_dT_* dir (has manifest.json + packed/inputs.npy)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical
from build_training_data_fixD_v2 import (build_L_hat, J_phys, L_op,
                                         compute_n_dot_analytical,
                                         compute_n_ddot_analytical)
from closure_operators import SCHEMES

DERIV_NAME = {1: 'N_dot', 2: 'N_ddot', 3: 'N_3dot', 4: 'N_4dot'}


def _Lk(field, L_hat, k):
    if k == 0:
        return field
    return to_physical((L_hat ** k) * to_spectral(field))


def compute_N(omega, derivative, F_phys):
    psi = to_physical(derivative.inv_laplacian * to_spectral(omega))
    N = -J_phys(psi, omega, derivative)
    if F_phys is not None:
        N = N + F_phys
    return N, psi


def compute_n_3dot(omega, derivative, L_hat, F_phys):
    """Extend the builder's chain rule one order: N''' (verified pattern)."""
    psi = to_physical(derivative.inv_laplacian * to_spectral(omega))
    N, _ = compute_N(omega, derivative, F_phys)
    omega_d = L_op(omega, L_hat) + N
    psi_d = to_physical(derivative.inv_laplacian * to_spectral(omega_d))
    N_d = -J_phys(psi_d, omega, derivative) - J_phys(psi, omega_d, derivative)
    omega_dd = L_op(omega_d, L_hat) + N_d
    psi_dd = to_physical(derivative.inv_laplacian * to_spectral(omega_dd))
    N_dd = (-J_phys(psi_dd, omega, derivative)
            - 2.0 * J_phys(psi_d, omega_d, derivative)
            - J_phys(psi, omega_dd, derivative))
    omega_3 = L_op(omega_dd, L_hat) + N_dd
    psi_3 = to_physical(derivative.inv_laplacian * to_spectral(omega_3))
    N_3 = (-J_phys(psi_3, omega, derivative)
           - 3.0 * J_phys(psi_dd, omega_d, derivative)
           - 3.0 * J_phys(psi_d, omega_dd, derivative)
           - J_phys(psi, omega_3, derivative))
    return N_3


def fro(x):
    return float(torch.sqrt((x.double() ** 2).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('member', type=Path)
    ap.add_argument('--eps', type=float, nargs='+', default=[0.026, 0.030, 0.040],
                    help='relative errors for N_dot, N_ddot, N_3dot, ...')
    ap.add_argument('--scheme', default='ab2cn2')
    ap.add_argument('--orders', type=int, nargs='+', default=[3, 4])
    ap.add_argument('--dt', type=float, default=None, help='override manifest Delta_T')
    ap.add_argument('--n-samples', type=int, default=32)
    ap.add_argument('--forcing', type=Path, default=None,
                    help='optional forcing field .npy (forced members; else F=0)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    man = json.loads((args.member / 'manifest.json').read_text())
    Nx, Ny = int(man['Nx']), int(man['Ny'])
    Lx, Ly = float(man['Lx']), float(man['Ly'])
    nu, mu, beta = float(man['nu']), float(man['mu']), float(man['beta'])
    h = float(args.dt if args.dt is not None else man['Delta_T'])
    dev = args.device if torch.cuda.is_available() else 'cpu'

    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=dev, precision='float64')
    derivative = Derivative(grid)
    for a in ('dx', 'dy', 'laplacian', 'inv_laplacian'):
        setattr(derivative, a, getattr(derivative, a).to(dev))
    L_hat = build_L_hat(derivative, nu=nu, mu=mu, B=beta).to(dev)

    F_phys = None
    if args.forcing is not None:
        F_phys = torch.tensor(np.load(args.forcing), dtype=torch.float64, device=dev)
    elif man.get('family', '') == 'forced' or beta != 0.0 and 'FRC' in str(args.member):
        print("  ! forced member but no --forcing given: using F=0 "
              "(N-derivative high-k structure is cascade-dominated, so magnitudes "
              "are representative; pass --forcing for the exact field).")

    inp = np.load(args.member / 'packed' / 'inputs.npy', mmap_mode='r')   # (Ns,2S,Ny,Nx)
    ns = min(args.n_samples, inp.shape[0])
    omega = torch.tensor(np.ascontiguousarray(inp[:ns, 0]),               # ch 0 = omega_0
                         dtype=torch.float64, device=dev)                  # (ns,Ny,Nx)

    # ---- true fields + N-derivatives (analytic, builder conventions) ---- #
    N, _ = compute_N(omega, derivative, F_phys)
    Ndot = compute_n_dot_analytical(omega, derivative, L_hat, F_phys)
    Nddot = compute_n_ddot_analytical(omega, derivative, L_hat, F_phys)
    N3 = compute_n_3dot(omega, derivative, L_hat, F_phys)
    Nderivs = [Ndot, Nddot, N3]                                  # N^(1..3)
    eps_of = {2: args.eps[0], 3: args.eps[1], 4: args.eps[2] if len(args.eps) > 2 else 0.0}

    coeffs_all = SCHEMES[args.scheme]['coeffs']
    denom_all = SCHEMES[args.scheme]['denom']

    # ---- assemble the full TRUE delta and the per-term error contributions ---- #
    delta = torch.zeros_like(omega)
    rows = []                       # (p,k,deriv,|weighted term|, eps, err_norm)
    err_field = torch.zeros_like(omega)   # correlated (aligned-error) worst case
    for p in args.orders:
        c = coeffs_all[p]; w = (h ** p) / denom_all[p]
        for k in range(0, p + 1):
            if k >= len(c) or c[k] == 0:
                continue
            field = omega if k == 0 else (N if k == 1 else Nderivs[k - 2])
            Lf = _Lk(field, L_hat, p - k)
            term = w * c[k] * Lf
            delta = delta + term
            if k >= 2:                          # learned -> carries error
                e = eps_of[k]
                tnorm = fro(term)               # ||weighted term||
                enorm = e * tnorm               # ||error from this term|| (rel-shape)
                err_field = err_field + e * term
                rows.append((p, k, DERIV_NAME[k - 1], tnorm, e, enorm))

    dnorm = fro(delta)

    # ---------------------------- report ---------------------------- #
    print(f"\nmember = {args.member.name}   scheme={args.scheme}  orders={args.orders}")
    print(f"  grid {Ny}x{Nx}  L={Lx:.4g}  nu={nu:.3g} mu={mu:.3g} beta={beta:.3g}  "
          f"Delta_T={h:g}  (avg over {ns} samples)")
    print(f"  eps: N_dot={args.eps[0]:.3f}  N_ddot={args.eps[1]:.3f}  "
          f"N_3dot={args.eps[2] if len(args.eps) > 2 else 0:.3f}\n")
    print(f"  ||delta|| (full assembled closure) = {dnorm:.4e}\n")

    print(f"  {'order':>5} {'term':>14} {'L^pow':>6} {'|coef|':>7} "
          f"{'||weighted term||':>18} {'eps':>6} {'err contrib':>12} {'/||delta||':>11}")
    print('  ' + '-' * 86)
    sq = 0.0; lin = 0.0
    for (p, k, nm, tnorm, e, enorm) in rows:
        rel = enorm / dnorm
        sq += enorm ** 2; lin += enorm
        print(f"  {p:>5} {f'L^{p-k} {nm}':>14} {p-k:>6} {abs(coeffs_all[p][k]):>7} "
              f"{tnorm:>18.4e} {e:>6.3f} {enorm:>12.4e} {rel:>11.2%}")
    print('  ' + '-' * 86)
    rms = (sq ** 0.5) / dnorm
    l1 = lin / dnorm
    corr = fro(err_field) / dnorm
    print(f"  closure error / ||delta||:")
    print(f"     RMS (independent errors)         = {rms:.2%}")
    print(f"     L1  (worst case, all aligned)    = {l1:.2%}")
    print(f"     correlated field (err aligned w/ terms) = {corr:.2%}")

    # explicit R3 / R4 error expressions
    print(f"\n  explicit error fields (||.||/||delta||):")
    for p in args.orders:
        c = coeffs_all[p]; w = (h ** p) / denom_all[p]
        ef = torch.zeros_like(omega); pieces = []
        for k in range(2, p + 1):
            if k >= len(c) or c[k] == 0:
                continue
            ef = ef + w * c[k] * eps_of[k] * _Lk(Nderivs[k - 2], L_hat, p - k)
            pieces.append(f"{c[k]:+g} L^{p-k}({eps_of[k]:.3f}*{DERIV_NAME[k-1]})")
        if pieces:
            print(f"     dR{p} = (dT^{p}/{denom_all[p]}) [ " + " ".join(pieces)
                  + f" ]   ->  {fro(ef)/dnorm:.2%}")


if __name__ == '__main__':
    main()
