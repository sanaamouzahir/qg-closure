"""
add_deriv_targets.py
====================
Augment each sweep_dT_<tag>/packed/ with the analytic N-time-derivative targets
needed for the *derivative-loss* closure model (the pre-6.1.2 setup):

    packed/deriv_anal_f64.npy   (N, 3, Ny, Nx)  =  [Ndot, Nddot, N3dot]  (float64)

WHY THIS EXISTS
---------------
The delta-pivot slicer (slice_delta_sweep.py) packs ONLY inputs + delta_{exact,rk4}.
It builds delta as  w[a+j] - AB2CN2(w[a],w[a-j]) -- a snapshot minus one AB2 step --
and NEVER computes the analytic N-derivatives. The derivative-loss model supervises
[Ndot, Nddot, N3dot] directly, so those targets must be reconstructed.

They are an EXACT analytic function of the anchor vorticity omega_0 (= inputs[:,0],
the model's own input channel 0) via the chain rule -- no DNS, no re-slicing:

    omega_dot = L omega + N ,   psi_* = inv_lap omega_* ,   N = -J(psi,omega) + F
    Ndot   = -J(psi_dot,  omega)     - J(psi, omega_dot)
    Nddot  = -J(psi_ddot, omega) - 2 J(psi_dot, omega_dot) - J(psi, omega_ddot)
    N3dot  = -J(psi_3,    omega) - 3 J(psi_dd, omega_d) - 3 J(psi_d, omega_dd)
                                                          - J(psi, omega_3)

This reuses the VERIFIED builder functions (compute_n_dot/ddot_analytical from
build_training_data_fixD_v2) so the targets match the original training data
exactly; N3dot extends the same pattern one order (identical to the verified
extension in closure_error_propagation.py).

FORCING
-------
F is time-independent (Fdot=Fddot=0), so it never appears in a derivative term
directly, but it enters every omega_dot (N carries F), hence Nddot, N3dot depend
on it. F is rebuilt from the manifest's `forcing` dict (the same build_F_phys the
slicer used to build delta): forced members get the EXACT field, decaying members
get F=0 (no `forcing` key). No external file is needed.

Run from .../training (so build_training_data_fixD_v2 + the qg package import).

Usage:
    python add_deriv_targets.py <ENSEMBLE_ROOT> [--members FRC-b1,FRC-b2 ...]
    python add_deriv_targets.py data/ensemble_N5 --device cuda
    python add_deriv_targets.py data/ensemble_N5/FRC-b075/sweep_dT_1p5em2   # single dir
    # add --overwrite to rebuild existing deriv_anal_f64.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical
from build_training_data_fixD_v2 import (build_L_hat, J_phys, L_op,
                                         compute_n_dot_analytical,
                                         compute_n_ddot_analytical)


def build_F_phys(forcing, Lx, Ly, Nx, Ny, device, dtype):
    """Rebuild the forcing field from the manifest dict (matches slice_delta_sweep).
    F = A cos(B x) + D cos(E y).  Returns None for decaying members (no forcing)."""
    if not forcing or not isinstance(forcing, dict):
        return None
    A = float(forcing.get('A', 0.0)); B = float(forcing.get('B', 0.0))
    D = float(forcing.get('D', 0.0)); E = float(forcing.get('E', 0.0))
    x = torch.linspace(0, Lx, Nx, device=device, dtype=dtype)
    y = torch.linspace(0, Ly, Ny, device=device, dtype=dtype)
    return (A * torch.cos(B * x[None, :]) + D * torch.cos(E * y[:, None]))[None]


def compute_n_3dot(omega, derivative, L_hat, F_phys):
    """N''' by extending the builder's chain rule one order (verified pattern,
    identical to closure_error_propagation.py)."""
    psi = to_physical(derivative.inv_laplacian * to_spectral(omega))
    N = -J_phys(psi, omega, derivative)
    if F_phys is not None:
        N = N + F_phys
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


def setup_solver(man, device):
    Nx, Ny = int(man['Nx']), int(man['Ny']); Lx, Ly = float(man['Lx']), float(man['Ly'])
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision='float64')
    der = Derivative(grid)
    for a in ('dx', 'dy', 'laplacian', 'inv_laplacian'):
        setattr(der, a, getattr(der, a).to(device))
    L_hat = build_L_hat(der, nu=float(man['nu']), mu=float(man['mu']),
                        B=float(man['beta'])).to(device)
    F = build_F_phys(man.get('forcing'), Lx, Ly, Nx, Ny, device, torch.float64)
    return der, L_hat, F


def build_member(sweep_dir: Path, chunk: int, device: str, overwrite: bool):
    man = json.loads((sweep_dir / 'manifest.json').read_text())
    pdir = sweep_dir / 'packed'
    inp = np.load(pdir / 'inputs.npy', mmap_mode='r')          # (N, 2S, Ny, Nx)
    N = inp.shape[0]
    Ny, Nx = int(man['Ny']), int(man['Nx'])
    out_path = pdir / 'deriv_anal_f64.npy'
    if out_path.exists() and not overwrite:
        print(f"  [{sweep_dir.parent.name}/{sweep_dir.name}] exists, skip "
              f"(use --overwrite)"); return
    der, L_hat, F = setup_solver(man, device)
    forced = F is not None
    out = open_memmap(out_path, mode='w+', dtype=np.float64, shape=(N, 3, Ny, Nx))
    rms = np.zeros(3, dtype=np.float64); w = 0
    for s0 in range(0, N, chunk):
        e = min(s0 + chunk, N)
        omega = torch.as_tensor(np.ascontiguousarray(inp[s0:e, 0]),   # ch 0 = omega_0
                                dtype=torch.float64, device=device)    # (b,Ny,Nx)
        Ndot = compute_n_dot_analytical(omega, der, L_hat, F)
        Nddot = compute_n_ddot_analytical(omega, der, L_hat, F)
        N3 = compute_n_3dot(omega, der, L_hat, F)
        block = torch.stack([Ndot, Nddot, N3], dim=1)               # (b,3,Ny,Nx)
        out[s0:e] = block.detach().cpu().numpy()
        rms += (block.double() ** 2).sum(dim=(0, 2, 3)).cpu().numpy()
        w += (e - s0) * Ny * Nx
    out.flush()
    rms = np.sqrt(rms / max(w, 1))
    # record the targets in the manifest so the loader/trainer can find them
    man.setdefault('extra_targets', {})
    man['extra_targets']['deriv_anal'] = dict(
        file='packed/deriv_anal_f64.npy', dtype='float64', shape=[N, 3, Ny, Nx],
        orders=['Ndot', 'Nddot', 'N3dot'], forced=bool(forced),
        rms=[float(r) for r in rms],
        definition='analytic N^(1..3) from omega_0 via chain rule '
                   '(F from manifest forcing; F=0 if absent)')
    (sweep_dir / 'manifest.json').write_text(json.dumps(man, indent=2))
    print(f"  [{sweep_dir.parent.name}/{sweep_dir.name}] N={N} forced={forced}  "
          f"|Ndot|={rms[0]:.3e} |Nddot|={rms[1]:.3e} |N3dot|={rms[2]:.3e}  -> {out_path.name}")


def find_sweeps(root: Path):
    """All sweep_dT_* dirs under root (or root itself if it is one)."""
    if (root / 'packed' / 'inputs.npy').exists() and 'sweep_dT_' in root.name:
        return [root]
    return sorted(d.parent for d in root.rglob('manifest.json')
                  if 'sweep_dT_' in d.parent.name
                  and (d.parent / 'packed' / 'inputs.npy').exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path, help='ensemble root, a member dir, or a sweep_dT_* dir')
    ap.add_argument('--members', type=str, default=None,
                    help='comma list to restrict (matched against the member dir name)')
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else 'cpu'
    want = set(s.strip() for s in args.members.split(',')) if args.members else None
    sweeps = find_sweeps(args.root)
    if want:
        sweeps = [s for s in sweeps if s.parent.name in want]
    if not sweeps:
        raise SystemExit(f"no sweep_dT_* dirs with packed/inputs.npy under {args.root}")
    print(f"[deriv-targets] {len(sweeps)} sweep dir(s)  device={dev}")
    for s in sweeps:
        build_member(s, args.chunk, dev, args.overwrite)
    print("[deriv-targets] done.")


if __name__ == '__main__':
    main()
