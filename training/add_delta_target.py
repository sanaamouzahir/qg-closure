"""
add_delta_target.py

Augment an already-built packed ensemble with the EMPIRICAL temporal-closure
target

    delta = RK4_fine(omega_0)  -  AB2CN2_coarse(omega_0, omega_m1)

i.e. the per-step correction that, added to a bare AB2CN2 step at Delta_T, lands
on the RK4 trajectory subcycled at Delta_T/K (>= 4th order, ~ exact). The network
learns delta directly => all-orders, free at inference (no T5/T6 Jacobian cascade,
no modified-equation Delta_T* wall).

Why this needs no DNS rerun
---------------------------
build_training_data_mmap.py already laid omega_0 and its lags as RK4(h_ultrafine)
states at Delta_T spacing -- the stable fine reference. So both halves of delta
come from the packed inputs that already exist:
  * AB2CN2_coarse : one ab2cn2_step_spectral(omega_0, omega_m1, Delta_T)
  * RK4_fine      : K = Delta_T/h_fine self-starting RK4 steps from omega_0
We reuse the builder's exact operators so dealiasing / L_hat / forcing match the
build bit-for-bit.

delta is float64 (it is a ~1e-9 difference of two O(1) fields -- the same
cancellation that forced e_total to float64; |delta| ~ 2.7e-9 at Delta_T=1e-3,
matching the verified e_total). It is written to packed/delta_f64.npy as (N,1,Ny,Nx)
alongside the float32 inputs/targets; the manifest records it under extra_targets.

Reference choice: RK4 (default) is self-starting -- cleaner than the AB2CN2-fine
reference in the builder diagnostics, which seeds its previous level with omega_m1
(a full Delta_T back). At K=100 the two differ by ~1e-13.  --ref ab2cn2 reproduces
the old e_total convention if you want to compare.

Usage
-----
    python add_delta_target.py <ENSEMBLE_ROOT>            # all members
    python add_delta_target.py <ENSEMBLE_ROOT> --members FRC-b2,DEC-hiRe
    python add_delta_target.py <ENSEMBLE_ROOT> --ref ab2cn2 --chunk 16
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical
# reuse the builder's operators verbatim so conventions match the build
from build_training_data_mmap import ab2cn2_step_spectral, rk4_step, build_L_hat


def build_F_phys(forcing, Lx, Ly, Nx, Ny, device, dtype):
    """Rebuild the static cosine forcing from manifest['forcing'] (or None)."""
    if not forcing or not isinstance(forcing, dict):
        return None
    if forcing.get('function') != 'unscaled_cosine':
        raise ValueError(f"unsupported forcing {forcing.get('function')!r}")
    A = float(forcing.get('A', 0.0)); B = float(forcing.get('B', 0.0))
    D = float(forcing.get('D', 0.0)); E = float(forcing.get('E', 0.0))
    x = torch.linspace(0, Lx, Nx, device=device, dtype=dtype)
    y = torch.linspace(0, Ly, Ny, device=device, dtype=dtype)
    return (A * torch.cos(B * x[None, :]) + D * torch.cos(E * y[:, None]))[None]


def setup_solver(man, device):
    Nx, Ny = int(man['Nx']), int(man['Ny'])
    Lx, Ly = float(man['Lx']), float(man['Ly'])
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision='float64')
    der = Derivative(grid)
    for a in ('dx', 'dy', 'laplacian', 'inv_laplacian'):
        setattr(der, a, getattr(der, a).to(device))
    L_hat = build_L_hat(der, nu=float(man['nu']), mu=float(man['mu']),
                        B=float(man['beta'])).to(device)
    F = build_F_phys(man.get('forcing'), Lx, Ly, Nx, Ny, device, torch.float64)
    return der, L_hat, F


def rk4_fine(om, K, h_fine, der, L_hat, F):
    for _ in range(K):
        om = rk4_step(om, h_fine, der, L_hat, F)
    return om


def ab2cn2_fine(qh0, qhm1, K, h_fine, der, L_hat, F):
    qm, qc = qhm1, qh0
    for _ in range(K):
        qn = ab2cn2_step_spectral(qc, qm, h_fine, der, L_hat, F)
        qm, qc = qc, qn
    return to_physical(qc)


def process_member(root: Path, ref: str, chunk: int, device: str):
    man = json.loads((root / 'manifest.json').read_text())
    der, L_hat, F = setup_solver(man, device)
    dT = float(man['Delta_T']); h_fine = float(man['h_fine'])
    K = int(round(dT / h_fine))
    Ny, Nx = int(man['Ny']), int(man['Nx'])

    inp = np.load(root / 'packed' / 'inputs.npy', mmap_mode='r')   # (N,14,Ny,Nx)
    N = inp.shape[0]
    out = open_memmap(root / 'packed' / 'delta_f64.npy', mode='w+',
                      dtype=np.float64, shape=(N, 1, Ny, Nx))

    rms_acc = 0.0
    t0 = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        om0 = torch.as_tensor(np.ascontiguousarray(inp[s:e, 0]),
                              dtype=torch.float64, device=device)
        omm1 = torch.as_tensor(np.ascontiguousarray(inp[s:e, 1]),
                               dtype=torch.float64, device=device)
        qh0, qhm1 = to_spectral(om0), to_spectral(omm1)
        om_coarse = to_physical(ab2cn2_step_spectral(qh0, qhm1, dT, der, L_hat, F))
        if ref == 'rk4':
            om_fine = rk4_fine(om0.clone(), K, h_fine, der, L_hat, F)
        else:
            om_fine = ab2cn2_fine(qh0, qhm1, K, h_fine, der, L_hat, F)
        delta = (om_fine - om_coarse)                      # (b,Ny,Nx) float64
        out[s:e, 0] = delta.detach().cpu().numpy()
        rms_acc += float((delta ** 2).sum().item())
        if (s // chunk) % 10 == 0:
            print(f"    [{e:4d}/{N}]  |delta|_rms(running)="
                  f"{np.sqrt(rms_acc / (e * Ny * Nx)):.3e}", flush=True)
    out.flush()
    rms = float(np.sqrt(rms_acc / (N * Ny * Nx)))

    man.setdefault('extra_targets', {})['delta_f64'] = dict(
        file='packed/delta_f64.npy', dtype='float64', shape=[N, 1, Ny, Nx],
        ref=ref, K=K, h_fine=h_fine, Delta_T=dT, rms=rms,
        definition='RK4_fine(omega_0) - AB2CN2_coarse(omega_0, omega_m1)'
        if ref == 'rk4'
        else 'AB2CN2_fine(omega_0,omega_m1) - AB2CN2_coarse(omega_0, omega_m1)',
    )
    (root / 'manifest.json').write_text(json.dumps(man, indent=2))
    print(f"  -> wrote delta_f64.npy  N={N}  |delta|_rms={rms:.3e}  "
          f"({(time.time()-t0)/60:.1f} min)")
    return rms


def find_members(ensemble_root: Path):
    roots = []
    for man in ensemble_root.rglob('manifest.json'):
        if (man.parent / 'packed' / 'inputs.npy').exists():
            roots.append(man.parent)
    return sorted(set(roots))


def member_label(root: Path):
    try:
        man = json.loads((root / 'manifest.json').read_text())
        p = man.get('source_omega_path', '')
        return Path(p).parts[-2] if p else root.parent.name
    except Exception:
        return root.parent.name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ensemble_root', type=Path)
    ap.add_argument('--members', type=str, default=None,
                    help='comma list of member labels (from source path) to limit to')
    ap.add_argument('--ref', choices=['rk4', 'ab2cn2'], default='rk4')
    ap.add_argument('--chunk', type=int, default=16)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--skip-existing', action='store_true',
                    help='skip members that already have packed/delta_f64.npy')
    args = ap.parse_args()

    roots = find_members(args.ensemble_root)
    if not roots:
        raise SystemExit(f"no packed members under {args.ensemble_root}")
    want = set(s.strip() for s in args.members.split(',')) if args.members else None

    print(f"[delta] ref={args.ref}  members found={len(roots)}")
    for root in roots:
        lab = member_label(root)
        if want and lab not in want:
            continue
        if args.skip_existing and (root / 'packed' / 'delta_f64.npy').exists():
            print(f"[delta] {lab}: exists, skip")
            continue
        print(f"[delta] {lab}  ({root})")
        process_member(root, args.ref, args.chunk, args.device)
    print("[delta] done.")


if __name__ == '__main__':
    main()
