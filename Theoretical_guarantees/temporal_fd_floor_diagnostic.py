"""
temporal_fd_floor_diagnostic.py

Isolate the TEMPORAL finite-difference floor of the cheap-deriv closure --
with NO model in the loop.

The question this answers
-------------------------
The trained 4-snapshot model plateaus at ~3% rel-L2 on Nddot (and similar on
Ndot / N3dot). Two competing explanations:

  (A) the wall is the 4-snapshot TIME stencil: from 4 levels spaced Delta_T,
      omega_ddot is only O(Delta_T^2) accurate and omega_3dot only O(Delta_T^1),
      so even a PERFECT spatial mixing cannot reconstruct N^(m) better than the
      FD time-derivatives allow.  --> the fix is more snapshots (7).
  (B) the wall is the model's learned mix / FD spatial grads / capacity, and a
      richer model (corrector, etc.) on the SAME 4 snapshots would do better.
      --> the fix is the corrector, not more data.

This script measures (A) directly. It takes the stored snapshots, forms the
time-derivatives with the model's EXACT TimeFD stencil (model_deriv_closure.py),
then assembles N^(m) using the builder's EXACT dealiased spectral Jacobian
(build_training_data_mmap.J_phys) and exact spectral inverse-Laplacian for the
psi-family -- i.e. perfect spatial operators. The ONLY difference from the
analytic targets is FD-in-time vs PDE-exact time-derivatives, so the rel-L2 it
reports IS the temporal-FD floor.

Read it as: if floor(n=4, Nddot) ~ 3%, the 4-snapshot stencil is the wall and
the corrector cannot beat it (build the 7-snapshot set). If floor(n=4) << 3%,
the temporal stencil has headroom the model isn't using -> the 3% is model
capacity / spatial path, and the corrector is the right lever.

It sweeps n_time from 3 (minimum for Nddot) up to whatever the packed set
contains, so on a 7-snapshot set it prints floor(4), floor(5), floor(6),
floor(7) in one shot -- the predicted gain from rebuilding at 7 snapshots.

Faithfulness
------------
* TimeFD weights: backward-FD Vandermonde on nodes x_j=-j*Delta_T, row k = the
  order-k stencil (already /Delta_T^k). Byte-identical to TimeFD in
  model_deriv_closure.py.
* J_phys: verbatim copy of build_training_data_mmap.J_phys (2/3-rule dealiased
  flux-form Jacobian) -- the SAME operator that built the targets.
* psi^(k): obtained by the model's TimeFD on the stored psi snapshots, which
  (psi = inv_lap omega, a linear time-independent operator) equals
  inv_lap(omega^(k)) exactly -- so the spatial side is exact.
* metric: per-sample ||fd - analytic|| / ||analytic||, averaged over samples --
  identical to train.py's relative_l2_perchannel_vec, so numbers are directly
  comparable to the reported val percentages.

Usage
-----
    python temporal_fd_floor_diagnostic.py \
        --root-dir .../_4snap_staging/forced_turbulence_dT_1em3 \
        --device cuda --n-samples 64 --split val

    # once the 7-snapshot set is built, same command on that root prints the
    # floor for n=3..7.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical


# --------------------------------------------------------------------------- #
# Spatial operator: VERBATIM copy of build_training_data_mmap.J_phys.          #
# Identical dealiased flux-form Jacobian that built the analytic targets, so   #
# the only difference vs the targets is the time-derivatives (FD vs PDE-exact).#
# --------------------------------------------------------------------------- #
def J_phys(psi_phys, omega_phys, derivative):
    psih = to_spectral(psi_phys)
    qh = to_spectral(omega_phys)
    u = to_physical(-1 * derivative.dy * psih)
    v = to_physical(+1 * derivative.dx * psih)
    q = to_physical(qh)
    uq_h = to_spectral(u * q).clone()
    vq_h = to_spectral(v * q).clone()
    derivative.dealias(uq_h)
    derivative.dealias(vq_h)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h
    return to_physical(j_hat)


# --------------------------------------------------------------------------- #
# Time-FD weights: EXACT replica of model_deriv_closure.TimeFD.                #
# --------------------------------------------------------------------------- #
def timefd_weights(n_time: int, dt: float) -> np.ndarray:
    """(nt, nt) backward-FD Vandermonde weights; row k = order-k stencil (/dt^k).

    Nodes x_j = -j*dt, j=0..nt-1; A[m,j] = x_j^m / m!; W = inv(A).T.
    """
    x = np.array([-j * dt for j in range(n_time)], dtype=np.float64)
    A = np.array([[x[j] ** m / math.factorial(m) for j in range(n_time)]
                  for m in range(n_time)], dtype=np.float64)
    return np.linalg.inv(A).T


def _lag_names(prefix: str, n: int):
    return [f'{prefix}_0'] + [f'{prefix}_m{k}' for k in range(1, n)]


# N^(m) -> stored analytic target field name (m = derivative order)
_TGT_OF_ORDER = {1: 'N_dot_0_anal', 2: 'N_ddot_0_anal', 3: 'N_3dot_0_anal'}
_LABEL_OF_ORDER = {1: 'Ndot ', 2: 'Nddot', 3: 'N3dot'}


def fd_n_derivatives(omega_snaps, psi_snaps, W, derivative, max_m):
    """FD-in-time N^(m) for m=1..max_m, exact spectral J, given physical snaps.

    omega_snaps / psi_snaps: lists of n (Ny,Nx) tensors ordered [t0, t-1, ...].
    W: (n,n) TimeFD weights. Returns {m: N^(m)} via the chain-rule binomial sum
        N^(m) = - sum_{j=0}^m C(m,j) J(psi^(m-j), omega^(j))
    -- identical assembly to compute_n_derivatives in the builder (F drops out
    for m>=1 since forcing is static).
    """
    n = len(omega_snaps)
    dev = omega_snaps[0].device
    dt_t = omega_snaps[0].dtype
    Wt = torch.from_numpy(W).to(dtype=dt_t, device=dev)
    # time-derivative orders 0..n-1
    omega_ord = [sum(Wt[k, j] * omega_snaps[j] for j in range(n)) for k in range(n)]
    psi_ord = [sum(Wt[k, j] * psi_snaps[j] for j in range(n)) for k in range(n)]
    out = {}
    for m in range(1, max_m + 1):
        Nm = torch.zeros_like(omega_snaps[0])
        for j in range(0, m + 1):
            i = m - j  # psi order
            Nm = Nm - math.comb(m, j) * J_phys(psi_ord[i], omega_ord[j], derivative)
        out[m] = Nm
    return out


def _meta_lookup(root: Path, pdir: Path):
    """Merge pack_meta.json (Lx/Ly/Delta_T/fields) with manifest.json fallback."""
    meta = {}
    with open(pdir / 'pack_meta.json') as f:
        meta.update(json.load(f))
    man_path = root / 'manifest.json'
    if man_path.exists():
        with open(man_path) as f:
            man = json.load(f)
        for k in ('Lx', 'Ly', 'Nx', 'Ny', 'Delta_T',
                  'input_fields', 'target_fields'):
            meta.setdefault(k, man.get(k))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root-dir', required=True, help='packed dataset root')
    ap.add_argument('--packed-subdir', default='packed')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--n-samples', type=int, default=64,
                    help='random samples from the split (<=0 -> all)')
    ap.add_argument('--n-list', default='',
                    help='comma list of n_time values; default sweeps 3..n_avail')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root_dir)
    pdir = root / args.packed_subdir
    meta = _meta_lookup(root, pdir)
    Lx, Ly = float(meta['Lx']), float(meta['Ly'])
    Nx, Ny = int(meta['Nx']), int(meta['Ny'])
    Delta_T = float(meta['Delta_T'])
    input_fields = list(meta['input_fields'])
    target_fields = list(meta['target_fields'])

    n_avail = sum(1 for f in input_fields if f.startswith('omega_'))
    if n_avail < 3:
        raise SystemExit(f"need >=3 omega snapshots for Nddot; packed set has {n_avail}")

    omega_cols = [input_fields.index(nm) for nm in _lag_names('omega', n_avail)]
    psi_cols = [input_fields.index(nm) for nm in _lag_names('psi', n_avail)]
    tgt_cols = {m: target_fields.index(name)
                for m, name in _TGT_OF_ORDER.items() if name in target_fields}

    if args.n_list.strip():
        n_values = [int(s) for s in args.n_list.split(',') if s.strip()]
    else:
        n_values = list(range(3, n_avail + 1))
    n_values = [n for n in n_values if 3 <= n <= n_avail]

    # sample indices
    split_path = pdir / 'split.npz'
    if not split_path.exists():
        split_path = root / 'split.npz'
    with np.load(split_path) as sp:
        if args.split == 'all':
            idx = np.concatenate([sp[k] for k in sp.files if k.endswith('_idx')])
        else:
            idx = sp[f'{args.split}_idx']
    idx = np.asarray(idx).astype(np.int64)
    if args.n_samples > 0 and len(idx) > args.n_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(idx, args.n_samples, replace=False)
    idx = np.sort(idx)

    # grid / spectral operators (float64, matching the build)
    device = args.device
    dtype = torch.float64
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision='float64')
    derivative = Derivative(grid)
    derivative.dx = derivative.dx.to(device)
    derivative.dy = derivative.dy.to(device)
    derivative.laplacian = derivative.laplacian.to(device)
    derivative.inv_laplacian = derivative.inv_laplacian.to(device)

    X = np.load(pdir / 'inputs.npy', mmap_mode='r')   # (N, C_in, Ny, Nx)
    Y = np.load(pdir / 'targets.npy', mmap_mode='r')  # (N, C_out, Ny, Nx)

    print(f"[fd-floor] root={root}")
    print(f"[fd-floor] grid=({Ny},{Nx}) Lx={Lx} Ly={Ly} Delta_T={Delta_T:g} "
          f"n_avail={n_avail} split={args.split} samples={len(idx)}")
    print(f"[fd-floor] sweeping n_time = {n_values}")
    print(f"[fd-floor] metric = per-sample rel-L2, mean over samples "
          f"(== train.py relative_l2_perchannel_vec)\n")

    # accumulate per (n, m): list of per-sample rel-L2
    per_sample = {(n, m): [] for n in n_values for m in range(1, min(3, n - 1) + 1)
                  if m in tgt_cols}

    for n in n_values:
        W = timefd_weights(n, Delta_T)
        max_m = min(3, n - 1)
        for s in idx:
            s = int(s)
            omega_snaps = [torch.as_tensor(np.asarray(X[s, omega_cols[k]]),
                                           dtype=dtype, device=device)
                           for k in range(n)]
            psi_snaps = [torch.as_tensor(np.asarray(X[s, psi_cols[k]]),
                                         dtype=dtype, device=device)
                         for k in range(n)]
            fd = fd_n_derivatives(omega_snaps, psi_snaps, W, derivative, max_m)
            for m in range(1, max_m + 1):
                if m not in tgt_cols:
                    continue
                tgt = torch.as_tensor(np.asarray(Y[s, tgt_cols[m]]),
                                      dtype=dtype, device=device)
                num = torch.linalg.vector_norm((fd[m] - tgt).flatten())
                den = torch.linalg.vector_norm(tgt.flatten()).clamp_min(1e-30)
                per_sample[(n, m)].append((num / den).item())

    # report
    print(f"{'n_time':>6} | " + " | ".join(f"{_LABEL_OF_ORDER[m]:>16}"
                                            for m in (1, 2, 3) if m in tgt_cols))
    print("-" * (8 + 19 * sum(1 for m in (1, 2, 3) if m in tgt_cols)))
    for n in n_values:
        cells = []
        for m in (1, 2, 3):
            if m not in tgt_cols:
                continue
            key = (n, m)
            if key in per_sample and per_sample[key]:
                a = np.asarray(per_sample[key])
                cells.append(f"{a.mean()*100:6.2f}% (+/-{a.std()*100:4.2f})")
            else:
                cells.append(f"{'--':>16}")
        print(f"{n:>6} | " + " | ".join(cells))

    print("\n[fd-floor] interpretation:")
    print("  * floor at n=4 ~ the trained model's val %  -> the 4-snapshot TIME")
    print("    stencil is the wall; the corrector cannot beat it. Build 7 snaps.")
    print("  * floor at n=4 << trained val %             -> temporal stencil has")
    print("    headroom; the 3% is model capacity/spatial path -> corrector is the")
    print("    right lever (and the gain from 7 snaps would be smaller).")
    print("  * the n=4 -> n=7 drop (on a 7-snapshot set) is the predicted payoff")
    print("    of rebuilding, with perfect spatial ops as the ceiling.")


if __name__ == '__main__':
    main()
