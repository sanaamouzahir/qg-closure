"""
slice_deriv_from_deep.py  --  build n-lag derivative-target sweep dirs by SLICING
the surviving deep sources (Re25k, combo, kf4), with ZERO regeneration and
WITHOUT touching the existing 4-lag sweep_dT_* dirs.

It writes deriv-schema sweep dirs (deriv_dataset.py contract) to a FRESH --out-root:
    <out-root>/<member>/sweep_dT_<tag>/
        packed/inputs.npy          (N, 2S, Ny, Nx)  [omega_0..S-1, psi_0..S-1]
        packed/deriv_anal_f64.npy  (N, 3,  Ny, Nx)  [Ndot, Nddot, N3dot]
        split.npz   manifest.json

Conventions are IDENTICAL to fd_depth_check.py (which matched the training plateau,
so they are validated): per deep window, anchor = mark 0 (most recent); a depth-S
lag stack at coarse spacing dt = j*dt_fine is marks [0, j, 2j, ..., (S-1)j]; the
analytic [Ndot,Nddot,N3dot] target is the DEALIASED chain rule at the anchor (so it
matches the solver RHS dealias + add_deriv_targets). Targets are dt-independent,
computed once per window, reused across dts.

Run from $QG_DIR/training. Example (slice BOTH depths for a clean 4-vs-7 A/B):
    python slice_deriv_from_deep.py \
        --sources data/ensemble_N5/FRC-{Re25k,combo,kf4}/forced_turbulence_dT_5em3 \
        --out-root data/ensemble_N5_7lag \
        --n-snapshots 7 --target-dts 5e-3 1e-2 1.5e-2 \
        --max-anchors 8 --device cuda --dtype float64
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import torch

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical


# ---- DEALIASED operators, verbatim from measure_truncation_magnitudes.py ---- #
def J_phys(psi_phys, omega_phys, derivative, dealias=True):
    psih = to_spectral(psi_phys); qh = to_spectral(omega_phys)
    u = to_physical(-1 * derivative.dy * psih)
    v = to_physical(+1 * derivative.dx * psih)
    q = to_physical(qh)
    uq_h = to_spectral(u * q); vq_h = to_spectral(v * q)
    if dealias:
        uq_h = uq_h.clone(); vq_h = vq_h.clone()
        derivative.dealias(uq_h); derivative.dealias(vq_h)
    return to_physical(derivative.dx * uq_h + derivative.dy * vq_h)


def L_apply(field_phys, L_hat, power=1):
    return to_physical((L_hat ** power) * to_spectral(field_phys))


def build_L_hat(derivative, nu, mu, B):
    L_hat = nu * derivative.laplacian - mu
    if B != 0.0:
        L_hat = L_hat - B * derivative.dx * derivative.inv_laplacian
    return L_hat


def inv_lap(omega_phys, derivative):
    return to_physical(derivative.inv_laplacian * to_spectral(omega_phys))


def compute_derivatives(omega, derivative, L_hat, F_phys, max_m=3, dealias=True):
    omega_list = [omega]; psi_list = [inv_lap(omega, derivative)]; N_list = []
    for m in range(0, max_m + 1):
        Nm = None
        for j in range(0, m + 1):
            term = math.comb(m, j) * J_phys(psi_list[m - j], omega_list[j],
                                            derivative, dealias=dealias)
            Nm = term if Nm is None else Nm + term
        Nm = -1.0 * Nm
        if m == 0 and F_phys is not None:
            Nm = Nm + F_phys
        N_list.append(Nm)
        omega_next = L_apply(omega_list[m], L_hat, 1) + N_list[m]
        omega_list.append(omega_next)
        psi_list.append(inv_lap(omega_next, derivative))
    return dict(omega=omega_list, psi=psi_list, N=N_list)


def torch_dtype(s):
    return {'float64': torch.float64, 'float32': torch.float32}[s]


def build_F_phys(manifest, grid, device, dtype):
    fc = manifest.get('forcing', None)
    if not isinstance(fc, dict):
        return None
    A = float(fc.get('A', 0.0)); B = float(fc.get('B', 0.0))
    D = float(fc.get('D', 0.0)); E = float(fc.get('E', 0.0))
    if A == 0.0 and D == 0.0:
        return None
    x = torch.linspace(0, grid.Lx, grid.Nx, device=device, dtype=dtype)
    y = torch.linspace(0, grid.Ly, grid.Ny, device=device, dtype=dtype)
    return A * torch.cos(B * x[None, :]) + D * torch.cos(E * y[:, None])


def dt_tag(dt):
    """0.005->5em3, 0.01->1em2, 0.015->1p5em2 (matches existing sweep_dT_ naming)."""
    exp = int(math.floor(math.log10(dt)))
    mant = round(dt / 10 ** exp, 6)
    mant_str = f"{mant:g}".replace('.', 'p')
    return f"{mant_str}em{-exp}" if exp < 0 else f"{mant_str}e{exp}"


def write_split(out_dir, N, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_tr = int(0.70 * N); n_va = int(0.15 * N)
    np.savez(out_dir / 'split.npz',
             train_idx=perm[:n_tr], val_idx=perm[n_tr:n_tr + n_va],
             test_idx=perm[n_tr + n_va:])


def slice_member(src, out_root, S, target_dts, max_anchors, device, dtype, seed):
    src = Path(src)
    man = json.loads((src / 'manifest.json').read_text())
    inp_path = src / 'inputs.npy'
    if not inp_path.exists():
        inp_path = src / 'packed' / 'inputs.npy'
    deep = np.load(inp_path, mmap_mode='r')          # (Nwin, 2M, Ny, Nx)
    Nwin, twoM, Ny, Nx = deep.shape
    M = twoM // 2
    member = src.parent.name
    tdt = torch_dtype(dtype)

    dt_fine = float(man.get('Delta_T', man.get('dt', man.get('dt_save', 0.0))))
    Lx = float(man.get('Lx', man.get('box', 2 * math.pi)))
    Ly = float(man.get('Ly', Lx))
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision=dtype)
    derivative = Derivative(grid).to(device)
    nu = float(man.get('nu', man.get('viscosity', 0.0)))
    mu = float(man.get('mu', man.get('drag', 0.0)))
    beta = float(man.get('beta', man.get('B', 0.0)))
    L_hat = build_L_hat(derivative, nu, mu, beta)
    F = build_F_phys(man, grid, device, tdt)

    print(f"\n[{member}] deep Nwin={Nwin} M={M} marks  dt_fine={dt_fine:.3e}  "
          f"S={S}  grid={Ny}x{Nx}  forced={F is not None}")

    for dt in target_dts:
        j = int(round(dt / dt_fine))
        span = (S - 1) * j                       # deepest mark index used
        if span > M - 1:
            print(f"  dt={dt:.3g} j={j}: needs {span+1}>{M} marks -- SKIP")
            continue
        # anchors a with a + span <= M-1  (anchor most-recent, lags go back)
        a_max = M - 1 - span
        anchors = list(range(0, min(max_anchors, a_max + 1)))
        N = Nwin * len(anchors)

        tag = dt_tag(dt)
        out_dir = Path(out_root) / member / f"sweep_dT_{tag}"
        pdir = out_dir / 'packed'; pdir.mkdir(parents=True, exist_ok=True)
        inp_mm = np.lib.format.open_memmap(pdir / 'inputs.npy', mode='w+',
                                           dtype=np.float32, shape=(N, 2 * S, Ny, Nx))
        tgt_mm = np.lib.format.open_memmap(pdir / 'deriv_anal_f64.npy', mode='w+',
                                           dtype=np.float64, shape=(N, 3, Ny, Nx))
        row = 0
        for n in range(Nwin):
            om_all = torch.as_tensor(np.asarray(deep[n, 0:M]),  device=device, dtype=tdt)
            ps_all = torch.as_tensor(np.asarray(deep[n, M:2*M]), device=device, dtype=tdt)
            for a in anchors:
                lags = [a + i * j for i in range(S)]
                inp_mm[row, 0:S]     = om_all[lags].to(torch.float32).cpu().numpy()
                inp_mm[row, S:2*S]   = ps_all[lags].to(torch.float32).cpu().numpy()
                d = compute_derivatives(om_all[a], derivative, L_hat, F, max_m=3)
                tgt_mm[row, 0] = d['N'][1].cpu().numpy()
                tgt_mm[row, 1] = d['N'][2].cpu().numpy()
                tgt_mm[row, 2] = d['N'][3].cpu().numpy()
                row += 1
        inp_mm.flush(); tgt_mm.flush()
        del inp_mm, tgt_mm

        out_man = dict(man)
        out_man.update(dict(Delta_T=float(dt), n_snapshots_per_sample=int(S),
                            Ny=int(Ny), Nx=int(Nx), Lx=float(Lx), Ly=float(Ly),
                            beta=float(beta), nu=float(nu), mu=float(mu),
                            target_fields=['N_dot_0_anal', 'N_ddot_0_anal',
                                           'N_3dot_0_anal'],
                            sliced_from=str(src), n_anchors=len(anchors)))
        (out_dir / 'manifest.json').write_text(json.dumps(out_man, indent=2))
        write_split(out_dir, N, seed)
        print(f"  dt={dt:.3g} j={j} anchors={len(anchors)} -> {out_dir}  N={N}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--out-root', required=True,
                   help='FRESH dir; sweep dirs land at <out-root>/<member>/sweep_dT_*')
    p.add_argument('--n-snapshots', type=int, default=7)
    p.add_argument('--target-dts', type=float, nargs='+', default=[5e-3, 1e-2, 1.5e-2])
    p.add_argument('--max-anchors', type=int, default=8,
                   help='anchors harvested per deep window (more = bigger dataset)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='float64', choices=['float64', 'float32'])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    op = Path(args.out_root)
    if op.exists() and any(op.iterdir()):
        raise SystemExit(f"--out-root {op} exists and is non-empty; refusing to risk "
                         f"an overwrite. Point at a fresh path.")
    torch.set_grad_enabled(False)
    for src in args.sources:
        slice_member(src, args.out_root, args.n_snapshots, args.target_dts,
                     args.max_anchors, args.device, args.dtype, args.seed)
    print("\nDone. Train with --n-snapshots matching the sliced S, --sweep-roots "
          f"{args.out_root}/FRC-*/sweep_dT_*")


if __name__ == '__main__':
    main()
