"""
temporal_fd_floor_deep.py  --  deep-source-native temporal-FD floor sweep.

Same question as temporal_fd_floor_diagnostic.py but reads the deep sources
directly (inputs.npy + manifest.json, truth computed on the fly), so it runs on
the survivors with no old-schema packed set. With PERFECT spatial operators (exact
dealiased spectral Jacobian + inv-Laplacian), the ONLY error is FD-in-time, so the
rel-L2 it reports IS the temporal-FD floor a depth-n closure can reach.

For each member x Delta_T it sweeps n_time = 3..n_max and reports the per-order
floor (Ndot, Nddot, N3dot):
    omega^(k)  = n-point backward TimeFD of the omega history (k=0..n-1)
    psi^(k)    = inv_lap(omega^(k))
    N_fd^(m)   = -sum_j C(m,j) J(psi^(m-j), omega^(j))   (dealiased, model's way)
    truth N^(m)= analytic dealiased chain rule from omega^0
    floor(n,m) = median_anchors relL2(N_fd^(m), N^(m))

Read it as the diminishing-returns curve: how far each extra lag drives the
per-order floor down, per dt, per regime. n=4 row ~ the trained plateau; the
n=4 -> n=7 drop is the predicted payoff of rebuilding at 7 lags.

Run from $QG_DIR/training:
    python temporal_fd_floor_deep.py \
        --sources data/ensemble_N5/FRC-{Re25k,combo,kf4}/forced_turbulence_dT_5em3 \
        --target-dts 5e-3 1e-2 1.5e-2 --n-list 3 4 5 6 7 \
        --n-samples 48 --device cuda --dtype float64
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import torch

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical


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


def backward_fd_rows(n_time):
    xu = np.array([-j for j in range(n_time)], dtype=np.float64)
    A = np.array([[xu[j] ** m / math.factorial(m) for j in range(n_time)]
                  for m in range(n_time)], dtype=np.float64)
    return np.linalg.inv(A).T


def fd_N_orders(marks, derivative, dt, n, max_m, dealias):
    """n-point TimeFD omega^(0..n-1) from marks, then N_fd^(m) for m=1..max_m."""
    W = backward_fd_rows(n)
    omega_ord, psi_ord = [], []
    for k in range(n):
        wk = torch.as_tensor(W[k] / dt ** k, dtype=marks.dtype, device=marks.device)
        ok = torch.einsum('i,ihw->hw', wk, marks)
        omega_ord.append(ok); psi_ord.append(inv_lap(ok, derivative))
    out = {}
    for m in range(1, max_m + 1):
        Nm = None
        for j in range(0, m + 1):
            term = math.comb(m, j) * J_phys(psi_ord[m - j], omega_ord[j],
                                            derivative, dealias=dealias)
            Nm = term if Nm is None else Nm + term
        out[m] = -1.0 * Nm
    return out


def relL2(a, b):
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


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


_ORD = {1: 'Ndot', 2: 'Nddot', 3: 'N3dot'}


def run_source(src, target_dts, n_list, n_samples, rng, device, dtype, dealias):
    src = Path(src)
    man = json.loads((src / 'manifest.json').read_text())
    inp_path = src / 'inputs.npy'
    if not inp_path.exists():
        inp_path = src / 'packed' / 'inputs.npy'
    X = np.load(inp_path, mmap_mode='r')
    Nwin, twoM, Ny, Nx = X.shape
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

    idx = rng.choice(Nwin, size=min(n_samples, Nwin), replace=False)
    print(f"\n=== {member} ===  nu={nu:.2e} mu={mu:.2e} beta={beta:.3g}  "
          f"Re~{(1.0/nu) if nu>0 else float('inf'):.3g}  M={M} marks  "
          f"dt_fine={dt_fine:.3e}  dealias={dealias}")

    for dt in target_dts:
        j = int(round(dt / dt_fine))
        print(f"  dt={dt:.3g} (j={j})   per-order temporal-FD floor (median relL2):")
        print(f"    {'n_time':>6} | {'Ndot':>10} | {'Nddot':>10} | {'N3dot':>10}")
        print("    " + "-" * 46)
        for n in n_list:
            need = (n - 1) * j + 1
            if need > M:
                print(f"    {n:>6} |  need {need}>{M} marks -- skip")
                continue
            max_m = min(3, n - 1)
            acc = {m: [] for m in range(1, max_m + 1)}
            for s in idx:
                om = torch.as_tensor(np.asarray(X[s, 0:M]), device=device, dtype=tdt)
                marks = om[0:need:j][:n]
                truth = compute_derivatives(om[0], derivative, L_hat, F,
                                            max_m=max_m, dealias=dealias)
                fd = fd_N_orders(marks, derivative, dt, n, max_m, dealias)
                for m in range(1, max_m + 1):
                    acc[m].append(relL2(fd[m], truth['N'][m]))
            cells = []
            for m in (1, 2, 3):
                if m in acc and acc[m]:
                    cells.append(f"{float(np.median(acc[m]))*100:9.3f}%")
                else:
                    cells.append(f"{'--':>10}")
            print(f"    {n:>6} | " + " | ".join(cells))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--target-dts', type=float, nargs='+', default=[5e-3, 1e-2, 1.5e-2])
    p.add_argument('--n-list', type=int, nargs='+', default=[3, 4, 5, 6, 7])
    p.add_argument('--n-samples', type=int, default=48)
    p.add_argument('--dealias', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='float64', choices=['float64', 'float32'])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.set_grad_enabled(False)
    for src in args.sources:
        run_source(src, args.target_dts, sorted(args.n_list), args.n_samples,
                   rng, args.device, args.dtype, args.dealias)
    print("\nRead: each extra lag drives the per-order floor down; the n=4 row ~ the "
          "trained\nplateau, and the n=4 -> n=7 drop is the predicted payoff of "
          "rebuilding at 7 lags.")


if __name__ == '__main__':
    main()
