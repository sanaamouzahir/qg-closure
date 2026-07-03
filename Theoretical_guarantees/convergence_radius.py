"""
convergence_radius.py  --  estimate the modified-equation series radius dT* per case.

The AB2CN2 truncation closure is a power series in Delta_T:
        delta(Delta_T) = sum_{p>=3} c_p * Delta_T^p,   c_p = R_p / D_p
with R_p / D_p the modified-equation coefficients (R_DEFS / D_P below, copied from
measure_truncation_magnitudes.py). Its radius of convergence is
        dT* = 1 / limsup_p ||c_p||^{1/p}          (Cauchy-Hadamard / root test)
        dT* ~ ||c_p|| / ||c_{p+1}||                (ratio test, more stable few-term)
Past dT* NO finite-order closure (and no finite-lag stencil) converges -- this is the
same wall the 7-lag FD depth check sees, so the two should agree per member.

Each c_p is dominated by its highest N-derivative N^(p-1); the cascade makes
||N^(m)|| grow ~geometrically per order, and 1/(that growth) ~ dT*. We report both
the series-coefficient radius and the raw ||N^(m+1)||/||N^(m)|| growth.

CAVEAT: only p=3..6 are available (N^5 = 5 nested Jacobians), so these are
asymptotic estimates from few terms with growing discretization error at high p.
The reliable signal is the per-member RANKING and order of magnitude, cross-checked
against where the FD depth check stops improving.

Run from $QG_DIR/training:
    python convergence_radius.py \
        --sources data/ensemble_N5/FRC-{Re25k,combo,kf4}/forced_turbulence_dT_5em3 \
        --max-order 5 --n-samples 32 --device cuda --dtype float64
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import torch

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical


# ---- DEALIASED operators + chain rule (verbatim from measure_truncation_*) ---- #
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


def compute_derivatives(omega, derivative, L_hat, F_phys, max_m=5, dealias=True):
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
    return dict(omega=omega_list, N=N_list)


# ---- modified-equation series coefficients (verbatim from measure_truncation_*) ---- #
R_DEFS = {
    3: [( 1, 'omega', 3), ( 1, 'N', 2), ( 1, 'N1', 1), (-5, 'N2', 0)],
    4: [( 2, 'omega', 4), ( 2, 'N', 3), ( 2, 'N1', 2), (-4, 'N2', 1), ( 1, 'N3', 0)],
    5: [(13, 'omega', 5), (13, 'N', 4), (13, 'N1', 3), (-17,'N2', 2), ( 8, 'N3', 1), (-7, 'N4', 0)],
    6: [(43, 'omega', 6), (43, 'N', 5), (43, 'N1', 4), (-47,'N2', 3), (28, 'N3', 2), (-17,'N4', 1), ( 4, 'N5', 0)],
}
D_P = {3: 12, 4: 24, 5: 240, 6: 1440}
_KIND_TO_IDX = {'N': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'N4': 4, 'N5': 5}


def _rms(t):
    return float(torch.sqrt(torch.mean(t ** 2)))


def backward_fd_rows(n_time):
    xu = np.array([-j for j in range(n_time)], dtype=np.float64)
    A = np.array([[xu[j] ** m / math.factorial(m) for j in range(n_time)]
                  for m in range(n_time)], dtype=np.float64)
    return np.linalg.inv(A).T


def relL2(a, b):
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def interp_crossing(xs, ys, thr):
    """First x where y crosses thr (log-log linear interp). None if never / always."""
    xs = list(xs); ys = list(ys)
    if all(y < thr for y in ys):
        return ('>', xs[-1])           # wall beyond probeable range
    if ys[0] >= thr:
        return ('<', xs[0])            # already past wall at smallest probed dt
    for i in range(1, len(xs)):
        if ys[i] >= thr > ys[i - 1]:
            lx0, lx1 = math.log(xs[i - 1]), math.log(xs[i])
            ly0, ly1 = math.log(ys[i - 1]), math.log(ys[i])
            f = (math.log(thr) - ly0) / (ly1 - ly0)
            return ('=', math.exp(lx0 + f * (lx1 - lx0)))
    return ('>', xs[-1])


def assemble_R(p, derivs, L_hat):
    omega0 = derivs['omega'][0]; N_list = derivs['N']
    R = None
    for coeff, kind, Lpow in R_DEFS[p]:
        base = omega0 if kind == 'omega' else N_list[_KIND_TO_IDX[kind]]
        term = L_apply(base, L_hat, Lpow) if Lpow > 0 else base
        contrib = coeff * term
        R = contrib if R is None else R + contrib
    return R


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


def run_source(src, max_order, n_samples, rng, device, dtype, wall_k=7, wall_thr=0.5):
    src = Path(src)
    man = json.loads((src / 'manifest.json').read_text())
    inp_path = src / 'inputs.npy'
    if not inp_path.exists():
        inp_path = src / 'packed' / 'inputs.npy'
    deep = np.load(inp_path, mmap_mode='r')
    Nwin, twoM, Ny, Nx = deep.shape
    M = twoM // 2
    member = src.parent.name
    tdt = torch_dtype(dtype)

    Lx = float(man.get('Lx', man.get('box', 2 * math.pi)))
    Ly = float(man.get('Ly', Lx))
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision=dtype)
    derivative = Derivative(grid).to(device)
    nu = float(man.get('nu', man.get('viscosity', 0.0)))
    mu = float(man.get('mu', man.get('drag', 0.0)))
    beta = float(man.get('beta', man.get('B', 0.0)))
    L_hat = build_L_hat(derivative, nu, mu, beta)
    F = build_F_phys(man, grid, device, tdt)

    L_hat = build_L_hat(derivative, nu, mu, beta)
    F = build_F_phys(man, grid, device, tdt)
    dt_fine = float(man.get('Delta_T', man.get('dt', man.get('dt_save', 0.0))))

    ps = list(range(3, max_order + 2))            # available series orders (R_3..R_{max+1})
    idx = rng.choice(Nwin, size=min(n_samples, Nwin), replace=False)
    cmag = {p: [] for p in ps}                    # ||c_p|| = ||R_p||/D_p per anchor
    ngrow = {m: [] for m in range(0, max_order)}   # ||N^(m+1)||/||N^(m)||

    # inner stencil wall: dt where depth-`wall_k` backward FD of omega_ddot vs the
    # analytic truth crosses `wall_thr`. dt = j*dt_fine, j integer, (wall_k-1)j<=M-1.
    probe_js = [j for j in range(1, M) if (wall_k - 1) * j <= M - 1] if dt_fine > 0 else []
    w2_unit = backward_fd_rows(wall_k)[2]          # order-2 row, unit spacing
    wall_rel = {j: [] for j in probe_js}

    for n in idx:
        om0 = torch.as_tensor(np.asarray(deep[n, 0]), device=device, dtype=tdt)
        d = compute_derivatives(om0, derivative, L_hat, F, max_m=max_order)
        for p in ps:
            cmag[p].append(_rms(assemble_R(p, d, L_hat)) / D_P[p])
        nrm = [_rms(d['N'][m]) for m in range(0, max_order + 1)]
        for m in range(0, max_order):
            ngrow[m].append(nrm[m + 1] / nrm[m] if nrm[m] > 0 else float('nan'))
        if probe_js:
            om_all = torch.as_tensor(np.asarray(deep[n, 0:M]), device=device, dtype=tdt)
            odd_true = d['omega'][2]
            for j in probe_js:
                dt = j * dt_fine
                w2 = torch.as_tensor(w2_unit / dt ** 2, dtype=tdt, device=device)
                lags = [i * j for i in range(wall_k)]
                odd_fd = torch.einsum('i,ihw->hw', w2, om_all[lags])
                wall_rel[j].append(relL2(odd_fd, odd_true))

    c = {p: float(np.median(cmag[p])) for p in ps}
    roots = {p: c[p] ** (-1.0 / p) for p in ps}
    ratios = [c[p] / c[p + 1] for p in ps[:-1] if c[p + 1] > 0]
    dT_star = float(np.median(list(roots.values())))   # HEADLINE: Cauchy-Hadamard

    print(f"\n=== {member} ===  nu={nu:.2e} mu={mu:.2e} beta={beta:.3g}  "
          f"Re~{(1.0/nu) if nu>0 else float('inf'):.3g}  grid={Ny}x{Nx}")
    print("  series-coefficient magnitudes  ||c_p|| = ||R_p||/D_p  (median over anchors):")
    for p in ps:
        print(f"    p={p}: ||c_p|| = {c[p]:.4e}    root  c_p^(-1/p) = {roots[p]:.4e}")
    print(f"  dT* (root test / Cauchy-Hadamard, median over p) = {dT_star:.3e}")
    if ratios:
        print(f"  ratio-test BRACKET (sandwiches dT*, not an estimate): "
              f"[{min(ratios):.3e}, {max(ratios):.3e}]  (spread = non-geometric c_p)")
    print("  N-derivative growth  ||N^(m+1)||/||N^(m)||  (median):")
    for m in range(0, max_order):
        print(f"    m={m}->{m+1}: {float(np.median(ngrow[m])):.3g}x")

    # inner stencil wall
    inner = None
    if probe_js:
        dts = [j * dt_fine for j in probe_js]
        med = [float(np.median(wall_rel[j])) for j in probe_js]
        print(f"  inner stencil wall (k={wall_k}, omega_ddot relL2 vs truth, "
              f"thr={wall_thr}):")
        for dt, mrel in zip(dts, med):
            print(f"    dt={dt:.3g}: relL2={mrel:.3e}")
        kind, x = interp_crossing(dts, med, wall_thr)
        inner = x
        label = {'=': '~', '>': '> (beyond reach)', '<': '< (already past at)'}[kind]
        print(f"    -> inner wall {label} {x:.3e}")
    print(f"  >>> OUTER dT* (series radius) = {dT_star:.3e}   |   "
          f"INNER stencil wall (k={wall_k}) "
          f"{'~ '+format(inner,'.3e') if inner else 'n/a'}")
    return member, dT_star, inner


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--max-order', type=int, default=5, choices=[3, 4, 5],
                   help='highest N-derivative (5 -> through R_6, three radius ratios)')
    p.add_argument('--n-samples', type=int, default=32)
    p.add_argument('--wall-k', type=int, default=7,
                   help='stencil depth for the inner omega_ddot wall (default 7)')
    p.add_argument('--wall-thr', type=float, default=0.5,
                   help='relL2 threshold defining the inner stencil wall (default 0.5)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='float64', choices=['float64', 'float32'])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.set_grad_enabled(False)
    summary = []
    for src in args.sources:
        summary.append(run_source(src, args.max_order, args.n_samples,
                                   rng, args.device, args.dtype,
                                   args.wall_k, args.wall_thr))
    print("\n========= per-member walls (outer series radius / inner stencil) =========")
    print(f"  {'member':<16} {'OUTER dT* (root)':>18} {'INNER wall k='+str(args.wall_k):>18}")
    for member, dT_star, inner in sorted(summary, key=lambda kv: kv[1]):
        istr = f"{inner:.3e}" if inner else "n/a"
        print(f"  {member:<16} {dT_star:>18.3e} {istr:>18}")
    print("Two NESTED walls. OUTER dT* (root test): past it adding higher analytic Rp\n"
          "diverges -- no finite-order closure helps. INNER stencil wall: where a fixed-\n"
          "lag backward FD stops recovering omega_ddot, hit FIRST (inside dT*). At dt below\n"
          "the inner wall the binding limit is finite lags -> more lags help (the FD-check\n"
          "result); between inner and outer, more lags help but the series still converges;\n"
          "past dT* nothing finite does. NOTE both are finite-p estimates, trust ~20-30%.")


if __name__ == '__main__':
    main()