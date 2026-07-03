"""
fd_depth_check.py  --  Is going from 4 lags to 7 lags worth regenerating the data?

TRAINING-FREE. Predicts the omega_ddot / N_ddot truncation FLOOR that a depth-k
time-FD stencil can reach at coarse Delta_T, BEFORE paying to rebuild any data.
Runs on the SURVIVING deep sources (Re25k, combo, +1) which still hold >= 7 deep
omega marks, so NO regeneration is needed to get the answer.

DEALIASING (this is the subtle part):
  The production solver dealiases the ASSEMBLED RHS every step
  (qg/.../operator __call__: derivative.dealias(sum(patches))), so the effective
  nonlinearity it propagates is P[N], the stored omega snapshots are band-limited,
  and the analytic targets are built with the 2/3 rule applied INSIDE every
  Jacobian, at every N-derivative stage. We therefore reuse the canonical
  dealiased chain rule from measure_truncation_magnitudes.py (J_phys(dealias=True),
  compute_derivatives) for the truth, and apply dealias=True inside the FD-built
  N_ddot Jacobians too. Pass --no-dealias to see the raw aliased error for contrast.

Why this can go either way (the whole reason we check first):
  A k-point backward stencil gives omega_ddot to O(dt^{k-2}) only if the temporal
  signal is resolved at the coarse spacing. If the coarse-dt plateau is TRUNCATION-
  limited, k=7 drops sharply at dt=1e-2/1.5e-2 -> regeneration justified. If it is
  TEMPORAL-UNDER-RESOLUTION-limited (omega_ddot's time variation simply isn't
  sampled finely enough for any stencil), k=7 ~ k=4 -> more lags won't move the
  plateau; report the validated dt<=1e-2 range instead.

What it reports, per member x Delta_T x depth k in {4,7}:
  relL2(omega_ddot_FD, omega_ddot_true)  and  relL2(N_ddot_FD, N_ddot_true)
  truth  = compute_derivatives(omega^0)  (dealiased chain rule, exact from omega^0)
  FD     = depth-k backward stencils -> omega_dot/omega_ddot -> N_ddot the model's way

Usage (run from $QG_DIR/training so qg + measure_truncation_magnitudes import):
    python fd_depth_check.py \
        --sources data/ensemble_N5/FRC-Re25k/<deep_dir> \
                  data/ensemble_N5/FRC-combo/<deep_dir> \
        --target-dts 5e-3 1e-2 1.5e-2 --depths 4 7 \
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


# --------------------------------------------------------------------------- #
# Canonical DEALIASED operators + chain rule -- copied VERBATIM from
# measure_truncation_magnitudes.py so the truth matches the solver RHS dealias
# (operator __call__: derivative.dealias(sum(patches))) and the target builder.
# Inlined (not imported) to keep this script standalone.
# --------------------------------------------------------------------------- #
def J_phys(psi_phys, omega_phys, derivative, dealias=True):
    """+J(psi, omega) via spectral derivatives, with 2/3 dealiasing on the
    nonlinear products (matches the production solver + the closure code)."""
    psih = to_spectral(psi_phys)
    qh   = to_spectral(omega_phys)
    uh = -1 * derivative.dy * psih
    vh = +1 * derivative.dx * psih
    u = to_physical(uh)
    v = to_physical(vh)
    q = to_physical(qh)
    uq_h = to_spectral(u * q)
    vq_h = to_spectral(v * q)
    if dealias:
        uq_h = uq_h.clone(); vq_h = vq_h.clone()
        derivative.dealias(uq_h); derivative.dealias(vq_h)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h
    return to_physical(j_hat)


def L_apply(field_phys, L_hat, power=1):
    qh = to_spectral(field_phys)
    return to_physical((L_hat ** power) * qh)


def build_L_hat(derivative, nu, mu, B):
    L_hat = nu * derivative.laplacian - mu
    if B != 0.0:
        L_hat = L_hat - B * derivative.dx * derivative.inv_laplacian
    return L_hat


def inv_lap(omega_phys, derivative):
    return to_physical(derivative.inv_laplacian * to_spectral(omega_phys))


def _binom(m, j):
    return math.comb(m, j)


def compute_derivatives(omega, derivative, L_hat, F_phys, max_m=5, dealias=True):
    """omega^(k), psi^(k) for k=0..max_m+1, and N^(m) for m=0..max_m via the
    interleaved chain-rule bootstrap (dealiased J at every stage):
        N^(0)   = -J(psi, w) + F
        w^(k)   = L w^(k-1) + N^(k-1)
        psi^(k) = inv_lap(w^(k))
        N^(m)   = -sum_j C(m,j) J(psi^(m-j), w^(j))    (F drops out for m>=1)
    """
    omega_list = [omega]
    psi_list   = [inv_lap(omega, derivative)]
    N_list     = []
    for m in range(0, max_m + 1):
        Nm = None
        for j in range(0, m + 1):
            term = _binom(m, j) * J_phys(psi_list[m - j], omega_list[j],
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
    """Steady forcing, EXACT replica of forcing.unscaled_cosine at t=0:
       w = A cos(B X) + D cos(E Y),  X,Y = linspace(0, L, N) (endpoint included)."""
    fc = manifest.get('forcing', None)
    if not isinstance(fc, dict):
        return None
    A = float(fc.get('A', 0.0)); B = float(fc.get('B', 0.0))
    D = float(fc.get('D', 0.0)); E = float(fc.get('E', 0.0))
    if A == 0.0 and D == 0.0:
        return None
    x = torch.linspace(0, grid.Lx, grid.Nx, device=device, dtype=dtype)
    y = torch.linspace(0, grid.Ly, grid.Ny, device=device, dtype=dtype)
    X, Y = x[None, :], y[:, None]
    return A * torch.cos(B * X) + D * torch.cos(E * Y)


def backward_fd_rows(n_time):
    """Unit-spacing backward-difference weights; row k = order-k stencil on x_j=-j."""
    xu = np.array([-j for j in range(n_time)], dtype=np.float64)
    A = np.array([[xu[j] ** m / math.factorial(m) for j in range(n_time)]
                  for m in range(n_time)], dtype=np.float64)
    return np.linalg.inv(A).T   # (n_time, n_time)


def fd_Nddot(omega_marks, derivative, dt, k, psi0, dealias):
    """omega_marks: (k, Ny, Nx) at coarse spacing dt, most-recent first.
       Build omega_dot/omega_ddot by depth-k backward FD, then N_ddot the model's
       way, with dealias applied inside every Jacobian (matches the pipeline)."""
    W = backward_fd_rows(k)
    w1 = torch.as_tensor(W[1] / dt,      dtype=omega_marks.dtype, device=omega_marks.device)
    w2 = torch.as_tensor(W[2] / dt ** 2, dtype=omega_marks.dtype, device=omega_marks.device)
    omega_dot_fd  = torch.einsum('i,ihw->hw', w1, omega_marks)
    omega_ddot_fd = torch.einsum('i,ihw->hw', w2, omega_marks)
    psi_dot_fd  = inv_lap(omega_dot_fd,  derivative)
    psi_ddot_fd = inv_lap(omega_ddot_fd, derivative)
    om0 = omega_marks[0]
    N_ddot_fd = (-J_phys(psi_ddot_fd, om0,           derivative, dealias=dealias)
                 - 2 * J_phys(psi_dot_fd, omega_dot_fd, derivative, dealias=dealias)
                 - J_phys(psi0, omega_ddot_fd,        derivative, dealias=dealias))
    return omega_ddot_fd, N_ddot_fd


def relL2(a, b):
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def run_source(src, target_dts, depths, n_samples, rng, device, dtype, dt_fine_cli,
               dealias=True):
    src = Path(src)
    man = json.loads((src / 'manifest.json').read_text())
    inp_path = src / 'inputs.npy'
    if not inp_path.exists():
        inp_path = src / 'packed' / 'inputs.npy'
    X = np.load(inp_path, mmap_mode='r')           # (N, 2M, Ny, Nx)
    Nsamp, twoM, Ny, Nx = X.shape
    M = twoM // 2

    dt_fine = dt_fine_cli if dt_fine_cli is not None else float(
        man.get('Delta_T', man.get('dt', man.get('dt_save', 0.0))))
    if dt_fine <= 0:
        raise SystemExit(f"[{src.name}] could not read mark spacing; pass --dt-fine")

    Lx = float(man.get('Lx', man.get('box', 2 * math.pi)))
    Ly = float(man.get('Ly', Lx))
    tdt = torch_dtype(dtype)
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision=dtype)
    derivative = Derivative(grid).to(device)   # k-arrays are built on CPU; move them
    nu = float(man.get('nu', man.get('viscosity', 0.0)))
    mu = float(man.get('mu', man.get('drag', 0.0)))
    beta = float(man.get('beta', man.get('B', 0.0)))
    L_hat = build_L_hat(derivative, nu, mu, beta)
    F = build_F_phys(man, grid, device, tdt)

    js = {dt: int(round(dt / dt_fine)) for dt in target_dts}

    print(f"\n=== {src.name} ===  grid={Ny}x{Nx} Lx={Lx:.4f} M={M} marks  "
          f"dt_fine={dt_fine:.3e}  nu={nu:.2e} mu={mu:.2e} beta={beta:.3g}  "
          f"forced={F is not None}  dealias={dealias}")
    print(f"    target dt -> j (marks back per lag): "
          + ", ".join(f"{dt:.3g}->{js[dt]}" for dt in target_dts))
    hdr = f"    {'dt':>8} | " + " | ".join(
        f"k={k}: omega_ddot   N_ddot" for k in depths)
    print(hdr); print("    " + "-" * (len(hdr) - 4))

    idx = rng.choice(Nsamp, size=min(n_samples, Nsamp), replace=False)
    for dt in target_dts:
        j = js[dt]
        cells = []
        for k in depths:
            need = (k - 1) * j + 1
            if need > M:
                cells.append((k, None, None, f"need {need}>{M} marks"))
                continue
            r_w, r_N = [], []
            for n in idx:
                om = torch.as_tensor(np.asarray(X[n, 0:M]), device=device, dtype=tdt)
                marks = om[0:need:j][:k]                       # (k,Ny,Nx)
                d = compute_derivatives(om[0], derivative, L_hat, F,
                                        max_m=2, dealias=dealias)
                od_fd, Nd_fd = fd_Nddot(marks, derivative, dt, k,
                                        d['psi'][0], dealias)
                r_w.append(relL2(od_fd, d['omega'][2]))
                r_N.append(relL2(Nd_fd, d['N'][2]))
            cells.append((k, float(np.median(r_w)), float(np.median(r_N)), None))
        parts = []
        for k, rw, rN, note in cells:
            parts.append(f"k={k}: {note:>22}" if note
                         else f"k={k}: {rw:>9.3e}   {rN:>9.3e}")
        print(f"    {dt:>8.3g} | " + " | ".join(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sources', nargs='+', required=True,
                   help='deep-source dirs (each with manifest.json + inputs.npy)')
    p.add_argument('--target-dts', type=float, nargs='+', default=[5e-3, 1e-2, 1.5e-2])
    p.add_argument('--depths', type=int, nargs='+', default=[4, 7])
    p.add_argument('--n-samples', type=int, default=48)
    p.add_argument('--dt-fine', type=float, default=None,
                   help='override mark spacing if not in manifest')
    p.add_argument('--dealias', action=argparse.BooleanOptionalAction, default=True,
                   help='2/3 dealias inside every Jacobian (default ON, matches the '
                        'solver RHS dealias + target builder). --no-dealias for the '
                        'raw aliased contrast.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='float64', choices=['float64', 'float32'])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.set_grad_enabled(False)
    for src in args.sources:
        run_source(src, args.target_dts, args.depths, args.n_samples,
                   rng, args.device, args.dtype, args.dt_fine, args.dealias)

    print("\nVERDICT GUIDE: compare N_ddot at k=7 vs k=4 for dt=1e-2 and 1.5e-2.\n"
          "  >2x lower   -> truncation-limited; regeneration justified.\n"
          "  ~equal/worse -> temporal under-resolution; more lags will NOT move the\n"
          "                  plateau. Do not regenerate; report validated dt<=1e-2.")


if __name__ == '__main__':
    main()