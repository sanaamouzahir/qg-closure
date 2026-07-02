"""
measure_truncation_magnitudes.py
================================

Empirically measure the size of every term in the AB2-CN2 high-order
truncation operators R_3, R_4, R_5, R_6 derived in
qg_ab2cn2_highorder_truncation.pdf:

    tau = -(h^3/12) R_3 - (h^4/24) R_4 - (h^5/240) R_5 - (h^6/1440) R_6 + O(h^7)

    R_3 = L^3 w + L^2 N + L Ndot - 5 Nddot
    R_4 = 2L^4 w + 2L^3 N + 2L^2 Ndot - 4 L Nddot + Ndddot
    R_5 = 13L^5 w + 13L^4 N + 13L^3 Ndot - 17L^2 Nddot + 8 L Ndddot - 7 N4
    R_6 = 43L^6 w + 43L^5 N + 43L^4 Ndot - 47L^3 Nddot + 28L^2 Ndddot
          - 17 L N4 + 4 N5

The point: the analytical pieces (L^p w, L^{p-1} N) shrink predictably with
order, but the LEARNED pieces depend on the unknown norms
||Ndot||, ||Nddot||, ||Ndddot||, ||N4||, ||N5||.  In a turbulent cascade these
may be large/growing, so the per-order contribution to the LTE must be
MEASURED, not assumed.

This script:
  1. Builds omega^(k) (k=0..6) and N^(m) (m=0..5) at representative snapshots
     via the recursive bootstrap (eqs. 2 and 4-8 of the document).
  2. Validates each N^(m) against central finite differences of N along an
     RK4 trajectory through the snapshot (degrades at high order -- expected,
     since FD of a p-th derivative amplifies roundoff by 1/dt^p).
  3. Prints the RMS of every monomial in R_3..R_6, the assembled ||R_p||, and
     the products (h^p / D_p) ||R_p|| at Delta_T in {1e-3, 1e-2, 1e-1}, i.e.
     the measured contribution of each order to the LTE.

Works for BOTH decaying turbulence (F=None) and forced turbulence (F built
from the YAML's unscaled_cosine block).

Usage
-----
  # decaying turbulence (no forcing)
  python measure_truncation_magnitudes.py \
      --omega-source /gdata/.../decaying_turb_restart_t60_dt1em5/DNS_FR_omega.npy \
      --times        /gdata/.../decaying_turb_restart_t60_dt1em5/DNS_FR_times.npy \
      --source-yaml  /gdata/.../decaying_turbulence.yaml \
      --scenario decaying_turbulence \
      --n-snaps 5 --device cuda

  # forced turbulence (forcing from yaml)
  python measure_truncation_magnitudes.py \
      --omega-source /gdata/.../forced_turb_dt_sweep_v2/dt_1em5/DNS_FR_omega.npy \
      --times        /gdata/.../forced_turb_dt_sweep_v2/dt_1em5/DNS_FR_times.npy \
      --source-yaml  /gdata/.../forced_turbulence.yaml \
      --scenario forced_turbulence \
      --n-snaps 5 --device cuda
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from qg.solver.opt.basis import to_spectral, to_physical


# --------------------------------------------------------------------------- #
# Operators (match build_training_data_fixD_v2.py / step3 conventions)        #
# --------------------------------------------------------------------------- #

def J_phys(psi_phys, omega_phys, derivative, dealias=True):
    """+J(psi, omega) via spectral derivatives.  Optional 2/3 dealiasing on
    the nonlinear products (matches the production solver + the closure code).
    """
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


def N_of(omega_phys, derivative, F_phys, dealias=True):
    psi = inv_lap(omega_phys, derivative)
    N = -1.0 * J_phys(psi, omega_phys, derivative, dealias=dealias)
    if F_phys is not None:
        N = N + F_phys
    return N


def rk4_step(omega, dt, derivative, L_hat, F_phys, dealias=True):
    def rhs(om):
        return L_apply(om, L_hat, 1) + N_of(om, derivative, F_phys, dealias=dealias)
    k1 = rhs(omega)
    k2 = rhs(omega + 0.5 * dt * k1)
    k3 = rhs(omega + 0.5 * dt * k2)
    k4 = rhs(omega + dt * k3)
    return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# --------------------------------------------------------------------------- #
# Recursive chain-rule bootstrap: omega^(k), psi^(k), N^(m)                   #
# --------------------------------------------------------------------------- #

def _binom(m, j):
    return math.comb(m, j)


def compute_derivatives(omega, derivative, L_hat, F_phys, max_m=5, dealias=True):
    """Build omega^(k) for k=0..max_m+1, psi^(k) for k=0..max_m+1, and
    N^(m) for m=0..max_m via the interleaved bootstrap:

        N^(0) = -J(psi, w) + F
        w^(k) = L w^(k-1) + N^(k-1)
        psi^(k) = inv_lap(w^(k))
        N^(m) = -sum_j C(m,j) J(psi^(m-j), w^(j))     (F drops out for m>=1)

    Returns dict with lists omega_list, psi_list, N_list.
    """
    omega_list = [omega]
    psi_list   = [inv_lap(omega, derivative)]
    N_list     = []
    for m in range(0, max_m + 1):
        # N^(m) from omega_list[0..m], psi_list[0..m]
        Nm = None
        for j in range(0, m + 1):
            term = _binom(m, j) * J_phys(psi_list[m - j], omega_list[j],
                                         derivative, dealias=dealias)
            Nm = term if Nm is None else Nm + term
        Nm = -1.0 * Nm
        if m == 0 and F_phys is not None:
            Nm = Nm + F_phys           # F enters only N^(0); F-derivatives = 0
        N_list.append(Nm)
        # omega^(m+1) = L omega^(m) + N^(m);  psi^(m+1) = inv_lap(.)
        omega_next = L_apply(omega_list[m], L_hat, 1) + N_list[m]
        omega_list.append(omega_next)
        psi_list.append(inv_lap(omega_next, derivative))
    return dict(omega=omega_list, psi=psi_list, N=N_list)


# --------------------------------------------------------------------------- #
# Finite-difference validation of N^(m) along an RK4 trajectory               #
# --------------------------------------------------------------------------- #

# central-difference stencils for the m-th derivative: {offset: coeff}, /dt^m
_FD_STENCILS = {
    1: ({-1: -0.5, 1: 0.5}, 1),
    2: ({-1: 1.0, 0: -2.0, 1: 1.0}, 2),
    3: ({-2: -0.5, -1: 1.0, 1: -1.0, 2: 0.5}, 3),
    4: ({-2: 1.0, -1: -4.0, 0: 6.0, 1: -4.0, 2: 1.0}, 4),
    5: ({-3: -0.5, -2: 2.0, -1: -2.5, 1: 2.5, 2: -2.0, 3: 0.5}, 5),
}


def fd_validate(omega0, derivative, L_hat, F_phys, max_m, fd_dt, dealias=True):
    """Build N at offsets k*fd_dt around omega0 via RK4, then central-difference
    to estimate N^(m).  Returns dict m -> FD estimate (physical tensor)."""
    max_off = max(max(abs(o) for o in st[0]) for st in _FD_STENCILS.values())
    # March forward and backward from omega0 to cover [-max_off, +max_off]*fd_dt
    N_at = {0: N_of(omega0, derivative, F_phys, dealias=dealias)}
    # forward
    om = omega0.clone()
    for k in range(1, max_off + 1):
        om = rk4_step(om, +fd_dt, derivative, L_hat, F_phys, dealias=dealias)
        N_at[k] = N_of(om, derivative, F_phys, dealias=dealias)
    # backward
    om = omega0.clone()
    for k in range(1, max_off + 1):
        om = rk4_step(om, -fd_dt, derivative, L_hat, F_phys, dealias=dealias)
        N_at[-k] = N_of(om, derivative, F_phys, dealias=dealias)
    out = {}
    for m in range(1, max_m + 1):
        stencil, p = _FD_STENCILS[m]
        acc = None
        for off, c in stencil.items():
            term = c * N_at[off]
            acc = term if acc is None else acc + term
        out[m] = acc / (fd_dt ** p)
    return out


# --------------------------------------------------------------------------- #
# R_p assembly                                                                #
# --------------------------------------------------------------------------- #

def _rms(t):
    return float(torch.sqrt(torch.mean(t ** 2)))


# Each R_p as a list of (coeff, 'kind', Lpow) where kind in
# {'omega','N','N1','N2','N3','N4','N5'} and Lpow is the power of L applied.
R_DEFS = {
    3: [( 1, 'omega', 3), ( 1, 'N', 2), ( 1, 'N1', 1), (-5, 'N2', 0)],
    4: [( 2, 'omega', 4), ( 2, 'N', 3), ( 2, 'N1', 2), (-4, 'N2', 1), ( 1, 'N3', 0)],
    5: [(13, 'omega', 5), (13, 'N', 4), (13, 'N1', 3), (-17,'N2', 2), ( 8, 'N3', 1), (-7, 'N4', 0)],
    6: [(43, 'omega', 6), (43, 'N', 5), (43, 'N1', 4), (-47,'N2', 3), (28, 'N3', 2), (-17,'N4', 1), ( 4, 'N5', 0)],
}
D_P = {3: 12, 4: 24, 5: 240, 6: 1440}

_KIND_TO_IDX = {'N': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'N4': 4, 'N5': 5}


def assemble_R(p, derivs, L_hat, derivative):
    """Return (R_p tensor, dict of per-monomial RMS)."""
    omega_list = derivs['omega']
    N_list     = derivs['N']
    R = None
    monos = {}
    for coeff, kind, Lpow in R_DEFS[p]:
        if kind == 'omega':
            base = omega_list[0]
        else:
            base = N_list[_KIND_TO_IDX[kind]]
        term_field = L_apply(base, L_hat, Lpow) if Lpow > 0 else base
        label = (f"L^{Lpow} " if Lpow > 0 else "") + \
                ({'omega': 'w', 'N': 'N', 'N1': 'Ndot', 'N2': 'Nddot',
                  'N3': 'Ndddot', 'N4': 'N4', 'N5': 'N5'}[kind])
        monos[label] = (coeff, _rms(term_field))
        contrib = coeff * term_field
        R = contrib if R is None else R + contrib
    return R, monos


# --------------------------------------------------------------------------- #
# Snapshot loading                                                            #
# --------------------------------------------------------------------------- #

def load_snapshots(omega_source, times_path, batch_index, n_snaps, t_start):
    suffix = omega_source.suffix.lower()
    if suffix == '.npy':
        arr = np.load(omega_source, mmap_mode='r')
        if arr.ndim == 4:
            arr = arr[batch_index]            # (T, Ny, Nx)
        if arr.ndim == 3:
            if times_path is not None:
                times = np.load(times_path)
                if times.ndim == 2:
                    times = times[batch_index]
            else:
                times = np.arange(arr.shape[0], dtype=float)
            # choose snapshots evenly spaced over [t_start, t_end]
            valid = np.where(times >= t_start)[0]
            if len(valid) == 0:
                valid = np.arange(arr.shape[0])
            idxs = np.linspace(valid[0], arr.shape[0] - 1, n_snaps).round().astype(int)
            idxs = sorted(set(int(i) for i in idxs))
            snaps = [(float(times[i]), np.asarray(arr[i]).astype(np.float64)) for i in idxs]
            return snaps
        elif arr.ndim == 2:                   # a single restart IC
            return [(0.0, np.asarray(arr).astype(np.float64))]
        else:
            sys.exit(f"ERROR: omega ndim={arr.ndim}")
    elif suffix == '.npz':                     # dataset sample
        rec = np.load(omega_source)
        key = 'omega_0' if 'omega_0' in rec.files else rec.files[0]
        a = rec[key].astype(np.float64)
        if a.ndim == 3: a = a[0]
        t = float(rec['seed_t']) if 'seed_t' in rec.files else 0.0
        return [(t, a)]
    else:
        sys.exit(f"ERROR: unrecognized omega-source suffix '{suffix}'")


def build_F_phys(yaml_cfg, scenario, Lx, Ly, Nx, Ny, device, dtype):
    if scenario != 'forced_turbulence':
        return None
    fcfg = yaml_cfg.get('qg', {}).get('forcing') or yaml_cfg.get('forcing')
    if not fcfg or fcfg.get('function') != 'unscaled_cosine':
        sys.exit(f"ERROR: scenario=forced_turbulence but no unscaled_cosine "
                 f"forcing in yaml (got {fcfg})")
    A = float(fcfg.get('A', 0.0)); B = float(fcfg.get('B', 0.0))
    C = float(fcfg.get('C', 0.0)); D = float(fcfg.get('D', 0.0))
    E = float(fcfg.get('E', 0.0)); Fc = float(fcfg.get('F', 0.0))
    if abs(C) > 1e-30 or abs(Fc) > 1e-30:
        sys.exit(f"ERROR: time-dependent forcing (C={C}, F={Fc}) breaks "
                 f"Fdot=Fddot=0 assumption.")
    x = torch.linspace(0.0, Lx, Nx, device=device, dtype=dtype)
    y = torch.linspace(0.0, Ly, Ny, device=device, dtype=dtype)
    F2d = A * torch.cos(B * x)[None, :] + D * torch.cos(E * y)[:, None]
    print(f"[mag] forcing F = {A}*cos({B}x)+{D}*cos({E}y), |F|_rms={_rms(F2d):.4e}")
    return F2d[None, ...]


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--omega-source', type=Path, required=True)
    p.add_argument('--times', type=Path, default=None)
    p.add_argument('--source-yaml', type=Path, required=True)
    p.add_argument('--scenario', type=str, default='decaying_turbulence',
                   choices=['decaying_turbulence', 'forced_turbulence',
                            'flow_past_cylinder'])
    p.add_argument('--batch-index', type=int, default=0)
    p.add_argument('--n-snaps', type=int, default=5)
    p.add_argument('--t-start', type=float, default=0.0,
                   help='earliest physical time to sample snapshots from')
    p.add_argument('--max-order', type=int, default=5, choices=[2, 3, 4, 5],
                   help='highest N-derivative to compute (5 -> through R_6)')
    p.add_argument('--fd-dt', type=float, default=2.0e-3,
                   help='dt for the RK4 finite-difference validation stencil '
                        '(near-optimal ~2e-3 for mid orders; high orders are '
                        'roundoff-limited regardless)')
    p.add_argument('--no-dealias', action='store_true',
                   help='disable 2/3 dealiasing (NOT recommended; closure uses it)')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    dtype = torch.float64
    dealias = not args.no_dealias

    with open(args.source_yaml) as f:
        yaml_cfg = yaml.safe_load(f)

    def yget(path, default=None):
        cur = yaml_cfg
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    Nx = int(yget(['qg', 'grid', 'Nx'], yget(['grid', 'Nx'], 256)))
    Ny = int(yget(['qg', 'grid', 'Ny'], yget(['grid', 'Ny'], 256)))
    Lx = float(yget(['qg', 'grid', 'Lx'], yget(['grid', 'Lx'], 2 * math.pi)))
    Ly = float(yget(['qg', 'grid', 'Ly'], yget(['grid', 'Ly'], 2 * math.pi)))
    nu = float(yget(['qg', 'pde', 'nu'], yget(['pde', 'nu'], 0.0)))
    mu = float(yget(['qg', 'pde', 'mu'], yget(['pde', 'mu'], 0.0)))
    B  = float(yget(['qg', 'pde', 'B'],  yget(['pde', 'B'], 0.0)))
    print(f"[mag] scenario={args.scenario}  Nx={Nx} Ny={Ny} Lx={Lx:.4f} Ly={Ly:.4f}")
    print(f"[mag] nu={nu} mu={mu} B(beta)={B}  dealias={dealias}  device={device}")

    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision='float64')
    derivative = Derivative(grid)
    for attr in ('dx', 'dy', 'laplacian', 'inv_laplacian'):
        setattr(derivative, attr, getattr(derivative, attr).to(device))
    L_hat = build_L_hat(derivative, nu=nu, mu=mu, B=B).to(device)

    F_phys = build_F_phys(yaml_cfg, args.scenario, Lx, Ly, Nx, Ny, device, dtype)

    snaps = load_snapshots(args.omega_source, args.times, args.batch_index,
                           args.n_snaps, args.t_start)
    print(f"[mag] {len(snaps)} snapshots at t = "
          f"{[f'{t:.3f}' for t, _ in snaps]}")

    max_m = args.max_order
    p_max = max_m + 1   # R_{max_m+1} is the highest assembled

    # Accumulators for averaging over snapshots
    mono_acc = {p: {} for p in range(3, p_max + 1)}
    Rnorm_acc = {p: [] for p in range(3, p_max + 1)}
    fd_rel_acc = {m: [] for m in range(1, max_m + 1)}

    for (t_snap, om_np) in snaps:
        omega = torch.tensor(om_np, dtype=dtype, device=device)[None]
        derivs = compute_derivatives(omega, derivative, L_hat, F_phys,
                                     max_m=max_m, dealias=dealias)

        # ---- FD validation of N^(m) ----
        fd = fd_validate(omega, derivative, L_hat, F_phys, max_m,
                         args.fd_dt, dealias=dealias)
        print(f"\n[mag] === snapshot t={t_snap:.3f} ===")
        print(f"  |omega|_rms = {_rms(omega):.4e}   |N|_rms = {_rms(derivs['N'][0]):.4e}")
        print(f"  chain-rule vs finite-difference (fd_dt={args.fd_dt:g}):")
        names = {1: 'Ndot', 2: 'Nddot', 3: 'Ndddot', 4: 'N4', 5: 'N5'}
        for m in range(1, max_m + 1):
            an = derivs['N'][m]
            fm = fd[m]
            rel = _rms(an - fm) / max(_rms(fm), 1e-30)
            fd_rel_acc[m].append(rel)
            flag = '' if rel < 0.05 else ('  <-- FD roundoff-limited' if m >= 4 else '  <-- CHECK')
            print(f"    {names[m]:>7s}: |an|={_rms(an):.4e}  |fd|={_rms(fm):.4e}  "
                  f"rel.diff={rel:.3e}{flag}")

        # ---- R_p monomials + norms ----
        for pp in range(3, p_max + 1):
            R, monos = assemble_R(pp, derivs, L_hat, derivative)
            Rnorm_acc[pp].append(_rms(R))
            for label, (coeff, rms) in monos.items():
                mono_acc[pp].setdefault(label, []).append(rms)

    # ---- Averaged report ----
    print("\n" + "=" * 72)
    print("AVERAGED OVER SNAPSHOTS")
    print("=" * 72)

    print("\nFinite-difference validation (mean rel.diff, lower orders should be small):")
    for m in range(1, max_m + 1):
        print(f"  N^({m}): {np.mean(fd_rel_acc[m]):.3e}")

    print("\nPer-monomial RMS in each R_p (coeff x ||term||):")
    for pp in range(3, p_max + 1):
        print(f"  --- R_{pp}  (||R_{pp}|| = {np.mean(Rnorm_acc[pp]):.4e}) ---")
        for label, vals in mono_acc[pp].items():
            print(f"      {label:>12s} : {np.mean(vals):.4e}")

    print("\nPer-order contribution to the LTE:  (h^p / D_p) * ||R_p||")
    print(f"  D_p = {D_P}")
    header = "  Delta_T  " + "".join(f"|  h^{pp}/D_{pp} ||R_{pp}|| " for pp in range(3, p_max + 1))
    print(header)
    for h in (1e-3, 1e-2, 1e-1):
        row = f"  {h:7.0e}  "
        prev = None
        for pp in range(3, p_max + 1):
            contrib = (h ** pp) / D_P[pp] * np.mean(Rnorm_acc[pp])
            row += f"|  {contrib:.3e}      "
        print(row)

    print("\nRatio of each order to the previous (measured):")
    for h in (1e-3, 1e-2, 1e-1):
        contribs = {pp: (h ** pp) / D_P[pp] * np.mean(Rnorm_acc[pp])
                    for pp in range(3, p_max + 1)}
        ratios = []
        for pp in range(4, p_max + 1):
            r = contribs[pp] / max(contribs[pp - 1], 1e-300)
            ratios.append(f"R{pp}/R{pp-1}={r:.3e}")
        print(f"  Delta_T={h:.0e}: " + "  ".join(ratios))

    print("\n[mag] DONE.")
    print("[mag] Interpretation: if (h^4/24)||R_4|| is not <<(h^3/12)||R_3|| at")
    print("[mag] your deployment Delta_T, the higher-order closure is worth it;")
    print("[mag] if it is orders of magnitude smaller, R_3 alone suffices.")


if __name__ == '__main__':
    main()
