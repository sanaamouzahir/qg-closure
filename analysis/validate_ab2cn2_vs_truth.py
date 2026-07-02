"""
validate_ab2cn2_vs_truth.py
===========================

Verify the FULL AB2-CN2 truncation-error expansion against the exact flow
(fine RK4 as the practical "truth"), confirming every operator R_3..R_6 of

    tau = omega(t+h) - omega_AB2CN2(t+h)
        = -(h^3/12) R_3 - (h^4/24) R_4 - (h^5/240) R_5 - (h^6/1440) R_6 + O(h^7)

    R_3 = L^3 w + L^2 N + L Ndot - 5 Nddot
    R_4 = 2L^4 w + 2L^3 N + 2L^2 Ndot - 4 L Nddot + Ndddot
    R_5 = 13L^5 w + 13L^4 N + 13L^3 Ndot - 17L^2 Nddot + 8 L Ndddot - 7 N4
    R_6 = 43L^6 w + 43L^5 N + 43L^4 Ndot - 47L^3 Nddot + 28L^2 Ndddot
          - 17 L N4 + 4 N5

This is the companion to step3's test_c.  test_c compared coarse AB2CN2 to
K-fine AB2CN2 (the (1 - 1/K^2) factor) and only exercised R_3.  Here we
compare a SINGLE bare AB2CN2 step of size h, started from the exact history,
against the exact flow, and check the local truncation error term by term.

THE TEST (per snapshot, swept over h):
  1.  omega^{n-1} = exact omega(t - h)      [RK4-fine, backward]
  2.  omega_truth = exact omega(t + h)      [RK4-fine, forward]
  3.  omega_ab2   = ONE AB2CN2 step of size h from (omega^n, omega^{n-1})
  4.  tau_emp     = omega_truth - omega_ab2                 (measured LTE)
  5.  tau_pred^{(P)} = -sum_{p=3}^{P} (h^p / D_p) R_p       (analytical)
  6.  residual^{(P)} = || tau_emp - tau_pred^{(P)} ||

CONVERGENCE SIGNATURE (the proof):
    || tau_emp ||         ~ h^3      (slope 3)
    || residual^{(3)} ||  ~ h^4      (keep R_3  -> next order is h^4)
    || residual^{(4)} ||  ~ h^5      (add R_4)
    || residual^{(5)} ||  ~ h^6      (add R_5)
    || residual^{(6)} ||  ~ h^7      (add R_6)
Each added operator drops the residual order by one.  Slopes up to 6 are
cleanly resolvable in float64; the slope-7 line (R_6) sits at the roundoff
floor because h^7 underflows quickly -- R_6 is confirmed symbolically instead.

Works for decaying turbulence (F=None) and forced turbulence (F from yaml).

Usage
-----
  python validate_ab2cn2_vs_truth.py \
      --omega-source /gdata/.../decaying_turb_restart_t60_dt1em5/DNS_FR_omega.npy \
      --times        /gdata/.../decaying_turb_restart_t60_dt1em5/DNS_FR_times.npy \
      --source-yaml  /gdata/.../decaying_turbulence.yaml \
      --scenario decaying_turbulence \
      --n-snaps 3 --device cuda --out-dir .
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qg.solver.opt.basis import to_spectral, to_physical


# --------------------------------------------------------------------------- #
# Operators (match step3 / build_training_data conventions)                   #
# --------------------------------------------------------------------------- #

def J_phys(psi_phys, omega_phys, derivative, dealias=True):
    """+J(psi, omega).  Optional 2/3 dealiasing on the nonlinear products
    (the production solver and closure dealias; step3's bare J did not).
    Must be consistent across truth, scheme, and the analytical R's."""
    psih = to_spectral(psi_phys)
    qh   = to_spectral(omega_phys)
    uh = -1 * derivative.dy * psih
    vh = +1 * derivative.dx * psih
    u = to_physical(uh); v = to_physical(vh); q = to_physical(qh)
    uq_h = to_spectral(u * q); vq_h = to_spectral(v * q)
    if dealias:
        uq_h = uq_h.clone(); vq_h = vq_h.clone()
        derivative.dealias(uq_h); derivative.dealias(vq_h)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h
    return to_physical(j_hat)


def L_apply(field_phys, L_hat, power=1):
    return to_physical((L_hat ** power) * to_spectral(field_phys))


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


def ab2cn2_step_spectral(qh_n, qh_nm1, dt, derivative, L_hat, F_phys, dealias=True):
    """One bare AB2CN2 IMEX step (identical to step3 / the solver)."""
    def N_at_qh(qh):
        psi = to_physical(derivative.inv_laplacian * qh)
        omega = to_physical(qh)
        N_phys = -1.0 * J_phys(psi, omega, derivative, dealias=dealias)
        if F_phys is not None:
            N_phys = N_phys + F_phys
        return to_spectral(N_phys)
    Nh_n   = N_at_qh(qh_n)
    Nh_nm1 = N_at_qh(qh_nm1)
    AB2_Nh = 1.5 * Nh_n - 0.5 * Nh_nm1
    rhs_hat   = qh_n + dt * (0.5 * L_hat * qh_n + AB2_Nh)
    denom_hat = 1.0 - 0.5 * dt * L_hat
    return rhs_hat / denom_hat


def rk4_step(omega, dt, derivative, L_hat, F_phys, dealias=True):
    def rhs(om):
        return L_apply(om, L_hat, 1) + N_of(om, derivative, F_phys, dealias=dealias)
    k1 = rhs(omega)
    k2 = rhs(omega + 0.5 * dt * k1)
    k3 = rhs(omega + 0.5 * dt * k2)
    k4 = rhs(omega + dt * k3)
    return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_rk4(omega, h, h_sub, derivative, L_hat, F_phys, dealias=True):
    """Integrate the exact PDE by total time h using RK4 substeps of size
    ~h_sub (sign of h sets direction).  This is the 'truth'."""
    n = max(4, int(round(abs(h) / h_sub)))
    dt = h / n
    om = omega.clone()
    for _ in range(n):
        om = rk4_step(om, dt, derivative, L_hat, F_phys, dealias=dealias)
    return om


# --------------------------------------------------------------------------- #
# Recursive chain-rule bootstrap  omega^(k), N^(m)                            #
# --------------------------------------------------------------------------- #

def compute_derivatives(omega, derivative, L_hat, F_phys, max_m=5, dealias=True):
    omega_list = [omega]
    psi_list   = [inv_lap(omega, derivative)]
    N_list     = []
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


# --------------------------------------------------------------------------- #
# R_p assembly + cumulative analytical tau                                    #
# --------------------------------------------------------------------------- #

R_DEFS = {
    3: [( 1, 'omega', 3), ( 1, 'N', 2), ( 1, 'N1', 1), (-5, 'N2', 0)],
    4: [( 2, 'omega', 4), ( 2, 'N', 3), ( 2, 'N1', 2), (-4, 'N2', 1), ( 1, 'N3', 0)],
    5: [(13, 'omega', 5), (13, 'N', 4), (13, 'N1', 3), (-17,'N2', 2), ( 8, 'N3', 1), (-7, 'N4', 0)],
    6: [(43, 'omega', 6), (43, 'N', 5), (43, 'N1', 4), (-47,'N2', 3), (28, 'N3', 2), (-17,'N4', 1), ( 4, 'N5', 0)],
}
D_P = {3: 12, 4: 24, 5: 240, 6: 1440}
_KIND = {'N': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'N4': 4, 'N5': 5}


def assemble_R(p, derivs, L_hat):
    omega0 = derivs['omega'][0]
    N_list = derivs['N']
    R = None
    for coeff, kind, Lpow in R_DEFS[p]:
        base = omega0 if kind == 'omega' else N_list[_KIND[kind]]
        field = L_apply(base, L_hat, Lpow) if Lpow > 0 else base
        contrib = coeff * field
        R = contrib if R is None else R + contrib
    return R


def tau_pred_cumulative(h, derivs, L_hat, P):
    """-sum_{p=3}^{P} (h^p / D_p) R_p."""
    acc = None
    for p in range(3, P + 1):
        term = -(h ** p) / D_P[p] * assemble_R(p, derivs, L_hat)
        acc = term if acc is None else acc + term
    return acc


def _rms(t):
    return float(torch.sqrt(torch.mean(t ** 2)))


# --------------------------------------------------------------------------- #
# FD validation of N-derivatives (direct chain-rule check)                    #
# --------------------------------------------------------------------------- #

_FD = {
    1: ({-1: -0.5, 1: 0.5}, 1),
    2: ({-1: 1.0, 0: -2.0, 1: 1.0}, 2),
    3: ({-2: -0.5, -1: 1.0, 1: -1.0, 2: 0.5}, 3),
    4: ({-2: 1.0, -1: -4.0, 0: 6.0, 1: -4.0, 2: 1.0}, 4),
    5: ({-3: -0.5, -2: 2.0, -1: -2.5, 1: 2.5, 2: -2.0, 3: 0.5}, 5),
}


def fd_validate(omega0, derivative, L_hat, F_phys, max_m, fd_dt, dealias=True):
    max_off = max(max(abs(o) for o in st[0]) for st in _FD.values())
    N_at = {0: N_of(omega0, derivative, F_phys, dealias=dealias)}
    om = omega0.clone()
    for k in range(1, max_off + 1):
        om = rk4_step(om, +fd_dt, derivative, L_hat, F_phys, dealias=dealias)
        N_at[k] = N_of(om, derivative, F_phys, dealias=dealias)
    om = omega0.clone()
    for k in range(1, max_off + 1):
        om = rk4_step(om, -fd_dt, derivative, L_hat, F_phys, dealias=dealias)
        N_at[-k] = N_of(om, derivative, F_phys, dealias=dealias)
    out = {}
    for m in range(1, max_m + 1):
        stencil, p = _FD[m]
        acc = None
        for off, c in stencil.items():
            term = c * N_at[off]
            acc = term if acc is None else acc + term
        out[m] = acc / (fd_dt ** p)
    return out


# --------------------------------------------------------------------------- #
# Snapshot loading + forcing                                                  #
# --------------------------------------------------------------------------- #

def load_snapshots(omega_source, times_path, batch_index, n_snaps, t_start):
    suffix = omega_source.suffix.lower()
    if suffix == '.npy':
        arr = np.load(omega_source, mmap_mode='r')
        if arr.ndim == 4:
            arr = arr[batch_index]
        if arr.ndim == 3:
            times = np.load(times_path) if times_path is not None \
                else np.arange(arr.shape[0], dtype=float)
            if times.ndim == 2:
                times = times[batch_index]
            valid = np.where(times >= t_start)[0]
            lo = valid[0] if len(valid) else 0
            idxs = sorted(set(np.linspace(lo, arr.shape[0] - 1, n_snaps)
                              .round().astype(int).tolist()))
            return [(float(times[i]), np.asarray(arr[i]).astype(np.float64)) for i in idxs]
        elif arr.ndim == 2:
            return [(0.0, np.asarray(arr).astype(np.float64))]
    elif suffix == '.npz':
        rec = np.load(omega_source)
        key = 'omega_0' if 'omega_0' in rec.files else rec.files[0]
        a = rec[key].astype(np.float64)
        if a.ndim == 3: a = a[0]
        t = float(rec['seed_t']) if 'seed_t' in rec.files else 0.0
        return [(t, a)]
    sys.exit(f"ERROR: bad omega-source '{omega_source}'")


def build_F_phys(yaml_cfg, scenario, Lx, Ly, Nx, Ny, device, dtype):
    if scenario != 'forced_turbulence':
        return None
    fcfg = yaml_cfg.get('qg', {}).get('forcing') or yaml_cfg.get('forcing')
    if not fcfg or fcfg.get('function') != 'unscaled_cosine':
        sys.exit(f"ERROR: forced_turbulence but no unscaled_cosine forcing ({fcfg})")
    A = float(fcfg.get('A', 0)); B = float(fcfg.get('B', 0)); C = float(fcfg.get('C', 0))
    D = float(fcfg.get('D', 0)); E = float(fcfg.get('E', 0)); Fc = float(fcfg.get('F', 0))
    if abs(C) > 1e-30 or abs(Fc) > 1e-30:
        sys.exit(f"ERROR: time-dependent forcing (C={C},F={Fc}) breaks Fdot=Fddot=0")
    x = torch.linspace(0.0, Lx, Nx, device=device, dtype=dtype)
    y = torch.linspace(0.0, Ly, Ny, device=device, dtype=dtype)
    F2d = A * torch.cos(B * x)[None, :] + D * torch.cos(E * y)[:, None]
    print(f"[vs-truth] forcing |F|_rms={_rms(F2d):.4e}")
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
                   choices=['decaying_turbulence', 'forced_turbulence', 'flow_past_cylinder'])
    p.add_argument('--batch-index', type=int, default=0)
    p.add_argument('--n-snaps', type=int, default=3)
    p.add_argument('--t-start', type=float, default=0.0)
    p.add_argument('--h-max', type=float, default=2.0e-2,
                   help='largest step in the convergence sweep (default 2e-2; '
                        'keep below the convergence radius ~2e-2 for forced turb)')
    p.add_argument('--h-min', type=float, default=2.0e-3,
                   help='smallest step (default 2e-3; below this the h^6,h^7 '
                        'residuals underflow the float64 roundoff floor)')
    p.add_argument('--n-h', type=int, default=8, help='number of h values (log-spaced)')
    p.add_argument('--h-sub', type=float, default=1.0e-5,
                   help='RK4 substep for the truth integration (default 1e-5)')
    p.add_argument('--fd-dt', type=float, default=2.0e-3,
                   help='dt for the secondary FD chain-rule check')
    p.add_argument('--no-dealias', action='store_true')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out-dir', type=Path, default=Path('.'))
    args = p.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    dtype = torch.float64
    dealias = not args.no_dealias
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.source_yaml) as f:
        yaml_cfg = yaml.safe_load(f)

    def yget(path, d=None):
        cur = yaml_cfg
        for k in path:
            if isinstance(cur, dict) and k in cur: cur = cur[k]
            else: return d
        return cur
    Nx = int(yget(['qg', 'grid', 'Nx'], yget(['grid', 'Nx'], 256)))
    Ny = int(yget(['qg', 'grid', 'Ny'], yget(['grid', 'Ny'], 256)))
    Lx = float(yget(['qg', 'grid', 'Lx'], yget(['grid', 'Lx'], 2 * math.pi)))
    Ly = float(yget(['qg', 'grid', 'Ly'], yget(['grid', 'Ly'], 2 * math.pi)))
    nu = float(yget(['qg', 'pde', 'nu'], yget(['pde', 'nu'], 0.0)))
    mu = float(yget(['qg', 'pde', 'mu'], yget(['pde', 'mu'], 0.0)))
    B  = float(yget(['qg', 'pde', 'B'],  yget(['pde', 'B'], 0.0)))
    print(f"[vs-truth] scenario={args.scenario} Nx={Nx} Ny={Ny} Lx={Lx:.4f} Ly={Ly:.4f}")
    print(f"[vs-truth] nu={nu} mu={mu} beta={B} dealias={dealias} device={device}")

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
    print(f"[vs-truth] {len(snaps)} snapshots at t={[f'{t:.3f}' for t,_ in snaps]}")

    h_list = np.geomspace(args.h_max, args.h_min, args.n_h)

    # Accumulators: per-h lists of RMS, averaged over snapshots
    emp   = {h: [] for h in h_list}
    res   = {P: {h: [] for h in h_list} for P in (3, 4, 5, 6)}
    fd_rel = {m: [] for m in range(1, 6)}

    for (t_snap, om_np) in snaps:
        omega = torch.tensor(om_np, dtype=dtype, device=device)[None]
        derivs = compute_derivatives(omega, derivative, L_hat, F_phys,
                                     max_m=5, dealias=dealias)
        # secondary: direct FD check of the chain-rule derivatives
        fd = fd_validate(omega, derivative, L_hat, F_phys, 5, args.fd_dt, dealias=dealias)
        print(f"\n[vs-truth] snapshot t={t_snap:.3f}  |omega|={_rms(omega):.3e}")
        names = {1:'Ndot',2:'Nddot',3:'Ndddot',4:'N4',5:'N5'}
        for m in range(1, 6):
            rel = _rms(derivs['N'][m] - fd[m]) / max(_rms(fd[m]), 1e-30)
            fd_rel[m].append(rel)
            print(f"    chain-rule {names[m]:>7s} vs FD: rel.diff={rel:.2e}")

        for h in h_list:
            h = float(h)
            om_minus = integrate_rk4(omega, -h, args.h_sub, derivative, L_hat,
                                     F_phys, dealias=dealias)
            om_truth = integrate_rk4(omega, +h, args.h_sub, derivative, L_hat,
                                     F_phys, dealias=dealias)
            qh_n  = to_spectral(omega)
            qh_m1 = to_spectral(om_minus)
            qh_ab2 = ab2cn2_step_spectral(qh_n, qh_m1, h, derivative, L_hat,
                                          F_phys, dealias=dealias)
            om_ab2 = to_physical(qh_ab2)
            tau_emp = om_truth - om_ab2
            emp[h].append(_rms(tau_emp))
            for P in (3, 4, 5, 6):
                tp = tau_pred_cumulative(h, derivs, L_hat, P)
                res[P][h].append(_rms(tau_emp - tp))

    # ---- averaged + slopes ----
    def avg(d): return np.array([np.mean(d[h]) for h in h_list])
    lh = np.log(h_list)
    def slope(y): return float(np.polyfit(lh, np.log(y + 1e-300), 1)[0])

    emp_a = avg(emp)
    res_a = {P: avg(res[P]) for P in (3, 4, 5, 6)}

    print("\n" + "=" * 72)
    print("CONVERGENCE OF AB2CN2 LTE vs EXACT FLOW (averaged over snapshots)")
    print("=" * 72)
    print(f"\nChain-rule vs FD (mean rel.diff): " +
          "  ".join(f"N^{m}={np.mean(fd_rel[m]):.1e}" for m in range(1, 6)))

    print(f"\n{'h':>10s} {'||tau_emp||':>13s} {'res R3':>11s} {'res R3-4':>11s}"
          f" {'res R3-5':>11s} {'res R3-6':>11s}")
    for i, h in enumerate(h_list):
        print(f"{h:10.3e} {emp_a[i]:13.4e} {res_a[3][i]:11.3e} {res_a[4][i]:11.3e}"
              f" {res_a[5][i]:11.3e} {res_a[6][i]:11.3e}")

    print("\nFitted log-log slopes (over full h-range):")
    print(f"  ||tau_emp||            slope = {slope(emp_a):.3f}   (expect 3)")
    print(f"  residual keep R3       slope = {slope(res_a[3]):.3f}   (expect 4)")
    print(f"  residual keep R3..R4   slope = {slope(res_a[4]):.3f}   (expect 5)")
    print(f"  residual keep R3..R5   slope = {slope(res_a[5]):.3f}   (expect 6)")
    print(f"  residual keep R3..R6   slope = {slope(res_a[6]):.3f}   (expect 7; "
          f"floor-limited in float64)")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.loglog(h_list, emp_a, 'ko-', lw=1.6, label=r'$\|\tau_{\rm emp}\|$ (slope 3)')
    cols = {3: 'C0', 4: 'C1', 5: 'C2', 6: 'C3'}
    for P in (3, 4, 5, 6):
        ax.loglog(h_list, res_a[P] + 1e-300, 'o-', color=cols[P], lw=1.3,
                  label=rf'residual keep $R_3..R_{{{P}}}$ (slope {P+1})')
    # reference slope guides anchored at h_max
    h0 = h_list[0]
    for order, y0, c in [(3, emp_a[0], 'k'), (4, res_a[3][0], 'C0'),
                         (5, res_a[4][0], 'C1'), (6, res_a[5][0], 'C2'),
                         (7, res_a[6][0], 'C3')]:
        ax.loglog(h_list, y0 * (h_list / h0) ** order, '--', color=c, alpha=0.4, lw=0.9)
    ax.set_xlabel(r'$h$'); ax.set_ylabel(r'RMS error')
    ax.set_title('AB2-CN2 local truncation error vs exact flow:\n'
                 'residual order drops by one per operator added')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig_path = args.out_dir / f'validate_ab2cn2_vs_truth_{args.scenario}.png'
    fig.savefig(fig_path, dpi=120, bbox_inches='tight')
    print(f"\n[vs-truth] wrote {fig_path}")
    print("[vs-truth] DONE.")


if __name__ == '__main__':
    main()
