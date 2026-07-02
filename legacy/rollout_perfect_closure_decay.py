"""
rollout_perfect_closure.py
==========================

Standalone rollout test for the closure CEILING. No NN, no dataset, no manifest.

What it does
------------
1. Read omega(t*) from a float64 source .npy file (the parent FR run's DNS_FR_omega.npy)
2. RK4 warmup to build the AB2-CN2 stencil at coarse-Delta_T spacing:
       omega(t* + 1*Delta_T) = omega_m1
       omega(t* + 2*Delta_T) = omega_m2
       omega(t* + 3*Delta_T) = omega_0   <-- start the rollout here
3. Run THREE parallel trajectories from omega_0:
       truth:   K fine AB2-CN2 steps of size h_fine = Delta_T/K, per coarse step
       bare:    1 coarse AB2-CN2 step of size Delta_T
       closure: 1 coarse AB2-CN2 step + e_anal_inc + fraction * f_NN_target_inc
   where:
       e_anal       = (1/12) * [L^3 omega + L^2 N]
       f_NN_target  = (1/12) * [L*Ndot - 5*Nddot]   <-- computed analytically via chain rule
       both increments are scaled by  -coef = -Delta_T^3 * (1 - 1/K^2)
4. Plot all the usual diagnostics, including the same 4-row figure as the
   full rollout script.

Usage
-----
  python rollout_perfect_closure.py \
      --source-omega /path/to/DNS_FR_omega.npy \
      --source-times /path/to/DNS_FR_times.npy \
      --scenario-yaml /path/to/decaying_turbulence.yaml \
      --seed-time 5.0 \
      --n-steps 1000 \
      --Delta-T 1e-3 \
      --K 100 \
      --h-ultrafine 1e-5 \
      --perfect-nn-fraction 0.95 \
      --device cpu \
      --out-dir .

What gets written
-----------------
<out-dir>/rollout_perfect_<tag>.png
<out-dir>/rollout_perfect_<tag>.npz
"""

from __future__ import annotations
import argparse
import sys
import time
import yaml
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SPECTRUM_KMIN = 3.0
SPECTRUM_KMAX = 60.0


# --------------------------------------------------------------------------- #
# Solver primitives (lifted from rollout_multistep_comparison.py).            #
# --------------------------------------------------------------------------- #

def J_phys(psi_phys, omega_phys, derivative):
    """Dealiased J(psi, omega) via spectral derivatives."""
    from qg.solver.opt.basis import to_spectral, to_physical
    psih = to_spectral(psi_phys)
    qh   = to_spectral(omega_phys)
    uh = -1 * derivative.dy * psih
    vh = +1 * derivative.dx * psih
    u = to_physical(uh)
    v = to_physical(vh)
    q = to_physical(qh)
    uq_h = to_spectral(u * q).clone()
    vq_h = to_spectral(v * q).clone()
    derivative.dealias(uq_h)
    derivative.dealias(vq_h)
    j_hat = derivative.dx * uq_h + derivative.dy * vq_h
    return to_physical(j_hat)


def build_L_hat(derivative, nu, mu, beta):
    L_hat = nu * derivative.laplacian - mu
    if beta != 0.0:
        L_hat = L_hat - beta * derivative.dx * derivative.inv_laplacian
    return L_hat


def L_op(omega_phys, L_hat):
    from qg.solver.opt.basis import to_spectral, to_physical
    return to_physical(L_hat * to_spectral(omega_phys))


def ab2cn2_step_spectral(qh_n, qh_nm1, dt, derivative, L_hat, F_phys):
    from qg.solver.opt.basis import to_spectral, to_physical
    def N_at_qh(qh):
        psi = to_physical(derivative.inv_laplacian * qh)
        omega = to_physical(qh)
        N_phys = -1.0 * J_phys(psi, omega, derivative)
        if F_phys is not None:
            N_phys = N_phys + F_phys
        return to_spectral(N_phys)
    Nh_n   = N_at_qh(qh_n)
    Nh_nm1 = N_at_qh(qh_nm1)
    AB2_Nh = 1.5 * Nh_n - 0.5 * Nh_nm1
    rhs_hat   = qh_n + dt * (0.5 * L_hat * qh_n + AB2_Nh)
    denom_hat = 1.0 - 0.5 * dt * L_hat
    return rhs_hat / denom_hat


def rk4_step(omega, dt, derivative, L_hat, F_phys):
    def rhs(om):
        from qg.solver.opt.basis import to_spectral, to_physical
        psi = to_physical(derivative.inv_laplacian * to_spectral(om))
        N = -1.0 * J_phys(psi, om, derivative)
        if F_phys is not None:
            N = N + F_phys
        return L_op(om, L_hat) + N
    k1 = rhs(omega)
    k2 = rhs(omega + 0.5 * dt * k1)
    k3 = rhs(omega + 0.5 * dt * k2)
    k4 = rhs(omega + dt * k3)
    return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def psi_from_omega(omega_phys, derivative):
    from qg.solver.opt.basis import to_spectral, to_physical
    return to_physical(derivative.inv_laplacian * to_spectral(omega_phys))


# --------------------------------------------------------------------------- #
# Analytical closure pieces                                                   #
# --------------------------------------------------------------------------- #

def E_analytical_phys(omega_phys, derivative, L_hat, F_phys):
    """f_anal = (1/12) * [L^3 omega + L^2 N]."""
    from qg.solver.opt.basis import to_spectral, to_physical
    qh = to_spectral(omega_phys)
    L3_omega = to_physical(L_hat ** 3 * qh)
    psi = to_physical(derivative.inv_laplacian * qh)
    N = -1.0 * J_phys(psi, omega_phys, derivative)
    if F_phys is not None:
        N = N + F_phys
    Nh = to_spectral(N)
    L2_N = to_physical(L_hat ** 2 * Nh)
    return (1.0 / 12.0) * (L3_omega + L2_N)


def compute_f_NN_target_phys(omega_phys, derivative, L_hat, F_phys):
    """f_NN_target = (1/12) * [L*Ndot - 5*Nddot], via chain rule.

    Chain rule (F time-independent, J bilinear):
        omega_dot  = L*omega + N
        psi_dot    = inv_lap(omega_dot)
        Ndot       = -J(psi_dot, omega) - J(psi, omega_dot)
        omega_ddot = L*omega_dot + Ndot
        psi_ddot   = inv_lap(omega_ddot)
        Nddot      = -J(psi_ddot, omega) - 2*J(psi_dot, omega_dot)
                       - J(psi, omega_ddot)
    """
    from qg.solver.opt.basis import to_spectral, to_physical
    qh = to_spectral(omega_phys)
    psi = to_physical(derivative.inv_laplacian * qh)
    L_om = to_physical(L_hat * qh)
    N = -1.0 * J_phys(psi, omega_phys, derivative)
    if F_phys is not None:
        N = N + F_phys
    omega_dot = L_om + N
    psi_dot   = to_physical(derivative.inv_laplacian * to_spectral(omega_dot))

    N_dot = -1.0 * J_phys(psi_dot, omega_phys, derivative) \
            - 1.0 * J_phys(psi, omega_dot, derivative)

    L_om_dot = to_physical(L_hat * to_spectral(omega_dot))
    omega_ddot = L_om_dot + N_dot
    psi_ddot   = to_physical(derivative.inv_laplacian * to_spectral(omega_ddot))

    N_ddot = -1.0 * J_phys(psi_ddot, omega_phys, derivative) \
             - 2.0 * J_phys(psi_dot,  omega_dot,  derivative) \
             - 1.0 * J_phys(psi,      omega_ddot, derivative)

    L_N_dot = to_physical(L_hat * to_spectral(N_dot))
    return (1.0 / 12.0) * (L_N_dot - 5.0 * N_ddot)


# --------------------------------------------------------------------------- #
# Spectra                                                                     #
# --------------------------------------------------------------------------- #

def radial_sum_spectrum(field2d, Lx, Ly):
    Ny, Nx = field2d.shape
    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2.0 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    Kmag = np.sqrt(KX ** 2 + KY ** 2)
    fhat = np.fft.fft2(field2d) / (Nx * Ny)
    psd = np.abs(fhat) ** 2
    dk = max(2.0 * np.pi / Lx, 2.0 * np.pi / Ly)
    kmax = float(Kmag.max())
    bins = np.arange(0.0, kmax + dk, dk)
    sp, _ = np.histogram(Kmag.ravel(), bins=bins, weights=psd.ravel())
    kc = 0.5 * (bins[:-1] + bins[1:])
    return kc, sp / dk


def enstrophy_spectrum(omega_phys, Lx, Ly):
    kc, sp = radial_sum_spectrum(omega_phys, Lx, Ly)
    return kc, 0.5 * sp


def energy_spectrum(omega_phys, Lx, Ly):
    kc, sp = radial_sum_spectrum(omega_phys, Lx, Ly)
    k_safe = np.where(kc <= 0, np.inf, kc)
    return kc, 0.5 * sp / (k_safe ** 2)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description="Standalone perfect-closure ceiling rollout. No NN, no dataset."
    )
    p.add_argument('--source-omega', type=Path, required=True,
                   help='float64 .npy with shape (n_batches, T, Ny, Nx) or (T, Ny, Nx)')
    p.add_argument('--source-times', type=Path, default=None,
                   help='float64 .npy with the time array. Optional; if not '
                        'given, assumes uniform spacing from --source-dt')
    p.add_argument('--source-dt', type=float, default=0.05,
                   help='Time spacing in the source .npy (default 0.05). '
                        'Ignored if --source-times is given.')
    p.add_argument('--source-batch', type=int, default=0,
                   help='Which batch (first dim) to use, if shape is 4D.')
    p.add_argument('--scenario-yaml', type=Path, required=True,
                   help='Scenario YAML with physical params: Nx/Ny, Lx/Ly, nu, etc.')
    p.add_argument('--seed-time', type=float, default=5.0,
                   help='Pick the source snapshot closest to this time.')
    p.add_argument('--n-steps', type=int, default=1000,
                   help='Number of coarse rollout steps.')
    p.add_argument('--Delta-T', type=float, default=1e-3,
                   help='Coarse timestep.')
    p.add_argument('--K', type=int, default=100,
                   help='Fine-steps-per-coarse-step.')
    p.add_argument('--h-ultrafine', type=float, default=1e-5,
                   help='Ultrafine RK4 sub-step for the warmup. Default = h_fine.')
    p.add_argument('--perfect-nn-fraction', type=float, default=0.95,
                   help='Fraction of analytical f_NN_target to apply. 1.0 = '
                        'full ceiling, 0.95 = 5%% deficit (simulated NN error).')
    p.add_argument('--snapshot-fracs', type=str, default='0,0.25,0.5,1.0',
                   help='Fractions of n_steps at which to render omega panels.')
    p.add_argument('--device', type=str, default='cpu',
                   choices=['cpu', 'cuda'])
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--tag', type=str, default='test',
                   help='Suffix for output files.')
    args = p.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[rollout] CUDA not available, falling back to CPU")
        device = 'cpu'
    dtype = torch.float64
    args.out_dir.mkdir(parents=True, exist_ok=True)

    h_fine = args.Delta_T / args.K
    if args.h_ultrafine > h_fine:
        print(f"[rollout] WARNING: h_ultrafine ({args.h_ultrafine:g}) > h_fine "
              f"({h_fine:g}). Using h_fine for warmup.")
        args.h_ultrafine = h_fine

    print(f"[rollout] device={device}  dtype=float64")
    print(f"[rollout] Delta_T={args.Delta_T:g}  K={args.K}  h_fine={h_fine:g}  "
          f"h_uf={args.h_ultrafine:g}")
    print(f"[rollout] perfect-NN fraction = {args.perfect_nn_fraction:.4f}")

    # ----- Read scenario yaml ----- #
    with open(args.scenario_yaml) as f:
        scen = yaml.safe_load(f)
    Nx = int(scen.get('Nx', 256))
    Ny = int(scen.get('Ny', 256))
    Lx = float(scen.get('Lx', 2 * np.pi))
    Ly = float(scen.get('Ly', 2 * np.pi))
    nu = float(scen.get('nu', 1.025e-5))
    mu = float(scen.get('mu', 0.0))
    beta = float(scen.get('beta', 0.0))
    print(f"[rollout] grid Nx={Nx} Ny={Ny} Lx={Lx:.4f} Ly={Ly:.4f}")
    print(f"[rollout] nu={nu} mu={mu} beta={beta}")

    # ----- Load source omega ----- #
    omega_src = np.load(args.source_omega, mmap_mode='r')
    print(f"[rollout] source omega: shape={omega_src.shape}, dtype={omega_src.dtype}")
    if omega_src.ndim == 4:
        print(f"[rollout] using batch index {args.source_batch}")
        omega_src_traj = omega_src[args.source_batch]
    elif omega_src.ndim == 3:
        omega_src_traj = omega_src
    else:
        raise RuntimeError(f"unexpected shape {omega_src.shape}")
    T_save = omega_src_traj.shape[0]

    if args.source_times is not None:
        t_save = np.load(args.source_times)
        if t_save.ndim == 2:
            t_save = t_save[args.source_batch]
        assert len(t_save) == T_save, f"time/omega length mismatch"
        print(f"[rollout] source times: T={T_save}, t[0]={t_save[0]:.4f}, t[-1]={t_save[-1]:.4f}")
    else:
        t_save = np.arange(T_save) * args.source_dt
        print(f"[rollout] using inferred times with dt={args.source_dt:g}")

    # Find seed snapshot
    seed_idx = int(np.argmin(np.abs(t_save - args.seed_time)))
    t_seed = float(t_save[seed_idx])
    print(f"[rollout] seed snapshot: index {seed_idx}, t={t_seed:.4f}")

    omega_seed = np.asarray(omega_src_traj[seed_idx], dtype=np.float64)
    if omega_seed.shape != (Ny, Nx):
        raise RuntimeError(f"omega_seed shape {omega_seed.shape} != ({Ny}, {Nx})")
    print(f"[rollout] |omega_seed|_rms = {float(np.sqrt(np.mean(omega_seed**2))):.4e}")

    # ----- Grid + operators ----- #
    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device,
                          precision='float64')
    derivative = Derivative(grid)
    for attr in ('dx', 'dy', 'laplacian', 'inv_laplacian'):
        setattr(derivative, attr, getattr(derivative, attr).to(device))
    L_hat = build_L_hat(derivative, nu=nu, mu=mu, beta=beta).to(device)
    F_phys = None
    print(f"[rollout] operators built")

    # ----- RK4 warmup: build (omega_m2, omega_m1, omega_0) at Delta_T spacing ----- #
    print(f"[rollout] RK4 warmup: {3*int(round(args.Delta_T/args.h_ultrafine))} "
          f"sub-steps of size {args.h_ultrafine:g}")
    omega_t = torch.tensor(omega_seed, dtype=dtype, device=device)[None]
    n_uf_per_DT = int(round(args.Delta_T / args.h_ultrafine))
    omega_m2_t = omega_m1_t = omega_0_t = None
    t_warm_start = time.time()
    for step in range(3 * n_uf_per_DT):
        omega_t = rk4_step(omega_t, args.h_ultrafine, derivative, L_hat, F_phys)
        if step + 1 == 1 * n_uf_per_DT:
            omega_m2_t = omega_t.clone()
        elif step + 1 == 2 * n_uf_per_DT:
            omega_m1_t = omega_t.clone()
        elif step + 1 == 3 * n_uf_per_DT:
            omega_0_t = omega_t.clone()
    print(f"[rollout] warmup done in {time.time()-t_warm_start:.1f}s")
    print(f"[rollout] |omega_m2|={float(torch.sqrt((omega_m2_t**2).mean())):.4e}  "
          f"|omega_m1|={float(torch.sqrt((omega_m1_t**2).mean())):.4e}  "
          f"|omega_0|={float(torch.sqrt((omega_0_t**2).mean())):.4e}")

    psi_0_t  = psi_from_omega(omega_0_t,  derivative)
    psi_m1_t = psi_from_omega(omega_m1_t, derivative)
    psi_m2_t = psi_from_omega(omega_m2_t, derivative)

    omega_init = [omega_0_t, omega_m1_t, omega_m2_t]
    psi_init   = [psi_0_t,   psi_m1_t,   psi_m2_t]

    # ----- Snapshot steps ----- #
    log_stride = max(1, args.n_steps // 500)
    frac_list = [float(s) for s in args.snapshot_fracs.split(',')]
    raw_snap = [int(round(f * args.n_steps)) for f in frac_list]
    snap_aligned = sorted({(s // log_stride) * log_stride for s in raw_snap})
    snapshot_steps = snap_aligned
    snap_set = set(snapshot_steps)
    print(f"[rollout] snapshot steps: {snapshot_steps}")

    # ============================================================== #
    # Run the triple rollout                                          #
    # ============================================================== #
    from qg.solver.opt.basis import to_spectral, to_physical

    om_t = omega_init[0].clone()
    om_t_minus = rk4_step(om_t, -h_fine, derivative, L_hat, F_phys)
    qh_t_curr  = to_spectral(om_t)
    qh_t_minus = to_spectral(om_t_minus)

    bare_omega = [w.clone() for w in omega_init]
    clos_omega = [w.clone() for w in omega_init]
    clos_psi   = [w.clone() for w in psi_init]

    K_factor = 1.0 - 1.0 / (args.K ** 2)
    coef = (args.Delta_T ** 3) * K_factor

    n = args.n_steps
    rel_bare = np.zeros(n + 1); abs_bare = np.zeros(n + 1)
    rel_clos = np.zeros(n + 1); abs_clos = np.zeros(n + 1)
    fine_rms = np.zeros(n + 1); bare_rms = np.zeros(n + 1); clos_rms = np.zeros(n + 1)
    fine_Z = np.zeros(n + 1);   bare_Z = np.zeros(n + 1);   clos_Z = np.zeros(n + 1)
    fine_frames = {}; bare_frames = {}; clos_frames = {}

    def _stash(step, om_t_np, om_b_np, om_c_np):
        fine_frames[step] = om_t_np.astype(np.float32).copy()
        bare_frames[step] = om_b_np.astype(np.float32).copy()
        clos_frames[step] = om_c_np.astype(np.float32).copy()

    om_t_np_0 = om_t[0].cpu().numpy()
    rms_t0 = float(np.sqrt(np.mean(om_t_np_0 ** 2)))
    fine_rms[0] = bare_rms[0] = clos_rms[0] = rms_t0
    Z_t0 = 0.5 * float(np.mean(om_t_np_0 ** 2))
    fine_Z[0] = bare_Z[0] = clos_Z[0] = Z_t0
    _stash(0, om_t_np_0, om_t_np_0, om_t_np_0)

    print(f"\n[rollout] === triple rollout ({n} coarse steps) ===")
    t0_wall = time.time()
    for step in range(1, n + 1):
        # Truth: K AB2-CN2 steps at h_fine
        for _ in range(args.K):
            qh_t_next = ab2cn2_step_spectral(qh_t_curr, qh_t_minus, h_fine,
                                              derivative, L_hat, F_phys)
            qh_t_minus, qh_t_curr = qh_t_curr, qh_t_next
        om_t = to_physical(qh_t_curr)

        # Bare: 1 AB2-CN2 step at Delta_T
        qh_b_curr  = to_spectral(bare_omega[0])
        qh_b_minus = to_spectral(bare_omega[1])
        qh_b_new = ab2cn2_step_spectral(qh_b_curr, qh_b_minus, args.Delta_T,
                                          derivative, L_hat, F_phys)
        om_b = to_physical(qh_b_new)
        bare_omega = [om_b, bare_omega[0], bare_omega[1]]

        # Closure: 1 AB2-CN2 step + analytical corrections
        qh_c_curr  = to_spectral(clos_omega[0])
        qh_c_minus = to_spectral(clos_omega[1])
        qh_c_bare = ab2cn2_step_spectral(qh_c_curr, qh_c_minus, args.Delta_T,
                                           derivative, L_hat, F_phys)
        om_c_bare = to_physical(qh_c_bare)
        E_anal_now = E_analytical_phys(clos_omega[0], derivative, L_hat, F_phys)
        e_anal_inc = -coef * E_anal_now
        f_NN_target = compute_f_NN_target_phys(clos_omega[0], derivative,
                                                L_hat, F_phys)
        e_NN_inc = -coef * args.perfect_nn_fraction * f_NN_target
        om_c = om_c_bare + e_anal_inc + e_NN_inc
        clos_omega = [om_c, clos_omega[0], clos_omega[1]]
        clos_psi   = [psi_from_omega(om_c, derivative),
                      clos_psi[0], clos_psi[1]]

        # Per-step scalars
        om_t_np = om_t[0].cpu().numpy()
        om_b_np = om_b[0].cpu().numpy()
        om_c_np = om_c[0].cpu().numpy()
        t_norm = float(np.sqrt(np.mean(om_t_np ** 2)))
        fine_rms[step] = t_norm
        bare_rms[step] = float(np.sqrt(np.mean(om_b_np ** 2)))
        clos_rms[step] = float(np.sqrt(np.mean(om_c_np ** 2)))
        fine_Z[step]   = 0.5 * float(np.mean(om_t_np ** 2))
        bare_Z[step]   = 0.5 * float(np.mean(om_b_np ** 2))
        clos_Z[step]   = 0.5 * float(np.mean(om_c_np ** 2))
        db = om_b_np - om_t_np
        dc = om_c_np - om_t_np
        abs_bare[step] = float(np.sqrt(np.mean(db ** 2)))
        abs_clos[step] = float(np.sqrt(np.mean(dc ** 2)))
        den = max(t_norm, 1e-30)
        rel_bare[step] = abs_bare[step] / den
        rel_clos[step] = abs_clos[step] / den

        if (step % log_stride == 0) or (step in snap_set) or (step == n):
            _stash(step, om_t_np, om_b_np, om_c_np)

        if step % max(1, n // 20) == 0 or step <= 3:
            el = time.time() - t0_wall
            eta = el * (n - step) / max(step, 1)
            ratio = rel_clos[step] / max(rel_bare[step], 1e-30)
            print(f"  step {step:5d}/{n}  rel_bare={rel_bare[step]:.3e}  "
                  f"rel_clos={rel_clos[step]:.3e}  ratio={ratio:.3f}  "
                  f"elapsed={el:.1f}s  eta={eta:.1f}s")

    print(f"\n[rollout] DONE in {(time.time()-t0_wall)/60:.1f} min")
    print(f"[rollout] final: rel_bare={rel_bare[-1]:.4e}  "
          f"rel_clos={rel_clos[-1]:.4e}  "
          f"improvement={rel_bare[-1]/max(rel_clos[-1], 1e-30):.2f}x")

    # ----- Save + plot ----- #
    retained_steps = sorted(fine_frames.keys())
    fine_hist = np.stack([fine_frames[s] for s in retained_steps], axis=0)
    bare_hist = np.stack([bare_frames[s] for s in retained_steps], axis=0)
    clos_hist = np.stack([clos_frames[s] for s in retained_steps], axis=0)
    retained_steps_arr = np.asarray(retained_steps, dtype=np.int64)

    fname_base = f"rollout_perfect_{args.tag}"
    npz_out = args.out_dir / f"{fname_base}.npz"
    np.savez(npz_out,
             fine_hist=fine_hist, bare_hist=bare_hist, clos_hist=clos_hist,
             retained_steps=retained_steps_arr,
             rel_bare=rel_bare, abs_bare=abs_bare,
             rel_clos=rel_clos, abs_clos=abs_clos,
             snapshot_steps=np.asarray(snapshot_steps, dtype=np.int64),
             fine_rms=fine_rms, bare_rms=bare_rms, clos_rms=clos_rms,
             fine_Z=fine_Z, bare_Z=bare_Z, clos_Z=clos_Z,
             Delta_T=args.Delta_T, K=args.K, h_fine=h_fine,
             Lx=Lx, Ly=Ly, nu=nu, mu=mu, beta=beta,
             perfect_nn_fraction=args.perfect_nn_fraction)
    print(f"[rollout] wrote {npz_out}")

    fig_path = args.out_dir / f"{fname_base}.png"
    render_figure(fine_hist, bare_hist, clos_hist, retained_steps_arr,
                   rel_bare, rel_clos, abs_bare, abs_clos,
                   fine_rms, bare_rms, clos_rms,
                   fine_Z, bare_Z, clos_Z,
                   snapshot_steps, args.Delta_T, args.K, Lx, Ly,
                   args.perfect_nn_fraction, fig_path)


def render_figure(fine_hist, bare_hist, clos_hist, retained_steps,
                   rel_bare, rel_clos, abs_bare, abs_clos,
                   fine_rms, bare_rms, clos_rms,
                   fine_Z, bare_Z, clos_Z,
                   snapshot_steps, Delta_T, K, Lx, Ly,
                   perfect_nn_fraction, out_path):
    step_to_idx = {int(s): i for i, s in enumerate(retained_steps)}
    def get_frames(step):
        i = step_to_idx[int(step)]
        return fine_hist[i], bare_hist[i], clos_hist[i]

    n_snaps = len(snapshot_steps)
    n_cols = 5
    n_bottom = 4
    fig_h = 3.2 * n_snaps + n_bottom * 3.0
    fig = plt.figure(figsize=(n_cols * 3.0, fig_h))
    outer = gridspec.GridSpec(n_snaps + n_bottom, n_cols, figure=fig,
                               hspace=0.55, wspace=0.30)

    for row_i, step in enumerate(snapshot_steps):
        omega_truth, omega_bare, omega_clos = get_frames(step)
        v = float(np.max(np.abs(np.stack([omega_truth, omega_bare, omega_clos]))))
        if v == 0: v = 1.0
        diff_bare = omega_bare - omega_truth
        diff_clos = omega_clos - omega_truth
        v_diff = float(np.max(np.abs(np.concatenate([diff_bare.ravel(),
                                                      diff_clos.ravel()]))))
        if v_diff == 0: v_diff = 1.0
        t_phys = step * Delta_T

        ax = fig.add_subplot(outer[row_i, 0])
        ax.imshow(omega_truth, cmap='RdBu_r', vmin=-v, vmax=v,
                  origin='lower', aspect='equal', interpolation='gaussian')
        ax.set_title(rf'truth  $t = {t_phys:.3f}$' '\n' rf'(step {step})',
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(rf'step {step}', fontsize=10)

        ax = fig.add_subplot(outer[row_i, 1])
        ax.imshow(omega_bare, cmap='RdBu_r', vmin=-v, vmax=v,
                  origin='lower', aspect='equal', interpolation='gaussian')
        ax.set_title('bare coarse', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[row_i, 2])
        ax.imshow(omega_clos, cmap='RdBu_r', vmin=-v, vmax=v,
                  origin='lower', aspect='equal', interpolation='gaussian')
        ax.set_title('coarse + closure', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[row_i, 3])
        ax.imshow(diff_bare, cmap='seismic', vmin=-v_diff, vmax=v_diff,
                  origin='lower', aspect='equal', interpolation='gaussian')
        rel_b = np.sqrt(np.mean(diff_bare**2)) / np.sqrt(np.mean(omega_truth**2)+1e-30)
        ax.set_title(rf'bare $-$ truth' '\n' rf'rel $L^2={rel_b:.3f}$',
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[row_i, 4])
        ax.imshow(diff_clos, cmap='seismic', vmin=-v_diff, vmax=v_diff,
                  origin='lower', aspect='equal', interpolation='gaussian')
        rel_c = np.sqrt(np.mean(diff_clos**2)) / np.sqrt(np.mean(omega_truth**2)+1e-30)
        ax.set_title(rf'closure $-$ truth' '\n' rf'rel $L^2={rel_c:.3f}$',
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    n_steps = len(rel_bare) - 1
    t_axis = np.arange(0, n_steps + 1) * Delta_T

    ax = fig.add_subplot(outer[n_snaps, 0:2])
    ax.semilogy(t_axis, rel_bare + 1e-30, 'C0-', lw=1.6, label='bare coarse')
    ax.semilogy(t_axis, rel_clos + 1e-30, 'C2-', lw=1.6, label='coarse + closure')
    ax.set_xlabel(r'physical time  $t = m\Delta T$', fontsize=10)
    ax.set_ylabel(r'$\|\omega(t)-\omega_{\Delta T}(t)\|_2 \,/\, \|\omega(t)\|_2$',
                  fontsize=10)
    ax.set_title('relative $L^2$ error vs truth', fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    ax = fig.add_subplot(outer[n_snaps, 2:4])
    ax.semilogy(t_axis, abs_bare + 1e-30, 'C0-', lw=1.6, label='bare coarse')
    ax.semilogy(t_axis, abs_clos + 1e-30, 'C2-', lw=1.6, label='coarse + closure')
    ax.set_xlabel(r'physical time  $t = m\Delta T$', fontsize=10)
    ax.set_ylabel(r'$\|\omega(t)-\omega_{\Delta T}(t)\|_2$', fontsize=10)
    ax.set_title('absolute RMS error vs truth', fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    ax = fig.add_subplot(outer[n_snaps, 4])
    improvement = rel_bare / np.maximum(rel_clos, 1e-30)
    ax.semilogy(t_axis, improvement, 'k-', lw=1.6)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xlabel(r'$t$', fontsize=10)
    ax.set_ylabel(r'rel-error ratio (bare / closure)', fontsize=9)
    ax.set_title('improvement factor', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    ax = fig.add_subplot(outer[n_snaps + 1, 0:2])
    ax.plot(t_axis, fine_rms, 'k-',  lw=1.6, label='truth')
    ax.plot(t_axis, bare_rms, 'C0-', lw=1.2, label='bare coarse')
    ax.plot(t_axis, clos_rms, 'C2-', lw=1.2, label='coarse + closure')
    ax.set_xlabel(r'$t$', fontsize=10)
    ax.set_ylabel(r'$\|\omega(t)\|_2$', fontsize=10)
    ax.set_title('rms vorticity', fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='best')

    ax = fig.add_subplot(outer[n_snaps + 1, 2:4])
    ax.plot(t_axis, fine_Z, 'k-',  lw=1.6, label='truth')
    ax.plot(t_axis, bare_Z, 'C0-', lw=1.2, label='bare coarse')
    ax.plot(t_axis, clos_Z, 'C2-', lw=1.2, label='coarse + closure')
    ax.set_xlabel(r'$t$', fontsize=10)
    ax.set_ylabel(r'$Z(t)$', fontsize=10)
    ax.set_title('mean enstrophy', fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='best')

    ax = fig.add_subplot(outer[n_snaps + 1, 4])
    rel_Z_bare = np.abs(bare_Z - fine_Z) / np.maximum(fine_Z, 1e-30)
    rel_Z_clos = np.abs(clos_Z - fine_Z) / np.maximum(fine_Z, 1e-30)
    ax.semilogy(t_axis, rel_Z_bare + 1e-30, 'C0-', lw=1.2, label='bare')
    ax.semilogy(t_axis, rel_Z_clos + 1e-30, 'C2-', lw=1.2, label='closure')
    ax.set_xlabel(r'$t$', fontsize=10)
    ax.set_ylabel(r'rel. $Z$ error', fontsize=9)
    ax.set_title('enstrophy drift', fontsize=10)
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8, loc='best')

    # Spectra
    omega_truth_0 = fine_hist[0]
    kc, E_truth_0 = energy_spectrum(omega_truth_0, Lx, Ly)
    _,  Z_truth_0_k = enstrophy_spectrum(omega_truth_0, Lx, Ly)
    omega_truth_f = fine_hist[-1]
    omega_bare_f  = bare_hist[-1]
    omega_clos_f  = clos_hist[-1]
    _,  E_truth_f = energy_spectrum(omega_truth_f, Lx, Ly)
    _,  E_bare_f  = energy_spectrum(omega_bare_f,  Lx, Ly)
    _,  E_clos_f  = energy_spectrum(omega_clos_f,  Lx, Ly)
    _,  Z_truth_f_k = enstrophy_spectrum(omega_truth_f, Lx, Ly)
    _,  Z_bare_f_k  = enstrophy_spectrum(omega_bare_f,  Lx, Ly)
    _,  Z_clos_f_k  = enstrophy_spectrum(omega_clos_f,  Lx, Ly)

    kk_full = kc[1:]
    k_mask = (kk_full >= SPECTRUM_KMIN) & (kk_full <= SPECTRUM_KMAX)
    kk = kk_full[k_mask]
    def _clip(d): return d[1:][k_mask]

    def _slope_anchor(spec, slope):
        if len(kk) <= 4 or spec.max() <= 0:
            return np.zeros_like(kk, dtype=bool), 1.0
        k_peak = float(kk[int(np.argmax(spec))])
        k_lo = max(k_peak, kk[0])
        k_hi = min(k_peak * 10.0, kk[-1] * 0.95)
        mask = (kk >= k_lo) & (kk <= k_hi) & (spec > 0)
        if not mask.any(): return mask, 1.0
        anchor = spec[mask][0] / (kk[mask][0] ** slope)
        return mask, anchor

    def _plot_spectrum(ax, curves, title, ylabel, ref_slope, ref_label):
        for label, color, style, lw, data in curves:
            ax.loglog(kk, data + 1e-30, color=color, linestyle=style,
                      lw=lw, label=label)
        truth_data = curves[0][4]
        mask, anchor = _slope_anchor(truth_data, ref_slope)
        if mask.any():
            ax.loglog(kk[mask], anchor * kk[mask] ** ref_slope,
                      'r--', lw=1, alpha=0.7, label=ref_label)
        ax.set_xlabel(r'$k$', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(SPECTRUM_KMIN, SPECTRUM_KMAX)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    ax = fig.add_subplot(outer[n_snaps + 2, 0:2])
    _plot_spectrum(ax, [('truth IC', 'k', '-', 1.6, _clip(E_truth_0))],
                    r'$E(k)$ at $t = 0$ (IC)', r'$E(k)$',
                    -5/3, r'$k^{-5/3}$ (ref)')

    ax = fig.add_subplot(outer[n_snaps + 2, 2:4])
    _plot_spectrum(ax, [('truth IC', 'k', '-', 1.6, _clip(Z_truth_0_k))],
                    r'$Z(k)$ at $t = 0$ (IC)', r'$Z(k)$',
                    -3, r'$k^{-3}$ (ref)')

    t_final = n_steps * Delta_T
    ax = fig.add_subplot(outer[n_snaps + 3, 0:2])
    _plot_spectrum(ax, [
        ('truth',   'k',  '-', 1.6, _clip(E_truth_f)),
        ('bare',    'C0', '-', 1.2, _clip(E_bare_f)),
        ('closure', 'C2', '-', 1.2, _clip(E_clos_f)),
    ], rf'$E(k)$ at $t = {t_final:.3f}$', r'$E(k)$',
       -5/3, r'$k^{-5/3}$ (ref)')

    ax = fig.add_subplot(outer[n_snaps + 3, 2:4])
    _plot_spectrum(ax, [
        ('truth',   'k',  '-', 1.6, _clip(Z_truth_f_k)),
        ('bare',    'C0', '-', 1.2, _clip(Z_bare_f_k)),
        ('closure', 'C2', '-', 1.2, _clip(Z_clos_f_k)),
    ], rf'$Z(k)$ at $t = {t_final:.3f}$', r'$Z(k)$',
       -3, r'$k^{-3}$ (ref)')

    frac_pct = int(round(perfect_nn_fraction * 100))
    fig.suptitle(rf'Perfect-closure ceiling rollout: truth ($K={K}$) vs bare '
                 rf'($\Delta T = {Delta_T:g}$) vs closure '
                 rf'[{frac_pct}\% of analytical $f_{{NN}}$ target]  --  '
                 rf'{n_steps} coarse steps',
                 fontsize=13, y=0.998)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"[rollout] wrote {out_path}")


if __name__ == '__main__':
    main()
