"""
rollout_load_truth_compare.py
=============================

Like rollout_multistep_comparison.py, but the TRUTH and BARE trajectories are
LOADED from precomputed simulations instead of regenerated on the fly.  Only
the CLOSURE trajectory (augmented AB2CN2 + ML correction) is computed here.

    truth  : loaded from a fine DNS run    (e.g. dt=1e-5 restart sim)
    bare   : loaded from a coarse DNS run  (e.g. dt=1e-3 restart sim)
    closure: COMPUTED -- AB2CN2 at Delta_T, each step augmented by
             e_anal + e_NN_pred, self-evolving from the same IC.

Why
---
Generating the K-fine truth on the fly costs M*K fine AB2CN2 steps (e.g.
1e7 steps for T=100).  Since you've already run the fine (dt=1e-5) and coarse
(dt=1e-3) sims as standalone jobs, we just read those back and spend compute
only on the cheap closure rollout (M coarse steps + M NN evals).

Assumptions
-----------
* Both reference sims were started from the SAME IC (the t=60 restart
  snapshot) and saved at the SAME physical-time cadence (e.g. every 0.1 time
  units).  Their internal clocks both start at t=0 (they don't know they are
  continuations); we compare by frame index / nearest-time.
* Both have been converted to DNS_FR_omega.npy (+ DNS_FR_times.npy) via the
  Step1 --convert-only path.  Shapes: (B, T_save, Ny, Nx) or (T_save, Ny, Nx).
* The closure rollout uses Delta_T from the dataset manifest (1e-3) and saves
  at the reference cadence so all three line up frame-for-frame.

Outputs
-------
<out-dir>/rollout_loaded_<ic_tag>.png    figure (truth/bare/closure + curves)
<out-dir>/rollout_loaded_<ic_tag>.npz    all arrays (per-frame)
<out-dir>/rollout_loaded_<ic_tag>.json   metadata

Usage
-----
  python rollout_load_truth_compare.py \
      --run-dir   <training_run_dir> \
      --root-dir  <dataset_root_dir> \
      --truth-omega /gdata/.../decaying_turb_restart_t60_dt1em5/DNS_FR_omega.npy \
      --truth-times /gdata/.../decaying_turb_restart_t60_dt1em5/DNS_FR_times.npy \
      --bare-omega  /gdata/.../decaying_turb_restart_t60_dt1em3/DNS_FR_omega.npy \
      --bare-times  /gdata/.../decaying_turb_restart_t60_dt1em3/DNS_FR_times.npy \
      --batch-index 0 \
      --device cuda \
      --ic-tag restart_t60 \
      --out-dir .
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _find_training_dir():
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        if (ancestor / 'dataset.py').exists():
            return ancestor
    return here
_training_dir = _find_training_dir()
sys.path.insert(0, str(_training_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path('.').resolve()))
print(f"[rollout] training dir on sys.path: {_training_dir}")


# --------------------------------------------------------------------------- #
# Solver primitives (identical to build_training_data_fixD_v2.py).            #
# Only needed for the CLOSURE rollout + stencil bootstrap.                    #
# --------------------------------------------------------------------------- #

def J_phys(psi_phys, omega_phys, derivative):
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


def L_op(omega_phys, L_hat):
    from qg.solver.opt.basis import to_spectral, to_physical
    return to_physical(L_hat * to_spectral(omega_phys))


def build_L_hat(derivative, nu, mu, beta):
    L_hat = nu * derivative.laplacian - mu
    if beta != 0.0:
        L_hat = L_hat - beta * derivative.dx * derivative.inv_laplacian
    return L_hat


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


def E_analytical_phys(omega_phys, derivative, L_hat, F_phys):
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


def psi_from_omega(omega_phys, derivative):
    from qg.solver.opt.basis import to_spectral, to_physical
    return to_physical(derivative.inv_laplacian * to_spectral(omega_phys))


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
    psd  = np.abs(fhat) ** 2
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
# Model + inputs                                                              #
# --------------------------------------------------------------------------- #

def load_model(run_dir, n_in, hidden, kernel, base_channels, depth, model_name,
               device='cpu'):
    if model_name in ('bilinear_closure', 'bilin', 'fixd_v2'):
        from model_fixD import build_model
        model = build_model(model_name, in_channels=n_in, hidden=hidden, kernel=kernel)
    else:
        from model import build_model
        if model_name == 'unet':
            model = build_model('unet', in_channels=n_in,
                                base_channels=base_channels, kernel=kernel)
        else:
            model = build_model('cnn', in_channels=n_in,
                                hidden_channels=hidden, depth=depth, kernel=kernel)
    ckpt_path = run_dir / 'model_best.pt'
    if not ckpt_path.exists():
        ckpt_path = run_dir / 'model_last.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no model_best.pt or model_last.pt in {run_dir}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    elif 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device).eval()
    print(f"  loaded checkpoint: {ckpt_path}")
    return model


def assemble_inputs(input_fields, omega_stack, psi_stack, derivative,
                    L_hat, F_phys, dtype, device):
    from qg.solver.opt.basis import to_spectral, to_physical
    chans = []
    def grab_omega(k):
        return omega_stack[k] if k < len(omega_stack) else torch.zeros_like(omega_stack[0])
    def grab_psi(k):
        return psi_stack[k] if k < len(psi_stack) else torch.zeros_like(psi_stack[0])
    def grad_psi_sq_phys():
        psih = to_spectral(psi_stack[0])
        u = to_physical(-1 * derivative.dy * psih)
        v = to_physical(+1 * derivative.dx * psih)
        return u * u + v * v
    def omega_grad(axis):
        qh = to_spectral(omega_stack[0])
        return to_physical((derivative.dx if axis == 'x' else derivative.dy) * qh)
    def N_now():
        N = -1.0 * J_phys(psi_stack[0], omega_stack[0], derivative)
        if F_phys is not None:
            N = N + F_phys
        return N
    def L_omega_phys(power):
        qh = to_spectral(omega_stack[0])
        return to_physical((L_hat ** power) * qh)
    def L_N_phys(power):
        Nh = to_spectral(N_now())
        return to_physical((L_hat ** power) * Nh)
    builders = {
        'omega_0':  lambda: grab_omega(0), 'omega_m1': lambda: grab_omega(1),
        'omega_m2': lambda: grab_omega(2), 'psi_0': lambda: grab_psi(0),
        'psi_m1': lambda: grab_psi(1), 'psi_m2': lambda: grab_psi(2),
        'grad_psi_sq': grad_psi_sq_phys,
        'omega_x': lambda: omega_grad('x'), 'omega_y': lambda: omega_grad('y'),
        'N_0': N_now,
        'L_omega': lambda: L_omega_phys(1), 'L2_omega': lambda: L_omega_phys(2),
        'L3_omega': lambda: L_omega_phys(3),
        'L_N': lambda: L_N_phys(1), 'L2_N': lambda: L_N_phys(2),
    }
    for field in input_fields:
        if field not in builders:
            raise ValueError(f"input field '{field}' not supported in rollout")
        chans.append(builders[field]())
    x = torch.stack([c[0] for c in chans], dim=0)[None]
    return x.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Reference-trajectory loader                                                 #
# --------------------------------------------------------------------------- #

def load_reference(omega_path, times_path, batch_index, label):
    """Load a precomputed trajectory's frames + times for ONE batch.

    Returns (frames float32 (T,Ny,Nx), times (T,)).
    """
    omega = np.load(omega_path, mmap_mode='r')
    print(f"[rollout] {label} omega: shape={omega.shape} dtype={omega.dtype}")
    if omega.ndim == 4:
        if batch_index >= omega.shape[0]:
            sys.exit(f"ERROR: {label} batch_index {batch_index} >= {omega.shape[0]}")
        traj = np.asarray(omega[batch_index]).astype(np.float32)
    elif omega.ndim == 3:
        traj = np.asarray(omega).astype(np.float32)
    else:
        sys.exit(f"ERROR: {label} omega ndim={omega.ndim}, expected 3 or 4")
    times = np.load(times_path)
    if times.ndim == 2:
        times = times[batch_index]
    if len(times) != traj.shape[0]:
        # Some converters store one fewer time than frames; trim.
        n = min(len(times), traj.shape[0])
        print(f"[rollout] {label} time/frame length mismatch "
              f"({len(times)} vs {traj.shape[0]}); trimming to {n}")
        times = times[:n]; traj = traj[:n]
    print(f"[rollout] {label}: T={traj.shape[0]} frames, "
          f"t[0]={times[0]:.4f}, t[-1]={times[-1]:.4f}, "
          f"dt_save={times[1]-times[0]:.4f}")
    return traj, times


def nearest_indices(target_times, source_times):
    """For each t in target_times, the index of the nearest source time."""
    src = np.asarray(source_times)
    return np.array([int(np.argmin(np.abs(src - t))) for t in target_times],
                    dtype=np.int64)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir',  type=Path, required=True)
    p.add_argument('--root-dir', type=Path, required=True)
    p.add_argument('--truth-omega', type=Path, required=True)
    p.add_argument('--truth-times', type=Path, required=True)
    p.add_argument('--bare-omega',  type=Path, required=True)
    p.add_argument('--bare-times',  type=Path, required=True)
    p.add_argument('--batch-index', type=int, default=0,
                   help='which batch of the reference sims to use (default 0)')
    p.add_argument('--max-frames', type=int, default=-1,
                   help='cap on number of reference frames to roll through '
                        '(-1 = all). Useful for quick tests.')
    p.add_argument('--snapshot-fracs', type=str, default='0,0.25,0.5,0.75,1.0',
                   help='fractions of the rollout at which to render fields')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--ic-tag', type=str, default='loaded')
    args = p.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    dtype = torch.float64
    out_dir = args.out_dir or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ic_tag = args.ic_tag
    print(f"[rollout] device={device}  state dtype=float64")

    # ----- Manifest (grid, physics, Delta_T, K) ----- #
    with open(args.root_dir / 'manifest.json') as f:
        manifest = json.load(f)
    Nx, Ny = int(manifest['Nx']), int(manifest['Ny'])
    Lx, Ly = float(manifest['Lx']), float(manifest['Ly'])
    nu     = float(manifest['nu'])
    mu     = float(manifest.get('mu', 0.0))
    beta   = float(manifest.get('beta', 0.0))
    Delta_T = float(manifest['Delta_T'])
    K       = int(manifest.get('K', 100))
    h_fine  = float(manifest.get('h_fine', Delta_T / K))
    print(f"[rollout] manifest: Nx={Nx} Ny={Ny} Lx={Lx:.4f} Ly={Ly:.4f}")
    print(f"[rollout]           nu={nu} mu={mu} beta={beta}")
    print(f"[rollout]           Delta_T={Delta_T} K={K} h_fine={h_fine}")

    # ----- Training config ----- #
    cfg = json.loads((args.run_dir / 'config.json').read_text())
    input_fields = tuple(cfg.get('input_fields',
                                 ['omega_0', 'omega_m1', 'omega_m2',
                                  'psi_0', 'psi_m1', 'psi_m2']))
    target_field = cfg.get('target_field', 'f_NN_target')
    model_name   = cfg.get('model', 'bilinear_closure')
    normalize    = cfg.get('normalize', True)
    print(f"[rollout] inputs: {input_fields}")
    print(f"[rollout] target: {target_field}  model: {model_name}  normalize={normalize}")

    # ----- Operators (float64) ----- #
    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision='float64')
    derivative = Derivative(grid)
    for attr in ('dx', 'dy', 'laplacian', 'inv_laplacian'):
        setattr(derivative, attr, getattr(derivative, attr).to(device))
    L_hat = build_L_hat(derivative, nu=nu, mu=mu, beta=beta).to(device)
    F_phys = None
    print(f"[rollout] operators built (L_hat dtype {L_hat.dtype})")

    # ----- Normalization stats ----- #
    from dataset import ClosureDataset
    ds = ClosureDataset(root_dir=args.root_dir, split='val',
                         input_fields=input_fields,
                         target_field=target_field, normalize=normalize)
    if normalize:
        target_mean = torch.as_tensor(ds.target_mean, dtype=dtype, device=device)
        target_std  = torch.as_tensor(ds.target_std,  dtype=dtype, device=device)
        input_mean = ds.input_mean.to(dtype=dtype, device=device).reshape(1, -1, 1, 1)
        input_std  = ds.input_std .to(dtype=dtype, device=device).reshape(1, -1, 1, 1)
        print(f"[rollout] target norm: mean={float(target_mean):.4e} std={float(target_std):.4e}")
    else:
        target_mean = target_std = input_mean = input_std = None

    def normalize_inputs(x):
        return (x - input_mean) / input_std if normalize else x
    def denormalize_target(yhat):
        return yhat * target_std + target_mean if normalize else yhat

    model = load_model(args.run_dir, n_in=len(input_fields),
                       hidden=cfg.get('hidden_channels', 64),
                       kernel=cfg.get('kernel', 3),
                       base_channels=cfg.get('base_channels', 32),
                       depth=cfg.get('depth', 6),
                       model_name=model_name, device=device)
    model_dtype = next(model.parameters()).dtype
    print(f"[rollout] model dtype: {model_dtype}")

    # ----- Load truth + bare references ----- #
    truth_frames, truth_times = load_reference(
        args.truth_omega, args.truth_times, args.batch_index, 'truth(dt_fine)')
    bare_frames, bare_times = load_reference(
        args.bare_omega, args.bare_times, args.batch_index, 'bare(dt_coarse)')

    if truth_frames.shape[1:] != (Ny, Nx):
        sys.exit(f"ERROR: truth frame shape {truth_frames.shape[1:]} != ({Ny},{Nx})")
    if bare_frames.shape[1:] != (Ny, Nx):
        sys.exit(f"ERROR: bare frame shape {bare_frames.shape[1:]} != ({Ny},{Nx})")

    # ----- Determine the comparison time grid (use truth's saved times) ----- #
    n_frames = truth_frames.shape[0]
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    truth_times = truth_times[:n_frames]
    truth_frames = truth_frames[:n_frames]

    # Coarse steps per saved frame
    dt_save = float(truth_times[1] - truth_times[0])
    frames_per_save = int(round(dt_save / Delta_T))
    if frames_per_save < 1:
        sys.exit(f"ERROR: dt_save={dt_save} < Delta_T={Delta_T}; cannot subdivide")
    n_steps = (n_frames - 1) * frames_per_save
    print(f"[rollout] reference cadence dt_save={dt_save:.4f} = {frames_per_save} "
          f"coarse steps of Delta_T={Delta_T}")
    print(f"[rollout] closure rollout: {n_steps} coarse steps -> {n_frames} saved frames "
          f"(physical span {truth_times[0]:.3f} -> {truth_times[-1]:.3f})")

    # Map each truth frame to the nearest bare frame (in case cadences differ)
    bare_idx_for = nearest_indices(truth_times, bare_times)

    # ----- IC + closure stencil ----- #
    # IC = truth frame 0 (the t=60 restart snapshot).  Bootstrap the AB2
    # stencil (omega_0, omega_m1, omega_m2) at Delta_T spacing via RK4 backward
    # from the IC.  This is the only "truth-like" compute we do, and it's tiny
    # (2 backward RK4 steps of size Delta_T).
    omega_0_np = truth_frames[0].astype(np.float64)
    omega_0_t = torch.tensor(omega_0_np, dtype=dtype, device=device)[None]
    print(f"[rollout] IC = truth frame 0, |omega_0|_rms="
          f"{float(torch.sqrt((omega_0_t**2).mean())):.4e}")
    omega_m1_t = rk4_step(omega_0_t,  -Delta_T, derivative, L_hat, F_phys)
    omega_m2_t = rk4_step(omega_m1_t, -Delta_T, derivative, L_hat, F_phys)
    psi_0_t  = psi_from_omega(omega_0_t,  derivative)
    psi_m1_t = psi_from_omega(omega_m1_t, derivative)
    psi_m2_t = psi_from_omega(omega_m2_t, derivative)
    clos_omega = [omega_0_t, omega_m1_t, omega_m2_t]
    clos_psi   = [psi_0_t,   psi_m1_t,   psi_m2_t]

    K_factor = 1.0 - 1.0 / (K ** 2)
    coef = (Delta_T ** 3) * K_factor

    # ----- Snapshot (figure) steps ----- #
    frac_list = [float(s) for s in args.snapshot_fracs.split(',')]
    snap_frames = sorted({int(round(f * (n_frames - 1))) for f in frac_list})
    print(f"[rollout] figure snapshot frames: {snap_frames}")

    # ----- Per-frame scalar arrays ----- #
    rel_bare = np.zeros(n_frames); abs_bare = np.zeros(n_frames)
    rel_clos = np.zeros(n_frames); abs_clos = np.zeros(n_frames)
    fine_rms = np.zeros(n_frames); bare_rms = np.zeros(n_frames); clos_rms = np.zeros(n_frames)
    fine_Z   = np.zeros(n_frames); bare_Z   = np.zeros(n_frames); clos_Z   = np.zeros(n_frames)
    clos_frames_store = np.zeros((n_frames, Ny, Nx), dtype=np.float32)

    from qg.solver.opt.basis import to_spectral, to_physical

    def record_frame(fi, om_c_np):
        om_t_np = truth_frames[fi].astype(np.float64)
        om_b_np = bare_frames[bare_idx_for[fi]].astype(np.float64)
        clos_frames_store[fi] = om_c_np.astype(np.float32)
        t_norm = float(np.sqrt(np.mean(om_t_np ** 2)))
        fine_rms[fi] = t_norm
        bare_rms[fi] = float(np.sqrt(np.mean(om_b_np ** 2)))
        clos_rms[fi] = float(np.sqrt(np.mean(om_c_np ** 2)))
        fine_Z[fi] = 0.5 * float(np.mean(om_t_np ** 2))
        bare_Z[fi] = 0.5 * float(np.mean(om_b_np ** 2))
        clos_Z[fi] = 0.5 * float(np.mean(om_c_np ** 2))
        db = om_b_np - om_t_np
        dc = om_c_np - om_t_np
        abs_bare[fi] = float(np.sqrt(np.mean(db ** 2)))
        abs_clos[fi] = float(np.sqrt(np.mean(dc ** 2)))
        den = max(t_norm, 1e-30)
        rel_bare[fi] = abs_bare[fi] / den
        rel_clos[fi] = abs_clos[fi] / den

    # frame 0 = IC for all three
    record_frame(0, omega_0_np)

    # ----- Closure rollout ----- #
    print(f"\n[rollout] === closure rollout ({n_steps} coarse steps) ===")
    t0 = time.time()
    next_save_frame = 1
    for step in range(1, n_steps + 1):
        qh_c_curr  = to_spectral(clos_omega[0])
        qh_c_minus = to_spectral(clos_omega[1])
        qh_c_bare  = ab2cn2_step_spectral(qh_c_curr, qh_c_minus, Delta_T,
                                           derivative, L_hat, F_phys)
        om_c_bare = to_physical(qh_c_bare)
        E_anal = E_analytical_phys(clos_omega[0], derivative, L_hat, F_phys)
        e_anal_inc = -coef * E_anal
        x = assemble_inputs(input_fields, clos_omega, clos_psi,
                            derivative, L_hat, F_phys, dtype, device)
        x_norm = normalize_inputs(x)
        with torch.no_grad():
            yhat = model(x_norm.to(dtype=model_dtype))
        yhat = yhat.to(dtype=dtype)
        f_NN_pred = denormalize_target(yhat)[0, 0][None]
        e_NN_inc = -coef * f_NN_pred
        om_c = om_c_bare + e_anal_inc + e_NN_inc
        clos_omega = [om_c, clos_omega[0], clos_omega[1]]
        clos_psi   = [psi_from_omega(om_c, derivative), clos_psi[0], clos_psi[1]]

        if step % frames_per_save == 0:
            fi = step // frames_per_save
            if fi < n_frames:
                record_frame(fi, om_c[0].cpu().numpy())
            if step % max(1, n_steps // 20) == 0:
                el = time.time() - t0
                eta = el * (n_steps - step) / max(step, 1)
                print(f"      step {step:7d}/{n_steps}  frame {fi:5d}/{n_frames-1}  "
                      f"rel_bare={rel_bare[fi]:.3e}  rel_clos={rel_clos[fi]:.3e}  "
                      f"ratio={rel_clos[fi]/max(rel_bare[fi],1e-30):.3f}  "
                      f"elapsed={el:.1f}s eta={eta:.1f}s")

    print(f"\n[rollout] DONE in {(time.time()-t0)/60:.1f} min")
    print(f"[rollout] final rel-L2: bare={rel_bare[-1]:.4e} closure={rel_clos[-1]:.4e} "
          f"improvement={rel_bare[-1]/max(rel_clos[-1],1e-30):.2f}x")

    # ----- Save npz ----- #
    npz_out = out_dir / f'rollout_loaded_{ic_tag}.npz'
    np.savez(npz_out,
             clos_hist=clos_frames_store,
             truth_times=truth_times,
             rel_bare=rel_bare, abs_bare=abs_bare,
             rel_clos=rel_clos, abs_clos=abs_clos,
             fine_rms=fine_rms, bare_rms=bare_rms, clos_rms=clos_rms,
             fine_Z=fine_Z, bare_Z=bare_Z, clos_Z=clos_Z,
             snap_frames=np.asarray(snap_frames, dtype=np.int64),
             Delta_T=Delta_T, K=K, h_fine=h_fine, Lx=Lx, Ly=Ly,
             nu=nu, mu=mu, beta=beta,
             truth_omega=str(args.truth_omega), bare_omega=str(args.bare_omega))
    print(f"[rollout] wrote {npz_out}")

    # ----- Figure ----- #
    fig_path = out_dir / f'rollout_loaded_{ic_tag}.png'
    render_figure(truth_frames, bare_frames, bare_idx_for, clos_frames_store,
                   truth_times, rel_bare, rel_clos, abs_bare, abs_clos,
                   fine_rms, bare_rms, clos_rms, fine_Z, bare_Z, clos_Z,
                   snap_frames, Delta_T, K, Lx, Ly, model_name, fig_path)

    # ----- Metadata ----- #
    json_out = out_dir / f'rollout_loaded_{ic_tag}.json'
    with open(json_out, 'w') as fj:
        json.dump(dict(
            run_dir=str(args.run_dir), root_dir=str(args.root_dir),
            truth_omega=str(args.truth_omega), bare_omega=str(args.bare_omega),
            batch_index=int(args.batch_index),
            n_frames=int(n_frames), n_steps=int(n_steps),
            frames_per_save=int(frames_per_save), dt_save=float(dt_save),
            Delta_T=Delta_T, K=K, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
            nu=nu, mu=mu, beta=beta, model_name=model_name,
            final_rel_err_bare=float(rel_bare[-1]),
            final_rel_err_closure=float(rel_clos[-1]),
            error_reduction_factor=float(rel_bare[-1] / max(rel_clos[-1], 1e-30)),
        ), fj, indent=2)
    print(f"[rollout] wrote {json_out}")
    print(f"[rollout] DONE.")


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #

def render_figure(truth_frames, bare_frames, bare_idx_for, clos_frames,
                   truth_times, rel_bare, rel_clos, abs_bare, abs_clos,
                   fine_rms, bare_rms, clos_rms, fine_Z, bare_Z, clos_Z,
                   snap_frames, Delta_T, K, Lx, Ly, model_name, out_path):
    n_snaps = len(snap_frames)
    n_cols = 5
    n_bottom = 4
    fig_h = 3.2 * n_snaps + n_bottom * 3.0
    fig = plt.figure(figsize=(n_cols * 3.0, fig_h))
    outer = gridspec.GridSpec(n_snaps + n_bottom, n_cols, figure=fig,
                               hspace=0.55, wspace=0.30)

    for row_i, fi in enumerate(snap_frames):
        omega_truth = truth_frames[fi].astype(np.float64)
        omega_bare  = bare_frames[bare_idx_for[fi]].astype(np.float64)
        omega_clos  = clos_frames[fi].astype(np.float64)
        v = float(np.max(np.abs(np.stack([omega_truth, omega_bare, omega_clos]))))
        if v == 0: v = 1.0
        diff_bare = omega_bare - omega_truth
        diff_clos = omega_clos - omega_truth
        v_diff = float(np.max(np.abs(np.concatenate([diff_bare.ravel(), diff_clos.ravel()]))))
        if v_diff == 0: v_diff = 1.0
        t_phys = float(truth_times[fi])

        ax = fig.add_subplot(outer[row_i, 0])
        ax.imshow(omega_truth, cmap='RdBu_r', vmin=-v, vmax=v, origin='lower',
                  aspect='equal', interpolation='gaussian')
        ax.set_title(rf'truth  $t={t_phys:.2f}$' '\n' rf'(frame {fi})', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[row_i, 1])
        ax.imshow(omega_bare, cmap='RdBu_r', vmin=-v, vmax=v, origin='lower',
                  aspect='equal', interpolation='gaussian')
        ax.set_title('bare (loaded)', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[row_i, 2])
        ax.imshow(omega_clos, cmap='RdBu_r', vmin=-v, vmax=v, origin='lower',
                  aspect='equal', interpolation='gaussian')
        ax.set_title('coarse + closure', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[row_i, 3])
        ax.imshow(diff_bare, cmap='seismic', vmin=-v_diff, vmax=v_diff, origin='lower',
                  aspect='equal', interpolation='gaussian')
        rb = np.sqrt(np.mean(diff_bare**2)) / np.sqrt(np.mean(omega_truth**2)+1e-30)
        ax.set_title(rf'bare $-$ truth' '\n' rf'rel $L^2={rb:.3f}$', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(outer[row_i, 4])
        ax.imshow(diff_clos, cmap='seismic', vmin=-v_diff, vmax=v_diff, origin='lower',
                  aspect='equal', interpolation='gaussian')
        rc = np.sqrt(np.mean(diff_clos**2)) / np.sqrt(np.mean(omega_truth**2)+1e-30)
        ax.set_title(rf'closure $-$ truth' '\n' rf'rel $L^2={rc:.3f}$', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    t_axis = truth_times

    ax = fig.add_subplot(outer[n_snaps, 0:2])
    ax.semilogy(t_axis, rel_bare + 1e-30, 'C0-', lw=1.6, label='bare (loaded)')
    ax.semilogy(t_axis, rel_clos + 1e-30, 'C2-', lw=1.6, label='coarse + closure')
    ax.set_xlabel(r'physical time $t$', fontsize=10)
    ax.set_ylabel(r'rel. $L^2$ error vs truth', fontsize=10)
    ax.set_title('relative error growth', fontsize=11)
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)

    ax = fig.add_subplot(outer[n_snaps, 2:4])
    ax.semilogy(t_axis, abs_bare + 1e-30, 'C0-', lw=1.6, label='bare (loaded)')
    ax.semilogy(t_axis, abs_clos + 1e-30, 'C2-', lw=1.6, label='coarse + closure')
    ax.set_xlabel(r'physical time $t$', fontsize=10)
    ax.set_ylabel(r'$\|\omega(t)-\omega_{\mathrm{ref}}(t)\|$', fontsize=10)
    ax.set_title('absolute RMS error growth', fontsize=11)
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)

    ax = fig.add_subplot(outer[n_snaps, 4])
    improvement = rel_bare / np.maximum(rel_clos, 1e-30)
    ax.semilogy(t_axis, improvement, 'k-', lw=1.6)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xlabel(r'$t$', fontsize=10)
    ax.set_ylabel('rel-error ratio (bare/closure)', fontsize=9)
    ax.set_title('improvement factor', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    ax = fig.add_subplot(outer[n_snaps + 1, 0:2])
    ax.plot(t_axis, fine_rms, 'k-', lw=1.6, label='truth')
    ax.plot(t_axis, bare_rms, 'C0-', lw=1.2, label='bare')
    ax.plot(t_axis, clos_rms, 'C2-', lw=1.2, label='closure')
    ax.set_xlabel(r'$t$', fontsize=10); ax.set_ylabel(r'$\|\omega(t)\|_2$', fontsize=10)
    ax.set_title('rms vorticity', fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = fig.add_subplot(outer[n_snaps + 1, 2:4])
    ax.plot(t_axis, fine_Z, 'k-', lw=1.6, label='truth')
    ax.plot(t_axis, bare_Z, 'C0-', lw=1.2, label='bare')
    ax.plot(t_axis, clos_Z, 'C2-', lw=1.2, label='closure')
    ax.set_xlabel(r'$t$', fontsize=10); ax.set_ylabel(r'$Z(t)$', fontsize=10)
    ax.set_title('mean enstrophy', fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = fig.add_subplot(outer[n_snaps + 1, 4])
    rZb = np.abs(bare_Z - fine_Z) / np.maximum(fine_Z, 1e-30)
    rZc = np.abs(clos_Z - fine_Z) / np.maximum(fine_Z, 1e-30)
    ax.semilogy(t_axis, rZb + 1e-30, 'C0-', lw=1.2, label='bare')
    ax.semilogy(t_axis, rZc + 1e-30, 'C2-', lw=1.2, label='closure')
    ax.set_xlabel(r'$t$', fontsize=10); ax.set_ylabel('rel. $Z$ error', fontsize=9)
    ax.set_title('enstrophy drift', fontsize=10)
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8)

    # spectra at t0 and t_final
    def _spec_row(row, fi, title_t):
        ot = truth_frames[fi].astype(np.float64)
        ob = bare_frames[bare_idx_for[fi]].astype(np.float64)
        oc = clos_frames[fi].astype(np.float64)
        kc, E_t = energy_spectrum(ot, Lx, Ly)
        _,  E_b = energy_spectrum(ob, Lx, Ly)
        _,  E_c = energy_spectrum(oc, Lx, Ly)
        _,  Z_t = enstrophy_spectrum(ot, Lx, Ly)
        _,  Z_b = enstrophy_spectrum(ob, Lx, Ly)
        _,  Z_c = enstrophy_spectrum(oc, Lx, Ly)
        kk = kc[1:]
        axE = fig.add_subplot(outer[row, 0:2])
        axE.loglog(kk, E_t[1:]+1e-30, 'k-', lw=1.6, label='truth')
        axE.loglog(kk, E_b[1:]+1e-30, 'C0-', lw=1.2, label='bare')
        axE.loglog(kk, E_c[1:]+1e-30, 'C2-', lw=1.2, label='closure')
        axE.set_xlabel(r'$k$'); axE.set_ylabel(r'$E(k)$')
        axE.set_title(rf'$E(k)$ at $t={title_t:.2f}$', fontsize=11)
        axE.grid(True, which='both', alpha=0.3); axE.legend(fontsize=8)
        axZ = fig.add_subplot(outer[row, 2:4])
        axZ.loglog(kk, Z_t[1:]+1e-30, 'k-', lw=1.6, label='truth')
        axZ.loglog(kk, Z_b[1:]+1e-30, 'C0-', lw=1.2, label='bare')
        axZ.loglog(kk, Z_c[1:]+1e-30, 'C2-', lw=1.2, label='closure')
        axZ.set_xlabel(r'$k$'); axZ.set_ylabel(r'$Z(k)$')
        axZ.set_title(rf'$Z(k)$ at $t={title_t:.2f}$', fontsize=11)
        axZ.grid(True, which='both', alpha=0.3); axZ.legend(fontsize=8)

    _spec_row(n_snaps + 2, 0, float(truth_times[0]))
    _spec_row(n_snaps + 3, len(truth_times) - 1, float(truth_times[-1]))

    fig.suptitle(rf'Loaded-reference rollout: truth (dt fine) vs bare (dt coarse, loaded) '
                 rf'vs coarse$+$closure ({model_name})  --  '
                 rf'{len(truth_times)} frames', fontsize=13, y=0.998)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"[rollout] wrote figure: {out_path}")


if __name__ == '__main__':
    main()
