"""
benchmark_walltime.py -- measure wall-clock time of the QG AB2-CN2 solver
as a function of dt.

Sweeps dt in {1e-5, 5e-5, 1e-4, 5e-4, 1e-3} integrating to a fixed final time
T = 1.0, on the same grid / nu / IC as the production decaying-turbulence run.

Outputs a log-log plot of wall time vs dt. Expected slope = -1 (cost
inversely proportional to dt -> total #steps).

Run:
    cd $QG_DIR
    source $QG_ROOT/qg-env/bin/activate
    python benchmark_walltime.py --device cuda --T 1.0 --out walltime.png

Then drop walltime.png into slide 27 of the deck.
"""
import argparse
import time
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_initial(Nx, Ny, Lx, Ly, peak_k=(3, 5), energy=0.1, seed=0, device='cpu', dtype=torch.float64):
    """Random Gaussian IC band-passed at peak wavenumbers, rescaled to target energy."""
    rng = np.random.default_rng(seed)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    kxg, kyg = np.meshgrid(kx, ky, indexing='xy')
    k = np.sqrt(kxg ** 2 + kyg ** 2)
    # Band-pass mask
    mask = ((k >= peak_k[0]) & (k <= peak_k[1])).astype(np.float64)
    # Random complex coefficients
    a = rng.standard_normal((Ny, Nx)) + 1j * rng.standard_normal((Ny, Nx))
    a *= mask
    omega = np.real(np.fft.ifft2(a))
    # Rescale to target energy E = (1/2) <|u|^2> = (1/2) <|grad psi|^2> approx
    # easier: just rescale omega to match a target rms.
    omega *= np.sqrt(2 * energy) / (np.sqrt(np.mean(omega ** 2)) + 1e-30)
    return torch.from_numpy(omega).to(device=device, dtype=dtype)[None]  # (1, Ny, Nx)


def run_one(dt, T, Nx, Ny, Lx, Ly, nu, device='cpu', dtype=torch.float64,
            warmup_steps=5):
    """Run AB2-CN2 from random IC for T/dt steps, time only the inner loop."""
    # Defer import to here so we get the actual codebase
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral, to_physical
    from qg.solver.grid.cartesian import CartesianGrid

    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision='float64')
    derivative = Derivative(grid).to(device)
    # L_hat for QG diffusion in spectral space.
    # derivative.laplacian already encodes -(kx^2 + ky^2), so the diffusive
    # linear operator is L = nu * laplacian (a negative-definite multiplier).
    L_hat = nu * derivative.laplacian  # (Ny, Nx) for full FFT, shape matches qh
    # Precompute CN2 multipliers
    cn_plus  = 1.0 + 0.5 * dt * L_hat
    cn_minus = 1.0 - 0.5 * dt * L_hat
    cn_inv   = 1.0 / cn_minus

    omega = make_initial(Nx, Ny, Lx, Ly, device=device, dtype=dtype)

    # Need previous N for AB2 - bootstrap with a single Forward Euler step
    qh = to_spectral(omega)
    psi_hat = derivative.inv_laplacian * qh
    u = to_physical(-1 * derivative.dy * psi_hat)
    v = to_physical(+1 * derivative.dx * psi_hat)
    q = to_physical(qh)
    uq_h = to_spectral(u * q)
    vq_h = to_spectral(v * q)
    Nh_prev = -(derivative.dx * uq_h + derivative.dy * vq_h)  # spectral

    n_steps = int(round(T / dt))

    # Warmup (compile, allocate, etc) - not timed
    for _ in range(warmup_steps):
        qh = to_spectral(omega)
        psi_hat = derivative.inv_laplacian * qh
        u = to_physical(-1 * derivative.dy * psi_hat)
        v = to_physical(+1 * derivative.dx * psi_hat)
        q = to_physical(qh)
        uq_h = to_spectral(u * q)
        vq_h = to_spectral(v * q)
        Nh = -(derivative.dx * uq_h + derivative.dy * vq_h)
        rhs = cn_plus * qh + dt * (1.5 * Nh - 0.5 * Nh_prev)
        qh_new = cn_inv * rhs
        omega = to_physical(qh_new)
        Nh_prev = Nh

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(n_steps):
        qh = to_spectral(omega)
        psi_hat = derivative.inv_laplacian * qh
        u = to_physical(-1 * derivative.dy * psi_hat)
        v = to_physical(+1 * derivative.dx * psi_hat)
        q = to_physical(qh)
        uq_h = to_spectral(u * q)
        vq_h = to_spectral(v * q)
        Nh = -(derivative.dx * uq_h + derivative.dy * vq_h)
        rhs = cn_plus * qh + dt * (1.5 * Nh - 0.5 * Nh_prev)
        qh_new = cn_inv * rhs
        omega = to_physical(qh_new)
        Nh_prev = Nh

    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return {
        'dt': dt,
        'T': T,
        'n_steps': n_steps,
        'wall_time_s': t1 - t0,
        'time_per_step_ms': 1000 * (t1 - t0) / n_steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--T', type=float, default=1.0,
                   help='final integration time')
    p.add_argument('--Nx', type=int, default=256)
    p.add_argument('--Ny', type=int, default=256)
    p.add_argument('--Lx', type=float, default=2 * np.pi)
    p.add_argument('--Ly', type=float, default=2 * np.pi)
    p.add_argument('--nu', type=float, default=1.025e-5)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', type=str, default='walltime.png')
    p.add_argument('--out-json', type=str, default='walltime.json')
    args = p.parse_args()

    dtype = torch.float64
    dts = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]

    print(f"benchmark: T={args.T}, grid {args.Nx}x{args.Ny}, nu={args.nu}, "
          f"device={args.device}")
    results = []
    for dt in dts:
        n_steps = int(round(args.T / dt))
        # Skip if it'd take forever
        if n_steps > 200000:
            print(f"  dt={dt:.1e}: skipping ({n_steps} steps, too long)")
            continue
        print(f"  dt={dt:.1e}: {n_steps} steps...", flush=True)
        r = run_one(dt, args.T, args.Nx, args.Ny, args.Lx, args.Ly, args.nu,
                    device=args.device, dtype=dtype)
        print(f"      wall={r['wall_time_s']:.2f}s  per-step={r['time_per_step_ms']:.3f}ms")
        results.append(r)

    # Save JSON
    with open(args.out_json, 'w') as f:
        json.dump({'config': vars(args), 'results': results}, f, indent=2)
    print(f"saved {args.out_json}")

    # Plot
    dt_arr   = np.array([r['dt'] for r in results])
    wall_arr = np.array([r['wall_time_s'] for r in results])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.loglog(dt_arr, wall_arr, 'ko-', ms=8, lw=1.5, label='measured')
    # Reference slope -1: cost ~ 1/dt
    anchor_idx = np.argmax(dt_arr)
    ref_slope = wall_arr[anchor_idx] * (dt_arr[anchor_idx] / dt_arr)
    ax.loglog(dt_arr, ref_slope, 'b--', lw=1.0,
              label=r'reference: cost $\propto 1/\Delta t$')
    ax.set_xlabel(r'$\Delta t$', fontsize=13)
    ax.set_ylabel(f'Wall time to integrate to T = {args.T:g}  (s)', fontsize=13)
    ax.set_title(f'AB2-CN2 wall time vs $\\Delta t$, Case: Decaying Turbulence  '
                 f'({args.Nx}\u00d7{args.Ny}, $\\nu={args.nu:.2e}$, on {args.device})',
                 fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")

    # Print speedup table
    print("\nSpeedup table (relative to dt = {:.0e}):".format(dts[-1]))
    base = wall_arr[-1]
    for r in results:
        ratio = r['wall_time_s'] / base
        print(f"  dt={r['dt']:.1e}:  {r['wall_time_s']:8.2f}s  ({ratio:6.1f}x of dt={dts[-1]:.0e})")


if __name__ == '__main__':
    main()
