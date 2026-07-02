"""
convergence_plot_bulk.py  (DECAYING TURBULENCE -- BULK QUANTITIES)

Bulk-quantity temporal convergence plot for the decaying-turbulence dt sweep,
designed to be CHAOS-INSENSITIVE.

Why this differs from convergence_plot.py:
  The strong-norm test ||q_dt - q_ref||_L2 saturates at O(1) in 2D turbulence
  because chaotic divergence amplifies tiny truncation errors as exp(lambda_1 t).
  At T_sim = 20, that exponential factor has blown up the strong-norm error
  for every dt in the sweep -> flat slopes.

  Bulk integrals (E, Z, shell-averaged E(k)) are INVARIANT under spatial
  shuffling of vortices, so they're blind to chaotic phase-shift errors and
  see only truncation-driven amplitude/dissipation differences. They should
  show clean slope-2 convergence even at long T_sim.

Three panels (single PNG):
  (a) |E_dt(t*) - E_ref(t*)|       vs Delta t
  (b) |Z_dt(t*) - Z_ref(t*)|       vs Delta t
  (c) ||E_dt(k,t*) - E_ref(k,t*)|| vs Delta t   (ell^2 over k)

All three quantities are evaluated at a single fixed physical time t*
(default: the final saved time, same as convergence_plot.py).

Usage:
    python convergence_plot_bulk.py \
        --sweep-root /path/to/decaying_turb_dt_sweep_v2 \
        --out-dir   /path/to/figures
        [--ref-subdir dt_1em5]
        [--Lx 6.2832]  [--Ly 6.2832]
        [--t-star -1]    # -1 = use final saved time
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SWEEP: List[Tuple[float, str, str]] = [
    (2.0e-3,   'dt_2em3',     r'$\Delta t = 2\times 10^{-3}$'),
    (1.0e-3,   'dt_1em3',     r'$\Delta t = 10^{-3}$'),
    (5.0e-4,   'dt_5em4',     r'$\Delta t = 5\times 10^{-4}$'),
    (2.5e-4,   'dt_2p5em4',   r'$\Delta t = 2.5\times 10^{-4}$'),
    (1.25e-4,  'dt_1p25em4',  r'$\Delta t = 1.25\times 10^{-4}$'),
    (2.0e-5,   'dt_2em5',     r'$\Delta t = 2\times 10^{-5}$'),
    (1.0e-5,   'dt_1em5',     r'$\Delta t = 10^{-5}$ (ref)'),
]
REF_SUBDIR = 'dt_1em5'


# --------------------------------------------------------------------------- #
# .npy reader                                                                 #
# --------------------------------------------------------------------------- #

class _NpyReader:
    """Memory-safe streamer for an .npy file of shape (T, Ny, Nx)."""

    def __init__(self, path: Path):
        from numpy.lib import format as np_format
        self.path = Path(path)
        with open(self.path, 'rb') as f:
            major, minor = np_format.read_magic(f)
            if (major, minor) == (1, 0):
                shape, fortran_order, dtype = np_format.read_array_header_1_0(f)
            elif (major, minor) == (2, 0):
                shape, fortran_order, dtype = np_format.read_array_header_2_0(f)
            else:
                raise RuntimeError(f"unsupported .npy version {major}.{minor}")
            self.header_offset = f.tell()
        if fortran_order:
            raise RuntimeError("Fortran-order .npy not supported here")
        if len(shape) != 3:
            raise RuntimeError(f"expected 3D .npy (T, Ny, Nx); got shape={shape}")
        self.shape = shape
        self.dtype = dtype
        self.T, self.Ny, self.Nx = shape
        self._snap_bytes = self.Ny * self.Nx * self.dtype.itemsize

    def read_snapshot(self, t_idx: int) -> np.ndarray:
        if not (0 <= t_idx < self.T):
            raise IndexError(f"t_idx={t_idx} out of [0, {self.T})")
        offset = self.header_offset + t_idx * self._snap_bytes
        with open(self.path, 'rb') as f:
            f.seek(offset)
            buf = f.read(self._snap_bytes)
        if len(buf) != self._snap_bytes:
            raise RuntimeError(f"short read at t_idx={t_idx}")
        return np.frombuffer(buf, dtype=self.dtype) \
                 .reshape(self.Ny, self.Nx).copy()


# --------------------------------------------------------------------------- #
# Bulk-quantity computations                                                  #
# --------------------------------------------------------------------------- #

def _make_k_grids(Nx: int, Ny: int, Lx: float, Ly: float):
    """Build wavevector grids and shell-binning structure."""
    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    Kmag = np.sqrt(K2)

    dk = min(2*np.pi/Lx, 2*np.pi/Ly)
    k_max = int(np.floor(Kmag.max() / dk))
    k_bins = np.arange(0, k_max + 2) * dk
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    K2_safe = K2.copy()
    K2_safe[0, 0] = 1.0
    inv_K2 = 1.0 / K2_safe
    inv_K2[0, 0] = 0.0

    return KX, KY, Kmag, inv_K2, k_bins, k_centers


def compute_bulk_at_tstar(omega_snap: np.ndarray,
                          KX, KY, Kmag, inv_K2,
                          k_bins,
                          Nx: int, Ny: int) -> Tuple[float, float, np.ndarray]:
    """
    Given vorticity omega(x, y), compute:
      E       = (1/2) <|u|^2>_{xy}            (energy density)
      Z       = (1/2) <omega^2>_{xy}          (enstrophy density)
      E(k)    = shell-averaged 1D energy spectrum
    """
    omega = omega_snap.astype(np.float64)

    # enstrophy
    Z = 0.5 * float(np.mean(omega**2))

    # velocity from streamfunction:  psi_hat = -omega_hat / k^2
    omega_hat = np.fft.fft2(omega)
    psi_hat = -omega_hat * inv_K2

    u_hat =  1j * KY * psi_hat
    v_hat = -1j * KX * psi_hat
    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real

    # energy
    E = 0.5 * float(np.mean(u**2 + v**2))

    # 1D shell-averaged energy spectrum
    Euv_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (Nx * Ny)**2
    Ek, _ = np.histogram(Kmag.ravel(), bins=k_bins, weights=Euv_2d.ravel())

    return E, Z, Ek


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-root', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--ref-subdir', type=str, default=REF_SUBDIR)
    p.add_argument('--Lx', type=float, default=2 * np.pi)
    p.add_argument('--Ly', type=float, default=2 * np.pi)
    p.add_argument('--t-star', type=float, default=-1.0,
                   help='physical time t* at which to evaluate bulk quantities '
                        '(-1 = use final saved time)')
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = args.sweep_root / args.ref_subdir
    ref_omega = ref_dir / 'DNS_FR_omega.npy'
    ref_times_path = ref_dir / 'DNS_FR_times.npy'
    if not ref_omega.exists():
        sys.exit(f"ERROR: reference omega not found at {ref_omega}")
    if not ref_times_path.exists():
        sys.exit(f"ERROR: reference times not found at {ref_times_path}")

    ref_reader = _NpyReader(ref_omega)
    ref_times = np.load(ref_times_path)
    Ny, Nx = ref_reader.Ny, ref_reader.Nx
    print(f"reference: {args.ref_subdir} -- shape {ref_reader.shape}")
    print(f"  saved data spans:  t = {ref_times[0]:.4f} ... {ref_times[-1]:.4f}")

    if args.t_star < 0:
        i_star = len(ref_times) - 1
    else:
        i_star = int(np.argmin(np.abs(ref_times - args.t_star)))
    t_star = float(ref_times[i_star])
    print(f"  evaluating bulk quantities at t* = {t_star:.4f} (index {i_star})")

    KX, KY, Kmag, inv_K2, k_bins, k_centers = _make_k_grids(Nx, Ny, args.Lx, args.Ly)
    print(f"  spectrum: {len(k_centers)} shells, k in [{k_centers[0]:.3f}, "
          f"{k_centers[-1]:.3f}]")

    omega_ref_star = ref_reader.read_snapshot(i_star)
    E_ref, Z_ref, Ek_ref = compute_bulk_at_tstar(
        omega_ref_star, KX, KY, Kmag, inv_K2, k_bins, Nx, Ny
    )
    print(f"  reference: E(t*)={E_ref:.6e}  Z(t*)={Z_ref:.6e}")

    rows_dt    = []
    rows_dE    = []
    rows_dZ    = []
    rows_dEk   = []

    for dt_value, sub, label in SWEEP:
        if sub == args.ref_subdir:
            continue
        run_dir = args.sweep_root / sub
        run_omega = run_dir / 'DNS_FR_omega.npy'
        run_times_path = run_dir / 'DNS_FR_times.npy'
        if not run_omega.exists():
            print(f"  skipping {sub}: no omega data found")
            continue

        run_times = np.load(run_times_path)
        if not np.allclose(run_times, ref_times, atol=1e-6):
            sys.exit(f"ERROR: time grid for {sub} differs from reference.")

        t0 = time.time()
        run_reader = _NpyReader(run_omega)
        omega_run_star = run_reader.read_snapshot(i_star)

        E_run, Z_run, Ek_run = compute_bulk_at_tstar(
            omega_run_star, KX, KY, Kmag, inv_K2, k_bins, Nx, Ny
        )

        dE  = abs(E_run - E_ref)
        dZ  = abs(Z_run - Z_ref)
        dEk = float(np.sqrt(np.sum((Ek_run - Ek_ref)**2)))

        rows_dt.append(dt_value)
        rows_dE.append(dE)
        rows_dZ.append(dZ)
        rows_dEk.append(dEk)

        print(f"  {sub:14s}  E={E_run:.6e}  |dE|={dE:.4e}  "
              f"|dZ|={dZ:.4e}  |dE(k)|_2={dEk:.4e}  ({time.time()-t0:.1f}s)")

    rows_dt  = np.array(rows_dt)
    rows_dE  = np.array(rows_dE)
    rows_dZ  = np.array(rows_dZ)
    rows_dEk = np.array(rows_dEk)

    if len(rows_dt) == 0:
        sys.exit("no usable runs found in sweep; nothing to plot")

    order = np.argsort(rows_dt)
    dts  = rows_dt[order]
    dEs  = rows_dE[order]
    dZs  = rows_dZ[order]
    dEks = rows_dEk[order]

    def _slope_lines(ax, x, y_anchor):
        if len(x) < 2:
            return
        dt_line = np.array([x.min(), x.max()])
        ref2 = y_anchor * (dt_line / dt_line[-1]) ** 2
        ref1 = y_anchor * (dt_line / dt_line[-1]) ** 1
        ax.loglog(dt_line, ref2, 'k--', alpha=0.6, lw=1.0, label=r'Slope $= 2$')
        ax.loglog(dt_line, ref1, 'k:',  alpha=0.6, lw=1.0, label=r'Slope $= 1$')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.loglog(dts, dEs, 'o-', color='C0', ms=8, lw=1.5,
              label=r'$|E_{\Delta t}(t^*) - E_{\rm ref}(t^*)|$')
    _slope_lines(ax, dts, dEs[-1])
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$|E_{\Delta t}(t^*) - E_{\rm ref}(t^*)|$')
    ax.set_title(r'(a) Energy at $t^*$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')

    ax = axes[1]
    ax.loglog(dts, dZs, 's-', color='C2', ms=8, lw=1.5,
              label=r'$|Z_{\Delta t}(t^*) - Z_{\rm ref}(t^*)|$')
    _slope_lines(ax, dts, dZs[-1])
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$|Z_{\Delta t}(t^*) - Z_{\rm ref}(t^*)|$')
    ax.set_title(r'(b) Enstrophy at $t^*$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')

    ax = axes[2]
    ax.loglog(dts, dEks, '^-', color='C3', ms=8, lw=1.5,
              label=r'$\|E_{\Delta t}(k, t^*) - E_{\rm ref}(k, t^*)\|_{\ell^2(k)}$')
    _slope_lines(ax, dts, dEks[-1])
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$\|E_{\Delta t}(k,t^*) - E_{\rm ref}(k,t^*)\|_{\ell^2(k)}$')
    ax.set_title(r'(c) Spectrum at $t^*$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')

    fig.suptitle(
        r'AB2-CN2 Temporal Convergence (Bulk Quantities) vs $\Delta t$ '
        rf'(Decaying Turbulence, $N = 256^2$, $L_x = L_y = 2\pi$, '
        rf'$t^* = {t_star:.2f}$, Reference $\Delta t = 10^{{-5}}$)',
        fontsize=12,
    )
    fig.tight_layout()
    out_path = args.out_dir / 'convergence_plot_decay_bulk.png'
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"\nwrote {out_path}")

    csv_path = args.out_dir / 'convergence_plot_decay_bulk.csv'
    with open(csv_path, 'w') as f:
        f.write(f'# t_star = {t_star:.6f}\n')
        f.write(f'# E_ref = {E_ref:.10e}, Z_ref = {Z_ref:.10e}\n')
        f.write('dt,abs_dE,abs_dZ,l2_dEk\n')
        for i in range(len(dts)):
            f.write(f'{dts[i]:.6e},{dEs[i]:.6e},{dZs[i]:.6e},{dEks[i]:.6e}\n')
    print(f"wrote {csv_path}")


if __name__ == '__main__':
    main()
