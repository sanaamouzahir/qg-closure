"""
step1_convergence_plot.py  (DECAYING TURBULENCE)

Temporal convergence analysis for the AB2CN2 IMEX scheme on the decaying-
turbulence dt sweep at 256^2, T=60. Decaying turbulence has NO obstacle,
NO sponge, NO forcing -- a clean QG-with-diffusion-only setup, on a
doubly-periodic domain of size (2*pi, 2*pi). Used as an artifact-free
reference for the closure validation pipeline.

Convergence diagnostics (1 x 4 panel layout):

  (a) trajectory difference  ||omega(t) - omega_ref(t)||_2,
      plotted vs t for every dt run.

  (b) enstrophy time series Z(t), one curve per dt run.

  (c) energy time series E(t), one curve per dt run.

  (d) statistical convergence: |<Z>_run - <Z>_ref|, |<E>_run - <E>_ref|
      averaged over the post-spinup window t >= t_spinup, vs dt on a
      log-log plot. Should converge with slope 2 (AB2CN2 is 2nd-order).

NOTE on multiple ICs. The decaying-turbulence YAML is configured with
n_batch=20, so DNS_FR_omega.npy has shape (20, T, Ny, Nx). We use batch
0 only by default; pass --batch-index to select a different batch.

MEMORY-EFFICIENT IMPLEMENTATION (mmap):
  Reads omega from .npy files (NOT .npz) using np.load(..., mmap_mode='r').
  PREREQUISITE: run prepare_npz_for_mmap.py first to convert each DNS_FR.npz
  into DNS_FR_omega.npy + DNS_FR_times.npy.

Inputs (under sweep-root):
  dt_2em3/    DNS_FR_omega.npy   DNS_FR_times.npy
  dt_1em3/    DNS_FR_omega.npy   DNS_FR_times.npy
  dt_5em4/    DNS_FR_omega.npy   DNS_FR_times.npy
  dt_2p5em4/  DNS_FR_omega.npy   DNS_FR_times.npy
  dt_1p25em4/ DNS_FR_omega.npy   DNS_FR_times.npy
  dt_2em5/    DNS_FR_omega.npy   DNS_FR_times.npy
  dt_1em5/    DNS_FR_omega.npy   DNS_FR_times.npy   <-- reference

Outputs:
  step1_convergence_decay.png  -- 1x4 figure
  step1_convergence_decay.csv  -- per-(dt, t) trajectory errors
  step1_statistics_decay.csv   -- per-(dt, t) energy and enstrophy

Usage:
  python step1_convergence_plot.py \\
      --sweep-root /path/to/outputs/decaying_turb_dt_sweep \\
      --out-dir   . \\
      [--t-spinup 5.0] [--chunk 50] [--batch-index 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Iterator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# --------------------------------------------------------------------------- #
# Sweep configuration
# --------------------------------------------------------------------------- #

SWEEP: List[Tuple[float, str, str]] = [
    (2.0e-3,   'dt_2em3',     r'$\Delta t = 2\times 10^{-3}$'),
    (1.0e-3,   'dt_1em3',     r'$\Delta t = 10^{-3}$'),
    (5.0e-4,   'dt_5em4',     r'$\Delta t = 5\times 10^{-4}$'),
    (2.5e-4,   'dt_2p5em4',   r'$\Delta t = 2.5\times 10^{-4}$'),
    (1.25e-4,  'dt_1p25em4',  r'$\Delta t = 1.25\times 10^{-4}$'),
    (2.0e-5,   'dt_2em5',     r'$\Delta t = 2\times 10^{-5}$'),
    (1.0e-5,   'dt_1em5',     r'$\Delta t = 10^{-5}$ (ref)'),
]
REF_INDEX = -1


# --------------------------------------------------------------------------- #
# Lazy run handle: mmap-backed access to omega and times
# --------------------------------------------------------------------------- #

class RunHandle:
    """
    Memory-mapped wrapper around DNS_FR_omega.npy + DNS_FR_times.npy.

    The arrays live on disk; the OS pages in just the parts we touch. Slicing
    or fancy-indexing a chunk of `omega` materializes only that chunk in RAM.
    """

    def __init__(self, run_dir: Path):
        omega_path = run_dir / 'DNS_FR_omega.npy'
        times_path = run_dir / 'DNS_FR_times.npy'
        if not omega_path.exists():
            raise FileNotFoundError(
                f"Missing {omega_path}. "
                f"Did you run prepare_npz_for_mmap.py first?"
            )
        if not times_path.exists():
            raise FileNotFoundError(f"Missing {times_path}")

        # mmap the omega array; we promise not to write to it.
        self._omega_mmap = np.load(omega_path, mmap_mode='r')
        # squeeze leading batch axis if present; this stays a memmap view
        if self._omega_mmap.ndim == 4:
            if self._omega_mmap.shape[0] != 1:
                print(f"  warning: B={self._omega_mmap.shape[0]} != 1 in "
                      f"{run_dir.name}; using batch 0")
            self._omega_mmap = self._omega_mmap[0]   # (T, Ny, Nx) memmap view

        # times is small; just read it eagerly into memory
        self._times = np.asarray(np.load(times_path), dtype=np.float64)

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def n_times(self) -> int:
        return self._omega_mmap.shape[0]

    @property
    def shape_yx(self) -> Tuple[int, int]:
        return self._omega_mmap.shape[-2], self._omega_mmap.shape[-1]

    @property
    def dtype(self):
        return self._omega_mmap.dtype

    def get_slab(self, t_start: int, t_stop: int) -> np.ndarray:
        """Materialize omega[t_start:t_stop] as a fresh float64 array.

        Only the requested range is read from disk thanks to mmap.
        """
        return np.asarray(self._omega_mmap[t_start:t_stop], dtype=np.float64)

    def get_indexed(self, indices: np.ndarray) -> np.ndarray:
        """Materialize omega[indices] as a fresh float64 array.

        Used for sparse/scattered time selections.
        """
        return np.asarray(self._omega_mmap[indices], dtype=np.float64)

    def iter_slabs(self, chunk: int) -> Iterator[Tuple[int, int, np.ndarray]]:
        """Yield (t_start, t_stop, slab) over the full time axis."""
        T = self.n_times
        for t_start in range(0, T, chunk):
            t_stop = min(t_start + chunk, T)
            yield t_start, t_stop, self.get_slab(t_start, t_stop)

    def close(self):
        """Drop the memmap. Subsequent access will raise."""
        # numpy memmap objects don't have an explicit close; deletion frees them
        del self._omega_mmap


# --------------------------------------------------------------------------- #
# Time matching
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Spatial mask (downstream-near-wake window for cylinder)                     #
# --------------------------------------------------------------------------- #
#
# The cylinder runs at very fine dt show wrap-around artifacts that grow as
# dt shrinks (sponge becomes too stiff for explicit AB2 at small dt, leaks
# into upstream and propagates through periodicity). Empirically, the
# near-wake window x in [0.30 Lx, 0.55 Lx] is essentially unaffected:
# RMS varies only ~12% across all 7 dt values vs the ~30x growth of upstream
# contamination. Restricting all norms and statistics to this window gives a
# clean convergence study without re-running the sweep.

def build_x_mask(Nx: int, x_min_frac: float, x_max_frac: float) -> np.ndarray:
    """1D bool mask along x: keep columns with x in [x_min_frac, x_max_frac] * Lx."""
    if not (0.0 <= x_min_frac < x_max_frac <= 1.0):
        raise ValueError(
            f"need 0 <= x_min_frac < x_max_frac <= 1, "
            f"got [{x_min_frac}, {x_max_frac}]"
        )
    col_frac = (np.arange(Nx) + 0.5) / Nx
    return (col_frac >= x_min_frac) & (col_frac <= x_max_frac)


def apply_x_mask(slab: np.ndarray, x_mask: np.ndarray) -> np.ndarray:
    """Restrict a slab (..., Ny, Nx) to the masked x columns."""
    return slab[..., :, x_mask]


def matched_time_indices(times_a: np.ndarray, times_b: np.ndarray,
                         tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """Indices ia, ib with times_a[ia] == times_b[ib] within tol."""
    qa = np.round(times_a / tol).astype(np.int64)
    qb = np.round(times_b / tol).astype(np.int64)
    common, ia, ib = np.intersect1d(qa, qb, return_indices=True)
    order = np.argsort(ia)
    return ia[order], ib[order]


# --------------------------------------------------------------------------- #
# Streaming reductions
# --------------------------------------------------------------------------- #

def streaming_traj_errors(
    run: RunHandle,
    ref: RunHandle,
    chunk: int,
    x_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Per-time error norms |run - ref| computed on the masked downstream window
    only. Memory cost: 2 * chunk * Ny * Nx_masked * 8 B.
    """
    ia, ib = matched_time_indices(run.times, ref.times)
    n_match = len(ia)
    if n_match == 0:
        return dict(t=np.array([]), l2=np.array([]),
                    linf=np.array([]), rel_l2=np.array([]))

    t_out      = np.empty(n_match, dtype=np.float64)
    l2_out     = np.empty(n_match, dtype=np.float64)
    linf_out   = np.empty(n_match, dtype=np.float64)
    rel_l2_out = np.empty(n_match, dtype=np.float64)

    for c0 in range(0, n_match, chunk):
        c1 = min(c0 + chunk, n_match)
        ia_chunk = ia[c0:c1]
        ib_chunk = ib[c0:c1]
        # mmap fancy-indexing materializes only these rows; mask immediately
        run_slab = apply_x_mask(run.get_indexed(ia_chunk), x_mask)
        ref_slab = apply_x_mask(ref.get_indexed(ib_chunk), x_mask)
        diff = run_slab - ref_slab

        l2     = np.sqrt(np.mean(diff**2, axis=(-1, -2)))
        linf   = np.max(np.abs(diff), axis=(-1, -2))
        norm_r = np.sqrt(np.mean(ref_slab**2, axis=(-1, -2)))
        norm_r_safe = np.where(norm_r > 0, norm_r, np.nan)
        rel    = l2 / norm_r_safe

        t_out[c0:c1]      = run.times[ia_chunk]
        l2_out[c0:c1]     = l2
        linf_out[c0:c1]   = linf
        rel_l2_out[c0:c1] = rel

        del run_slab, ref_slab, diff

    return dict(t=t_out, l2=l2_out, linf=linf_out, rel_l2=rel_l2_out)


def streaming_energy_enstrophy(
    run: RunHandle,
    Lx: float, Ly: float,
    chunk: int,
    x_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Streaming E(t) and Z(t), restricted to the masked downstream window.

    Energy needs a Poisson solve to recover (u, v) from omega, which is a
    global operation. We therefore do the FFT on the *full* periodic field,
    inverse-FFT to get u, v in physical space, and then apply the x-mask
    when computing the spatial mean. This is well-defined even though the
    masked region is not itself periodic.

    Z is computed directly on the masked region: Z = 0.5 * <omega^2>_masked.
    """
    Ny, Nx = run.shape_yx
    Nx_masked = int(x_mask.sum())

    # spectral inverse-Laplacian weights on the FULL (periodic) grid
    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    inv_K2 = 1.0 / K2
    inv_K2[0, 0] = 0.0

    T = run.n_times
    E = np.empty(T, dtype=np.float64)
    Z = np.empty(T, dtype=np.float64)

    for t0, t1, slab_full in run.iter_slabs(chunk):
        # Enstrophy on masked region
        slab_masked = apply_x_mask(slab_full, x_mask)
        Z[t0:t1] = 0.5 * np.mean(slab_masked**2, axis=(-1, -2))

        # Energy: Poisson-solve for psi on full domain, then u = -dpsi/dy,
        # v = +dpsi/dx, then physical-space mean of u^2+v^2 over masked window.
        omega_hat = np.fft.fft2(slab_full)
        psi_hat = -omega_hat * inv_K2          # (-omega = lap psi  =>  psi_hat = -omega_hat / K^2)
        u_hat   = -1j * KY * psi_hat
        v_hat   = +1j * KX * psi_hat
        u_full  = np.fft.ifft2(u_hat).real
        v_full  = np.fft.ifft2(v_hat).real
        u_masked = apply_x_mask(u_full, x_mask)
        v_masked = apply_x_mask(v_full, x_mask)
        E[t0:t1] = 0.5 * np.mean(u_masked**2 + v_masked**2, axis=(-1, -2))

        del (slab_full, slab_masked, omega_hat, psi_hat,
             u_hat, v_hat, u_full, v_full, u_masked, v_masked)

    return run.times.copy(), E, Z


# --------------------------------------------------------------------------- #
# Decorrelation detection
# --------------------------------------------------------------------------- #

def detect_decorrelation_time(err_curves: List[dict],
                              rel_threshold: float,
                              t_min: float = 0.0) -> float:
    """
    t_break = first time *after* t_min the median rel_L2 exceeds rel_threshold.

    The impulsive cylinder startup produces large rel_L2 in the first ~10 t.u.
    that is NOT trajectory decorrelation -- it's just the runs producing
    slightly different transient shear layers as they adjust to the obstacle.
    We skip this window via t_min (typically equal to t_spinup).
    """
    if not err_curves:
        return float('inf')
    t0 = err_curves[0]['t']
    rel_stack = []
    for c in err_curves:
        rel = np.interp(t0, c['t'], c['rel_l2'])
        rel_stack.append(rel)
    rel_arr = np.stack(rel_stack, axis=0)
    median_rel = np.nanmedian(rel_arr, axis=0)
    # Mask out times before t_min so the startup transient doesn't trigger
    valid = t0 >= t_min
    crossing = np.where(valid & (median_rel > rel_threshold))[0]
    if len(crossing) == 0:
        return float('inf')
    return float(t0[crossing[0]])


# --------------------------------------------------------------------------- #
# Main computation
# --------------------------------------------------------------------------- #

def _try_load_ens(run_dir: Path, filename: str, expected_T: int = -1) -> np.ndarray:
    """
    Try to load DNS_FR_Z_ens.npy or DNS_FR_E_ens.npy from a run directory.
    Returns None if the file doesn't exist (graceful fallback when
    prepare_npz_for_mmap.py was run without ensemble support). If
    expected_T is given (>0), warns when shape doesn't match.
    """
    p = run_dir / filename
    if not p.exists():
        return None
    arr = np.load(p)
    if expected_T > 0 and arr.shape[0] != expected_T:
        print(f"    warning: {p.name} has T={arr.shape[0]}, "
              f"expected {expected_T}")
    return arr


def compute_all(
    sweep_root: Path,
    rel_threshold: float,
    t_spinup: float,
    Lx: float, Ly: float,
    chunk: int,
    x_min_frac: float,
    x_max_frac: float,
) -> Tuple[List[dict], Dict[str, dict], float, np.ndarray]:
    """
    Open the reference once (mmap; cheap). For each non-reference run:
      * compute streaming trajectory errors vs reference (on masked region)
      * compute streaming energy/enstrophy (on masked region)
      * release the run's mmap before opening the next.
    Also compute energy/enstrophy for the reference itself.

    The x-mask is built once from the reference's grid shape and reused.
    """
    ref_subdir = SWEEP[REF_INDEX][1]
    ref_dir = sweep_root / ref_subdir
    print(f"opening reference: {ref_dir}")
    ref = RunHandle(ref_dir)
    Ny, Nx = ref.shape_yx
    print(f"  shape (T, Ny, Nx) = ({ref.n_times}, {Ny}, {Nx}) "
          f"dtype = {ref.dtype}")

    # Build x-mask once
    x_mask = build_x_mask(Nx, x_min_frac, x_max_frac)
    Nx_masked = int(x_mask.sum())
    print(f"  x-mask: x in [{x_min_frac:.3f} Lx, {x_max_frac:.3f} Lx]  "
          f"-> {Nx_masked}/{Nx} columns kept ({100.0*Nx_masked/Nx:.1f}%)")

    err_curves: List[dict] = []
    stats: Dict[str, dict] = {}

    print("  computing reference E(t), Z(t) on masked window ...")
    t_ref, E_ref, Z_ref = streaming_energy_enstrophy(ref, Lx, Ly, chunk, x_mask)

    # Try to load ensemble-averaged Z, E (from prepare_npz_for_mmap.py); these
    # are computed across ALL batches B=20 for decaying-turb, and give a
    # statistically meaningful target for convergence.
    Z_ens_ref = _try_load_ens(ref_dir, 'DNS_FR_Z_ens.npy', expected_T=ref.n_times)
    E_ens_ref = _try_load_ens(ref_dir, 'DNS_FR_E_ens.npy', expected_T=ref.n_times)
    has_ens = (Z_ens_ref is not None) and (E_ens_ref is not None)
    if has_ens:
        print(f"    loaded ensemble Z_ens, E_ens (T={len(Z_ens_ref)})")
    else:
        print("    [info] no DNS_FR_{Z,E}_ens.npy found; "
              "the ensemble convergence figure will be skipped")

    stats[ref_subdir] = dict(
        dt=SWEEP[REF_INDEX][0], label=SWEEP[REF_INDEX][2],
        t=t_ref, E=E_ref, Z=Z_ref,
        Z_ens=Z_ens_ref, E_ens=E_ens_ref,
    )
    print(f"    <E>_masked={np.mean(E_ref[t_ref >= t_spinup]):.4e}  "
          f"<Z>_masked={np.mean(Z_ref[t_ref >= t_spinup]):.4e}")

    for dt_val, subdir, label in SWEEP:
        if subdir == ref_subdir:
            continue
        run_dir = sweep_root / subdir
        print(f"opening run: {run_dir}")
        run = RunHandle(run_dir)
        print(f"  shape (T, Ny, Nx) = ({run.n_times}, {run.shape_yx[0]}, {run.shape_yx[1]})")

        print(f"  computing trajectory errors vs reference (masked) ...")
        errs = streaming_traj_errors(run, ref, chunk, x_mask)
        if len(errs['t']) > 0:
            err_curves.append(dict(
                dt=dt_val, label=label, subdir=subdir, **errs,
            ))
            print(f"    n_matched_times = {len(errs['t'])}")

        print(f"  computing E(t), Z(t) (masked) ...")
        t_run, E_run, Z_run = streaming_energy_enstrophy(run, Lx, Ly, chunk, x_mask)
        Z_ens_run = _try_load_ens(run_dir, 'DNS_FR_Z_ens.npy', expected_T=run.n_times)
        E_ens_run = _try_load_ens(run_dir, 'DNS_FR_E_ens.npy', expected_T=run.n_times)
        stats[subdir] = dict(
            dt=dt_val, label=label, t=t_run, E=E_run, Z=Z_run,
            Z_ens=Z_ens_run, E_ens=E_ens_run,
        )
        print(f"    <E>_masked={np.mean(E_run[t_run >= t_spinup]):.4e}  "
              f"<Z>_masked={np.mean(Z_run[t_run >= t_spinup]):.4e}")

        run.close()

    ref.close()

    t_break = detect_decorrelation_time(err_curves, rel_threshold, t_min=t_spinup)
    if np.isfinite(t_break):
        print(f"\ndetected decorrelation time t_break = {t_break:.3f} "
              f"(median rel_L2 first crosses {rel_threshold*100:.0f}% after t_spinup={t_spinup})")
    else:
        t_break = err_curves[0]['t'][-1] if err_curves else 0.0
        print(f"\nno decorrelation past t_spinup={t_spinup}; using full window, t_break = {t_break:.3f}")

    return err_curves, stats, t_break, x_mask


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_convergence(err_curves: List[dict],
                     stats: Dict[str, dict],
                     t_break: float,
                     rel_threshold: float,
                     t_spinup: float,
                     out_path: Path) -> None:
    r"""
    PNG 1: three panels (1 x 3) using batch-0 data only.
      (a) Per-time normalized trajectory difference $\|\bar\omega(t) -
          \bar\omega_{\rm ref}(t)\|_2$ vs $t$ (log-y).
      (b) Enstrophy time series $Z(t)$ for each $\Delta t$.
      (c) Energy time series $E(t)$ for each $\Delta t$.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = mpl.colormaps['viridis']
    n_curves = max(len(err_curves), 1)

    # ===== (a) per-time trajectory difference ===== #
    ax = axes[0]
    for i, c in enumerate(err_curves):
        color = cmap(i / max(n_curves - 1, 1))
        ax.semilogy(c['t'], c['l2'], color=color, label=c['label'], lw=1.4)
    ax.axvline(t_spinup, color='k', ls=':', alpha=0.6,
               label=fr'$t_{{\rm spinup}} = {t_spinup:.1f}$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\|\bar\omega(t) - \bar\omega_{\rm ref}(t)\|_2$')
    ax.set_title(r'(a) $\bar\omega(t) - \bar\omega_{\rm ref}(t)$, Batch #0')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7, loc='best')

    # ===== (b) enstrophy Z(t), batch 0 ===== #
    ax = axes[1]
    for i, (dt_val, subdir, label) in enumerate(SWEEP):
        if subdir not in stats:
            continue
        s = stats[subdir]
        color = cmap(i / max(len(SWEEP) - 1, 1))
        ax.plot(s['t'], s['Z'], color=color, label=label, lw=1.2)
    ax.axvline(t_spinup, color='k', ls=':', alpha=0.6,
               label=fr'$t_{{\rm spinup}} = {t_spinup:.1f}$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$Z(t) = \frac{1}{2} \langle \bar\omega^2 \rangle$')
    ax.set_title(r'(b) Enstrophy $Z(t)$, Batch #0')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')

    # ===== (c) energy E(t), batch 0 ===== #
    ax = axes[2]
    for i, (dt_val, subdir, label) in enumerate(SWEEP):
        if subdir not in stats:
            continue
        s = stats[subdir]
        color = cmap(i / max(len(SWEEP) - 1, 1))
        ax.plot(s['t'], s['E'], color=color, label=label, lw=1.2)
    ax.axvline(t_spinup, color='k', ls=':', alpha=0.6)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$E(t) = \frac{1}{2} \langle |\nabla\bar\psi|^2 \rangle$')
    ax.set_title(r'(c) Energy $E(t)$, Batch #0')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')

    fig.suptitle(
        r'AB2-CN2 Temporal Convergence Analysis '
        r'(Case #2 Decaying Turbulence, $N = 256^2$, '
        r'$L_x = L_y = 2\pi$, $T = 60$)',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_statistical_convergence(stats: Dict[str, dict],
                                 t_spinup: float,
                                 T_total: float,
                                 out_path: Path) -> None:
    r"""
    PNG 2: three panels (1 x 3) of the ensemble-averaged statistics.
      (a) Ensemble-averaged enstrophy $\langle Z \rangle_{\rm ens}(t)$.
      (b) Ensemble-averaged energy $\langle E \rangle_{\rm ens}(t)$.
      (c) Convergence vs $\Delta t$: time-integrated absolute difference
          $\frac{1}{T - t_{\rm spinup}} \int_{t_{\rm spinup}}^{T}
          | \langle Z \rangle_{\rm ens}(t) - \langle Z \rangle_{\rm ens, ref}(t) |
          \, dt$ (and likewise for $E$). Both slope-1 and slope-2 reference
          lines are plotted.

    Skipped silently if any run lacks DNS_FR_{Z,E}_ens.npy.
    """
    ref_subdir = SWEEP[REF_INDEX][1]
    ref = stats.get(ref_subdir)
    if (ref is None) or (ref.get('Z_ens') is None) or (ref.get('E_ens') is None):
        print("[skip] no ensemble Z_ens / E_ens for the reference run; "
              "did you regenerate with the latest prepare_npz_for_mmap.py?")
        return

    # Verify all SWEEP runs have ensemble data
    runs_with_ens = [(dt, sub, lab) for (dt, sub, lab) in SWEEP
                     if sub in stats
                     and stats[sub].get('Z_ens') is not None
                     and stats[sub].get('E_ens') is not None]
    if len(runs_with_ens) < 2:
        print(f"[skip] only {len(runs_with_ens)} runs have ensemble data; "
              f"need at least 2.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = mpl.colormaps['viridis']

    # ===== (a) <Z>_ens(t) ===== #
    ax = axes[0]
    for i, (dt_val, subdir, label) in enumerate(SWEEP):
        if subdir not in stats or stats[subdir].get('Z_ens') is None:
            continue
        s = stats[subdir]
        color = cmap(i / max(len(SWEEP) - 1, 1))
        ax.plot(s['t'], s['Z_ens'], color=color, label=label, lw=1.3)
    ax.axvline(t_spinup, color='k', ls=':', alpha=0.6,
               label=fr'$t_{{\rm spinup}} = {t_spinup:.1f}$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\langle Z \rangle_{\rm ens}(t)$')
    ax.set_title(r'(a) Ensemble-Averaged Enstrophy $\langle Z \rangle_{\rm ens}(t)$')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')

    # ===== (b) <E>_ens(t) ===== #
    ax = axes[1]
    for i, (dt_val, subdir, label) in enumerate(SWEEP):
        if subdir not in stats or stats[subdir].get('E_ens') is None:
            continue
        s = stats[subdir]
        color = cmap(i / max(len(SWEEP) - 1, 1))
        ax.plot(s['t'], s['E_ens'], color=color, label=label, lw=1.3)
    ax.axvline(t_spinup, color='k', ls=':', alpha=0.6)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\langle E \rangle_{\rm ens}(t)$')
    ax.set_title(r'(b) Ensemble-Averaged Energy $\langle E \rangle_{\rm ens}(t)$')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')

    # ===== (c) Convergence: time-integrated abs diff vs dt ===== #
    ax = axes[2]
    t_ref = ref['t']
    Z_ref = ref['Z_ens']
    E_ref = ref['E_ens']
    mask_ref = t_ref >= t_spinup
    if not mask_ref.any():
        print("[skip] no reference times past t_spinup")
        plt.close(fig)
        return
    window_T = float(t_ref[mask_ref][-1] - t_ref[mask_ref][0])
    if window_T <= 0:
        window_T = 1.0  # fallback

    dts, dZ_int, dE_int = [], [], []
    for dt_val, subdir, _ in SWEEP:
        if subdir == ref_subdir:
            continue
        if subdir not in stats:
            continue
        s = stats[subdir]
        if s.get('Z_ens') is None or s.get('E_ens') is None:
            continue
        # Match times -- they should be identical between runs at the same
        # save_rate, but be tolerant.
        t_run = s['t']
        # Take intersection of save grids inside the post-spinup window
        # (in practice all runs use identical save grids, so this is just
        # a guard).
        m_run = t_run >= t_spinup
        if not m_run.any():
            continue
        # If lengths match, use elementwise; otherwise interpolate ref to run.
        if len(t_run[m_run]) == len(t_ref[mask_ref]) and \
           np.allclose(t_run[m_run], t_ref[mask_ref], atol=1e-6):
            dZ_t = np.abs(s['Z_ens'][m_run] - Z_ref[mask_ref])
            dE_t = np.abs(s['E_ens'][m_run] - E_ref[mask_ref])
            t_use = t_run[m_run]
        else:
            Z_ref_interp = np.interp(t_run[m_run], t_ref, Z_ref)
            E_ref_interp = np.interp(t_run[m_run], t_ref, E_ref)
            dZ_t = np.abs(s['Z_ens'][m_run] - Z_ref_interp)
            dE_t = np.abs(s['E_ens'][m_run] - E_ref_interp)
            t_use = t_run[m_run]
        # 1/(T-t_spinup) * integral over post-spinup window
        # numpy 2.x renamed trapz -> trapezoid
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        dZ_int.append(_trapz(dZ_t, t_use) / window_T)
        dE_int.append(_trapz(dE_t, t_use) / window_T)
        dts.append(dt_val)

    if len(dts) == 0:
        print("[skip] no comparable runs for convergence panel")
        plt.close(fig)
        return

    dts = np.array(dts); dZ_int = np.array(dZ_int); dE_int = np.array(dE_int)
    order = np.argsort(dts)
    dts = dts[order]; dZ_int = dZ_int[order]; dE_int = dE_int[order]

    ax.loglog(dts, dZ_int, 'D-',
              label=(r'$\frac{1}{T - t_{\rm spinup}}\int_{t_{\rm spinup}}^{T}'
                     r'|\langle Z\rangle_{\rm ens} - \langle Z\rangle_{\rm ens,ref}|\,dt$'),
              ms=8, lw=1.8, color='C2')
    ax.loglog(dts, dE_int, 'v-',
              label=(r'$\frac{1}{T - t_{\rm spinup}}\int_{t_{\rm spinup}}^{T}'
                     r'|\langle E\rangle_{\rm ens} - \langle E\rangle_{\rm ens,ref}|\,dt$'),
              ms=8, lw=1.8, color='C4')

    # Slope-1 and slope-2 reference lines, anchored at the smallest non-ref dt
    if len(dts) >= 1 and dZ_int[0] > 0:
        ref_y2 = dZ_int[0] * (dts / dts[0])**2
        ax.loglog(dts, ref_y2, 'k--', alpha=0.6, label=r'Slope $= 2$')
        ref_y1 = dZ_int[0] * (dts / dts[0])
        ax.loglog(dts, ref_y1, 'k:', alpha=0.6, label=r'Slope $= 1$')

    # Slope readouts
    if len(dts) >= 3:
        if np.all(dZ_int > 0):
            sZ, _ = np.polyfit(np.log(dts), np.log(dZ_int), 1)
            ax.text(0.05, 0.95, fr'$Z$ Slope $= {sZ:.2f}$',
                    transform=ax.transAxes, va='top', color='C2',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        if np.all(dE_int > 0):
            sE, _ = np.polyfit(np.log(dts), np.log(dE_int), 1)
            ax.text(0.05, 0.85, fr'$E$ Slope $= {sE:.2f}$',
                    transform=ax.transAxes, va='top', color='C4',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'Time-Integrated Absolute Difference')
    ax.set_title(r'(c) Convergence Of $\langle Z \rangle_{\rm ens}$ And $\langle E \rangle_{\rm ens}$ Vs $\Delta t$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7, loc='best')

    fig.suptitle(
        r'AB2-CN2, Statistical Convergence '
        r'(Case #2 Decaying Turbulence, $N = 256^2$, '
        r'$L_x = L_y = 2\pi$, $T = 60$)',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------------- #

def write_trajectory_csv(err_curves: List[dict], out_path: Path) -> None:
    rows = ['dt,subdir,t,l2,linf,rel_l2']
    for c in err_curves:
        for i, t in enumerate(c['t']):
            rows.append(
                f"{c['dt']:.6e},{c['subdir']},{t:.6f},"
                f"{c['l2'][i]:.6e},{c['linf'][i]:.6e},{c['rel_l2'][i]:.6e}"
            )
    out_path.write_text('\n'.join(rows))
    print(f"wrote {out_path}  ({len(rows)-1} rows)")


def write_stats_csv(stats: Dict[str, dict], out_path: Path) -> None:
    rows = ['dt,subdir,t,E,Z']
    for dt_val, subdir, _ in SWEEP:
        if subdir not in stats:
            continue
        s = stats[subdir]
        for i, t in enumerate(s['t']):
            rows.append(f"{dt_val:.6e},{subdir},{t:.6f},{s['E'][i]:.6e},{s['Z'][i]:.6e}")
    out_path.write_text('\n'.join(rows))
    print(f"wrote {out_path}  ({len(rows)-1} rows)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--sweep-root', type=Path, required=True,
                   help='root containing dt_1em3/, dt_5em4/, ... subdirectories')
    p.add_argument('--out-dir', type=Path, default=Path.cwd(),
                   help='where to write step1_convergence.{png,csv}')
    p.add_argument('--rel-error-threshold', type=float, default=0.30,
                   help='relative L2 threshold for detecting trajectory '
                        'decorrelation (default 0.30)')
    p.add_argument('--t-spinup', type=float, default=5.0,
                   help='spin-up time for statistical observables (default '
                        '5.0 for decaying turbulence)')
    p.add_argument('--Lx', type=float, default=2 * np.pi,
                   help='domain size in x for energy computation '
                        '(default 2*pi for decaying turbulence)')
    p.add_argument('--Ly', type=float, default=2 * np.pi,
                   help='domain size in y (default 2*pi for decaying turbulence)')
    p.add_argument('--chunk', type=int, default=50,
                   help='time-axis chunk size for streaming reductions '
                        '(default 50)')
    p.add_argument('--x-min-frac', type=float, default=0.0,
                   help='lower x bound of analysis window as fraction of Lx '
                        '(default 0.0 = full domain; decaying turb has no '
                        'boundary artifacts to mask out)')
    p.add_argument('--x-max-frac', type=float, default=1.0,
                   help='upper x bound of analysis window as fraction of Lx '
                        '(default 1.0 = full domain)')
    p.add_argument('--batch-index', type=int, default=0,
                   help='which IC batch to use (default 0; YAML configures n_batch=20)')
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    err_curves, stats, t_break, x_mask = compute_all(
        args.sweep_root,
        rel_threshold=args.rel_error_threshold,
        t_spinup=args.t_spinup,
        Lx=args.Lx, Ly=args.Ly,
        chunk=args.chunk,
        x_min_frac=args.x_min_frac,
        x_max_frac=args.x_max_frac,
    )

    plot_convergence(err_curves, stats, t_break,
                     rel_threshold=args.rel_error_threshold,
                     t_spinup=args.t_spinup,
                     out_path=args.out_dir / 'step1_convergence_decay.png')
    write_trajectory_csv(err_curves, args.out_dir / 'step1_convergence_decay.csv')
    write_stats_csv(stats,           args.out_dir / 'step1_statistics_decay.csv')

    # PNG 2: ensemble-averaged statistics. Determined by whether
    # DNS_FR_{Z,E}_ens.npy were produced by prepare_npz_for_mmap.py.
    # Reference run T (via stats) gives the integration window.
    ref_subdir = SWEEP[REF_INDEX][1]
    if ref_subdir in stats and stats[ref_subdir].get('Z_ens') is not None:
        T_total = float(stats[ref_subdir]['t'][-1])
    else:
        T_total = 60.0  # fallback for decaying-turb default
    plot_statistical_convergence(
        stats,
        t_spinup=args.t_spinup,
        T_total=T_total,
        out_path=args.out_dir / 'step1_statistical_convergence_decay.png',
    )

    print()
    print(f"summary:")
    if args.x_min_frac > 0.0 or args.x_max_frac < 1.0:
        print(f"  analysis window: x in [{args.x_min_frac:.3f} Lx, {args.x_max_frac:.3f} Lx]")
    else:
        print(f"  analysis window: full domain")
    print(f"  t_spinup = {args.t_spinup:.1f}")


if __name__ == '__main__':
    main()