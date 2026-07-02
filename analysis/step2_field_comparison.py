"""
step2_field_comparison.py  (FLOW PAST CYLINDER, Re ~ 200)

Side-by-side vorticity field comparison across the decaying-turbulence dt sweep at
three representative times. The point is to make visible *what* truncation
error looks like in physical space for deterministic vortex shedding.

This is the cleanest test case: decaying turbulence has no forcing, no obstacle,
flow is deterministic (after spin-up) so trajectories don't decorrelate by
chaotic divergence. Differences between dt levels manifest primarily as
*shedding-cycle phase drift* and as small-scale numerical dissipation in
the wake. The expected pattern in the 'diff' column is therefore an
oscillating dipole shifted along the wake axis.

Three runs are shown:
  - largest dt  (coarsest -- biggest expected error)
  - middle dt   (transitional)
  - smallest dt (reference -- best available "truth")

Three times are chosen to cover early, mid, and late shedding regimes:
  t1 = "early"  -- 1-2 shedding cycles in (default 5.0)
  t2 = "middle" -- mid-simulation, fully developed wake (default 25.0)
  t3 = "late"   -- statistical regime (default 45.0)

You can override these via --t-early/--t-middle/--t-late if needed.

The figure has a 3 x 4 grid:
  rows    = times
  columns = (coarsest, middle, smallest=ref) field, then (coarsest - ref) diff

Color scales:
  - field columns: shared symmetric vmin/vmax across all field panels at a row
  - diff column:  symmetric vmin/vmax based on the diff itself

Usage:
  python step2_field_comparison.py \\
      --sweep-root /path/to/decaying_turb_dt_sweep \\
      [--convergence-csv /path/to/step1_convergence_decay.csv] \\
      --out-dir   .
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Same sweep config as Step 1
SWEEP: List[Tuple[float, str, str]] = [
    (2.0e-3,   'dt_2em3',     r'$\Delta t = 2\times 10^{-3}$'),
    (1.0e-3,   'dt_1em3',     r'$\Delta t = 10^{-3}$'),
    (5.0e-4,   'dt_5em4',     r'$\Delta t = 5\times 10^{-4}$'),
    (2.5e-4,   'dt_2p5em4',   r'$\Delta t = 2.5\times 10^{-4}$'),
    (1.25e-4,  'dt_1p25em4',  r'$\Delta t = 1.25\times 10^{-4}$'),
    (2.0e-5,   'dt_2em5',     r'$\Delta t = 2\times 10^{-5}$'),
    (1.0e-5,   'dt_1em5',     r'$\Delta t = 10^{-5}$ (ref)'),
]

# SWEEP indices for the 4 dts displayed in Step 2 (ordered as the user
# specified: smallest non-ref first, then a moderate one, then the two
# largest). Reference is the smallest dt, dt_1em5 = SWEEP[6].
#
#   col 0:  dt = 2e-5    (index 5)  -- closest to ref
#   col 1:  dt = 2.5e-4  (index 3)
#   col 2:  dt = 2e-3    (index 0)  -- coarsest
#   col 3:  dt = 1e-3    (index 1)
#   ref:    dt = 1e-5    (index 6)
SELECTED_INDICES = [5, 3, 0, 1]   # the dts to show (NOT including reference)
REF_INDEX        = 6


# --------------------------------------------------------------------------- #
# Lazy run handle (same idea as step 1)
# --------------------------------------------------------------------------- #

class RunHandle:
    """
    Lazy snapshot loader for one dt run. Tries np.load(mmap_mode='r') first
    (fast random access). If that fails with OSError -- which happens on
    memory-constrained machines where the full virtual-address reservation
    is rejected -- falls back to reading each snapshot directly from the
    .npy file using its header layout, materializing only one snapshot at
    a time.
    """

    def __init__(self, run_dir: Path):
        omega_path = run_dir / 'DNS_FR_omega.npy'
        times_path = run_dir / 'DNS_FR_times.npy'
        if not omega_path.exists():
            raise FileNotFoundError(
                f"Missing {omega_path}. Did you run prepare_npz_for_mmap.py first?"
            )
        if not times_path.exists():
            raise FileNotFoundError(f"Missing {times_path}")

        self._times = np.asarray(np.load(times_path), dtype=np.float64)
        self._omega_path = omega_path

        # Always parse the header so we know shape/dtype regardless of mode.
        from numpy.lib import format as np_format
        with open(omega_path, 'rb') as f:
            major, minor = np_format.read_magic(f)
            if (major, minor) == (1, 0):
                shape, fortran_order, dtype = np_format.read_array_header_1_0(f)
            elif (major, minor) == (2, 0):
                shape, fortran_order, dtype = np_format.read_array_header_2_0(f)
            else:
                raise RuntimeError(f"unsupported .npy version {major}.{minor}")
            self._header_offset = f.tell()
        if fortran_order:
            raise RuntimeError("Fortran-order .npy not supported here")
        self._dtype = dtype

        # Layouts: (T, Ny, Nx) or (B, T, Ny, Nx). Strip leading B==1 if present.
        self._batch_index = None
        if len(shape) == 4:
            B, T, Ny, Nx = shape
            if B == 1:
                self._batch_index = 0
                self._n_snapshots = T
                self._spatial_shape = (Ny, Nx)
            else:
                # multi-batch: pick batch 0 by default and warn
                print(f"  [warn] {run_dir.name}: omega has B={B} batches; "
                      f"using batch 0. Ideally regenerate with "
                      f"prepare_npz_for_mmap.py --batch-index 0 to strip the dim.")
                self._batch_index = 0
                self._n_snapshots = T
                self._spatial_shape = (Ny, Nx)
        elif len(shape) == 3:
            self._n_snapshots = shape[0]
            self._spatial_shape = (shape[1], shape[2])
        else:
            raise RuntimeError(f"unexpected omega shape {shape}")

        # Try mmap; fall back to per-snapshot reads if mmap fails
        try:
            mm = np.load(omega_path, mmap_mode='r')
            if mm.ndim == 4:
                mm = mm[self._batch_index if self._batch_index is not None else 0]
            self._omega_mmap = mm
            self._mode = 'mmap'
        except OSError as e:
            print(f"  [info] mmap of {omega_path.name} failed ({e}); "
                  f"falling back to per-snapshot reads")
            self._omega_mmap = None
            self._mode = 'fallback'

    @property
    def times(self) -> np.ndarray:
        return self._times

    def _read_snapshot(self, idx: int) -> np.ndarray:
        """Return the idx-th snapshot as a fresh numpy array."""
        if self._mode == 'mmap':
            return np.asarray(self._omega_mmap[idx], dtype=np.float64)
        # Fallback: seek + read just the bytes for this (batch, idx) snapshot
        Ny, Nx = self._spatial_shape
        per_snapshot_bytes = Ny * Nx * self._dtype.itemsize
        # Layout: if 4D with B>1 we need (batch_index * T + idx) * per-snapshot
        if self._batch_index is not None:
            T = self._n_snapshots
            absolute_idx = self._batch_index * T + idx
        else:
            absolute_idx = idx
        offset = self._header_offset + absolute_idx * per_snapshot_bytes
        with open(self._omega_path, 'rb') as f:
            f.seek(offset)
            buf = f.read(per_snapshot_bytes)
        if len(buf) != per_snapshot_bytes:
            raise RuntimeError(f"short read at idx={idx}: "
                               f"got {len(buf)} bytes, expected {per_snapshot_bytes}")
        return np.frombuffer(buf, dtype=self._dtype).reshape(Ny, Nx).astype(np.float64)

    def get_at_time(self, t_target: float, tol: float = 1e-6) -> Tuple[float, np.ndarray]:
        """
        Return (actual_time, omega_field) for the saved snapshot closest to t_target.
        Only that single snapshot is materialized into memory.
        """
        idx = int(np.argmin(np.abs(self._times - t_target)))
        actual_t = float(self._times[idx])
        if abs(actual_t - t_target) > 0.05:   # half a save interval
            print(f"  warning: closest saved time to t={t_target:.3f} is "
                  f"t={actual_t:.3f} (off by {actual_t-t_target:.3f})")
        field = self._read_snapshot(idx)
        return actual_t, field


# --------------------------------------------------------------------------- #
# Pick the three time slices from the convergence CSV
# --------------------------------------------------------------------------- #

def pick_times_from_csv(
    csv_path: Path,
    early_threshold: float = 0.05,
    middle_threshold: float = 0.30,
    late_threshold: float = 0.80,
) -> Tuple[float, float, float]:
    """
    For decaying turbulence we use fixed times keyed to the simulation length T=60:
        t_early  = 10.0   (~1/6 of total simulation time)
        t_middle = 30.0  (mid-simulation; large-scale vortices have merged)
        t_late   = 50.0  (late dissipation regime, near end of T=60)

    The CSV-based auto-detection from the FT version doesn't work cleanly here:
    For reference, the cylinder version had this problem due to impulsive startup;
    so the early/middle thresholds both trigger at t=0.05 and produce duplicate
    rows in the figure. Fixed times are simpler and more interpretable for a
    deterministic shedding flow.

    The CSV path is unused by this function (kept for API compatibility).

    Returns (t_early, t_middle, t_late).
    """
    t_early  = 10.0
    t_middle = 30.0
    t_late   = 50.0
    print(f"selected times (decaying-turb fixed): "
          f"early={t_early:.2f}  middle={t_middle:.2f}  late={t_late:.2f}")
    return t_early, t_middle, t_late


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_field_grid(
    sweep_root: Path,
    times: Tuple[float, float, float],
    out_path: Path,
) -> None:
    r"""
    3 x 4 figure: rows are 3 representative times; columns are
    $\bar\omega_{\Delta t}(t) - \bar\omega_{\rm ref}(t)$ for 4 different
    $\Delta t$ values. Each row uses a shared symmetric colorbar so the
    growth of the error with $\Delta t$ is directly visible across columns.
    """
    selected = [SWEEP[i] for i in SELECTED_INDICES]
    selected_dirs = [sweep_root / sub for _, sub, _ in selected]
    selected_labels = [lbl for _, _, lbl in selected]

    ref_dt, ref_sub, ref_label = SWEEP[REF_INDEX]
    ref_dir = sweep_root / ref_sub

    # Open all handles -- each is mmap (or fallback) so cheap.
    run_handles = [RunHandle(d) for d in selected_dirs]
    ref = RunHandle(ref_dir)

    n_rows = len(times)
    n_cols = len(selected)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 11),
                             gridspec_kw=dict(hspace=0.18, wspace=0.05))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, None]

    for row, t_target in enumerate(times):
        # Read reference snapshot once per row
        t_ref_actual, ref_field = ref.get_at_time(t_target)

        # Compute all four diffs at this time
        diff_fields = []
        actual_times = []
        for h in run_handles:
            t_run_actual, run_field = h.get_at_time(t_target)
            diff_fields.append(run_field - ref_field)
            actual_times.append(t_run_actual)

        # Symmetric colorscale shared across the row so the error magnitudes
        # are directly comparable across dt's.
        vmax_row = float(np.max([np.max(np.abs(d)) for d in diff_fields]))
        if vmax_row == 0.0:
            vmax_row = 1e-12

        for col, (dfield, t_act, lbl) in enumerate(
            zip(diff_fields, actual_times, selected_labels)
        ):
            ax = axes[row, col]
            im = ax.imshow(dfield, origin='lower', cmap='seismic',
                           vmin=-vmax_row, vmax=vmax_row,
                           interpolation='gaussian', aspect='equal')

            # title only on top row: a single common diff label
            if row == 0:
                ax.set_title(
                    r'$\bar\omega_{\Delta t}(t) - \bar\omega_{\rm ref}(t)$',
                    fontsize=10,
                )
            # row label only on leftmost column: target time
            if col == 0:
                ax.set_ylabel(fr'$t \approx {t_act:.2f}$', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

        # one shared colorbar per row, attached to the last column
        cbar = fig.colorbar(im, ax=axes[row, -1], pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(r'$\bar\omega - \bar\omega_{\rm ref}$', fontsize=8)

    # Bottom-row x-label spells out the dt for that column.
    for col, lbl in enumerate(selected_labels):
        axes[-1, col].set_xlabel(lbl, fontsize=10)

    fig.suptitle(
        r'Vorticity Field Differences vs. Reference Run '
        r'(Decaying Turbulence, $N = 256^2$, $L_x = L_y = 2\pi$, $T = 60$, '
        r'Batch #0)',
        fontsize=12,
    )

    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


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
    p.add_argument('--convergence-csv', type=Path, default=None,
                   help='step1_convergence.csv (used to auto-pick times). '
                        'If omitted, falls back to t=1, 5, 25.')
    p.add_argument('--out-dir', type=Path, default=Path.cwd(),
                   help='where to write step2_fields.png')
    p.add_argument('--early-threshold',  type=float, default=0.05,
                   help='median rel_L2 threshold for the "early" time')
    p.add_argument('--middle-threshold', type=float, default=0.30,
                   help='median rel_L2 threshold for the "middle" time')
    p.add_argument('--late-threshold',   type=float, default=0.80,
                   help='median rel_L2 threshold for the "late" time')
    p.add_argument('--t-early',  type=float, default=None,
                   help='manually override the early time')
    p.add_argument('--t-middle', type=float, default=None,
                   help='manually override the middle time')
    p.add_argument('--t-late',   type=float, default=None,
                   help='manually override the late time')
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # default csv path: same dir as out-dir
    csv_path = args.convergence_csv
    if csv_path is None:
        csv_path = args.out_dir / 'step1_convergence_decay.csv'

    # auto-pick times unless overridden
    t_early, t_middle, t_late = pick_times_from_csv(
        csv_path,
        early_threshold=args.early_threshold,
        middle_threshold=args.middle_threshold,
        late_threshold=args.late_threshold,
    )
    if args.t_early is not None:
        t_early = args.t_early
    if args.t_middle is not None:
        t_middle = args.t_middle
    if args.t_late is not None:
        t_late = args.t_late

    plot_field_grid(
        args.sweep_root,
        times=(t_early, t_middle, t_late),
        out_path=args.out_dir / 'step2_fields_decay.png',
    )


if __name__ == '__main__':
    main()