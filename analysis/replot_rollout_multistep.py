"""
replot_rollout_multistep.py
============================

Read the npz dumped by rollout_multistep_comparison.py and remake the figure
WITHOUT re-running the rollout.  Saves a lot of time when iterating on
matplotlib labels, layout, colorbars, slope reference lines, etc.

Usage
-----
  python replot_rollout_multistep.py \
      --npz <path>/rollout_multistep_<ic_tag>.npz \
      --out <path>/rollout_multistep_<ic_tag>.png \
      [--snapshot-fracs "0,0.25,0.5,1.0"] \
      [--model-name bilinear_closure]

If --out is omitted, writes alongside the input npz with the same stem and a
.png extension.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

# Reuse render_figure from the main script (must live in the same directory).
sys.path.insert(0, str(Path(__file__).parent))
from rollout_multistep_comparison import render_figure


def main():
    p = argparse.ArgumentParser(
        description="Re-render the rollout figure from a saved npz.")
    p.add_argument('--npz',  type=Path, required=True,
                   help='path to rollout_multistep_*.npz')
    p.add_argument('--out',  type=Path, default=None,
                   help='output figure path (default: <npz>.png)')
    p.add_argument('--snapshot-fracs', type=str, default=None,
                   help='comma-separated fractions for snapshot rows '
                        '(default: use the values saved in the npz)')
    p.add_argument('--model-name', type=str, default='bilinear_closure',
                   help='model name for the figure title')
    args = p.parse_args()

    if not args.npz.exists():
        sys.exit(f"npz not found: {args.npz}")

    out_path = args.out or args.npz.with_suffix('.png')
    print(f"[replot] reading: {args.npz}")
    z = np.load(args.npz)
    print(f"[replot] arrays: {sorted(z.files)}")

    # Cast back to float64 for the plotting code (npz was saved float32 for the
    # field histories; spectra and error curves were already float64).
    fine_hist = z['fine_hist'].astype(np.float64)
    bare_hist = z['bare_hist'].astype(np.float64)
    clos_hist = z['clos_hist'].astype(np.float64)
    rel_bare  = z['rel_bare'].astype(np.float64)
    rel_clos  = z['rel_clos'].astype(np.float64)
    abs_bare  = z['abs_bare'].astype(np.float64)
    abs_clos  = z['abs_clos'].astype(np.float64)
    fine_rms  = z['fine_rms'].astype(np.float64)
    bare_rms  = z['bare_rms'].astype(np.float64)
    clos_rms  = z['clos_rms'].astype(np.float64)
    fine_Z    = z['fine_Z'].astype(np.float64)
    bare_Z    = z['bare_Z'].astype(np.float64)
    clos_Z    = z['clos_Z'].astype(np.float64)

    Delta_T = float(z['Delta_T'])
    K       = int(z['K'])
    Lx      = float(z['Lx'])
    Ly      = float(z['Ly'])

    if args.snapshot_fracs is not None:
        # User wants custom snapshot fractions.  Use the dense step count
        # (rel_bare has shape (n_steps + 1,)).
        n_steps = rel_bare.shape[0] - 1
        fracs = [float(s) for s in args.snapshot_fracs.split(',')]
        raw_snap = [int(round(f * n_steps)) for f in fracs]
        # Round each requested snapshot to the nearest retained step
        retained = z['retained_steps']
        snapshot_steps = sorted({
            int(retained[np.argmin(np.abs(retained - s))]) for s in raw_snap
        })
    else:
        snapshot_steps = z['snapshot_steps'].tolist()

    retained_steps = z['retained_steps']
    print(f"[replot] Delta_T={Delta_T}  K={K}  Lx={Lx}  Ly={Ly}")
    print(f"[replot] snapshot steps: {snapshot_steps}")
    print(f"[replot] retained frames: {len(retained_steps)}  "
          f"dense scalar length: {rel_bare.shape[0]}")
    print(f"[replot] field history shape: {fine_hist.shape}")

    render_figure(fine_hist, bare_hist, clos_hist, retained_steps,
                  rel_bare, rel_clos, abs_bare, abs_clos,
                  fine_rms, bare_rms, clos_rms,
                  fine_Z, bare_Z, clos_Z,
                  snapshot_steps, Delta_T, K, Lx, Ly, args.model_name,
                  out_path)
    print(f"[replot] wrote {out_path}")


if __name__ == '__main__':
    main()
