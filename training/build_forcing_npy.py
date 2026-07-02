#!/usr/bin/env python
r"""
build_forcing_npy.py  --  rebuild an FRC member's static forcing field F(x,y) from
its sliced sweep_dT_* manifest, byte-matching the builder/slicer construction, and
save it as <member_dir>/forcing.npy for closure_error_propagation.py --forcing.

The forcing enters the N-derivatives through the tendency (omega_dot = L omega + N,
N = -J + F), so even with F_dot = 0 the field F is NOT ignorable for exact
amplification norms -- hence this helper (vs the script's F=0 fallback).

Construction is verbatim from build_training_data_mmap.py / slice_deriv_from_deep.py:
    F(x,y) = A cos(B x) + D cos(E y),   x=linspace(0,Lx,Nx), y=linspace(0,Ly,Ny)
(endpoint=True linspace, matching the target build -- consistency over periodicity).
Guards: C or F nonzero => time-dependent forcing, incompatible with static closure.

Usage:
    python build_forcing_npy.py data/ensemble_N5_7lag_sliced/FRC-combo/sweep_dT_5em3
    # writes .../sweep_dT_5em3/forcing.npy   (shape (Ny, Nx), float64)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) != 2:
        print("usage: python build_forcing_npy.py MEMBER_SWEEP_DIR")
        raise SystemExit(1)
    mdir = Path(sys.argv[1])
    man = json.loads((mdir / 'manifest.json').read_text())

    fc = man.get('forcing', None)
    Nx, Ny = int(man['Nx']), int(man['Ny'])
    Lx, Ly = float(man['Lx']), float(man['Ly'])

    if not isinstance(fc, dict):
        print(f"[forcing] {mdir.name}: manifest has no forcing block -> writing F=0 "
              f"(unforced member, or forcing not recorded).")
        F = np.zeros((Ny, Nx), dtype=np.float64)
    else:
        A = float(fc.get('A', 0.0)); B = float(fc.get('B', 0.0))
        C = float(fc.get('C', 0.0)); D = float(fc.get('D', 0.0))
        E = float(fc.get('E', 0.0)); Ff = float(fc.get('F', 0.0))
        if C != 0.0 or Ff != 0.0:
            raise SystemExit(f"[forcing] {mdir.name}: C={C} or F={Ff} nonzero "
                             f"(time-dependent forcing) is incompatible with the "
                             f"static-forcing closure. Refusing to fake it.")
        x = np.linspace(0.0, Lx, Nx, dtype=np.float64)
        y = np.linspace(0.0, Ly, Ny, dtype=np.float64)
        F = A * np.cos(B * x)[None, :] + D * np.cos(E * y)[:, None]   # (Ny, Nx)
        print(f"[forcing] {mdir.name}: F = {A}*cos({B}x) + {D}*cos({E}y)  "
              f"grid {Ny}x{Nx}  |F|_rms={np.sqrt((F**2).mean()):.4e}")

    out = mdir / 'forcing.npy'
    np.save(out, F)
    print(f"[forcing] wrote {out}")


if __name__ == '__main__':
    main()
