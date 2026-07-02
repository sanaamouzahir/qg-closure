#!/usr/bin/env python
"""
extract_restart_ic.py
=====================

Extract the last time-snapshot of a multi-batch DNS run as an IC file
that can be loaded via ic.py's `from_file` option to restart/continue
the simulation.

Usage:
    python extract_restart_ic.py \
        --source /gdata/.../dt_2em5/DNS_FR_omega.npy \
        --out    /gdata/.../dt_2em5/restart_ic_t60.npy

Reads the source npy (4D: B, T, Ny, Nx) and writes a 3D array
(B, Ny, Nx) = source[:, -1, :, :].  ic.py recognizes (B, Ny, Nx)
shape and broadcasts to n_batch=B in the new run.
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True,
                   help='DNS_FR_omega.npy (4D B,T,Ny,Nx or 3D T,Ny,Nx)')
    p.add_argument('--out', type=Path, required=True,
                   help='output .npy with shape (B, Ny, Nx) or (1, Ny, Nx)')
    p.add_argument('--time-index', type=int, default=-1,
                   help='which time index to extract (default -1 = last)')
    args = p.parse_args()

    arr = np.load(args.source, mmap_mode='r')
    print(f"source shape: {arr.shape}, dtype: {arr.dtype}")

    if arr.ndim == 4:           # (B, T, Ny, Nx)
        last = np.asarray(arr[:, args.time_index, :, :])
        print(f"extracted (B, Ny, Nx) = {last.shape}")
    elif arr.ndim == 3:         # (T, Ny, Nx)
        last = np.asarray(arr[args.time_index, :, :])[None]
        print(f"extracted (1, Ny, Nx) = {last.shape}")
    else:
        raise RuntimeError(f"unexpected ndim={arr.ndim}")

    # Save as float64 to preserve precision
    last = last.astype(np.float64)
    np.save(args.out, last)
    print(f"wrote {args.out}  ({last.shape}, {last.dtype})")
    print(f"rms |omega|: {float(np.sqrt(np.mean(last**2))):.4e}")


if __name__ == '__main__':
    main()
