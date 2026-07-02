"""
pack_dataset_mmap.py

One-time repack of a per-sample .npz dataset (built by build_training_data*.py)
into two CONTIGUOUS, UNCOMPRESSED, float32 memmap arrays:

    <root>/<out-subdir>/inputs.npy    shape (N, C_in,  Ny, Nx)  float32
    <root>/<out-subdir>/targets.npy   shape (N, C_out, Ny, Nx)  float32
    <root>/<out-subdir>/pack_meta.json
    <root>/<out-subdir>/split.npz     (copied from root)

Why: the per-sample files are np.savez_compressed archives holding ~27 fields
each. Every epoch the loader opens N zip files and DEFLATE-inflates the few keys
it needs -- pure I/O + CPU, and it does not fit in page cache at 512^2, so it is
re-read from NFS every epoch. The packed memmaps remove the zip-opens and the
decompression, halve the bytes (float64 -> float32), and let the OS cache the
contiguous file. float32 is sufficient for TRAINING: the float64 in the build
was only to survive the e_total cancellation at build time; the N-derivative
targets here are O(1)..O(1e4) and lose nothing in float32.

The packed dir is self-describing: dataset.py reads pack_meta.json to map each
requested field name to its column, so field ORDER here does not have to match
the order you later pass to --input-fields / --target-fields.

Usage:
    python pack_dataset_mmap.py \\
        --root-dir .../forced_turbulence_dT_1em3 \\
        --input-fields  omega_0 omega_m1 omega_m2 psi_0 psi_m1 psi_m2 \\
        --target-fields N_dot_0_anal N_ddot_0_anal N_3dot_0_anal

Run it once (it reads the whole dataset a single time -- a few tens of minutes
as a CPU job). Training then auto-detects <root>/packed/ and uses it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-dir', type=Path, required=True,
                   help='dataset root (contains manifest.json, split.npz, samples/)')
    p.add_argument('--input-fields', type=str, nargs='+', required=True,
                   help='fields to pack as input channels (in this order)')
    p.add_argument('--target-fields', type=str, nargs='+', required=True,
                   help='fields to pack as target channels (in this order)')
    p.add_argument('--out-subdir', type=str, default='packed',
                   help='subdir under root to write the memmaps (default: packed)')
    p.add_argument('--flush-every', type=int, default=200,
                   help='flush memmaps to disk every N samples (default 200)')
    p.add_argument('--overwrite', action='store_true',
                   help='overwrite an existing packed dir')
    args = p.parse_args()

    root = args.root_dir
    with open(root / 'manifest.json') as f:
        manifest = json.load(f)
    N = int(manifest['n_total'])
    Ny, Nx = int(manifest['Ny']), int(manifest['Nx'])
    C_in, C_out = len(args.input_fields), len(args.target_fields)

    out_dir = root / args.out_subdir
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already exists; pass --overwrite to replace it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    in_bytes = N * C_in * Ny * Nx * 4
    tg_bytes = N * C_out * Ny * Nx * 4
    print(f"[pack] N={N}  grid={Ny}x{Nx}  C_in={C_in}  C_out={C_out}")
    print(f"[pack] inputs.npy  ~ {in_bytes/1024**3:.1f} GiB")
    print(f"[pack] targets.npy ~ {tg_bytes/1024**3:.1f} GiB")
    print(f"[pack] input fields:  {args.input_fields}")
    print(f"[pack] target fields: {args.target_fields}")

    inp = open_memmap(out_dir / 'inputs.npy', mode='w+', dtype=np.float32,
                      shape=(N, C_in, Ny, Nx))
    tgt = open_memmap(out_dir / 'targets.npy', mode='w+', dtype=np.float32,
                      shape=(N, C_out, Ny, Nx))

    samples_dir = root / 'samples'
    t0 = time.time()
    n_missing = 0
    for i in range(N):
        spath = samples_dir / f'sample_{i:06d}.npz'
        if not spath.exists():
            n_missing += 1
            if n_missing <= 10:
                print(f"[pack] WARN: missing {spath.name}; row {i} left zero.")
            continue
        with np.load(spath) as zf:               # lazy: only requested keys inflate
            for c, fld in enumerate(args.input_fields):
                inp[i, c] = np.asarray(zf[fld], dtype=np.float32)
            for c, fld in enumerate(args.target_fields):
                tgt[i, c] = np.asarray(zf[fld], dtype=np.float32)
        if (i + 1) % args.flush_every == 0 or (i + 1) == N:
            inp.flush(); tgt.flush()
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-9)
            eta = (N - (i + 1)) / max(rate, 1e-9)
            print(f"[pack] {i+1:6d}/{N}  {rate:5.1f} samp/s  "
                  f"elapsed {dt/60:5.1f} min  ETA {eta/60:5.1f} min")

    inp.flush(); tgt.flush()
    del inp, tgt

    # Copy the split so the packed dir is self-contained.
    if (root / 'split.npz').exists():
        shutil.copy2(root / 'split.npz', out_dir / 'split.npz')

    meta = dict(
        source_root=str(root),
        N=N, Ny=Ny, Nx=Nx,
        input_fields=list(args.input_fields),
        target_fields=list(args.target_fields),
        dtype='float32',
        n_missing=int(n_missing),
        Lx=float(manifest.get('Lx', 0.0)),
        Ly=float(manifest.get('Ly', 0.0)),
        Delta_T=float(manifest.get('Delta_T', 0.0)),
    )
    with open(out_dir / 'pack_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    total = (time.time() - t0) / 60
    print(f"[pack] done in {total:.1f} min  ({n_missing} missing samples)")
    print(f"[pack] wrote {out_dir}/inputs.npy, targets.npy, pack_meta.json, split.npz")


if __name__ == '__main__':
    main()
