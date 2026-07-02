"""
extract_omega_from_dns_npy.py

The raw solver output DNS.npy is a 5D tensor (B, T, C, Ny, Nx) where:
  C = 0  : vorticity   omega
  C = 1  : streamfunction psi
  C = 2  : velocity u
  C = 3  : velocity v

The dataset.py packaging step downcasts this to float32 when it writes
DNS_FR.npz.  To preserve the simulation's true float64 precision we extract
channel 0 directly from DNS.npy (which IS float64) into a 4D file:

    DNS.npy             (B, T, 4, Ny, Nx)  float64    <-- input
    DNS_FR_omega.npy    (B, T,    Ny, Nx)  float64    <-- output
    DNS_FR_times.npy    (T,)               float64    <-- output

The output names match what build_training_data_fixD_v2.py looks for, so the
build pipeline works unchanged afterward.

Memory cost: streamed batch-by-batch, peak ~ (T*Ny*Nx*8) bytes = ~600 MB
for T=1200, Ny=Nx=256.  Disk cost: ~12.6 GB for the same shape at float64.

Usage:
  python extract_omega_from_dns_npy.py \\
      --run-dir /gdata/.../decaying_turb_dt_sweep_float64/dt_2em5 \\
      [--vorticity-channel 0]              \\
      [--times-source DNS_FR.npz]
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib import format as np_format


def parse_npy_header(path: Path):
    """Return (shape, fortran_order, dtype, header_offset)."""
    with open(path, 'rb') as f:
        major, minor = np_format.read_magic(f)
        if (major, minor) == (1, 0):
            shape, fortran, dtype = np_format.read_array_header_1_0(f)
        elif (major, minor) == (2, 0):
            shape, fortran, dtype = np_format.read_array_header_2_0(f)
        else:
            raise RuntimeError(f'unsupported .npy version {major}.{minor}')
        return shape, fortran, dtype, f.tell()


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', type=Path, required=True,
                   help='directory containing DNS.npy')
    p.add_argument('--dns-name', type=str, default='DNS.npy',
                   help='filename of the raw solver output (default DNS.npy)')
    p.add_argument('--out-name-omega', type=str, default='DNS_FR_omega.npy',
                   help='output omega filename (default DNS_FR_omega.npy)')
    p.add_argument('--out-name-times', type=str, default='DNS_FR_times.npy',
                   help='output times filename (default DNS_FR_times.npy)')
    p.add_argument('--vorticity-channel', type=int, default=0,
                   help='which channel of DNS.npy is omega (default 0)')
    p.add_argument('--times-source', type=str, default='DNS_FR.npz',
                   help='where to read times from: a .npz with a "times" '
                        'key, or a .npy file of times directly (default '
                        'DNS_FR.npz)')
    p.add_argument('--force', action='store_true',
                   help='overwrite output file if it already exists')
    args = p.parse_args()

    dns_path     = args.run_dir / args.dns_name
    out_omega    = args.run_dir / args.out_name_omega
    out_times    = args.run_dir / args.out_name_times

    if not dns_path.exists():
        sys.exit(f'ERROR: {dns_path} not found')
    if out_omega.exists() and not args.force:
        sys.exit(f'ERROR: {out_omega} exists; pass --force to overwrite')

    # ---- Parse DNS.npy header ---- #
    shape, fortran, dtype, hdr_off = parse_npy_header(dns_path)
    print(f'[extract] DNS source: {dns_path}')
    print(f'[extract]   shape: {shape}')
    print(f'[extract]   dtype: {dtype}')
    if fortran:
        sys.exit('ERROR: Fortran-order .npy not supported')
    if len(shape) != 5:
        sys.exit(f'ERROR: expected 5D (B,T,C,Ny,Nx); got shape {shape}')
    B, T, C, Ny, Nx = shape
    if args.vorticity_channel >= C:
        sys.exit(f'ERROR: --vorticity-channel={args.vorticity_channel} >= '
                 f'channels={C}')
    c = args.vorticity_channel
    print(f'[extract]   B={B}, T={T}, C={C} (using ch {c}), Ny={Ny}, Nx={Nx}')

    # ---- Read times ---- #
    times_source = args.run_dir / args.times_source
    if not times_source.exists():
        sys.exit(f'ERROR: times source {times_source} not found; pass '
                 f'--times-source explicitly to skip this step.')
    if times_source.suffix == '.npz':
        with np.load(times_source) as zf:
            if 'times' not in zf.files:
                sys.exit(f'ERROR: no "times" key in {times_source}; keys={zf.files}')
            times = np.asarray(zf['times'])
    else:
        times = np.load(times_source)
    print(f'[extract] times: shape={times.shape}, dtype={times.dtype}, '
          f't[0]={float(times[0]):.4f}, t[-1]={float(times[-1]):.4f}')

    # ---- Write the output 4D array, streamed batch-by-batch ---- #
    print(f'[extract] writing {out_omega} '
          f'({B*T*Ny*Nx*8/1e9:.2f} GB at float64) ...')
    out_shape = (B, T, Ny, Nx)
    # Allocate the .npy file with the correct header, then memory-map for write.
    # This avoids holding the full array in RAM.
    np.lib.format.open_memmap(
        str(out_omega), mode='w+', dtype=np.float64, shape=out_shape)
    out_mm = np.load(out_omega, mmap_mode='r+')

    elem_size = dtype.itemsize
    # Per (b, t, c) slice in source: contiguous Ny*Nx*elem_size bytes
    # Source row-major layout: (b, t, c, y, x).  Offset of slice (b, t, c, :, :):
    #   hdr_off + ((b*T + t)*C + c) * Ny*Nx*elem_size
    Nyx_bytes = Ny * Nx * elem_size

    t_start = time.time()
    with open(dns_path, 'rb') as f:
        for b in range(B):
            t0 = time.time()
            # Read T full snapshots of channel c into RAM as a single chunk-ish
            # operation.  Since the channels are interleaved at the inner-most
            # position before (Ny,Nx), the slice for (b, :, c, :, :) is not
            # contiguous on disk -- we need T separate reads.  Slow but
            # bounded-memory.
            for t in range(T):
                offset = hdr_off + ((b * T + t) * C + c) * Nyx_bytes
                f.seek(offset)
                buf = f.read(Nyx_bytes)
                if len(buf) != Nyx_bytes:
                    sys.exit(f'short read at (b={b}, t={t}): '
                             f'{len(buf)}/{Nyx_bytes}')
                snap = np.frombuffer(buf, dtype=dtype).reshape(Ny, Nx)
                # Cast up (in case source was somehow float32; for true
                # float64 source this is a no-op view).
                out_mm[b, t] = snap.astype(np.float64, copy=False)
            print(f'[extract]   batch {b+1}/{B} done in {time.time()-t0:.1f}s '
                  f'(total {time.time()-t_start:.0f}s)')
    out_mm.flush()
    del out_mm
    print(f'[extract] saved {out_omega}')

    # ---- Save times ---- #
    np.save(out_times, np.asarray(times, dtype=np.float64))
    print(f'[extract] saved {out_times}')

    # ---- Sanity check ---- #
    verify = np.load(out_omega, mmap_mode='r')
    print(f'[extract] verify: shape={verify.shape}, dtype={verify.dtype}, '
          f'rms[batch0,t=0]={np.sqrt((verify[0,0]**2).mean()):.4e}')

    print('[extract] DONE.')


if __name__ == '__main__':
    main()
