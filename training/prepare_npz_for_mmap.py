"""
prepare_npz_for_mmap.py

Convert DNS_FR.npz files (which can't be memory-mapped because they're zip
archives) into raw .npy files in the same directory:

    DNS_FR.npz   -->  DNS_FR_omega.npy   (the omega_FR array)
                      DNS_FR_times.npy   (the times array)

The .npy format supports np.load(..., mmap_mode='r'), which is what makes
the streaming convergence analysis use bounded memory.

This is a one-time preprocessing step:
  - Disk cost: roughly doubles the storage for each run during the conversion
    (you can delete the original .npz after if you trust the .npy copy).
  - Time cost: a few minutes total at 1024^2 x 1000 timesteps -- it's just
    unzipping a ZIP file and writing the arrays out.

Usage:
  python prepare_npz_for_mmap.py --sweep-root /path/to/forced_turbulence_dt_sweep/beta_0p0
  python prepare_npz_for_mmap.py --sweep-root ... --skip-existing  # default; only convert if .npy missing
  python prepare_npz_for_mmap.py --sweep-root ... --force          # re-convert even if .npy exists
  python prepare_npz_for_mmap.py --sweep-root ... --delete-npz     # remove .npz after successful conversion
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np


SWEEP_SUBDIRS = ['dt_1em3', 'dt_2em3', 'dt_5em4', 'dt_2p5em4', 'dt_1p25em4', 'dt_2em5', 'dt_1em5']


class _NpyFileReader:
    """
    Read slabs of an .npy file via direct seek+read, never mmap'ing the whole
    file. Cluster login/compute nodes with limited virtual address space may
    refuse the mmap reservation (OSError: cannot allocate memory) for files
    larger than a few GB; this class avoids that by reading only one slab at
    a time.

    Supports the layouts we encounter:
      * (T, Ny, Nx)         -- 3D
      * (B, T, Ny, Nx)      -- 4D (multi-batch)
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        from numpy.lib import format as np_format
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
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

    def read_batch(self, b: int) -> np.ndarray:
        """Return batch b as a (T, Ny, Nx) float32 ndarray. 4D arrays only."""
        if self.ndim != 4:
            raise RuntimeError("read_batch requires 4D layout")
        B, T, Ny, Nx = self.shape
        if not (0 <= b < B):
            raise IndexError(f"batch index {b} out of [0, {B})")
        per_batch_bytes = T * Ny * Nx * self.dtype.itemsize
        offset = self.header_offset + b * per_batch_bytes
        with open(self.path, 'rb') as f:
            f.seek(offset)
            buf = f.read(per_batch_bytes)
        if len(buf) != per_batch_bytes:
            raise RuntimeError(f"short read at batch {b}: got {len(buf)} of {per_batch_bytes}")
        return np.frombuffer(buf, dtype=self.dtype).reshape(T, Ny, Nx).copy()

    def read_all(self) -> np.ndarray:
        """For 3D arrays only -- read the whole array into memory."""
        if self.ndim != 3:
            raise RuntimeError("read_all here is only for 3D arrays")
        T, Ny, Nx = self.shape
        per_array_bytes = T * Ny * Nx * self.dtype.itemsize
        with open(self.path, 'rb') as f:
            f.seek(self.header_offset)
            buf = f.read(per_array_bytes)
        return np.frombuffer(buf, dtype=self.dtype).reshape(T, Ny, Nx).copy()


def _compute_ensemble_z_e_streaming(
    reader: '_NpyFileReader',
    Lx: float,
    Ly: float,
    time_chunk: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ensemble-averaged Z(t) and E(t) by reading one batch at a time
    and processing time in chunks of `time_chunk` frames. Peak memory per
    chunk is ~7 * time_chunk * Ny * Nx * 16 bytes (complex128 working set
    for FFT). For Ny=Nx=256, time_chunk=64, that's ~470 MB.
    """
    if reader.ndim == 3:
        T, Ny, Nx = reader.shape
        B = 1
    elif reader.ndim == 4:
        B, T, Ny, Nx = reader.shape
    else:
        raise RuntimeError(f"unsupported ndim={reader.ndim}")

    # Spectral inv-Laplacian weights, computed once
    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    inv_K2 = 1.0 / K2
    inv_K2[0, 0] = 0.0

    Z_acc = np.zeros(T, dtype=np.float64)
    E_acc = np.zeros(T, dtype=np.float64)

    for b in range(B):
        if reader.ndim == 4:
            slab = reader.read_batch(b)        # (T, Ny, Nx) float32 on disk
        else:
            slab = reader.read_all()
        # slab is float32 on disk; we keep it that way to halve memory
        # versus float64. The FFT promotes to complex128 internally either
        # way, but the slab itself stays small.

        for c0 in range(0, T, time_chunk):
            c1 = min(c0 + time_chunk, T)
            chunk = slab[c0:c1].astype(np.float64, copy=False)  # (chunk, Ny, Nx)
            # Z = 0.5 * <omega^2>
            Z_acc[c0:c1] += 0.5 * np.mean(chunk**2, axis=(-1, -2))
            # E = 0.5 * <|grad psi|^2>; psi_hat = -omega_hat / k^2
            omega_hat = np.fft.fft2(chunk)
            psi_hat   = -omega_hat * inv_K2
            u_hat     = -1j * KY * psi_hat
            v_hat     = +1j * KX * psi_hat
            # Free omega_hat and psi_hat before allocating u, v
            del omega_hat, psi_hat
            u = np.fft.ifft2(u_hat).real
            del u_hat
            v = np.fft.ifft2(v_hat).real
            del v_hat
            E_acc[c0:c1] += 0.5 * np.mean(u**2 + v**2, axis=(-1, -2))
            del u, v, chunk
        del slab

    return Z_acc / B, E_acc / B


def _compute_ensemble_z_e(
    omega_4d: np.ndarray,
    Lx: float,
    Ly: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backwards-compatible wrapper for in-memory testing. The streaming
    workhorse is _compute_ensemble_z_e_streaming above.
    """
    B, T, Ny, Nx = omega_4d.shape

    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    inv_K2 = 1.0 / K2
    inv_K2[0, 0] = 0.0

    Z_acc = np.zeros(T, dtype=np.float64)
    E_acc = np.zeros(T, dtype=np.float64)

    for b in range(B):
        slab = np.asarray(omega_4d[b], dtype=np.float64)
        Z_acc += 0.5 * np.mean(slab**2, axis=(-1, -2))
        omega_hat = np.fft.fft2(slab)
        psi_hat = -omega_hat * inv_K2
        u_hat = -1j * KY * psi_hat
        v_hat = +1j * KX * psi_hat
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        E_acc += 0.5 * np.mean(u**2 + v**2, axis=(-1, -2))
        del slab, omega_hat, psi_hat, u_hat, v_hat, u, v

    return Z_acc / B, E_acc / B


def convert_one(npz_path: Path, force: bool, delete_npz: bool,
                batch_index: int = 0,
                Lx: float = 2 * np.pi,
                Ly: float = 2 * np.pi) -> None:
    """
    Convert one DNS_FR.npz -> DNS_FR_omega.npy, DNS_FR_times.npy,
                              DNS_FR_Z_ens.npy, DNS_FR_E_ens.npy.

    The npz is a multi-batch dataset of shape (B, T, Ny, Nx). To avoid
    decompressing the entire (often multi-GB) array into RAM, we:

      1. Stream the inner omega_FR.npy out of the npz zip to a temp file.
      2. mmap the temp file.
      3. Compute ensemble-averaged Z(t) and E(t) batch-by-batch, then
         write DNS_FR_Z_ens.npy and DNS_FR_E_ens.npy (small, T floats).
      4. Materialize batch_index only and write DNS_FR_omega.npy.
      5. Delete the temp file.

    batch_index: which IC realization to save in DNS_FR_omega.npy
                 (default 0; ensemble Z and E always use ALL batches).
    Lx, Ly: physical domain size, needed only for the energy computation
            (default 2p for the decaying-turbulence YAML).
    """
    omega_path = npz_path.parent / 'DNS_FR_omega.npy'
    times_path = npz_path.parent / 'DNS_FR_times.npy'
    z_ens_path = npz_path.parent / 'DNS_FR_Z_ens.npy'
    e_ens_path = npz_path.parent / 'DNS_FR_E_ens.npy'

    if not force and all(p.exists() for p in
                          [omega_path, times_path, z_ens_path, e_ens_path]):
        print(f"  skip {npz_path.parent.name} (already converted)")
        return

    if not npz_path.exists():
        print(f"  WARNING: {npz_path} does not exist, skipping")
        return

    print(f"  converting {npz_path.parent.name}/DNS_FR.npz ...")
    t0 = time.time()

    import zipfile
    import tempfile

    with zipfile.ZipFile(npz_path) as zf:
        names = zf.namelist()

        times_member = next((n for n in names if n.startswith('times')), None)
        if times_member is None:
            print(f"  ERROR: no 'times' member in {npz_path}")
            return
        with zf.open(times_member) as f:
            times = np.lib.format.read_array(f, allow_pickle=False)

        omega_member = next((n for n in names if n.startswith('omega_FR')), None)
        if omega_member is None:
            print(f"  ERROR: no 'omega_FR' member in {npz_path}")
            return

        # Stream the inner .npy member to a temp file
        with tempfile.NamedTemporaryFile(
            dir=npz_path.parent, prefix='.tmp_omega_', suffix='.npy', delete=False
        ) as tf:
            tmp_npy = Path(tf.name)
            with zf.open(omega_member) as src:
                CHUNK = 64 * 1024 * 1024
                while True:
                    buf = src.read(CHUNK)
                    if not buf:
                        break
                    tf.write(buf)
        print(f"    extracted to temp {tmp_npy.name} in {time.time()-t0:.1f}s")

    try:
        reader = _NpyFileReader(tmp_npy)
        print(f"    parsed shape={reader.shape}, dtype={reader.dtype} (no mmap, streaming)")

        # ---- ensemble Z(t), E(t) across ALL batches ---- #
        if reader.ndim == 4:
            B = reader.shape[0]
            print(f"    computing ensemble Z(t), E(t) across B={B} batches "
                  f"(Lx=Ly={Lx:.4f}) ...")
            t1 = time.time()
            Z_ens, E_ens = _compute_ensemble_z_e_streaming(reader, Lx, Ly)
            print(f"      done in {time.time()-t1:.1f}s")
        else:
            print(f"    computing Z(t), E(t) (single-batch 3D file) ...")
            Z_ens, E_ens = _compute_ensemble_z_e_streaming(reader, Lx, Ly)

        np.save(z_ens_path, Z_ens.astype(np.float64))
        np.save(e_ens_path, E_ens.astype(np.float64))
        print(f"    saved {z_ens_path.name}, {e_ens_path.name} "
              f"(shapes {Z_ens.shape}, {E_ens.shape})")

        # ---- batch-{batch_index} omega slab ---- #
        if reader.ndim == 4:
            B = reader.shape[0]
            if not (0 <= batch_index < B):
                raise IndexError(f"batch_index={batch_index} out of [0, {B})")
            if B == 1:
                print(f"    B=1; saving the only batch")
            else:
                print(f"    B={B}; saving batch {batch_index} for "
                      f"trajectory comparisons (use --batch-index to choose another)")
            omega_slice = reader.read_batch(batch_index)  # (T, Ny, Nx)
        elif reader.ndim == 3:
            omega_slice = reader.read_all()
        else:
            raise RuntimeError(f"unexpected omega ndim={reader.ndim}")

        print(f"    saving {omega_path.name} "
              f"({omega_slice.nbytes / 1e9:.3f} GB, shape={omega_slice.shape}) ...")
        t1 = time.time()
        np.save(omega_path, omega_slice)
        print(f"      done in {time.time()-t1:.1f}s")

        print(f"    saving {times_path.name} ({times.nbytes} bytes) ...")
        np.save(times_path, times)

        del omega_slice
    finally:
        try:
            tmp_npy.unlink()
        except OSError:
            pass

    if delete_npz:
        size_mb = npz_path.stat().st_size / 1e6
        npz_path.unlink()
        print(f"    deleted original {npz_path.name} ({size_mb:.0f} MB freed)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--sweep-root', type=Path, required=True,
                   help='root containing dt_1em3/, dt_5em4/, ... subdirectories')
    p.add_argument('--force', action='store_true',
                   help='re-convert even if .npy files already exist')
    p.add_argument('--delete-npz', action='store_true',
                   help='delete the original .npz after successful conversion')
    p.add_argument('--batch-index', type=int, default=0,
                   help='which IC batch (B-axis index) to extract from each '
                        'multi-batch npz (default 0; decaying-turbulence YAML '
                        'configures n_batch=20)')
    p.add_argument('--Lx', type=float, default=2 * np.pi,
                   help='physical domain size in x for energy computation '
                        '(default 2*pi for decaying turbulence)')
    p.add_argument('--Ly', type=float, default=2 * np.pi,
                   help='physical domain size in y for energy computation '
                        '(default 2*pi for decaying turbulence)')
    args = p.parse_args()

    if not args.sweep_root.is_dir():
        raise SystemExit(f"sweep root not found: {args.sweep_root}")

    print(f"converting sweeps under {args.sweep_root}")
    if args.force:
        print("  (--force: re-converting even existing .npy files)")
    if args.delete_npz:
        print("  (--delete-npz: original .npz will be deleted after success)")
    print(f"  (--batch-index={args.batch_index}: saving only batch "
          f"{args.batch_index} from each multi-batch npz)")
    print(f"  (--Lx={args.Lx:.4f}, --Ly={args.Ly:.4f}: domain size for E(t))")
    print()

    found_any = False
    for subdir in SWEEP_SUBDIRS:
        npz_path = args.sweep_root / subdir / 'DNS_FR.npz'
        if not npz_path.exists():
            # also check if it's already been converted+deleted
            omega_path = npz_path.parent / 'DNS_FR_omega.npy'
            if omega_path.exists():
                print(f"  {subdir}: only .npy exists (already converted, .npz removed)")
                found_any = True
            else:
                print(f"  {subdir}: no DNS_FR.npz found, skipping")
            continue
        convert_one(npz_path, force=args.force, delete_npz=args.delete_npz,
                    batch_index=args.batch_index,
                    Lx=args.Lx, Ly=args.Ly)
        found_any = True

    if not found_any:
        raise SystemExit("no runs found to convert")

    print("\nconversion done.")


if __name__ == '__main__':
    main()