#!/usr/bin/env python
r"""spectral_regrid.py -- exact spectral grid transfer for restart ICs (SGS branch).

Charter S5.1.2: convergence-tier runs share ONE developed-flow IC, transferred
across grids spectrally -- 2048 -> 1024 by spectral truncation, 2048 -> 4096 by
spectral zero-padding. Operates on the restart-IC format of
extract_restart_ic.py: real omega .npy, shape (B, Ny, Nx) or (Ny, Nx), float64.
Isotropic square grids only (Ny == Nx), matching the domain rule of this repo.

Convention (documented for the convergence report):
  rfft2 layout; modes with |k| < n_keep/2 are copied, the Nyquist row/column
  (k = n_keep/2) is ZEROED on both truncation and padding. This keeps the
  operation self-adjoint and the round-trip N -> 2N -> N an exact identity on
  the non-Nyquist modes. Solver fields are 2/3-rule dealiased, so their Nyquist
  content is already zero and the round-trip recovers the field to machine
  precision -- the script measures and prints exactly that number (charter
  requires it in the report), plus the energy fraction actually dropped by a
  truncation. Amplitudes scale by (N_out/N_in)^2 to compensate numpy's
  unnormalized FFT pair.

Usage (from training/):
    python spectral_regrid.py --source restart_ic_t30.npy --out ic_1024.npy --N-out 1024
    python spectral_regrid.py --source restart_ic_t30.npy --out ic_4096.npy --N-out 4096
    python spectral_regrid.py --source restart_ic_t30.npy --self-test   # round-trip number
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def regrid(field: np.ndarray, n_out: int) -> np.ndarray:
    """Spectral truncation/zero-padding of real (B, N, N) -> (B, n_out, n_out)."""
    assert field.ndim == 3 and field.shape[1] == field.shape[2], \
        f'expected (B, N, N), got {field.shape} -- anisotropic grids unsupported'
    n_in = field.shape[1]
    if n_in == n_out:
        f = np.fft.rfft2(field)
        h = n_in // 2
        f[:, h, :] = 0.0
        f[:, :, h] = 0.0
        return np.fft.irfft2(f, s=(n_out, n_out))
    f_in = np.fft.rfft2(field)
    f_out = np.zeros((field.shape[0], n_out, n_out // 2 + 1), dtype=np.complex128)
    h = min(n_in, n_out) // 2               # keep |k| < h, zero Nyquist k = h
    f_out[:, :h, :h] = f_in[:, :h, :h]              # ky in [0, h)
    f_out[:, -(h - 1):, :h] = f_in[:, -(h - 1):, :h]  # ky in [-(h-1), -1]
    f_out *= (n_out / n_in) ** 2
    return np.fft.irfft2(f_out, s=(n_out, n_out))


def dropped_energy_fraction(field: np.ndarray, n_out: int) -> float:
    """Fraction of sum|F_k|^2 (rfft-weighted) not representable at n_out (incl. Nyquist zeroing)."""
    f_in = np.fft.rfft2(field)
    w = np.ones_like(f_in.real)
    w[..., 1:field.shape[2] // 2] = 2.0     # rfft half-plane double-count weight
    tot = float((w * np.abs(f_in) ** 2).sum())
    kept = np.zeros_like(f_in)
    h = min(field.shape[1], n_out) // 2
    kept[:, :h, :h] = f_in[:, :h, :h]
    kept[:, -(h - 1):, :h] = f_in[:, -(h - 1):, :h]
    return 1.0 - float((w * np.abs(kept) ** 2).sum()) / tot


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True,
                   help='restart IC .npy, (B, N, N) or (N, N), real, float64')
    p.add_argument('--out', type=Path, help='output .npy (required unless --self-test)')
    p.add_argument('--N-out', type=int, help='target grid size (required unless --self-test)')
    p.add_argument('--self-test', action='store_true',
                   help='round-trip N -> 2N -> N; print max rel L2/Linf (charter number)')
    args = p.parse_args()

    field = np.load(args.source).astype(np.float64)
    if field.ndim == 2:
        field = field[None]
    n_in = field.shape[1]
    print(f'source {args.source}  shape={field.shape}  rms={np.sqrt(np.mean(field**2)):.6e}')

    if args.self_test:
        ref = regrid(field, n_in)           # identity-with-Nyquist-zeroed reference
        nyq = np.sqrt(np.mean((field - ref) ** 2)) / np.sqrt(np.mean(field ** 2))
        back = regrid(regrid(field, 2 * n_in), n_in)
        rel_l2 = np.linalg.norm(back - ref) / np.linalg.norm(ref)
        rel_linf = np.abs(back - ref).max() / np.abs(ref).max()
        print(f'source Nyquist content (rel rms, should be ~0 for dealiased fields): {nyq:.3e}')
        print(f'round-trip {n_in} -> {2*n_in} -> {n_in}:  rel_L2={rel_l2:.3e}  rel_Linf={rel_linf:.3e}')
        return

    if args.out is None or args.N_out is None:
        raise SystemExit('--out and --N-out required (or use --self-test)')
    if args.N_out < n_in:
        print(f'truncation {n_in} -> {args.N_out}: dropped energy fraction = '
              f'{dropped_energy_fraction(field, args.N_out):.6e}')
    out = regrid(field, args.N_out)
    np.save(args.out, out)
    print(f'wrote {args.out}  shape={out.shape}  rms={np.sqrt(np.mean(out**2)):.6e}')


if __name__ == '__main__':
    main()
