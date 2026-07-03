#!/usr/bin/env python
r"""
check_f32_quantization.py -- decisive test of the float32-input-storage hypothesis.

Loads a few consecutive float64 snapshots straight from a member's DNS npz,
computes the 7-node FD N-derivatives twice: (a) from the float64 fields, and
(b) from the SAME fields round-tripped through float32 (exactly what disk
storage does). Reports rel-L2( (b) vs (a) ) per order per dt.

If the hypothesis is right, this prints ~ the init-eval numbers (N3dot O(10+)
at dt=5e-3, falling ~1/dt^3), with zero other moving parts.

Usage (from $QG_DIR/training):
    python check_f32_quantization.py \
        --npz $QG_DIR/outputs/Step_size_resolution_closure_ensemble/FRC-256/DNS_FR.npz \
        --t-start 15.0 --dts 5e-3 1e-2 1.5e-2
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import torch


def fd_weights(S: int) -> np.ndarray:
    x = np.arange(0, -S, -1, dtype=float)
    A = np.array([[x[j] ** m / math.factorial(m) for j in range(S)]
                  for m in range(S)])
    return np.linalg.inv(A).T          # row k = order-k stencil, unit spacing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True, help='DNS_FR.npz of one member')
    ap.add_argument('--t-start', type=float, default=15.0)
    ap.add_argument('--dts', type=float, nargs='+', default=[5e-3, 1e-2, 1.5e-2])
    ap.add_argument('--S', type=int, default=7)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    d = np.load(args.npz)
    # field key: prefer omega/q; adapt if named differently
    key = next(k for k in ('omega', 'q', 'w') if k in d.files)
    t = d['t'] if 't' in d.files else None
    F = d[key]                                        # (T, Ny, Nx) float64
    print(f"[chk] {args.npz.name}: field '{key}' shape={F.shape} dtype={F.dtype}")
    if F.dtype != np.float64:
        print("[chk] WARNING: source is not float64 -- test is only meaningful "
              "from a float64 source.")
    if t is not None:
        dt_fine = float(t[1] - t[0])
    else:
        dt_fine = 5e-3
        print(f"[chk] no 't' in npz; assuming dt_fine={dt_fine}")
    i0 = int(round(args.t_start / dt_fine)) if t is None else int(np.argmin(np.abs(t - args.t_start)))

    W = fd_weights(args.S)
    dev = args.device
    for dt in args.dts:
        j = int(round(dt / dt_fine))
        idx = [i0 - m * j for m in range(args.S)]     # newest first, backward
        if min(idx) < 0:
            print(f"  dT={dt}: not enough history at t_start, skip"); continue
        snaps64 = torch.tensor(F[idx], dtype=torch.float64, device=dev)  # (S,Ny,Nx)
        snaps32 = snaps64.to(torch.float32).to(torch.float64)            # disk round-trip

        line = [f"  dT={dt:<8g} j={j}"]
        for k in (1, 2, 3):
            w = torch.tensor(W[k], dtype=torch.float64, device=dev)
            d64 = torch.einsum('s,shw->hw', w, snaps64) / dt ** k
            d32 = torch.einsum('s,shw->hw', w, snaps32) / dt ** k
            rel = (torch.norm(d32 - d64) / torch.norm(d64).clamp_min(1e-30)).item()
            line.append(f"omega^({k}) f32-vs-f64 rel={rel:.4f}")
        print('   '.join(line))

    print("\n[chk] Interpretation: these are the field-level errors. The Jacobian "
          "adds a gradient boost (white noise ~k_max vs physical ~k_char) and four "
          "binomial slots, so N^(k) errors are several x larger. If omega^(3) shows "
          "O(0.1-1) at 5e-3 falling ~1/dt^3, the float32-storage hypothesis is "
          "confirmed and the deep builds must be rebuilt at float64.")


if __name__ == '__main__':
    main()
