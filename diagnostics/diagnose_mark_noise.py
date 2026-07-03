#!/usr/bin/env python
r"""
diagnose_mark_noise.py -- measure per-mark noise in a deep 28-mark build.

For a smooth trajectory sampled at dt, the k-th FINITE DIFFERENCE (no 1/dt^k)
satisfies ||Delta^k w|| / ||w|| ~ (dt*sigma)^k. Per-mark iid noise of relative
amplitude eta flattens that decay at ~ 2^(k/2) * eta. So plotting the measured
||Delta^k|| against k separates:
    float32-only storage: flattens near ~1e-6
    the observed S=7 pathology: needs a flatten near ~1e-4..1e-3

Usage (from $QG_DIR/training):
    python diagnose_mark_noise.py \
        data/ensemble_N5_7lag/FRC-256/forced_turbulence_dT_5em3 \
        data/ensemble_N5_7lag/FRC-Re25k/forced_turbulence_dT_5em3 \
        data/ensemble_N5_7lag/FRC-combo/forced_turbulence_dT_5em3
(pass any mix of members: rebuilds vs original survivors -- if survivors are
clean and rebuilds are noisy, the rebuild path is the culprit.)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


def probe(root: Path, n_windows: int = 3):
    man = json.loads((root / 'manifest.json').read_text())
    M = int(man.get('n_snapshots_per_sample', 0))
    dt = float(man['Delta_T'])
    inp = np.load(root / 'packed' / 'inputs.npy', mmap_mode='r')
    N = inp.shape[0]
    print(f"\n[{root.parent.name}/{root.name}] marks={M} dt={dt} "
          f"windows={N} inputs dtype={inp.dtype}")
    if M < 8:
        print("  not a deep build (need >=8 marks); skip")
        return

    for w in np.linspace(0, N - 1, n_windows, dtype=int):
        # omega block = first M channels, newest-first
        om = np.asarray(inp[w, :M], dtype=np.float64)      # (M, Ny, Nx)
        base = np.linalg.norm(om[0])
        line = [f"  win {w:4d} ||w||={base:.3e}  ||Delta^k w||/||w||:"]
        d = om.copy()
        for k in range(1, 7):
            d = np.diff(d, axis=0)                          # k-th difference
            line.append(f"k{k}={np.linalg.norm(d, axis=(1, 2)).mean() / base:.3e}")
        print('  '.join(line))

    print("  reference decay if smooth: (dt*sigma)^k with sigma~15-25 -> "
          + '  '.join(f"k{k}~{(dt*20)**k:.1e}" for k in range(1, 7)))
    print("  float32-storage floor: ~ 2^(k/2)*1.2e-7 -> "
          + '  '.join(f"k{k}~{(2**(k/2))*1.19e-7:.1e}" for k in range(1, 7)))


if __name__ == '__main__':
    roots = [Path(a) for a in sys.argv[1:]]
    if not roots:
        print(__doc__)
        raise SystemExit(1)
    for r in roots:
        probe(r)
