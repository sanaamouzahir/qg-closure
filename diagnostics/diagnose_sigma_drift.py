#!/usr/bin/env python
r"""diagnose_sigma_drift.py -- how sample-local is the cond_deriv conditioning input?

cond_deriv conditions each sample on x(kappa) = dT * sigma_hat_omega(kappa), read
from that sample's own two newest omega marks. Within one deep WINDOW the slicer
harvests n_anchors samples (rows 3n..3n+n_anchors-1 in window-major packing), each
shifted by one fine step h_fine. If sigma_hat(kappa) is nearly identical across a
window's anchors, the conditioning input is a stable, physical regime read (good);
if it swings anchor-to-anchor, the model is effectively conditioning on per-sample
noise -- a risk worth knowing BEFORE training.

This measures the relative drift of the conditioning group x(kappa) between the
first and last anchor of each window, aggregated over windows and the resolved
shell band.

Analysis-only. Run from anywhere:
    python diagnostics/diagnose_sigma_drift.py
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_TRAIN = Path(__file__).resolve().parent.parent / 'training'
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))
_DATA = _TRAIN / 'data' / 'ensemble_N5_7lag'

DEFAULT_ROOTS = [
    'FRC-256/sweep_dT_5em3',
    'FRC-256/sweep_dT_1p5em2',
    'FRC-kf4/sweep_dT_1em2',
    'FRC-Re25k/sweep_dT_1em2',
]


def analyze(root: Path, n_windows: int, dev: str):
    from cond_grad import sigma_hat, _CACHE
    man = json.loads((root / 'manifest.json').read_text())
    dt = float(man['Delta_T'])
    Nx, Ny = int(man['Nx']), int(man['Ny'])
    Lx, Ly = float(man['Lx']), float(man['Ly'])
    na = int(man.get('n_anchors', 3))
    if na < 2:
        print(f"[skip] {root.parent.name}/{root.name}: n_anchors={na} (<2)")
        return
    inp = np.load(root / 'packed' / 'inputs.npy', mmap_mode='r')
    S = int(man['n_snapshots_per_sample'])
    Ntot = inp.shape[0]
    Nwin = Ntot // na
    step = max(Nwin // n_windows, 1)
    wins = list(range(0, Nwin, step))[:n_windows]
    dtv = torch.tensor([dt], device=dev, dtype=torch.float64)

    # energy-weighted shell mask: restrict to shells carrying real omega energy
    c = _CACHE.get(Ny, Nx, Lx, Ly, dev, torch.float64)

    drifts = []          # per-window, per-shell relative drift of x=dT*sigma
    x0_all = []          # anchor-0 x for band selection
    for w in wins:
        r0, rL = na * w + 0, na * w + (na - 1)
        om0_0 = torch.tensor(np.asarray(inp[r0, 0], np.float64), device=dev)[None]
        om1_0 = torch.tensor(np.asarray(inp[r0, 1], np.float64), device=dev)[None]
        om0_L = torch.tensor(np.asarray(inp[rL, 0], np.float64), device=dev)[None]
        om1_L = torch.tensor(np.asarray(inp[rL, 1], np.float64), device=dev)[None]
        sig0, _ = sigma_hat(om0_0, om1_0, dtv, Lx, Ly)   # (1, n_sh)
        sigL, _ = sigma_hat(om0_L, om1_L, dtv, Lx, Ly)
        x0 = (sig0 * dt)[0]
        xL = (sigL * dt)[0]
        rel = (xL - x0).abs() / x0.clamp_min(1e-12)
        drifts.append(rel)
        x0_all.append(x0)

    D = torch.stack(drifts)          # (W, n_sh)
    X0 = torch.stack(x0_all)         # (W, n_sh)
    # band: shells where the conditioning input is non-trivial (x >= 1% of the
    # per-window max x) -- the shells the MLP actually keys on.
    band = X0 >= 0.01 * X0.max(dim=1, keepdim=True).values
    d_band = D[band]
    xr = X0[X0 > 0]
    print(f"[{root.parent.name}/{root.name}] dt={dt} {Ny}x{Nx} n_anchors={na} "
          f"windows={len(wins)} (anchor 0 vs {na-1}, {na-1} fine-steps apart)")
    print(f"   conditioning x=dT*sigma range: [{float(xr.min()):.2e}, {float(xr.max()):.2e}]")
    q = torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64)
    qs = torch.quantile(d_band.double(), q)
    print(f"   |dx|/x over resolved band: median={float(qs[0]):.2e}  "
          f"p90={float(qs[1]):.2e}  p99={float(qs[2]):.2e}  max={float(d_band.max()):.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roots', nargs='*', default=DEFAULT_ROOTS,
                    help='roots under training/data/ensemble_N5_7lag')
    ap.add_argument('--n-windows', type=int, default=40)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    for r in args.roots:
        root = _DATA / r
        if not (root / 'packed' / 'inputs.npy').exists():
            print(f"[skip] {r}: missing inputs.npy")
            continue
        analyze(root, args.n_windows, args.device)


if __name__ == '__main__':
    main()
