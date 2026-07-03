#!/usr/bin/env python
r"""
diagnose_error_distribution.py -- per-sample init-model error distribution on the
val split of one sweep. Separates "uniformly bad" from "heavy tail of broken
samples", and identifies the offenders.

Prints median / p90 / p99 / max per order, then the worst offenders with:
window id, anchor, target norms, and stack roughness (Delta^2 of the omega
stack -- frozen/degenerate stacks show ~float32 floor ~1e-6 instead of ~1e-3).

Usage (from $QG_DIR/training):
    python diagnose_error_distribution.py \
        --sliced data/ensemble_N5_7lag/FRC-256/sweep_dT_5em3 \
        --ckpt data/ensemble_N5_7lag/training_runs/deriv7_equalw_R3R4/init.pt \
        --grad-kernel 15 --split val --n-worst 12
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deriv_dataset import DerivMemberDataset
from model_deriv_closure import build_model

ONAMES = ['Ndot', 'Nddot', 'N3dot']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sliced', type=Path, required=True)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--grad-kernel', type=int, default=15)
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--n-worst', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = args.device

    man = json.loads((args.sliced / 'manifest.json').read_text())
    S = int(man['n_snapshots_per_sample'])
    na = int(man.get('n_anchors', 1))
    dt = float(man['Delta_T'])
    Nx, Ny = int(man['Nx']), int(man['Ny'])
    dx_, dy_ = float(man['Lx']) / Nx, float(man['Ly']) / Ny

    ds = DerivMemberDataset(args.sliced, split=args.split, n_snapshots=S,
                            compute_dtype='float64')
    # recover the raw packed indices this split maps to (for window ids)
    raw_idx = getattr(ds, 'indices', None)
    if raw_idx is None:
        raw_idx = getattr(ds, 'idx', np.arange(len(ds)))
    raw_idx = np.asarray(raw_idx)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    mdl = build_model('cheap_deriv', in_channels=2 * S, out_orders=3,
                      grad_kernel=args.grad_kernel, dt=dt, dx=dx_, dy=dy_,
                      physics_init=True, learnable_stencils=True).to(dev).double()
    mdl.load_state_dict(ck['model']); mdl.eval()

    # per-shape dealias
    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral, to_physical
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=float(man['Lx']), Ly=float(man['Ly']),
                        device=dev, precision='float64')
    keep = (~Derivative(grid).alias_mask).to(dev)

    rels, tnorms, rough = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(ds), args.batch_size):
            ii = range(s0, min(s0 + args.batch_size, len(ds)))
            xs, ys, rg = zip(*[ds[i] for i in ii])
            x = torch.stack(xs).to(dev).double()
            y = torch.stack(ys).to(dev).double()[:, :3]
            reg = torch.stack(rg).to(dev)
            nd = mdl(x, dt=reg[:, 0].double(), dx=reg[:, 4].double(),
                     dy=reg[:, 5].double())
            nd = to_physical(to_spectral(nd) * keep)
            p = nd.flatten(2); t = y.flatten(2)
            r = (torch.norm(p - t, dim=2) / torch.norm(t, dim=2).clamp_min(1e-30))
            rels.append(r.cpu().numpy())
            tnorms.append(torch.norm(t, dim=2).cpu().numpy())
            # stack roughness: ||Delta^2 omega|| / ||omega_0|| per sample
            om = x[:, :S]
            d2 = torch.diff(om, n=2, dim=1)
            rough.append((torch.norm(d2.flatten(2), dim=2).mean(1)
                          / torch.norm(om[:, 0].flatten(1), dim=1)).cpu().numpy())
    rels = np.concatenate(rels)          # (N,3)
    tnorms = np.concatenate(tnorms)      # (N,3)
    rough = np.concatenate(rough)        # (N,)

    print(f"[dist] {args.sliced.parent.name}/{args.sliced.name} split={args.split} "
          f"n={len(rels)}  (mean is what the eval reported)")
    for m, name in enumerate(ONAMES):
        r = rels[:, m]
        print(f"  {name:6s} mean={r.mean():10.3f}  median={np.median(r):8.4f}  "
              f"p90={np.percentile(r, 90):8.3f}  p99={np.percentile(r, 99):10.3f}  "
              f"max={r.max():12.3f}")

    worst = np.argsort(rels[:, 2])[::-1][:args.n_worst]
    print(f"\n[dist] worst {args.n_worst} by N3dot:")
    print("   split-idx  raw-idx  window anchor   relN3dot     ||tN3||    "
          "stackD2/||w||")
    for i in worst:
        ri = int(raw_idx[i])
        print(f"   {i:9d}  {ri:7d}  {ri // na:6d} {ri % na:6d}  "
              f"{rels[i, 2]:10.3f}  {tnorms[i, 2]:10.3e}  {rough[i]:12.3e}")

    good = np.argsort(rels[:, 2])[:3]
    print("\n[dist] 3 best by N3dot (for contrast):")
    for i in good:
        ri = int(raw_idx[i])
        print(f"   {i:9d}  {ri:7d}  {ri // na:6d} {ri % na:6d}  "
              f"{rels[i, 2]:10.4f}  {tnorms[i, 2]:10.3e}  {rough[i]:12.3e}")


if __name__ == '__main__':
    main()
