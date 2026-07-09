#!/usr/bin/env python
r"""
eval_deriv_by_root.py -- per-(member, dt, order) rel-L2 of a trained deriv model.

The pooled training val averages easy (5e-3) and impossible (1.5e-2 near the
radius) samples into one number. This breaks it out: one row per sweep dir
(member x dt), columns Ndot/Nddot/N3dot, on the requested split.

Mirrors train_deriv.py exactly: per-sample dt/dx/dy from the regime vector,
per-shape dealias projection, same rel-L2. Loads best.pt (or any ckpt).

Usage (from $QG_DIR/training):
    python eval_deriv_by_root.py \
        --ckpt data/ensemble_N5_7lag/training_runs/deriv7_equalw_R3R4/best.pt \
        --sweep-roots data/ensemble_N5_7lag/FRC-*/sweep_dT_* \
        --n-snapshots 7 --grad-kernel 15 --split val \
        --compute-dtype float64
Output: table to stdout + eval_by_root.csv next to the ckpt.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from deriv_dataset import DerivMemberDataset
from model_deriv_closure import build_model

ORDER_NAMES = ['Ndot', 'Nddot', 'N3dot', 'N4dot']


def rel_l2_vec(pred, target):
    p = pred.flatten(start_dim=2)
    t = target.flatten(start_dim=2)
    return (torch.norm(p - t, dim=2) / torch.norm(t, dim=2).clamp_min(1e-30))  # (B,C)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--sweep-roots', type=Path, nargs='+', required=True)
    ap.add_argument('--n-snapshots', type=int, default=7)
    ap.add_argument('--out-orders', type=int, default=3)
    ap.add_argument('--grad-kernel', type=int, default=15)
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--compute-dtype', choices=['float32', 'float64'], default='float64')
    ap.add_argument('--dealias-pred', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-batches', type=int, default=0,
                    help='cap batches per root (0 = all; use e.g. 20 for a fast pass)')
    ap.add_argument('--model', default='auto',
                    choices=['auto', 'cheap_deriv', 'cond_local', 'cond_deriv'],
                    help="'auto' reads config.json next to the ckpt "
                         "(falls back to cheap_deriv)")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else 'cpu'
    adtype = torch.float64 if args.compute_dtype == 'float64' else torch.float32

    roots = [r for r in args.sweep_roots if (r / 'manifest.json').exists()
             and (r / 'packed' / 'inputs.npy').exists()]
    if not roots:
        raise SystemExit("no valid sweep roots")

    # ---- reference grid = most common (Ny,Nx,Lx,Ly), matching the trainer ----
    ident = {}
    for r in roots:
        m = json.loads((r / 'manifest.json').read_text())
        ident.setdefault((int(m['Ny']), int(m['Nx']),
                          round(float(m['Lx']), 6), round(float(m['Ly']), 6)), []).append(r)
    ref = max(ident, key=lambda k: len(ident[k]))
    Ny0, Nx0, Lx0, Ly0 = ref
    dx0, dy0 = Lx0 / Nx0, Ly0 / Ny0

    # ---- model at reference spacing, load ckpt ----
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model_name = args.model
    if model_name == 'auto':
        cfg_path = args.ckpt.parent / 'config.json'
        model_name = (json.loads(cfg_path.read_text()).get('model', 'cheap_deriv')
                      if cfg_path.exists() else 'cheap_deriv')
        print(f"[eval] --model auto -> {model_name} "
              f"({'from ' + str(cfg_path) if cfg_path.exists() else 'default'})")
    dt0 = 5e-3  # placeholder; per-sample dt overrides at forward
    model = build_model(model_name, in_channels=2 * args.n_snapshots,
                        out_orders=args.out_orders, grad_kernel=args.grad_kernel,
                        dt=dt0, dx=dx0, dy=dy0, physics_init=True,
                        learnable_stencils=True).to(dev).to(adtype)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"[eval] ckpt={args.ckpt}  epoch={ck.get('epoch','?')}  "
          f"train-val-at-save={ck.get('val','?')}")

    # ---- per-shape dealias (matches trainer) ----
    projs = {}
    if args.dealias_pred:
        from qg.solver.grid.cartesian import CartesianGrid
        from qg.solver.opt.derivative import Derivative
        from qg.solver.opt.basis import to_spectral, to_physical
        for (Ny, Nx, Lx, Ly) in ident:
            if (Ny, Nx) in projs:
                continue
            g = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=dev,
                              precision='float64')
            keep = (~Derivative(g).alias_mask).to(dev)
            def _p(p, keep=keep):
                return to_physical(to_spectral(p) * keep.to(device=p.device,
                                                            dtype=p.dtype))
            projs[(Ny, Nx)] = _p
    else:
        projs = None

    onames = ORDER_NAMES[:args.out_orders]
    rows = []
    with torch.no_grad():
        for r in sorted(roots):
            ds = DerivMemberDataset(r, split=args.split,
                                    n_snapshots=args.n_snapshots,
                                    compute_dtype=args.compute_dtype)
            if len(ds) == 0:
                continue
            member = r.parent.name
            dt_val = float(ds.man['Delta_T'])
            shape = (ds.Ny, ds.Nx)
            proj = (projs[shape] if projs else (lambda p: p))

            acc = []
            nb = 0
            for s in range(0, len(ds), args.batch_size):
                idxs = range(s, min(s + args.batch_size, len(ds)))
                xs, ys, rg = zip(*[ds[i] for i in idxs])
                x = torch.stack(xs).to(dev)
                y = torch.stack(ys).to(dev).to(adtype)[:, :args.out_orders]
                reg = torch.stack(rg).to(dev)
                dT = reg[:, 0].to(x.dtype)
                dX = reg[:, 4].to(x.dtype)
                dY = reg[:, 5].to(x.dtype)
                nd = proj(model(x, dt=dT, dx=dX, dy=dY))
                acc.append(rel_l2_vec(nd, y).cpu())
                nb += 1
                if args.max_batches and nb >= args.max_batches:
                    break
            per = torch.cat(acc, 0).mean(0).numpy()          # (C,)
            rows.append((member, dt_val, shape[0], len(ds)) + tuple(per))
            brk = '  '.join(f'{o}={v:.4f}' for o, v in zip(onames, per))
            print(f"  {member:12s} dT={dt_val:<8g} N{shape[0]:<4d} "
                  f"n={len(ds):<5d} {brk}")

    # ---- summary by dt tier ----
    print("\n[eval] mean over members, by dT:")
    for dt_val in sorted({r[1] for r in rows}):
        sel = np.array([r[4:] for r in rows if r[1] == dt_val])
        brk = '  '.join(f'{o}={v:.4f}' for o, v in zip(onames, sel.mean(0)))
        print(f"  dT={dt_val:<8g} ({len(sel)} members)  {brk}")

    out_csv = args.ckpt.parent / f'eval_by_root_{args.split}.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['member', 'Delta_T', 'Ny', 'n_samples'] + onames)
        w.writerows(rows)
    print(f"\n[eval] wrote {out_csv}")


if __name__ == '__main__':
    main()
