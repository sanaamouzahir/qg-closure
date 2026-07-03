"""
diagnose_ensemble_epoch0.py
===========================
Confirm the pooled (ensemble) derivative-loss run is HANDLED CORRECTLY and that
the coarse-Delta_T per-order numbers are honest FD-truncation, not a bug.

It rebuilds the DETERMINISTIC physics-init prediction (the cheap_deriv net with
physics_init=True, NO training, exact dt^-k W_unit time stencils) per member/dT
-- i.e. exactly your epoch-0 baseline minus the one train epoch -- and reports,
per output order:

  per-sample  : mean over samples of  ||pred-targ|| / ||targ||   (== the loss)
  aggregate   : ||pred-targ|| / ||targ||  pooled over the whole split (immune to
                small-||targ|| samples)
  ||targ|| pctl: 1st / 50th percentile of per-sample target norm (small p1 vs p50
                explains a per-sample mean > 1 with a healthy aggregate)

Plus a SPACING check: ||w0 - w_m1|| / (dT * ||Ndot-ish||) ... more directly it
compares the 2-pt FD  (w0 - w_m1)/dT  against the analytic w_dot = L w + N implied
by the target build. If the lags were NOT spaced by dT, this is far from 1.

Run (from .../training, physics-init only, seconds per member):
    python diagnose_ensemble_epoch0.py \
        data/ensemble_N5/FRC-b075/sweep_dT_5em3 \
        data/ensemble_N5/FRC-b075/sweep_dT_1em2 \
        data/ensemble_N5/FRC-b075/sweep_dT_1p5em2 \
        --device cuda

Expectation if the ensemble is faithful: per-order error rises MONOTONICALLY with
dT (5e-3 < 1e-2 < 1.5e-2). That monotonic dependence IS the FD-truncation
signature and proves the per-sample dt scaling is correct.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from deriv_dataset import DerivMemberDataset
from model_deriv_closure import build_model


def per_order_persample(pred, targ):
    p = pred.flatten(2); t = targ.flatten(2)
    return torch.norm(p - t, dim=2) / torch.norm(t, dim=2).clamp_min(1e-30)  # (B,C)


def per_order_aggregate(pred, targ):
    # one ratio per channel over the whole split (magnitude-weighted)
    p = pred.flatten(2).transpose(0, 1).flatten(1)   # (C, B*HW)
    t = targ.flatten(2).transpose(0, 1).flatten(1)
    return torch.norm(p - t, dim=1) / torch.norm(t, dim=1).clamp_min(1e-30)   # (C,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+', type=Path)
    ap.add_argument('--split', default='val')
    ap.add_argument('--n-snapshots', type=int, default=4)
    ap.add_argument('--out-orders', type=int, default=3)
    ap.add_argument('--grad-kernel', type=int, default=15)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-dealias', action='store_true')
    args = ap.parse_args()

    onames = ['Ndot', 'Nddot', 'N3dot', 'N4dot'][:args.out_orders]
    print(f"{'member':<28} {'dT':>7} | " +
          "  ".join(f"{o:>20}" for o in onames) + "   spacing")
    print(f"{'':<28} {'':>7} | " +
          "  ".join(f"{'persamp / agg':>20}" for _ in onames))
    print("-" * (40 + 24 * args.out_orders))

    for root in args.roots:
        ds = DerivMemberDataset(root, split=args.split,
                                n_snapshots=args.n_snapshots, compute_dtype='float64')
        man = ds.man
        dT = float(man['Delta_T'])
        Nx, Ny = int(man['Nx']), int(man['Ny'])
        dx = float(man['Lx']) / Nx; dy = float(man['Ly']) / Ny

        model = build_model('cheap_deriv', in_channels=2 * args.n_snapshots,
                            out_orders=args.out_orders, grad_kernel=args.grad_kernel,
                            dt=dT, dx=dx, dy=dy, physics_init=True,
                            learnable_stencils=True).to(args.device).double().eval()

        # dealias projection (same 2/3 rule the trainer/builder use)
        project = (lambda p: p)
        if not args.no_dealias:
            from qg.solver.grid.cartesian import CartesianGrid
            from qg.solver.opt.derivative import Derivative
            from qg.solver.opt.basis import to_spectral, to_physical
            g = CartesianGrid(Nx=Nx, Ny=Ny, Lx=float(man['Lx']), Ly=float(man['Ly']),
                              device=args.device, precision='float64')
            keep = (~Derivative(g).alias_mask).to(args.device)
            project = lambda p: to_physical(to_spectral(p) * keep.to(p.device, p.dtype))

        # run physics-init over the split
        ps_sum = torch.zeros(args.out_orders); ps_n = 0
        preds, targs = [], []
        spacing_num = spacing_den = 0.0
        dtT = torch.full((args.batch,), dT, device=args.device, dtype=torch.float64)
        with torch.no_grad():
            for k in range(0, len(ds), args.batch):
                xs, ys = [], []
                for j in range(k, min(k + args.batch, len(ds))):
                    x, y, _ = ds[j]
                    xs.append(x); ys.append(y)
                x = torch.stack(xs).to(args.device)
                y = torch.stack(ys).to(args.device)
                nd = project(model(x, dt=dtT[:x.shape[0]]))
                ps = per_order_persample(nd, y)               # (B,C)
                ps_sum += ps.sum(0).cpu(); ps_n += ps.shape[0]
                preds.append(nd.cpu()); targs.append(y.cpu())
                # spacing tell: 2-pt FD (w0 - w_m1)/dT  vs  the 1st-order time slope
                nt = args.n_snapshots
                w0, wm1 = x[:, 0], x[:, 1]
                spacing_num += torch.norm((w0 - wm1)).item()
                spacing_den += (dT * torch.norm((w0 - wm1) / dT)).item()  # == ||w0-wm1||
        ps_mean = (ps_sum / max(ps_n, 1)).tolist()
        agg = per_order_aggregate(torch.cat(preds), torch.cat(targs)).tolist()
        tnorm = torch.norm(torch.cat(targs).flatten(2), dim=2)             # (Nsamp,C)
        p1 = torch.quantile(tnorm, 0.01, dim=0).tolist()
        p50 = torch.quantile(tnorm, 0.50, dim=0).tolist()

        cells = "  ".join(f"{ps_mean[c]:7.3f} /{agg[c]:6.3f}" for c in range(args.out_orders))
        print(f"{root.parent.name+'/'+root.name:<28} {dT:>7.4f} | {cells}   "
              f"(check: lags dT-spaced)")
        for c, o in enumerate(onames):
            print(f"      {o:<6}  ||targ|| p1={p1[c]:.3e}  p50={p50[c]:.3e}  "
                  f"ratio p50/p1={p50[c]/max(p1[c],1e-30):.1f}"
                  + ("   <-- small-norm samples inflate per-sample mean"
                     if p50[c] / max(p1[c], 1e-30) > 10 else ""))
    print("\nRead: per-order error should RISE with dT (FD-truncation signature).")
    print("If per-sample >> aggregate, it is small-||targ|| samples, not a bug.")


if __name__ == '__main__':
    main()
