"""
train_deriv.py - the pre-6.1.2 derivative-loss temporal closure, on the ensemble.

The cheap_deriv network predicts the LOCAL N-time-derivatives [Ndot, Nddot, N3dot]
DIRECTLY from the minimal 8-channel snapshot input

    [omega_0, omega_m1, omega_m2, omega_m3,  psi_0, psi_m1, psi_m2, psi_m3]   (n_time=4)

(four consecutive time levels make the derivatives available by FD; the chain-rule
binomials are the physics-init of the 1x1 mix). The L^k weightings of the truncation
operators are NOT learned -- they are applied analytically at assembly/inference. So
training here supervises the derivatives themselves; the loss is the per-order
relative-L2 against the analytic targets [Ndot, Nddot, N3dot] (add_deriv_targets.py).

This is the SAME model and SAME objective as the single-case run that gave the
rollout (cheap_deriv, derivative loss), just POOLED over the ensemble. No corrector,
no delta-assembly, no L^k in the loss -- the error-propagation analysis showed the
closure error == the per-operator error (no amplification), so the derivative loss
IS the closure objective, and N_ddot's accuracy sets the rollout ceiling.

dT is a per-sample model input only through the cheap_deriv TimeFD scaling (one model
across the Delta_T sweep). FD stencils are frozen by default (recommended across a dT
sweep; the 1/dt^k weights make a single learnable stencil cross-dt-imbalanced).

MULTIGRID scope: pools ANY mix of grids/domains. The cheap_deriv spatial stencil
is built at a REFERENCE grid (the most common (Ny,Nx,Lx,Ly)); every sample's
gradient is rescaled by dx0/dx_i inside SpatialGrad, so one learnable dimensionless
operator serves all grids. Batches are shape-homogeneous (GridHomogeneousBatchSampler)
but may mix domains of equal shape and unequal dx -- hence the rescale is per-SAMPLE.
The dealias projection is per-shape (the 2/3 mask is mode-index based, L-independent).
On a single grid this is bit-identical to the single-grid trainer (rescale = x1).

Usage:
    python train_deriv.py \
        --sweep-roots $QG_DIR/training/data/ensemble_N5/FRC-*/sweep_dT_* \
        --n-snapshots 4 --out-orders 3 --grad-kernel 7 \
        --epochs 200 --lr 3e-4 --batch-size 4 --compute-dtype float64
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from deriv_dataset import make_deriv_loaders
from model_deriv_closure import build_model

ORDER_NAMES = ['Ndot', 'Nddot', 'N3dot', 'N4dot']


def relative_l2_perchannel(pred, target):
    """EXACT pre-6.1.2 training loss: per-SAMPLE, per-CHANNEL relative L2, averaged
    over channels and batch (each N-derivative order, spanning orders of magnitude,
    is normalized independently so the largest does not dominate). This is the
    `rel_l2` + multi-target criterion from the original train.py."""
    p = pred.flatten(start_dim=2)
    t = target.flatten(start_dim=2)
    num = torch.norm(p - t, dim=2)
    den = torch.norm(t, dim=2).clamp_min(1e-30)
    return (num / den).mean()


def relative_l2_perchannel_vec(pred, target):
    """(C,) per-operator relative L2, averaged over batch (diagnostic): tracks each
    of Ndot, Nddot, N3dot separately. mean() of this == relative_l2_perchannel."""
    p = pred.flatten(start_dim=2)
    t = target.flatten(start_dim=2)
    num = torch.norm(p - t, dim=2)
    den = torch.norm(t, dim=2).clamp_min(1e-30)
    return (num / den).mean(dim=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-roots', type=Path, nargs='+', required=True,
                   help='sweep_dT_* dirs (across members and Delta_T); each must have '
                        'packed/deriv_anal_f64.npy (run add_deriv_targets.py first)')
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--out-root', type=Path, default=None,
                   help='where training_runs/<run> lands (default: ensemble root)')
    p.add_argument('--grid', type=str, default=None,
                   help="restrict to one grid 'NyxNx' (default: most common in pool)")

    p.add_argument('--n-snapshots', type=int, default=4,
                   help='lags fed to the model (4 -> 8 channels, n_time=4; gives '
                        'N^(1..3) exactly in the FD span)')
    p.add_argument('--out-orders', type=int, default=3,
                   help='number of N-derivative outputs [Ndot, Nddot, N3dot]')
    p.add_argument('--grad-kernel', type=int, default=7,
                   help='spatial FD stencil width (7 = 6th order; narrows the '
                        'FD-vs-spectral gap on high-k fields)')
    p.add_argument('--no-physics-init', action='store_true',
                   help='disable chain-rule binomial init of the 1x1 mix')
    p.add_argument('--learnable-stencils', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='let the SPATIAL grad stencils train (default ON, matching '
                        'the pre-6.1.2 single-trajectory run -- the width-15 stencil '
                        'refines toward the spectral derivative, tightening the high-k '
                        'Nddot gap that sets the rollout ceiling). NOTE: the TIME-FD '
                        'stencils use the exact dt^-k W_unit path across the sweep '
                        'regardless of this flag (a single learnable time stencil '
                        'cannot serve multiple dt), so this governs only the spatial '
                        'kernels. Use --no-learnable-stencils for the frozen-FD ablation.')

    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--compute-dtype', choices=['float32', 'float64'], default='float64')
    p.add_argument('--dealias-pred', action=argparse.BooleanOptionalAction, default=True,
                   help='project predictions onto the 2/3 band (same cutoff as the '
                        'solver/builder) before the loss. The analytic targets are '
                        'dealiased by the Jacobian, so the prediction must be too '
                        '(matches the pre-6.1.2 run). --no-dealias-pred to disable.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--print-every', type=int, default=5)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- resolve roots; KEEP ALL GRIDS (multigrid) ----
    roots = [r for r in args.sweep_roots if (r / 'manifest.json').exists()
             and (r / 'packed' / 'inputs.npy').exists()]
    if not roots:
        raise SystemExit("no valid sweep roots (need manifest.json + packed/inputs.npy)")
    # full grid identity = (Ny, Nx, Lx, Ly); the most common one is the REFERENCE the
    # spatial stencil is built at (all others rescale to it per-sample). --grid still
    # filters by shape 'NyxNx' if you want to pin to one shape.
    ident = {}
    for r in roots:
        m = json.loads((r / 'manifest.json').read_text())
        shape = f"{int(m['Ny'])}x{int(m['Nx'])}"
        if args.grid and shape != args.grid:
            continue
        ident.setdefault((int(m['Ny']), int(m['Nx']),
                          round(float(m['Lx']), 6), round(float(m['Ly']), 6)), []).append(r)
    if not ident:
        raise SystemExit(f"no roots match --grid {args.grid!r}")
    roots = [r for v in ident.values() for r in v]
    ref = max(ident, key=lambda k: len(ident[k]))         # reference full grid
    shapes = sorted({(k[0], k[1]) for k in ident})
    print(f"[deriv-train] MULTIGRID  roots={len(roots)}  full-grids={len(ident)}  "
          f"shapes={shapes}  reference(Ny,Nx,Lx,Ly)={ref}")

    # ---- data ----
    tl, vl, te, train_ds, val_ds, test_ds = make_deriv_loaders(
        roots, batch_size=args.batch_size, num_workers=args.num_workers,
        n_snapshots=args.n_snapshots, compute_dtype=args.compute_dtype, seed=0)

    adtype = torch.float64 if args.compute_dtype == 'float64' else torch.float32
    # reference spacings: the stencil is baked here; every other grid rescales to it.
    Ny0, Nx0, Lx0, Ly0 = ref
    dx0 = Lx0 / Nx0; dy0 = Ly0 / Ny0; dt0 = float(train_ds.subsets[0].man['Delta_T'])

    # ---- dealias projections, ONE PER SHAPE (the 2/3 mask is mode-index based, so
    # it depends only on (Ny,Nx), NOT on L -- two domains of the same shape share it).
    # Built from a representative member of each shape. project_by_shape[(Ny,Nx)] -> fn.
    project_by_shape = {}
    if args.dealias_pred:
        from qg.solver.grid.cartesian import CartesianGrid
        from qg.solver.opt.derivative import Derivative
        from qg.solver.opt.basis import to_spectral, to_physical
        rep = {}
        for s in train_ds.subsets:
            rep.setdefault((int(s.man['Ny']), int(s.man['Nx'])), s.man)
        for (Ny, Nx), m in rep.items():
            g = CartesianGrid(Nx=Nx, Ny=Ny, Lx=float(m['Lx']), Ly=float(m['Ly']),
                              device=args.device, precision='float64')
            keep = (~Derivative(g).alias_mask).to(args.device)
            def _proj(p, keep=keep):
                return to_physical(to_spectral(p) * keep.to(device=p.device, dtype=p.dtype))
            project_by_shape[(Ny, Nx)] = _proj
        print(f"[deriv-train] dealiasing predictions before loss (2/3 rule), "
              f"{len(project_by_shape)} per-shape projection(s)")
    else:
        for s in train_ds.subsets:
            project_by_shape[(int(s.man['Ny']), int(s.man['Nx']))] = lambda p: p

    # ---- model (corrector OFF; built at the REFERENCE grid spacing) ----
    in_ch = 2 * args.n_snapshots
    model = build_model('cheap_deriv', in_channels=in_ch, out_orders=args.out_orders,
                        grad_kernel=args.grad_kernel, dt=dt0,
                        dx=dx0, dy=dy0, physics_init=not args.no_physics_init,
                        learnable_stencils=args.learnable_stencils).to(args.device).to(adtype)
    trainable = [q for q in model.parameters() if q.requires_grad]
    n_params = sum(q.numel() for q in trainable)
    print(f"[deriv-train] cheap_deriv trainable params={n_params:,}  in_ch={in_ch}  "
          f"out_orders={args.out_orders}  spatial_stencils={'learn' if args.learnable_stencils else 'frozen'} "
          f"(time-FD = exact dt^-k W_unit; spatial rescaled per-sample to ref dx0={dx0:.4e})  "
          f"dtype={args.compute_dtype}")

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    out_root = args.out_root or roots[0].parent.parent
    run = args.run_name or f"deriv_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = out_root / 'training_runs' / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(vars(args) | {
        'multigrid': True, 'shapes': [f"{a}x{b}" for a, b in shapes],
        'reference_grid': list(ref), 'ref_dx0': dx0, 'ref_dy0': dy0,
        'roots': [str(r) for r in roots]}, indent=2, default=str))
    log = run_dir / 'log.csv'
    onames = ORDER_NAMES[:args.out_orders]
    log.write_text('epoch,lr,train_relL2,val_relL2,best_val,'
                   + ','.join(f'val_{o}' for o in onames) + ',elapsed_s\n')

    def run_epoch(loader, train):
        model.train(train)
        tot = 0.0; per = np.zeros(args.out_orders); nb = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y, regime in loader:
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True).to(adtype)
                regime = regime.to(args.device, non_blocking=True)
                dT = regime[:, 0].to(x.dtype)
                dxb = regime[:, 4].to(x.dtype)             # MULTIGRID: per-sample dx
                dyb = regime[:, 5].to(x.dtype)             # MULTIGRID: per-sample dy
                nd = model(x, dt=dT, dx=dxb, dy=dyb)        # (B, out_orders, H, W)
                project = project_by_shape[(x.shape[-2], x.shape[-1])]  # by batch shape
                nd = project(nd)
                loss = relative_l2_perchannel(nd, y)        # per-sample, per-channel mean
                if train:
                    optim.zero_grad(); loss.backward(); optim.step()
                tot += loss.item()
                per += relative_l2_perchannel_vec(nd, y).detach().cpu().numpy()
                nb += 1
        nb = max(nb, 1)
        # headline = mean per-order rel-L2 (== the per-channel criterion); per = per-order
        return tot / nb, per / nb

    best = float('inf'); t0 = time.time()
    for ep in range(args.epochs):
        te0 = time.time()
        # multigrid: advance the GridHomogeneousBatchSampler so batches reshuffle
        # each epoch (set_epoch is a no-op / absent on the single-grid DataLoader path).
        for _ld in (tl, vl, te):
            _bs = getattr(_ld, 'batch_sampler', None)
            if _bs is not None and hasattr(_bs, 'set_epoch'):
                _bs.set_epoch(ep)
        tr, _ = run_epoch(tl, True)
        va, va_per = run_epoch(vl, False)
        sched.step()
        improved = va < best
        if improved:
            best = va
            torch.save({'model': model.state_dict(), 'epoch': ep, 'val': va,
                        'config': vars(args), 'reference_grid': list(ref),
                        'ref_dx0': dx0, 'ref_dy0': dy0}, run_dir / 'best.pt')
        torch.save({'model': model.state_dict(), 'epoch': ep, 'val': va},
                   run_dir / 'last.pt')
        with open(log, 'a') as f:
            f.write(f'{ep},{optim.param_groups[0]["lr"]:.3e},{tr:.6e},{va:.6e},'
                    f'{best:.6e},' + ','.join(f'{v:.6e}' for v in va_per)
                    + f',{time.time()-t0:.1f}\n')
        if ep % args.print_every == 0 or improved:
            brk = ' '.join(f'{o}={v:.3e}' for o, v in zip(onames, va_per))
            print(f"  ep {ep:4d} {'*' if improved else ' '}  "
                  f"train={tr:.4e}  val={va:.4e}  best={best:.4e} (mean relL2/order)  "
                  f"[{brk}]  ({time.time()-te0:.1f}s)")

    print(f"\n[deriv-train] done in {(time.time()-t0)/60:.1f} min, best val={best:.4e}")
    ckpt = torch.load(run_dir / 'best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    test, test_per = run_epoch(te, False)
    brk = ' '.join(f'{o}={v:.4e}' for o, v in zip(onames, test_per))
    print(f"[deriv-train] TEST relL2={test:.4e}  [{brk}]")
    with open(log, 'a') as f:
        f.write(f'-1,0.0,0.0,{test:.6e},{best:.6e},'
                + ','.join(f'{v:.6e}' for v in test_per) + f',{time.time()-t0:.1f}\n')


if __name__ == '__main__':
    main()