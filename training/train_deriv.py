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


def relative_l2_perchannel(pred, target, floor=None):
    """Per-SAMPLE, per-CHANNEL relative L2, averaged over channels and batch.

    floor: optional (B,C) tensor of per-sample denominator floors
    (= rel_floor * member-median ||t_c||, carried in regime[:,6:9]).
    Denominator = max(||t_c||, floor). Caps the leverage of near-zero-target
    samples (e.g. Ndot zero-crossings at ||N(t)|| extrema) without dropping
    them. floor=None == exact pre-6.1.2 loss."""
    p = pred.flatten(start_dim=2)
    t = target.flatten(start_dim=2)
    num = torch.norm(p - t, dim=2)
    den = torch.norm(t, dim=2).clamp_min(1e-30)
    if floor is not None:
        den = torch.maximum(den, floor.to(den.dtype))
    return (num / den).mean()


def relative_l2_perchannel_vec(pred, target, floor=None):
    """(C,) per-operator relative L2, averaged over batch (diagnostic): tracks each
    of Ndot, Nddot, N3dot separately. mean() of this == relative_l2_perchannel."""
    p = pred.flatten(start_dim=2)
    t = target.flatten(start_dim=2)
    num = torch.norm(p - t, dim=2)
    den = torch.norm(t, dim=2).clamp_min(1e-30)
    if floor is not None:
        den = torch.maximum(den, floor.to(den.dtype))
    return (num / den).mean(dim=0)


def relative_l2_persample(pred, target, floor=None):
    """(B,C) per-sample per-operator relative L2 with the SAME floored
    denominator as the loss. Used to report per-order MEDIANS each epoch
    (F1, incident 1827034): the floored-mean stays the headline/best metric,
    the median is printed+logged alongside so plateaus stay comparable to
    the 0.19/0.26/0.33 physics-init convention (rule 16)."""
    p = pred.flatten(start_dim=2)
    t = target.flatten(start_dim=2)
    num = torch.norm(p - t, dim=2)
    den = torch.norm(t, dim=2).clamp_min(1e-30)
    if floor is not None:
        den = torch.maximum(den, floor.to(den.dtype))
    return num / den


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

    p.add_argument('--model', default='cheap_deriv',
                   choices=['cheap_deriv', 'cond_local', 'cond_deriv'],
                   help="'cheap_deriv' = learned local FD stencils (control); "
                        "'cond_local' = control + sigma-hat tap modulation "
                        "(the deliverable; inference = control + 2 FFTs/step); "
                        "'cond_deriv' = conditioned spectral gradients "
                        "(~24 FFTs/step; ceiling-measurement instrument only, "
                        "ignores --grad-kernel)")
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
    p.add_argument('--rel-floor', type=float, default=0.1,
                   help='loss denominator floor = rel_floor * member-median '
                        '||t_c|| per channel (from regime[:,6:9]). Caps the '
                        'leverage of near-zero-target samples (Ndot zero-'
                        'crossings at ||N(t)|| extrema). 0 disables (exact '
                        'pre-6.1.2 relative loss).')
    p.add_argument('--init-ckpt', type=Path, default=None,
                   help='warm start from a train_deriv-lineage ckpt. If its '
                        'grad_kernel is NARROWER than --grad-kernel, the '
                        'stencils and cond head are widened EXACTLY '
                        '(center-embed, zeros outside): the widened model '
                        'computes the identical function at step 0, the new '
                        'outer taps are pure added capacity.')
    p.add_argument('--order-weights', type=str, default=None,
                   help="per-order loss weights, e.g. '1,1,0' to ditch "
                        "N3dot from the objective (Sanaa 2026-07-14; N3dot "
                        "is 2.6%% of the closure and its FD floor is high). "
                        "Weighted loss also drives best-ckpt selection; "
                        "per-order columns stay logged unweighted. Default: "
                        "equal weights.")
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

    # GUARD: the per-SHAPE dealias projection below is valid ONLY for isotropic
    # (Lx==Ly) domains -- then the 2/3 mask is mode-index based (L cancels) so all
    # members of a shape share one mask. A rectangular (Lx!=Ly) member has an
    # aspect-ratio-dependent mask; per-shape projection would silently mis-dealias
    # it. All current scenarios are isotropic (4pi, 8pi squares); refuse loudly if
    # that ever changes rather than corrupt the loss.
    aniso = [(Ny, Nx, Lx, Ly) for (Ny, Nx, Lx, Ly) in ident
             if abs(Lx - Ly) > 1e-9 or Ny != Nx]
    if aniso:
        raise SystemExit(
            "[deriv-train] ANISOTROPIC domain(s) detected (Lx!=Ly or Ny!=Nx): "
            f"{aniso}. The per-shape dealias projection is INVALID for these (the "
            "2/3 mask depends on the aspect ratio). Key the projection AND the "
            "batch sampler by (Ny,Nx,Lx/Ly) before running -- ask for the "
            "aspect-aware variant.")

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

    if args.rel_floor > 0:
        print(f"[deriv-train] norm-floored loss: denominator = "
              f"max(||t||, {args.rel_floor} * member-median ||t_c||)")
    else:
        print("[deriv-train] RAW relative loss (no floor) -- near-zero-target "
              "samples have unbounded leverage")

    # ---- model (corrector OFF; built at the REFERENCE grid spacing) ----
    in_ch = 2 * args.n_snapshots
    model = build_model(args.model, in_channels=in_ch, out_orders=args.out_orders,
                        grad_kernel=args.grad_kernel, dt=dt0,
                        dx=dx0, dy=dy0, physics_init=not args.no_physics_init,
                        learnable_stencils=args.learnable_stencils).to(args.device).to(adtype)
    # ---- warm start (with exact widening if the ckpt stencil is narrower) ----
    if args.init_ckpt is not None:
        ck = torch.load(args.init_ckpt, map_location=args.device,
                        weights_only=False)
        w_in = int(ck['config'].get('grad_kernel', args.grad_kernel))
        w_out = args.grad_kernel
        if w_in == w_out:
            model.load_state_dict(ck['model'])
            print(f"[deriv-train] warm start from {args.init_ckpt} "
                  f"(epoch={ck.get('epoch')})")
        elif w_in < w_out:
            pad = (w_out - w_in) // 2
            sd_new = model.state_dict()
            loaded = {}
            for k, v_new in sd_new.items():
                if k not in ck['model']:
                    raise SystemExit(f"[init-ckpt] key {k} missing in ckpt")
                v_old = ck['model'][k].to(v_new.dtype)
                if tuple(v_old.shape) == tuple(v_new.shape):
                    loaded[k] = v_old
                elif k.endswith('grad.wx') or k.endswith('grad.wy'):
                    # (C,1,w,w) -> (C,1,W,W): zeros outside == same conv output
                    t = torch.zeros_like(v_new)
                    t[..., pad:pad + w_in, pad:pad + w_in] = v_old
                    loaded[k] = t
                elif k.endswith('cond.head.weight'):
                    C = v_old.shape[0] // (2 * w_in)
                    t = torch.zeros_like(v_new)      # (C*2*W, hidden)
                    t.view(C, 2, w_out, -1)[:, :, pad:pad + w_in, :] = \
                        v_old.view(C, 2, w_in, -1)
                    loaded[k] = t
                elif k.endswith('cond.head.bias'):
                    C = v_old.shape[0] // (2 * w_in)
                    t = torch.zeros_like(v_new)
                    t.view(C, 2, w_out)[:, :, pad:pad + w_in] = \
                        v_old.view(C, 2, w_in)
                    loaded[k] = t
                else:
                    raise SystemExit(f"[init-ckpt] non-widenable mismatch on "
                                     f"{k}: {tuple(v_old.shape)} vs "
                                     f"{tuple(v_new.shape)}")
            model.load_state_dict(loaded)
            print(f"[deriv-train] warm start from {args.init_ckpt} "
                  f"(epoch={ck.get('epoch')}) WIDENED {w_in}->{w_out} "
                  f"(exact function preserved; outer taps start at 0)")
        else:
            raise SystemExit(f"[init-ckpt] ckpt grad_kernel {w_in} > "
                             f"--grad-kernel {w_out}: narrowing unsupported")

    # ---- per-order loss weights (e.g. 1,1,0 = ditch N3dot) ----
    order_w = None
    if args.order_weights is not None:
        order_w = torch.tensor([float(v) for v in
                                args.order_weights.split(',')],
                               dtype=adtype, device=args.device)
        if order_w.numel() != args.out_orders or (order_w < 0).any() \
                or order_w.sum() <= 0:
            raise SystemExit(f"--order-weights needs {args.out_orders} "
                             f"non-negative values with positive sum")
        print(f"[deriv-train] order-weighted loss: "
              f"{dict(zip(ORDER_NAMES, order_w.tolist()))} "
              f"(weighted mean drives loss AND best-ckpt selection)")

    trainable = [q for q in model.parameters() if q.requires_grad]
    n_params = sum(q.numel() for q in trainable)
    print(f"[deriv-train] {args.model} trainable params={n_params:,}  in_ch={in_ch}  "
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
    # F1: val_{o} columns stay the floored per-order MEANS (backward compatible);
    # val_med_{o} medians are APPENDED after elapsed_s (positional parsers unaffected).
    log.write_text('epoch,lr,train_relL2,val_relL2,best_val,'
                   + ','.join(f'val_{o}' for o in onames) + ',elapsed_s,'
                   + ','.join(f'val_med_{o}' for o in onames) + '\n')

    def run_epoch(loader, train):
        model.train(train)
        tot = 0.0; per = np.zeros(args.out_orders); nb = 0
        samp = []   # per-sample (B,out_orders) floored rel-L2 -> per-order medians
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y, regime in loader:
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True).to(adtype)
                regime = regime.to(args.device, non_blocking=True)
                dT = regime[:, 0].to(x.dtype)
                dxb = regime[:, 4].to(x.dtype)             # MULTIGRID: per-sample dx
                dyb = regime[:, 5].to(x.dtype)             # MULTIGRID: per-sample dy
                # norm floor: rel_floor * member-median ||t_c|| (regime[:,6:9]).
                # Caps leverage of near-zero-target samples (Ndot zero-crossings).
                floor = (args.rel_floor * regime[:, 6:6 + args.out_orders]
                         if args.rel_floor > 0 else None)
                nd = model(x, dt=dT, dx=dxb, dy=dyb)        # (B, out_orders, H, W)
                project = project_by_shape[(x.shape[-2], x.shape[-1])]  # by batch shape
                nd = project(nd)
                if order_w is None:
                    loss = relative_l2_perchannel(nd, y, floor)
                else:
                    vec = relative_l2_perchannel_vec(nd, y, floor)
                    loss = (order_w * vec).sum() / order_w.sum()
                if train:
                    optim.zero_grad(); loss.backward(); optim.step()
                tot += loss.item()
                r = relative_l2_persample(nd, y, floor).detach().cpu().numpy()
                per += r.mean(axis=0)   # == relative_l2_perchannel_vec (batch mean)
                samp.append(r)
                nb += 1
        nb = max(nb, 1)
        # headline/best = mean per-order FLOORED rel-L2 (same denominator as the
        # loss); med = per-order MEDIAN over all samples (F1, printed+logged).
        med = (np.median(np.concatenate(samp, axis=0), axis=0) if samp
               else np.zeros(args.out_orders))
        return tot / nb, per / nb, med

    best = float('inf'); t0 = time.time()
    for ep in range(args.epochs):
        te0 = time.time()
        # multigrid: advance the GridHomogeneousBatchSampler so batches reshuffle
        # each epoch (set_epoch is a no-op / absent on the single-grid DataLoader path).
        for _ld in (tl, vl, te):
            _bs = getattr(_ld, 'batch_sampler', None)
            if _bs is not None and hasattr(_bs, 'set_epoch'):
                _bs.set_epoch(ep)
        tr, _, _ = run_epoch(tl, True)
        va, va_per, va_med = run_epoch(vl, False)
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
                    + f',{time.time()-t0:.1f},'
                    + ','.join(f'{v:.6e}' for v in va_med) + '\n')
        if ep % args.print_every == 0 or improved:
            brk = ' '.join(f'{o}={v:.3e}' for o, v in zip(onames, va_per))
            brm = ' '.join(f'{o}={v:.3e}' for o, v in zip(onames, va_med))
            print(f"  ep {ep:4d} {'*' if improved else ' '}  "
                  f"train={tr:.4e}  val={va:.4e}  best={best:.4e} (floored-mean relL2/order)  "
                  f"[mean: {brk}]  [med: {brm}]  ({time.time()-te0:.1f}s)")

    print(f"\n[deriv-train] done in {(time.time()-t0)/60:.1f} min, best val={best:.4e}")
    ckpt = torch.load(run_dir / 'best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    test, test_per, test_med = run_epoch(te, False)
    brk = ' '.join(f'{o}={v:.4e}' for o, v in zip(onames, test_per))
    brm = ' '.join(f'{o}={v:.4e}' for o, v in zip(onames, test_med))
    print(f"[deriv-train] TEST relL2={test:.4e}  [mean: {brk}]  [med: {brm}]")
    with open(log, 'a') as f:
        f.write(f'-1,0.0,0.0,{test:.6e},{best:.6e},'
                + ','.join(f'{v:.6e}' for v in test_per) + f',{time.time()-t0:.1f},'
                + ','.join(f'{v:.6e}' for v in test_med) + '\n')


if __name__ == '__main__':
    main()
