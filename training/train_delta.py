"""
train_delta.py - train the EMPIRICAL temporal closure (the delta-R pivot).

Target is the empirical per-step correction
    delta = Phi_ref(omega; dT) - Phi_AB2CN2(omega, omega_m1; dT),
with the reference chosen by --reference:
    exact : Phi_ref = fine RK4 (~exact)   -> delta = R3,R4,R5,...      (match exact / RK4-at-dT/K)
    rk4   : Phi_ref = one coarse RK4 step  -> delta = R3,R4,R5-T5,...   (match RK4 stability)
sliced by slice_delta_sweep.py over a Delta_T sweep and pooled across the ensemble.

Hybrid prediction (agreed design):
    delta_pred = delta_anal(R3,R4 from predicted Ndot,Nddot,N3dot ; per-sample dT)
               + delta_tail(Delta_T/regime-conditioned corrector ; learns R5 and up)
Loss is relative-L2 on the dT^3-NORMALIZED field delta/dT^3 (an O(R3) target across
the whole sweep), so dT spans no longer wreck the loss scale.

dT is now a model input: it enters (1) the per-sample FD time-scaling inside the
cheap_deriv TimeFD, and (2) the dT^p closure weights in closure_increment_batched,
and conditions the tail corrector via FiLM on [dT, beta, nu, mu].

v1 scope: ONE grid per run (the cheap_deriv spatial stencil and the assembly operators
bake the grid spacing). Pass --grid to pick; default = the most common grid in the pool.
Mixed-grid (per-batch dx/L_hat) and the rollout-loss curriculum are follow-ups.

Usage:
    python train_delta.py \
        --sweep-roots $QG_DIR/training/data/ensemble_N5/*/sweep_dT_* \
        --reference exact --n-snapshots 4 --orders 3 4 \
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
import torch.nn as nn

from delta_dataset import make_delta_loaders
from model_deriv_closure import build_model
from closure_operators import closure_increment_batched
from build_training_data_mmap import J_phys

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative


# --------------------------------------------------------------------------- #
# Delta_T / regime-conditioned residual corrector (learns the R5+ tail)        #
# --------------------------------------------------------------------------- #

class DeltaTailNet(nn.Module):
    """Zero-init residual CNN, FiLM-conditioned on the (normalized) regime vector,
    predicting the all-orders tail delta - delta_anal (1 channel).

    Zero-init output => at epoch 0 delta_pred == the analytic R3,R4 assembly, so the
    physics-init cheap_deriv already lands near delta at small dT; training only ADDS
    the R5+ correction. Per-channel instance norm tames the ~5-decade spread across the
    [lags, N-derivs, delta_anal] feature stack.
    """

    def __init__(self, in_ch: int, n_regime: int = 4, hidden: int = 48,
                 depth: int = 4, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.norm = nn.GroupNorm(in_ch, in_ch, affine=False)
        self.film = nn.Sequential(nn.Linear(n_regime, hidden), nn.GELU(),
                                  nn.Linear(hidden, 2 * hidden))
        self.in_conv = nn.Conv2d(in_ch, hidden, kernel, padding=pad,
                                 padding_mode='circular')
        self.mid = nn.ModuleList(
            nn.Conv2d(hidden, hidden, kernel, padding=pad, padding_mode='circular')
            for _ in range(max(depth - 2, 0)))
        self.out = nn.Conv2d(hidden, 1, kernel, padding=pad, padding_mode='circular')
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # no-op at init
        self.act = nn.GELU()

    def forward(self, feat: torch.Tensor, regime_norm: torch.Tensor) -> torch.Tensor:
        g, b = self.film(regime_norm).chunk(2, dim=1)         # (B,hidden) each
        h = self.act(self.in_conv(self.norm(feat)))
        h = g[:, :, None, None] * h + b[:, :, None, None]      # FiLM
        for conv in self.mid:
            h = self.act(conv(h))
        return self.out(h)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def relative_l2(pred, target):
    """Batch-aggregate relative L2: one ratio over the whole batch, NOT a mean of
    per-sample ratios. Per-sample normalization blows up on quiescent anchors where
    ||target|| ~ 0 (a single near-zero-delta sample sends the mean to 1e4+) and, worse,
    weights the least-important samples hardest. The aggregate form is magnitude-weighted
    (active samples dominate, as they should for a closure), immune to small-norm samples,
    and still == 1.0 at zero-init (pred=0 => ||t||/||t|| = 1)."""
    p = pred.flatten(start_dim=1); t = target.flatten(start_dim=1)
    return torch.norm(p - t) / torch.norm(t).clamp_min(1e-30)


def normalize_regime(regime):
    """[dT, beta, nu, mu] -> roughly O(1) features for the FiLM head.
    log-scale the strictly-positive, wide-range dT/nu/mu; leave beta linear/10."""
    dT, beta, nu, mu = regime[:, 0], regime[:, 1], regime[:, 2], regime[:, 3]
    f = torch.stack([
        torch.log10(dT.clamp_min(1e-12)) + 3.0,        # 1e-3..1.5e-2 -> ~0..1.2
        beta / 2.0,                                    # 0..2.5 -> 0..1.25
        torch.log10(nu.clamp_min(1e-12)) + 4.0,        # ~1e-6..1e-4 -> ~-2..0
        torch.log10(mu.clamp_min(1e-12)) + 2.0,
    ], dim=1)
    return torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def build_grid_ops(man, device):
    Nx, Ny = int(man['Nx']), int(man['Ny'])
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=float(man['Lx']), Ly=float(man['Ly']),
                         device=device, precision='float64')
    der = Derivative(grid)
    for a in ('dx', 'dy', 'laplacian', 'inv_laplacian'):
        setattr(der, a, getattr(der, a).to(device))
    lap = der.laplacian                       # spectral symbol (H,W)
    betasym = der.dx * der.inv_laplacian       # beta-drift symbol (H,W)
    return der, lap, betasym, (Ny, Nx)


def lhat_batched(regime, lap, betasym, dtype):
    """Per-sample L_hat spectral symbol from [dT,beta,nu,mu]: (B,1,H,W).
    L = nu*lap - mu - beta*(dx*inv_lap)  (same form as build_L_hat, vectorized)."""
    beta = regime[:, 1].view(-1, 1, 1, 1).to(dtype)
    nu = regime[:, 2].view(-1, 1, 1, 1).to(dtype)
    mu = regime[:, 3].view(-1, 1, 1, 1).to(dtype)
    lap_ = lap.reshape(1, 1, *lap.shape[-2:])         # -> (1,1,H,W) regardless of native rank
    bsym_ = betasym.reshape(1, 1, *betasym.shape[-2:])
    return nu * lap_ - mu - beta * bsym_


def physics_part(model, x, regime, der, lap, betasym, scheme, orders, adtype,
                 pure=False, dx=None, dy=None):
    """Reference-INDEPENDENT analytic part: returns (delta_anal, feat).

    delta_anal = R3,R4 assembled from the model's N-derivatives at per-sample dT;
    feat = [lags, N-derivs, delta_anal] for the corrector. This is identical for
    every reference -- exact and rk4 only differ at R5+, which is the corrector's
    job -- so in 'both' mode it is computed ONCE and the corrector runs per-bit.
    """
    n = x.shape[1] // 2
    dT = regime[:, 0].to(x.dtype)
    omega0 = x[:, 0:1].to(adtype)
    nd = model(x, dt=dT, dx=dx, dy=dy)               # (B, out_orders, H, W)
    if pure:
        # pure-empirical: no L^k assembly, no amplification. delta_anal := 0,
        # corrector learns delta directly. nd kept as (free) input features.
        delta_anal = torch.zeros_like(omega0)
    else:
        psi0 = x[:, n:n + 1].to(adtype)
        N0 = -1.0 * J_phys(psi0, omega0, der)        # F dropped (static; tiny bias)
        Nderivs = [nd[:, k:k + 1].to(adtype) for k in range(nd.shape[1])]
        L_hat = lhat_batched(regime, lap, betasym, adtype)
        delta_anal = closure_increment_batched(scheme, omega0, N0, Nderivs, L_hat,
                                               dT.to(adtype), orders=orders)
    feat = torch.cat([x.to(nd.dtype), nd, delta_anal.to(nd.dtype)], dim=1)
    return delta_anal, feat


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-roots', type=Path, nargs='+', required=True,
                   help='sweep_dT_* dirs (across members and Delta_T)')
    p.add_argument('--run-name', type=str, default=None)
    p.add_argument('--out-root', type=Path, default=None,
                   help='where training_runs/<run> lands (default: first root parent)')
    p.add_argument('--reference', choices=['exact', 'rk4', 'both'], default='exact',
                   help="target reference. 'both' trains ONE reference-conditioned "
                        "corrector on delta_exact AND delta_rk4 (a ref bit is appended "
                        "to the FiLM input); its two modes then differ by tau_RK4.")
    p.add_argument('--grid', type=str, default=None,
                   help="restrict to one grid 'NyxNx' (default: most common in pool)")

    p.add_argument('--n-snapshots', type=int, default=4,
                   help='lags fed to the model (>=4 for N3dot in R4; 4 is the sweet '
                        'spot -- higher orders are FD-noisy and unused for R3,R4)')
    p.add_argument('--orders', type=int, nargs='+', default=[3, 4],
                   help='analytic R_p orders to assemble; the rest is the learned tail')
    p.add_argument('--scheme', type=str, default='ab2cn2')
    p.add_argument('--out-orders', type=int, default=3)
    p.add_argument('--grad-kernel', type=int, default=7)
    p.add_argument('--no-physics-init', action='store_true')
    p.add_argument('--freeze-stencils', action='store_true', default=True,
                   help='freeze the FD stencils (recommended across a dT sweep)')
    p.add_argument('--freeze-physics', action='store_true',
                   help='freeze the WHOLE cheap_deriv (stencils + physics-init mix) so '
                        'delta_anal is a fixed reference-independent prior and the '
                        'corrector carries all the learning. Recommended for the '
                        'exact-vs-rk4 comparison -- keeps delta_anal bit-identical, so '
                        'corrector(exact) - corrector(rk4) is exactly the empirical tau_RK4.')
    p.add_argument('--pure-empirical', action='store_true',
                   help='zero delta_anal entirely; the corrector learns delta directly '
                        'from [lags, FD-derivs]. No L^k assembly, no Sense-A amplification. '
                        'epoch-0 relL2 == 1.0 by construction. This is the clean pivot baseline.')
    p.add_argument('--tail-hidden', type=int, default=48)
    p.add_argument('--tail-depth', type=int, default=4)

    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--compute-dtype', choices=['float32', 'float64'], default='float64')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--print-every', type=int, default=5)
    args = p.parse_args()

    # ---- resolve roots; mixed grids handled per-batch (signature-homogeneous) ----
    roots = [r for r in args.sweep_roots if (r / 'manifest.json').exists()
             and (r / 'packed' / 'inputs.npy').exists()]
    if not roots:
        raise SystemExit("no valid sweep roots (need manifest.json + packed/inputs.npy)")
    grids = {}
    for r in roots:
        m = json.loads((r / 'manifest.json').read_text())
        grids.setdefault(f"{int(m['Ny'])}x{int(m['Nx'])}", []).append(r)
    if args.grid:                                  # optional: restrict to one resolution
        roots = grids[args.grid]
        print(f"[delta-train] grid={args.grid} (restricted)  roots={len(roots)}")
    else:
        print(f"[delta-train] ALL grids  roots={len(roots)}  "
              f"resolutions={ {k: len(v) for k, v in grids.items()} }")

    # ---- data ----
    tl, vl, te, train_ds, val_ds, test_ds = make_delta_loaders(
        roots, batch_size=args.batch_size, num_workers=args.num_workers,
        n_snapshots=args.n_snapshots, reference=args.reference,
        compute_dtype=args.compute_dtype, seed=0)

    adtype = torch.float64 if args.compute_dtype == 'float64' else torch.float32
    # per-signature spectral ops + spacing, keyed by (Ny,Nx,Lx,Ly). Built once per
    # unique grid/domain in the pool; selected per batch in run_epoch. This is what
    # lets one model train over mixed 256^2/512^2 (and mixed Lx) members correctly:
    # each batch is signature-homogeneous, so its der/lap/betasym/dx/dy are unique.
    sig_ops = {}
    for ds in train_ds.subsets:
        if ds.sig in sig_ops:
            continue
        der_s, lap_s, bsym_s, _ = build_grid_ops(ds.man, args.device)
        sig_ops[ds.sig] = (der_s, lap_s.to(adtype), bsym_s.to(adtype), ds.dx, ds.dy)
    print(f"[delta-train] built ops for {len(sig_ops)} unique grid signature(s): "
          f"{sorted((s[0], s[1]) for s in sig_ops)}")
    man0 = train_ds.subsets[0].man               # reference spacing for model build
    dx0 = float(man0['Lx']) / int(man0['Nx']); dy0 = float(man0['Ly']) / int(man0['Ny'])

    # ---- model + corrector ----
    in_ch = 2 * args.n_snapshots
    learnable = not (args.freeze_stencils or args.freeze_physics)
    model = build_model('cheap_deriv', in_channels=in_ch, out_orders=args.out_orders,
                        grad_kernel=args.grad_kernel, dt=float(man0['Delta_T']),
                        dx=dx0, dy=dy0, physics_init=not args.no_physics_init,
                        learnable_stencils=learnable).to(args.device).to(adtype)
    if args.freeze_physics:
        for q in model.parameters():
            q.requires_grad_(False)
        print("[delta-train] cheap_deriv FROZEN -> delta_anal is a fixed prior")
    both = args.reference == 'both'
    n_regime = 5 if both else 4                            # +1 reference bit in 'both'
    tail_in = in_ch + args.out_orders + 1                  # [lags, N-derivs, delta_anal]
    corrector = DeltaTailNet(tail_in, n_regime=n_regime, hidden=args.tail_hidden,
                             depth=args.tail_depth).to(args.device).to(adtype)
    trainable = [q for q in list(model.parameters()) + list(corrector.parameters())
                 if q.requires_grad]
    n_params = sum(q.numel() for q in trainable)
    print(f"[delta-train] trainable params={n_params:,}  in_ch={in_ch}  "
          f"orders={args.orders}  ref={args.reference}  dtype={args.compute_dtype}")

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    out_root = args.out_root or roots[0].parent.parent
    run = args.run_name or f"delta_{args.reference}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = out_root / 'training_runs' / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(vars(args) | {
        'grids': sorted(f"{s[0]}x{s[1]}@Lx{s[2]}" for s in sig_ops),
        'roots': [str(r) for r in roots]}, indent=2, default=str))
    log = run_dir / 'log.csv'
    log.write_text('epoch,lr,train_relL2,val_relL2,best_val,elapsed_s\n')

    def run_epoch(loader, train):
        model.train(train); corrector.train(train)
        tot = tot_e = tot_r = 0.0; nb = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                if both:
                    x, ye, yr, regime = batch
                else:
                    x, y, regime = batch
                x = x.to(args.device, non_blocking=True)
                regime = regime.to(args.device, non_blocking=True)
                dT3 = (regime[:, 0] ** 3).view(-1, 1, 1, 1).to(adtype)
                # select per-grid ops from the batch signature (regime[:,4:8]);
                # batch is signature-homogeneous, so row 0 keys the whole batch.
                sig = (int(regime[0, 4]), int(regime[0, 5]),
                       round(float(regime[0, 6]), 4), round(float(regime[0, 7]), 4))
                der, lap, betasym, dx_b, dy_b = sig_ops[sig]
                delta_anal, feat = physics_part(model, x, regime, der, lap, betasym,
                                                args.scheme, tuple(args.orders), adtype,
                                                pure=args.pure_empirical, dx=dx_b, dy=dy_b)
                base = normalize_regime(regime).to(feat.dtype)
                if both:
                    ye = ye.to(args.device).to(adtype); yr = yr.to(args.device).to(adtype)
                    one = torch.ones(x.shape[0], 1, device=x.device, dtype=feat.dtype)
                    de = delta_anal + dT3 * corrector(feat, torch.cat([base, 0 * one], 1)).to(adtype)
                    dr = delta_anal + dT3 * corrector(feat, torch.cat([base, one], 1)).to(adtype)
                    le = relative_l2(de / dT3, ye / dT3)
                    lr = relative_l2(dr / dT3, yr / dT3)
                    loss = le + lr
                    tot_e += le.item(); tot_r += lr.item()
                else:
                    y = y.to(args.device).to(adtype)
                    dp = delta_anal + dT3 * corrector(feat, base).to(adtype)
                    loss = relative_l2(dp / dT3, y / dT3)        # corrector predicts delta/dT^3
                if train:
                    optim.zero_grad(); loss.backward(); optim.step()
                tot += loss.item(); nb += 1
        nb = max(nb, 1)
        # headline = MEAN relative-L2 per reference leg (0..1 interpretable; ~1 = zero
        # prediction). backprop still used the summed loss above. per-leg returned too.
        if both:
            return (0.5 * (tot_e + tot_r) / nb, (tot_e / nb, tot_r / nb))
        return (tot / nb, None)

    best = float('inf'); t0 = time.time()
    for ep in range(args.epochs):
        te0 = time.time()
        tr, _ = run_epoch(tl, True)
        va, va_aux = run_epoch(vl, False)
        sched.step()
        improved = va < best
        if improved:
            best = va
            torch.save({'model': model.state_dict(), 'corrector': corrector.state_dict(),
                        'epoch': ep, 'val': va, 'config': vars(args),
                        'grids': sorted(f"{s[0]}x{s[1]}" for s in sig_ops)},
                       run_dir / 'best.pt')
        torch.save({'model': model.state_dict(), 'corrector': corrector.state_dict(),
                    'epoch': ep, 'val': va}, run_dir / 'last.pt')
        with open(log, 'a') as f:
            f.write(f'{ep},{optim.param_groups[0]["lr"]:.3e},{tr:.6e},{va:.6e},'
                    f'{best:.6e},{time.time()-t0:.1f}\n')
        if ep % args.print_every == 0 or improved:
            extra = f"  [exact={va_aux[0]:.3e} rk4={va_aux[1]:.3e}]" if va_aux else ""
            print(f"  ep {ep:4d} {'*' if improved else ' '}  "
                  f"train={tr:.4e}  val={va:.4e}  best={best:.4e} (mean relL2/leg){extra}  "
                  f"({time.time()-te0:.1f}s)")

    print(f"\n[delta-train] done in {(time.time()-t0)/60:.1f} min, best val={best:.4e}")
    ckpt = torch.load(run_dir / 'best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['model']); corrector.load_state_dict(ckpt['corrector'])
    test, test_aux = run_epoch(te, False)
    extra = f"  [exact={test_aux[0]:.4e} rk4={test_aux[1]:.4e}]" if test_aux else ""
    print(f"[delta-train] TEST relL2={test:.4e}{extra}")
    with open(log, 'a') as f:
        f.write(f'-1,0.0,0.0,{test:.6e},{best:.6e},{time.time()-t0:.1f}\n')


if __name__ == '__main__':
    main()