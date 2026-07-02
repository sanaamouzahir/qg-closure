"""
train.py - Train a closure NN to predict f_NN_target from input fields.

Usage:
    python train.py \\
        --root-dir ../data/decaying_turbulence_dT_1em3 \\
        --model unet \\
        --input-fields omega_0 psi_0 \\
        --batch-size 4 \\
        --epochs 200 \\
        --lr 3e-4 \\
        [--normalize]  [--device cuda]  [--num-workers 2]

    # cheap multi-output N-derivative closure:
    python train.py \\
        --root-dir ../data/forced_turbulence_dT_1em3 \\
        --model cheap_deriv \\
        --input-fields omega_0 omega_m1 omega_m2 psi_0 psi_m1 psi_m2 \\
        --target-fields N_dot_0_anal N_ddot_0_anal N_3dot_0_anal \\
        --loss rel_l2 \\
        [--dealias-pred]    # project predictions onto the 2/3 band before the loss

Saves under root_dir/training_runs/<run_name>/:
    config.json       - reproducibility
    model_best.pt     - best val checkpoint
    model_last.pt     - latest checkpoint
    log.csv           - per-epoch train/val/test loss + LR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import ClosureDataset, make_loaders
from model import build_model


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2: ||pred-target||_2 / ||target||_2 averaged over batch."""
    flat_pred = pred.flatten(start_dim=1)
    flat_targ = target.flatten(start_dim=1)
    num = torch.norm(flat_pred - flat_targ, dim=1)
    den = torch.norm(flat_targ, dim=1).clamp_min(1e-30)
    return (num / den).mean()


def relative_l2_perchannel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample, per-channel relative L2, averaged over channels and batch.

    Use for multi-channel targets (e.g. [Ndot, Nddot, N3dot]) whose channels
    differ by orders of magnitude: a flat relative L2 would let the largest
    channel dominate, this normalizes each channel independently.
    """
    p = pred.flatten(start_dim=2)
    t = target.flatten(start_dim=2)
    num = torch.norm(p - t, dim=2)
    den = torch.norm(t, dim=2).clamp_min(1e-30)
    return (num / den).mean()


def relative_l2_perchannel_vec(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(C,) per-operator relative L2, averaged over batch (diagnostic only).

    One entry per output channel, so each learned operator (Ndot, Nddot,
    N3dot, ...) can be tracked separately rather than lumped together.
    """
    p = pred.flatten(start_dim=2)
    t = target.flatten(start_dim=2)
    num = torch.norm(p - t, dim=2)               # (B, C)
    den = torch.norm(t, dim=2).clamp_min(1e-30)  # (B, C)
    return (num / den).mean(dim=0)               # (C,)


def count_macs(model, in_channels, Ny, Nx, device):
    """One-forward MAC count over Conv2d/Linear layers at the inference grid.

    NB: does NOT count elementwise products or einsum. For cheap_deriv add by
    hand ~ (n_time^2 * 2) MACs/grid-point for the Jacobian products.
    """
    macs = [0]
    handles = []

    def conv_hook(m, _inp, out):
        c_out, h, w = out.shape[-3:]
        kh, kw = m.kernel_size
        macs[0] += c_out * h * w * (m.in_channels // m.groups) * kh * kw

    def lin_hook(m, _inp, out):
        macs[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(lin_hook))

    was_training = model.training
    model.eval()
    mdtype = next(model.parameters()).dtype
    with torch.no_grad():
        model(torch.zeros(1, in_channels, Ny, Nx, device=device, dtype=mdtype))
    for h in handles:
        h.remove()
    model.train(was_training)
    return macs[0]


def evaluate(model, loader, device, criterion=None, project=None):
    if criterion is None:
        criterion = nn.MSELoss()
    if project is None:
        project = lambda p: p
    model.eval()
    total_loss = 0.0
    rel_vec = None
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = project(model(x))
            loss = criterion(pred, y)
            rv = relative_l2_perchannel_vec(pred, y)
            total_loss += loss.item()
            rel_vec = rv if rel_vec is None else rel_vec + rv
            n_batches += 1
    if n_batches == 0:
        return float('nan'), None
    return total_loss / n_batches, (rel_vec / n_batches).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-dir', type=Path, required=True,
                   help='dataset root (contains manifest.json + samples/)')
    p.add_argument('--run-name', type=str, default=None,
                   help='subdir name under root_dir/training_runs/. '
                        'Default: timestamp-based.')

    # Data
    p.add_argument('--input-fields', type=str, nargs='+', default=['omega_0'],
                   help='which fields to stack as NN input channels')
    p.add_argument('--target-field', type=str, default='f_NN_target')
    p.add_argument('--target-fields', type=str, nargs='+', default=None,
                   help='multi-channel target, e.g. N_dot_0_anal N_ddot_0_anal '
                        'N_3dot_0_anal. Overrides --target-field if given.')
    p.add_argument('--normalize', action='store_true',
                   help='normalize inputs and target by per-channel stats')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--dealias-pred', action='store_true',
                   help='project predictions onto the 2/3 band (same cutoff as '
                        'derivative.dealias) before the loss; use with dealiased '
                        'datasets. Deployed model stays FFT-free (rollout dealiases).')

    # Model
    p.add_argument('--model', type=str, default='unet',
                   choices=['cnn', 'unet', 'bilinear_closure', 'bilin', 'fixd_v2',
                            'cheap_deriv'])
    p.add_argument('--hidden-channels', type=int, default=64,
                   help='for cnn: hidden width')
    p.add_argument('--depth', type=int, default=6,
                   help='for cnn: number of conv blocks')
    p.add_argument('--base-channels', type=int, default=32,
                   help='for unet: width of the highest-res layer')
    p.add_argument('--kernel', type=int, default=3)
    p.add_argument('--grad-kernel', type=int, default=7,
                   help='odd width of the spatial central-difference stencil '
                        '(3=2nd order, 5=4th, 7=6th); wider narrows the '
                        'FD-vs-spectral gradient gap on high-k fields.')
    p.add_argument('--out-orders', type=int, default=3,
                   help='cheap_deriv: number of N-derivative output channels '
                        '(Ndot, Nddot, N3dot, ...). Must match #target-fields.')
    p.add_argument('--refine-channels', type=int, default=0,
                   help='cheap_deriv: width of optional spatial refinement '
                        '(0 = pure bilinear, cheapest)')
    p.add_argument('--corrector-channels', type=int, default=16,
                   help='cheap_deriv: width of the zero-init residual corrector '
                        'that mops up the FD-truncation residual the linear mix '
                        'cannot. 0 = OFF (pure physics, cheapest at rollout -- the '
                        'default). This is the ONLY knob that adds rollout cost; '
                        'keep it small (8-16) if you need it at all. Do NOT reuse '
                        '--hidden-channels here (that defaults to 64 for the CNN).')
    p.add_argument('--corrector-depth', type=int, default=2,
                   help='cheap_deriv: conv layers in the corrector when '
                        '--corrector-channels>0 (ignored when 0). 2 is plenty for '
                        'a local truncation correction.')
    p.add_argument('--physics-init', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='cheap_deriv: initialise the 1x1 mix to the analytic '
                        'binomial coefficients (default on). --no-physics-init '
                        'gives the random-init ablation.')

    # Training
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--lr-schedule', type=str, default='cosine',
                   choices=['cosine', 'plateau', 'none'])
    p.add_argument('--patience', type=int, default=20,
                   help='for plateau schedule: epochs without improvement')
    p.add_argument('--loss', type=str, default='mse',
                   choices=['mse', 'rel_l2'],
                   help='training loss: mse (default) or relative L2 '
                        '(per-channel when --target-fields is set)')

    # Compute
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--compute-dtype', type=str, default='float64',
                   choices=['float32', 'float64'],
                   help='training precision (packed path). float64 keeps the '
                        'high-order time-FD (N3dot) clean; pair with float64 '
                        'packed inputs. Model is tiny so f64 compute is cheap.')

    # Logging
    p.add_argument('--print-every', type=int, default=1)

    args = p.parse_args()

    # ---- setup ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.run_name is None:
        args.run_name = time.strftime('%Y%m%d_%H%M%S')
    run_dir = args.root_dir / 'training_runs' / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config for reproducibility
    config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"[train] run dir: {run_dir}")
    print(f"[train] config: {json.dumps(config_dict, indent=2)}")

    # Load dataset manifest for shape info
    with open(args.root_dir / 'manifest.json') as f:
        manifest = json.load(f)
    Ny, Nx = int(manifest['Ny']), int(manifest['Nx'])
    print(f"[train] grid: {Ny} x {Nx}")

    # optional: project predictions onto the resolved (2/3) band before the loss,
    # using the SAME cutoff as the solver/builder (derivative.alias_mask). The mask
    # multiply is out-of-place so it stays autograd-safe; to_spectral is rfftn, so
    # alias_mask is already the right (Ny, Nx//2+1) shape.
    if args.dealias_pred:
        from qg.solver.grid.cartesian import CartesianGrid
        from qg.solver.opt.derivative import Derivative
        from qg.solver.opt.basis import to_spectral, to_physical
        _grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=float(manifest['Lx']),
                              Ly=float(manifest['Ly']), device=args.device,
                              precision='float64')
        _keep = (~Derivative(_grid).alias_mask).to(args.device)
        def project(p):
            return to_physical(to_spectral(p) * _keep.to(device=p.device, dtype=p.dtype))
        print("[train] dealiasing predictions before loss (matches solver 2/3 rule)")
    else:
        def project(p):
            return p

    # ---- data ----
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_loaders(
        args.root_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        input_fields=tuple(args.input_fields), target_field=args.target_field,
        target_fields=args.target_fields,
        normalize=args.normalize,
        compute_dtype=args.compute_dtype,
    )
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ---- per-operator channel labels (for the diagnostic) ----
    if args.target_fields:
        ch_labels = [f.replace('_0_anal', '').replace('_anal', '')
                     for f in args.target_fields]
    else:
        ch_labels = [args.target_field]
    print(f"[train] target operators: {ch_labels}")

    # ---- model ----
    in_channels = len(args.input_fields)
    if args.model == 'cnn':
        model = build_model('cnn', in_channels=in_channels,
                            hidden_channels=args.hidden_channels,
                            depth=args.depth, kernel=args.kernel)
    elif args.model in ('bilinear_closure', 'bilin', 'fixd_v2'):
        from model_fixD import build_model as build_model_fixD
        model = build_model_fixD(args.model, in_channels=in_channels,
                                 hidden_channels=args.hidden_channels,
                                 kernel=args.kernel)
    elif args.model == 'cheap_deriv':
        from model_deriv_closure import build_model as build_model_deriv
        # Physical scalings so the FD/gradient features are in true units and the
        # ideal mix weights are the O(1) chain-rule binomials (not ~1e6+).
        dx = float(manifest['Lx']) / Nx
        dy = float(manifest['Ly']) / Ny
        dt = float(manifest['Delta_T'])
        print(f"[train] cheap_deriv spacings: dt={dt:.3e}  dx={dx:.6e}  dy={dy:.6e}")
        model = build_model_deriv('cheap_deriv', in_channels=in_channels,
                                  out_orders=args.out_orders,
                                  refine_channels=args.refine_channels,
                                  kernel=args.kernel,
                                  grad_kernel=args.grad_kernel,
                                  dt=dt, dx=dx, dy=dy,
                                  physics_init=args.physics_init,
                                  hidden_channels=args.corrector_channels,
                                  depth=args.corrector_depth)
    else:
        model = build_model('unet', in_channels=in_channels,
                            base_channels=args.base_channels, kernel=args.kernel)
    model = model.to(args.device)
    mdtype = torch.float64 if args.compute_dtype == 'float64' else torch.float32
    model = model.to(mdtype)
    n_params = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, in_channels, Ny, Nx, args.device)
    print(f"[train] model: {args.model}, {n_params:,} params, "
          f"{macs:,} MACs/forward at {Ny}x{Nx} "
          f"({macs/(Ny*Nx):.1f} MACs/grid-point)")

    # ---- optimizer ----
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    if args.lr_schedule == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    elif args.lr_schedule == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.5, patience=args.patience)
    else:
        sched = None
    if args.loss == 'rel_l2':
        criterion = relative_l2_perchannel if args.target_fields else relative_l2
    else:
        criterion = nn.MSELoss()

    # ---- training loop ----
    log_path = run_dir / 'log.csv'
    tr_cols = ','.join(f'train_rel_{c}' for c in ch_labels)
    va_cols = ','.join(f'val_rel_{c}' for c in ch_labels)
    with open(log_path, 'w') as f:
        f.write('epoch,lr,train_loss,val_loss,best_val,'
                f'{tr_cols},{va_cols},elapsed_s\n')

    best_val = float('inf')
    t_run_start = time.time()

    for epoch in range(args.epochs):
        t_epoch = time.time()
        # train
        model.train()
        train_loss = 0.0
        train_relvec = None
        n_b = 0
        for x, y in train_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            pred = project(model(x))
            loss = criterion(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
            with torch.no_grad():
                rv = relative_l2_perchannel_vec(pred, y)
            train_relvec = rv if train_relvec is None else train_relvec + rv
            n_b += 1
        train_loss /= max(n_b, 1)
        train_relvec = (train_relvec / max(n_b, 1)).cpu()

        # val (per-operator)
        val_loss, val_relvec = evaluate(model, val_loader, args.device, criterion, project)

        # schedule step
        cur_lr = optim.param_groups[0]['lr']
        if sched is not None:
            if args.lr_schedule == 'plateau':
                sched.step(val_loss)
            else:
                sched.step()

        # checkpoint best
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_rel_per_op': dict(zip(ch_labels, val_relvec.tolist())),
                'config': config_dict,
            }, run_dir / 'model_best.pt')

        # always save 'last'
        torch.save({
            'model_state': model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'config': config_dict,
        }, run_dir / 'model_last.pt')

        elapsed = time.time() - t_epoch
        total_elapsed = time.time() - t_run_start
        tr_vals = ','.join(f'{v:.6e}' for v in train_relvec.tolist())
        va_vals = ','.join(f'{v:.6e}' for v in val_relvec.tolist())
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{cur_lr:.6e},{train_loss:.6e},{val_loss:.6e},'
                    f'{best_val:.6e},{tr_vals},{va_vals},{total_elapsed:.1f}\n')

        if epoch % args.print_every == 0 or improved:
            tag = '*' if improved else ' '
            per_op = '  '.join(f'{c}={v:.3f}'
                               for c, v in zip(ch_labels, val_relvec.tolist()))
            print(f"  ep {epoch:4d} {tag}  lr={cur_lr:.2e}  "
                  f"train: loss={train_loss:.3e}  val: loss={val_loss:.3e}  "
                  f"best={best_val:.3e}  |  val rel/op: {per_op}  ({elapsed:.1f}s)")

    # ---- final test ----
    print(f"\n[train] training done in {(time.time()-t_run_start)/60:.1f} min, "
          f"best val={best_val:.4e}")
    print("[train] loading best model and evaluating on test set...")
    ckpt = torch.load(run_dir / 'model_best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_loss, test_relvec = evaluate(model, test_loader, args.device, criterion, project)
    per_op = '  '.join(f'{c}={v:.4f}' for c, v in zip(ch_labels, test_relvec.tolist()))
    print(f"[train] TEST  loss={test_loss:.4e}  |  rel/op: {per_op}")
    n_ch = len(ch_labels)
    zeros = ','.join(['0.0'] * n_ch)                       # train cols (n/a for test row)
    te_vals = ','.join(f'{v:.6e}' for v in test_relvec.tolist())
    with open(log_path, 'a') as f:
        f.write(f'-1,0.0,0.0,{test_loss:.6e},{best_val:.6e},'
                f'{zeros},{te_vals},{time.time()-t_run_start:.1f}\n')


if __name__ == '__main__':
    main()