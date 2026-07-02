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


def evaluate(model, loader, device, criterion=None):
    if criterion is None:
        criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    total_relL2 = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            r = relative_l2(pred, y)
            total_loss += loss.item()
            total_relL2 += r.item()
            n_batches += 1
    if n_batches == 0:
        return float('nan'), float('nan')
    return total_loss / n_batches, total_relL2 / n_batches


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
    p.add_argument('--normalize', action='store_true',
                   help='normalize inputs and target by per-channel stats')
    p.add_argument('--num-workers', type=int, default=2)

    # Model
    p.add_argument('--model', type=str, default='unet',
                   choices=['cnn', 'unet', 'bilinear_closure', 'bilin', 'fixd_v2'])
    p.add_argument('--hidden-channels', type=int, default=64,
                   help='for cnn: hidden width')
    p.add_argument('--depth', type=int, default=6,
                   help='for cnn: number of conv blocks')
    p.add_argument('--base-channels', type=int, default=32,
                   help='for unet: width of the highest-res layer')
    p.add_argument('--kernel', type=int, default=3)

    # Training
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--lr-schedule', type=str, default='cosine',
                   choices=['cosine', 'plateau', 'none'])
    p.add_argument('--patience', type=int, default=20,
                   help='for plateau schedule: epochs without improvement')

    # Compute
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=0)

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

    # ---- data ----
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_loaders(
        args.root_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        input_fields=tuple(args.input_fields), target_field=args.target_field,
        normalize=args.normalize,
    )
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

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
    else:
        model = build_model('unet', in_channels=in_channels,
                            base_channels=args.base_channels, kernel=args.kernel)
    model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model: {args.model}, {n_params:,} params")

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
    criterion = nn.MSELoss()

    # ---- training loop ----
    log_path = run_dir / 'log.csv'
    with open(log_path, 'w') as f:
        f.write('epoch,lr,train_loss,train_relL2,val_loss,val_relL2,best_val,elapsed_s\n')

    best_val = float('inf')
    t_run_start = time.time()

    for epoch in range(args.epochs):
        t_epoch = time.time()
        # train
        model.train()
        train_loss = 0.0
        train_rel  = 0.0
        n_b = 0
        for x, y in train_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
            with torch.no_grad():
                train_rel += relative_l2(pred, y).item()
            n_b += 1
        train_loss /= max(n_b, 1)
        train_rel /= max(n_b, 1)

        # val
        val_loss, val_rel = evaluate(model, val_loader, args.device, criterion)

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
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{cur_lr:.6e},{train_loss:.6e},{train_rel:.6e},'
                    f'{val_loss:.6e},{val_rel:.6e},{best_val:.6e},{total_elapsed:.1f}\n')

        if epoch % args.print_every == 0 or improved:
            tag = '*' if improved else ' '
            print(f"  ep {epoch:4d} {tag}  lr={cur_lr:.2e}  "
                  f"train: loss={train_loss:.3e} rel={train_rel:.3e}  "
                  f"val: loss={val_loss:.3e} rel={val_rel:.3e}  "
                  f"best={best_val:.3e}  ({elapsed:.1f}s)")

    # ---- final test ----
    print(f"\n[train] training done in {(time.time()-t_run_start)/60:.1f} min, "
          f"best val={best_val:.4e}")
    print("[train] loading best model and evaluating on test set...")
    ckpt = torch.load(run_dir / 'model_best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_loss, test_rel = evaluate(model, test_loader, args.device, criterion)
    print(f"[train] TEST  loss={test_loss:.4e}  relL2={test_rel:.4e}")
    with open(log_path, 'a') as f:
        f.write(f'-1,0.0,0.0,0.0,{test_loss:.6e},{test_rel:.6e},{best_val:.6e},'
                f'{time.time()-t_run_start:.1f}\n')


if __name__ == '__main__':
    main()
