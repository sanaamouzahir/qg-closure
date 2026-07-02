#!/usr/bin/env python
"""
diagnose_training_plateau.py
============================
Implements the plateau-diagnosis plan from the prior conversation:

  (A) Loss curve summary stats (train/val/best, plateau detection).
  (B) Train vs val crossover: val < train -> underfit, val > train -> overfit.
  (C) Per-sample val rel-L2 binned by ||target||  -> magnitude bias?
  (D) Histogram of ||f_NN_target|| across the dataset -> outliers / dead frames?
  (E) Residual panels: best / 25th-pct / median / worst val sample.
  (F) Average spectrum: pred vs target  -> where in k-space the model fails.

Outputs into <run-dir>/diagnosis/:
  loss_curves.png
  target_magnitude_histogram.png
  val_rel_l2_by_magnitude.png
  residuals_panel.png
  spectra_comparison.png
  summary.txt

Usage on the cluster:
  cd $QG_DIR/training
  python diagnose_training_plateau.py \
      --run-dir   $QG_DIR/training/data/decaying_turbulence_dT_1em3_fixD_v2_OLD_f32bug/training_runs/<run_name> \
      --root-dir  $QG_DIR/training/data/decaying_turbulence_dT_1em3_fixD_v2_OLD_f32bug \
      --device    cuda

Required to be importable from CWD (i.e., $QG_DIR/training/):
  dataset.py, model.py, model_fixD.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# (A) loss-curve diagnostics from log.csv (no model needed)
# =========================================================================

def read_log(log_path: Path) -> dict:
    cols: dict[str, list] = {}
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v != '' else float('nan'))
    return cols


def plot_loss_curves(cols: dict, out_path: Path) -> None:
    epochs = np.array(cols['epoch'])
    valid = epochs >= 0  # exclude the test row at epoch -1
    epochs = epochs[valid]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.semilogy(epochs, np.array(cols['train_loss'])[valid], label='train', alpha=0.85)
    ax.semilogy(epochs, np.array(cols['val_loss'])[valid], label='val', alpha=0.85)
    ax.semilogy(epochs, np.array(cols['best_val'])[valid], 'k--', lw=1, label='best val')
    ax.set_xlabel('epoch')
    ax.set_ylabel(r'$\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N}\sum_i (f_i^{\mathrm{pred}} - f_i^{\mathrm{target}})^2$')
    ax.set_title(r'MSE loss')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    train_rel = np.array(cols['train_relL2'])[valid]
    val_rel = np.array(cols['val_relL2'])[valid]
    ax.plot(epochs, train_rel, label='train', alpha=0.85)
    ax.plot(epochs, val_rel, label='val', alpha=0.85)
    ax.set_xlabel('epoch')
    ax.set_ylabel(r'$\mathrm{rel.}\,L^2 = \|f^{\mathrm{pred}}-f^{\mathrm{target}}\|_2 \,/\, \|f^{\mathrm{target}}\|_2$')
    ax.set_title('Relative L2 error')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(epochs, np.array(cols['lr'])[valid])
    ax.set_xlabel('epoch')
    ax.set_ylabel('learning rate')
    ax.set_title(r'Learning rate:  $\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\,[1 + \cos(\pi t / T_{\max})]$')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, val_rel / np.maximum(train_rel, 1e-30))
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('epoch')
    ax.set_ylabel(r'$\mathrm{rel.}\,L^2_{\mathrm{val}} \,/\, \mathrm{rel.}\,L^2_{\mathrm{train}}$')
    ax.set_title('Val / train ratio  (>1 overfit, <1 underfit)')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def loss_summary_stats(cols: dict) -> str:
    epochs = np.array(cols['epoch'])
    valid = epochs >= 0
    val_rel = np.array(cols['val_relL2'])[valid]
    train_rel = np.array(cols['train_relL2'])[valid]
    n = len(val_rel)
    if n == 0:
        return "No valid epochs in log."

    last10 = max(0, n - 10)
    plateau_val = float(val_rel[last10:].mean())
    plateau_std = float(val_rel[last10:].std())
    plateau_train = float(train_rel[last10:].mean())
    initial_val = float(val_rel[:5].mean()) if n >= 5 else float(val_rel[0])
    delta = initial_val - plateau_val

    lines = [
        "=" * 60,
        "(A) LOSS CURVE SUMMARY",
        "=" * 60,
        f"  total epochs                 : {n}",
        f"  initial val rel-L2 (ep 0-4)  : {initial_val:.4f}",
        f"  plateau val rel-L2 (last 10) : {plateau_val:.4f}  (std {plateau_std:.4f})",
        f"  plateau train rel-L2         : {plateau_train:.4f}",
        f"  total improvement            : {delta:+.4f}  ({100*delta/max(initial_val,1e-30):+.1f}%)",
        "",
    ]

    # (B) under/overfit diagnosis
    if plateau_val < plateau_train:
        lines.append("  >> val < train: UNDERFIT  (model not memorizing)")
        lines.append("     not a capacity issue per se -- could be signal-bound.")
    elif plateau_val > 1.5 * plateau_train:
        lines.append("  >> val >> train: OVERFIT")
        lines.append("     reduce capacity, add regularization, or get more data.")
    else:
        lines.append("  >> val ~ train: balanced regime.")

    # plateau character
    if plateau_val == 0:
        plateau_val = 1e-30
    if plateau_std / plateau_val < 0.01:
        lines.append("  >> plateau is FLAT (<1% std over last 10 epochs).")
        lines.append("     consider warm restarts / capacity bump / different loss.")
    elif plateau_std / plateau_val > 0.05:
        lines.append("  >> plateau is NOISY (>5% std over last 10 epochs).")
        lines.append("     batch-to-batch variance dominating; LR may be too high.")
    else:
        lines.append("  >> plateau still has some descent.")
    return "\n".join(lines)


# =========================================================================
# (D) target magnitude histogram (no model needed)
# =========================================================================

def compute_target_magnitudes(root_dir: Path, target_field: str = 'f_NN_target',
                              max_samples: int = 2000):
    files = sorted((root_dir / 'samples').glob('sample_*.npz'))[:max_samples]
    mags = []
    for f in files:
        d = np.load(f)
        if target_field in d.files:
            t = d[target_field].astype(np.float64).ravel()
            mags.append(float(np.sqrt(np.mean(t ** 2))))
        else:
            mags.append(np.nan)
    return np.array(mags), files


def plot_target_histogram(mags: np.ndarray, out_path: Path):
    valid = np.isfinite(mags)
    if not valid.any():
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(mags[valid], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(r'$\|f^{\mathrm{target}}_{\mathrm{NN}}\|_{\mathrm{RMS}}$')
    axes[0].set_ylabel('count')
    axes[0].set_title(f'Target magnitudes  (n = {valid.sum()})')
    axes[0].grid(alpha=0.3)

    pos = mags[valid] > 0
    if pos.any():
        axes[1].hist(
            mags[valid][pos],
            bins=np.logspace(np.log10(mags[valid][pos].min()),
                             np.log10(mags[valid][pos].max()), 50),
            edgecolor='black', alpha=0.7,
        )
        axes[1].set_xscale('log')
    axes[1].set_xlabel(r'$\|f^{\mathrm{target}}_{\mathrm{NN}}\|_{\mathrm{RMS}}$    (log)')
    axes[1].set_ylabel('count')
    axes[1].set_title('Same on log axis')
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        'min': float(mags[valid].min()),
        'max': float(mags[valid].max()),
        'mean': float(mags[valid].mean()),
        'std': float(mags[valid].std()),
        'median': float(np.median(mags[valid])),
        'n_zero_or_tiny': int((mags[valid] < 0.01 * np.median(mags[valid])).sum()),
    }


# =========================================================================
# Model loading helper (used by C, E, F)
# =========================================================================

def _load_model_and_loaders(run_dir: Path, root_dir: Path, device: str):
    sys.path.insert(0, str(Path('.').resolve()))

    import torch
    with open(run_dir / 'config.json') as f:
        config = json.load(f)

    model_name = config['model']
    in_channels = len(config['input_fields'])
    hidden = config.get('hidden_channels', 64)
    kernel = config.get('kernel', 3)

    if model_name in ('bilinear_closure', 'bilin', 'fixd_v2'):
        from model_fixD import build_model
        model = build_model(model_name, in_channels=in_channels,
                            hidden=hidden, kernel=kernel)
    else:
        from model import build_model
        if model_name == 'unet':
            model = build_model('unet', in_channels=in_channels,
                                base_channels=config.get('base_channels', 32),
                                kernel=kernel)
        else:
            model = build_model('cnn', in_channels=in_channels,
                                hidden_channels=hidden,
                                depth=config.get('depth', 6), kernel=kernel)
    model = model.to(device)
    ckpt = torch.load(run_dir / 'model_best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    from dataset import make_loaders
    train_loader, val_loader, test_loader, _, _, _ = make_loaders(
        root_dir, batch_size=1, num_workers=0,
        input_fields=tuple(config['input_fields']),
        target_field=config['target_field'],
        normalize=config.get('normalize', False),
    )
    return model, val_loader, config


# =========================================================================
# (C) per-sample val rel-L2 vs target magnitude
# =========================================================================

def per_sample_val_errors(model, val_loader, device):
    import torch
    rel_l2, tgt_rms, pred_rms = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device); y = y.to(device)
            p = model(x)
            num = float(torch.norm(p - y).item())
            den = float(torch.norm(y).item())
            rel_l2.append(num / max(den, 1e-30))
            tgt_rms.append(float(torch.sqrt((y ** 2).mean()).item()))
            pred_rms.append(float(torch.sqrt((p ** 2).mean()).item()))
    return np.array(rel_l2), np.array(tgt_rms), np.array(pred_rms)


def plot_rel_l2_by_magnitude(rel_l2, tgt_rms, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(tgt_rms, rel_l2, alpha=0.4, s=10)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\|f^{\mathrm{target}}\|_{\mathrm{RMS}}$')
    axes[0].set_ylabel(r'$\mathrm{rel.}\,L^2$')
    axes[0].set_title(r'Per-sample val rel. $L^2$ vs target magnitude')
    axes[0].grid(alpha=0.3)

    n_bins = 10
    pos = tgt_rms > 0
    if pos.any():
        log_mags = np.log10(np.maximum(tgt_rms[pos], 1e-30))
        bins = np.linspace(log_mags.min(), log_mags.max(), n_bins + 1)
        centers = 0.5 * (bins[1:] + bins[:-1])
        means, stds = [], []
        rel_pos = rel_l2[pos]
        for i in range(n_bins):
            mask = (log_mags >= bins[i]) & (log_mags < bins[i + 1])
            if mask.any():
                means.append(rel_pos[mask].mean())
                stds.append(rel_pos[mask].std())
            else:
                means.append(np.nan); stds.append(np.nan)
        axes[1].errorbar(10 ** centers, means, yerr=stds, fmt='o-', capsize=3)
        axes[1].set_xscale('log')
    axes[1].set_xlabel(r'$\|f^{\mathrm{target}}\|_{\mathrm{RMS}}$  (bin center)')
    axes[1].set_ylabel(r'mean rel. $L^2$')
    axes[1].set_title(r'Binned: rel. $L^2$ by target-magnitude bin')
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    sorted_idx = np.argsort(tgt_rms)
    n = len(rel_l2)
    if n < 4:
        return None
    return {
        'bottom_quartile_target_mean_rel_l2': float(rel_l2[sorted_idx[:n // 4]].mean()),
        'top_quartile_target_mean_rel_l2':    float(rel_l2[sorted_idx[3 * n // 4:]].mean()),
        'overall_mean_rel_l2':                float(rel_l2.mean()),
    }


# =========================================================================
# (E) residual panels: best / 25th / median / worst
# =========================================================================

def plot_residual_panels(model, val_loader, device, out_path):
    import torch

    preds, tgts, rels = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device); y = y.to(device)
            p = model(x)
            num = float(torch.norm(p - y).item())
            den = float(torch.norm(y).item())
            rels.append(num / max(den, 1e-30))
            preds.append(p.cpu().numpy()[0, 0])
            tgts.append(y.cpu().numpy()[0, 0])
    rels = np.array(rels)
    if len(rels) < 4:
        return rels
    order = np.argsort(rels)
    pick = [order[0], order[len(order) // 4], order[len(order) // 2], order[-1]]
    labels = ['best', '25th pct', 'median', 'worst']

    fig, axes = plt.subplots(len(pick), 3, figsize=(13, 3.5 * len(pick)))
    for row, (idx, lbl) in enumerate(zip(pick, labels)):
        pred = preds[idx]
        tgt = tgts[idx]
        diff = pred - tgt
        m = float(max(np.abs(tgt).max(), np.abs(pred).max()))
        if m == 0:
            m = 1.0
        for col, (a, t) in enumerate([
            (tgt,  f'{lbl}: target'),
            (pred, f'{lbl}: pred  (rel L2 = {rels[idx]:.3f})'),
            (diff, f'{lbl}: pred - target'),
        ]):
            axes[row, col].imshow(a, cmap='RdBu_r', vmin=-m, vmax=m, origin='lower')
            axes[row, col].set_title(t, fontsize=10)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return rels


# =========================================================================
# (F) average spectrum of pred vs target
# =========================================================================

def plot_spectra_comparison(model, val_loader, device, out_path, n_samples: int = 50):
    import torch

    pred_specs, tgt_specs, diff_specs = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= n_samples:
                break
            x = x.to(device); y = y.to(device)
            p = model(x)
            for arr, lst in [
                (p.cpu().numpy()[0, 0], pred_specs),
                (y.cpu().numpy()[0, 0], tgt_specs),
                ((p - y).cpu().numpy()[0, 0], diff_specs),
            ]:
                fhat = np.fft.rfft2(arr)
                psd = np.abs(fhat) ** 2
                Ny, Nx = arr.shape
                kx = np.fft.rfftfreq(Nx) * Nx
                ky = np.fft.fftfreq(Ny) * Ny
                kxg, kyg = np.meshgrid(kx, ky, indexing='xy')
                k = np.sqrt(kxg ** 2 + kyg ** 2)
                k_max = int(np.floor(k.max()))
                radial = np.zeros(k_max + 1)
                for k_idx in range(k_max + 1):
                    mask = (k >= k_idx - 0.5) & (k < k_idx + 0.5)
                    if mask.any():
                        radial[k_idx] = psd[mask].sum()
                lst.append(radial)

    if not pred_specs:
        return
    pred_avg = np.mean(np.stack(pred_specs), axis=0)
    tgt_avg = np.mean(np.stack(tgt_specs), axis=0)
    diff_avg = np.mean(np.stack(diff_specs), axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    k = np.arange(len(tgt_avg))
    ax.loglog(k[1:], tgt_avg[1:],  'k-',  lw=2, label='target')
    ax.loglog(k[1:], pred_avg[1:], 'C1-', lw=2, label='prediction')
    ax.loglog(k[1:], diff_avg[1:], 'r--', lw=1, label='residual (pred - target)')
    ax.set_xlabel(r'wavenumber $k$')
    ax.set_ylabel(r'power spectrum  $|\hat f(k)|^2$')
    ax.set_title(f'Average spectrum, {len(pred_specs)} val samples')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# =========================================================================
# main
# =========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True,
                   help='training run directory (has log.csv, config.json, model_best.pt)')
    p.add_argument('--root-dir', type=Path, required=True,
                   help='dataset root (manifest.json + samples/)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-samples', type=int, default=2000,
                   help='cap on samples for the magnitude histogram (D)')
    p.add_argument('--n-spectra-samples', type=int, default=50,
                   help='how many val samples to average the spectrum over (F)')
    p.add_argument('--skip-model', action='store_true',
                   help='skip model-based diagnostics (B/C/E/F); only do A/D')
    args = p.parse_args()

    out_dir = args.run_dir / 'diagnosis'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []

    # (A) Loss curves
    print("[A] reading log.csv")
    cols = read_log(args.run_dir / 'log.csv')
    plot_loss_curves(cols, out_dir / 'loss_curves.png')
    summary_lines.append(loss_summary_stats(cols))
    summary_lines.append("")

    # (D) Target magnitude histogram
    print(f"[D] target magnitude distribution (up to {args.max_samples} samples)")
    mags, _files = compute_target_magnitudes(args.root_dir, max_samples=args.max_samples)
    target_stats = plot_target_histogram(mags, out_dir / 'target_magnitude_histogram.png')
    if target_stats is not None:
        summary_lines.append("=" * 60)
        summary_lines.append("(D) TARGET MAGNITUDE DISTRIBUTION")
        summary_lines.append("=" * 60)
        for k, v in target_stats.items():
            summary_lines.append(f"  {k:30s} : {v}")
        summary_lines.append("")

    if not args.skip_model:
        try:
            print("[loading model + val loader]")
            model, val_loader, config = _load_model_and_loaders(
                args.run_dir, args.root_dir, args.device)

            print("[C] per-sample val rel-L2 vs target magnitude")
            rel, rms_t, _ = per_sample_val_errors(model, val_loader, args.device)
            mag_stats = plot_rel_l2_by_magnitude(rel, rms_t, out_dir / 'val_rel_l2_by_magnitude.png')
            if mag_stats is not None:
                summary_lines.append("=" * 60)
                summary_lines.append("(C) VAL REL-L2 BINNED BY TARGET MAGNITUDE")
                summary_lines.append("=" * 60)
                for k, v in mag_stats.items():
                    summary_lines.append(f"  {k:42s} : {v:.4f}")
                ratio = mag_stats['bottom_quartile_target_mean_rel_l2'] / max(
                    mag_stats['top_quartile_target_mean_rel_l2'], 1e-30)
                summary_lines.append(f"  bottom/top quartile rel-L2 ratio          : {ratio:.2f}")
                if ratio > 1.5:
                    summary_lines.append("  >> small-target samples have HIGHER rel-L2.")
                    summary_lines.append("     -> consider RELATIVE loss or per-sample normalization.")
                summary_lines.append("")

            print("[E] residual panels (best / 25th / median / worst)")
            plot_residual_panels(model, val_loader, args.device, out_dir / 'residuals_panel.png')

            print("[F] spectral comparison")
            plot_spectra_comparison(model, val_loader, args.device,
                                    out_dir / 'spectra_comparison.png',
                                    n_samples=args.n_spectra_samples)
        except Exception as e:
            summary_lines.append(f"WARNING: model-based diagnostics failed: {e}")
            print(f"  (skipped: {e})")

    summary = "\n".join(summary_lines)
    print()
    print(summary)
    (out_dir / 'summary.txt').write_text(summary)
    print(f"\nAll outputs in: {out_dir}")


if __name__ == '__main__':
    main()
