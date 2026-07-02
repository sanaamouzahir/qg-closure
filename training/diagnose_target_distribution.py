"""
diagnose_target_distribution.py - Inspect how the NN target magnitude varies
across an existing training dataset, to diagnose whether per-channel global
normalization is the right strategy.

What it computes:
  For every sample in the dataset (train + val + test):
    norm_f_NN_target = sqrt(mean(f_NN_target^2))      (per-sample L2)
    norm_e_NN_incr   = sqrt(mean(e_NN_incr^2))
    norm_e_anal_incr = sqrt(mean(e_anal_incr^2))
    norm_e_total     = sqrt(mean(e_total^2))
    norm_omega_0     = sqrt(mean(omega_0^2))
  Plus per-sample meta: seed_t, batch_idx, split_membership.

Outputs:
  - target_magnitude_distribution.png:
      a 2x3 grid of plots showing how target magnitude varies across:
        (top row)
          (a) histogram of log10(norm_f_NN_target) split by train/val/test
          (b) norm_f_NN_target vs seed_t, colored by batch_idx
          (c) norm_f_NN_target vs batch_idx, box plot
        (bottom row)
          (d) breakdown: norm_e_total, norm_e_anal_incr, norm_e_NN_incr vs seed_t
          (e) ratio norm_e_NN_incr / norm_e_total vs seed_t
              (how big the NN target is relative to total residual)
          (f) ratio norm_omega_0 vs seed_t (sanity check on field magnitude)

  - target_magnitude_diagnostics.csv:
      one row per sample with all the per-sample stats for further analysis.

Usage:
    python diagnose_target_distribution.py \
        --root-dir /path/to/decaying_turbulence_dT_1em3 \
        --out-dir  ./figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _norm_l2(a: np.ndarray) -> float:
    """sqrt(mean(a^2))."""
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-dir', type=Path, required=True,
                   help='dataset root (contains manifest.json + samples/ + split.npz)')
    p.add_argument('--out-dir', type=Path, default=None,
                   help='where to write figures + csv (default: <root-dir>/diagnostics)')
    p.add_argument('--max-samples', type=int, default=-1,
                   help='cap on number of samples to read (default: all)')
    args = p.parse_args()

    root = args.root_dir
    out_dir = args.out_dir if args.out_dir is not None else root / 'diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load manifest + split ---- #
    with open(root / 'manifest.json') as f:
        manifest = json.load(f)
    Ny, Nx = int(manifest['Ny']), int(manifest['Nx'])
    print(f"[diag] dataset root: {root}")
    print(f"[diag] grid: {Ny} x {Nx}")
    print(f"[diag] split mode: {manifest.get('split_mode', '?')}")
    print(f"[diag] n_total: {manifest.get('n_total', '?')}, "
          f"n_completed: {manifest.get('n_completed', '?')}")
    print(f"[diag] batches_used: {manifest.get('batches_used', '?')}, "
          f"seeds_per_batch: {manifest.get('seeds_per_batch', '?')}")

    with np.load(root / 'split.npz') as sp:
        train_idx = sp['train_idx'].astype(np.int64)
        val_idx   = sp['val_idx'].astype(np.int64)
        test_idx  = sp['test_idx'].astype(np.int64)
    split_of = {}
    for i in train_idx: split_of[int(i)] = 'train'
    for i in val_idx:   split_of[int(i)] = 'val'
    for i in test_idx:  split_of[int(i)] = 'test'

    print(f"[diag] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    samples_dir = root / 'samples'
    sample_files = sorted(samples_dir.glob('sample_*.npz'))
    if args.max_samples > 0:
        sample_files = sample_files[:args.max_samples]
    print(f"[diag] reading {len(sample_files)} samples...")

    # ---- collect per-sample stats ---- #
    rows = []  # list of dicts
    bad = 0
    for k, fp in enumerate(sample_files):
        idx = int(fp.stem.split('_')[-1])
        try:
            with np.load(fp) as zf:
                f_NN     = zf['f_NN_target']
                e_NN     = zf['e_NN_incr']
                e_anal   = zf['e_anal_incr']
                e_total  = zf['e_total']
                omega_0  = zf['omega_0']
                seed_t   = float(zf['seed_t'])
                seed_idx = int(zf['seed_idx'])
                batch_id = int(zf['batch_idx'])
        except Exception as e:
            bad += 1
            if bad < 5:
                print(f"  [warn] failed to read {fp.name}: {e}")
            continue

        rows.append(dict(
            index=idx,
            file=fp.name,
            split=split_of.get(idx, '?'),
            seed_t=seed_t,
            seed_idx=seed_idx,
            batch_idx=batch_id,
            norm_f_NN=_norm_l2(f_NN),
            norm_e_NN=_norm_l2(e_NN),
            norm_e_anal=_norm_l2(e_anal),
            norm_e_total=_norm_l2(e_total),
            norm_omega_0=_norm_l2(omega_0),
        ))

        if (k + 1) % 200 == 0 or k + 1 == len(sample_files):
            print(f"  read {k+1}/{len(sample_files)} ...")

    if bad:
        print(f"[diag] {bad} samples failed to read")
    if not rows:
        print("[diag] ERROR: no samples readable")
        return

    # ---- write csv ---- #
    csv_path = out_dir / 'target_magnitude_diagnostics.csv'
    cols = ['index', 'file', 'split', 'batch_idx', 'seed_idx', 'seed_t',
            'norm_f_NN', 'norm_e_NN', 'norm_e_anal', 'norm_e_total',
            'norm_omega_0']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f"[diag] wrote {csv_path} ({len(rows)} rows)")

    # ---- summary statistics ---- #
    arr_norm_f_NN     = np.array([r['norm_f_NN']     for r in rows])
    arr_norm_e_NN     = np.array([r['norm_e_NN']     for r in rows])
    arr_norm_e_anal   = np.array([r['norm_e_anal']   for r in rows])
    arr_norm_e_total  = np.array([r['norm_e_total']  for r in rows])
    arr_norm_omega_0  = np.array([r['norm_omega_0']  for r in rows])
    arr_seed_t        = np.array([r['seed_t']        for r in rows])
    arr_batch_idx     = np.array([r['batch_idx']     for r in rows])
    arr_split         = np.array([r['split']         for r in rows])

    print("\n[diag] per-sample magnitude summary:")
    for name, a in [('norm_f_NN_target', arr_norm_f_NN),
                    ('norm_e_NN_incr',   arr_norm_e_NN),
                    ('norm_e_anal_incr', arr_norm_e_anal),
                    ('norm_e_total',     arr_norm_e_total),
                    ('norm_omega_0',     arr_norm_omega_0)]:
        a_pos = a[a > 0]
        if len(a_pos):
            print(f"  {name:20s}: "
                  f"min={a_pos.min():.3e}  max={a.max():.3e}  "
                  f"median={np.median(a):.3e}  "
                  f"max/min={a.max()/a_pos.min():.1f}x  "
                  f"std/mean={a.std()/a.mean():.2f}")

    # ---- per-split summary ---- #
    print("\n[diag] norm_f_NN_target by split:")
    for s in ['train', 'val', 'test']:
        m = arr_split == s
        if m.sum() == 0:
            continue
        a = arr_norm_f_NN[m]
        print(f"  {s:5s}: n={m.sum():4d}  "
              f"mean={a.mean():.3e}  median={np.median(a):.3e}  "
              f"min={a.min():.3e}  max={a.max():.3e}")

    # ---- plot ---- #
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) histogram of log10(norm_f_NN) split by train/val/test
    ax = axes[0, 0]
    log_norm = np.log10(np.maximum(arr_norm_f_NN, 1e-30))
    bins = np.linspace(log_norm.min(), log_norm.max(), 50)
    for s, c in [('train', 'C0'), ('val', 'C1'), ('test', 'C2')]:
        m = arr_split == s
        if m.sum() == 0:
            continue
        ax.hist(log_norm[m], bins=bins, alpha=0.5, label=f'{s} (n={m.sum()})',
                color=c, density=True)
    ax.set_xlabel(r'$\log_{10}\|f_{NN,{\rm target}}\|_2$ (per sample)')
    ax.set_ylabel('density')
    ax.set_title('(a) Distribution of target magnitude, by split')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) norm_f_NN vs seed_t, colored by batch_idx
    ax = axes[0, 1]
    unique_batches = sorted(set(arr_batch_idx.tolist()))
    cmap = plt.colormaps['viridis']
    for i, b in enumerate(unique_batches):
        m = arr_batch_idx == b
        c = cmap(i / max(len(unique_batches) - 1, 1))
        ax.semilogy(arr_seed_t[m], arr_norm_f_NN[m], 'o', ms=2, alpha=0.4,
                    color=c, label=f'b={b}' if i < 5 else None)
    ax.set_xlabel(r'seed time $t$')
    ax.set_ylabel(r'$\|f_{NN,{\rm target}}\|_2$ (log)')
    ax.set_title(f'(b) Target magnitude vs t, by batch ({len(unique_batches)} batches)')
    if len(unique_batches) <= 5:
        ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # (c) box plot of norm_f_NN vs batch_idx
    ax = axes[0, 2]
    bp_data = [arr_norm_f_NN[arr_batch_idx == b] for b in unique_batches]
    ax.boxplot(bp_data, labels=[str(b) for b in unique_batches],
               whis=[5, 95], showfliers=False)
    ax.set_yscale('log')
    ax.set_xlabel('batch index')
    ax.set_ylabel(r'$\|f_{NN,{\rm target}}\|_2$ (log, 5--95 pct.)')
    ax.set_title('(c) Per-batch distribution of target magnitude')
    ax.grid(True, alpha=0.3, axis='y')

    # (d) breakdown of total / analytical / NN target norms vs seed_t
    ax = axes[1, 0]
    order = np.argsort(arr_seed_t)
    ax.semilogy(arr_seed_t[order], arr_norm_e_total[order], 'o', ms=2, alpha=0.5,
                label=r'$\|e_{\rm total}\|_2$', color='C3')
    ax.semilogy(arr_seed_t[order], arr_norm_e_anal[order], 'o', ms=2, alpha=0.5,
                label=r'$\|e_{\rm anal\;incr}\|_2$', color='C1')
    ax.semilogy(arr_seed_t[order], arr_norm_e_NN[order], 'o', ms=2, alpha=0.5,
                label=r'$\|e_{NN\;incr}\|_2$', color='C0')
    ax.set_xlabel(r'seed time $t$')
    ax.set_ylabel(r'norm (log)')
    ax.set_title('(d) Total / analytical / NN error increments vs t')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # (e) ratio NN / total vs seed_t
    ax = axes[1, 1]
    ratio = arr_norm_e_NN / np.maximum(arr_norm_e_total, 1e-30)
    ax.plot(arr_seed_t, ratio, 'o', ms=2, alpha=0.5, color='C0')
    ax.axhline(1.0, color='k', ls=':', alpha=0.5)
    ax.set_xlabel(r'seed time $t$')
    ax.set_ylabel(r'$\|e_{NN}\| / \|e_{\rm total}\|$')
    ax.set_title('(e) NN target as fraction of total error vs t')
    ax.grid(True, alpha=0.3)
    # auto-clip y to a sane range
    valid = np.isfinite(ratio) & (ratio < 100)
    if valid.any():
        ax.set_ylim(0, min(2.0, 1.2 * np.percentile(ratio[valid], 99)))

    # (f) field magnitude norm_omega_0 vs seed_t
    ax = axes[1, 2]
    ax.semilogy(arr_seed_t, arr_norm_omega_0, 'o', ms=2, alpha=0.5, color='C4')
    ax.set_xlabel(r'seed time $t$')
    ax.set_ylabel(r'$\|\bar\omega_0\|_2$ (log)')
    ax.set_title('(f) Field magnitude vs t (sanity check)')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Target distribution diagnostics: {root.name}',
                 fontsize=13)
    fig.tight_layout()

    out_png = out_dir / 'target_magnitude_distribution.png'
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"[diag] wrote {out_png}")

    # ---- key takeaways printed in text ---- #
    print("\n[diag] KEY DIAGNOSTICS")
    log_decade_range = log_norm.max() - log_norm.min()
    print(f"  norm_f_NN_target spans {log_decade_range:.1f} decades across samples")
    if log_decade_range > 2.0:
        print("    -> FLAG: target dynamic range > 2 decades. Per-channel global")
        print("       normalization will compress most samples to near zero;")
        print("       MSE loss will be dominated by the few large-magnitude samples.")
        print("       Consider per-sample normalization, log-magnitude target,")
        print("       or sample weighting by magnitude.")

    train_mean = arr_norm_f_NN[arr_split == 'train'].mean() if (arr_split == 'train').any() else 0
    val_mean   = arr_norm_f_NN[arr_split == 'val'].mean()   if (arr_split == 'val').any() else 0
    if train_mean > 0 and val_mean > 0:
        ratio_tv = max(train_mean, val_mean) / min(train_mean, val_mean)
        print(f"  train/val mean target magnitude ratio: {ratio_tv:.2f}x")
        if ratio_tv > 1.5:
            print("    -> FLAG: train and val have very different magnitude.")
            print("       Likely a by_batch split; rebuild with --split-mode by_time.")


if __name__ == '__main__':
    main()
