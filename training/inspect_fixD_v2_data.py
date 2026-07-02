"""
inspect_fixD_v2_data.py -- pre-training diagnostic for the Fix D v2 dataset.

Usage:
    python inspect_fixD_v2_data.py \
        --root /gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-simple-package-stable/src/qg/training/data/decaying_turbulence_dT_1em3_fixD_v2 \
        --out  diagnostics_fixD_v2/ \
        --n-samples 8

What it produces (in --out directory):
    1. summary.txt                 -- per-channel statistics + dataset metadata
    2. sample_NNNNNN_inputs.png    -- 6 input channels (omega^{0,-1,-2}, psi^{0,-1,-2})
    3. sample_NNNNNN_target.png    -- target (f_NN_target) + diagnostic overlays
    4. sample_NNNNNN_psi_check.png -- visual: spectral_lap(psi^{-k}) ?= omega^{-k}
    5. dataset_overview.png        -- 3x3 mosaic of f_NN_target across first 9 samples
    6. manifest_dump.json          -- copy of the manifest

Interpretation guide:
    * psi_check should show |Lap(psi) - omega|_inf / |omega|_inf ~ 1e-12 (float64
      roundoff). If not, your inverse Laplacian has a bug (wrong domain L,
      aliasing, etc.)
    * f_NN_target vs f_NN_target_from_e should agree in shape; magnitudes
      should differ by O(DT) (~0.1% at DT=1e-3). If they're 2x off everywhere,
      you have a sign or normalization bug.
    * Stats: at nu=1e-5 with E0=0.01, omega has |.|max ~ 5-20, psi has
      |.|max ~ 0.1, target |.|max should be O(0.1-10). If anything is 10^-9
      or 10^9, something's wrong.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_samples(root, n_max=None):
    samples_dir = root / 'samples'
    if not samples_dir.exists():
        sys.exit(f"ERROR: {samples_dir} does not exist")
    files = sorted(samples_dir.glob('sample_*.npz'))
    if not files:
        sys.exit(f"ERROR: no sample_*.npz files in {samples_dir}")
    if n_max is not None:
        files = files[:n_max]
    return files


def load_manifest(root):
    mf = root / 'manifest.json'
    if mf.exists():
        return json.loads(mf.read_text())
    return None


def spectral_lap(field, Lx=2*np.pi, Ly=2*np.pi):
    """Compute Laplacian via FFT. Field shape: (Ny, Nx)."""
    Ny, Nx = field.shape
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    kxg, kyg = np.meshgrid(kx, ky, indexing='xy')
    ksq = kxg**2 + kyg**2
    fhat = np.fft.fft2(field)
    return np.real(np.fft.ifft2(-ksq * fhat))


def stats(arr):
    a = np.asarray(arr).astype(np.float64)
    return (f"min={a.min():.3e} max={a.max():.3e} "
            f"mean={a.mean():.3e} std={a.std():.3e} "
            f"|.|max={np.abs(a).max():.3e}")


def maybe_squeeze(a):
    """Some fields may be (1, Ny, Nx); squeeze to (Ny, Nx)."""
    if a.ndim == 3 and a.shape[0] == 1:
        return a[0]
    return a


def plot_grid(arrs, titles, fig_path, ncols=3, suptitle=None):
    n = len(arrs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows),
                              squeeze=False)
    for idx, (arr, title) in enumerate(zip(arrs, titles)):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        vmax = max(abs(arr.min()), abs(arr.max())) if arr.size > 0 else 1.0
        if vmax == 0:
            vmax = 1.0
        im = ax.imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        origin='lower')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for idx in range(n, nrows*ncols):
        r, c = idx // ncols, idx % ncols
        axes[r][c].axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def diagnose_psi_inv_lap(rec, sample_name, out_dir):
    diag = []
    arrs, titles = [], []
    for tag in ['0', 'm1', 'm2']:
        omega_key = f'omega_{tag}'
        psi_key = f'psi_{tag}'
        if omega_key not in rec.files or psi_key not in rec.files:
            diag.append(f"  {psi_key}: KEY MISSING")
            continue
        omega = maybe_squeeze(rec[omega_key])
        psi = maybe_squeeze(rec[psi_key])
        lap_psi = spectral_lap(psi)
        residual = lap_psi - omega
        rel_err = np.abs(residual).max() / (np.abs(omega).max() + 1e-30)
        diag.append(f"  {psi_key}:  |Lap(psi) - omega|_inf / |omega|_inf = {rel_err:.3e}")
        arrs.extend([omega, lap_psi, residual])
        titles.extend([
            f"omega_{tag}\n{stats(omega)}",
            f"Lap(psi_{tag})\n{stats(lap_psi)}",
            f"residual\nrel_err={rel_err:.2e}",
        ])
    if arrs:
        plot_grid(arrs, titles,
                  out_dir / f'{sample_name}_psi_check.png',
                  ncols=3,
                  suptitle=f"{sample_name}: psi inverse-Laplacian sanity")
    return diag


def diagnose_target_consistency(rec, sample_name, out_dir):
    if 'f_NN_target' not in rec.files or 'f_NN_target_from_e' not in rec.files:
        return ["  target consistency: KEYS MISSING"]
    a = maybe_squeeze(rec['f_NN_target'])
    b = maybe_squeeze(rec['f_NN_target_from_e'])
    diff = b - a
    rel = np.linalg.norm(diff) / (np.linalg.norm(a) + 1e-30)
    diag = [
        f"  f_NN_target (analytical):     {stats(a)}",
        f"  f_NN_target_from_e (empirical):{stats(b)}",
        f"  ||emp - anal|| / ||anal|| = {rel:.3e}  (expect ~ DT = 1e-3)",
    ]
    plot_grid(
        [a, b, diff],
        [f"f_NN_target (analytical)\n{stats(a)}",
         f"f_NN_target_from_e (empirical)\n{stats(b)}",
         f"diff (emp - anal)\nrel L2 = {rel:.2e}"],
        out_dir / f'{sample_name}_target.png',
        ncols=3,
        suptitle=f"{sample_name}: closure target consistency"
    )
    return diag


def plot_inputs(rec, sample_name, out_dir):
    keys = ['omega_0', 'omega_m1', 'omega_m2',
            'psi_0',  'psi_m1',  'psi_m2']
    arrs, titles = [], []
    for k in keys:
        if k not in rec.files:
            continue
        a = maybe_squeeze(rec[k])
        arrs.append(a)
        titles.append(f"{k}\n{stats(a)}")
    if arrs:
        plot_grid(arrs, titles,
                  out_dir / f'{sample_name}_inputs.png',
                  ncols=3,
                  suptitle=f"{sample_name}: 6 input channels")


def aggregate_stats(samples):
    keys = ['omega_0', 'omega_m1', 'omega_m2',
            'psi_0', 'psi_m1', 'psi_m2',
            'f_NN_target', 'f_NN_target_from_e',
            'f_anal', 'e_total', 'e_NN_incr',
            'N_0', 'N_dot_0_anal', 'N_ddot_0_anal']
    accum = {k: {'min': np.inf, 'max': -np.inf, 'sum': 0.0,
                 'sumsq': 0.0, 'count': 0, 'absmax': 0.0}
             for k in keys}
    for f in samples:
        d = np.load(f)
        for k in keys:
            if k not in d.files:
                continue
            a = d[k].astype(np.float64).ravel()
            accum[k]['min'] = min(accum[k]['min'], a.min())
            accum[k]['max'] = max(accum[k]['max'], a.max())
            accum[k]['absmax'] = max(accum[k]['absmax'], np.abs(a).max())
            accum[k]['sum'] += a.sum()
            accum[k]['sumsq'] += (a**2).sum()
            accum[k]['count'] += a.size

    lines = ['',
             '=== Aggregate per-channel statistics ===',
             f"(across {len(samples)} samples)",
             '',
             f"{'channel':<22} {'min':>12} {'max':>12} {'mean':>12} {'std':>12} {'|.|max':>12}"]
    for k in keys:
        a = accum[k]
        if a['count'] == 0:
            lines.append(f"{k:<22}  (missing)")
            continue
        mean = a['sum'] / a['count']
        var = a['sumsq'] / a['count'] - mean**2
        std = np.sqrt(max(var, 0))
        lines.append(
            f"{k:<22} {a['min']:>12.3e} {a['max']:>12.3e} "
            f"{mean:>12.3e} {std:>12.3e} {a['absmax']:>12.3e}"
        )
    return '\n'.join(lines)


def overview_mosaic(samples, out_path, n=9):
    arrs, titles = [], []
    for i, f in enumerate(samples[:n]):
        d = np.load(f)
        if 'f_NN_target' not in d.files:
            continue
        a = maybe_squeeze(d['f_NN_target'])
        arrs.append(a)
        titles.append(f"sample {i}: |.|max={np.abs(a).max():.2e}")
    if arrs:
        plot_grid(arrs, titles, out_path, ncols=3,
                  suptitle="f_NN_target across first 9 samples")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('diagnostics_fixD_v2'))
    p.add_argument('--n-samples', type=int, default=8,
                   help='how many samples to visualise in detail')
    p.add_argument('--n-stats', type=int, default=200,
                   help='how many samples to use for aggregate stats')
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting dataset:  {args.root}")
    print(f"Output:              {args.out}")

    manifest = load_manifest(args.root)
    if manifest:
        with open(args.out / 'manifest_dump.json', 'w') as f:
            json.dump(manifest, f, indent=2)

    samples_to_show = find_samples(args.root, args.n_samples)
    summary_lines = ['=== Fix D v2 dataset diagnostics ===',
                     f'Dataset root: {args.root}',
                     f'Samples shown: {len(samples_to_show)}',
                     '']
    for f in samples_to_show:
        rec = np.load(f)
        name = f.stem
        print(f"  {name} (fields: {len(rec.files)})")
        summary_lines.append(f'--- {name} ---')
        summary_lines.append(f'  fields: {sorted(rec.files)}')
        plot_inputs(rec, name, args.out)
        d1 = diagnose_psi_inv_lap(rec, name, args.out)
        d2 = diagnose_target_consistency(rec, name, args.out)
        summary_lines.extend(d1)
        summary_lines.extend(d2)
        summary_lines.append('')

    samples_for_stats = find_samples(args.root, args.n_stats)
    summary_lines.append(aggregate_stats(samples_for_stats))

    (args.out / 'summary.txt').write_text('\n'.join(summary_lines))
    print(f"\nWrote {args.out / 'summary.txt'}")

    overview_mosaic(samples_to_show, args.out / 'dataset_overview.png', n=9)
    print(f"Wrote {args.out / 'dataset_overview.png'}")

    print("\nDone. Inspect the PNGs and summary.txt in", args.out)


if __name__ == '__main__':
    main()
