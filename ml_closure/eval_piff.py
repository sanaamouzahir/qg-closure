"""
A priori evaluation package (ML SPEC 01 S4) — run on the best checkpoint.

1. Pointwise val metrics (R2, RMSE, NLL) global and binned by zeta decile
   (degenerates to one bin when zeta is constant, e.g. FPC-const).
2. Calibration: reliability diagram (empirical coverage of +/-1,2,3 sigma vs
   nominal) and spread–skill plot (binned predictive sigma vs empirical |err|),
   global and per zeta-bin.
3. Field visualizations: 6 validation snapshots spanning the Re range —
   truth Pi*, predictive mean, predictive sigma, |error| (4-panel, seismic,
   aspect-preserving) + the Re_inlet(t) trace with snapshot times marked.
4. zeta ARD lengthscale in the summary (regime dependence not absorbed by FiLM).

Full frames are evaluated directly (CNN is convolutional + periodic; the GP is
pointwise) — crops are a training device only.

Usage:
    python eval_piff.py --ckpt runs_piff/<name>/best.pt [--config conf_piff.yaml] [--outdir ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import load_conf, build_runs, split_frames, _f
from model_piff import PiffModel
from train_piff import gaussian_nll

HERE = Path(__file__).resolve().parent
NOMINAL = {1.0: 0.682689, 2.0: 0.954500, 3.0: 0.997300}


@torch.no_grad()
def predict_frame(model, run, frame, device, gp_chunk):
    """Full-frame predictive mean/sigma on masked pixels; returns 2D fields
    (NaN outside mask) + flat arrays."""
    x, y, mask, zeta = run.full_frame(frame)
    gpin = model.masked_gp_inputs(x[None].to(device), zeta[None].to(device),
                                  mask[None].to(device))
    mus, vars_ = [], []
    for i0 in range(0, gpin.shape[0], gp_chunk):
        pred = model.likelihood(model.gp(gpin[i0:i0 + gp_chunk]))
        mus.append(pred.mean.cpu().numpy())
        vars_.append(pred.variance.cpu().numpy())
    mu, var = np.concatenate(mus), np.concatenate(vars_)
    m = mask.numpy()
    truth = y.numpy()
    mu2 = np.full_like(truth, np.nan); sg2 = np.full_like(truth, np.nan)
    mu2[m] = mu; sg2[m] = np.sqrt(var)
    return {'truth': truth, 'mask': m, 'mu2d': mu2, 'sigma2d': sg2,
            'y': truth[m], 'mu': mu, 'sigma': np.sqrt(var),
            'zeta': float(zeta), 't': float(run.times[frame]),
            'Re': float(run.Re_snap[frame])}


def metrics_block(y, mu, sigma):
    var = sigma ** 2
    return {
        'n': int(y.size),
        'r2': float(1.0 - np.sum((y - mu) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)),
        'rmse': float(np.sqrt(np.mean((y - mu) ** 2))),
        'nll': float(np.mean(gaussian_nll(y, mu, var))),
        'mean_sigma': float(np.mean(sigma)),
        'coverage': {f'{k:.0f}sigma': float(np.mean(np.abs(y - mu) <= k * sigma))
                     for k in NOMINAL},
    }


def spread_skill(y, mu, sigma, n_bins=12):
    """Bin by predicted sigma; empirical skill = RMSE of errors in bin."""
    q = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    q[-1] += 1e-12
    ctr, emp = [], []
    for i in range(n_bins):
        m = (sigma >= q[i]) & (sigma < q[i + 1])
        if m.sum() < 10:
            continue
        ctr.append(float(np.mean(sigma[m])))
        emp.append(float(np.sqrt(np.mean((y[m] - mu[m]) ** 2))))
    return np.array(ctr), np.array(emp)


def imshow_field(ax, f2d, run, title, vmax=None):
    vmax = vmax if vmax is not None else np.nanmax(np.abs(f2d))
    im = ax.imshow(f2d, cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower',
                   extent=[0, run.Lx, 0, run.Ly], aspect='equal')  # never stretch
    ax.set_title(title, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser(description="A priori Pi_FF eval (spec S4)")
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=str(HERE / 'conf_piff.yaml'))
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(args.config)
    ec = conf['eval']
    device = args.device or conf['train']['device']
    gp_chunk = int(conf['train']['gp_chunk'])
    outdir = Path(args.outdir or (Path(args.ckpt).parent / 'eval'))
    outdir.mkdir(parents=True, exist_ok=True)

    model = PiffModel(ckpt['conf']).to(device)   # conf as trained (film flag etc.)
    model.load_state_dict(ckpt['model'])
    model.eval()

    runs = build_runs(conf)
    frames = split_frames(runs, 'val', conf)
    print(f"[eval] {len(frames)} val frames from {[r.name for r in runs]}")

    preds = [predict_frame(model, runs[ri], fi, device, gp_chunk) for ri, fi in frames]
    y = np.concatenate([p['y'] for p in preds])
    mu = np.concatenate([p['mu'] for p in preds])
    sg = np.concatenate([p['sigma'] for p in preds])
    zt = np.concatenate([np.full(p['y'].size, p['zeta']) for p in preds])

    # ---- 1. metrics: global + zeta deciles -------------------------------- #
    summary = {'ckpt': str(Path(args.ckpt).resolve()), 'epoch': int(ckpt['epoch']),
               'seed': int(ckpt['seed']),
               'zeta_ard_lengthscale': model.zeta_ard_lengthscale(),
               'global': metrics_block(y, mu, sg), 'zeta_bins': []}
    edges = np.unique(np.quantile(zt, np.linspace(0, 1, int(ec['n_zeta_bins']) + 1)))
    if len(edges) < 2:
        edges = np.array([zt[0] - 0.5, zt[0] + 0.5])   # constant zeta (FPC-const)
    for i in range(len(edges) - 1):
        m = (zt >= edges[i]) & (zt <= edges[i + 1] if i == len(edges) - 2 else zt < edges[i + 1])
        if m.sum() < 10:
            continue
        blk = metrics_block(y[m], mu[m], sg[m])
        blk['zeta_range'] = [float(edges[i]), float(edges[i + 1])]
        summary['zeta_bins'].append(blk)

    # ---- 2. calibration ---------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ks = np.array(sorted(NOMINAL))
    nom = np.array([NOMINAL[k] for k in ks])
    emp = np.array([np.mean(np.abs(y - mu) <= k * sg) for k in ks])
    ax1.plot(nom, emp, 'o-', label='global')
    for i in range(len(edges) - 1):
        m = (zt >= edges[i]) & (zt < edges[i + 1] + (1e-12 if i == len(edges) - 2 else 0))
        if m.sum() < 100:
            continue
        e = [np.mean(np.abs(y[m] - mu[m]) <= k * sg[m]) for k in ks]
        ax1.plot(nom, e, '.--', alpha=0.5, label=f'zeta[{edges[i]:.2f},{edges[i+1]:.2f}]')
    ax1.plot([0, 1], [0, 1], 'k:', lw=1)
    ax1.set_xlabel('nominal coverage'); ax1.set_ylabel('empirical coverage')
    ax1.set_title('reliability (+/-1,2,3 sigma)'); ax1.legend(fontsize=6); ax1.grid(alpha=0.3)
    ctr, es = spread_skill(y, mu, sg)
    ax2.plot(ctr, es, 'o-')
    lim = [0, max(ctr.max(), es.max()) * 1.05]
    ax2.plot(lim, lim, 'k:', lw=1)
    ax2.set_xlabel('binned predictive sigma'); ax2.set_ylabel('empirical RMSE in bin')
    ax2.set_title('spread-skill'); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / 'calibration.png', dpi=130); plt.close(fig)

    # ---- 3. field figures: 6 snapshots spanning the Re range -------------- #
    n_show = min(int(ec['n_field_snapshots']), len(preds))
    order = np.argsort([p['Re'] for p in preds])
    sel = order[np.linspace(0, len(order) - 1, n_show).astype(int)]
    show = [preds[i] for i in sel]
    for j, (idx, p) in enumerate(zip(sel, show)):
        run = runs[frames[idx][0]]
        err = np.where(p['mask'], np.abs(p['truth'] - np.nan_to_num(p['mu2d'])), np.nan)
        tr = np.where(p['mask'], p['truth'], np.nan)
        fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
        vmax = np.nanmax(np.abs(tr))
        for ax, f2d, ttl, vm in zip(
                axs, [tr, p['mu2d'], p['sigma2d'], err],
                [f"truth Pi*  t={p['t']:.2f} Re={p['Re']:.0f}", 'predictive mean',
                 'predictive sigma', '|error|'],
                [vmax, vmax, None, None]):
            im = imshow_field(ax, f2d, run, ttl, vmax=vm)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(outdir / f'field_{j}_t{p["t"]:.2f}.png', dpi=130)
        plt.close(fig)

    # Re_inlet(t) trace with snapshot markers
    fig, ax = plt.subplots(figsize=(9, 3))
    for r in runs:
        tab = np.load(r.run_dir / r.man['files']['u_table'])
        ax.plot(tab['t'], tab['Re'], lw=0.8, label=r.name)
    for p in show:
        ax.axvline(p['t'], color='k', ls='--', lw=0.7)
    ax.set_xlabel('t'); ax.set_ylabel('Re_inlet'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / 'Re_trace.png', dpi=130); plt.close(fig)

    with open(outdir / 'summary.yaml', 'w') as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(json.dumps(summary['global'], indent=2))
    print(f"[eval] zeta ARD lengthscale = {summary['zeta_ard_lengthscale']:.4f}")
    print(f"[eval] package in {outdir}")


if __name__ == '__main__':
    main()
