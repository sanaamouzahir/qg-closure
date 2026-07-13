"""
Post-hoc scalar sigma recalibration (Sanaa GO 2026-07-12 chat).

The S4 eval showed the winner's predictive sigma ~1.9x overdispersed
(1-sigma coverage 0.975 vs 0.683 nominal). Standard variance-scaling fix:
sigma' = s * sigma with ONE global scalar. The NLL-optimal s has the closed
form s^2 = mean(z^2), z = (y - mu)/sigma  (d/ds mean[log(s sigma) +
(y-mu)^2/(2 s^2 sigma^2)] = 0). The GP mean is untouched.

Honesty split: s is FIT on the first half of the val window and REPORTED on
the second half (fit [t_val_lo, t_mid), test [t_mid, t_val_hi]) — one scalar
cannot meaningfully overfit, but the headline number stays out-of-sample.

Outputs (next to the ckpt): recalibration.yaml (s, before/after NLL +
coverage on both halves), fig_recalibration.png (reliability before/after).
The scalar is a SIDECAR — no ckpt mutation; downstream consumers read
recalibration.yaml['sigma_scale'].

Usage:
    python calibrate_piff.py --ckpt runs_piff/<name>/best.pt [--config conf_piff.yaml]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import load_conf, build_runs, split_frames, _f
from eval_piff import predict_frame, metrics_block

HERE = Path(__file__).resolve().parent
NOMINAL = {1.0: 0.682689, 2.0: 0.954500, 3.0: 0.997300}


def collect(model, runs, frames, device, gp_chunk):
    ys, mus, sgs, ts = [], [], [], []
    for ri, fi in frames:
        p = predict_frame(model, runs[ri], fi, device, gp_chunk)
        ys.append(p['y']); mus.append(p['mu']); sgs.append(p['sigma'])
        ts.append(np.full(p['y'].shape, p['t']))
    return (np.concatenate(ys), np.concatenate(mus),
            np.concatenate(sgs), np.concatenate(ts))


def reliability(y, mu, sigma, ks=np.linspace(0.25, 3.0, 12)):
    z = np.abs(y - mu) / sigma
    from scipy.special import erf
    nominal = erf(ks / np.sqrt(2.0))
    empirical = np.array([np.mean(z <= k) for k in ks])
    return ks, nominal, empirical


def main():
    ap = argparse.ArgumentParser(description="Post-hoc sigma recalibration")
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=str(HERE / 'conf_piff.yaml'))
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    conf = load_conf(args.config)
    device = args.device or conf['train']['device']
    gp_chunk = int(conf['train']['gp_chunk'])

    from model_piff import PiffModel
    ckpt_path = Path(args.ckpt).resolve()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = PiffModel(ckpt['conf']).to(device)   # conf as trained (film flag etc.)
    model.load_state_dict(ckpt['model'])
    model.eval()

    runs = build_runs(conf)
    frames = split_frames(runs, 'val', conf)
    t_lo, t_hi = _f(conf['data']['t_val_lo']), _f(conf['data']['t_val_hi'])
    t_mid = 0.5 * (t_lo + t_hi)
    fit_frames = [(ri, fi) for ri, fi in frames if runs[ri].times[fi] < t_mid]
    test_frames = [(ri, fi) for ri, fi in frames if runs[ri].times[fi] >= t_mid]
    if not fit_frames or not test_frames:
        raise SystemExit(f"empty half-split: {len(fit_frames)} fit / {len(test_frames)} test")
    print(f"[cal] val frames: {len(fit_frames)} fit (t<{t_mid}), {len(test_frames)} test")

    y_f, mu_f, sg_f, _ = collect(model, runs, fit_frames, device, gp_chunk)
    y_t, mu_t, sg_t, _ = collect(model, runs, test_frames, device, gp_chunk)

    z2 = ((y_f - mu_f) / sg_f) ** 2
    s = float(np.sqrt(np.mean(z2)))
    print(f"[cal] sigma_scale s = {s:.6f}  (mean z^2 = {np.mean(z2):.6f} on fit half)")

    out = {'ckpt': str(ckpt_path), 'sigma_scale': s,
           'fit_window': [float(t_lo), float(t_mid)],
           'test_window': [float(t_mid), float(t_hi)],
           'criterion': 'NLL-optimal variance scaling: s^2 = mean(z^2) on fit half'}
    for name, (y, mu, sg) in (('fit', (y_f, mu_f, sg_f)), ('test', (y_t, mu_t, sg_t))):
        out[name] = {'before': metrics_block(y, mu, sg),
                     'after': metrics_block(y, mu, s * sg)}
        b, a = out[name]['before'], out[name]['after']
        print(f"[cal] {name}: NLL {b['nll']:.4f} -> {a['nll']:.4f} | "
              f"1sig cov {b['coverage']['1sigma']:.4f} -> {a['coverage']['1sigma']:.4f}")

    outdir = ckpt_path.parent
    with open(outdir / 'recalibration.yaml', 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, sc, ttl in ((axes[0], 1.0, 'before'), (axes[1], s, f'after (s={s:.3f})')):
        ks, nom, emp = reliability(y_t, mu_t, sc * sg_t)
        ax.plot(nom, emp, 'o-', color='#b2182b', label='empirical (test half)')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
        ax.set_xlabel('nominal coverage'); ax.set_ylabel('empirical coverage')
        ax.set_title(f'reliability {ttl}'); ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(outdir / 'fig_recalibration.png', dpi=140)
    print(f"[cal] wrote {outdir}/recalibration.yaml + fig_recalibration.png")


if __name__ == '__main__':
    main()
