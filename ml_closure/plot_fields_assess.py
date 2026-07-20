#!/usr/bin/env python
"""plot_fields_assess.py -- "is the prediction good or garbage?" panel
(Sanaa 2026-07-20: the symlog field6 figures make error MAGNITUDE
unreadable - a 5% and a 50% error look alike).

Design rule: TRUTH, PREDICTION and ERROR share ONE LINEAR colour scale
(+-p99 of |truth| over ring-excluded valid pixels). On that scale a small
error is literally blank - no interpretation needed. Panels:

  1 truth Pi*            linear, shared scale
  2 predicted Pi*        linear, SAME scale
  3 error (pred-truth)   linear, SAME scale   <- the assessment panel
  4 error x5             same scale, amplified (is there structure left?)
  5 |error| / RMS(truth) 0..1 viridis-free seismic 0-centred: fraction of
                         the field's own RMS, so "0.2" = 20% of typical
                         signal; saturates white at >= 1
  6 pixel scatter        pred vs truth (2% sample), identity line, with
                         r2 / RMSE-over-RMS / p99-error printed

Title carries the numbers that answer the question outright:
  RMSE/RMS_truth (= sqrt(1-r2)), r2, % pixels with |err| > 0.25 RMS_truth,
  and the share of squared error held by the worst 0.1% of pixels.

Reuses replot_eval_fields' loaders verbatim (frozen modules, no production
file touched). Usage mirrors it:
  python plot_fields_assess.py --ckpt <best.pt> --config <conf.yaml>
                               [--per-member N] [--device cpu]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import load_conf, build_runs, split_frames
from model_piff import PiffModel
from piff_model_loader import load_piff_model  # two-band blend (Sanaa GO 2026-07-20): plain ckpt -> identical PiffModel path
from eval_piff import full_frame_slice
from replot_eval_fields import predict_frame_full

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--outdir', default=None,
                    help='default: <ckpt dir>/eval_assess')
    ap.add_argument('--device', default=None)
    ap.add_argument('--per-member', type=int, default=2)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(HERE / args.config if not Path(args.config).is_absolute()
                     else args.config)
    device = args.device or conf['train']['device']
    gp_chunk = int(conf['train']['gp_chunk'])
    outdir = Path(args.outdir or (Path(args.ckpt).parent / 'eval_assess'))
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_piff_model(ckpt, device, conf=conf)
    model.load_state_dict(ckpt['model'])
    model.eval()
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    ck_var = ckpt['conf'].get('data', {}).get('variant')
    if ck_var and not conf['data'].get('variant'):
        conf['data']['variant'] = ck_var
    conf['zeta']['tshed_smooth'] = ckpt['conf'].get('zeta', {}).get(
        'tshed_smooth', 2.992)

    variant = conf['data'].get('variant')
    print(f"[assess] DATA VARIANT (truth+pred filter) = {variant!r}  "
          f"ckpt variant = {ckpt['conf'].get('data', {}).get('variant')!r}  "
          f"scale = LINEAR shared +-p99|truth|", flush=True)
    runs = build_runs(conf)
    frames = split_frames(runs, 'val', conf)
    sel, tags = [], []
    for ri, run in enumerate(runs):
        fis = sorted(fi for rj, fi in frames if rj == ri)
        if not fis:
            continue
        n = min(args.per_member, len(fis))
        for k in np.linspace(0, len(fis) - 1, n).astype(int):
            sel.append((ri, fis[k]))
            tags.append(run.name)

    counts = {}
    for (ri, fi), tag in zip(sel, tags):
        run = runs[ri]
        p = predict_frame_full(model, run, fi, device, gp_chunk)
        m = p['mask']
        pred = np.nan_to_num(p['mu2d'])
        errs = np.where(m, pred - p['truth'], np.nan)      # signed
        tr = np.where(m, p['truth'], np.nan)
        pr = np.where(m, pred, np.nan)

        ring2d = run.sdf[full_frame_slice(run)] > 1.0 * run.D
        mr = m & ring2d
        base = mr if mr.any() else m
        tvals = p['truth'][base]
        evals = (pred - p['truth'])[base]
        rms_t = float(np.sqrt(np.mean(tvals ** 2)))
        rms_e = float(np.sqrt(np.mean(evals ** 2)))
        r2 = 1.0 - float(np.var(evals) / max(np.var(tvals), 1e-30))
        vmax = float(np.percentile(np.abs(tvals), 99.0))
        frac_bad = float((np.abs(evals) > 0.25 * rms_t).mean())
        se = np.sort(evals ** 2)[::-1]
        k = max(1, int(0.001 * se.size))
        tail_share = float(se[:k].sum() / max(se.sum(), 1e-30))

        relmag = np.where(m, np.abs(pred - p['truth']) / max(rms_t, 1e-30),
                          np.nan)

        fig, axs = plt.subplots(1, 6, figsize=(28, 4.4))
        kw = dict(cmap='seismic', origin='lower', aspect='equal',
                  extent=[0, run.Lx, 0, run.Ly], vmin=-vmax, vmax=vmax)
        for ax, f2d, ttl in (
                (axs[0], tr, 'TRUTH  Pi*  (linear, +-p99)'),
                (axs[1], pr, 'PREDICTION  (same scale)'),
                (axs[2], errs, 'ERROR = pred - truth  (SAME scale)'),
                (axs[3], errs * 5.0, 'ERROR x5  (same scale)')):
            im = ax.imshow(f2d, **kw)
            ax.set_title(ttl, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        im = axs[4].imshow(relmag, cmap='seismic', origin='lower',
                           aspect='equal', extent=[0, run.Lx, 0, run.Ly],
                           vmin=0.0, vmax=1.0)
        axs[4].set_title('|error| / RMS(truth)   (1.0 = a full RMS)',
                         fontsize=9)
        fig.colorbar(im, ax=axs[4], fraction=0.046)

        rng = np.random.default_rng(0)
        idx = rng.choice(tvals.size, size=min(20000, tvals.size),
                         replace=False)
        axs[5].plot(tvals[idx], (tvals + evals)[idx], '.', ms=1.0, alpha=0.25)
        lim = float(np.percentile(np.abs(tvals), 99.9))
        axs[5].plot([-lim, lim], [-lim, lim], 'r-', lw=1.0)
        axs[5].set_xlim(-lim, lim); axs[5].set_ylim(-lim, lim)
        axs[5].set_xlabel('truth'); axs[5].set_ylabel('prediction')
        axs[5].set_title('pixel scatter (2% sample) + identity', fontsize=9)
        axs[5].set_aspect('equal')

        fig.suptitle(
            f"{run.name}  t={p['t']:.2f}  Re={p['Re']:.0f}   |   "
            f"r2={r2:.3f}   RMSE/RMS_truth={rms_e / max(rms_t, 1e-30):.1%}   "
            f"pixels with |err|>0.25 RMS: {frac_bad:.1%}   "
            f"worst-0.1% pixels hold {tail_share:.0%} of squared error   |   "
            f"variant={variant}  LINEAR shared scale +-{vmax:.3g}  "
            f"RMS(truth)={rms_t:.3g}",
            fontsize=11)
        fig.tight_layout()
        fdir = outdir / tag
        fdir.mkdir(parents=True, exist_ok=True)
        j = counts.get(tag, 0)
        counts[tag] = j + 1
        fp = fdir / f"assess_{j}_t{p['t']:.2f}.png"
        fig.savefig(fp, dpi=130)
        plt.close(fig)
        print(f"[assess] {fp}  r2={r2:.3f} rmse/rms={rms_e / rms_t:.3f} "
              f"tail={tail_share:.2f}  [AUDIT member={run.name} t={p['t']:.2f} "
              f"Re={p['Re']:.0f} rms_truth={rms_t:.6e} vmax={vmax:.6e} "
              f"npix={tvals.size} variant={variant}]", flush=True)

    print(f"[assess] done: {len(sel)} figures in {outdir}", flush=True)


if __name__ == '__main__':
    main()
