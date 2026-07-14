"""
replot_eval_fields.py — P1 of the 2026-07-13 field-plot triage (standalone,
reusable). The original eval field panels use a LINEAR color scale to the frame
max of a heavy-tailed field, so the wake reads as "no flow". This regenerates
them as 5-panel figures with a SYMLOG color norm:

    [ omega_bar | truth Pi | predicted Pi | predicted sigma | |error| ]

symlog linthresh = 99th percentile of |truth Pi| over the frame's valid pixels;
one shared norm for truth/prediction/error, sigma on the same linthresh.
cmap seismic, aspect-preserving, origin/orientation identical to eval_piff.
Writes field5_*.png NEXT TO the existing field_*.png (never deletes them).
Frame selection reproduces eval_piff exactly (val frames sorted by Re, 6-point
linspace), so field5_j corresponds to field_j.

Reads the FROZEN modules (dataset_piff/model_piff) — no production file edited.

Usage (GPU, via piff_tool_job.sh):
  python replot_eval_fields.py --ckpt runs_piff/prod_ext150/best.pt --config conf_piff.yaml
  python replot_eval_fields.py --ckpt runs_piff/cape_base_100ep/best.pt --config conf_piff_cape.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from dataset_piff import load_conf, build_runs, split_frames
from model_piff import PiffModel

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def predict_frame_full(model, run, frame, device, gp_chunk):
    """Full-frame prediction, returning the omega* channel too (superset of
    eval_piff.predict_frame — same math, conditioning flags honored)."""
    x, y, mask, zeta, zeta_dot, g = run.full_frame(frame)
    gpin = model.masked_gp_inputs(
        x[None].to(device), zeta[None].to(device), mask[None].to(device),
        zeta_dot=(zeta_dot[None].to(device) if model.use_zeta_dot else None),
        g=(g[None].to(device) if model.use_grad_feature else None))
    gm = (g[None].to(device)[mask[None].to(device)]
          if getattr(model, 'noise_prior', 'none') == 'structural' else None)
    mus, vars_ = [], []
    for i0 in range(0, gpin.shape[0], gp_chunk):
        mu_p, var_p = model.predict_physical(
            gpin[i0:i0 + gp_chunk],
            g_masked=(gm[i0:i0 + gp_chunk] if gm is not None else None))
        mus.append(mu_p.cpu().numpy())
        vars_.append(var_p.cpu().numpy())
    mu, var = np.concatenate(mus), np.concatenate(vars_)
    m = mask.numpy()
    truth = y.numpy()
    mu2 = np.full_like(truth, np.nan); sg2 = np.full_like(truth, np.nan)
    mu2[m] = mu; sg2[m] = np.sqrt(var)
    return {'omega': x.numpy()[0], 'truth': truth, 'mask': m,
            'mu2d': mu2, 'sigma2d': sg2,
            't': float(run.times[frame]), 'Re': float(run.Re_snap[frame])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=str(HERE / 'conf_piff.yaml'))
    ap.add_argument('--outdir', default=None, help='default: <ckpt dir>/eval')
    ap.add_argument('--device', default=None)
    ap.add_argument('--n-snapshots', type=int, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(args.config)
    device = args.device or conf['train']['device']
    gp_chunk = int(conf['train']['gp_chunk'])
    n_show = int(args.n_snapshots or conf['eval']['n_field_snapshots'])
    outdir = Path(args.outdir or (Path(args.ckpt).parent / 'eval'))
    outdir.mkdir(parents=True, exist_ok=True)

    model = PiffModel(ckpt['conf']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # same conditioning plumbing as eval_piff (flags travel with the ckpt)
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    ck_tsm = ckpt['conf'].get('zeta', {}).get('tshed_smooth', 2.992)
    conf['zeta']['tshed_smooth'] = ck_tsm

    runs = build_runs(conf)
    frames = split_frames(runs, 'val', conf)
    print(f'[replot] {len(frames)} val frames from {[r.name for r in runs]}')

    # identical selection to eval_piff: sort ALL val frames by Re, 6-pt linspace
    Re_all = np.array([runs[ri].Re_snap[fi] for ri, fi in frames])
    order = np.argsort(Re_all)
    n_show = min(n_show, len(frames))
    sel = order[np.linspace(0, len(order) - 1, n_show).astype(int)]

    for j, idx in enumerate(sel):
        ri, fi = frames[idx]
        run = runs[ri]
        p = predict_frame_full(model, run, fi, device, gp_chunk)
        m = p['mask']
        tr = np.where(m, p['truth'], np.nan)
        err = np.where(m, np.abs(p['truth'] - np.nan_to_num(p['mu2d'])), np.nan)

        absval = np.abs(p['truth'][m])
        lt = max(float(np.percentile(absval, 99.0)), 1e-12)
        vmax = max(float(absval.max()), lt * 1.01)
        norm = SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10)
        norm_s = SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10)
        ovmax = float(np.percentile(np.abs(p['omega']), 99.5))

        fig, axs = plt.subplots(1, 5, figsize=(23, 4.2))
        specs = [
            (p['omega'], 'filtered vorticity omega_bar* (linear)',
             dict(vmin=-ovmax, vmax=ovmax)),
            (tr, f"truth Pi*  t={p['t']:.2f} Re={p['Re']:.0f} (symlog)", dict(norm=norm)),
            (p['mu2d'], 'predicted Pi* (same symlog)', dict(norm=norm)),
            (p['sigma2d'], 'predicted sigma (symlog)', dict(norm=norm_s)),
            (err, '|error| (same symlog)', dict(norm=norm)),
        ]
        for ax, (f2d, ttl, kw) in zip(axs, specs):
            im = ax.imshow(f2d, cmap='seismic', origin='lower',
                           extent=[0, run.Lx, 0, run.Ly], aspect='equal', **kw)
            ax.set_title(ttl, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{run.name}  (symlog linthresh = 99th pct |Pi*| = {lt:.3g})",
                     fontsize=10)
        fig.tight_layout()
        fp = outdir / f"field5_{j}_t{p['t']:.2f}.png"
        fig.savefig(fp, dpi=130)
        plt.close(fig)
        print(f'[replot] {fp}  (member {run.name}, Re {p["Re"]:.0f})')

    print(f'[replot] done: {n_show} figures in {outdir}')


if __name__ == '__main__':
    main()
