"""
triage_mask_audit.py — P3 of the 2026-07-13 field-plot triage (standalone).

The code audit (see DECISIONS.md 2026-07-13) shows sponge/body/strip pixels are
excluded from training loss, eval metrics, and kurtosis via RunData.valid.
This script renders the deliverable FIGURE: the valid-mask decomposition per
geometry, so the exclusion can be SEEN — which pixels are dropped and why:

    valid | analytic sponge strips | filtered chi_sponge support | obstacle body

plus the exact pixel fractions, printed and saved.

CPU-only (all.q): builds RunData from the frozen dataset_piff, no model.

Usage:
  python triage_mask_audit.py --config conf_piff.yaml \
      --members <...>/FPC-const <...>/FPCape-const \
      --outdir triage_plot_20260713/mask_audit
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from dataset_piff import load_conf, RunData, _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    base = load_conf(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for md in args.members:
        conf = yaml.safe_load(yaml.safe_dump(base))   # deep copy
        conf['data']['runs'] = [md]
        conf.setdefault('model', {})['use_grad_feature'] = False
        run = RunData(md, conf)
        dc = conf['data']

        # reproduce the mask components exactly as dataset_piff.RunData does
        man = run.man
        sxf = [_f(v) for v in man['sponge']['x_strip_frac']]
        syf = [_f(v) for v in man['sponge']['y_strip_frac']]
        xf = (np.arange(run.Nx) + 0.5) / run.Nx
        yf = (np.arange(run.Ny) + 0.5) / run.Ny
        strip_analytic = (xf[None, :] >= sxf[0]) | (yf[:, None] >= syf[0])
        strip_chi = run.chi_sponge > _f(dc['sponge_thresh'])
        body = run.sdf < 0.0
        valid = run.valid
        # cross-check against RunData (must be exact)
        recon = (~(strip_analytic | strip_chi)) & (~body)
        assert (recon == valid).all(), f'{run.name}: mask reconstruction mismatch'

        cat = np.zeros(valid.shape, dtype=np.int8)          # 0 = valid
        cat[strip_analytic] = 1
        cat[strip_chi & ~strip_analytic] = 2
        cat[body] = 3
        n = valid.size
        fr = {k: float(v) / n for k, v in {
            'valid': valid.sum(),
            'analytic strips': strip_analytic.sum(),
            'chi_sponge only': (strip_chi & ~strip_analytic).sum(),
            'body': body.sum()}.items()}
        print(f'[{run.name}] pixels {run.Ny}x{run.Nx}: '
              + ', '.join(f'{k} {100*v:.1f}%' for k, v in fr.items()))

        cmap = ListedColormap(['#FFFFFF', '#4477AA', '#66CCEE', '#222222'])
        fig, ax = plt.subplots(figsize=(9, 9 * run.Ly / run.Lx + 1))
        ax.imshow(cat, cmap=cmap, vmin=-0.5, vmax=3.5, origin='lower',
                  extent=[0, run.Lx, 0, run.Ly], aspect='equal',
                  interpolation='nearest')
        # faint vorticity contour of a developed frame for orientation
        fi = run.frames_in(60.0, 100.0)
        if len(fi):
            om = run.omega[fi[len(fi) // 2]]
            lv = np.percentile(np.abs(om), 98)
            x = (np.arange(run.Nx) + 0.5) * run.dx
            y = (np.arange(run.Ny) + 0.5) * run.dy
            ax.contour(x, y, om, levels=[-lv, lv], colors='gray',
                       linewidths=0.4, alpha=0.6)
        handles = [Patch(facecolor='#FFFFFF', edgecolor='k',
                         label=f"valid (used in loss + metrics): {100*fr['valid']:.1f}%"),
                   Patch(facecolor='#4477AA',
                         label=f"analytic sponge strips: {100*fr['analytic strips']:.1f}%"),
                   Patch(facecolor='#66CCEE',
                         label=f"filtered chi_sponge > {dc['sponge_thresh']}: "
                               f"{100*fr['chi_sponge only']:.1f}% (extra)"),
                   Patch(facecolor='#222222',
                         label=f"obstacle body (SDF < 0): {100*fr['body']:.1f}%")]
        ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.9)
        ax.set_title(f'{run.name}: pixels EXCLUDED from training loss, eval metrics '
                     f'and kurtosis (everything colored)\n'
                     f'gray contours: vorticity of a developed frame, for orientation',
                     fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        fig.tight_layout()
        fp = outdir / f'valid_mask_{run.name}.png'
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f'[{run.name}] figure -> {fp}')

        np.savez(outdir / f'valid_mask_{run.name}.npz', category=cat,
                 fractions=np.array([fr['valid'], fr['analytic strips'],
                                     fr['chi_sponge only'], fr['body']]),
                 n_valid=run.n_valid)


if __name__ == '__main__':
    main()
