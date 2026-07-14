#!/usr/bin/env python
"""diag_ring_side_by_side.py -- Sanaa's 4-panel ring check (2026-07-14 evening):
for each geometry, per frame: [#1 truth WITH ring | #2 truth WITHOUT ring
(near-body band sdf <= 1D blanked) | #3 with - without (= the ring band alone)
| #4 GP predictive mean]. All four panels share ONE color scale computed from
the ring-EXCLUDED truth (so the wake is visible; ring pixels saturate — that
saturation IS the message: it shows what the 83x artefact does to any shared
scale). cmap seismic, aspect preserved.

CPU job (plot rule):
  cd ml_closure && qsub -q all.q -N ringSbS -o ../logs/ringSbS.$JOB_ID.log \
      -j y -cwd -V ../scripts/sge/piff_tool_job.sh diag_ring_side_by_side.py \
      --geoms fpc cape --n-frames 2 --device cpu
Output: pngs/ring_side_by_side/<geom>/... + README.txt (full paths printed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import load_conf, build_runs, split_frames
from model_piff import PiffModel
from eval_piff import full_frame_slice, imshow_field
from replot_eval_fields import predict_frame_full

HERE = Path(__file__).resolve().parent

CKPTS = {'fpc': ('runs_piff/piff_fpc_gjs/best.pt', 'conf_piff_fpc_gjs.yaml'),
         'cape': ('runs_piff/piff_cape_gjs/best.pt', 'conf_piff_cape_gjs.yaml')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geoms', nargs='+', default=['fpc', 'cape'])
    ap.add_argument('--n-frames', type=int, default=2)
    ap.add_argument('--sdf-mult', type=float, default=1.0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    for geom in args.geoms:
        ck_rel, cf_rel = CKPTS[geom]
        ckpt = torch.load(HERE / ck_rel, map_location='cpu', weights_only=False)
        conf = load_conf(HERE / cf_rel)
        conf.setdefault('model', {})['use_grad_feature'] = \
            ckpt['conf'].get('model', {}).get('use_grad_feature', False)
        ck_var = ckpt['conf'].get('data', {}).get('variant')
        if ck_var and not conf['data'].get('variant'):
            conf['data']['variant'] = ck_var
        conf['zeta']['tshed_smooth'] = ckpt['conf'].get(
            'zeta', {}).get('tshed_smooth', 2.992)
        model = PiffModel(ckpt['conf']).to(args.device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        gp_chunk = int(conf['train']['gp_chunk'])

        runs = build_runs(conf)
        frames = split_frames(runs, 'val', conf)
        sel = frames[:: max(1, len(frames) // args.n_frames)][: args.n_frames]

        outdir = HERE / 'pngs' / 'ring_side_by_side' / geom
        outdir.mkdir(parents=True, exist_ok=True)
        made = []
        for ri, fi in sel:
            run = runs[ri]
            p = predict_frame_full(model, run, fi, args.device, gp_chunk)
            m = p['mask']
            ring_ok = run.sdf[full_frame_slice(run)] > args.sdf_mult * run.D
            tr_with = np.where(m, p['truth'], np.nan)                # #1
            tr_wo = np.where(m & ring_ok, p['truth'], np.nan)        # #2
            ring_only = np.where(m & ~ring_ok, p['truth'], np.nan)   # #3
            pred = np.where(m, p['mu2d'], np.nan)                    # #4
            vmax = np.nanmax(np.abs(tr_wo))                          # shared scale

            fig, axs = plt.subplots(1, 4, figsize=(19, 4.4))
            for ax, f2d, ttl in zip(
                    axs, [tr_with, tr_wo, ring_only, pred],
                    ['#1 truth WITH ring', '#2 truth WITHOUT ring (sdf>1D)',
                     '#3 with - without (the ring band)',
                     '#4 GP predictive mean']):
                im = imshow_field(ax, f2d, run, ttl, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(f"{geom} {run.name}  t={p['t']:.2f}  Re={p['Re']:.0f} "
                         f"— one shared scale from ring-excluded truth "
                         f"(ring saturates by design)")
            fig.tight_layout()
            fp = outdir / f"ring_side_by_side_{run.name}_t{p['t']:.2f}.png"
            fig.savefig(fp, dpi=140)
            plt.close(fig)
            made.append(fp)
            print(f"[ringSbS] wrote {fp.resolve()}")

        (outdir / 'README.txt').write_text(
            "Sanaa's 4-panel ring check (2026-07-14): per frame, panels are\n"
            "#1 truth WITH ring | #2 truth WITHOUT ring (near-body band\n"
            "sdf <= 1D blanked) | #3 with - without = the ring band alone |\n"
            "#4 GP predictive mean. One shared color scale from the\n"
            "ring-EXCLUDED truth; ring pixels saturate by design (that is the\n"
            "point: the 83x artefact swamps any shared scale). cmap seismic.\n"
            + "\n".join(str(f.resolve()) for f in made) + "\n")
        print(f"[ringSbS] README {(outdir / 'README.txt').resolve()}")


if __name__ == '__main__':
    main()
