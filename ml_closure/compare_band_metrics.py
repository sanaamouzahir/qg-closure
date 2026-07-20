#!/usr/bin/env python
"""compare_band_metrics.py -- the decisive wallv2 question (Sanaa 2026-07-20):
is wallv2 a NEAR-vs-FAR TRADE or a true regression?

plot_fields_assess scores the RING-EXCLUDED population (sdf > 1D) = the FAR
field only, and there wallv2 lost to ylp75/lap on every matched frame. But
wallv2 moves capacity INTO the near-wall band that metric cannot see. This
script scores the SAME ckpt/frames in THREE populations:

    NEAR  sdf <= 1.0 D      (the band wallv2 targets)
    FAR   sdf >  1.0 D      (what the assess panels showed)
    ALL   every valid pixel (what the gate diagnostics used)

Same truth (config data variant), same frames (per-member linspace over the
member's val window), metrics-only (no figures). Prints one row per
(model, member, frame, band): r2, rmse/rms_truth, and the worst-0.1% share.

Usage: python compare_band_metrics.py --ckpt <best.pt> --config <conf.yaml>
                                      [--per-member N] [--device cpu]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from dataset_piff import load_conf, build_runs, split_frames
from model_piff import PiffModel
from piff_model_loader import load_piff_model  # two-band blend (Sanaa GO 2026-07-20): plain ckpt -> identical PiffModel path
from eval_piff import full_frame_slice
from replot_eval_fields import predict_frame_full

HERE = Path(__file__).resolve().parent


def band_stats(truth, err):
    if truth.size == 0:
        return None
    rms_t = float(np.sqrt(np.mean(truth ** 2)))
    rms_e = float(np.sqrt(np.mean(err ** 2)))
    r2 = 1.0 - float(np.var(err) / max(np.var(truth), 1e-30))
    se = np.sort(err ** 2)[::-1]
    k = max(1, int(0.001 * se.size))
    tail = float(se[:k].sum() / max(se.sum(), 1e-30))
    return dict(n=int(truth.size), rms_t=rms_t,
                rel=rms_e / max(rms_t, 1e-30), r2=r2, tail=tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--per-member', type=int, default=2)
    ap.add_argument('--out-csv', default=None,
                    help='ALSO write the rows to this CSV (input to the '
                         'two-band gate table, Sanaa GO 2026-07-20). Absent '
                         '(default) => stdout only, exactly as before.')
    ap.add_argument('--model-tag', default=None,
                    help='label for the model column of --out-csv; '
                         'default = the ckpt directory name')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(HERE / args.config if not Path(args.config).is_absolute()
                     else args.config)
    gp_chunk = int(conf['train']['gp_chunk'])
    model = load_piff_model(ckpt, args.device, conf=conf)
    model.load_state_dict(ckpt['model'])
    model.eval()
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    ck_var = ckpt['conf'].get('data', {}).get('variant')
    if ck_var and not conf['data'].get('variant'):
        conf['data']['variant'] = ck_var
    conf['zeta']['tshed_smooth'] = ckpt['conf'].get('zeta', {}).get(
        'tshed_smooth', 2.992)
    tag = args.model_tag or Path(args.ckpt).parent.name
    print(f"[band] model={tag} variant={conf['data'].get('variant')}",
          flush=True)
    rows = []
    # BLENDED ONLY (G4 MINOR, 2026-07-20): record what share of the scored
    # pixels each expert actually owns. A blended run landing near a bar
    # cannot be interpreted without it -- if the overlap collar holds a large
    # share of the pixels, the result is a property of the HAND-OVER, not of
    # either specialist. No-op for plain PiffModels.
    bf_rows = []
    blended = hasattr(model, 'band_fractions')

    runs = build_runs(conf)
    frames = split_frames(runs, 'val', conf)
    sel = []
    for ri, run in enumerate(runs):
        fis = sorted(fi for rj, fi in frames if rj == ri)
        if not fis:
            continue
        n = min(args.per_member, len(fis))
        for k in np.linspace(0, len(fis) - 1, n).astype(int):
            sel.append((ri, fis[k]))

    for ri, fi in sel:
        run = runs[ri]
        p = predict_frame_full(model, run, fi, args.device, gp_chunk)
        m = p['mask']
        e2d = np.nan_to_num(p['mu2d']) - p['truth']
        if blended:
            xf, _, mf = run.full_frame(fi)[:3]
            bf = model.band_fractions(xf[None].to(args.device),
                                      mf[None].to(args.device))
            bf_rows.append(dict(member=run.name, t=float(p['t']), **bf))
            print(f"[band] {tag:26s} {run.name:12s} t={p['t']:7.2f} "
                  f"EXPERT SHARE pure_near={bf['pure_near']:.4f} "
                  f"overlap={bf['overlap']:.4f} pure_far={bf['pure_far']:.4f}",
                  flush=True)
        sdf2d = run.sdf[full_frame_slice(run)]
        near = m & (sdf2d <= 1.0 * run.D)
        far = m & (sdf2d > 1.0 * run.D)
        for bname, bmask in (('NEAR', near), ('FAR', far), ('ALL', m)):
            s = band_stats(p['truth'][bmask], e2d[bmask])
            if s is None:
                continue
            print(f"[band] {tag:26s} {run.name:12s} t={p['t']:7.2f} "
                  f"{bname:4s} n={s['n']:9d} rms_truth={s['rms_t']:.5e} "
                  f"r2={s['r2']:7.4f} rel_err={s['rel']:.4f} "
                  f"tail0.1%={s['tail']:.3f}", flush=True)
            rows.append(dict(model=tag, member=run.name, t=float(p['t']),
                             band=bname, **s))
    if args.out_csv:
        import csv as _csv
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, 'w', newline='') as fh:
            w = _csv.DictWriter(fh, fieldnames=['model', 'member', 't', 'band',
                                                'n', 'rms_t', 'rel', 'r2', 'tail'])
            w.writeheader()
            w.writerows(rows)
        print(f"[band] wrote {outp} ({len(rows)} rows)", flush=True)
        if bf_rows:
            # gate artifact: the expert-share census travels with the metrics
            import yaml as _yaml
            nbf = len(bf_rows)
            summary = {k: sum(r[k] for r in bf_rows) / nbf
                       for k in ('pure_near', 'overlap', 'pure_far')}
            bfp = outp.parent / 'band_fractions.yaml'
            bfp.write_text(_yaml.safe_dump(
                {'model': tag, 'n_frames': nbf,
                 'note': 'share of SCORED (masked) pixels owned by each expert; '
                         'overlap = the smoothstep hand-over collar, where the '
                         'prediction is a blend of both and neither specialist '
                         'alone explains the result',
                 'mean_over_frames': summary, 'per_frame': bf_rows},
                sort_keys=False))
            print(f"[band] EXPERT SHARE (mean over {nbf} frames): "
                  f"pure_near={summary['pure_near']:.4f} "
                  f"overlap={summary['overlap']:.4f} "
                  f"pure_far={summary['pure_far']:.4f} -> {bfp}", flush=True)
    print(f"[band] done {tag}", flush=True)


if __name__ == '__main__':
    main()
