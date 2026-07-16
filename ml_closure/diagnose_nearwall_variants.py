#!/usr/bin/env python
"""diagnose_nearwall_variants.py -- is the 12-35x near-wall truth excess of the
sharp/gaussian variants CHECKERBOARD ARTIFACT or SMOOTH PHYSICS?
(Sanaa GO 2026-07-15 night; follow-up to the reflection probe's test C.)

Pure data diagnostic (no model, CPU): for the top-N extreme-event pixels of
one member, per filter variant (sharp / gaussian / ylp75):
  1. zoomed truth-Pi window panels (N rows x 3 variants, seismic, shared row
     scale, body contour) -- the eyeball answer;
  2. adjacent-row correlation of Pi in the near-wall band (|sdf| < 1D):
     row_corr ~ -1 = y-Nyquist checkerboard (the ylp75 discovery signature);
  3. fraction of y-spectral power above 0.75*kN in the same band.
Verdict: ARTIFACT-STRIPED if sharp/gaussian row_corr < -0.5 while ylp75
row_corr > -0.2; else SMOOTH-PHYSICS (or MIXED).

Usage: python diagnose_nearwall_variants.py --config conf_piff_fpc_gjs_ylp75.yaml \
    --member FPC-const --events runs_piff/piff_fpc_gjs_ylp75/error_tails_diag/FPC-const/extreme_events.csv \
    [--n-events 3] [--half 24] [--report-run nearwall_variants_fpc]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import load_conf, build_runs

HERE = Path(__file__).resolve().parent
BRANCH_ROOT = HERE.parent
VARIANTS = [('sharp', ''), ('gaussian', 'gaussian'),
            ('jonly', 'gaussian_jonly'),      # isolates the commutator removal
            ('ylp75', 'gaussian_jonly_ylp75')]  # ...from the y-notch


def load_member(config, member, variant):
    conf = load_conf(HERE / config)
    conf['data']['variant'] = variant
    conf.setdefault('model', {})['use_grad_feature'] = False
    runs = build_runs(conf)
    r = next(x for x in runs if x.name == member)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--member', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--n-events', type=int, default=3)
    ap.add_argument('--half', type=int, default=24)
    ap.add_argument('--report-run', default=None)
    args = ap.parse_args()

    with open(args.events) as f:
        evs = list(csv.DictReader(f))[:args.n_events]
    fig_dir = HERE / 'pngs' / 'nearwall_variants' / args.member
    fig_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    fig, axs = plt.subplots(len(evs), len(VARIANTS),
                            figsize=(3.7 * len(VARIANTS), 3.6 * len(evs)),
                            squeeze=False)
    for vi, (vname, vkey) in enumerate(VARIANTS):
        run = load_member(args.config, args.member, vkey)
        band = np.abs(run.sdf) < 1.0 * run.D          # near-wall rows/cols
        rc_list, hk_list = [], []
        for ev in evs:
            fi = int(ev['frame'])
            U = run.U_snap[fi]
            pi = np.asarray(run.pi[fi], dtype=np.float64) * run.D ** 2 / U ** 2
            iy = int(round(float(ev['y']) / run.dy)) % run.Ny
            ix = int(round(float(ev['x']) / run.dx)) % run.Nx
            h = args.half
            ys = np.arange(iy - h, iy + h) % run.Ny
            xs = np.arange(ix - h, ix + h) % run.Nx
            win = pi[np.ix_(ys, xs)]
            ei = evs.index(ev)
            v = np.nanmax(np.abs(win)) or 1.0
            ax = axs[ei][vi]
            ax.imshow(win, origin='lower', cmap='seismic', vmin=-v, vmax=v,
                      aspect='equal', interpolation='nearest')
            ax.contour(run.sdf[np.ix_(ys, xs)], levels=[0.0], colors='k',
                       linewidths=0.8)
            ax.set_title(f"{vname} f{fi} max|Pi|={v:.0f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            # near-wall band stats on the window columns that touch the band
            bwin = band[np.ix_(ys, xs)]
            cols = np.where(bwin.any(axis=0))[0]
            if cols.size >= 2:
                sub = win[:, cols]
                a, b = sub[:-1].ravel(), sub[1:].ravel()
                if a.std() > 0 and b.std() > 0:
                    rc_list.append(float(np.corrcoef(a, b)[0, 1]))
                ps = np.abs(np.fft.rfft(sub, axis=0)) ** 2
                n_k = ps.shape[0]
                hi = ps[int(0.75 * (n_k - 1)):].sum()
                hk_list.append(float(hi / max(ps[1:].sum(), 1e-30)))
        stats[vname] = {
            'row_corr_median': float(np.median(rc_list)) if rc_list else None,
            'high_ky_power_frac_median':
                float(np.median(hk_list)) if hk_list else None}
        del run
    fig.suptitle(f'{args.member}: truth Pi* near the top extreme events, '
                 'per filter variant', fontsize=11)
    fig.savefig(fig_dir / 'nearwall_truth_windows_by_variant.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    rc = {k: (v['row_corr_median'] if v['row_corr_median'] is not None
              else 0.0) for k, v in stats.items()}
    if rc['sharp'] < -0.5 and rc['gaussian'] < -0.5 and rc['ylp75'] > -0.2:
        verdict = ('ARTIFACT-STRIPED: sharp/gaussian near-wall content is '
                   'y-Nyquist checkerboard; ylp75 kept the physical residue. '
                   'Fix = body-aware filtering, not model capacity.')
    elif rc['sharp'] > -0.2 and rc['gaussian'] > -0.2:
        verdict = ('SMOOTH-PHYSICS: the near-wall excess is coherent in every '
                   'variant -- real boundary subgrid stress; the lever is '
                   'near-wall model capacity/features (and possibly ylp75 '
                   'over-suppression of real content).')
    else:
        verdict = 'MIXED: see per-variant numbers; needs a ruling.'
    out = {'member': args.member, 'stats': stats, 'verdict': verdict,
           'events_used': [int(e['frame']) for e in evs]}
    out_p = fig_dir / 'nearwall_variants.yaml'
    with open(out_p, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f'[nearwall] {args.member} VERDICT: {verdict}')
    print(f'[nearwall] stats: {stats}', flush=True)

    if args.report_run:
        rep = BRANCH_ROOT / 'reports' / args.report_run
        rep.mkdir(parents=True, exist_ok=True)
        (rep / 'summary.md').write_text(
            f'# near-wall variant check -- {args.member}\n\n'
            f'verdict: **{verdict}**\n\n```yaml\n'
            + yaml.safe_dump(stats, sort_keys=False) + '```\n')
        dw = BRANCH_ROOT / 'diagnostics' / 'digest_writer.py'
        if dw.exists():
            subprocess.run([sys.executable, str(dw), '--repo-dir',
                            str(BRANCH_ROOT), '--run-name', args.report_run,
                            '--event', 'done', '--note',
                            f'{args.member}: {verdict[:80]}'],
                           capture_output=True, text=True)


if __name__ == '__main__':
    main()
