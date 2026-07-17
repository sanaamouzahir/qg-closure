#!/usr/bin/env python
"""diagnose_feature_candidates.py -- rank candidate pointwise features by how
well they explain the TRUE closure in the near-wall band, to design the new
feature engineering (Sanaa order 2026-07-15 21:35: missed physics at the
boundary; current [g^2, sdf, zeta] cannot represent it). CPU only.

Candidates (from omega_bar, ubar/vbar where present, sdf; centered diffs,
periodic): |grad om|, laplacian om, |om|, strain |S| & Okubo-Weiss (from u,v
if multi-frame else omega-only set), sdf, exp(-sdf/D), theta angle around the
body, om * exp(-sdf/D) (wall-weighted vorticity), |grad om| * exp(-sdf/D).
Rank by |Spearman| with Pi separately in NEAR (sdf<1D) and FAR bands, over the
val frames (subsampled). High near-band rank + low far-band rank = the wall
feature we are missing. Per member; pooled summary mailed + reports/ push.
Usage: python diagnose_feature_candidates.py --config <conf> [--every 4]
    [--report-run feature_candidates_<g>]
"""
from __future__ import annotations
import argparse, subprocess, sys
from datetime import datetime
from pathlib import Path
import numpy as np, yaml
from dataset_piff import load_conf, build_runs, split_frames

HERE = Path(__file__).resolve().parent
BRANCH_ROOT = HERE.parent
SPOOL = Path('/gdata/projects/ml_scope/Closure_modeling/QG-closure/'
             'reporting/pending_mail')


def spearman(a, b):
    if len(a) < 10 or a.std() == 0 or b.std() == 0:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def features(om, sdf, dx, dy, D, xc, yc, X, Y):
    gx = (np.roll(om, -1, 1) - np.roll(om, 1, 1)) / (2 * dx)
    gy = (np.roll(om, -1, 0) - np.roll(om, 1, 0)) / (2 * dy)
    g = np.sqrt(gx ** 2 + gy ** 2)
    lap = ((np.roll(om, -1, 1) + np.roll(om, 1, 1) - 2 * om) / dx ** 2
           + (np.roll(om, -1, 0) + np.roll(om, 1, 0) - 2 * om) / dy ** 2)
    wall = np.exp(-np.maximum(sdf, 0.0) / D)
    theta = np.arctan2(Y - yc, X - xc)
    return {'grad_om': g, 'lap_om': np.abs(lap), 'abs_om': np.abs(om),
            'sdf': sdf, 'wall_exp': wall, 'cos_theta': np.cos(theta),
            'om_wall': np.abs(om) * wall, 'grad_om_wall': g * wall,
            'lap_om_wall': np.abs(lap) * wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--every', type=int, default=4)
    ap.add_argument('--outdir', default=None,
                    help='also write summary.md here (STANDARD tree '
                         'passthrough); default: reports/ + mail only')
    ap.add_argument('--report-run', default=None)
    args = ap.parse_args()
    conf = load_conf(HERE / args.config)
    conf.setdefault('model', {})['use_grad_feature'] = False
    runs = build_runs(conf)
    split = split_frames(runs, 'val', conf)
    lines = []
    for ri, run in enumerate(runs):
        frames = [fi for rj, fi in split if rj == ri][::args.every]
        if not frames:
            continue
        X = (np.arange(run.Nx) * run.dx)[None, :].repeat(run.Ny, 0)
        Y = (np.arange(run.Ny) * run.dy)[:, None].repeat(run.Nx, 1)
        near = (run.sdf >= 0) & (run.sdf < 1.0 * run.D)
        far = (run.sdf >= 1.0 * run.D)
        acc = {}
        for fi in frames:
            om = np.asarray(run.omega[fi], dtype=np.float64)
            pi = np.asarray(run.pi[fi], dtype=np.float64)
            fs = features(om, run.sdf, run.dx, run.dy, run.D,
                          run.x_c, run.y_c, X, Y)
            for k, F in fs.items():
                for band, sel in (('near', near), ('far', far)):
                    s = spearman(F[sel].ravel()[::7], np.abs(pi[sel]).ravel()[::7])
                    acc.setdefault((k, band), []).append(s)
        row = {f'{k}_{b}': float(np.nanmedian(v)) for (k, b), v in acc.items()}
        ranked = sorted(((abs(row.get(f'{k}_near', 0)), k)
                         for k in {kk for kk, _ in acc}), reverse=True)
        lines.append(f"{run.name}: " + '  '.join(
            f"{k}={row.get(f'{k}_near', float('nan')):.2f}/"
            f"{row.get(f'{k}_far', float('nan')):.2f}"
            for _, k in ranked[:6]) + '   (near/far spearman vs |Pi|)')
        print(lines[-1], flush=True)
    body = ('Feature-candidate screen (|spearman| vs |Pi|, NEAR=sdf<1D / FAR):\n'
            + '\n'.join(lines)
            + '\n\nREAD: high near + low far = the missing wall feature. '
            'Design doc next session.')
    if args.outdir:                    # STANDARD tree passthrough
        od = Path(args.outdir)
        od.mkdir(parents=True, exist_ok=True)
        (od / 'summary.md').write_text('# feature candidates\n\n```\n'
                                       + body + '\n```\n')
    if args.report_run:
        rep = BRANCH_ROOT / 'reports' / args.report_run
        rep.mkdir(parents=True, exist_ok=True)
        (rep / 'summary.md').write_text('# feature candidates\n\n```\n'
                                        + body + '\n```\n')
        dw = BRANCH_ROOT / 'diagnostics' / 'digest_writer.py'
        if dw.exists():
            subprocess.run([sys.executable, str(dw), '--repo-dir',
                            str(BRANCH_ROOT), '--run-name', args.report_run,
                            '--event', 'done', '--note', 'feature screen done'],
                           capture_output=True)
    SPOOL.mkdir(parents=True, exist_ok=True)
    (SPOOL / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_featscreen.mail"
     ).write_text(f'To: sanaamz@mit.edu\nSubject: [QG][MONITOR][sgs-closure] '
                  f'feature-candidate screen ({args.config})\n\n{body}\n')


if __name__ == '__main__':
    main()
