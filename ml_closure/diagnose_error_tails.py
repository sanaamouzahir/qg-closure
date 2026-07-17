#!/usr/bin/env python
"""diagnose_error_tails.py -- WHERE are the huge errors, HOW MANY pixels carry
them, and is the low RMSE just zero-closure padding? (Sanaa order 2026-07-15,
follow-up to diagnose_mean_prediction.py's +-70..195 signed extrema.)

Per member (val split, filtered targets, full frames, a-priori; each member ->
its own subdirectory):

  1. WHERE: per-pixel exceedance-count map of |e| > 10*RMSE_member over the val
     window (log color, body outline = sdf 0-contour), plus the top-K extreme
     events as a table (frame, t, Re, x, y, sdf, truth, pred, e) AND as a
     domain scatter -- the literal answer to "where are they".
  2. HOW MANY: pixel-sample counts and fractions exceeding |e| > k*RMSE_member
     for k in {3, 10, 30} and above the |e| 99.9 / 99.99 percentiles; plus the
     share of TOTAL squared error carried by the worst 0.1% / 0.01% pixels
     (how tail-dominated the RMSE is).
  3. ZERO-PADDING CHECK: split valid pixels by |truth Pi*| into quiet
     (<= median), mid, active (>= q90); report per-set RMSE, R2 (own-mean),
     pixel share, and squared-error share. Also the fraction of pixels with
     |truth| < 0.05 * RMS(truth) ("closure essentially zero") and the RMSE
     there. If R2 collapses on the active set, the pooled R2 was padded.
  4. Re SANITY: Re_min/Re_max/n_unique per member; the Re-RMSE correlation is
     reported ONLY when Re varies, else the string
     'undefined -- Re constant in the val window' (the 2026-07-15 nan answer).

Cost note: one model forward per frame (same as diagnose_mean_prediction);
error stacks are cached in memory (E+T float32 + a float64 work copy: peak ~6x
the E stack alone -- fine at scale-4, recheck if the grid or window grows) so all
are computed in one pass. CPU job -- diagnostics never run on the GPU queue
(Sanaa ruling 2026-07-15).

Outputs:
  runs_piff/<model>/error_tails_diag/<member>/{metrics.yaml, extreme_events.csv}
  pngs/error_tails_diag/<model>/<member>/*.png
  <outdir>/summary_all_members.csv + summary.md
  --report-run <name>: summary copied to <branch>/reports/<name>/ + pushed
  (digest_writer, I22b/I23b).

Usage (via piff_tool_job.sh, all.q):
  cd ml_closure && python diagnose_error_tails.py \
      --ckpt runs_piff/piff_fpc_gjs_ylp75/best.pt \
      --config conf_piff_fpc_gjs_ylp75.yaml [--top-k 50] [--max-frames 0] \
      [--report-run error_tails_fpc]
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from dataset_piff import load_conf, build_runs, split_frames
from member_naming import member_dirname, member_stamp, modulation_name
from model_piff import PiffModel
from eval_piff import predict_frame, full_frame_slice

HERE = Path(__file__).resolve().parent
BRANCH_ROOT = HERE.parent
K_FACTORS = (3.0, 10.0, 30.0)
TAIL_QS = (99.9, 99.99)
ACTIVE_QS = (50.0, 90.0)          # quiet <= q50 < mid < q90 <= active
ZERO_FRAC = 0.05                  # 'essentially zero' = |truth| < 0.05*RMS(truth)


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def set_block(y, e, sel, name):
    """RMSE/R2 (own-mean)/shares on one pixel set."""
    n = int(sel.sum())
    if n == 0:
        return {'set': name, 'n': 0}
    ys, es = y[sel], e[sel]
    ss_res = float(np.sum(es ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return {'set': name, 'n': n,
            'pixel_share': float(n / y.size),
            'rmse': float(np.sqrt(np.mean(es ** 2))),
            'r2_own_mean': float(1.0 - ss_res / max(ss_tot, 1e-30)),
            'sq_error_share': ss_res}     # normalized later


@torch.no_grad()
def diagnose_member(model, run, frames, device, args, fig_root, out_root,
                    siblings=()):
    name = run.name
    # --plain-member-names (STANDARD 2026-07-17): modulation-named subdirs
    sub = member_dirname(name, args.plain_member_names, siblings)
    out_dir = out_root / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = fig_root / sub
    sl = full_frame_slice(run)
    X = (np.arange(run.Nx) * run.dx)[None, :].repeat(run.Ny, axis=0)[sl]
    Y = (np.arange(run.Ny) * run.dy)[:, None].repeat(run.Nx, axis=1)[sl]
    SDF = run.sdf[sl]

    # ---- single pass: cache 2D error/truth stacks (float32) ----
    e_stk, y_stk, m_stk, meta = [], [], [], []
    for fi in frames:
        p = predict_frame(model, run, fi, device, args.gp_chunk)
        e_stk.append((p['mu2d'] - p['truth']).astype(np.float32))
        y_stk.append(p['truth'].astype(np.float32))
        m_stk.append(p['mask'])
        meta.append((fi, p['t'], p['Re']))
    E = np.stack(e_stk); T = np.stack(y_stk); M = np.stack(m_stk)
    del e_stk, y_stk, m_stk
    ef = E[M].astype(np.float64)          # flat valid pixel-samples
    tf = T[M].astype(np.float64)
    n = ef.size
    rmse = float(np.sqrt(np.mean(ef ** 2)))
    ss_res = float(np.sum(ef ** 2))

    # ---- 2. counts / tail dominance ----
    ae = np.abs(ef)
    counts = {}
    for k in K_FACTORS:
        c = int((ae > k * rmse).sum())
        counts[f'gt_{k:g}x_rmse'] = {
            'threshold': k * rmse, 'count': c, 'fraction': c / n}
    for q in TAIL_QS:
        thr = float(np.percentile(ae, q))
        sel = ae > thr
        counts[f'gt_p{q:g}'] = {
            'threshold': thr, 'count': int(sel.sum()),
            'fraction': float(sel.mean()),
            'sq_error_share': float(np.sum(ef[sel] ** 2) / max(ss_res, 1e-30))}

    # ---- 3. zero-padding / activity split ----
    at = np.abs(tf)
    q50, q90 = np.percentile(at, ACTIVE_QS)
    rms_t = float(np.sqrt(np.mean(tf ** 2)))
    zero_thr = ZERO_FRAC * rms_t
    sets = [set_block(tf, ef, at <= q50, 'quiet_le_q50'),
            set_block(tf, ef, (at > q50) & (at < q90), 'mid'),
            set_block(tf, ef, at >= q90, 'active_ge_q90'),
            set_block(tf, ef, at < zero_thr,
                      f'near_zero_lt_{ZERO_FRAC:g}rms')]
    for s in sets:
        if s.get('sq_error_share') is not None:
            s['sq_error_share'] = float(s['sq_error_share'] / max(ss_res, 1e-30))

    # ---- 1. where: exceedance map + top-K events ----
    thr10 = 10.0 * rmse
    exceed = ((np.abs(E) > thr10) & M).sum(axis=0).astype(np.float64)
    aeE = np.where(M, np.abs(E), 0.0)
    # relative-error map (Sanaa 2026-07-15 follow-up): time-mean |e| per pixel
    # over RMS of the TRUE filtered closure -- error in closure units.
    cnt2d = M.sum(axis=0).astype(np.float64)
    mae2d = np.where(cnt2d > 0, aeE.sum(axis=0) / np.maximum(cnt2d, 1), np.nan)
    rel2d = mae2d / max(rms_t, 1e-30)
    rel_rmse = rmse / max(rms_t, 1e-30)
    np.savez_compressed(
        out_dir / 'fields.npz',
        mean_abs_error=mae2d.astype(np.float32),
        relative_error=rel2d.astype(np.float32),
        exceed_count_gt10x=exceed.astype(np.int32),
        X=X.astype(np.float32), Y=Y.astype(np.float32),
        sdf=SDF.astype(np.float32), rms_truth=rms_t, rmse=rmse)
    flat_idx = np.argpartition(aeE.ravel(), -args.top_k)[-args.top_k:]
    order = np.argsort(aeE.ravel()[flat_idx])[::-1]
    flat_idx = flat_idx[order]
    fdim, iy, ix = np.unravel_index(flat_idx, aeE.shape)
    events = []
    for f_i, y_i, x_i in zip(fdim, iy, ix):
        fi, t, Re = meta[f_i]
        ev = {'frame': int(fi), 't': float(t), 'Re': float(Re),
              'x': float(X[y_i, x_i]), 'y': float(Y[y_i, x_i]),
              'sdf': float(SDF[y_i, x_i]),
              'truth': float(T[f_i, y_i, x_i]),
              'error': float(E[f_i, y_i, x_i])}
        ev['pred'] = ev['truth'] + ev['error']
        events.append(ev)
    with open(out_dir / 'extreme_events.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        w.writeheader(); w.writerows(events)
    ev_sdf = np.array([ev['sdf'] for ev in events])
    ev_wake = float(np.mean((ev_sdf > 0) & (ev_sdf < 2.0)))

    # ---- 4. Re sanity ----
    Re_arr = np.array([m[2] for m in meta])
    per_rmse = np.array([np.sqrt(np.mean(E[i][M[i]].astype(np.float64) ** 2))
                         for i in range(len(meta))])
    re_blk = {'Re_min': float(Re_arr.min()), 'Re_max': float(Re_arr.max()),
              'n_unique': int(np.unique(np.round(Re_arr, 6)).size)}
    if re_blk['n_unique'] > 2:
        re_blk['pearson_Re_rmse'] = float(np.corrcoef(Re_arr, per_rmse)[0, 1])
    else:
        re_blk['pearson_Re_rmse'] = ('undefined -- Re constant in the val '
                                     'window (0/0, not zero correlation)')

    # ---- figures ----
    # STANDARD rule 2 (2026-07-17): titles state modulation + Re range
    stamp = member_stamp(name, re_blk['Re_min'], re_blk['Re_max'], siblings)
    ext = [X.min(), X.max(), Y.min(), Y.max()]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    if exceed.max() > 0:
        disp = np.where(exceed > 0, exceed, np.nan)
        im = ax.imshow(disp, origin='lower', extent=ext, aspect='equal',
                       cmap='seismic',
                       norm=LogNorm(vmin=1, vmax=max(exceed.max(), 2)),
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, shrink=0.85, label='exceedance count')
    else:
        ax.text(0.5, 0.5, 'no |e| > 10*RMSE pixel-samples in the val window',
                transform=ax.transAxes, ha='center')
    ax.contour(X, Y, SDF, levels=[0.0], colors='k', linewidths=1.0)
    ax.set_title(f'{stamp}:\nframes with |e| > 10*RMSE per pixel '
                 f'(RMSE {rmse:.3g}, {counts["gt_10x_rmse"]["count"]} px-samples '
                 f'= {counts["gt_10x_rmse"]["fraction"]:.2e})', fontsize=9)
    savefig(fig, fig_dir / 'huge_error_exceedance_count_map.png')

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    finite = np.isfinite(rel2d) & (rel2d > 0)
    if finite.any():
        vmin = max(float(np.nanpercentile(rel2d[finite], 1)), 1e-6)
        vmax = float(np.nanmax(rel2d[finite]))
        im = ax.imshow(np.where(finite, rel2d, np.nan), origin='lower',
                       extent=ext, aspect='equal', cmap='seismic',
                       norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)),
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, shrink=0.85, label='|e| / RMS(truth Pi*)')
    ax.contour(X, Y, SDF, levels=[0.0], colors='k', linewidths=1.0)
    ax.set_title(f'{stamp}:\ntime-mean |error| / RMS(true filtered closure) '
                 f'(rel-RMSE {rel_rmse:.3f})', fontsize=9)
    savefig(fig, fig_dir / 'relative_error_map_abs_error_over_rms_truth.png')

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sgn = np.array([np.sign(ev['error']) for ev in events])
    mag = np.array([abs(ev['error']) for ev in events])
    ax.scatter([ev['x'] for ev in events], [ev['y'] for ev in events],
               s=20 + 60 * mag / max(mag.max(), 1e-30), c=sgn, cmap='seismic',
               vmin=-1, vmax=1, edgecolors='k', linewidths=0.4)
    ax.contour(X, Y, SDF, levels=[0.0], colors='k', linewidths=1.0)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect('equal')
    ax.set_title(f'{stamp}:\ntop-{args.top_k} extreme errors '
                 f'(red +, blue -; {ev_wake:.0%} within 2 units of the body)',
                 fontsize=9)
    savefig(fig, fig_dir / 'top_extreme_errors_domain_scatter.png')

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    # hexbin raises ValueError on ANY non-positive value under log scaling
    # (G4 blocker 2026-07-15): exact zeros occur in |truth| and |error|; a
    # log-log plot cannot show them anyway, so they are dropped explicitly.
    pos = (at > 0) & (ae > 0)
    hb = ax.hexbin(at[pos], ae[pos], xscale='log', yscale='log', gridsize=60,
                   bins='log', mincnt=1)
    ax.axhline(thr10, color='k', ls=':', lw=0.8)
    ax.set_xlabel('|truth Pi*|'); ax.set_ylabel('|error|')
    ax.set_title(f'{stamp}:\nerror magnitude vs closure magnitude', fontsize=9)
    plt.colorbar(hb, ax=ax, shrink=0.85, label='log10 count')
    savefig(fig, fig_dir / 'error_magnitude_vs_closure_magnitude.png')

    metrics = {'member': name, 'results_subdir': sub,
               'member_modulation': modulation_name(name, siblings),
               'split': args.split, 'n_pixel_samples': int(n),
               'rmse': rmse, 'rms_truth': rms_t, 'rel_rmse': rel_rmse,
               'exceedance_counts': counts,
               'activity_split': sets,
               'extreme_events_within_2_of_body': ev_wake,
               'reynolds': re_blk}
    with open(out_dir / 'metrics.yaml', 'w') as f:
        yaml.safe_dump(metrics, f, sort_keys=False)
    c10 = counts['gt_10x_rmse']
    act = next(s for s in sets if s['set'] == 'active_ge_q90')
    qui = next(s for s in sets if s['set'] == 'quiet_le_q50')
    p999 = counts[f'gt_p{TAIL_QS[0]:g}']
    print(f"[{name}] |e|>10RMSE: {c10['count']} px ({c10['fraction']:.2e})  "
          f"worst-0.1% SS share {p999['sq_error_share']:.1%}  "
          f"R2(active) {act['r2_own_mean']:.3f} R2(quiet) {qui['r2_own_mean']:.3f}  "
          f"extremes near body {ev_wake:.0%}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', default='val', choices=['train', 'val'])
    ap.add_argument('--gp-chunk', type=int, default=200_000)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--fig-dir', default=None)
    ap.add_argument('--plain-member-names', action='store_true',
                    help='name per-member subdirs by modulation per the '
                         'STANDARD results tree; default keeps codenames')
    ap.add_argument('--report-run', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'cpu')
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out_root = Path(args.outdir or ckpt.parent / 'error_tails_diag')
    fig_root = Path(args.fig_dir or HERE / 'pngs' / 'error_tails_diag'
                    / ckpt.parent.name)
    conf = load_conf(HERE / args.config)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = PiffModel(ck['conf']).to(args.device)
    model.load_state_dict(ck['model'])
    model.eval()
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    ck_var = ck['conf'].get('data', {}).get('variant')
    if ck_var and not conf['data'].get('variant'):
        conf['data']['variant'] = ck_var
    ck_tsm = ck['conf'].get('zeta', {}).get('tshed_smooth')
    conf.setdefault('zeta', {})['tshed_smooth'] = float(
        ck_tsm if ck_tsm is not None else 2.992)
    runs = build_runs(conf)

    split = split_frames(runs, args.split, conf)
    siblings = [r.name for r in runs]
    rows = []
    for ri, run in enumerate(runs):
        frames = [fi for (rj, fi) in split if rj == ri]
        if args.max_frames:
            frames = frames[:args.max_frames]
        if not frames:
            print(f"[{run.name}] no {args.split} frames -- skipped", flush=True)
            continue
        met = diagnose_member(model, run, frames, args.device, args,
                              fig_root, out_root, siblings=siblings)
        c10 = met['exceedance_counts']['gt_10x_rmse']
        p999 = met['exceedance_counts'][f'gt_p{TAIL_QS[0]:g}']
        act = next(s for s in met['activity_split']
                   if s['set'] == 'active_ge_q90')
        qui = next(s for s in met['activity_split']
                   if s['set'] == 'quiet_le_q50')
        nz = next(s for s in met['activity_split']
                  if s['set'].startswith('near_zero'))
        re_c = met['reynolds']['pearson_Re_rmse']
        rows.append({
            'member': met['member'],
            'modulation': met['member_modulation'],
            'rmse': round(met['rmse'], 5),
            'rel_rmse': round(met['rel_rmse'], 3),
            'n_px': met['n_pixel_samples'],
            'n_gt_10x': c10['count'],
            'frac_gt_10x': f"{c10['fraction']:.2e}",
            'worst0.1pct_SS_share': f"{p999['sq_error_share']:.1%}",
            'r2_active_q90': round(act['r2_own_mean'], 3),
            'r2_quiet_q50': round(qui['r2_own_mean'], 3),
            'rmse_active': round(act['rmse'], 4),
            'rmse_quiet': round(qui['rmse'], 4),
            'near_zero_px_share': f"{nz.get('pixel_share', 0):.1%}",
            'extremes_near_body': f"{met['extreme_events_within_2_of_body']:.0%}",
            'Re_corr': (round(re_c, 3) if isinstance(re_c, float)
                        else 'undef(const)')})

    if not rows:
        raise SystemExit('no members produced metrics')
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / 'summary_all_members.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    lines = [f"# error-tails diagnostics -- {ckpt.parent.name} ({args.split}, "
             f"filtered {conf['data'].get('variant') or 'sharp'})", '',
             '| ' + ' | '.join(rows[0].keys()) + ' |',
             '|' + '---|' * len(rows[0])]
    lines += ['| ' + ' | '.join(str(v) for v in r.values()) + ' |' for r in rows]
    lines += ['', f'figures: {fig_root}', f'per-member yaml/csv: {out_root}']
    (out_root / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)

    if args.report_run:
        rep = BRANCH_ROOT / 'reports' / args.report_run
        rep.mkdir(parents=True, exist_ok=True)
        (rep / 'summary.md').write_text((out_root / 'summary.md').read_text())
        (rep / 'summary_all_members.csv').write_text(
            (out_root / 'summary_all_members.csv').read_text())
        dw = BRANCH_ROOT / 'diagnostics' / 'digest_writer.py'
        if dw.exists():
            r = subprocess.run([sys.executable, str(dw), '--repo-dir',
                                str(BRANCH_ROOT), '--run-name', args.report_run,
                                '--event', 'done', '--note',
                                f'error-tails diag {ckpt.parent.name} '
                                f'({len(rows)} members)'],
                               capture_output=True, text=True)
            print(f"[report] digest push rc={r.returncode} "
                  f"{(r.stdout + r.stderr).strip()[-200:]}", flush=True)


if __name__ == '__main__':
    main()
