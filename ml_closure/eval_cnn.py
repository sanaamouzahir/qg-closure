"""
CNN-only Pi_FF closure evaluation (Sanaa order 2026-07-22). PER-MEMBER, never
pooled (reporting rule 2026-07-22): every metric and every figure is one
member, and the per-pixel relative error is ALWAYS err_pixel / |truth_pixel|
— the pixel's OWN truth, never an averaged denominator.

Per member (val split, full frames):
  - fields_<t>.png: truth Pi*, prediction, error (ONE shared linear scale,
    Sanaa 2026-07-20 ruling), per-pixel |err|/|truth| (log scale), pred-truth
    scatter (correlation panel per the eval-plot convention).
  - rel_err_hist.png: log10 |err|/|truth| histogram, near vs far.
  - metrics row (near / far / all): RMSE, R2, median + p90 of |err|/|truth|,
    and frac(|truth| < 1e-3 * member rms) — the context for the relative
    metric where the truth is near zero.
Writes metrics_by_member.csv + summary.md; spools the table as a
[QG][LANDED][sgs] mail (diagnostics-table convention).
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from dataset_piff import load_conf, build_runs, split_frames, _f
from model_cnn import PiffCNN

HERE = Path(__file__).resolve().parent
PENDING = Path(os.environ.get(
    'QG_PENDING_MAIL',
    '/gdata/projects/ml_scope/Closure_modeling/QG-closure/reporting/pending_mail'))

try:
    from member_naming import member_stamp, member_dirname, geometry_name
except ImportError:                       # keep the tool usable off-branch
    member_stamp = None


def _stamp(r, siblings):
    if member_stamp is None:
        return r.name
    try:
        return member_stamp(r.name, re_lo=float(r.Re_snap.min()),
                            re_hi=float(r.Re_snap.max()), siblings=siblings)
    except Exception:
        return r.name


def _dirname(r, siblings):
    if member_stamp is None:
        return r.name
    try:
        return member_dirname(r.name, plain=True, siblings=siblings)
    except Exception:
        return r.name


def _imshow(ax, fld, title, vmin, vmax, cmap='seismic', norm=None):
    im = ax.imshow(fld, origin='lower', cmap=cmap, norm=norm,
                   **({} if norm else {'vmin': vmin, 'vmax': vmax}),
                   aspect='equal', interpolation='gaussian')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def ylp75_taper(field, lo=0.70, hi=0.80):
    """Global y-spectral cosine taper matching the ylp75 target build's gain:
    1 below lo*kN, 0 above hi*kN, cosine in between (build_ynotch_variant
    docstring: 'cut of k_y >= 0.75 k_Nyquist, cosine taper 0.70-0.80'). The
    build applied it only inside the quiescent band; globally on the TRUTH it
    is a measured near-no-op (printed per member)."""
    Ny = field.shape[0]
    fy = np.abs(np.fft.fftfreq(Ny) * Ny) / (Ny / 2.0)
    g = np.where(fy <= lo, 1.0,
                 np.where(fy >= hi, 0.0,
                          0.5 * (1.0 + np.cos(np.pi * (fy - lo) / (hi - lo)))))
    F = np.fft.fft(field, axis=0)
    return np.fft.ifft(F * g[:, None], axis=0).real


def main():
    ap = argparse.ArgumentParser(description="per-member eval of a CNN-only ckpt")
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=None,
                    help='override the ckpt-recorded config (e.g. other members)')
    ap.add_argument('--split', default='val', choices=['val', 'train'])
    ap.add_argument('--outdir', default=None,
                    help='default results/<geometry>/<run-name>/evaluation')
    ap.add_argument('--eval-split-D', type=float, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--pred-filter', default='none', choices=['none', 'ylp75'],
                    help='apply the target-variant ky-taper to the PREDICTION '
                         '(ylp75: cosine taper 0.70-0.80 kNyquist in y, the '
                         'same gain the target build used, applied globally). '
                         'The truth is untouched; the taper-removed truth '
                         'energy fraction is measured and printed per member '
                         'to confirm the truth already lacks that band.')
    ap.add_argument('--rel-floor', type=float, default=1.0e-2,
                    help='|truth| below this (Pi* units): the error-%% map '
                         'shows 0 there (no divide-by-tiny blow-ups; Sanaa '
                         '2026-07-22; default 1e-2 per her ruling - mutes '
                         'nodal-line + weak-column ratio artifacts). '
                         'Table stats unaffected.')
    ap.add_argument('--rel-floor-frac', type=float, default=0.25,
                    help='member-RELATIVE map floor (Sanaa ruling 2026-07-22b): '
                         'show error-%% only where |truth| >= frac * wake rms '
                         '(the signal-carrying pixels; the energy-weighted '
                         'analysis showed the rest carries ~1-2%% of Pi^2). '
                         '0 disables -> absolute --rel-floor only.')
    ap.add_argument('--rel-map-vmax', type=float, default=200.0,
                    help='linear color cap of the error-%% map, in percent')
    ap.add_argument('--no-mail', action='store_true')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(args.config) if args.config else ck['conf']
    if args.config:
        # G4 LOW 2026-07-22: an override conf with a different sdf_clip_D or
        # bin count silently shifts the sdf->sigma_loc mapping the ckpt's
        # recorded profile was built for (or crashes on sig_rms shape). Refuse.
        for sec, key in (('data', 'sdf_clip_D'), ('model', 'sigma_loc_bins')):
            a = _f(conf[sec].get(key, {'sdf_clip_D': 2.0, 'sigma_loc_bins': 24}[key]))
            b = _f(ck['conf'][sec].get(key, {'sdf_clip_D': 2.0, 'sigma_loc_bins': 24}[key]))
            if a != b:
                raise SystemExit(f"--config override {sec}.{key}={a} != ckpt "
                                 f"{b} — the recorded sigma_loc profile would "
                                 f"be misapplied; match the training value")
    run_name = Path(args.ckpt).resolve().parent.name
    split_D = _f(args.eval_split_D if args.eval_split_D is not None
                 else conf['data'].get('eval_split_D', 1.25))

    model = PiffCNN(conf).to(args.device)
    model.load_state_dict(ck['model'])
    model.eval()

    runs = build_runs(conf)
    siblings = [r.name for r in runs]
    geom = (geometry_name(runs[0].name) if member_stamp is not None
            else 'flow_past_cylinder')
    outdir = Path(args.outdir) if args.outdir else (
        HERE / 'results' / geom.replace(' ', '_') / run_name /
        ('evaluation' if args.pred_filter == 'none'
         else f'evaluation_predfilter_{args.pred_filter}'))
    outdir.mkdir(parents=True, exist_ok=True)

    frames_by_run = {}
    for ri, fi in split_frames(runs, args.split, conf):
        frames_by_run.setdefault(ri, []).append(fi)

    rows = []
    for ri, frames in sorted(frames_by_run.items()):
        r = runs[ri]
        near_m = r.valid & (r.sdf <= split_D * r.D)
        far_m = r.valid & (r.sdf > split_D * r.D)
        # wake band (2026-07-22): rel-error stats are only meaningful where
        # the truth is alive — outside the wake |Pi|~0 and |err|/|truth|
        # explodes regardless of skill. Same box as the crop sampler.
        dc = conf['data']
        xs = (np.arange(r.Nx) + 0.5) * r.dx
        ys = (np.arange(r.Ny) + 0.5) * r.dy
        wake_m = (r.valid
                  & (xs[None, :] >= r.x_c + _f(dc['wake_x_lo_D']) * r.D)
                  & (xs[None, :] <= r.x_c + _f(dc['wake_x_hi_D']) * r.D)
                  & (np.abs(ys[:, None] - r.y_c) <= _f(dc['wake_y_half_D']) * r.D))
        acc = {k: dict(sse=0.0, sy=0.0, sy2=0.0, n=0)
               for k in ('all', 'near', 'far', 'wake')}
        rel = {'near': [], 'far': [], 'wake': []}
        small = {'near': 0, 'far': 0, 'wake': 0}
        panel = None
        tt_num = tt_den = 0.0        # taper-removed truth energy (diagnostic)
        mid_fi = frames[len(frames) // 2]
        for fi in frames:
            x, y, m, zeta, zeta_dot, _, lap_pl, psi_pl = r.full_frame(fi)
            with torch.no_grad():
                pred = model.predict_physical(
                    x[None].to(args.device), zeta[None].to(args.device),
                    zeta_dot[None].to(args.device) if model.use_zeta_dot else None,
                    lap_pl[None].to(args.device) if model.use_lap_input else None,
                    psi_pl[None].to(args.device)
                    if getattr(model, 'use_psi_input', False) else None
                )[0].cpu().numpy().astype(np.float64)
            y = y.numpy().astype(np.float64)
            if args.pred_filter == 'ylp75':
                ty = ylp75_taper(y)
                tt_num += float(((y - ty) ** 2).sum())
                tt_den += float((y ** 2).sum())
                pred = ylp75_taper(pred)
            err = pred - y
            for key, sel in (('all', r.valid), ('near', near_m),
                             ('far', far_m), ('wake', wake_m)):
                e, yy = err[sel], y[sel]
                acc[key]['sse'] += float((e * e).sum())
                acc[key]['sy'] += float(yy.sum())
                acc[key]['sy2'] += float((yy * yy).sum())
                acc[key]['n'] += int(yy.size)
            for key, sel in (('near', near_m), ('far', far_m),
                             ('wake', wake_m)):
                rel[key].append((np.abs(err[sel]) /
                                 np.abs(y[sel])).astype(np.float32))
            if fi == mid_fi:
                panel = (y, pred, err, float(r.times[fi]))
        rms_y = np.sqrt(acc['all']['sy2'] / acc['all']['n'])
        for key, sel in (('near', near_m), ('far', far_m), ('wake', wake_m)):
            # truth-near-zero census for the relative metric (per member)
            tv = np.concatenate([np.abs(r.pi[fi][sel].astype(np.float64))
                                 * (r.D ** 2 / _f(r.U_snap[fi]) ** 2)
                                 for fi in frames])
            small[key] = float((tv < 1.0e-3 * rms_y).mean())

        if args.pred_filter == 'ylp75':
            print(f'[filter] {r.name}: global ylp75 taper removes '
                  f'{tt_num / max(tt_den, 1e-300):.3e} of TRUTH energy '
                  f'(must be <<1 for a clean comparison; pred was tapered)')
        stamp = _stamp(r, siblings)
        mdir = outdir / _dirname(r, siblings)
        mdir.mkdir(parents=True, exist_ok=True)
        for key in ('all', 'near', 'far', 'wake'):
            a = acc[key]
            var = a['sy2'] / a['n'] - (a['sy'] / a['n']) ** 2
            row = {'member': r.name, 'region': key, 'n_pixels': a['n'],
                   'rmse': float(np.sqrt(a['sse'] / a['n'])),
                   'r2': float(1.0 - (a['sse'] / a['n']) / max(var, 1e-30))}
            if key in rel:
                rv = np.concatenate(rel[key])
                # stats over FINITE ratios only (G4 MEDIUM 2026-07-22: exact-
                # zero truth pixels make |err|/|truth| = inf and np.percentile
                # then returns NaN, corrupting the mailed table). The dropped
                # fraction is reported, and frac_truth_lt_1e3rms carries the
                # near-zero-truth context.
                fin = np.isfinite(rv)
                rvf = rv[fin]
                row.update(rel_median=float(np.median(rvf)),
                           rel_p90=float(np.percentile(rvf, 90)),
                           frac_rel_nonfinite=float(1.0 - fin.mean()),
                           frac_truth_lt_1e3rms=small[key])
            rows.append(row)

        # ---- figures ------------------------------------------------------ #
        y, pred, err, t = panel
        vm = float(np.percentile(np.abs(y[r.valid]), 99.5))
        # error-%% map: LINEAR scale, zero where the truth is below the floor
        # (Sanaa 2026-07-22: log scale + near-zero-truth blow-ups made the
        # map unreadable). Table stats elsewhere are untouched by the floor.
        at = np.abs(y)
        rms_wake = float(np.sqrt(acc['wake']['sy2'] / max(acc['wake']['n'], 1)))
        floor_eff = max(args.rel_floor, args.rel_floor_frac * rms_wake)
        shown = float((at[wake_m] >= floor_eff).mean())
        relmap = np.where(at >= floor_eff,
                          100.0 * np.abs(err) / np.maximum(at, floor_eff),
                          0.0)
        relmap[~r.valid] = np.nan
        for f in (y, pred, err):
            f[~r.valid] = np.nan
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        _imshow(axs[0, 0], y, r'truth $\Pi_{FF}^*$', -vm, vm)
        _imshow(axs[0, 1], pred, r'prediction $\hat\Pi_{FF}^*$', -vm, vm)
        _imshow(axs[0, 2], err, r'error $\hat\Pi-\Pi$ (same scale)', -vm, vm)
        _imshow(axs[1, 0], relmap,
                r'per-pixel error % ($100\,|\hat\Pi-\Pi|/|\Pi|$; '
                r'signal pixels only, $|\Pi|\geq$' + f'{floor_eff:.2f}'
                + f' = {100 * shown:.0f}% of wake px)',
                0.0, args.rel_map_vmax, cmap='viridis')
        ax = axs[1, 1]
        sel = r.valid.copy()
        idx = np.flatnonzero(sel.ravel())
        sub = np.random.default_rng(0).choice(idx, size=min(20000, idx.size),
                                              replace=False)
        yt, yp = y.ravel()[sub], pred.ravel()[sub]
        ax.plot(yt, yp, '.', ms=1, alpha=0.3)
        lim = vm
        ax.plot([-lim, lim], [-lim, lim], 'k-', lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(r'$\Pi_{FF}^*$'); ax.set_ylabel(r'$\hat\Pi_{FF}^*$')
        cc = np.corrcoef(yt, yp)[0, 1]
        ax.set_title(f'scatter (corr {cc:.3f})', fontsize=9)
        ax.grid(alpha=0.3)
        ax = axs[1, 2]
        for key, col in (('near', 'tab:red'), ('far', 'tab:blue')):
            rv = np.concatenate(rel[key])
            rv = rv[np.isfinite(rv) & (rv > 0)]
            ax.hist(np.log10(rv), bins=80, histtype='step', density=True,
                    color=col, label=f'{key} (med {np.median(rv):.2f})')
        ax.set_xlabel(r'$\log_{10}\,|\hat\Pi-\Pi|/|\Pi|$')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title('per-pixel relative error', fontsize=9)
        note = '' if args.pred_filter == 'none' else '   [pred ylp75-tapered]'
        fig.suptitle(f'{stamp}   t={t:.2f}   {run_name}{note}', fontsize=11)
        fig.tight_layout()
        fig.savefig(mdir / f'fields_t{t:.1f}.png', dpi=140)
        plt.close(fig)
        print(f'[eval] {r.name}: ' + '  '.join(
            f"{q['region']} R2 {q['r2']:.4f} relmed "
            f"{q.get('rel_median', float('nan')):.3f}"
            for q in rows[-4:]))

    # ---- table + mail ----------------------------------------------------- #
    cols = ['member', 'region', 'n_pixels', 'rmse', 'r2', 'rel_median',
            'rel_p90', 'frac_rel_nonfinite', 'frac_truth_lt_1e3rms']
    with open(outdir / 'metrics_by_member.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, '') for c in cols})
    lines = [f'{run_name} — per-member eval ({args.split} split, '
             f'near/far at {split_D}D)', '']
    hdr = f"{'member':<14}{'region':<7}{'R2':>8}{'RMSE':>11}{'rel_med':>9}{'rel_p90':>9}"
    lines.append(hdr)
    for row in rows:
        lines.append(f"{row['member']:<14}{row['region']:<7}{row['r2']:>8.4f}"
                     f"{row['rmse']:>11.3e}"
                     f"{row.get('rel_median', float('nan')):>9.3f}"
                     f"{row.get('rel_p90', float('nan')):>9.3f}")
    body = '\n'.join(lines)
    (outdir / 'summary.md').write_text(body + '\n')
    print(body)
    if not args.no_mail:
        try:
            PENDING.mkdir(parents=True, exist_ok=True)
            p = PENDING / f'evalcnn_{int(time.time())}_{os.getpid()}.mail'
            p.write_text(f'To: sanaamz@mit.edu\n'
                         f'Subject: [QG][LANDED][sgs] {run_name} per-member '
                         f'eval table\n\n{body}\n\nCSV+figures: {outdir}\n')
            print(f'[eval] mail spooled: {p}')
        except OSError as e:
            print(f'[eval] mail spool failed ({e}) — table above is authoritative')


if __name__ == '__main__':
    main()
