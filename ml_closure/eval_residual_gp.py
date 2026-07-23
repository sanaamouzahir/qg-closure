"""
Per-member eval of a residual-GP-over-frozen-CNN checkpoint (Sanaa order
2026-07-22). Same conventions as eval_cnn.py: PER MEMBER, ylp75 pred taper
standard, LINEAR error-% map with the 1e-2 truth floor, per-pixel rel error =
err_pixel/|truth_pixel|. Adds the GP's uncertainty products: per-pixel sigma
map and calibration (68/95 coverage) per member x region.

Prediction:  Pi_final = ( pred_std_CNN + m_GP ) * sigma_loc,  tapered;
             sigma    = sqrt(v_GP + noise) * sigma_loc.
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

import gpytorch  # noqa: F401  (needed for the model classes)

from dataset_piff import load_conf, build_runs, split_frames, _f
from model_cnn import PiffCNN
from train_gp_residual import ResidualSVGP, gp_inputs_and_residual

HERE = Path(__file__).resolve().parent
PENDING = Path(os.environ.get(
    'QG_PENDING_MAIL',
    '/gdata/projects/ml_scope/Closure_modeling/QG-closure/reporting/pending_mail'))

try:
    from member_naming import member_stamp, member_dirname, geometry_name
except ImportError:
    member_stamp = None

from eval_cnn import _stamp, _dirname, _imshow, ylp75_taper


def main():
    ap = argparse.ArgumentParser(description="per-member eval of CNN+residual-GP")
    ap.add_argument('--ckpt', required=True, help='train_gp_residual best.pt')
    ap.add_argument('--split', default='val', choices=['val', 'train'])
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--eval-split-D', type=float, default=None)
    ap.add_argument('--gp-chunk', type=int, default=65536)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--rel-floor', type=float, default=1.0e-2)
    ap.add_argument('--rel-map-vmax', type=float, default=200.0)
    ap.add_argument('--no-mail', action='store_true')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = ck['conf']
    run_name = Path(args.ckpt).resolve().parent.name
    split_D = _f(args.eval_split_D if args.eval_split_D is not None
                 else conf['data'].get('eval_split_D', 1.25))
    dc = conf['data']

    cnn = PiffCNN(conf).to(args.device)
    cnn.load_state_dict(ck['cnn'])
    cnn.eval()
    M = int(ck['n_inducing'])
    # dim from the SAVED inducing points, not the recorded scalar (the early
    # v1-era trainer recorded a hardcoded 20 even for wider CNNs)
    gp_dim = int(ck['gp']['variational_strategy.inducing_points'].shape[-1])
    gp = ResidualSVGP(torch.zeros(M, gp_dim)).to(args.device)
    gp.load_state_dict(ck['gp'])
    lik = gpytorch.likelihoods.GaussianLikelihood().to(args.device)
    lik.load_state_dict(ck['lik'])
    gp.eval(); lik.eval()

    runs = build_runs(conf)
    siblings = [r.name for r in runs]
    geom = (geometry_name(runs[0].name) if member_stamp is not None
            else 'flow_past_cylinder')
    outdir = Path(args.outdir) if args.outdir else (
        HERE / 'results' / geom.replace(' ', '_') / run_name /
        'evaluation_predfilter_ylp75')
    outdir.mkdir(parents=True, exist_ok=True)

    frames_by_run = {}
    for ri, fi in split_frames(runs, args.split, conf):
        frames_by_run.setdefault(ri, []).append(fi)

    rows = []
    for ri, frames in sorted(frames_by_run.items()):
        r = runs[ri]
        near_m = r.valid & (r.sdf <= split_D * r.D)
        far_m = r.valid & (r.sdf > split_D * r.D)
        xs = (np.arange(r.Nx) + 0.5) * r.dx
        ys = (np.arange(r.Ny) + 0.5) * r.dy
        wake_m = (r.valid
                  & (xs[None, :] >= r.x_c + _f(dc['wake_x_lo_D']) * r.D)
                  & (xs[None, :] <= r.x_c + _f(dc['wake_x_hi_D']) * r.D)
                  & (np.abs(ys[:, None] - r.y_c) <= _f(dc['wake_y_half_D']) * r.D))
        regions = (('all', r.valid), ('near', near_m), ('far', far_m),
                   ('wake', wake_m))
        acc = {k: dict(sse=0.0, sy=0.0, sy2=0.0, n=0, c68=0, c95=0, sig=0.0)
               for k, _ in regions}
        rel = {'near': [], 'far': [], 'wake': []}
        panel = None
        mid_fi = frames[len(frames) // 2]
        for fi in frames:
            x, y, m, zeta, zeta_dot, _, lap_pl, psi_pl = r.full_frame(fi)
            b = {'x': x[None], 'y': y[None], 'mask': m[None],
                 'zeta': zeta[None], 'zeta_dot': zeta_dot[None]}
            if lap_pl is not None:
                b['lap'] = lap_pl[None]
            if psi_pl is not None:
                b['psi'] = psi_pl[None]
            with torch.no_grad():
                z, _, pred_std_m = gp_inputs_and_residual(cnn, b, args.device)
                mu = torch.empty(z.shape[0], device=args.device)
                var = torch.empty(z.shape[0], device=args.device)
                for i0 in range(0, z.shape[0], args.gp_chunk):
                    p = lik(gp(z[i0:i0 + args.gp_chunk]))
                    mu[i0:i0 + args.gp_chunk] = p.mean
                    var[i0:i0 + args.gp_chunk] = p.variance
                xg = x[None].to(args.device)
                sig_full = cnn.sigma_loc(xg)[0]
                # full-field CNN base (same object eval_cnn tapers) + GP
                # residual correction on valid pixels only
                pred_t = cnn.predict_physical(
                    xg, zeta[None].to(args.device),
                    zeta_dot[None].to(args.device) if cnn.use_zeta_dot else None,
                    lap_pl[None].to(args.device) if cnn.use_lap_input else None,
                    psi_pl[None].to(args.device)
                    if getattr(cnn, 'use_psi_input', False) else None
                )[0]
                sig_t = torch.zeros_like(sig_full)
                mk = m[None].to(args.device)[0]
                pred_t[mk] = pred_t[mk] + mu * sig_full[mk]
                sig_t[mk] = var.sqrt() * sig_full[mk]
            pred = ylp75_taper(pred_t.cpu().numpy().astype(np.float64))
            sig = sig_t.cpu().numpy().astype(np.float64)
            t = y.numpy().astype(np.float64)
            err = pred - t
            zsc = np.zeros_like(t)
            vv = r.valid & (sig > 0)
            zsc[vv] = np.abs(err[vv]) / sig[vv]
            for key, sel in regions:
                e, yy = err[sel], t[sel]
                a = acc[key]
                a['sse'] += float((e * e).sum())
                a['sy'] += float(yy.sum()); a['sy2'] += float((yy * yy).sum())
                a['n'] += int(yy.size)
                a['c68'] += int((zsc[sel] <= 1.0).sum())
                a['c95'] += int((zsc[sel] <= 2.0).sum())
                a['sig'] += float(sig[sel].sum())
            for key in ('near', 'far', 'wake'):
                sel = dict(regions)[key]
                rel[key].append((np.abs(err[sel]) /
                                 np.abs(t[sel])).astype(np.float32))
            if fi == mid_fi:
                panel = (t, pred, err, sig, float(r.times[fi]))

        stamp = _stamp(r, siblings)
        mdir = outdir / _dirname(r, siblings)
        mdir.mkdir(parents=True, exist_ok=True)
        for key, _ in regions:
            a = acc[key]
            n = max(a['n'], 1)               # G4 LOW: empty-region guard
            var_t = a['sy2'] / n - (a['sy'] / n) ** 2
            row = {'member': r.name, 'region': key, 'n_pixels': a['n'],
                   'rmse': float(np.sqrt(a['sse'] / n)),
                   'r2': float(1.0 - (a['sse'] / n) / max(var_t, 1e-30)),
                   'cov68': a['c68'] / n, 'cov95': a['c95'] / n,
                   'mean_sigma': a['sig'] / n}
            if key in rel:
                rv = np.concatenate(rel[key]) if rel[key] else np.array([])
                fin = np.isfinite(rv)
                if fin.any():
                    row.update(rel_median=float(np.median(rv[fin])),
                               rel_p90=float(np.percentile(rv[fin], 90)))
            rows.append(row)

        t, pred, err, sig, tt = panel
        vm = float(np.percentile(np.abs(t[r.valid]), 99.5))
        at = np.abs(t)
        relmap = np.where(at >= args.rel_floor,
                          100.0 * np.abs(err) / np.maximum(at, args.rel_floor),
                          0.0)
        for f_ in (t, pred, err, sig, relmap):
            f_[~r.valid] = np.nan
        fig, axs = plt.subplots(2, 4, figsize=(20, 9))
        _imshow(axs[0, 0], t, r'truth $\Pi_{FF}^*$', -vm, vm)
        _imshow(axs[0, 1], pred, r'prediction (CNN+GP) $\hat\Pi_{FF}^*$', -vm, vm)
        _imshow(axs[0, 2], err, r'error $\hat\Pi-\Pi$ (same scale)', -vm, vm)
        _imshow(axs[0, 3], sig, r'predictive $\sigma$ (physical)', 0,
                float(np.nanpercentile(sig, 99.5)), cmap='viridis')
        _imshow(axs[1, 0], relmap,
                r'per-pixel error % ; 0 where $|\Pi|<$' + f'{args.rel_floor:g}',
                0.0, args.rel_map_vmax, cmap='viridis')
        ax = axs[1, 1]
        idx = np.flatnonzero(r.valid.ravel())
        sub = np.random.default_rng(0).choice(idx, size=min(20000, idx.size),
                                              replace=False)
        yt, yp = t.ravel()[sub], pred.ravel()[sub]
        ax.plot(yt, yp, '.', ms=1, alpha=0.3)
        ax.plot([-vm, vm], [-vm, vm], 'k-', lw=0.8)
        ax.set_xlim(-vm, vm); ax.set_ylim(-vm, vm)
        cc = np.corrcoef(yt, yp)[0, 1]
        ax.set_title(f'scatter (corr {cc:.3f})', fontsize=9); ax.grid(alpha=0.3)
        ax = axs[1, 2]
        for key, col in (('near', 'tab:red'), ('far', 'tab:blue'),
                         ('wake', 'tab:green')):
            rv = np.concatenate(rel[key]); rv = rv[np.isfinite(rv) & (rv > 0)]
            ax.hist(np.log10(rv), bins=80, histtype='step', density=True,
                    color=col, label=f'{key} (med {np.median(rv):.2f})')
        ax.set_xlabel(r'$\log_{10}$ rel err'); ax.legend(fontsize=8)
        ax.grid(alpha=0.3); ax.set_title('per-pixel relative error', fontsize=9)
        ax = axs[1, 3]
        zz = (err / np.where(sig > 0, sig, np.nan))[r.valid]
        zz = zz[np.isfinite(zz)]
        ax.hist(zz, bins=100, range=(-5, 5), density=True)
        g = np.linspace(-5, 5, 200)
        ax.plot(g, np.exp(-g * g / 2) / np.sqrt(2 * np.pi), 'k-', lw=1)
        ax.set_title(f'z = err/sigma vs N(0,1)  (calibration)', fontsize=9)
        ax.grid(alpha=0.3)
        fig.suptitle(f'{stamp}   t={tt:.2f}   {run_name}   [pred ylp75-tapered]',
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(mdir / f'fields_t{tt:.1f}.png', dpi=140)
        plt.close(fig)
        print(f'[eval-gp] {r.name} done')

    cols = ['member', 'region', 'n_pixels', 'rmse', 'r2', 'rel_median',
            'rel_p90', 'cov68', 'cov95', 'mean_sigma']
    with open(outdir / 'metrics_by_member.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, '') for c in cols})
    lines = [f'{run_name} - per-member eval CNN+residual-GP ({args.split}, '
             f'near/far at {split_D}D; coverage targets 0.68/0.95)', '']
    lines.append(f"{'member':<14}{'region':<7}{'R2':>8}{'rel_med':>9}"
                 f"{'cov68':>7}{'cov95':>7}{'sigma':>10}")
    for row in rows:
        lines.append(f"{row['member']:<14}{row['region']:<7}{row['r2']:>8.4f}"
                     f"{row.get('rel_median', float('nan')):>9.3f}"
                     f"{row['cov68']:>7.3f}{row['cov95']:>7.3f}"
                     f"{row['mean_sigma']:>10.3e}")
    body = '\n'.join(lines)
    (outdir / 'summary.md').write_text(body + '\n')
    print(body)
    if not args.no_mail:
        try:
            PENDING.mkdir(parents=True, exist_ok=True)
            p = PENDING / f'evalgp_{int(time.time())}_{os.getpid()}.mail'
            p.write_text(f'To: sanaamz@mit.edu\nSubject: [QG][LANDED][sgs] '
                         f'{run_name} CNN+residual-GP per-member eval\n\n'
                         f'{body}\n\nCSV+figures: {outdir}\n')
            print(f'[eval-gp] mail spooled: {p}')
        except OSError as e:
            print(f'[eval-gp] mail spool failed ({e})')


if __name__ == '__main__':
    main()
