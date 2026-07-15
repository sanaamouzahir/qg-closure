#!/usr/bin/env python
"""diagnose_mean_prediction.py -- MEAN-prediction diagnostic suite, per member
(Sanaa order 2026-07-15: before sigma stage 3, characterize the mean itself).

For ONE checkpoint + config (the ylp75 production candidates; the config's
`variant: gaussian_jonly_ylp75` selects the FILTERED fields), over the val
split, per ensemble member (each member -> its OWN subdirectory):

  1. Global + per-frame R2, RMSE, bias, MAX signed error, MIN signed error
     (e = mu - truth; signed, NOT absolute -- per spec).
  2. Re--error relationship: per-frame RMSE and bias vs Re_snap(t)
     (Pearson + Spearman; degenerates gracefully for const members).
  3. Error--location: time-mean signed-error and |error| maps; Pearson
     correlation of e and |e| with x, y, sdf over all valid pixels; mean
     |error| profiles along x and along y.
  4. SPATIAL autocorrelation (masked, FFT-based, frame-averaged, radially
     averaged) of: the error, the TRUE closure, the PREDICTED closure; plus
     the pred--truth spatial CROSS-correlation. Integral length scale and
     1/e-crossing per field.
  5. Predicted-vs-true closure: hexbin scatter (a-priori), zero-lag
     pred-truth correlation coefficient.
  6. TEMPORAL autocorrelation of the per-frame bias and RMSE series.

Masked autocorrelation convention: f' = (f - mean_valid) * mask, zero outside;
    rho(dx) = IFFT(|FFT(f')|^2) / IFFT(|FFT(mask)|^2), normalized to rho(0)=1
(the mask-count normalization removes the sponge/body footprint; domain is
periodic). Cross-correlation likewise with FFT(a')*conj(FFT(b')).

Outputs (nothing existing overwritten):
  <outdir>/<member>/metrics.yaml            headline numbers + correlations
  <outdir>/<member>/per_frame_metrics.csv   t, Re, r2, rmse, bias, max_e, min_e
  <fig-dir>/<member>/*.png                  full-English filenames, seismic,
                                            aspect-preserving (hard rule 10)
  <outdir>/summary_all_members.csv          one row per member
  <outdir>/summary.md                       compact table
  <fig-dir>/error_vs_reynolds_all_members.png  pooled, member-colored
  --report-run <name>: summary.csv+md ALSO copied to <branch>/reports/<name>/
    and pushed via diagnostics/digest_writer.py (I22b/I23b) -- phone-readable.

Defaults: outdir = <ckpt dir>/mean_prediction_diag/, fig-dir =
pngs/mean_prediction_diag/<ckpt-dir-name>/.

Usage (GPU -- model forwards; via piff_tool_job.sh):
  cd ml_closure && python diagnose_mean_prediction.py \
      --ckpt runs_piff/piff_fpc_gjs_ylp75/best.pt \
      --config conf_piff_fpc_gjs_ylp75.yaml [--split val] [--acf-every 2] \
      [--report-run mean_prediction_diag_fpc]
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

from dataset_piff import load_conf, build_runs, split_frames
from model_piff import PiffModel
from eval_piff import predict_frame, full_frame_slice

HERE = Path(__file__).resolve().parent
BRANCH_ROOT = HERE.parent
MAX_SCATTER = 2_000_000          # reservoir cap for the hexbin pairs


# ------------------------------------------------------------------ helpers

def spearman(a, b):
    """Spearman rho without scipy.stats dependency drift: Pearson of ranks."""
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


class MaskedAcf2D:
    """Frame-averaged masked spatial autocorrelation accumulator.
    add(field, mask): accumulate IFFT(|FFT((f-mean)*m)|^2) and IFFT(|FFT(m)|^2);
    profile(): radially averaged rho(r), rho(0)=1."""

    def __init__(self, shape):
        self.num = np.zeros(shape, dtype=np.float64)
        self.den = np.zeros(shape, dtype=np.float64)
        self.n = 0

    @staticmethod
    def _corr(a, b):
        fa = np.fft.rfft2(a)
        fb = np.fft.rfft2(b)
        return np.fft.irfft2(fa * np.conj(fb), s=a.shape)

    def add(self, f, mask, g=None):
        m = mask.astype(np.float64)
        fp = np.where(mask, f - f[mask].mean(), 0.0)
        if g is None:
            self.num += self._corr(fp, fp)
        else:                                      # cross-correlation a<->b
            gp = np.where(mask, g - g[mask].mean(), 0.0)
            self.num += self._corr(fp, gp)
        self.den += self._corr(m, m)
        self.n += 1

    def rho2d(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            rho = self.num / np.maximum(self.den, self.n * 4.0)  # >=4 px overlap
        z = rho[0, 0]
        return rho / z if np.isfinite(z) and abs(z) > 0 else rho

    def radial_profile(self, dx, dy):
        rho = self.rho2d()
        ny, nx = rho.shape
        iy = np.fft.fftfreq(ny) * ny
        ix = np.fft.fftfreq(nx) * nx
        R = np.sqrt((iy[:, None] * dy) ** 2 + (ix[None, :] * dx) ** 2)
        dr = min(dx, dy)
        nb = int(R.max() / dr) + 1
        idx = np.minimum((R / dr).astype(np.int64), nb - 1)
        cnt = np.bincount(idx.ravel(), minlength=nb).astype(np.float64)
        s = np.bincount(idx.ravel(), weights=rho.ravel(), minlength=nb)
        prof = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        r = (np.arange(nb) + 0.5) * dr
        return r, prof


def integral_scale(r, prof):
    """Integral of rho dr up to the first zero crossing, plus 1/e crossing."""
    v = np.nan_to_num(prof, nan=0.0)
    zero = np.argmax(v <= 0.0) if (v <= 0.0).any() else len(v)
    L = float(np.trapz(v[:max(zero, 1)], r[:max(zero, 1)]))
    below = np.argmax(v < 1.0 / np.e) if (v < 1.0 / np.e).any() else len(v) - 1
    return L, float(r[below])


def temporal_acf(x, max_lag):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    v = float(np.dot(x, x))
    if v <= 0 or len(x) < 3:
        return np.array([1.0])
    L = min(max_lag, len(x) - 1)
    return np.array([np.dot(x[:len(x) - k], x[k:]) / v for k in range(L + 1)])


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def field_map(ax, arr2d, extent, title, vmax=None):
    """seismic, centered, aspect-preserving (hard rule 10)."""
    v = vmax if vmax else np.nanmax(np.abs(arr2d)) or 1.0
    im = ax.imshow(arr2d, origin='lower', cmap='seismic', vmin=-v, vmax=v,
                   extent=extent, aspect='equal', interpolation='nearest')
    ax.set_title(title, fontsize=9)
    return im


# ------------------------------------------------------------------- driver

@torch.no_grad()
def diagnose_member(model, run, frames, device, args, fig_root, out_root):
    name = run.name
    fig_dir = fig_root / name
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    sl = full_frame_slice(run)
    # coordinates: physical grid from dx,dy (origin at domain corner)
    X = (np.arange(run.Nx) * run.dx)[None, :].repeat(run.Ny, axis=0)[sl]
    Y = (np.arange(run.Ny) * run.dy)[:, None].repeat(run.Nx, axis=1)[sl]
    SDF = run.sdf[sl]

    shape = X.shape
    sum_e = np.zeros(shape); sum_ae = np.zeros(shape)
    cnt = np.zeros(shape)
    acf_err = MaskedAcf2D(shape)
    acf_tru = MaskedAcf2D(shape)
    acf_prd = MaskedAcf2D(shape)
    xcf_pt = MaskedAcf2D(shape)

    # running first/second moments for corr(e, loc) over ALL valid pixels
    feats = {'x': X, 'y': Y, 'sdf': SDF}
    mom = {k: np.zeros(5) for k in ('e', 'ae')}         # n, sum, sum2 (via [0:3])
    cross = {(ek, fk): 0.0 for ek in ('e', 'ae') for fk in feats}
    fsum = {fk: np.zeros(2) for fk in feats}            # sum, sum2 (valid px)
    n_px = 0

    rows = []
    rng = np.random.default_rng(0)
    sc_y = np.empty(0); sc_mu = np.empty(0)
    g_max, g_min = -np.inf, np.inf
    e_ss, y_sum, y_sq = 0.0, 0.0, 0.0                    # global R2 accumulators
    mu_sum, mu_sq, ymu_sum = 0.0, 0.0, 0.0               # exact zero-lag corr (G4 #2)

    for j, fi in enumerate(frames):
        p = predict_frame(model, run, fi, device, args.gp_chunk)
        m = p['mask']
        e2d = p['mu2d'] - p['truth']                     # SIGNED error, 2D
        e = p['mu'] - p['y']                             # flat, valid px

        rows.append({'frame': fi, 't': p['t'], 'Re': p['Re'],
                     'r2': 1.0 - np.sum(e ** 2)
                           / max(np.sum((p['y'] - p['y'].mean()) ** 2), 1e-30),
                     'rmse': float(np.sqrt(np.mean(e ** 2))),
                     'bias': float(np.mean(e)),
                     'max_err': float(e.max()), 'min_err': float(e.min())})
        g_max = max(g_max, float(e.max())); g_min = min(g_min, float(e.min()))
        e_ss += float(np.sum(e ** 2)); y_sum += float(np.sum(p['y']))
        y_sq += float(np.sum(p['y'] ** 2)); n_px += e.size
        mu_sum += float(np.sum(p['mu'])); mu_sq += float(np.sum(p['mu'] ** 2))
        ymu_sum += float(np.dot(p['y'], p['mu']))

        sum_e[m] += e2d[m]; sum_ae[m] += np.abs(e2d[m]); cnt[m] += 1.0

        ae = np.abs(e)
        for ek, v in (('e', e), ('ae', ae)):
            mom[ek][0] += v.size; mom[ek][1] += v.sum(); mom[ek][2] += (v ** 2).sum()
        for fk, F in feats.items():
            fv = F[m]
            fsum[fk] += (fv.sum(), (fv ** 2).sum())
            cross[('e', fk)] += float(np.dot(e, fv))
            cross[('ae', fk)] += float(np.dot(ae, fv))

        if j % args.acf_every == 0:
            acf_err.add(e2d, m)
            acf_tru.add(p['truth'], m)
            acf_prd.add(np.nan_to_num(p['mu2d']), m)
            xcf_pt.add(np.nan_to_num(p['mu2d']), m, g=p['truth'])

        take = min(max(MAX_SCATTER // max(len(frames), 1), 1000), e.size)
        ii = rng.choice(e.size, take, replace=False)
        sc_y = np.concatenate([sc_y, p['y'][ii]])
        sc_mu = np.concatenate([sc_mu, p['mu'][ii]])

    # ---- global numbers ----
    y_mean = y_sum / n_px
    ss_tot = y_sq - n_px * y_mean ** 2
    glob = {'n_frames': len(frames), 'n_pixels': int(n_px),
            'r2': float(1.0 - e_ss / max(ss_tot, 1e-30)),
            'rmse': float(np.sqrt(e_ss / n_px)),
            'bias': float(mom['e'][1] / n_px),
            'max_err': g_max, 'min_err': g_min}

    def _corr(ek, fk):
        n = mom[ek][0]
        me, mf = mom[ek][1] / n, fsum[fk][0] / n
        ve = mom[ek][2] / n - me ** 2
        vf = fsum[fk][1] / n - mf ** 2
        if ve <= 0 or vf <= 0:
            return float('nan')
        return float((cross[(ek, fk)] / n - me * mf) / np.sqrt(ve * vf))

    loc_corr = {f'{ek}_vs_{fk}': _corr(ek, fk)
                for ek in ('e', 'ae') for fk in feats}

    Re_arr = np.array([r['Re'] for r in rows])
    rmse_arr = np.array([r['rmse'] for r in rows])
    bias_arr = np.array([r['bias'] for r in rows])
    re_err = {'pearson_Re_rmse': pearson(Re_arr, rmse_arr),
              'spearman_Re_rmse': spearman(Re_arr, rmse_arr),
              'pearson_Re_bias': pearson(Re_arr, bias_arr),
              'Re_min': float(Re_arr.min()), 'Re_max': float(Re_arr.max())}

    scales = {}
    prof = {}
    for tag, acc in (('error', acf_err), ('truth', acf_tru),
                     ('pred', acf_prd), ('pred_x_truth', xcf_pt)):
        r, p_ = acc.radial_profile(run.dx, run.dy)
        prof[tag] = (r, p_)
        L, re_ = integral_scale(r, p_)
        scales[tag] = {'integral_scale': L, 'one_over_e_crossing': re_}
    # rho2d self-normalizes, so the cross accumulator's lag-0 is trivially 1
    # (G4 #3) -- the honest zero-lag number is the EXACT pixel correlation:
    my, mm = y_sum / n_px, mu_sum / n_px
    vy = y_sq / n_px - my ** 2
    vm = mu_sq / n_px - mm ** 2
    zero_lag_corr = (float((ymu_sum / n_px - my * mm) / np.sqrt(vy * vm))
                     if vy > 0 and vm > 0 else float('nan'))

    # ---- figures ----
    ext = [0.0, run.Nx * run.dx, 0.0, run.Ny * run.dy]
    with np.errstate(invalid='ignore'):
        me2d = np.where(cnt > 0, sum_e / np.maximum(cnt, 1), np.nan)
        mae2d = np.where(cnt > 0, sum_ae / np.maximum(cnt, 1), np.nan)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    im0 = field_map(axs[0], me2d, ext, f'{name}: time-mean SIGNED error (mu - truth)')
    plt.colorbar(im0, ax=axs[0], shrink=0.85)
    im1 = field_map(axs[1], mae2d, ext, f'{name}: time-mean |error|')
    plt.colorbar(im1, ax=axs[1], shrink=0.85)
    savefig(fig, fig_dir / 'time_mean_error_maps_signed_and_absolute.png')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].scatter(Re_arr, rmse_arr, s=12); axs[0].set_xlabel('Re(t)')
    axs[0].set_ylabel('per-frame RMSE')
    axs[0].set_title(f"RMSE vs Re  (pearson {re_err['pearson_Re_rmse']:.3f})")
    axs[1].scatter(Re_arr, bias_arr, s=12, color='tab:red')
    axs[1].axhline(0, lw=0.6, color='k')
    axs[1].set_xlabel('Re(t)'); axs[1].set_ylabel('per-frame bias')
    axs[1].set_title(f"bias vs Re  (pearson {re_err['pearson_Re_bias']:.3f})")
    fig.suptitle(name, fontsize=10)
    savefig(fig, fig_dir / 'error_vs_reynolds_number.png')

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for tag, style in (('truth', '-'), ('pred', '--'),
                       ('error', '-.'), ('pred_x_truth', ':')):
        r, p_ = prof[tag]
        keep = r <= 0.25 * min(ext[1], ext[3])
        ax.plot(r[keep], p_[keep], style,
                label=f"{tag} (L={scales[tag]['integral_scale']:.3g})")
    ax.axhline(0, lw=0.6, color='k'); ax.axhline(1 / np.e, lw=0.6, ls=':',
                                                 color='gray')
    ax.set_xlabel('r (physical units)'); ax.set_ylabel('rho(r)')
    ax.set_title(f'{name}: radial spatial (auto/cross)correlation')
    ax.legend(fontsize=8)
    savefig(fig, fig_dir / 'spatial_autocorrelation_radial_profiles.png')

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    hb = ax.hexbin(sc_y, sc_mu, gridsize=80, bins='log', mincnt=1)
    lim = np.nanmax(np.abs(np.concatenate([sc_y, sc_mu]))) or 1.0
    ax.plot([-lim, lim], [-lim, lim], 'k-', lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('true Pi* (filtered)'); ax.set_ylabel('predicted Pi*')
    ax.set_title(f"{name}: pred vs truth (R2 {glob['r2']:.3f}, "
                 f"corr {zero_lag_corr:.3f})")
    plt.colorbar(hb, ax=ax, shrink=0.85, label='log10 count')
    savefig(fig, fig_dir / 'predicted_vs_true_closure_hexbin.png')

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    with np.errstate(invalid='ignore'):
        w = cnt.sum(axis=0)
        axs[0].plot(X[0], np.nansum(sum_ae, axis=0) / np.maximum(w, 1))
        h = cnt.sum(axis=1)
        axs[1].plot(np.nansum(sum_ae, axis=1) / np.maximum(h, 1), Y[:, 0])
    axs[0].set_xlabel('x'); axs[0].set_ylabel('mean |error|')
    axs[1].set_ylabel('y'); axs[1].set_xlabel('mean |error|')
    fig.suptitle(f'{name}: mean |error| profiles along x and y '
                 '(reflection/boundary signature check)', fontsize=10)
    savefig(fig, fig_dir / 'error_profiles_along_x_and_y.png')

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for series, lab in ((bias_arr, 'bias'), (rmse_arr, 'rmse')):
        ac = temporal_acf(series, max_lag=min(40, len(series) - 2))
        ax.plot(np.arange(len(ac)), ac, marker='.', label=lab)
    ax.axhline(0, lw=0.6, color='k')
    ax.set_xlabel('lag (frames)'); ax.set_ylabel('temporal ACF')
    ax.set_title(f'{name}: temporal autocorrelation of per-frame error stats')
    ax.legend(fontsize=8)
    savefig(fig, fig_dir / 'temporal_autocorrelation_of_frame_error.png')

    # ---- artifacts ----
    with open(out_dir / 'per_frame_metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    metrics = {'member': name, 'split': args.split,
               'variant': run.variant or 'sharp',
               'global': glob, 'location_correlation': loc_corr,
               'reynolds_vs_error': re_err,
               'spatial_correlation_scales': scales,
               'pred_truth_zero_lag_corr': zero_lag_corr,
               'acf_every': args.acf_every}
    with open(out_dir / 'metrics.yaml', 'w') as f:
        yaml.safe_dump(metrics, f, sort_keys=False)
    print(f"[{name}] R2 {glob['r2']:.4f}  RMSE {glob['rmse']:.4f}  "
          f"bias {glob['bias']:+.2e}  max_e {glob['max_err']:+.3f}  "
          f"min_e {glob['min_err']:+.3f}  L_err "
          f"{scales['error']['integral_scale']:.3g} vs L_truth "
          f"{scales['truth']['integral_scale']:.3g}", flush=True)
    return metrics, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', default='val', choices=['train', 'val'])
    ap.add_argument('--gp-chunk', type=int, default=200_000)
    ap.add_argument('--acf-every', type=int, default=2,
                    help='spatial ACF every Nth frame (cost control)')
    ap.add_argument('--max-frames', type=int, default=0,
                    help='per-member frame cap, 0 = all (smoke: 6)')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--fig-dir', default=None)
    ap.add_argument('--report-run', default=None,
                    help='also copy summary into <branch>/reports/<name>/ and '
                         'push via digest_writer (I23b)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'cpu')
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out_root = Path(args.outdir or ckpt.parent / 'mean_prediction_diag')
    fig_root = Path(args.fig_dir or HERE / 'pngs' / 'mean_prediction_diag'
                    / ckpt.parent.name)
    conf = load_conf(HERE / args.config)
    # eval_piff conventions: the model is built from the ckpt's OWN conf
    # (film/ORDER-3 flags as trained), and the ckpt's data variant + grad
    # feature propagate into the data conf -- a gaussian-trained model is
    # never silently evaluated on sharp targets.
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = PiffModel(ck['conf']).to(args.device)
    model.load_state_dict(ck['model'])
    model.eval()
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    ck_var = ck['conf'].get('data', {}).get('variant')
    if ck_var and not conf['data'].get('variant'):
        conf['data']['variant'] = ck_var
    ck_tsm = ck['conf'].get('zeta', {}).get('tshed_smooth')
    ev_tsm = conf.get('zeta', {}).get('tshed_smooth')
    if (ck_tsm is not None and ev_tsm is not None
            and abs(float(ev_tsm) - float(ck_tsm)) > 1e-12):
        raise SystemExit(f'tshed_smooth mismatch: ckpt {ck_tsm} vs conf {ev_tsm} '
                         '(zeta_dot smoothing must match training)')
    # eval_piff parity (G4 #1): the TRAINING smoothing scale always wins
    conf.setdefault('zeta', {})['tshed_smooth'] = float(
        ck_tsm if ck_tsm is not None else 2.992)
    runs = build_runs(conf)

    split = split_frames(runs, args.split, conf)
    all_metrics, summary_rows = [], []
    for ri, run in enumerate(runs):
        frames = [fi for (rj, fi) in split if rj == ri]
        if args.max_frames:
            frames = frames[:args.max_frames]
        if not frames:
            print(f"[{run.name}] no {args.split} frames -- skipped", flush=True)
            continue
        met, _rows = diagnose_member(model, run, frames, args.device, args,
                                     fig_root, out_root)
        all_metrics.append(met)
        g, s = met['global'], met['spatial_correlation_scales']
        summary_rows.append({
            'member': run.name, 'n_frames': g['n_frames'],
            'r2': round(g['r2'], 4), 'rmse': round(g['rmse'], 5),
            'bias': f"{g['bias']:+.3e}",
            'max_err': round(g['max_err'], 4), 'min_err': round(g['min_err'], 4),
            'pearson_Re_rmse': round(met['reynolds_vs_error']['pearson_Re_rmse'], 3)
                if np.isfinite(met['reynolds_vs_error']['pearson_Re_rmse']) else 'nan',
            'corr_ae_sdf': round(met['location_correlation']['ae_vs_sdf'], 3),
            'corr_ae_y': round(met['location_correlation']['ae_vs_y'], 3),
            'L_error': round(s['error']['integral_scale'], 4),
            'L_truth': round(s['truth']['integral_scale'], 4),
            'L_pred': round(s['pred']['integral_scale'], 4),
            'pred_truth_corr': round(met['pred_truth_zero_lag_corr'], 4)})

    if not summary_rows:
        raise SystemExit(f'no members produced {args.split} metrics -- '
                         'check the split windows / lomo_holdout')
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / 'summary_all_members.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    # pooled Re--error, member-colored
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for met in all_metrics:
        member = met['member']
        pf = out_root / member / 'per_frame_metrics.csv'
        with open(pf) as f:
            rr = list(csv.DictReader(f))
        ax.scatter([float(r['Re']) for r in rr], [float(r['rmse']) for r in rr],
                   s=10, label=member, alpha=0.7)
    ax.set_xlabel('Re(t)'); ax.set_ylabel('per-frame RMSE')
    ax.set_title(f'error vs Reynolds number, all members ({args.split}, '
                 f'{ckpt.parent.name})')
    ax.legend(fontsize=7)
    savefig(fig, fig_root / 'error_vs_reynolds_all_members.png')

    lines = [f"# mean-prediction diagnostics -- {ckpt.parent.name} "
             f"({args.split}, filtered variant "
             f"{conf['data'].get('variant') or 'sharp'})", '',
             '| ' + ' | '.join(summary_rows[0].keys()) + ' |',
             '|' + '---|' * len(summary_rows[0])]
    lines += ['| ' + ' | '.join(str(v) for v in r.values()) + ' |'
              for r in summary_rows]
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
                                f'mean-prediction diag {ckpt.parent.name} '
                                f'({len(summary_rows)} members)'],
                               capture_output=True, text=True)
            print(f"[report] digest push rc={r.returncode} "
                  f"{(r.stdout + r.stderr).strip()[-200:]}", flush=True)
            if r.returncode != 0:
                print('[report] WARNING: the phone-readable summary may NOT '
                      'be on origin -- check reports/ next session', flush=True)
        else:
            print(f"[report] digest_writer missing at {dw} -- summary "
                  "written locally only", flush=True)


if __name__ == '__main__':
    main()
