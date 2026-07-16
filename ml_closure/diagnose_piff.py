"""
diagnose_piff.py — standalone Pi_FF diagnostics package (Sanaa ORDER 2, 2026-07-13).

Eight diagnostics on a trained checkpoint, PER MEMBER (never pooled):
  1. calibration reliability diagram per member
  2. R^2 / NLL binned by zeta (regime-resolved skill)
  3. sigma vs |grad omega_bar| binned scatter (sigma-shape evidence for Arm F)
  4. residual excess kurtosis + residual radial spectra per member (float64)
  5. coverage split by obstacle distance (SDF bins) + wake-box vs freestream
  6. backscatter sign accuracy overall / wake-only / strong-pixel
  7. zeta-ARD lengthscale trajectory over training (log parsing, CPU only)
  8. prediction-error drift across a shedding cycle (error autocorrelation vs lag)

Standalone by design: imports ONLY dataset_piff (RunData) and model_piff
(PiffModel) as-is; the small helpers from train/eval are copied here so
concurrent edits to train_piff.py/eval_piff.py cannot break this tool.

Usage (GPU, items 1-6+8):
  python diagnose_piff.py --mode model --ckpt runs_piff/<run>/best.pt \
      --base-config conf_piff.yaml --members <run_dir> [<run_dir> ...] \
      [--outdir runs_piff/<run>/diagnostics]

Usage (CPU, item 7):
  python diagnose_piff.py --mode logs --pairs <run_dir>=<trainer_log> ... \
      [--combined-out <dir>]
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import load_conf, RunData, _f
from model_piff import PiffModel

HERE = Path(__file__).resolve().parent

# +/- k sigma central coverage of a Gaussian (erf(k/sqrt(2))), k = 0.25..3
K_LEVELS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
try:
    from scipy.special import erf
    NOMINAL_COV = erf(K_LEVELS / np.sqrt(2.0))
except ImportError:                                   # scipy is a dataset dep; belt+braces
    NOMINAL_COV = np.array([0.197413, 0.382925, 0.546746, 0.682689,
                            0.866386, 0.954500, 0.987581, 0.997300])

SDF_BIN_EDGES_D = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]   # obstacle distance in D units


# --------------------------------------------------------------------------- #
# helpers copied from train_piff / eval_piff (frozen against concurrent edits)
# --------------------------------------------------------------------------- #

def gaussian_nll(y, mu, var):
    return 0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)


@torch.no_grad()
def predict_frame(model, run, frame, device, gp_chunk):
    """Full-frame predictive mean/sigma on masked pixels (copy of eval_piff)."""
    x, y, mask, zeta, zeta_dot, g, lap = run.full_frame(frame)
    gpin = model.masked_gp_inputs(
        x[None].to(device), zeta[None].to(device), mask[None].to(device),
        zeta_dot=(zeta_dot[None].to(device) if model.use_zeta_dot else None),
        g=(g[None].to(device) if model.use_grad_feature else None),
        lap=(lap[None].to(device) if getattr(model, 'use_lap_feature', False) else None))
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
    return {'x': x.numpy(), 'truth': truth, 'mask': m,
            'y': truth[m].astype(np.float64), 'mu': mu.astype(np.float64),
            'sigma': np.sqrt(var).astype(np.float64),
            'zeta': float(zeta), 't': float(run.times[frame]),
            'Re': float(run.Re_snap[frame])}


def r2_nll(y, mu, sigma):
    ss = float(np.sum((y - y.mean()) ** 2))
    return (float(1.0 - np.sum((y - mu) ** 2) / max(ss, 1e-30)),
            float(np.mean(gaussian_nll(y, mu, sigma ** 2))))


def coverage(y, mu, sigma, k):
    return float(np.mean(np.abs(y - mu) <= k * sigma))


def excess_kurtosis(r):
    r = np.asarray(r, dtype=np.float64)
    c = r - r.mean()
    v = np.mean(c * c)
    return float(np.mean(c ** 4) / max(v * v, 1e-300) - 3.0)


def radial_spectrum(f2d):
    """Isotropic radially-binned power spectrum, float64. Invalid pixels must be
    zero-filled by the caller (windowing note in the figure caption)."""
    f = np.asarray(f2d, dtype=np.float64)
    F = np.fft.fft2(f)
    P = np.abs(F) ** 2 / f.size
    ky = np.fft.fftfreq(f.shape[0]) * f.shape[0]
    kx = np.fft.fftfreq(f.shape[1]) * f.shape[1]
    kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    kbin = np.rint(kr).astype(np.int64)
    kmax = min(f.shape) // 2
    spec = np.bincount(kbin.ravel(), weights=P.ravel(), minlength=kmax + 1)[:kmax + 1]
    return np.arange(kmax + 1), spec


def wake_box_mask(run, conf):
    """The sampler's wake window as a pixel mask (Ny, Nx)."""
    dc = conf['data']
    x = (np.arange(run.Nx) + 0.5) * run.dx
    y = (np.arange(run.Ny) + 0.5) * run.dy
    inx = (x[None, :] >= run.x_c + _f(dc['wake_x_lo_D']) * run.D) & \
          (x[None, :] <= run.x_c + _f(dc['wake_x_hi_D']) * run.D)
    iny = np.abs(y[:, None] - run.y_c) <= _f(dc['wake_y_half_D']) * run.D
    return inx & iny


# --------------------------------------------------------------------------- #
# per-member model diagnostics (items 1-6, 8)
# --------------------------------------------------------------------------- #

def diagnose_member(model, run, conf, device, gp_chunk, outdir):
    dc = conf['data']
    frames = run.frames_in(_f(dc['t_val_lo']), _f(dc['t_val_hi']) + 1e-9)
    if len(frames) == 0:
        return None
    name = run.name
    wake2d = wake_box_mask(run, conf)

    # accumulate flats + per-frame error fields (for drift) + spectra
    ys, mus, sgs, zts, gms = [], [], [], [], []
    wake_flat, sdf_flat = [], []
    err_fields, times = [], []
    spec_res_acc = spec_tru_acc = None
    for fi in frames:
        p = predict_frame(model, run, int(fi), device, gp_chunk)
        m = p['mask']
        ys.append(p['y']); mus.append(p['mu']); sgs.append(p['sigma'])
        zts.append(np.full(p['y'].size, p['zeta']))
        om = p['x'][0].astype(np.float64)                       # omega* channel
        gy, gx = np.gradient(om, run.dy, run.dx)
        gms.append(np.sqrt(gx * gx + gy * gy)[m])
        wake_flat.append(wake2d[m]); sdf_flat.append(run.sdf[m] / run.D)
        e2d = np.zeros(m.shape, dtype=np.float64)
        e2d[m] = p['y'] - p['mu']
        t2d = np.where(m, p['truth'].astype(np.float64), 0.0)
        err_fields.append(e2d); times.append(p['t'])
        k, sr = radial_spectrum(e2d)
        _, st = radial_spectrum(t2d)
        spec_res_acc = sr if spec_res_acc is None else spec_res_acc + sr
        spec_tru_acc = st if spec_tru_acc is None else spec_tru_acc + st
    y = np.concatenate(ys); mu = np.concatenate(mus); sg = np.concatenate(sgs)
    zt = np.concatenate(zts); gm = np.concatenate(gms)
    wk = np.concatenate(wake_flat); sd = np.concatenate(sdf_flat)
    nf = len(frames)
    spec_res = spec_res_acc / nf; spec_tru = spec_tru_acc / nf
    times = np.asarray(times)

    out = {'member': name, 'n_val_frames': int(nf), 'n_pixels': int(y.size)}

    # ---- 1. reliability ---------------------------------------------------- #
    out['reliability_emp'] = [coverage(y, mu, sg, k) for k in K_LEVELS]
    out['cov_1s'] = coverage(y, mu, sg, 1.0)
    out['cov_2s'] = coverage(y, mu, sg, 2.0)
    out['cov_3s'] = coverage(y, mu, sg, 3.0)

    # ---- 2. R2/NLL by zeta -------------------------------------------------- #
    zbins = []
    uz = np.unique(np.round(zt, 6))
    if uz.size <= 1:
        r2, nll = r2_nll(y, mu, sg)
        zbins.append({'zeta_lo': float(uz[0]), 'zeta_hi': float(uz[0]),
                      'r2': r2, 'nll': nll, 'n': int(y.size)})
    else:
        edges = np.unique(np.quantile(zt, np.linspace(0, 1, 7)))
        for i in range(len(edges) - 1):
            m = (zt >= edges[i]) & ((zt < edges[i + 1]) | (i == len(edges) - 2))
            if m.sum() < 100:
                continue
            r2, nll = r2_nll(y[m], mu[m], sg[m])
            zbins.append({'zeta_lo': float(edges[i]), 'zeta_hi': float(edges[i + 1]),
                          'r2': r2, 'nll': nll, 'n': int(m.sum())})
    out['zeta_bins'] = zbins
    out['r2_global'], out['nll_global'] = r2_nll(y, mu, sg)
    out['rmse_global'] = float(np.sqrt(np.mean((y - mu) ** 2)))

    # ---- 3. sigma vs |grad omega| ------------------------------------------ #
    qe = np.quantile(gm, np.linspace(0, 1, 11)); qe[-1] += 1e-12
    g_ctr, s_med, e_med = [], [], []
    for i in range(10):
        m = (gm >= qe[i]) & (gm < qe[i + 1])
        if m.sum() < 50:
            continue
        g_ctr.append(float(np.median(gm[m])))
        s_med.append(float(np.median(sg[m])))
        e_med.append(float(np.median(np.abs(y[m] - mu[m]))))
    out['grad_bins'] = {'grad': g_ctr, 'sigma_med': s_med, 'abserr_med': e_med}
    sub = np.random.default_rng(0).choice(y.size, size=min(200000, y.size), replace=False)
    try:
        from scipy.stats import spearmanr
        out['spearman_sigma_grad'] = float(spearmanr(sg[sub], gm[sub]).statistic)
        out['spearman_abserr_grad'] = float(spearmanr(np.abs(y - mu)[sub], gm[sub]).statistic)
    except Exception:
        out['spearman_sigma_grad'] = out['spearman_abserr_grad'] = float('nan')
    # dynamic range of sigma vs of |err| across grad deciles (shape mismatch metric)
    if s_med and e_med:
        out['sigma_dyn_range'] = float(max(s_med) / max(min(s_med), 1e-30))
        out['abserr_dyn_range'] = float(max(e_med) / max(min(e_med), 1e-30))

    # ---- 4. residual kurtosis + spectra ------------------------------------ #
    out['residual_kurtosis'] = excess_kurtosis(y - mu)
    out['spectrum_k'] = k.tolist()
    out['spectrum_res'] = spec_res.tolist()
    out['spectrum_truth'] = spec_tru.tolist()

    # ---- 5. coverage by SDF distance + wake/freestream --------------------- #
    sdf_rows = []
    for i in range(len(SDF_BIN_EDGES_D) - 1):
        m = (sd >= SDF_BIN_EDGES_D[i]) & (sd < SDF_BIN_EDGES_D[i + 1])
        if m.sum() < 100:
            continue
        sdf_rows.append({'d_lo_D': float(SDF_BIN_EDGES_D[i]),
                         'd_hi_D': float(min(SDF_BIN_EDGES_D[i + 1], 1e9)),
                         'n': int(m.sum()),
                         'cov1': coverage(y[m], mu[m], sg[m], 1.0),
                         'cov2': coverage(y[m], mu[m], sg[m], 2.0),
                         'r2': r2_nll(y[m], mu[m], sg[m])[0]})
    out['sdf_bins'] = sdf_rows
    for tag, m in (('wake', wk), ('freestream', ~wk)):
        out[f'cov1_{tag}'] = coverage(y[m], mu[m], sg[m], 1.0)
        out[f'cov2_{tag}'] = coverage(y[m], mu[m], sg[m], 2.0)
        out[f'r2_{tag}'] = r2_nll(y[m], mu[m], sg[m])[0]

    # ---- 6. sign accuracy --------------------------------------------------- #
    sgn = np.sign(mu) == np.sign(y)
    strong = np.abs(y) >= np.median(np.abs(y))
    out['sign_acc'] = float(np.mean(sgn))
    out['sign_acc_wake'] = float(np.mean(sgn[wk]))
    out['sign_acc_strong'] = float(np.mean(sgn[strong]))
    out['sign_acc_wake_strong'] = float(np.mean(sgn[wk & strong]))
    out['backscatter_frac_true'] = float(np.mean(y < 0.0))   # Pi*<0 = backscatter

    # ---- 8. error drift across a shedding cycle ---------------------------- #
    max_lag = min(14, nf - 1)
    corr = []
    for lag in range(1, max_lag + 1):
        cs = []
        for i in range(nf - lag):
            a = err_fields[i][run.valid]; b = err_fields[i + lag][run.valid]
            ca = a - a.mean(); cb = b - b.mean()
            den = np.sqrt(np.sum(ca * ca) * np.sum(cb * cb))
            if den > 0:
                cs.append(float(np.sum(ca * cb) / den))
        corr.append(float(np.mean(cs)) if cs else float('nan'))
    dt_med = float(np.median(np.diff(times))) if nf > 1 else float('nan')
    out['drift_lag_dt'] = dt_med
    out['drift_corr'] = corr
    return out


# --------------------------------------------------------------------------- #
# figures across members
# --------------------------------------------------------------------------- #

def fig_reliability(results, outdir):
    n = len(results)
    fig, axs = plt.subplots(1, n, figsize=(3.1 * n, 3.4), squeeze=False)
    for ax, r in zip(axs[0], results):
        ax.plot(NOMINAL_COV, r['reliability_emp'], 'o-', ms=3)
        ax.plot([0, 1], [0, 1], 'k:', lw=1)
        ax.set_title(f"{r['member']}\ncov@1s={r['cov_1s']:.3f}", fontsize=8)
        ax.set_xlabel('nominal'); ax.grid(alpha=0.3)
    axs[0][0].set_ylabel('empirical coverage')
    fig.suptitle('reliability per member (k sigma = 0.25..3)', fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / 'reliability_per_member.png', dpi=130); plt.close(fig)


def fig_zeta_binned(results, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        zc = [0.5 * (b['zeta_lo'] + b['zeta_hi']) for b in r['zeta_bins']]
        ax1.plot(zc, [b['r2'] for b in r['zeta_bins']], 'o-', ms=4, label=r['member'])
        ax2.plot(zc, [b['nll'] for b in r['zeta_bins']], 'o-', ms=4, label=r['member'])
    ax1.set_xlabel('zeta'); ax1.set_ylabel('R^2'); ax1.grid(alpha=0.3)
    ax2.set_xlabel('zeta'); ax2.set_ylabel('NLL'); ax2.grid(alpha=0.3)
    ax1.legend(fontsize=7); ax2.legend(fontsize=7)
    fig.suptitle('regime-resolved skill: R^2 and NLL binned by zeta, per member', fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / 'metrics_by_zeta.png', dpi=130); plt.close(fig)


def fig_sigma_grad(results, outdir):
    n = len(results)
    fig, axs = plt.subplots(1, n, figsize=(3.1 * n, 3.4), squeeze=False)
    for ax, r in zip(axs[0], results):
        gb = r['grad_bins']
        ax.loglog(gb['grad'], gb['sigma_med'], 'o-', ms=3, label='median sigma')
        ax.loglog(gb['grad'], gb['abserr_med'], 's--', ms=3, label='median |err|')
        ax.set_title(f"{r['member']}\nrho(sig,g)={r['spearman_sigma_grad']:.2f} "
                     f"rho(|e|,g)={r['spearman_abserr_grad']:.2f}", fontsize=7)
        ax.set_xlabel('|grad omega*|'); ax.grid(alpha=0.3, which='both')
    axs[0][0].set_ylabel('sigma / |err| (target units)')
    axs[0][0].legend(fontsize=6)
    fig.suptitle('sigma vs |grad omega| (grad-decile medians) — Arm F evidence', fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / 'sigma_vs_gradomega.png', dpi=130); plt.close(fig)


def fig_spectra(results, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        k = np.asarray(r['spectrum_k'][1:], dtype=float)
        ax1.loglog(k, r['spectrum_res'][1:], lw=1, label=f"{r['member']} res")
        ax2.loglog(k, np.asarray(r['spectrum_res'][1:]) /
                   np.maximum(np.asarray(r['spectrum_truth'][1:]), 1e-300),
                   lw=1, label=r['member'])
    ax1.set_xlabel('k'); ax1.set_ylabel('residual power'); ax1.grid(alpha=0.3, which='both')
    ax2.set_xlabel('k'); ax2.set_ylabel('residual / truth power'); ax2.grid(alpha=0.3, which='both')
    ax2.axhline(1.0, color='k', ls=':', lw=1)
    ax1.legend(fontsize=6); ax2.legend(fontsize=6)
    fig.suptitle('residual radial spectra per member (masked region zero-filled)', fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / 'residual_spectra.png', dpi=130); plt.close(fig)


def fig_sdf_coverage(results, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        d = [0.5 * (b['d_lo_D'] + min(b['d_hi_D'], 16.0)) for b in r['sdf_bins']]
        ax1.plot(d, [b['cov1'] for b in r['sdf_bins']], 'o-', ms=4, label=r['member'])
        ax2.plot(d, [b['r2'] for b in r['sdf_bins']], 'o-', ms=4, label=r['member'])
    ax1.axhline(0.6827, color='k', ls=':', lw=1)
    ax1.set_xlabel('obstacle distance (D)'); ax1.set_ylabel('coverage @ 1 sigma')
    ax2.set_xlabel('obstacle distance (D)'); ax2.set_ylabel('R^2')
    ax1.grid(alpha=0.3); ax2.grid(alpha=0.3); ax1.legend(fontsize=7); ax2.legend(fontsize=7)
    fig.suptitle('calibration and skill vs obstacle distance (SDF bins)', fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / 'coverage_by_sdf.png', dpi=130); plt.close(fig)


def fig_drift(results, outdir, t_shed=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in results:
        lags = np.arange(1, len(r['drift_corr']) + 1) * r['drift_lag_dt']
        ax.plot(lags, r['drift_corr'], 'o-', ms=4, label=r['member'])
    if t_shed:
        ax.axvline(t_shed, color='k', ls='--', lw=1, label=f'T_shed~{t_shed:g}')
        ax.axvline(0.5 * t_shed, color='k', ls=':', lw=1)
    ax.set_xlabel('lag (time units)'); ax.set_ylabel('error-field correlation')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title('prediction-error temporal coherence across the shedding cycle')
    fig.tight_layout()
    fig.savefig(outdir / 'drift_shedding_cycle.png', dpi=130); plt.close(fig)


def write_table(results, outdir):
    cols = ['member', 'n_val_frames', 'r2_global', 'rmse_global', 'nll_global',
            'cov_1s', 'cov_2s', 'cov_3s', 'cov1_wake', 'cov1_freestream',
            'r2_wake', 'r2_freestream', 'residual_kurtosis',
            'sign_acc', 'sign_acc_wake', 'sign_acc_strong', 'sign_acc_wake_strong',
            'backscatter_frac_true', 'spearman_sigma_grad', 'spearman_abserr_grad',
            'sigma_dyn_range', 'abserr_dyn_range']
    with open(outdir / 'per_member_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c, '') for c in cols])


# --------------------------------------------------------------------------- #
# item 7: zeta_ls trajectories from trainer logs (CPU)
# --------------------------------------------------------------------------- #

LOG_RE = re.compile(
    r'\[ep\s+(\d+)\].*?val NLL ([\d.eE+-]+) RMSE ([\d.eE+-]+) R2 ([\d.eE+-]+) '
    r'sigma ([\d.eE+-]+)\s+zeta_ls ([\d.eE+-]+)')


def parse_log(path):
    rows = []
    for line in Path(path).read_text(errors='replace').splitlines():
        m = LOG_RE.search(line)
        if m:
            rows.append([float(g) for g in m.groups()])
    return np.asarray(rows)   # (E, 6): ep, nll, rmse, r2, sigma, zeta_ls


def mode_logs(pairs, combined_out):
    all_curves = []
    for spec in pairs:
        run_dir, log_path = spec.split('=', 1)
        run_dir = Path(run_dir)
        rows = parse_log(log_path)
        if rows.size == 0:
            print(f"[logs] {log_path}: NO epoch lines parsed — skipped")
            continue
        dd = run_dir / 'diagnostics'
        dd.mkdir(parents=True, exist_ok=True)
        with open(dd / 'zeta_ls_trajectory.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'val_nll', 'val_rmse', 'val_r2', 'val_sigma', 'zeta_ls'])
            w.writerows(rows.tolist())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
        ax1.plot(rows[:, 0], rows[:, 5], lw=1.2)
        ax1.axhline(np.log(2.0), color='r', ls=':', lw=1, label='ln 2 = 0.6931 (init/frozen)')
        ax1.set_xlabel('epoch'); ax1.set_ylabel('zeta ARD lengthscale'); ax1.legend(fontsize=7)
        ax2.plot(rows[:, 0], rows[:, 3], lw=1.2)
        ax2.set_xlabel('epoch'); ax2.set_ylabel('val R^2')
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
        fig.suptitle(f"{run_dir.name}: zeta_ls identifiability ({Path(log_path).name})", fontsize=10)
        fig.tight_layout()
        fig.savefig(dd / 'zeta_ls_trajectory.png', dpi=130); plt.close(fig)
        all_curves.append((run_dir.name, rows))
        print(f"[logs] {run_dir.name}: {rows.shape[0]} epochs, zeta_ls "
              f"{rows[0, 5]:.4f} -> {rows[-1, 5]:.4f}, final val R2 {rows[-1, 3]:.4f}")
    if combined_out and all_curves:
        co = Path(combined_out); co.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, rows in all_curves:
            ax.plot(rows[:, 0], rows[:, 5], lw=1.2, label=name)
        ax.axhline(np.log(2.0), color='k', ls=':', lw=1)
        ax.set_xlabel('epoch'); ax.set_ylabel('zeta ARD lengthscale')
        ax.set_title('zeta_ls identifiability across all runs (0.6931 = frozen at init)')
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(co / 'zeta_ls_all_runs.png', dpi=130); plt.close(fig)
        print(f"[logs] combined figure -> {co / 'zeta_ls_all_runs.png'}")


# --------------------------------------------------------------------------- #

def mode_model(args):
    device = args.device
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = PiffModel(ckpt['conf']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    base = load_conf(args.base_config)
    gp_chunk = int(base['train']['gp_chunk'])
    outdir = Path(args.outdir or (Path(args.ckpt).parent / 'diagnostics'))
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for md in args.members:
        conf_m = yaml.safe_load(yaml.safe_dump(base))   # deep copy
        conf_m['data']['runs'] = [md]
        run = RunData(md, conf_m)
        print(f"[diag] {run.name}: grid {run.Ny}x{run.Nx}, "
              f"{len(run.frames_in(_f(conf_m['data']['t_val_lo']), _f(conf_m['data']['t_val_hi']) + 1e-9))} val frames")
        r = diagnose_member(model, run, conf_m, device, gp_chunk, outdir)
        if r is None:
            print(f"[diag] {run.name}: no val frames — skipped")
            continue
        results.append(r)
        print(f"[diag] {run.name}: R2 {r['r2_global']:.4f} NLL {r['nll_global']:.3f} "
              f"cov1 {r['cov_1s']:.3f} (wake {r['cov1_wake']:.3f} / free {r['cov1_freestream']:.3f}) "
              f"kurt {r['residual_kurtosis']:.1f} signacc {r['sign_acc']:.3f} "
              f"(wake {r['sign_acc_wake']:.3f})")

    fig_reliability(results, outdir)
    fig_zeta_binned(results, outdir)
    fig_sigma_grad(results, outdir)
    fig_spectra(results, outdir)
    fig_sdf_coverage(results, outdir)
    fig_drift(results, outdir, t_shed=args.t_shed)
    write_table(results, outdir)
    with open(outdir / 'diagnostics_summary.yaml', 'w') as f:
        yaml.safe_dump({'ckpt': str(Path(args.ckpt).resolve()),
                        'epoch': int(ckpt.get('epoch', -1)),
                        'zeta_ard_lengthscale': model.zeta_ard_lengthscale(),
                        'members': results}, f, sort_keys=False)
    print(f"[diag] package in {outdir}")


def main():
    ap = argparse.ArgumentParser(description='Pi_FF diagnostics (ORDER 2)')
    ap.add_argument('--mode', choices=['model', 'logs'], required=True)
    ap.add_argument('--ckpt')
    ap.add_argument('--base-config', default=str(HERE / 'conf_piff.yaml'))
    ap.add_argument('--members', nargs='*', default=[])
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--t-shed', type=float, default=2.992)
    ap.add_argument('--pairs', nargs='*', default=[], help='run_dir=trainer_log')
    ap.add_argument('--combined-out', default=None)
    args = ap.parse_args()
    if args.mode == 'logs':
        mode_logs(args.pairs, args.combined_out)
    else:
        if not args.ckpt or not args.members:
            raise SystemExit('--mode model needs --ckpt and --members')
        mode_model(args)


if __name__ == '__main__':
    main()
