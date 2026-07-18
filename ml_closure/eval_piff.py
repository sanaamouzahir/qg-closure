"""
A priori evaluation package (ML SPEC 01 S4) — run on the best checkpoint.

1. Pointwise val metrics (R2, RMSE, NLL) global and binned by zeta decile
   (degenerates to one bin when zeta is constant, e.g. FPC-const).
2. Calibration: reliability diagram (empirical coverage of +/-1,2,3 sigma vs
   nominal) and spread–skill plot (binned predictive sigma vs empirical |err|),
   global and per zeta-bin.
3. Field visualizations: 6 validation snapshots spanning the Re range —
   truth Pi*, predictive mean, predictive sigma, |error| (4-panel, seismic,
   aspect-preserving) + the Re_inlet(t) trace with snapshot times marked.
4. zeta ARD lengthscale in the summary (regime dependence not absorbed by FiLM).

Full frames are evaluated directly (CNN is convolutional + periodic; the GP is
pointwise) — crops are a training device only.

Usage:
    python eval_piff.py --ckpt runs_piff/<name>/best.pt [--config conf_piff.yaml] [--outdir ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import load_conf, build_runs, split_frames, _f
from member_naming import modulation_name, member_stamp
from model_piff import PiffModel
from train_piff import gaussian_nll, student_t_nll, t_central_halfwidth


def full_frame_slice(run):
    """Reproduce dataset_piff.RunData.full_frame's periodic crop indices so
    per-run 2D masks line up pixel-for-pixel with predict_frame outputs.
    (Same helper as wake_restricted_r2_check.full_frame_slice; defined here
    because that module imports eval_piff -- importing it back is circular.)"""
    size = max(run.Ny, run.Nx)
    iy = (run.Ny // 2 - size // 2 + np.arange(size)) % run.Ny
    ix = (run.Nx // 2 - size // 2 + np.arange(size)) % run.Nx
    return np.ix_(iy, ix)

HERE = Path(__file__).resolve().parent
NOMINAL = {1.0: 0.682689, 2.0: 0.954500, 3.0: 0.997300}


@torch.no_grad()
def predict_frame(model, run, frame, device, gp_chunk):
    """Full-frame predictive mean/sigma on masked pixels; returns 2D fields
    (NaN outside mask) + flat arrays."""
    x, y, mask, zeta, zeta_dot, g, lap = run.full_frame(frame)
    gpin = model.masked_gp_inputs(
        x[None].to(device), zeta[None].to(device), mask[None].to(device),
        zeta_dot=(zeta_dot[None].to(device) if model.use_zeta_dot else None),
        g=(g[None].to(device) if model.use_grad_feature else None),
        lap=(lap[None].to(device) if getattr(model, 'use_lap_feature', False) else None))
    gm = (g[None].to(device)[mask[None].to(device)]
          if getattr(model, 'noise_prior', 'none') == 'structural' else None)
    student = model.is_student_t()
    mus, vars_, scales, pvars, nvars = [], [], [], [], []
    for i0 in range(0, gpin.shape[0], gp_chunk):
        gm_c = gm[i0:i0 + gp_chunk] if gm is not None else None
        mu_p, var_p = model.predict_physical(gpin[i0:i0 + gp_chunk], g_masked=gm_c)
        mus.append(mu_p.cpu().numpy())
        vars_.append(var_p.cpu().numpy())
        if student:
            scales.append(model.student_scale(gm_c).cpu().numpy())
            # std/GP-space post-var and het_noise scale^2 (ratio diagnostic)
            pvars.append(model.gp(gpin[i0:i0 + gp_chunk]).variance.cpu().numpy())
            nvars.append(model.het_noise(gm_c).cpu().numpy())
    mu, var = np.concatenate(mus), np.concatenate(vars_)
    m = mask.numpy()
    truth = y.numpy()
    mu2 = np.full_like(truth, np.nan); sg2 = np.full_like(truth, np.nan)
    mu2[m] = mu; sg2[m] = np.sqrt(var)
    out = {'truth': truth, 'mask': m, 'mu2d': mu2, 'sigma2d': sg2,
           'y': truth[m], 'mu': mu, 'sigma': np.sqrt(var),
           'zeta': float(zeta), 't': float(run.times[frame]),
           'Re': float(run.Re_snap[frame])}
    if student:
        out['scale'] = np.concatenate(scales)      # physical Student-t scale
        out['post_var'] = np.concatenate(pvars)     # GP posterior var (std space)
        out['noise_var'] = np.concatenate(nvars)    # het_noise scale^2 (std space)
    return out


def metrics_block(y, mu, sigma, scale=None, nu=None):
    """Predictive metrics. Gaussian: NLL from (mu,sigma), coverage at
    +/-1,2,3 sigma. Student-t (scale + nu given): NLL under the t, coverage of
    the CENTRAL t-intervals at the same nominal probs (t-quantile half-widths,
    NOT the Gaussian z=1)."""
    var = sigma ** 2
    blk = {
        'n': int(y.size),
        'r2': float(1.0 - np.sum((y - mu) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)),
        'rmse': float(np.sqrt(np.mean((y - mu) ** 2))),
        'mean_sigma': float(np.mean(sigma)),
    }
    if scale is not None and nu is not None:
        blk['nll'] = float(np.mean(student_t_nll(y, mu, scale, nu)))
        blk['nu'] = float(nu)
        blk['mean_scale'] = float(np.mean(scale))
        # central t-interval coverage at the SAME nominal probs (apples-to-apples
        # reliability): half-width = t_nu.ppf(0.5 + p/2) in scale units
        blk['coverage'] = {
            f'{k:.0f}sigma': float(np.mean(
                np.abs(y - mu) <= t_central_halfwidth(NOMINAL[k], nu) * scale))
            for k in NOMINAL}
    else:
        blk['nll'] = float(np.mean(gaussian_nll(y, mu, var)))
        blk['coverage'] = {f'{k:.0f}sigma': float(np.mean(np.abs(y - mu) <= k * sigma))
                           for k in NOMINAL}
    return blk


def spread_skill(y, mu, sigma, n_bins=12):
    """Bin by predicted sigma; empirical skill = RMSE of errors in bin."""
    q = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    q[-1] += 1e-12
    ctr, emp = [], []
    for i in range(n_bins):
        m = (sigma >= q[i]) & (sigma < q[i + 1])
        if m.sum() < 10:
            continue
        ctr.append(float(np.mean(sigma[m])))
        emp.append(float(np.sqrt(np.mean((y[m] - mu[m]) ** 2))))
    return np.array(ctr), np.array(emp)


def imshow_field(ax, f2d, run, title, vmax=None):
    vmax = vmax if vmax is not None else np.nanmax(np.abs(f2d))
    im = ax.imshow(f2d, cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower',
                   extent=[0, run.Lx, 0, run.Ly], aspect='equal')  # never stretch
    ax.set_title(title, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser(description="A priori Pi_FF eval (spec S4)")
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=str(HERE / 'conf_piff.yaml'))
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--fig-dir', default=None,
                    help='figure directory (STANDARD tree passthrough); '
                         'default: same as --outdir')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(args.config)
    ec = conf['eval']
    device = args.device or conf['train']['device']
    gp_chunk = int(conf['train']['gp_chunk'])
    outdir = Path(args.outdir or (Path(args.ckpt).parent / 'eval'))
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.fig_dir) if args.fig_dir else outdir
    figdir.mkdir(parents=True, exist_ok=True)

    model = PiffModel(ckpt['conf']).to(device)   # conf as trained (film flag etc.)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # the data must carry whatever conditioning the CKPT was trained with,
    # regardless of the eval conf (ORDER-3 flags travel with the model)
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    conf['model']['use_lap_feature'] = getattr(model, 'use_lap_feature', False)
    # the data variant (sharp vs gaussian filter) travels with the ckpt too —
    # evaluating a gaussian-trained model on sharp targets is a different
    # experiment and must be asked for explicitly, never happen by default
    ck_var = ckpt['conf'].get('data', {}).get('variant')
    if ck_var and not conf['data'].get('variant'):
        conf['data']['variant'] = ck_var
    # tshed_smooth must equal the TRAINING value or zeta_dot is computed at a
    # different smoothing scale than the recorded zdot_sd normalizes (G4 LOW)
    ck_tsm = ckpt['conf'].get('zeta', {}).get('tshed_smooth', 2.992)
    ev_tsm = conf['zeta'].get('tshed_smooth')
    if ev_tsm is not None and abs(float(ev_tsm) - float(ck_tsm)) > 1e-12:
        raise ValueError(f"eval tshed_smooth {ev_tsm} != training {ck_tsm}")
    conf['zeta']['tshed_smooth'] = ck_tsm
    runs = build_runs(conf)
    frames = split_frames(runs, 'val', conf)
    print(f"[eval] {len(frames)} val frames from {[r.name for r in runs]}")

    preds = [predict_frame(model, runs[ri], fi, device, gp_chunk) for ri, fi in frames]
    # STANDARD rule 2 (Sanaa 2026-07-17): every figure title states the
    # member modulation function + the Reynolds number (per-frame Re(t) on
    # the field panels, pooled Re range on the pooled figures).
    member_names = [r.name for r in runs]
    pool_stamp = ('members: '
                  + ', '.join(f"{n} [{modulation_name(n, member_names).replace('_', ' ')}]"
                              for n in member_names)
                  + f" | Re {min(p['Re'] for p in preds):.0f}-"
                    f"{max(p['Re'] for p in preds):.0f}")
    y = np.concatenate([p['y'] for p in preds])
    mu = np.concatenate([p['mu'] for p in preds])
    sg = np.concatenate([p['sigma'] for p in preds])
    zt = np.concatenate([np.full(p['y'].size, p['zeta']) for p in preds])
    student = model.is_student_t()
    nu = model.student_nu() if student else None
    sc = np.concatenate([p['scale'] for p in preds]) if student else None

    def _block(sel):
        return metrics_block(y[sel], mu[sel], sg[sel],
                             scale=(sc[sel] if student else None), nu=nu)

    # ring-excluded population (Sanaa 2026-07-14): the Pi_J body-column ringing
    # is a proven numerical artefact (target == split Pi_J to 8e-12) that the
    # smooth GP cannot fit and the het-noise explains away; report metrics with
    # the near-body ring excluded ALONGSIDE global -- never instead of it.
    # Band = the standing wake_restricted convention (sdf > 1D).
    ring = np.concatenate([
        (runs[frames[i][0]].sdf[full_frame_slice(runs[frames[i][0]])]
         > 1.0 * runs[frames[i][0]].D)[p['mask']]
        for i, p in enumerate(preds)])

    # ---- 1. metrics: global + zeta deciles -------------------------------- #
    summary = {'ckpt': str(Path(args.ckpt).resolve()), 'epoch': int(ckpt['epoch']),
               'seed': int(ckpt['seed']),
               'likelihood': model.likelihood_type,
               'student_nu': nu,
               'zeta_ard_lengthscale': model.zeta_ard_lengthscale(),
               'ard_lengthscales': model.ard_lengthscales(),
               'global': _block(slice(None)),
               'ring_excluded_sdf_gt_1D': _block(ring), 'zeta_bins': []}
    if student:
        # Review Finding 1: verify the t-NLL's noise-only scale omits a
        # negligible GP posterior variance (accept criterion #2 gate). Ratio in
        # consistent std/GP space; nll_with_postvar folds post-var into the
        # t-scale in PHYSICAL units (scale_total = y_sd*sqrt(noise_var+post_var)).
        pv = np.concatenate([p['post_var'] for p in preds])
        nv = np.concatenate([p['noise_var'] for p in preds])
        summary['post_var_over_noise'] = float(np.mean(pv) / max(np.mean(nv), 1e-30))
        scale_total = float(model.y_sd) * np.sqrt(nv + pv)
        summary['global']['nll_with_postvar'] = float(
            np.mean(student_t_nll(y, mu, scale_total, nu)))
    edges = np.unique(np.quantile(zt, np.linspace(0, 1, int(ec['n_zeta_bins']) + 1)))
    if len(edges) < 2:
        edges = np.array([zt[0] - 0.5, zt[0] + 0.5])   # constant zeta (FPC-const)
    for i in range(len(edges) - 1):
        m = (zt >= edges[i]) & (zt <= edges[i + 1] if i == len(edges) - 2 else zt < edges[i + 1])
        if m.sum() < 10:
            continue
        blk = _block(m)
        blk['zeta_range'] = [float(edges[i]), float(edges[i + 1])]
        summary['zeta_bins'].append(blk)

    # ---- 2. calibration ---------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ks = np.array(sorted(NOMINAL))
    nom = np.array([NOMINAL[k] for k in ks])
    # half-width in scale/sigma units per nominal prob: Gaussian z=k, or the
    # central Student-t quantile at nu (the correct yardstick for heavy tails)
    hw = ({k: t_central_halfwidth(NOMINAL[k], nu) for k in ks} if student
          else {k: float(k) for k in ks})
    disp = sc if student else sg           # scale for t, sigma for gaussian
    emp = np.array([np.mean(np.abs(y - mu) <= hw[k] * disp) for k in ks])
    ax1.plot(nom, emp, 'o-', label='global')
    for i in range(len(edges) - 1):
        m = (zt >= edges[i]) & (zt < edges[i + 1] + (1e-12 if i == len(edges) - 2 else 0))
        if m.sum() < 100:
            continue
        e = [np.mean(np.abs(y[m] - mu[m]) <= hw[k] * disp[m]) for k in ks]
        ax1.plot(nom, e, '.--', alpha=0.5, label=f'zeta[{edges[i]:.2f},{edges[i+1]:.2f}]')
    ax1.plot([0, 1], [0, 1], 'k:', lw=1)
    ax1.set_xlabel('nominal coverage'); ax1.set_ylabel('empirical coverage')
    ax1.set_title(f"reliability ({'Student-t central' if student else '+/-1,2,3 sigma'})")
    ax1.legend(fontsize=6); ax1.grid(alpha=0.3)
    ctr, es = spread_skill(y, mu, sg)
    ax2.plot(ctr, es, 'o-')
    lim = [0, max(ctr.max(), es.max()) * 1.05]
    ax2.plot(lim, lim, 'k:', lw=1)
    ax2.set_xlabel('binned predictive sigma'); ax2.set_ylabel('empirical RMSE in bin')
    ax2.set_title('spread-skill'); ax2.grid(alpha=0.3)
    fig.suptitle(pool_stamp, fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / 'calibration.png', dpi=130); plt.close(fig)

    # ---- 3. field figures: per MEMBER, spanning that member's Re range ---- #
    # (Sanaa 2026-07-17: more fields per member; the old pooled-Re pick gave
    # uneven 1-2 panels per member)
    n_per = int(ec.get('n_field_snapshots_per_member', 5))
    by_run: dict = {}
    for i in range(len(preds)):
        by_run.setdefault(frames[i][0], []).append(i)
    sel = []
    for ri in sorted(by_run):
        o = sorted(by_run[ri], key=lambda i: preds[i]['Re'])
        picks = np.linspace(0, len(o) - 1,
                            min(n_per, len(o))).astype(int)
        sel.extend((o[q], j) for j, q in enumerate(dict.fromkeys(picks)))
    for idx, j in sel:
        p = preds[idx]
        run = runs[frames[idx][0]]
        err = np.where(p['mask'], np.abs(p['truth'] - np.nan_to_num(p['mu2d'])), np.nan)
        tr = np.where(p['mask'], p['truth'], np.nan)
        fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
        # color scale from the ring-excluded truth (sdf > 1D): the body-column
        # ringing is ~83x background and otherwise sets vmax, washing out the
        # wake the model is actually scored on (ring pixels saturate instead)
        ring2d = run.sdf[full_frame_slice(run)] > 1.0 * run.D
        vmax = np.nanmax(np.abs(np.where(ring2d, tr, np.nan)))
        if not np.isfinite(vmax):
            vmax = np.nanmax(np.abs(tr))
        # Sanaa convention 2026-07-16: ALL panels on ONE color scale -- the
        # TRUTH's (ring-excluded max) -- and every title carries t + Re(t);
        # STANDARD 2026-07-17: plus the member modulation function.
        mod = modulation_name(run.name, member_names).replace('_', ' ')
        stamp = f"{run.name} [{mod}]  t={p['t']:.2f} Re(t)={p['Re']:.0f}"
        for ax, f2d, ttl in zip(
                axs, [tr, p['mu2d'], p['sigma2d'], err],
                [f"truth Pi*  {stamp}", f"predictive mean  {stamp}",
                 f"predictive sigma  {stamp}", f"|error|  {stamp}"]):
            im = imshow_field(ax, f2d, run, ttl, vmax=vmax)
        fig.colorbar(im, ax=list(axs), fraction=0.02, pad=0.02,
                     label='truth color scale (shared)')
        # STANDARD 2026-07-17: every plot in a per-member modulation subdir
        mdir = figdir / modulation_name(run.name, member_names)
        mdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(mdir / f'field_{j}_t{p["t"]:.2f}.png', dpi=130)
        plt.close(fig)

    # Re_inlet(t) trace with snapshot markers
    fig, ax = plt.subplots(figsize=(9, 3))
    for r in runs:
        tab = np.load(r.run_dir / r.man['files']['u_table'])
        mod = modulation_name(r.name, member_names).replace('_', ' ')
        ax.plot(tab['t'], tab['Re'], lw=0.8, label=f"{r.name} [{mod}]")
    for p in show:
        ax.axvline(p['t'], color='k', ls='--', lw=0.7)
    ax.set_title(f'inlet Reynolds number trace per member (evaluated '
                 f'snapshots marked) | Re {min(p["Re"] for p in preds):.0f}-'
                 f'{max(p["Re"] for p in preds):.0f}', fontsize=9)
    ax.set_xlabel('t'); ax.set_ylabel('Re_inlet'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figdir / 'Re_trace.png', dpi=130); plt.close(fig)

    # stage-1 conformal (Sanaa approval 2026-07-14): every eval summary
    # surfaces the deployable conformal calibration and its held-out
    # coverage. Computed by sigma_conformal_prototype.py on the
    # recal-standardized pipeline (honesty test half) -- referenced here,
    # NOT recomputed (this eval's sigma is the raw predictive one).
    cpath = Path(args.ckpt).parent / 'conformal_calibration.yaml'
    if cpath.exists():
        cc = yaml.safe_load(cpath.read_text())
        summary['conformal'] = {'source': str(cpath.resolve()),
                                'coverage_test': cc.get('coverage_test')}
        print("[eval] conformal calibration surfaced from", cpath)

    with open(outdir / 'summary.yaml', 'w') as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(json.dumps(summary['global'], indent=2))
    print(f"[eval] zeta ARD lengthscale = {summary['zeta_ard_lengthscale']:.4f}")
    print(f"[eval] package in {outdir}")


if __name__ == '__main__':
    main()
