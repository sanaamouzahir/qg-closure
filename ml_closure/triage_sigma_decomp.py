"""
triage_sigma_decomp.py — P4 of the 2026-07-13 field-plot triage (standalone).

Question: is the predictive uncertainty dominated by the homoscedastic
likelihood noise (collapse-to-flat-sigma), rather than the input-dependent
posterior GP variance? Decomposition, per ensemble member, over its val frames:

    total predictive variance = posterior GP variance (input-dependent)
                              + likelihood noise sigma_n^2 (one global constant)

both reported in PHYSICAL target units (x y_sd^2). Outputs per geometry:
  1. bar chart: sigma_n^2 vs mean posterior GP variance per member (log scale)
  2. paper-quality "before" figure: median predicted sigma vs median actual
     |error| across vorticity-gradient deciles (flat sigma = the finding)
  3. one consolidated .npz + printed ratios.

Reads the FROZEN modules only. GPU (model forwards).

Usage (via piff_tool_job.sh):
  python triage_sigma_decomp.py --ckpt runs_piff/prod_ext150/best.pt \
      --config conf_piff.yaml --label fpc_prod_ext150 \
      --members <5 member dirs> --outdir triage_plot_20260713/sigma_decomp
"""

import argparse
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
MEMBER_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377']


@torch.no_grad()
def frame_stats(model, run, frame, device, gp_chunk):
    """Posterior GP variance (no likelihood), total predictive sigma, |error|,
    and |grad omega*| — physical target units, masked pixels."""
    x, y, mask, zeta, zeta_dot, g, lap = run.full_frame(frame)
    gpin = model.masked_gp_inputs(
        x[None].to(device), zeta[None].to(device), mask[None].to(device),
        zeta_dot=(zeta_dot[None].to(device) if model.use_zeta_dot else None),
        g=(g[None].to(device) if model.use_grad_feature else None),
        lap=(lap[None].to(device) if getattr(model, 'use_lap_feature', False) else None))
    y_sd2 = float(model.y_sd) ** 2
    mus, pvars, tvars = [], [], []
    for i0 in range(0, gpin.shape[0], gp_chunk):
        post = model.gp(gpin[i0:i0 + gp_chunk])
        pvars.append((post.variance * y_sd2).cpu().numpy())
        mu_p, var_p = model.predict_physical(gpin[i0:i0 + gp_chunk])
        mus.append(mu_p.cpu().numpy())
        tvars.append(var_p.cpu().numpy())
    m = mask.numpy()
    om = x.numpy()[0].astype(np.float64)
    gy, gx = np.gradient(om, run.dy, run.dx)
    gm = np.sqrt(gx * gx + gy * gy)[m]
    yv = y.numpy()[m].astype(np.float64)
    return (np.concatenate(pvars), np.concatenate(tvars),
            np.abs(yv - np.concatenate(mus)), gm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=str(HERE / 'conf_piff.yaml'))
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-frames', type=int, default=24,
                    help='evenly-spaced cap on val frames per member')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = PiffModel(ckpt['conf']).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    base = load_conf(args.config)
    base.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    base['zeta']['tshed_smooth'] = ckpt['conf'].get('zeta', {}).get('tshed_smooth', 2.992)
    gp_chunk = int(base['train']['gp_chunk'])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    y_sd2 = float(model.y_sd) ** 2
    noise_phys = float(model.likelihood.noise) * y_sd2
    print(f'[{args.label}] likelihood noise sigma_n^2 (physical) = {noise_phys:.4e} '
          f'(GP-space {float(model.likelihood.noise):.4e}, y_sd^2 {y_sd2:.4e})')

    names, post_means, ratios, deciles = [], [], [], {}
    for md in args.members:
        conf = yaml.safe_load(yaml.safe_dump(base))
        conf['data']['runs'] = [md]
        try:
            run = RunData(md, conf)
        except Exception as e:
            print(f'[{args.label}] SKIP {Path(md).name}: {type(e).__name__}: {e}')
            continue
        dc = conf['data']
        fr = run.frames_in(_f(dc['t_val_lo']), _f(dc['t_val_hi']) + 1e-9)
        if len(fr) == 0:
            print(f'[{args.label}] SKIP {run.name}: no val frames')
            continue
        if len(fr) > args.max_frames:
            fr = fr[np.linspace(0, len(fr) - 1, args.max_frames).astype(int)]
        pv, tv, ae, gm = [], [], [], []
        for fi in fr:
            a, b, c, d = frame_stats(model, run, int(fi), args.device, gp_chunk)
            pv.append(a); tv.append(b); ae.append(c); gm.append(d)
        pv = np.concatenate(pv); tv = np.concatenate(tv)
        ae = np.concatenate(ae); gm = np.concatenate(gm)
        mp = float(pv.mean())
        names.append(run.name)
        post_means.append(mp)
        ratios.append(noise_phys / max(mp, 1e-300))
        # gradient-decile medians of total predictive sigma and actual |error|
        qe = np.quantile(gm, np.linspace(0, 1, 11)); qe[-1] += 1e-12
        g_ctr, s_med, e_med = [], [], []
        for i in range(10):
            sel = (gm >= qe[i]) & (gm < qe[i + 1])
            if sel.sum() < 50:
                continue
            g_ctr.append(float(np.median(gm[sel])))
            s_med.append(float(np.median(np.sqrt(tv[sel]))))
            e_med.append(float(np.median(ae[sel])))
        deciles[run.name] = (g_ctr, s_med, e_med)
        print(f'[{args.label}] {run.name}: {len(fr)} frames, '
              f'mean posterior GP var {mp:.4e}, sigma_n^2 {noise_phys:.4e}, '
              f'ratio noise/posterior = {ratios[-1]:.1f}x')

    # ---- figure 1: bar chart per member (log scale) ------------------------- #
    xpos = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.9 * len(names) + 3, 4.6))
    ax.bar(xpos - w / 2, [noise_phys] * len(names), w,
           color='#EE6677', label='likelihood noise sigma_n^2 (one global constant)')
    ax.bar(xpos + w / 2, post_means, w,
           color='#4477AA', label='mean posterior GP variance (input-dependent part)')
    for i, r in enumerate(ratios):
        ytop = max(noise_phys, post_means[i]) * 1.15
        lbl = f'{r:.0f}x' if r >= 1 else f'{r:.2g}x'
        ax.text(xpos[i], ytop, lbl, ha='center', fontsize=8)
    ax.set_yscale('log')
    ax.set_xticks(xpos); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('variance, physical target units')
    ax.set_title(f'{args.label}: constant likelihood noise vs input-dependent '
                 'posterior GP variance, per member\n(number above bars = noise / posterior-GP-variance ratio)',
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fp1 = outdir / f'sigma_decomposition_{args.label}.png'
    fig.savefig(fp1, dpi=150); plt.close(fig)
    print(f'[{args.label}] bar chart -> {fp1}')

    # ---- figure 2: flat-sigma "before" figure (paper quality) --------------- #
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for (nm, (g, s, e)), col in zip(deciles.items(), MEMBER_COLORS):
        ax.loglog(g, s, 'o-', color=col, ms=4, lw=1.6, label=nm)
        ax.loglog(g, e, 's--', color=col, ms=4, lw=1.2, alpha=0.65)
    ax.set_xlabel('vorticity gradient magnitude (decile medians, normalized units)')
    ax.set_ylabel('predicted uncertainty (solid) and actual error (dashed)\n'
                  '(median per decile, physical target units)')
    ax.set_title('Before conditioning: predicted uncertainty is nearly flat across\n'
                 'flow regimes while the actual error grows with the local gradient',
                 fontsize=11)
    ax.legend(fontsize=8, title='ensemble member', title_fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fp2 = outdir / f'flat_sigma_before_{args.label}.png'
    fig.savefig(fp2, dpi=150); plt.close(fig)
    print(f'[{args.label}] before-figure -> {fp2}')

    np.savez(outdir / f'sigma_decomp_{args.label}.npz',
             members=np.array(names), noise_phys=noise_phys,
             posterior_var_mean=np.array(post_means), ratio=np.array(ratios),
             **{f'decile_{nm}': np.array(v, dtype=object) for nm, v in deciles.items()})
    print(f'[{args.label}] ratios noise/posterior: '
          + ', '.join(f'{n} {r:.0f}x' for n, r in zip(names, ratios)))


if __name__ == '__main__':
    main()
