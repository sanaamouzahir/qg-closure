"""
Two-parameter post-hoc sigma recalibration for STRUCTURAL-noise models
(Sanaa GO 2026-07-14 chat: "I like your 2 parameter recalibration").

The 2026-07-14 finals diagnostics showed the structural prior
sigma^2 = softplus(a) + softplus(b) * g^2/g2_scale works qualitatively
(Spearman sigma-vs-|grad| 0.75-0.92) but is FLOOR-DOMINATED: the softplus(a)
floor is too big for the quiet freestream (cov1 0.999 there vs 0.84
near-body) while the b-term under-responds ~3x in the top gradient decile.
One global scalar (calibrate_piff.py) cannot fix both — it shrinks floor and
tail together. This fits TWO scalars on the noise decomposition instead:

    var'(x) = var_gp_posterior(x)
            + clamp( s_a * softplus(a) + s_b * softplus(b) * g^2/g2_scale,
                     1e-3, 10 )                      [standardized units]

(the clamp band is the VALIDATED arm-F band — kept so the recalibrated noise
can never leave it). The GP mean and posterior variance are untouched; only
the likelihood-noise split is rescaled. NLL-optimal (s_a, s_b) found by Adam
on log-parameters (no closed form once two terms + clamp are present).

Honesty split (same as calibrate_piff.py): fit on the FIRST half of the val
time-window, report on the SECOND half — headline numbers stay out-of-sample.

Outputs:
  <ckpt dir>/recalibration_structural.yaml   sidecar (s_a, s_b, before/after
      NLL + coverage on both halves + per-gradient-decile sigma medians).
      No ckpt mutation; consumers apply the two scalars at inference.
  --fig-dir (optional): the two before/after figures (reliability; sigma vs
      |grad| deciles), full-English filenames per CONVENTION.md 5a.

Usage (GPU, via piff_tool_job.sh):
  python recalibrate_structural_sigma.py --ckpt runs_piff/piff_fpc_gjs/best.pt \
      --config conf_piff_fpc_gjs.yaml --model-name cylinder_jacobian_only_structural_sigma \
      --fig-dir pngs/jacobian_structural_sigma_final_models_eval/two_parameter_sigma_recalibration
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_piff import load_conf, build_runs, split_frames, _f
from model_piff import PiffModel

HERE = Path(__file__).resolve().parent
CLAMP_LO, CLAMP_HI = 1.0e-3, 10.0   # validated arm-F variance band (model_piff.het_noise)


@torch.no_grad()
def collect_decomposed(model, runs, frames, device, gp_chunk):
    """Per-pixel standardized residual + variance decomposition over frames.
    Returns float64 numpy arrays: r (y_std - mu_std), v_gp (GP posterior
    variance), s_feat (g^2/g2_scale), t (frame time per pixel)."""
    assert getattr(model, 'noise_prior', 'none') == 'structural', \
        "this recalibration is only defined for structural-noise models"
    rs, vgs, sfs, ts = [], [], [], []
    y_mu = float(model.y_mu); y_sd = float(model.y_sd)
    for ri, fi in frames:
        run = runs[ri]
        x, y, mask, zeta, zeta_dot, g = run.full_frame(fi)
        gpin = model.masked_gp_inputs(
            x[None].to(device), zeta[None].to(device), mask[None].to(device),
            zeta_dot=(zeta_dot[None].to(device) if model.use_zeta_dot else None),
            g=(g[None].to(device) if model.use_grad_feature else None))
        gm = g[None].to(device)[mask[None].to(device)]
        y_std = (y.numpy()[mask.numpy()].astype(np.float64) - y_mu) / y_sd
        mus, vgs_f, sfs_f = [], [], []
        for i0 in range(0, gpin.shape[0], gp_chunk):
            post = model.gp(gpin[i0:i0 + gp_chunk])       # latent posterior
            mus.append(post.mean.double().cpu().numpy())
            vgs_f.append(post.variance.double().cpu().numpy())
            sf = (gm[i0:i0 + gp_chunk] ** 2) / model.g2_scale.to(gm.dtype)
            sfs_f.append(sf.double().cpu().numpy())
        rs.append(y_std - np.concatenate(mus))
        vgs.append(np.concatenate(vgs_f))
        sfs.append(np.concatenate(sfs_f))
        ts.append(np.full(y_std.shape, float(run.times[fi])))
    return (np.concatenate(rs), np.concatenate(vgs),
            np.concatenate(sfs), np.concatenate(ts))


def total_var(vgp, sfeat, spa, spb, sa, sb):
    """Recalibrated standardized predictive variance (numpy or torch)."""
    noise = sa * spa + sb * spb * sfeat
    clip = noise.clamp if torch.is_tensor(noise) else lambda a, b: np.clip(noise, a, b)
    return vgp + clip(CLAMP_LO, CLAMP_HI)


def nll(r, v):
    return 0.5 * (np.log(2 * np.pi * v) + r ** 2 / v)


def coverage(r, v, k):
    return float(np.mean(np.abs(r) <= k * np.sqrt(v)))


def fit_two_scalars(r, vgp, sfeat, spa, spb, device, iters=600, lr=0.05, seed=0):
    """Adam on (log s_a, log s_b); full-batch NLL. Data already float64."""
    g = torch.Generator().manual_seed(seed)   # subsample determinism
    n = r.shape[0]
    idx = torch.randperm(n, generator=g)[:min(n, 5_000_000)].numpy()
    rt = torch.as_tensor(r[idx], device=device)
    vt = torch.as_tensor(vgp[idx], device=device)
    st = torch.as_tensor(sfeat[idx], device=device)
    log_s = torch.zeros(2, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([log_s], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        sa, sb = torch.exp(log_s[0]), torch.exp(log_s[1])
        v = total_var(vt, st, spa, spb, sa, sb)
        loss = (0.5 * (torch.log(2 * np.pi * v) + rt ** 2 / v)).mean()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == iters - 1:
            print(f'[recal] iter {it:4d}  fit-NLL(std) {loss.item():.6f} '
                  f' s_a {float(torch.exp(log_s[0])):.4g}  s_b {float(torch.exp(log_s[1])):.4g}')
    return float(torch.exp(log_s[0])), float(torch.exp(log_s[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=str(HERE / 'conf_piff.yaml'))
    ap.add_argument('--model-name', required=True,
                    help='plain-English model name for figure filenames (CONVENTION.md rule 2)')
    ap.add_argument('--fig-dir', default=None)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(args.config)
    device = args.device or conf['train']['device']
    gp_chunk = int(conf['train']['gp_chunk'])

    model = PiffModel(ckpt['conf']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    conf['zeta']['tshed_smooth'] = ckpt['conf'].get('zeta', {}).get('tshed_smooth', 2.992)

    spa = float(torch.nn.functional.softplus(model.noise_a))
    spb = float(torch.nn.functional.softplus(model.noise_b))
    y_sd = float(model.y_sd)

    runs = build_runs(conf)
    frames = split_frames(runs, 'val', conf)
    print(f'[recal] {len(frames)} val frames from {[r.name for r in runs]}; '
          f'sp(a)={spa:.4g} sp(b)={spb:.4g} (standardized var units)')

    r, vgp, sfeat, t = collect_decomposed(model, runs, frames, device, gp_chunk)

    t_lo, t_hi = _f(conf['data']['t_val_lo']), _f(conf['data']['t_val_hi'])
    t_mid = 0.5 * (t_lo + t_hi)
    fit_m, test_m = t < t_mid, t >= t_mid
    if not fit_m.any() or not test_m.any():
        raise SystemExit(f'[recal] empty honesty half (fit {int(fit_m.sum())} px, '
                         f'test {int(test_m.sum())} px) — val window does not '
                         f'straddle t_mid={t_mid:.2f}; refusing to fit')
    print(f'[recal] honesty split at t={t_mid:.2f}: fit {int(fit_m.sum())} px, '
          f'test {int(test_m.sum())} px')

    s_a, s_b = fit_two_scalars(r[fit_m], vgp[fit_m], sfeat[fit_m], spa, spb, device)

    out = {'s_a': s_a, 's_b': s_b,
           'softplus_a': spa, 'softplus_b': spb, 'y_sd': y_sd,
           'formula': "var'_std = var_gp + clamp(s_a*softplus(a) + "
                      "s_b*softplus(b)*g^2/g2_scale, 1.0e-3, 10.0); "
                      "sigma_physical = sqrt(var'_std)*y_sd",
           'fit_window_t': [float(t_lo), float(t_mid)],
           'test_window_t': [float(t_mid), float(t_hi)]}
    for name, msk in (('fit_half', fit_m), ('test_half', test_m)):
        v0 = total_var(vgp[msk], sfeat[msk], spa, spb, 1.0, 1.0)
        v1 = total_var(vgp[msk], sfeat[msk], spa, spb, s_a, s_b)
        out[name] = {
            'nll_before': float(np.mean(nll(r[msk], v0)) + np.log(y_sd)),
            'nll_after':  float(np.mean(nll(r[msk], v1)) + np.log(y_sd)),
            'cov1_before': coverage(r[msk], v0, 1.0), 'cov1_after': coverage(r[msk], v1, 1.0),
            'cov2_before': coverage(r[msk], v0, 2.0), 'cov2_after': coverage(r[msk], v1, 2.0),
            'cov3_before': coverage(r[msk], v0, 3.0), 'cov3_after': coverage(r[msk], v1, 3.0),
        }
        print(f"[recal] {name}: NLL {out[name]['nll_before']:.4f} -> "
              f"{out[name]['nll_after']:.4f}  cov1 {out[name]['cov1_before']:.4f} -> "
              f"{out[name]['cov1_after']:.4f}")

    # per-gradient-decile medians on the TEST half (the arm-F staircase, after)
    g_test = sfeat[test_m]
    edges = np.quantile(g_test, np.linspace(0, 1, 11))
    dec = {'grad_feature_hi': [], 'sigma_med_before': [], 'sigma_med_after': [],
           'abserr_med': []}
    v0 = total_var(vgp[test_m], sfeat[test_m], spa, spb, 1.0, 1.0)
    v1 = total_var(vgp[test_m], sfeat[test_m], spa, spb, s_a, s_b)
    for i in range(10):
        m = (g_test >= edges[i]) & (g_test <= edges[i + 1] if i == 9 else g_test < edges[i + 1])
        if not m.any():   # zero-inflated g -> duplicate quantile edges
            continue
        dec['grad_feature_hi'].append(float(edges[i + 1]))
        dec['sigma_med_before'].append(float(np.median(np.sqrt(v0[m])) * y_sd))
        dec['sigma_med_after'].append(float(np.median(np.sqrt(v1[m])) * y_sd))
        dec['abserr_med'].append(float(np.median(np.abs(r[test_m][m])) * y_sd))
    out['test_half_gradient_deciles'] = dec

    yml = Path(args.ckpt).parent / 'recalibration_structural.yaml'
    yml.write_text(yaml.safe_dump(out, sort_keys=False))
    print(f'[recal] wrote {yml}')

    if args.fig_dir:
        fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
        mn = args.model_name
        ks = np.linspace(0.25, 3.0, 12)
        from scipy.special import erf
        nominal = erf(ks / np.sqrt(2.0))
        rt_, v0_, v1_ = r[test_m], v0, v1
        emp0 = [np.mean(np.abs(rt_) <= k * np.sqrt(v0_)) for k in ks]
        emp1 = [np.mean(np.abs(rt_) <= k * np.sqrt(v1_)) for k in ks]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(nominal, nominal, 'k--', label='perfect')
        ax.plot(nominal, emp0, 'o-', label='before')
        ax.plot(nominal, emp1, 's-', label=f'after (s_a={s_a:.3g}, s_b={s_b:.3g})')
        ax.set_xlabel('nominal coverage'); ax.set_ylabel('empirical coverage')
        ax.set_title('reliability, held-out half of val window'); ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f'reliability_before_after_two_parameter_recalibration_{mn}.png', dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        xs = np.arange(1, len(dec['sigma_med_before']) + 1)
        ax.semilogy(xs, dec['sigma_med_before'], 'o-', label='median sigma BEFORE')
        ax.semilogy(xs, dec['sigma_med_after'], 's-', label='median sigma AFTER')
        ax.semilogy(xs, dec['abserr_med'], 'k^--', label='median |error| (target)')
        ax.set_xlabel('|grad omega| decile (held-out half)')
        ax.set_ylabel('physical units')
        ax.set_title('sigma staircase vs error staircase, before/after')
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / f'sigma_vs_sharpness_deciles_before_after_two_parameter_recalibration_{mn}.png', dpi=130)
        plt.close(fig)
        print(f'[recal] figures in {fig_dir}')


if __name__ == '__main__':
    main()
