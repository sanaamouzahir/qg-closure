"""
replot_eval_fields.py — P1 of the 2026-07-13 field-plot triage (standalone,
reusable). The original eval field panels use a LINEAR color scale to the frame
max of a heavy-tailed field, so the wake reads as "no flow". This regenerates
them as 6-panel figures with a SYMLOG color norm:

    [ omega_bar | truth Pi | predicted Pi | predicted sigma | |error| | rel. error ]

symlog linthresh = 99th percentile of |truth Pi| over the frame's valid pixels;
one shared norm for truth/prediction/error, sigma on the same linthresh.
Sixth panel (Sanaa FINAL rule, 2026-07-14 afternoon — CONVENTION.md 5b):
SIGNED relative error (pred - truth) / (|truth| + 0.01*max|truth|), seismic,
colorbar FIXED to [-1, 1]. (Supersedes the same-day threshold rule.)

--per-member N (Sanaa 2026-07-14): instead of 6 frames pooled across members,
write N frames PER ensemble member, evenly spread over that member's val
window, into <outdir>/<member>/. If recalibration_structural.yaml exists next
to the ckpt, the sigma panel applies the (s_a, s_b) two-parameter
recalibration (title says so); mean prediction is unchanged by recal.
cmap seismic, aspect-preserving, origin/orientation identical to eval_piff.
Writes field5_*.png NEXT TO the existing field_*.png (never deletes them).
Frame selection reproduces eval_piff exactly (val frames sorted by Re, 6-point
linspace), so field5_j corresponds to field_j.

Reads the FROZEN modules (dataset_piff/model_piff) — no production file edited.

Usage (GPU, via piff_tool_job.sh):
  python replot_eval_fields.py --ckpt runs_piff/prod_ext150/best.pt --config conf_piff.yaml
  python replot_eval_fields.py --ckpt runs_piff/cape_base_100ep/best.pt --config conf_piff_cape.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from dataset_piff import load_conf, build_runs, split_frames
from model_piff import PiffModel
from piff_model_loader import load_piff_model  # two-band blend (Sanaa GO 2026-07-20): plain ckpt -> identical PiffModel path
from eval_piff import full_frame_slice

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def predict_frame_full(model, run, frame, device, gp_chunk, recal=None):
    """Full-frame prediction, returning the omega* channel too (superset of
    eval_piff.predict_frame — same math, conditioning flags honored).
    recal = {'s_a','s_b'} applies the two-parameter noise recalibration
    (structural models only): var' = var_gp + clamp(s_a*sp(a) +
    s_b*sp(b)*g^2/g2_scale, 1e-3, 10), exactly recalibrate_structural_sigma's
    formula. Mean is untouched."""
    x, y, mask, zeta, zeta_dot, g, lap = run.full_frame(frame)
    gpin = model.masked_gp_inputs(
        x[None].to(device), zeta[None].to(device), mask[None].to(device),
        zeta_dot=(zeta_dot[None].to(device) if model.use_zeta_dot else None),
        g=(g[None].to(device) if model.use_grad_feature else None),
        lap=(lap[None].to(device) if getattr(model, 'use_lap_feature', False) else None))
    structural = getattr(model, 'noise_prior', 'none') == 'structural'
    gm = (g[None].to(device)[mask[None].to(device)] if structural else None)
    if recal is not None and not structural:
        raise SystemExit('--recal requires a structural-noise model')
    mus, vars_ = [], []
    for i0 in range(0, gpin.shape[0], gp_chunk):
        if recal is not None:
            post = model.gp(gpin[i0:i0 + gp_chunk])
            sfeat = (gm[i0:i0 + gp_chunk] ** 2) / model.g2_scale.to(gm.dtype)
            spa = torch.nn.functional.softplus(model.noise_a).to(gm.dtype)
            spb = torch.nn.functional.softplus(model.noise_b).to(gm.dtype)
            noise = (recal['s_a'] * spa + recal['s_b'] * spb * sfeat).clamp(1.0e-3, 10.0)
            # tau_pv tempers the GP posterior variance (3-param recal,
            # Sanaa 2026-07-14); old 2-param sidecars imply tau_pv = 1.0
            var_p = (recal.get('tau_pv', 1.0) * post.variance + noise) \
                * model.y_sd * model.y_sd
            mu_p = post.mean * model.y_sd + model.y_mu
        else:
            mu_p, var_p = model.predict_physical(
                gpin[i0:i0 + gp_chunk],
                g_masked=(gm[i0:i0 + gp_chunk] if gm is not None else None))
        mus.append(mu_p.cpu().numpy())
        vars_.append(var_p.cpu().numpy())
    mu, var = np.concatenate(mus), np.concatenate(vars_)
    m = mask.numpy()
    truth = y.numpy()
    mu2 = np.full_like(truth, np.nan); sg2 = np.full_like(truth, np.nan)
    mu2[m] = mu; sg2[m] = np.sqrt(var)
    return {'omega': x.numpy()[0], 'truth': truth, 'mask': m,
            'mu2d': mu2, 'sigma2d': sg2,
            't': float(run.times[frame]), 'Re': float(run.Re_snap[frame])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=str(HERE / 'conf_piff.yaml'))
    ap.add_argument('--outdir', default=None, help='default: <ckpt dir>/eval')
    ap.add_argument('--device', default=None)
    ap.add_argument('--n-snapshots', type=int, default=None)
    ap.add_argument('--per-member', type=int, default=None, metavar='N',
                    help='write N frames PER member into <outdir>/<member>/ '
                         '(evenly over each member\'s val window) instead of '
                         'the pooled 6-frame selection')
    ap.add_argument('--recal', default='auto', choices=['auto', 'on', 'off'],
                    help='apply recalibration_structural.yaml next to the ckpt '
                         'to the sigma panel (auto = if the file exists)')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = load_conf(args.config)
    device = args.device or conf['train']['device']
    gp_chunk = int(conf['train']['gp_chunk'])
    n_show = int(args.n_snapshots or conf['eval']['n_field_snapshots'])
    outdir = Path(args.outdir or (Path(args.ckpt).parent / 'eval'))
    outdir.mkdir(parents=True, exist_ok=True)

    model = load_piff_model(ckpt, device, conf=conf)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # same conditioning plumbing as eval_piff (flags travel with the ckpt)
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    # data variant travels with the ckpt too (mirror eval_piff:157-159): a
    # jonly ckpt replotted with a full-Pi config would silently show
    # body-force ringing in the truth panel (latent trap, fixed 2026-07-14)
    ck_var = ckpt['conf'].get('data', {}).get('variant')
    if ck_var and not conf['data'].get('variant'):
        conf['data']['variant'] = ck_var
    ck_tsm = ckpt['conf'].get('zeta', {}).get('tshed_smooth', 2.992)
    conf['zeta']['tshed_smooth'] = ck_tsm

    # (s_a, s_b) recalibration sidecar (Sanaa GO 2026-07-14)
    recal = None
    recal_path = Path(args.ckpt).parent / 'recalibration_structural.yaml'
    if args.recal == 'on' or (args.recal == 'auto' and recal_path.exists()):
        import yaml
        rc = yaml.safe_load(recal_path.read_text())
        recal = {'s_a': float(rc['s_a']), 's_b': float(rc['s_b']),
                 'tau_pv': float(rc.get('tau_pv', 1.0))}
        print(f"[replot] sigma recalibration ON: s_a={recal['s_a']:.4g} "
              f"s_b={recal['s_b']:.4g} tau_pv={recal['tau_pv']:.4g} "
              f"({recal_path})")

    runs = build_runs(conf)
    frames = split_frames(runs, 'val', conf)
    print(f'[replot] {len(frames)} val frames from {[r.name for r in runs]}')

    if args.per_member:
        # N frames per member, evenly over each member's val window (time order)
        sel, tags = [], []
        for ri, run in enumerate(runs):
            fis = sorted(fi for rj, fi in frames if rj == ri)
            if not fis:
                continue
            n = min(args.per_member, len(fis))
            for k in np.linspace(0, len(fis) - 1, n).astype(int):
                sel.append((ri, fis[k]))
                tags.append(run.name)
    else:
        # identical selection to eval_piff: sort ALL val frames by Re, 6-pt linspace
        Re_all = np.array([runs[ri].Re_snap[fi] for ri, fi in frames])
        order = np.argsort(Re_all)
        n_show = min(n_show, len(frames))
        sel = [frames[i] for i in order[np.linspace(0, len(order) - 1, n_show).astype(int)]]
        tags = [None] * len(sel)

    counts = {}
    for (ri, fi), tag in zip(sel, tags):
        run = runs[ri]
        p = predict_frame_full(model, run, fi, device, gp_chunk, recal=recal)
        m = p['mask']
        tr = np.where(m, p['truth'], np.nan)
        abserr2d = np.abs(np.nan_to_num(p['mu2d']) - p['truth'])
        err = np.where(m, abserr2d, np.nan)

        # scale stats from the ring-excluded population (sdf > 1D): the body-
        # column ringing (~83x background numerical artefact) otherwise sets
        # vmax; ring pixels saturate instead (Sanaa 2026-07-14 apples-to-apples)
        ring2d = run.sdf[full_frame_slice(run)] > 1.0 * run.D
        mr = m & ring2d
        absval = np.abs(p['truth'][mr if mr.any() else m])
        lt = max(float(np.percentile(absval, 99.0)), 1e-12)
        vmax = max(float(absval.max()), lt * 1.01)
        norm = SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10)
        norm_s = SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10)
        ovmax = float(np.percentile(np.abs(p['omega']), 99.5))

        # rule 5b (Sanaa final rule 2026-07-14 afternoon): SIGNED relative
        # error (pred-truth)/(|truth| + 0.01*max|truth|), colorbar FIXED [-1,1]
        denom = np.abs(p['truth']) + 0.01 * float(absval.max())
        rel = np.where(m, (np.nan_to_num(p['mu2d']) - p['truth']) / denom, np.nan)

        sig_ttl = ('predicted sigma RECALIBRATED (symlog)' if recal is not None
                   else 'predicted sigma (symlog)')
        fig, axs = plt.subplots(1, 6, figsize=(27.5, 4.2))
        specs = [
            (p['omega'], 'filtered vorticity omega_bar* (linear)',
             dict(vmin=-ovmax, vmax=ovmax)),
            (tr, f"truth Pi*  t={p['t']:.2f} Re={p['Re']:.0f} (symlog)", dict(norm=norm)),
            (p['mu2d'], 'predicted Pi* (same symlog)', dict(norm=norm)),
            (p['sigma2d'], sig_ttl, dict(norm=norm_s)),
            (err, '|error| (same symlog)', dict(norm=norm)),
            (rel, 'relative error (pred-truth)/(|truth|+1%max), fixed [-1,1]',
             dict(vmin=-1.0, vmax=1.0)),
        ]
        for ax, (f2d, ttl, kw) in zip(axs, specs):
            im = ax.imshow(f2d, cmap='seismic', origin='lower',
                           extent=[0, run.Lx, 0, run.Ly], aspect='equal', **kw)
            ax.set_title(ttl, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{run.name}  (symlog linthresh = 99th pct |Pi*| = {lt:.3g})",
                     fontsize=10)
        fig.tight_layout()
        if tag is not None:
            fdir = outdir / tag
            fdir.mkdir(parents=True, exist_ok=True)
        else:
            fdir = outdir
        j = counts.get(tag, 0)
        counts[tag] = j + 1
        fp = fdir / f"field6_{j}_t{p['t']:.2f}.png"
        fig.savefig(fp, dpi=130)
        plt.close(fig)
        print(f'[replot] {fp}  (member {run.name}, Re {p["Re"]:.0f})')

    print(f'[replot] done: {len(sel)} figures in {outdir}')


if __name__ == '__main__':
    main()
