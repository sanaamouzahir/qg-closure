"""
Shape-vs-amplitude decomposition of a CNN-only Pi_FF checkpoint (Sanaa
question 2026-07-22: "is it the shape that is mostly wrong, or the amplitude,
or both?"). PER MEMBER, never pooled.

Two exact decompositions over the val split:

1. MURPHY (per frame, per region near/wake, medians over frames):
       MSE = bias^2 + (sigma_p - sigma_t)^2 + 2 sigma_p sigma_t (1 - rho)
   The three terms ARE bias / AMPLITUDE error / SHAPE error and sum exactly
   to the measured MSE. Also reported: amplitude ratio sigma_p/sigma_t and
   pattern correlation rho.

2. SPECTRAL (pooled over frames, full periodic field, radial shells):
       amp_ratio(k) = sqrt(S_pp(k) / S_tt(k))         [amplitude per scale]
       coherence(k) = Re S_pt(k) / sqrt(S_pp S_tt)    [shape per scale]
   S_ab(k) = sum over shell of A(k) conj(B(k)), accumulated f64 over frames.
   Localizes WHERE (in scale) shape is lost vs merely damped.

Outputs per member: one 3-panel figure + rows in shape_amplitude.csv +
printed table; spools the table as [QG][LANDED][sgs] mail.
CPU-friendly (FFT2 512^2 x ~45 frames x 5 members).
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

from dataset_piff import load_conf, build_runs, split_frames, _f
from model_cnn import PiffCNN

HERE = Path(__file__).resolve().parent
PENDING = Path(os.environ.get(
    'QG_PENDING_MAIL',
    '/gdata/projects/ml_scope/Closure_modeling/QG-closure/reporting/pending_mail'))


def murphy(p, t):
    """Exact MSE decomposition on 1-D pixel arrays. Returns dict; terms sum
    to mse to round-off (asserted by the caller's self-check)."""
    bias = float(p.mean() - t.mean())
    sp, st = float(p.std()), float(t.std())
    rho = float(np.corrcoef(p, t)[0, 1]) if p.size > 1 else np.nan
    mse = float(((p - t) ** 2).mean())
    return {'mse': mse, 'bias2': bias * bias, 'amp': (sp - st) ** 2,
            'shape': 2.0 * sp * st * (1.0 - rho),
            'amp_ratio': sp / st if st > 0 else np.nan, 'rho': rho}


def main():
    ap = argparse.ArgumentParser(description="shape vs amplitude decomposition")
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'train'])
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--no-mail', action='store_true')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = ck['conf']
    run_name = Path(args.ckpt).resolve().parent.name
    split_D = _f(conf['data'].get('eval_split_D', 1.25))
    dc = conf['data']

    model = PiffCNN(conf).to(args.device)
    model.load_state_dict(ck['model'])
    model.eval()
    runs = build_runs(conf)

    outdir = Path(args.outdir) if args.outdir else (
        HERE / 'results' / 'flow_past_cylinder' / run_name / 'shape_amplitude')
    outdir.mkdir(parents=True, exist_ok=True)

    frames_by_run = {}
    for ri, fi in split_frames(runs, args.split, conf):
        frames_by_run.setdefault(ri, []).append(fi)

    rows, lines = [], []
    for ri, frames in sorted(frames_by_run.items()):
        r = runs[ri]
        near_m = r.valid & (r.sdf <= split_D * r.D)
        xs = (np.arange(r.Nx) + 0.5) * r.dx
        ys = (np.arange(r.Ny) + 0.5) * r.dy
        wake_m = (r.valid
                  & (xs[None, :] >= r.x_c + _f(dc['wake_x_lo_D']) * r.D)
                  & (xs[None, :] <= r.x_c + _f(dc['wake_x_hi_D']) * r.D)
                  & (np.abs(ys[:, None] - r.y_c) <= _f(dc['wake_y_half_D']) * r.D))

        per_frame = {'near': [], 'wake': []}
        # spectral accumulators (f64), radial integer shells to N/2
        nsh = min(r.Ny, r.Nx) // 2
        ky = np.fft.fftfreq(r.Ny) * r.Ny
        kx = np.fft.fftfreq(r.Nx) * r.Nx
        kmag = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        shell = np.minimum(kmag.astype(np.int64), nsh)
        S_pp = np.zeros(nsh + 1); S_tt = np.zeros(nsh + 1); S_pt = np.zeros(nsh + 1)

        for fi in frames:
            x, y, m, zeta, zeta_dot, _, _ = r.full_frame(fi)
            with torch.no_grad():
                pred = model.predict_physical(
                    x[None].to(args.device), zeta[None].to(args.device),
                    zeta_dot[None].to(args.device) if model.use_zeta_dot else None
                )[0].cpu().numpy().astype(np.float64)
            t = y.numpy().astype(np.float64)
            for key, sel in (('near', near_m), ('wake', wake_m)):
                per_frame[key].append(murphy(pred[sel], t[sel]))
            P = np.fft.fft2(pred); T = np.fft.fft2(t)
            S_pp += np.bincount(shell.ravel(), weights=np.abs(P.ravel()) ** 2,
                                minlength=nsh + 1)
            S_tt += np.bincount(shell.ravel(), weights=np.abs(T.ravel()) ** 2,
                                minlength=nsh + 1)
            S_pt += np.bincount(shell.ravel(),
                                weights=(P * np.conj(T)).real.ravel(),
                                minlength=nsh + 1)

        # self-check (verify-before-reporting): terms sum to mse per frame
        for key in ('near', 'wake'):
            for d in per_frame[key]:
                assert abs(d['bias2'] + d['amp'] + d['shape'] - d['mse']) \
                    <= 1e-9 * max(d['mse'], 1e-30), 'murphy terms do not sum'

        for key in ('near', 'wake'):
            med = {q: float(np.median([d[q] for d in per_frame[key]]))
                   for q in ('mse', 'bias2', 'amp', 'shape', 'amp_ratio', 'rho')}
            tot = max(med['bias2'] + med['amp'] + med['shape'], 1e-30)
            rows.append({'member': r.name, 'region': key, **med,
                         'share_bias': med['bias2'] / tot,
                         'share_amp': med['amp'] / tot,
                         'share_shape': med['shape'] / tot})

        with np.errstate(invalid='ignore', divide='ignore'):
            amp_k = np.sqrt(S_pp / S_tt)
            coh_k = S_pt / np.sqrt(S_pp * S_tt)
        np.savez(outdir / f'spectral_{r.name}.npz', amp_ratio_k=amp_k,
                 coherence_k=coh_k, S_pp=S_pp, S_tt=S_tt, S_pt=S_pt)

        fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
        ax = axs[0]
        for key, col in (('near', 'tab:red'), ('wake', 'tab:blue')):
            d = [q for q in rows if q['member'] == r.name and q['region'] == key][0]
            ax.bar([f'bias\n{key}', f'amp\n{key}', f'shape\n{key}'],
                   [d['share_bias'], d['share_amp'], d['share_shape']],
                   color=col, alpha=0.6)
        ax.set_title('MSE share: bias / amplitude / shape', fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        ax = axs[1]
        k = np.arange(nsh + 1)
        ax.loglog(k[1:], S_tt[1:] / len(frames), 'k-', label=r'$E_{truth}(k)$')
        ax.loglog(k[1:], S_pp[1:] / len(frames), 'r--', label=r'$E_{pred}(k)$')
        ax.set_xlabel('k shell'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title('spectra (pooled frames)', fontsize=9)
        ax = axs[2]
        ax.semilogx(k[1:], coh_k[1:], 'b-', label='coherence (shape)')
        ax.semilogx(k[1:], amp_k[1:], 'r--', label='amp ratio')
        ax.axhline(1.0, color='k', lw=0.5)
        ax.set_ylim(-0.1, 1.5); ax.set_xlabel('k shell')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title('per-scale: shape vs amplitude', fontsize=9)
        fig.suptitle(f'{r.name}  shape-vs-amplitude  {run_name}', fontsize=11)
        fig.tight_layout()
        fig.savefig(outdir / f'shape_amplitude_{r.name}.png', dpi=140)
        plt.close(fig)

    cols = ['member', 'region', 'rho', 'amp_ratio', 'share_shape', 'share_amp',
            'share_bias', 'mse']
    with open(outdir / 'shape_amplitude.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)
    lines.append(f'{run_name} — shape vs amplitude ({args.split}; medians over frames)')
    lines.append(f"{'member':<14}{'region':<7}{'rho':>7}{'amp_ratio':>11}"
                 f"{'shape%':>8}{'amp%':>7}{'bias%':>7}")
    for row in rows:
        lines.append(f"{row['member']:<14}{row['region']:<7}{row['rho']:>7.3f}"
                     f"{row['amp_ratio']:>11.3f}{100 * row['share_shape']:>8.1f}"
                     f"{100 * row['share_amp']:>7.1f}{100 * row['share_bias']:>7.1f}")
    body = '\n'.join(lines)
    (outdir / 'summary.md').write_text(body + '\n')
    print(body)
    if not args.no_mail:
        try:
            PENDING.mkdir(parents=True, exist_ok=True)
            p = PENDING / f'shapeamp_{int(time.time())}_{os.getpid()}.mail'
            p.write_text(f'To: sanaamz@mit.edu\nSubject: [QG][LANDED][sgs] '
                         f'{run_name} shape-vs-amplitude decomposition\n\n'
                         f'{body}\n\nFigures+npz: {outdir}\n')
            print(f'[diag] mail spooled: {p}')
        except OSError as e:
            print(f'[diag] mail spool failed ({e})')


if __name__ == '__main__':
    main()
