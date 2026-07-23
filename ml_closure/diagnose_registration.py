"""
Registration (out-of-phaseness) diagnostic (Sanaa question 2026-07-22: can we
isolate 'slightly off location' from 'wrong content'?). PER MEMBER.

Method: PIV-style local registration. The wake box is tiled (TILE px, stride
STRIDE); per tile and per frame the prediction is slid by every integer shift
(dx, dy) in [-SMAX, SMAX]^2 over the truth tile and the normalized correlation
is recorded at zero shift and at the best shift. Band-passed variants (mid
15<=|k|<130, fine |k|>=130, radial shells on the full periodic field) give the
frequency-vs-location view.

Per member outputs:
  - energy-weighted mean tile corr at zero shift vs at best local shift, per
    band; RECOVERED = (best - zero)/(1 - zero) = the fraction of the
    incoherence explained by small local displacements. The remainder is
    genuinely different structure (missing/invented).
  - median and p90 |shift| (px), and the mean shift VECTOR (a systematic
    displacement — e.g. everything downstream — is a fixable bias).
  - figure: shift-magnitude map + quiver, corr bars per band, |shift| hist.
CSV + summary + [QG][LANDED][sgs] mail. CPU-friendly.
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

TILE, STRIDE, SMAX = 32, 16, 8
BANDS = {'full': None, 'mid': (15, 130), 'fine': (130, 10 ** 9)}


def bandpass(f, klo, khi):
    Ny, Nx = f.shape
    ky = np.fft.fftfreq(Ny) * Ny
    kx = np.fft.fftfreq(Nx) * Nx
    km = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    m = (km >= klo) & (km < khi)
    return np.fft.ifft2(np.fft.fft2(f) * m).real


def tile_register(t, p):
    """Normalized corr of truth tile t (TILE^2) vs pred patch p
    (TILE+2SMAX)^2 at every shift; returns (corr0, corr_best, dx, dy)."""
    tn = t - t.mean()
    nt = np.sqrt((tn * tn).sum())
    if nt < 1e-12:
        return None
    best, b_dx, b_dy, c0 = -2.0, 0, 0, 0.0
    for dy in range(2 * SMAX + 1):
        for dx in range(2 * SMAX + 1):
            w = p[dy:dy + TILE, dx:dx + TILE]
            wn = w - w.mean()
            nw = np.sqrt((wn * wn).sum())
            if nw < 1e-12:
                continue
            c = float((tn * wn).sum() / (nt * nw))
            if dy == SMAX and dx == SMAX:
                c0 = c
            if c > best:
                best, b_dx, b_dy = c, dx - SMAX, dy - SMAX
    return c0, best, b_dx, b_dy, float(nt)


def main():
    ap = argparse.ArgumentParser(description="local-shift registration diagnostic")
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--frame-stride', type=int, default=4)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--no-mail', action='store_true')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    conf = ck['conf']
    dc = conf['data']
    run_name = Path(args.ckpt).resolve().parent.name
    model = PiffCNN(conf).to(args.device)
    model.load_state_dict(ck['model'])
    model.eval()
    runs = build_runs(conf)
    outdir = Path(args.outdir) if args.outdir else (
        HERE / 'results' / 'flow_past_cylinder' / run_name / 'registration')
    outdir.mkdir(parents=True, exist_ok=True)

    frames_by_run = {}
    for ri, fi in split_frames(runs, args.split, conf):
        frames_by_run.setdefault(ri, []).append(fi)

    rows, lines = [], []
    for ri, frames in sorted(frames_by_run.items()):
        r = runs[ri]
        frames = frames[::args.frame_stride]
        xs = (np.arange(r.Nx) + 0.5) * r.dx
        ys = (np.arange(r.Ny) + 0.5) * r.dy
        x_lo = r.x_c + _f(dc['wake_x_lo_D']) * r.D
        x_hi = r.x_c + _f(dc['wake_x_hi_D']) * r.D
        y_half = _f(dc['wake_y_half_D']) * r.D
        cy_ok = np.flatnonzero(np.abs(ys - r.y_c) <= y_half)
        cx_ok = np.flatnonzero((xs >= x_lo) & (xs <= x_hi))
        j0, j1 = cy_ok[0], cy_ok[-1]
        i0, i1 = cx_ok[0], cx_ok[-1]

        res = {b: [] for b in BANDS}          # (c0, cb, dx, dy, w) tuples
        shift_map = {}                        # (ty,tx) -> (dx,dy,|d|) full band
        for fi in frames:
            x, y, m, zeta, zeta_dot, _, lap_pl, psi_pl = r.full_frame(fi)
            with torch.no_grad():
                pred = model.predict_physical(
                    x[None].to(args.device), zeta[None].to(args.device),
                    zeta_dot[None].to(args.device) if model.use_zeta_dot else None,
                    lap_pl[None].to(args.device) if getattr(model, 'use_lap_input', False) else None,
                    psi_pl[None].to(args.device) if getattr(model, 'use_psi_input', False) else None
                )[0].cpu().numpy().astype(np.float64)
            t_full = y.numpy().astype(np.float64)
            for band, kk in BANDS.items():
                tb = t_full if kk is None else bandpass(t_full, *kk)
                pb = pred if kk is None else bandpass(pred, *kk)
                for ty in range(j0, j1 - TILE, STRIDE):
                    for tx in range(i0, i1 - TILE, STRIDE):
                        if (ty < SMAX or tx < SMAX
                                or ty + TILE + SMAX >= r.Ny
                                or tx + TILE + SMAX >= r.Nx):
                            continue
                        out = tile_register(
                            tb[ty:ty + TILE, tx:tx + TILE],
                            pb[ty - SMAX:ty + TILE + SMAX,
                               tx - SMAX:tx + TILE + SMAX])
                        if out is None:
                            continue
                        res[band].append(out)
                        if band == 'full':
                            c0, cb, dx, dy, w = out
                            k = (ty, tx)
                            e = shift_map.get(k, (0.0, 0.0, 0.0, 0.0))
                            shift_map[k] = (e[0] + w * dx, e[1] + w * dy,
                                            e[2] + w, e[3] + w * np.hypot(dx, dy))
        for band in BANDS:
            a = np.array(res[band])            # (n, 5)
            if a.size == 0:
                continue
            w = a[:, 4]
            c0 = float((a[:, 0] * w).sum() / w.sum())
            cb = float((a[:, 1] * w).sum() / w.sum())
            d = np.hypot(a[:, 2], a[:, 3])
            rows.append({'member': r.name, 'band': band, 'n_tiles': len(a),
                         'corr0': c0, 'corr_best': cb,
                         'recovered': (cb - c0) / max(1.0 - c0, 1e-9),
                         'shift_med_px': float(np.median(d)),
                         'shift_p90_px': float(np.percentile(d, 90)),
                         'mean_dx_px': float((a[:, 2] * w).sum() / w.sum()),
                         'mean_dy_px': float((a[:, 3] * w).sum() / w.sum())})

        # figure: weighted mean shift field + bars
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
        tys = sorted({k[0] for k in shift_map})
        txs = sorted({k[1] for k in shift_map})
        mag = np.full((len(tys), len(txs)), np.nan)
        U = np.zeros_like(mag); V = np.zeros_like(mag)
        for (ty, tx), (sx, sy, sw, sm) in shift_map.items():
            iy, ix = tys.index(ty), txs.index(tx)
            mag[iy, ix] = sm / sw
            U[iy, ix], V[iy, ix] = sx / sw, sy / sw
        im = axs[0].imshow(mag, origin='lower', cmap='viridis', vmin=0,
                           vmax=SMAX, aspect='equal')
        axs[0].quiver(U, V, color='w', scale=60)
        plt.colorbar(im, ax=axs[0], fraction=0.046)
        axs[0].set_title('mean local shift |px| + direction (full band)',
                         fontsize=9)
        ax = axs[1]
        bl = [q for q in rows if q['member'] == r.name]
        xpos = np.arange(len(bl))
        ax.bar(xpos - 0.15, [q['corr0'] for q in bl], 0.3, label='corr @ 0 shift')
        ax.bar(xpos + 0.15, [q['corr_best'] for q in bl], 0.3,
               label='corr @ best local shift')
        ax.set_xticks(xpos); ax.set_xticklabels([q['band'] for q in bl])
        ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
        ax.set_title('coherence recovered by local shifts', fontsize=9)
        ax = axs[2]
        a = np.array(res['full'])
        ax.hist(np.hypot(a[:, 2], a[:, 3]), bins=np.arange(0, SMAX + 1.5) - 0.5,
                density=True)
        ax.set_xlabel('|shift| px (full band)'); ax.grid(alpha=0.3)
        ax.set_title('local shift magnitudes', fontsize=9)
        fig.suptitle(f'{r.name}  registration  {run_name}', fontsize=11)
        fig.tight_layout()
        fig.savefig(outdir / f'registration_{r.name}.png', dpi=140)
        plt.close(fig)
        print(f'[reg] {r.name} done ({len(frames)} frames)')

    cols = ['member', 'band', 'n_tiles', 'corr0', 'corr_best', 'recovered',
            'shift_med_px', 'shift_p90_px', 'mean_dx_px', 'mean_dy_px']
    with open(outdir / 'registration.csv', 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        for row in rows:
            wtr.writerow(row)
    lines.append(f'{run_name} — local-shift registration ({args.split}, '
                 f'tile {TILE}px, max shift {SMAX}px)')
    lines.append(f"{'member':<10}{'band':<6}{'corr0':>7}{'corrB':>7}{'recov':>7}"
                 f"{'d_med':>7}{'d_p90':>7}{'<dx,dy>':>12}")
    for row in rows:
        lines.append(f"{row['member']:<10}{row['band']:<6}{row['corr0']:>7.3f}"
                     f"{row['corr_best']:>7.3f}{row['recovered']:>7.1%}"
                     f"{row['shift_med_px']:>7.1f}{row['shift_p90_px']:>7.1f}"
                     f"   {row['mean_dx_px']:+.1f},{row['mean_dy_px']:+.1f}")
    body = '\n'.join(lines)
    (outdir / 'summary.md').write_text(body + '\n')
    print(body)
    if not args.no_mail:
        try:
            PENDING.mkdir(parents=True, exist_ok=True)
            p = PENDING / f'reg_{int(time.time())}_{os.getpid()}.mail'
            p.write_text(f'To: sanaamz@mit.edu\nSubject: [QG][LANDED][sgs] '
                         f'{run_name} registration diagnostic\n\n{body}\n\n'
                         f'Figures: {outdir}\n')
            print(f'[reg] mail spooled: {p}')
        except OSError as e:
            print(f'[reg] mail spool failed ({e})')


if __name__ == '__main__':
    main()
