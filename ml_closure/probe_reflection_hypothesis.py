#!/usr/bin/env python
"""probe_reflection_hypothesis.py -- is the near-wall missed-peak error caused
by the upstream-reflection contamination of the reference velocity?
(Sanaa GO 2026-07-15 evening; full data access granted incl. DNS_FR.)

Three tests on ONE member (the geometry's worst: FPC-const / FPCape-sine),
restricted to the frames that carry the top-K extreme events:

  A. MEASURED per-frame contamination ratio r(t) = U_local(t)/U_inlet(t).
     u from DNS_FR omega (2048^2) by spectral inversion (psi_hat = -omega_hat
     /|k|^2, u = d psi/d y), slab-averaged over the July-14 study's band
     x in [x_c-4D, x_c-1.5D] (all y), assuming the k=0 mean flow equals the
     inlet table U(t). SIGN CALIBRATION: the same slab mean is computed from
     the LES ubar single frame -- the spectral sign convention is validated
     against it and against the study's 0.859 (fpc); both are recorded, the
     probe hard-fails if neither sign reproduces the LES measurement.

  B. ZETA-OVERRIDE: re-predict the extreme frames with zeta computed from
     Re_eff(t) = Re(t) * r  -- (i) r = measured r(t) from A, (ii) a constant-
     ratio sweep. zeta_dot scales by r (affine approx). Everything else
     untouched. Readout: prediction at the top-K event pixels vs truth, and
     RMSE on the |truth|>=q90 active set of those frames. If conditioning on
     the true incident velocity moves the near-wall peaks toward truth, the
     reflection hypothesis stands; if flat, it is exonerated.

  C. FILTER-VARIANT CONTROL: truth Pi* at the event pixels read from
     DNS_LES_s4.npz (sharp), _gaussian.npz, _gaussian_jonly_ylp75.npz --
     same normalization D^2/U(t)^2. Peaks present in all variants = physics
     being missed; peaks in one variant only = target-side artifact.

Outputs: runs_piff/<model>/reflection_probe/<member>/{metrics.yaml,
events_augmented.csv} + pngs/reflection_probe/<model>/<member>/*.png +
reports/<report-run>/ push. CPU only (diagnostics-never-GPU ruling).

Usage (via piff_tool_job.sh, all.q):
  cd ml_closure && python probe_reflection_hypothesis.py \
      --ckpt runs_piff/piff_fpc_gjs_ylp75/best.pt \
      --config conf_piff_fpc_gjs_ylp75.yaml --member FPC-const \
      --ratios 0.7,0.8,0.859,0.9,1.0 [--report-run reflection_probe_fpc]
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

from dataset_piff import load_conf, build_runs, split_frames, _f
from model_piff import PiffModel
from eval_piff import predict_frame

HERE = Path(__file__).resolve().parent
BRANCH_ROOT = HERE.parent
VARIANT_FILES = ['DNS_LES_s4.npz', 'DNS_LES_s4_gaussian.npz',
                 'DNS_LES_s4_gaussian_jonly_ylp75.npz']


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- test A

def u_slab_from_omega(om, Lx, Ly, x_lo, x_hi, sign, y_c, half_w):
    """Slab-mean streamwise velocity induced by the vorticity field:
    psi_hat = sign * omega_hat / |k|^2 (k=0 -> 0), u = d psi / d y.
    The y-band MUST be restricted (|y - y_c| <= half_w): the full-column mean
    of d psi/d y is identically zero by periodicity (2026-07-15 degenerate-
    ratio bug), so an all-y average measures nothing."""
    ny, nx = om.shape
    oh = np.fft.rfft2(om)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=Lx / nx)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    k2[0, 0] = 1.0
    psi_h = sign * oh / k2
    psi_h[0, 0] = 0.0
    u = np.fft.irfft2(1j * ky[:, None] * psi_h, s=om.shape)
    dxg, dyg = Lx / nx, Ly / ny
    i_lo, i_hi = int(x_lo / dxg), int(x_hi / dxg)
    j_lo = max(0, int((y_c - half_w) / dyg))
    j_hi = min(ny, int((y_c + half_w) / dyg))
    return float(u[j_lo:j_hi, i_lo:i_hi].mean())


def measure_ratio(run, frames, fr_dir, band_D=(-4.0, -1.5)):
    """r(t) = 1 + mean_slab(u_omega)/U_in(t) on the LES frames, from DNS_FR.
    Sign calibrated against the LES ubar single frame."""
    om_mm = np.load(fr_dir / 'DNS_FR_omega.npy', mmap_mode='r')
    if om_mm.ndim == 4:          # stored with the (1, T, Ny, Nx) batch dim
        om_mm = om_mm[0]
    fr_t = np.load(fr_dir / 'DNS_FR_times.npy')
    if fr_t.ndim > 1:
        fr_t = fr_t.reshape(-1)
    x_lo = run.x_c + band_D[0] * run.D
    x_hi = run.x_c + band_D[1] * run.D
    # LES ubar reference (single stored frame): slab mean / U at that frame
    iu_lo, iu_hi = int(x_lo / run.dx), int(x_hi / run.dx)
    ub = run.u if run.u.ndim == 2 else run.u[0]
    u_les = float(np.asarray(ub, dtype=np.float64)[:, iu_lo:iu_hi].mean())
    # ubar is stored normalized by U at its frame (spec S1.2) -> deficit
    les_deficit = u_les if abs(u_les) < 2.0 else None

    # calibrate the inversion sign on the first requested frame
    hw = 1.5 * run.D
    j0 = int(np.argmin(np.abs(fr_t - run.times[frames[0]])))
    cal = {}
    for sign in (+1.0, -1.0):
        du = u_slab_from_omega(np.asarray(om_mm[j0], dtype=np.float64),
                               run.Lx, run.Ly, x_lo, x_hi, sign, run.y_c, hw)
        cal[float(sign)] = float(1.0 + du / run.U_snap[frames[0]])
    # pick the sign giving a DEFICIT consistent with the LES measurement
    # (study: U_local/U_in < 1); record both.
    sign = min(cal, key=lambda s: abs(cal[s] - (1.0 + (les_deficit or 0.0) - 1.0))
               if les_deficit is not None else abs(cal[s] - 0.859))
    if not (0.05 < cal[sign] < 1.5):
        raise SystemExit(f'[probe] sign calibration failed: {cal} vs LES '
                         f'{les_deficit} -- refusing to guess')
    ratios = {}
    for fi in frames:
        j = int(np.argmin(np.abs(fr_t - run.times[fi])))
        du = u_slab_from_omega(np.asarray(om_mm[j], dtype=np.float64),
                               run.Lx, run.Ly, x_lo, x_hi, sign, run.y_c, hw)
        ratios[fi] = float(1.0 + du / run.U_snap[fi])
    return ratios, {'sign': float(sign), 'both_signs_frame0': cal,
                    'les_ubar_slab_over_U':
                        (float(les_deficit) if les_deficit is not None else None),
                    'band_x_D': [float(b) for b in band_D],
                    'band_y_half_D': 1.5}


# ---------------------------------------------------------------- test B

def predict_with_ratio(model, run, frame, device, chunk, ratio, re0, re_scale):
    """predict_frame with zeta(t) recomputed from Re_eff = Re(t)*ratio;
    zeta_dot scaled by ratio (affine approx). Arrays restored afterwards."""
    z0 = float(run.zeta_snap[frame]); zd0 = float(run.zeta_dot_snap[frame])
    try:
        run.zeta_snap[frame] = (run.Re_snap[frame] * ratio - re0) / re_scale
        run.zeta_dot_snap[frame] = zd0 * ratio
        return predict_frame(model, run, frame, device, chunk)
    finally:
        run.zeta_snap[frame] = z0
        run.zeta_dot_snap[frame] = zd0


# ---------------------------------------------------------------- test C

def truth_variants_at_events(run, events):
    """Pi* at the event pixels from each stored filter variant, normalized
    with the SAME D^2/U(t)^2 as training."""
    out = {}
    for fn in VARIANT_FILES:
        p = run.run_dir / fn
        if not p.exists():
            continue
        z = np.load(p)
        key = next((k for k in z.files if 'pi' in k.lower()), None)
        if key is None:
            continue
        pi = z[key]
        pi = pi[0] if pi.ndim == 4 else pi
        vals = []
        for ev in events:
            fi = ev['frame']
            iy = int(round(ev['y'] / run.dy)) % run.Ny
            ix = int(round(ev['x'] / run.dx)) % run.Nx
            U = run.U_snap[fi]
            vals.append(float(pi[fi, iy, ix]) * run.D ** 2 / U ** 2)
        out[fn.replace('DNS_LES_s4', 's4').replace('.npz', '')] = vals
        del z, pi
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--member', required=True)
    ap.add_argument('--ratios', default='0.7,0.8,0.859,0.9,1.0')
    ap.add_argument('--gp-chunk', type=int, default=200_000)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--report-run', default=None)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    sweep = [float(r) for r in args.ratios.split(',')]

    ckpt = Path(args.ckpt)
    conf = load_conf(HERE / args.config)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    model = PiffModel(ck['conf']).to(args.device)
    model.load_state_dict(ck['model'])
    model.eval()
    conf.setdefault('model', {})['use_grad_feature'] = model.use_grad_feature
    ck_var = ck['conf'].get('data', {}).get('variant')
    if ck_var and not conf['data'].get('variant'):
        conf['data']['variant'] = ck_var
    conf.setdefault('zeta', {})['tshed_smooth'] = float(
        ck['conf'].get('zeta', {}).get('tshed_smooth', 2.992))
    re0 = _f(conf['zeta']['re0']); re_scale = _f(conf['zeta']['re_scale'])

    runs = build_runs(conf)
    run = next((r for r in runs if r.name == args.member), None)
    if run is None:
        raise SystemExit(f'member {args.member} not in {args.config}')

    ev_csv = (ckpt.parent / 'error_tails_diag' / args.member
              / 'extreme_events.csv')
    if not ev_csv.exists():
        raise SystemExit(f'{ev_csv} missing -- run diagnose_error_tails first')
    with open(ev_csv) as f:
        events = [{k: (int(v) if k == 'frame' else float(v))
                   for k, v in row.items()}
                  for row in list(csv.DictReader(f))[:args.top_k]]
    frames = sorted({ev['frame'] for ev in events})
    print(f'[probe] {args.member}: {len(events)} events on {len(frames)} '
          f'frames', flush=True)

    out_dir = ckpt.parent / 'reflection_probe' / args.member
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = HERE / 'pngs' / 'reflection_probe' / ckpt.parent.name / args.member

    # A. measured contamination -- NON-FATAL: a failure here degrades the
    # probe to sweep-only (the constant-ratio sweep still yields a verdict)
    try:
        ratios_t, cal = measure_ratio(run, frames, run.run_dir)
        r_mean = float(np.mean(list(ratios_t.values())))
        print(f"[probe] measured U_local/U_in: mean {r_mean:.3f} "
              f"(sign cal {cal['both_signs_frame0']}, LES ubar "
              f"{cal['les_ubar_slab_over_U']})", flush=True)
    except Exception as e:
        ratios_t, cal, r_mean = None, {'error': repr(e)}, float('nan')
        print(f"[probe] WARNING: measured-ratio unavailable ({e!r}) -- "
              "sweep-only mode", flush=True)

    # B. predictions: baseline + measured r(t) + constant sweep
    ev_by_frame = {}
    for i, ev in enumerate(events):
        ev_by_frame.setdefault(ev['frame'], []).append(i)
    modes = [('baseline', None)] \
        + ([('measured', 'measured')] if ratios_t is not None else []) \
        + [(f'r={r:g}', r) for r in sweep]
    pred_at_events = {m: np.zeros(len(events)) for m, _ in modes}
    rmse_active = {m: [] for m, _ in modes}
    with torch.no_grad():
        for fi in frames:
            for mode, r in modes:
                rv = (ratios_t[fi] if r == 'measured'
                      else (r if r is not None else 1.0))
                p = (predict_frame(model, run, fi, args.device, args.gp_chunk)
                     if mode == 'baseline' else
                     predict_with_ratio(model, run, fi, args.device,
                                        args.gp_chunk, rv, re0, re_scale))
                mu2d = p['mu2d']
                for i in ev_by_frame[fi]:
                    ev = events[i]
                    iy = int(round(ev['y'] / run.dy)) % run.Ny
                    ix = int(round(ev['x'] / run.dx)) % run.Nx
                    pred_at_events[mode][i] = float(mu2d[iy, ix])
                y, mu = p['y'], p['mu']
                act = np.abs(y) >= np.percentile(np.abs(y), 90)
                rmse_active[mode].append(
                    float(np.sqrt(np.mean((mu[act] - y[act]) ** 2))))
            print(f'[probe] frame {fi} done', flush=True)

    truth = np.array([ev['truth'] for ev in events])
    gap = {m: float(np.mean(np.abs(pred_at_events[m] - truth)))
           for m, _ in modes}
    ra = {m: float(np.mean(v)) for m, v in rmse_active.items()}

    # C. filter-variant truth
    variants = truth_variants_at_events(run, events)

    # ---- outputs ----
    rows = []
    for i, ev in enumerate(events):
        row = dict(ev)
        row['r_measured'] = (ratios_t[ev['frame']]
                             if ratios_t is not None else float('nan'))
        for m, _ in modes:
            row[f'pred_{m}'] = pred_at_events[m][i]
        for vn, vals in variants.items():
            row[f'truth_{vn}'] = vals[i]
        rows.append(row)
    with open(out_dir / 'events_augmented.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # discriminant: the measured-r gap if available, else the best sweep point
    sweep_gaps = [gap[m] for m, r in modes if isinstance(r, float)]
    disc = gap.get('measured', min(sweep_gaps) if sweep_gaps else 9e9)
    verdict = ('REFLECTION-SUPPORTED' if disc < 0.8 * gap['baseline'] else
               'REFLECTION-EXONERATED (conditioning moves <20% of the gap)')
    metrics = {'member': args.member, 'n_events': len(events),
               'sign_calibration': cal,
               'ratio_measured_mean': r_mean,
               'ratio_per_frame': ({int(k): float(v)
                                    for k, v in ratios_t.items()}
                                   if ratios_t is not None else 'unavailable'),
               'mean_abs_gap_at_events': gap,
               'rmse_active_q90_extreme_frames': ra,
               'variant_truth_rms_at_events':
                   {k: float(np.sqrt(np.mean(np.square(v))))
                    for k, v in variants.items()},
               'verdict': verdict}
    with open(out_dir / 'metrics.yaml', 'w') as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xs, ys = [], []
    for m, r in modes:
        if r is None or r == 'measured':
            continue
        xs.append(r); ys.append(gap[m])
    ax.plot(xs, ys, 'o-', label='constant-ratio sweep')
    ax.axhline(gap['baseline'], color='k', ls='--', label='baseline (r=1)')
    if 'measured' in gap:
        ax.plot([r_mean], [gap['measured']], 'r*', ms=14,
                label=f'measured r(t) (mean {r_mean:.3f})')
    ax.set_xlabel('U_local / U_inlet ratio fed to zeta')
    ax.set_ylabel('mean |pred - truth| at the top-50 event pixels')
    ax.set_title(f'{args.member}: does effective-Re conditioning close the '
                 'missed peaks?', fontsize=9)
    ax.legend(fontsize=8)
    savefig(fig, fig_dir / 'event_gap_vs_conditioning_ratio.png')

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    order = np.argsort(truth)
    ax.plot(truth[order], 'k.-', label='truth (ylp75, eval target)')
    ax.plot(pred_at_events['baseline'][order], 'b.', label='pred baseline')
    if 'measured' in pred_at_events:
        ax.plot(pred_at_events['measured'][order], 'r.',
                label='pred @ measured r(t)')
    for vn, vals in variants.items():
        if 'ylp75' in vn:
            continue
        ax.plot(np.array(vals)[order], 'x', ms=4, alpha=0.6,
                label=f'truth {vn}')
    ax.set_xlabel('event rank (sorted by truth)')
    ax.set_ylabel('Pi* (normalized)')
    ax.set_title(f'{args.member}: events -- truth variants vs predictions',
                 fontsize=9)
    ax.legend(fontsize=7)
    savefig(fig, fig_dir / 'events_truth_variants_vs_predictions.png')

    print(f"[probe] VERDICT: {verdict}")
    print(f"[probe] gaps: {gap}")
    print(f"[probe] variant truth RMS at events: "
          f"{metrics['variant_truth_rms_at_events']}", flush=True)

    if args.report_run:
        rep = BRANCH_ROOT / 'reports' / args.report_run
        rep.mkdir(parents=True, exist_ok=True)
        (rep / 'summary.md').write_text(
            f"# reflection probe -- {args.member} ({ckpt.parent.name})\n\n"
            f"verdict: **{verdict}**\n\n"
            f"measured U_local/U_in mean: {r_mean:.3f} "
            f"(LES ubar cal: {cal['les_ubar_slab_over_U']})\n\n"
            f"mean |pred-truth| at top-{len(events)} events per mode:\n\n"
            + '\n'.join(f"- {m}: {g:.3f}" for m, g in gap.items())
            + '\n\nactive-set RMSE (extreme frames):\n\n'
            + '\n'.join(f"- {m}: {v:.4f}" for m, v in ra.items())
            + '\n\nvariant truth RMS at events:\n\n'
            + '\n'.join(f"- {k}: {v:.3f}" for k, v in
                        metrics['variant_truth_rms_at_events'].items())
            + '\n')
        dw = BRANCH_ROOT / 'diagnostics' / 'digest_writer.py'
        if dw.exists():
            r = subprocess.run([sys.executable, str(dw), '--repo-dir',
                                str(BRANCH_ROOT), '--run-name',
                                args.report_run, '--event', 'done', '--note',
                                f'reflection probe {args.member}: {verdict}'],
                               capture_output=True, text=True)
            print(f"[report] digest push rc={r.returncode}", flush=True)


if __name__ == '__main__':
    main()
