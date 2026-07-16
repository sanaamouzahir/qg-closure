#!/usr/bin/env python
"""analyze_sponge_sweep.py -- S1/S2 analyzer of the reflection pipeline.
[fable-authored 2026-07-15]

Inlet-cleanliness metrics from a run's DNS_FR.npz (omega + times + U table in
config.yaml's inlet_table), on an inlet strip x in [0, 1.5D]:
    u_deficit = |mean_strip(u_omega)| / U(t)      (k=0 mean flow == U assumed)
    v_rms     = RMS_strip(v_omega) / U(t)         (v is fully omega-induced)
    om_rms    = RMS_strip(omega) * D / U(t)
    st_leak   = fraction of strip-v temporal power in the shedding band
                f in [0.5, 2] * St_ref * U/D (St_ref 0.2) -- energy at the
                shedding frequency AT THE INLET can only arrive by
                reflection/wraparound.
PASS thresholds (each, time-median over the window): u_deficit < 1e-3,
v_rms < 1e-3, om_rms < 1e-3, st_leak < 0.05.

Modes:
  --sweep-root <dir> --geometry g --t-window a b --out <file>
      score every p*/ subdir, pick the SMALLEST penalty whose metrics pass
      AND whose successor also passes (margin); write it to --out, email a
      table via the pending-mail spool (works from network-less nodes: the
      spool is shared FS). No qualifying penalty -> no out file + FLAG mail.
  --check-run <member-dir> --member NAME --append <file>
      same metrics on the last 10 time units; append 'NAME PASS|FAIL ...'.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

TOL = {'u_deficit': 1e-3, 'v_rms': 1e-3, 'om_rms': 1e-3, 'st_leak': 0.05}
ST_REF = 0.2
SPOOL = Path('/gdata/projects/ml_scope/Closure_modeling/QG-closure/'
             'reporting/pending_mail')
EMAIL = 'sanaamz@mit.edu'


def mail(tag, subject, body):
    SPOOL.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    (SPOOL / f'{stamp}_{tag}.mail').write_text(
        f'To: {EMAIL}\nSubject: {subject}\n\n{body}\n')


def run_metrics(run_dir, t_lo=None, t_hi=None):
    run_dir = Path(run_dir)
    z = np.load(run_dir / 'DNS_FR.npz')
    om_all = z['omega_FR'] if 'omega_FR' in z.files else z['omega']
    if om_all.ndim == 4:
        om_all = om_all[0]
    times = z['times']
    cfg_p = run_dir / 'config.yaml'
    if not cfg_p.exists():
        cfg_p = run_dir / '.hydra' / 'config.yaml'
    cfg = yaml.safe_load(cfg_p.read_text())
    cfg = cfg.get('qg', cfg)          # hydra run dirs nest under qg:
    Lx = float(cfg['grid']['Lx']); Ly = float(cfg['grid']['Ly'])
    D = float(cfg.get('bc', {}).get('D', 1.0) or 1.0)
    tab_path = str(cfg.get('bc', {}).get('inlet_table') or '')
    U_t = None
    if tab_path and Path(tab_path).exists():
        tab = np.load(tab_path)
        U_t = np.interp(times, tab['t'], tab['U'])
    sel = np.ones(len(times), bool)
    if t_lo is not None:
        sel &= (times >= t_lo) & (times <= t_hi)
    idx = np.nonzero(sel)[0]
    if idx.size == 0:
        raise SystemExit(f'{run_dir}: no frames in window')
    ny, nx = om_all.shape[-2:]
    dxg = Lx / nx
    i_hi = max(2, int(1.5 * D / dxg))
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=Lx / nx)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    k2[0, 0] = 1.0
    ud, vr, orr, v_series = [], [], [], []
    for j in idx:
        om = np.asarray(om_all[j], dtype=np.float64)
        U = float(U_t[j]) if U_t is not None else 1.0
        oh = np.fft.rfft2(om)
        psi = -oh / k2
        psi[0, 0] = 0.0
        u = np.fft.irfft2(1j * ky[:, None] * psi, s=om.shape)[:, :i_hi]
        v = np.fft.irfft2(-1j * kx[None, :] * psi, s=om.shape)[:, :i_hi]
        ud.append(abs(u.mean()) / U)
        vr.append(float(np.sqrt((v ** 2).mean())) / U)
        orr.append(float(np.sqrt((om[:, :i_hi] ** 2).mean())) * D / U)
        v_series.append(float(v.mean()))
    ts = times[idx]
    vs = np.asarray(v_series) - np.mean(v_series)
    st_leak = 0.0
    if len(vs) > 8:
        dt = float(np.median(np.diff(ts)))
        ps = np.abs(np.fft.rfft(vs)) ** 2
        fr = np.fft.rfftfreq(len(vs), d=dt)
        U_m = float(np.mean(U_t[idx])) if U_t is not None else 1.0
        f_st = ST_REF * U_m / D
        band = (fr >= 0.5 * f_st) & (fr <= 2.0 * f_st)
        tot = float(ps[1:].sum())
        st_leak = float(ps[band].sum() / tot) if tot > 0 else 0.0
    m = {'u_deficit': float(np.median(ud)), 'v_rms': float(np.median(vr)),
         'om_rms': float(np.median(orr)), 'st_leak': st_leak,
         'n_frames': int(idx.size)}
    # Sanaa 2026-07-15 21:20: u/v/omega are the binding checks; St-leak is
    # ADVISORY when they pass (St may sit within range of truth).
    m['pass'] = all(m[k] < TOL[k] for k in ('u_deficit', 'v_rms', 'om_rms'))
    m['st_advisory_exceeded'] = bool(st_leak >= TOL['st_leak'])
    return m


def snapshot_plot(run_dir, out_png, title):
    """Last-snapshot omega + inlet-strip panels on a VERY TIGHT color scale
    (Sanaa: triple-sure visual -- residual reflection shows at clim ~1e-3)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = np.load(Path(run_dir) / 'DNS_FR.npz')
    om = z['omega_FR'] if 'omega_FR' in z.files else z['omega']
    om = om[0] if om.ndim == 4 else om
    last = np.asarray(om[-1], dtype=np.float64)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, clim, tag in ((axs[0], 1.0, 'omega, clim +-1'),
                          (axs[1], 1e-3, 'omega, clim +-1e-3 (reflection scale)')):
        im = ax.imshow(last, origin='lower', cmap='seismic',
                       vmin=-clim, vmax=clim, aspect='equal',
                       interpolation='nearest')
        ax.set_title(f'{title}: {tag}', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-root')
    ap.add_argument('--geometry', default='?')
    ap.add_argument('--t-window', nargs=2, type=float, default=[25.0, 35.0])
    ap.add_argument('--out')
    ap.add_argument('--check-run')
    ap.add_argument('--member')
    ap.add_argument('--append')
    args = ap.parse_args()

    if args.check_run:
        try:
            z = np.load(Path(args.check_run) / 'DNS_FR.npz')
            t_end = float(z['times'][-1]); del z
            m = run_metrics(args.check_run, t_end - 10.0, t_end)
            line = (f"{args.member} {'PASS' if m['pass'] else 'FAIL'} "
                    + ' '.join(f'{k}={m[k]:.2e}' for k in TOL))
        except Exception as e:
            line = f'{args.member} FAIL exception={e!r}'
        with open(args.append, 'a') as f:
            f.write(line + '\n')
        print(line, flush=True)
        return

    root = Path(args.sweep_root)
    rows, passing = [], []
    for d in sorted(root.glob('p*')):
        p = float(d.name[1:].replace('p', '.'))
        try:
            m = run_metrics(d, *args.t_window)
        except Exception as e:
            rows.append((p, f'ERROR {e!r}')); continue
        rows.append((p, m))
        if m['pass']:
            passing.append(p)
    pick = None
    for i, p in enumerate(passing):
        if any(abs(q - (p + 0.1)) < 1e-6 for q in passing):
            pick = p + 0.1          # smallest passing WITH passing successor,
            break                   # deployed with the successor as margin
    body = '\n'.join(
        f'p={p}: ' + (r if isinstance(r, str) else
                      ' '.join(f'{k}={r[k]:.2e}' for k in TOL)
                      + (' PASS' if r['pass'] else ' fail'))
        for p, r in rows)
    if pick is not None:
        Path(args.out).write_text(f'{pick}\n')
        # triple-sure snapshots (tight colorbar) -> reports/, pushed by the
        # relay cron; email carries the paths
        rep = Path('/gdata/projects/ml_scope/Closure_modeling/QG-closure/'
                   f'qg-sgs-closure/reports/sponge_sweep_{args.geometry}')
        for p, r in rows:
            if isinstance(r, str) or abs(p - pick) > 0.051:
                continue
            d = root / f"p{str(p).replace('.', 'p')}"
            try:
                snapshot_plot(d, rep / f'last_snapshot_p{p}.png',
                              f'{args.geometry} penalty {p}')
            except Exception as e:
                print(f'[snapshot] {p}: {e!r}', flush=True)
        mail(f'sweep_{args.geometry}',
             f'[QG][MONITOR][sgs-closure] sponge sweep {args.geometry}: '
             f'PICKED penalty {pick}',
             body + f'\n\nPICK: {pick} (smallest passing with passing '
             'successor; deployed value = successor for margin).\n'
             'S2 member reruns fire on the next pipeline tick.')
    elif len([r for _, r in rows if not isinstance(r, str)]) >= 11:
        # no-pick mail only when the FULL sweep has reported (partial early
        # runs stay silent -- Sanaa 21:35 early-exit mode reruns this often)
        mail(f'sweep_{args.geometry}',
             f'[QG][FLAG][sgs-closure] sponge sweep {args.geometry}: '
             'NO penalty in 1.25-2.25 passes',
             body + '\n\nNo out file written -- pipeline holds at S2. '
             'Likely needs a wider sponge RAMP, not more penalty (impedance '
             'mismatch); your ruling.')
    print(body, flush=True)


if __name__ == '__main__':
    main()
