#!/usr/bin/env python
"""gate1_overlays.py — the two remaining Gate-1 checks (AMENDMENT_01 SC.3):

1. U_inlet-vs-table EXACT overlay, per recorder case: the recorder's
   U_inlet(t_row) must equal the table value the bc assigned at that step,
   U_table[step_row + offset], EXACTLY (float equality, max|diff| == 0.0)
   for one alignment offset in {-1, 0, +1} (reported; expected 0 — bc doc:
   U[n] at step n = round(t/dt), recorder stamps the same step's value).

2. dt-consistency STRICT-IDENTITY overlay on the table pair(s): the ou (and
   telegraph) tables were realized on the DT_MICRO=1.25e-4 micro-grid and
   exactly subsampled to 2.5e-4, so U_coarse[k] == U_fine[2k] bitwise.

Branch-supervisor authored (plotting/log-analysis lane). Prints numbers,
writes PNGs + gate1_overlays_summary.yaml under --outdir.
Usage:
  python gate1_overlays.py --g1-root <SGS_closure_gate1> \
      [--cases const_rec sine ou] [--dt-pairs ou telegraph]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def u_vs_table(g1, case, summary):
    rd = os.path.join(g1, case)
    with open(os.path.join(rd, 'config.yaml')) as fh:
        cfg = yaml.safe_load(fh)
    tab_path = cfg['qg']['bc']['inlet_table']
    z = np.load(os.path.join(rd, 'scalars.npz'), allow_pickle=True)
    tab = np.load(tab_path)
    U_tab = np.asarray(tab['U'], dtype=np.float64)
    t_tab = np.asarray(tab['t'], dtype=np.float64)
    step = np.asarray(z['step'], dtype=np.int64)
    U_rec = np.asarray(z['U_inlet'], dtype=np.float64)
    if U_rec.ndim == 2:
        U_rec = U_rec[:, 0]
    t_rec = np.asarray(z['t'], dtype=np.float64)

    diffs = {}
    for off in (-1, 0, 1):
        idx = step + off
        ok = (idx >= 0) & (idx < U_tab.size)
        diffs[off] = float(np.max(np.abs(U_rec[ok] - U_tab[idx[ok]])))
    exact = [off for off, d in diffs.items() if d == 0.0]
    print(f'== {case}: U_inlet vs table {os.path.basename(tab_path)}')
    print(f'   rows={U_rec.size} max|diff| per offset: ' +
          ', '.join(f'{o:+d}: {d:.3e}' for o, d in sorted(diffs.items())))
    print(f'   EXACT (==0.0) at offset(s): {exact if exact else "NONE"}')

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1]))
    a0.plot(t_tab, U_tab, lw=0.8, label='table U(t)')
    sl = slice(None, None, max(1, U_rec.size // 400))
    a0.plot(t_rec[sl], U_rec[sl], 'o', ms=2.5, mfc='none',
            label='recorder U_inlet (every 10 steps, decimated for plot)')
    a0.set_ylabel('U')
    a0.legend(fontsize=8)
    a0.set_title(f'{case}: U_inlet vs inlet table (max|diff| = '
                 f'{diffs.get(0, np.nan):.1e} at offset 0)')
    idx0 = np.clip(step, 0, U_tab.size - 1)
    a1.semilogy(t_rec, np.maximum(np.abs(U_rec - U_tab[idx0]), 1e-20), lw=0.6)
    a1.set_ylabel('|diff|')
    a1.set_xlabel('t [t.u.]')
    fig.tight_layout()
    out = os.path.join(rd, 'fig_gate1_u_vs_table.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'   figure: {out}')
    summary['u_vs_table'][case] = dict(
        table=tab_path, rows=int(U_rec.size),
        max_abs_diff_by_offset={str(o): d for o, d in diffs.items()},
        exact_offsets=exact)


def dt_pair(g1, stem, summary):
    tdir = os.path.join(g1, 'tables')
    fc = os.path.join(tdir, f'{stem}_dt2p5e-4.npz')
    ff = os.path.join(tdir, f'{stem}_dt1p25e-4.npz')
    zc, zf = np.load(fc), np.load(ff)
    Uc = np.asarray(zc['U'], dtype=np.float64)
    Uf = np.asarray(zf['U'], dtype=np.float64)
    n = min(Uc.size, (Uf.size + 1) // 2)
    d = float(np.max(np.abs(Uc[:n] - Uf[: 2 * n : 2])))
    ident = bool(d == 0.0)
    print(f'== dt-consistency [{stem}]: coarse N={Uc.size} fine N={Uf.size} '
          f'compared n={n}')
    print(f'   max|U_2p5[k] - U_1p25[2k]| = {d:.3e} -> '
          f'{"STRICT IDENTITY" if ident else "NOT identical"}')
    tc = np.asarray(zc['t'], dtype=np.float64)
    tf = np.asarray(zf['t'], dtype=np.float64)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.2))
    a0.plot(tf, Uf, lw=0.7, label='dt=1.25e-4')
    a0.plot(tc[:: max(1, Uc.size // 800)], Uc[:: max(1, Uc.size // 800)],
            'o', ms=2, mfc='none', label='dt=2.5e-4 (decimated)')
    a0.set_xlabel('t')
    a0.set_ylabel('U')
    a0.set_title(f'{stem}: table pair overlay (max|diff|={d:.1e})')
    a0.legend(fontsize=8)
    m = (tf >= 5.0) & (tf <= 6.0)
    a1.plot(tf[m], Uf[m], lw=0.9, label='fine')
    mc = (tc >= 5.0) & (tc <= 6.0)
    a1.plot(tc[mc], Uc[mc], 'o', ms=3, mfc='none', label='coarse')
    a1.set_title('zoom t in [5, 6]')
    a1.set_xlabel('t')
    fig.tight_layout()
    out = os.path.join(tdir, f'fig_gate1_dtconsistency_{stem}.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'   figure: {out}')
    summary['dt_consistency'][stem] = dict(
        coarse=fc, fine=ff, n_compared=int(n), max_abs_diff=d,
        strict_identity=ident)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--g1-root', required=True)
    p.add_argument('--cases', nargs='+', default=['const_rec', 'sine', 'ou'])
    p.add_argument('--dt-pairs', nargs='+', default=['ou', 'telegraph'])
    a = p.parse_args()
    summary = {'u_vs_table': {}, 'dt_consistency': {}}
    for c in a.cases:
        u_vs_table(a.g1_root, c, summary)
    for s in a.dt_pairs:
        dt_pair(a.g1_root, s, summary)
    out = os.path.join(a.g1_root, 'gate1_overlays_summary.yaml')
    with open(out, 'w') as fh:
        yaml.safe_dump(json.loads(json.dumps(summary, default=float)), fh,
                       sort_keys=False)
    print(f'summary: {out}')


if __name__ == '__main__':
    main()
