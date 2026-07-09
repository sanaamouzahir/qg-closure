#!/usr/bin/env python
"""st_verdict_gate1.py — development check behind the Gate-1 St question
(branch-supervisor authored: log-analysis/diagnostic-support lane).

For each given scalars.npz: C_L(t) amplitude envelope (|analytic signal| of
band-passed Cl_mid, same band as shedding_tracker) split into thirds of the
analysis window, half-window rms, observed cycle count vs the quasi-steady
expectation St_ref*U_med/D, and the Welch resolution df = 1/T_window.
Prints numbers only (+ one Cl(t)+envelope PNG per case next to the input).
Usage: python st_verdict_gate1.py <scalars.npz> [...] [--t-min 5.0]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shedding_tracker import bandpass  # sibling import (flat dir)

ST_REF = 0.21
BAND = (0.15, 0.55)


def analyze_one(path, t_min):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z['meta']))
    D = float(meta['length'])
    t = np.asarray(z['t'], dtype=np.float64)
    cl = np.asarray(z['Cl_mid'], dtype=np.float64)
    u_in = np.asarray(z['U_inlet'], dtype=np.float64)
    if cl.ndim == 2:
        cl, u_in = cl[:, 0], u_in[:, 0]
    win = t >= t_min
    t, cl, u_in = t[win], cl[win], u_in[win]
    n = t.size
    T_win = float(t[-1] - t[0])
    U_med = float(np.nanmedian(u_in))

    f_qs = ST_REF * U_med / D                 # quasi-steady expected line
    exp_cycles = f_qs * T_win
    df_welch = 1.0 / T_win

    x = cl - np.nanmean(cl)
    x_bp = bandpass(x, float(np.median(np.diff(t))), *BAND)
    an = hilbert(x_bp)
    env = np.abs(an)
    phi = np.unwrap(np.angle(an))
    obs_cycles = float((phi[-1] - phi[0]) / (2 * np.pi))

    thirds = np.array_split(np.arange(n), 3)
    env_thirds = [float(np.median(env[ix])) for ix in thirds]
    halves = np.array_split(np.arange(n), 2)
    rms_halves = [float(np.sqrt(np.nanmean(x[ix] ** 2))) for ix in halves]

    name = os.path.basename(os.path.dirname(path))
    print(f'===== {name} =====')
    print(f'  window [{t[0]:.3f}, {t[-1]:.3f}] T_win={T_win:.3f} t.u. '
          f'= {U_med * T_win / D:.1f} convective D/U units '
          f'(t_wait {t_min:.1f} = {U_med * t_min / D:.1f} conv. units)')
    print(f'  U_med={U_med:.4f} D={D:.6f} -> f_qs={f_qs:.4f} '
          f'(T_qs={1/f_qs:.3f}) expected cycles in window: {exp_cycles:.2f}')
    print(f'  observed cycles (Hilbert phase advance): {obs_cycles:.2f}')
    print(f'  Welch resolution df=1/T_win={df_welch:.3f} '
          f'(gap 0.21-0.11 in St = {0.10 * U_med / D:.3f} in f, '
          f'= {0.10 * U_med / D / df_welch:.1f} bins)')
    print(f'  Cl envelope medians by thirds: '
          f'{env_thirds[0]:.4g} -> {env_thirds[1]:.4g} -> {env_thirds[2]:.4g} '
          f'(last/first = {env_thirds[2] / max(env_thirds[0], 1e-300):.2f})')
    print(f'  Cl rms halves: {rms_halves[0]:.4g} -> {rms_halves[1]:.4g} '
          f'(2nd/1st = {rms_halves[1] / max(rms_halves[0], 1e-300):.2f})')
    grow = env_thirds[2] > 1.2 * env_thirds[1] > 0
    print(f'  VERDICT bit: envelope still growing at window end: '
          f'{"YES" if grow else "no (see thirds)"}')

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, x, lw=0.6, label=r'$C_L$ (mid, demeaned)')
    ax.plot(t, env, lw=1.4, label='envelope |analytic|')
    ax.plot(t, -env, lw=1.4, color=ax.lines[-1].get_color())
    ax.set_xlabel('t [t.u.]')
    ax.set_ylabel(r'$C_L$')
    ax.set_title(f'{name}: lift amplitude development')
    ax.legend(loc='upper left', fontsize=8)
    out = os.path.join(os.path.dirname(path), 'shedding',
                       'fig_cl_envelope.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  figure: {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('scalars', nargs='+')
    p.add_argument('--t-min', type=float, default=5.0)
    a = p.parse_args()
    for s in a.scalars:
        analyze_one(s, a.t_min)


if __name__ == '__main__':
    main()
