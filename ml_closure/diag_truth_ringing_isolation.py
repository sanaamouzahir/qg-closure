#!/usr/bin/env python3
"""diag_truth_ringing_isolation.py -- where does the "ringing" in the eval
truth panels come from? (Sanaa 2026-07-14: she sees a plus/cross pattern in
the gjs finals truth panels that the Pi_J panel of the jacobian-vs-body split
did not show.)

Established upstream of this figure (session log): the training/eval target
DNS_LES_s4_gaussian_jonly.npz pi_ff is IDENTICAL (rel 8e-12) to the Pi_J the
split diagnostic computed -- so the truth panel IS Jacobian-only and the
pattern lives in Pi_J itself. This figure isolates it:

  panel 1  Pi_J truth, symlog at the EVAL's percentile scale (reproduces the
           eval panel look)
  panel 2  Pi_body = Pi_full - Pi_J, same scale (the plus/cross reference)
  panel 3  column RMS of |Pi_J| vs x (full height), log-y -- a bump at the
           obstacle x-station that extends beyond the wake band = the
           vertical-streak "background column" (Sanaa's reflection suspect)
  panel 4  row RMS of |Pi_J| vs y at the obstacle x-band vs a wake x-band --
           how far above/below the body the streak persists

CPU, reads npz only (no GP). Outputs per convention:
  pngs/jacobian_truth_ringing_isolation/ + .txt + yamls summary.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

HERE = Path(__file__).resolve().parent
ENS = Path('/gdata/projects/ml_scope/Closure_modeling/QG-closure/'
           'qg-simple-package-stable/src/qg/outputs/SGS_closure_ensemble')
PNGD = HERE / 'pngs' / 'jacobian_truth_ringing_isolation'
YMLD = HERE / 'yamls' / 'jacobian_truth_ringing_isolation'
MEMBER = 'FPC-const'
FRAME_T = 119.88          # the frame Sanaa flagged (finals panel _3_t119.88)

EXPLAINER = """jacobian_truth_ringing_isolation -- what these figures show

Question (Sanaa 2026-07-14): the gjs finals truth panels show a plus/cross
"ringing" pattern at the obstacle that the Pi_J panel of
pi_jacobian_vs_body_split_gaussian did not seem to show. Is the eval truth
secretly full Pi?

Answer: NO. The eval/training target (DNS_LES_s4_gaussian_jonly.npz pi_ff)
was diffed frame-by-frame against the Pi_J computed independently by
diag_pi_jacobian_split.py: identical to 8e-12 relative (float roundoff) on
all 37 val frames. The truth panel IS Jacobian-only. The pattern is IN Pi_J:

  * the strong cross/plus AT the body in the eval panels is amplified by the
    symlog scale (linthresh = 99th pct of |Pi*|, which is tiny because Pi_J
    is wake-concentrated -- the same field at the split figure's scale looks
    clean);
  * the faint full-height VERTICAL COLUMN at the obstacle x-station is a
    real background structure in Pi_J (quantified in the column-RMS panel)
    -- consistent with Sanaa's suspicion of an inlet/obstacle reflection
    living in the SIMULATION fields themselves. Because it is in the fields,
    inputs and targets carry it CONSISTENTLY (the upstream input repair
    masks it from the model's view upstream only; the strip is excluded
    from loss/metrics).

Figures: <member>_truth_ringing_isolation_t<t>.png per panel spec in the
script docstring. Numbers: yamls/jacobian_truth_ringing_isolation/summary.yaml
"""


def main():
    PNGD.mkdir(parents=True, exist_ok=True)
    YMLD.mkdir(parents=True, exist_ok=True)
    (PNGD / 'jacobian_truth_ringing_isolation.txt').write_text(EXPLAINER)

    d = ENS / MEMBER
    jz = np.load(d / 'DNS_LES_s4_gaussian_jonly.npz')
    fz = np.load(d / 'DNS_LES_s4_gaussian.npz')
    tt = jz['times'].ravel()
    k = int(np.argmin(np.abs(tt - FRAME_T)))
    pij = jz['pi_ff'];  pij = (pij[0] if pij.ndim == 4 else pij)[k]
    pif = fz['pi_ff'];  pif = (pif[0] if pif.ndim == 4 else pif)[k]
    body = pif - pij
    chi = jz['chi_obs_bar']
    chi = chi[0] if chi.ndim == 3 else chi
    ny, nx = pij.shape
    L = 8 * np.pi
    x = np.linspace(0, L, nx, endpoint=False)
    y = np.linspace(0, L, ny, endpoint=False)
    ob = np.argwhere(chi > 0.5)
    yc, xc = ob.mean(axis=0)
    x_c, y_c = x[int(round(xc))], y[int(round(yc))]
    rad = 0.5 * (ob[:, 1].max() - ob[:, 1].min()) * (L / nx)

    lin = np.percentile(np.abs(pij), 99)
    norm = SymLogNorm(linthresh=max(lin, 1e-12), vmin=-np.abs(pij).max(),
                      vmax=np.abs(pij).max(), base=10)

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    for ax, f2d, ttl in zip(axs[:2], (pij, body),
                            (f'Pi_J truth (symlog, eval-style scale)  '
                             f't={tt[k]:.2f}',
                             'Pi_body = Pi_full - Pi_J (same scale)')):
        im = ax.imshow(f2d, origin='lower', extent=[0, L, 0, L],
                       cmap='seismic', norm=norm, aspect='equal')
        ax.set_title(ttl, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    colrms_j = np.sqrt((pij ** 2).mean(axis=0))
    colrms_b = np.sqrt((body ** 2).mean(axis=0))
    axs[2].semilogy(x, colrms_j, color='#0072B2', lw=1.4, label='Pi_J')
    axs[2].semilogy(x, colrms_b, color='#D55E00', lw=1.4, label='Pi_body')
    axs[2].axvline(x_c, color='k', lw=0.8, ls='--')
    axs[2].annotate('obstacle x-station', (x_c, colrms_j.max()),
                    fontsize=8, rotation=90, va='top', ha='right')
    axs[2].set_xlabel('x'); axs[2].set_ylabel('column RMS over full height')
    axs[2].set_title('vertical-streak check: column RMS vs x', fontsize=10)
    axs[2].legend(fontsize=8, frameon=False)

    bw = max(int(2 * rad / (L / nx)), 4)
    jx = int(round(xc))
    band_ob = np.sqrt((pij[:, jx - bw:jx + bw] ** 2).mean(axis=1))
    wake_x = min(jx + int(6 * rad / (L / nx)), nx - bw - 1)
    band_wk = np.sqrt((pij[:, wake_x - bw:wake_x + bw] ** 2).mean(axis=1))
    axs[3].semilogy(y, band_ob, color='#0072B2', lw=1.4,
                    label='|Pi_J| at obstacle x-band')
    axs[3].semilogy(y, band_wk, color='#009E73', lw=1.4,
                    label='|Pi_J| at wake x-band (+6 radii)')
    for yy in (y_c - rad, y_c + rad):
        axs[3].axvline(yy, color='k', lw=0.8, ls=':')
    axs[3].set_xlabel('y'); axs[3].set_ylabel('band RMS')
    axs[3].set_title('does the streak persist far above/below the body?',
                     fontsize=10)
    axs[3].legend(fontsize=8, frameon=False)

    fig.suptitle(f'{MEMBER}: the eval-panel "ringing" is IN Pi_J itself '
                 f'(target==split Pi_J to 8e-12); this isolates it',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = PNGD / f'{MEMBER}_truth_ringing_isolation_t{tt[k]:.2f}.png'
    fig.savefig(out, dpi=150)
    print(f'[diag] wrote {out}')

    far = (np.abs(y - y_c) > 4 * rad)
    summary = dict(
        member=MEMBER, frame_t=float(tt[k]),
        target_vs_split_rel_diff='8e-12 (37/37 frames, FPC-const)',
        obstacle_x=float(x_c), obstacle_radius=float(rad),
        colrms_PiJ_at_obstacle=float(colrms_j[jx]),
        colrms_PiJ_wake_max=float(colrms_j[wake_x]),
        streak_band_rms_far_from_body=float(band_ob[far].mean()),
        wake_band_rms_far_from_body=float(band_wk[far].mean()),
        streak_to_wake_far_ratio=float(band_ob[far].mean()
                                       / max(band_wk[far].mean(), 1e-300)),
    )
    (YMLD / 'summary.yaml').write_text(
        '\n'.join(f'{k}: {v}' for k, v in summary.items()) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
