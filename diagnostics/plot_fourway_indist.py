#!/usr/bin/env python3
"""plot_fourway_indist.py -- the four-way in-distribution ladder verdict, one figure.

Three panels (one per Delta_T), six in-distribution (member, IC) cases on x,
log-scale error-reduction factor over bare AB2CN2 on y. Three arms:
  p1-NN   rollout-stability fine-tuned ckpt (rollout_ft_p1_lam01)
  cond_v2 PURE a-priori conditioned ckpt (deriv7_cond_local_v2)
  true    analytic R3 closure = the ceiling
cond_v2 blow-ups are drawn as X markers in a shaded strip below parity, tagged
with the step (of 16) where the rollout went non-finite. Reads
fourway_indist_table.csv (no recompute).
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = Path('/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning'
           '/diagnostics/Results/apost_indist_condv2_20260714')
TABLE = RES / 'fourway_indist_table.csv'
OUT = RES / 'fourway_indist_verdict.png'

# Okabe-Ito (colorblind-safe by construction)
C_P1 = '#0072B2'    # blue    - stabilized NN
C_CV = '#009E73'    # green   - pure conditioned NN
C_TR = '#555555'    # gray    - analytic ceiling
C_BLOW = '#D55E00'  # vermilion - blow-up marker

CASES = [('kf4', 532), ('kf4', 912), ('kf4', 1356),
         ('256', 549), ('256', 933), ('256', 1357)]
DTS = [0.005, 0.01, 0.015]
BLOW_Y = 1.4e-3     # where the blow-up strip sits (below every real value)


def main():
    rows = list(csv.DictReader(open(TABLE)))
    get = {(r['member'], int(r['ic']), float(r['dT'])): r for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    xs = range(len(CASES))
    labels = [f'{m}\nic{ic}' for m, ic in CASES]

    for ax, dt in zip(axes, DTS):
        ax.set_yscale('log')
        ax.axhline(1.0, color='#999999', lw=1.0, ls='--', zorder=1)
        ax.axhspan(6.0e-4, 2.6e-3, color=C_BLOW, alpha=0.08, zorder=0)
        for x, (m, ic) in zip(xs, CASES):
            r = get[(m, ic, dt)]
            # true ceiling: wide gray dash
            tx = float(r['true_x'])
            ax.plot([x - 0.28, x + 0.28], [tx, tx], color=C_TR, lw=2.2,
                    solid_capstyle='butt', zorder=2)
            # p1-NN
            if r['p1nn_verdict'] == 'STABLE' and r['p1nn_x'] not in ('', 'nan'):
                ax.plot(x - 0.13, float(r['p1nn_x']), 'o', ms=8, color=C_P1,
                        mec='white', mew=0.8, zorder=4)
            else:
                ax.plot(x - 0.13, BLOW_Y, 'X', ms=9, color=C_P1, zorder=4)
            # cond_v2
            if r['condv2_verdict'] == 'STABLE':
                ax.plot(x + 0.13, float(r['condv2_x']), 'o', ms=8, color=C_CV,
                        mec='white', mew=0.8, zorder=4)
            else:
                ax.plot(x + 0.13, BLOW_Y, 'X', ms=9, color=C_BLOW, zorder=4)
                ax.annotate(f"step {r['condv2_blowup']}", (x + 0.13, BLOW_Y),
                            textcoords='offset points', xytext=(0, -13),
                            ha='center', fontsize=7, color=C_BLOW)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f'$\\Delta T$ = {dt:g}', fontsize=11)
        ax.grid(axis='y', color='#dddddd', lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=9)

    axes[0].set_ylabel('error reduction over bare AB2CN2\n'
                       '(final rel-L2, 16-step ladder)  —  log scale', fontsize=10)
    axes[0].set_ylim(6.0e-4, 400)
    axes[0].annotate('parity with bare', (0.02, 1.0), xycoords=('axes fraction', 'data'),
                     fontsize=8, color='#777777', va='bottom')
    axes[1].annotate('blew up (X = non-finite before step 16)',
                     (0.5, 3.2e-3), xycoords=('axes fraction', 'data'),
                     fontsize=8, color=C_BLOW, ha='center')

    handles = [
        plt.Line2D([], [], marker='o', ls='none', ms=8, color=C_P1,
                   mec='white', label='p1-NN (rollout-stability FT)'),
        plt.Line2D([], [], marker='o', ls='none', ms=8, color=C_CV,
                   mec='white', label='cond_v2 (pure conditioned, no FT)'),
        plt.Line2D([], [], color=C_TR, lw=2.2, label='true analytic closure (ceiling)'),
        plt.Line2D([], [], marker='X', ls='none', ms=9, color=C_BLOW,
                   label='cond_v2 blow-up'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Pure conditioning is MORE accurate where it survives, but only '
                 'survives at $\\Delta T$=5e-3 — stabilization is necessary, '
                 'current FT is too blunt',
                 fontsize=11, y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=170, bbox_inches='tight')
    print(f'[plot] wrote {OUT}')


if __name__ == '__main__':
    main()
