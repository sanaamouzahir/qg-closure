#!/usr/bin/env python
r"""
compute_vs_dt.py
================
Theoretical cost-to-solve curves: total compute to integrate a fixed horizon T
as a function of the coarse step Delta_T.

  cost(Delta_T) = c_scheme * (T / Delta_T)        [units: bare AB2CN2 steps]

per-step cost factors c_scheme (RHS/FFT-bundle evaluations per step):
  AB2CN2 = 1, AB4CN2 = 1 (AB reuses stored N; 1 RHS/step),
  RK4 = 4 (4 stages), AB2CN2+NN = 2.6 (measured ~2.6x bare step).
The two NN curves (match RK4 / match exact) COINCIDE: same network, different
training target, identical inference cost.

TRUTH ceiling is FLAT: truth = RK4 at an ultrafine dt_uf, run at dt_uf regardless
of the coarse Delta_T, so its cost = 4 * (T / dt_uf) = const.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--T', type=float, default=10.0, help='integration horizon')
    p.add_argument('--dt-uf', type=float, default=1e-6, help='truth (RK4) ultrafine step')
    p.add_argument('--nn-cost', type=float, default=2.6, help='AB2CN2+NN per-step factor')
    p.add_argument('--dt-lo', type=float, default=1e-4)
    p.add_argument('--dt-hi', type=float, default=1e-1)
    p.add_argument('--op-dt', type=float, default=1e-2, help='operating Delta_T to annotate')
    p.add_argument('--out', type=str, default='compute_vs_dt.png')
    args = p.parse_args()

    T, dt_uf, cNN = args.T, args.dt_uf, args.nn_cost
    dT = np.logspace(np.log10(args.dt_lo), np.log10(args.dt_hi), 200)

    # per-step cost factors (units: one bare AB2CN2 step)
    cost = {
        'AB2CN2':         1.0 * T / dT,
        'AB4CN2':         1.0 * T / dT,
        'RK4':            4.0 * T / dT,
        'AB2CN2+NN1':     cNN * T / dT,
        'AB2CN2+NN2':     cNN * T / dT,
    }
    truth = 4.0 * T / dt_uf                       # flat ceiling

    STYLE = {                                     # color, linestyle, lw, z
        'AB2CN2':     ('C0', '-',  2.4, 3),
        'AB4CN2':     ('C1', '--', 2.0, 2),       # overlaps AB2CN2 (same cost)
        'RK4':        ('k',  '-',  2.2, 3),
        'AB2CN2+NN1': ('C3', '-',  2.8, 5),
        'AB2CN2+NN2': ('C2', (0, (1, 2)), 2.4, 6),  # dotted, on top of NN1 (coincide)
    }
    LABEL = {
        'AB2CN2':     r'$\mathrm{AB2CN2}$  ($1\times$/step)',
        'AB4CN2':     r'$\mathrm{AB4CN2}$  ($1\times$/step, $\approx$AB2CN2)',
        'RK4':        r'$\mathrm{RK4}$  ($4\times$/step)',
        'AB2CN2+NN1': rf'$\mathrm{{AB2CN2}}+E_{{NN1}}$ (match RK4)  (${cNN:g}\times$/step)',
        'AB2CN2+NN2': rf'$\mathrm{{AB2CN2}}+E_{{NN2}}$ (match exact)  (${cNN:g}\times$/step)',
    }

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for name, (c, ls, lw, z) in STYLE.items():
        ax.plot(dT, cost[name], ls=ls, color=c, lw=lw, zorder=z, label=LABEL[name])

    # truth ceiling (flat)
    ax.axhline(truth, color='0.25', ls='-.', lw=2.0, zorder=1,
               label=rf'truth $=\mathrm{{RK4}}(\delta t={dt_uf:g})$ ceiling '
                     rf'($4T/\delta t$)')

    # operating Delta_T marker + speedup annotation (NN vs truth)
    od = args.op_dt
    c_nn_op = cNN * T / od
    ax.axvline(od, color='0.6', ls=':', lw=1.4, zorder=1)
    ax.scatter([od], [c_nn_op], s=55, color='C3', zorder=7, edgecolor='k', linewidth=0.6)
    speedup = truth / c_nn_op
    ax.annotate(rf'NN @ $\Delta T={od:g}$: {c_nn_op:.0f} steps'
                + '\n' + rf'$\Rightarrow {speedup:,.0f}\times$ below truth',
                xy=(od, c_nn_op), xytext=(od*1.15, c_nn_op*40),
                fontsize=9.5, color='C3',
                arrowprops=dict(arrowstyle='->', color='C3', lw=1.3))
    # bracket the NN<->truth gap
    ax.annotate('', xy=(od, truth), xytext=(od, c_nn_op),
                arrowprops=dict(arrowstyle='<->', color='0.4', lw=1.1, ls=':'))

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'coarse step $\Delta T$', fontsize=13)
    ax.set_ylabel(rf'compute to integrate $T={T:g}$  '
                  r'(units: bare AB2CN2 steps)', fontsize=13)
    ax.set_title(r'Cost to solve vs.\ step size  '
                 r'(theoretical; per-step factors $\times\,T/\Delta T$)', fontsize=12)
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.95)
    ax.set_xlim(args.dt_lo, args.dt_hi)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(args.out.rsplit('.', 1)[0] + '.' + ext, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # console summary at the operating point
    print(f"horizon T={T}, dt_uf={dt_uf}, NN per-step={cNN}x, operating dT={od}")
    for name in cost:
        print(f"  {name:14s} @dT={od:g}: {(STYLE[name] and 0) or ''}"
              f"{ {'AB2CN2':1.,'AB4CN2':1.,'RK4':4.,'AB2CN2+NN1':cNN,'AB2CN2+NN2':cNN}[name]*T/od:>12.1f} steps")
    print(f"  {'truth':14s}        : {truth:>12.1f} steps  (flat ceiling)")
    print(f"  speedup NN vs truth @ dT={od:g}: {speedup:,.0f}x")


if __name__ == '__main__':
    main()
