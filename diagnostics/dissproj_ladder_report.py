"""dissproj_ladder_report.py -- verdict table + figures for the P0
dissipative-projection A/B ladder (Sanaa 2026-07-13 plan, gate G3).

Reads the consolidated case npzs under
Results/apost_dissproj_20260713/<member>_ic<IC>/case_*.npz
(variant 'full' = proj OFF, 'full_dissproj' = proj ON; horizon in the
ckpt label suffix _h64/_h128, else 16-step) and writes:

  <dir>/dissproj_AB_table.csv          every case, one row (merged CSV)
  <dir>/dissproj_verdict.txt           the three PRE-REGISTERED bars verdict
  <pngs>/f1_ab_final_relL2.png         off-vs-on final rel-L2 per dT
  <pngs>/f2_proj_activity.png          projected shells per step, per dT
  <pngs>/f3_horizon_curves.png         5e-3 h64/h128 error curves (med draws)

PRE-REGISTERED BARS (verbatim from the P0 spec):
  proj-on -> 1e-2 10/10 stable, 1.5e-2 improves, accuracy cost < 5% at 5e-3.

Usage:
  python dissproj_ladder_report.py --dir <Results/apost_dissproj_20260713> \
      --pngs <diagnostics/pngs/dissipative_projection_ladder>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_cases(root: Path):
    rows = []
    for f in sorted(root.glob('*_ic*/case_*.npz')):
        z = np.load(f, allow_pickle=False)
        cfg = json.loads(str(z['config_json']))
        member = Path(cfg['root_dir']).parent.name
        name = f.stem                      # case_<label>_<variant>_<dt>
        on = '_dissproj_' in name or name.endswith('_dissproj')
        horizon = int(z['M'])
        rows.append(dict(
            file=f, member=member, ic=int(cfg['ic_index']),
            dT=float(z['Delta_T']), horizon=horizon, proj_on=on,
            verdict=str(z['verdict']), blowup=int(z['blowup_step']),
            fin_bare=float(z['final_relL2_bare']),
            fin_clos=float(z['final_relL2_closure']),
            imp=float(z['improvement_x']),
            t=np.asarray(z['t']), relL2_bare=np.asarray(z['relL2_bare']),
            relL2_clos=np.asarray(z['relL2_closure']),
            proj_counts=(np.asarray(z['proj_shell_count'])
                         if 'proj_shell_count' in z else None),
            proj_frac=(np.asarray(z['proj_removed_frac'])
                       if 'proj_removed_frac' in z else None)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--pngs', type=Path, required=True)
    args = ap.parse_args()
    rows = load_cases(args.dir)
    if not rows:
        raise SystemExit(f'[report] no case npzs under {args.dir}')
    args.pngs.mkdir(parents=True, exist_ok=True)

    # ---- merged CSV ---- #
    csv_path = args.dir / 'dissproj_AB_table.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['member', 'ic', 'dT', 'horizon', 'proj', 'verdict',
                    'blowup_step', 'final_relL2_bare', 'final_relL2_closure',
                    'improvement_x', 'proj_shells_mean', 'proj_shells_max',
                    'proj_removed_frac_mean'])
        for r in sorted(rows, key=lambda r: (r['dT'], r['horizon'],
                                             r['member'], r['ic'],
                                             r['proj_on'])):
            pc = r['proj_counts']
            w.writerow([r['member'], r['ic'], r['dT'], r['horizon'],
                        'on' if r['proj_on'] else 'off', r['verdict'],
                        r['blowup'],
                        f"{r['fin_bare']:.6e}",
                        '' if not np.isfinite(r['fin_clos'])
                        else f"{r['fin_clos']:.6e}",
                        '' if not np.isfinite(r['imp'])
                        else f"{r['imp']:.2f}",
                        '' if pc is None else f'{pc.mean():.1f}',
                        '' if pc is None else int(pc.max()),
                        '' if r['proj_frac'] is None
                        else f"{r['proj_frac'].mean():.3e}"])
    print(f'[report] merged table -> {csv_path} ({len(rows)} cases)')

    def sel(dT=None, horizon=None, on=None):
        out = [r for r in rows
               if (dT is None or abs(r['dT'] - dT) < 1e-12)
               and (horizon is None or r['horizon'] == horizon)
               and (on is None or r['proj_on'] == on)]
        return sorted(out, key=lambda r: (r['member'], r['ic']))

    lines = ['PRE-REGISTERED BARS (verbatim from the P0 spec): proj-on -> '
             '1e-2 10/10 stable, 1.5e-2 improves, accuracy cost < 5% at '
             '5e-3.', '']

    # BAR-1: 1e-2, 16 steps, ON: all stable
    on10 = sel(1e-2, 16, True)
    n_st = sum(r['verdict'] == 'STABLE' for r in on10)
    b1 = n_st == len(on10) and len(on10) == 10
    lines.append(f'BAR-1  1e-2 proj-on stable: {n_st}/{len(on10)} '
                 f'(need 10/10)  -> {"PASS" if b1 else "FAIL"}')
    off10 = sel(1e-2, 16, False)
    lines.append(f'       (context: proj-off stable '
                 f'{sum(r["verdict"] == "STABLE" for r in off10)}'
                 f'/{len(off10)})')

    # BAR-2: 1.5e-2, 16 steps: ON improves over OFF
    on15, off15 = sel(1.5e-2, 16, True), sel(1.5e-2, 16, False)
    ns_on = sum(r['verdict'] == 'STABLE' for r in on15)
    ns_off = sum(r['verdict'] == 'STABLE' for r in off15)
    later, worse = 0, 0
    for a, b in zip(off15, on15):
        assert (a['member'], a['ic']) == (b['member'], b['ic'])
        sa = a['blowup'] if a['blowup'] >= 0 else 10 ** 9
        sb = b['blowup'] if b['blowup'] >= 0 else 10 ** 9
        later += sb > sa
        worse += sb < sa
    b2 = (ns_on > ns_off) or (ns_on == ns_off and later > 0 and worse == 0)
    lines.append(f'BAR-2  1.5e-2 proj-on improves: stable {ns_off}->{ns_on}, '
                 f'blowup later in {later}, earlier in {worse} of '
                 f'{len(on15)} draws -> {"PASS" if b2 else "FAIL"}')

    # BAR-3: 5e-3, 16 steps: accuracy cost < 5%
    on5, off5 = sel(5e-3, 16, True), sel(5e-3, 16, False)
    costs = []
    for a, b in zip(off5, on5):
        assert (a['member'], a['ic']) == (b['member'], b['ic'])
        if np.isfinite(a['fin_clos']) and np.isfinite(b['fin_clos']) \
                and a['fin_clos'] > 0:
            costs.append(b['fin_clos'] / a['fin_clos'] - 1.0)
    med_cost = float(np.median(costs)) if costs else float('nan')
    max_cost = float(np.max(costs)) if costs else float('nan')
    b3 = np.isfinite(med_cost) and med_cost < 0.05
    lines.append(f'BAR-3  5e-3 accuracy cost: median {med_cost:+.2%}, '
                 f'max {max_cost:+.2%} over {len(costs)} draws '
                 f'(need median < +5%) -> {"PASS" if b3 else "FAIL"}')
    lines.append('')

    # per-dT A/B table (16-step)
    lines.append(f'{"dT":>8} {"proj":>5} {"stable":>7} {"med finL2":>12} '
                 f'{"med imp_x":>10} {"med shells/step":>16} '
                 f'{"med rm-frac":>12}')
    for dT in (5e-3, 1e-2, 1.5e-2):
        for on in (False, True):
            g = sel(dT, 16, on)
            if not g:
                continue
            fins = [r['fin_clos'] for r in g if np.isfinite(r['fin_clos'])]
            imps = [r['imp'] for r in g if np.isfinite(r['imp'])]
            sh = [r['proj_counts'].mean() for r in g
                  if r['proj_counts'] is not None]
            fr = [r['proj_frac'].mean() for r in g
                  if r['proj_frac'] is not None]
            lines.append(
                f'{dT:>8g} {"on" if on else "off":>5} '
                f'{sum(r["verdict"] == "STABLE" for r in g):>4}/{len(g):<2} '
                f'{np.median(fins) if fins else float("nan"):>12.4e} '
                f'{np.median(imps) if imps else float("nan"):>10.2f} '
                f'{np.median(sh) if sh else float("nan"):>16.1f} '
                f'{np.median(fr) if fr else float("nan"):>12.3e}')
    lines.append('')

    # horizon outcome (5e-3, 64/128)
    for H in (64, 128):
        for on in (False, True):
            g = sel(5e-3, H, on)
            if not g:
                continue
            fins = [r['fin_clos'] for r in g if np.isfinite(r['fin_clos'])]
            lines.append(
                f'HORIZON h={H:>3} 5e-3 proj-{"on " if on else "off"}: '
                f'stable {sum(r["verdict"] == "STABLE" for r in g)}'
                f'/{len(g)}  med final relL2 '
                f'{np.median(fins) if fins else float("nan"):.4e}  '
                f'med imp_x {np.median([r["imp"] for r in g if np.isfinite(r["imp"])] or [float("nan")]):.2f}')
    lines.append('')
    lines.append(f'OVERALL: {"ALL BARS PASS" if (b1 and b2 and b3) else "BAR(S) FAILED"}'
                 f'  [B1={b1} B2={b2} B3={b3}]')
    vt = args.dir / 'dissproj_verdict.txt'
    vt.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'[report] verdict -> {vt}')

    # ---- figures ---- #
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # f1: off-vs-on final rel-L2 (16-step), one marker set per dT
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sent = None
    for dT, c, mk in ((5e-3, 'C0', 'o'), (1e-2, 'C1', 's'),
                      (1.5e-2, 'C3', '^')):
        off, on = sel(dT, 16, False), sel(dT, 16, True)
        xs, ys = [], []
        for a, b in zip(off, on):
            x = a['fin_clos'] if np.isfinite(a['fin_clos']) else np.nan
            y = b['fin_clos'] if np.isfinite(b['fin_clos']) else np.nan
            xs.append(x); ys.append(y)
        xs, ys = np.asarray(xs), np.asarray(ys)
        fin = np.isfinite(xs) & np.isfinite(ys)
        ax.loglog(xs[fin], ys[fin], mk, color=c, ms=7, alpha=0.85,
                  label=f'dT={dT:g}')
        blo = ~np.isfinite(xs) & np.isfinite(ys)   # off blew, on finished
        if blo.any():
            sent = np.nanmax(np.concatenate([xs[fin], ys[fin]])) \
                if fin.any() else 1.0
            ax.loglog(np.full(blo.sum(), sent * 3), ys[blo], mk,
                      color=c, ms=9, mfc='none', mew=2)
    lo, hi = ax.get_xlim()
    ax.loglog([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
    if sent is not None:
        ax.axvline(sent * 3, color='k', lw=0.6, ls=':', alpha=0.6)
        ax.text(sent * 3, ax.get_ylim()[0] * 1.3, ' off BLEW UP',
                rotation=90, fontsize=7, va='bottom')
    ax.set_xlabel('final rel-L2, projection OFF')
    ax.set_ylabel('final rel-L2, projection ON')
    ax.set_title('dissipative projection A/B, 16 steps, 10 draws\n'
                 '(open markers at dotted line: OFF arm blew up)')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.pngs / 'f1_ab_final_relL2.png', dpi=130)
    plt.close(fig)

    # f2: projected shells per step (ON arms), median +- IQR across draws
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), sharey=True)
    panels = [(5e-3, 16, 'dT=5e-3 (16)'), (1e-2, 16, 'dT=1e-2 (16)'),
              (1.5e-2, 16, 'dT=1.5e-2 (16)'), (5e-3, 128, 'dT=5e-3 (128)')]
    for ax, (dT, H, ttl) in zip(axes, panels):
        g = [r for r in sel(dT, H, True) if r['proj_counts'] is not None]
        if g:
            L = max(len(r['proj_counts']) for r in g)
            M_ = np.full((len(g), L), np.nan)
            for i, r in enumerate(g):
                M_[i, :len(r['proj_counts'])] = r['proj_counts']
            med = np.nanmedian(M_, 0)
            q1, q3 = np.nanpercentile(M_, 25, 0), np.nanpercentile(M_, 75, 0)
            st = np.arange(1, L + 1)
            ax.plot(st, med, '-', color='C0', lw=1.5, label='median')
            ax.fill_between(st, q1, q3, color='C0', alpha=0.25,
                            label='IQR (draws)')
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('coarse step')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('shells projected / step')
    axes[0].legend(fontsize=7)
    fig.suptitle('dissipative projection activity (quiet at 5e-3, busy at '
                 '1.5e-2?)', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.pngs / 'f2_proj_activity.png', dpi=130)
    plt.close(fig)

    # f3: 5e-3 horizon error curves (median across draws)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, H in zip(axes, (64, 128)):
        for on, c, lab in ((False, 'C1', 'closure, proj off'),
                           (True, 'C0', 'closure, proj on')):
            g = sel(5e-3, H, on)
            if not g:
                continue
            tt = g[0]['t']
            C = np.full((len(g), len(tt)), np.nan)
            for i, r in enumerate(g):
                n = min(len(r['relL2_clos']), len(tt))
                C[i, :n] = r['relL2_clos'][:n]
            ax.semilogy(tt, np.nanmedian(C, 0), '-', color=c, label=lab)
        g = sel(5e-3, H, False)
        if g:
            tt = g[0]['t']
            B = np.full((len(g), len(tt)), np.nan)
            for i, r in enumerate(g):
                n = min(len(r['relL2_bare']), len(tt))
                B[i, :n] = r['relL2_bare'][:n]
            ax.semilogy(tt, np.nanmedian(B, 0), 'k--', lw=1.0, label='bare')
        ax.set_title(f'{H} coarse steps @ dT=5e-3', fontsize=9)
        ax.set_xlabel('t')
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('rel-L2 vs RK4 truth (median over draws)')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.pngs / 'f3_horizon_curves.png', dpi=130)
    plt.close(fig)
    print(f'[report] figures -> {args.pngs}')


if __name__ == '__main__':
    main()
