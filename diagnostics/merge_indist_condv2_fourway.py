#!/usr/bin/env python3
"""merge_indist_condv2_fourway.py -- four-way in-distribution ladder table.

Combines, per (member, IC, dT), the M=16 a-posteriori ladder outcomes of:
  bare        : baseline AB2CN2 at Delta_T (identical across all runs)
  p1-NN       : rollout-stability-FT ckpt rollout_ft_p1_lam01 (existing CSV)
  cond_v2     : PURE a-priori conditioned ckpt deriv7_cond_local_v2 (this job)
  true        : full analytic R3 closure (session-14c ceiling)

Error-reduction ratio per arm = final_relL2_bare / final_relL2_<arm>.
Reports medians by dT over the STABLE cond_v2 draws, and flags cond_v2 blowups
(the stability cost we are quantifying). No recompute -- reads three CSVs.
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path
from statistics import median

WT = Path('/gdata/projects/ml_scope/Closure_modeling/QG-closure/qg-wiener-conditioning')
RES = WT / 'diagnostics' / 'Results'
CSVS = {
    'p1nn':   RES / 'apost_p1lam01_20260714' / 'ladder_matrix_summary_ALL.csv',
    'condv2': RES / 'apost_indist_condv2_20260714' / 'ladder_matrix_summary_ALL.csv',
    'true':   RES / 'apost_indist_trueclosure_20260714' / 'ladder_matrix_summary_ALL.csv',
}
OUT = RES / 'apost_indist_condv2_20260714' / 'fourway_indist_table.csv'

# the 6 in-distribution pairs
PAIRS = [('kf4', 532), ('kf4', 912), ('kf4', 1356),
         ('256', 549), ('256', 933), ('256', 1357)]
DTS = [0.005, 0.01, 0.015]

CASE_RE = re.compile(r'_(kf4|256)_ic(\d+)_full_(0p\d+)$')


def dt_of(tok: str) -> float:
    return float(tok.replace('0p', '0.'))


def load(path: Path) -> dict:
    """key (member, ic, dt) -> dict(bare, closure, improvement_x, verdict, blowup_step)."""
    if not path.exists():
        raise SystemExit(f'[merge] FATAL missing input CSV: {path}')
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            m = CASE_RE.search(row['case'])
            if not m:
                continue
            member, ic, dttok = m.group(1), int(m.group(2)), m.group(3)
            key = (member, ic, dt_of(dttok))
            bare = float(row['final_relL2_bare'] or 'nan')
            clos = float(row['final_relL2_closure'] or 'nan')
            out[key] = dict(bare=bare, closure=clos,
                            verdict=row.get('verdict', ''),
                            blowup_step=row.get('blowup_step', ''))
    return out


def ratio(bare: float, arm: float) -> float:
    return bare / arm if arm and arm == arm and arm > 0 else float('nan')


def main():
    D = {k: load(p) for k, p in CSVS.items()}

    rows = []
    for member, ic in PAIRS:
        for dt in DTS:
            key = (member, ic, dt)
            r = {'member': member, 'ic': ic, 'dT': dt}
            # bare: prefer cond_v2's copy (crosscheck vs the others below)
            bares = {k: D[k][key]['bare'] for k in D if key in D[k]}
            if not bares:
                print(f'[merge] WARN no data for {key}', file=sys.stderr)
                continue
            bare = bares.get('condv2', next(iter(bares.values())))
            r['bare_relL2'] = bare
            # crosscheck bare identity
            r['bare_crosscheck_max'] = max(abs(v - bare) for v in bares.values())
            for arm in ('p1nn', 'condv2', 'true'):
                if key in D[arm]:
                    c = D[arm][key]['closure']
                    r[f'{arm}_relL2'] = c
                    r[f'{arm}_x'] = ratio(bare, c)
                    r[f'{arm}_verdict'] = D[arm][key]['verdict']
                    r[f'{arm}_blowup'] = D[arm][key]['blowup_step']
                else:
                    r[f'{arm}_relL2'] = float('nan')
                    r[f'{arm}_x'] = float('nan')
                    r[f'{arm}_verdict'] = 'MISSING'
                    r[f'{arm}_blowup'] = ''
            rows.append(r)

    # write full table
    cols = ['member', 'ic', 'dT', 'bare_relL2', 'bare_crosscheck_max',
            'p1nn_relL2', 'p1nn_x', 'p1nn_verdict', 'p1nn_blowup',
            'condv2_relL2', 'condv2_x', 'condv2_verdict', 'condv2_blowup',
            'true_relL2', 'true_x', 'true_verdict', 'true_blowup']
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    print(f'[merge] wrote {OUT}  ({len(rows)} rows)')

    # ---- medians by dT ------------------------------------------------------
    def stable(r, arm):
        v = r.get(f'{arm}_verdict', '')
        x = r.get(f'{arm}_x', float('nan'))
        return v == 'STABLE' and x == x

    print('\n============ FOUR-WAY MEDIANS BY dT (error-reduction x over bare) ============')
    print(f'{"dT":>7} | {"n":>3} | {"p1-NN":>18} | {"cond_v2":>26} | {"true":>10}')
    print('-' * 78)
    for dt in DTS:
        sub = [r for r in rows if r['dT'] == dt]
        # cond_v2: split stable / blown
        c_stable = [r for r in sub if stable(r, 'condv2')]
        c_blown = [r for r in sub if not stable(r, 'condv2')]
        p_stable = [r for r in sub if stable(r, 'p1nn')]
        t_all = [r for r in sub if r.get('true_x') == r.get('true_x')]
        med = lambda lst, arm: (median([r[f'{arm}_x'] for r in lst])
                                if lst else float('nan'))
        # cond_v2 median over its OWN stable draws + p1-NN median over the SAME draws
        c_keys = {(r['member'], r['ic']) for r in c_stable}
        p_on_cstable = [r for r in p_stable if (r['member'], r['ic']) in c_keys]
        print(f'{dt:>7.3f} | {len(sub):>3} | '
              f'{med(p_stable,"p1nn"):>7.2f}x (n={len(p_stable)})  | '
              f'{med(c_stable,"condv2"):>7.2f}x (n_st={len(c_stable)}, '
              f'blown={len(c_blown)}) | '
              f'{med(t_all,"true"):>7.1f}x')
    print('\n============ cond_v2 vs p1-NN on cond_v2-STABLE draws (paired) ============')
    for dt in DTS:
        sub = [r for r in rows if r['dT'] == dt]
        paired = [r for r in sub if stable(r, 'condv2') and stable(r, 'p1nn')]
        if not paired:
            print(f'{dt:>7.3f} | no paired-stable draws')
            continue
        cmed = median([r['condv2_x'] for r in paired])
        pmed = median([r['p1nn_x'] for r in paired])
        tmed = median([r['true_x'] for r in paired])
        print(f'{dt:>7.3f} | paired n={len(paired)}: '
              f'cond_v2 {cmed:.2f}x  vs  p1-NN {pmed:.2f}x  '
              f'(cond_v2/p1nn = {cmed/pmed:.2f}x)  | true {tmed:.1f}x  '
              f'| cond_v2 delivers {100*cmed/tmed:.1f}% of ceiling')

    print('\n============ cond_v2 BLOWUPS (the stability cost) ============')
    any_blow = False
    for r in rows:
        if not stable(r, 'condv2'):
            any_blow = True
            print(f'  {r["member"]}_ic{r["ic"]} @ dT={r["dT"]}: '
                  f'verdict={r.get("condv2_verdict")} '
                  f'blowup_step={r.get("condv2_blowup")}  '
                  f'(p1-NN here: {r.get("p1nn_verdict")}, '
                  f'x={r.get("p1nn_x"):.2f})')
    if not any_blow:
        print('  NONE -- cond_v2 stable on all 18 draws.')


if __name__ == '__main__':
    main()
