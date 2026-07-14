"""merge_indist_trueclosure_table.py -- three-way in-distribution table
(Sanaa order 2026-07-14): per (member, IC, dT) merge

  bare final relL2      from the true-closure ladder (apost_indist_trueclosure_20260714)
  NN final relL2        from the EXISTING p1lam01 ladder CSV (NOT rerun)
  true final relL2      from the true-closure ladder (r3anal arm)

and the ratios NN/bare (improvement_x of the NN), true/bare (the analytic
promise on the same grid), and gap = NN_relL2 / true_relL2 (how far the NN
is from the promise). Also cross-checks the two bare legs (same refs, same
stepper -- must agree to ~1e-3 rel).

Usage (from diagnostics/):
  python merge_indist_trueclosure_table.py \
      --true-csv Results/apost_indist_trueclosure_20260714/ladder_matrix_summary_ALL.csv \
      --nn-csv   Results/apost_p1lam01_20260714/ladder_matrix_summary_ALL.csv \
      --out      Results/apost_indist_trueclosure_20260714/threeway_indist_table.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PAIRS = [('kf4', '532'), ('kf4', '912'), ('kf4', '1356'),
         ('256', '549'), ('256', '933'), ('256', '1357')]
DTS = ['0.005', '0.01', '0.015']


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def key_of(row):
    m = re.search(r'(?:^|_)(kf4|256|combo)_ic(\d+)_', row['case'])
    if not m:
        raise SystemExit(f"cannot parse case name: {row['case']}")
    return m.group(1), m.group(2), f"{float(row['Delta_T']):g}"


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float('nan')
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--true-csv', type=Path, required=True)
    ap.add_argument('--nn-csv', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    tc = {key_of(r): r for r in load(args.true_csv)}
    nn = {key_of(r): r for r in load(args.nn_csv)}

    rows = []
    for dt in DTS:
        for member, ic in PAIRS:
            k = (member, ic, f'{float(dt):g}')
            rt, rn = tc.get(k), nn.get(k)
            if rt is None or rn is None:
                raise SystemExit(f'missing row for {k}: true={rt is not None} nn={rn is not None}')
            fb = float(rt['final_relL2_bare']) if rt['final_relL2_bare'] else float('nan')
            ft = float(rt['final_relL2_closure']) if rt['final_relL2_closure'] else float('nan')
            fn = float(rn['final_relL2_closure']) if rn['final_relL2_closure'] else float('nan')
            fb_nn = float(rn['final_relL2_bare']) if rn['final_relL2_bare'] else float('nan')
            bare_mismatch = abs(fb - fb_nn) / fb if fb == fb and fb_nn == fb_nn and fb else float('nan')
            rows.append(dict(
                member=member, ic=ic, Delta_T=k[2],
                final_relL2_bare=f'{fb:.4e}',
                final_relL2_nn=f'{fn:.4e}',
                final_relL2_true=f'{ft:.4e}',
                nn_over_bare_x=f'{fb / fn:.2f}' if fn == fn and fn else '',
                true_over_bare_x=f'{fb / ft:.2f}' if ft == ft and ft else '',
                gap_nn_over_true=f'{fn / ft:.1f}' if fn == fn and ft == ft and ft else '',
                verdict_true=rt['verdict'], verdict_nn=rn['verdict'],
                bare_crosscheck_rel=f'{bare_mismatch:.2e}' if bare_mismatch == bare_mismatch else ''))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[merge] wrote {args.out}')

    # per-dT medians over stable pairs
    print('\nPER-dT MEDIANS (stable pairs only):')
    print(f"{'dT':>7} {'NN x':>8} {'TRUE x':>9} {'gap NN/true':>12} {'delivered %':>12}")
    for dt in DTS:
        sel = [r for r in rows if r['Delta_T'] == f'{float(dt):g}'
               and r['verdict_true'] == 'STABLE' and r['verdict_nn'] == 'STABLE'
               and r['nn_over_bare_x'] and r['true_over_bare_x']]
        nx = median([float(r['nn_over_bare_x']) for r in sel])
        tx = median([float(r['true_over_bare_x']) for r in sel])
        gap = median([float(r['gap_nn_over_true']) for r in sel])
        print(f'{float(dt):>7g} {nx:>8.2f} {tx:>9.1f} {gap:>12.1f} {100*nx/tx:>11.1f}%'
              f'   (n={len(sel)})')


if __name__ == '__main__':
    main()
