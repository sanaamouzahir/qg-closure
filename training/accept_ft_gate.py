#!/usr/bin/env python
"""accept_ft_gate.py -- option-4 acceptance gate (Sanaa GO 2026-07-14):
per-member a-priori Nddot BEFORE vs AFTER a stability fine-tune / clip.

The p1lam01 lesson made quantitative: a-priori Nddot is the rollout floor
(benign amplification, 1:1), so any stabilization arm that degrades it
per-member beyond tolerance is REJECTED regardless of how stable it looks.

Inputs: two eval_by_root CSVs (from eval_deriv_by_root.py, columns
member, Delta_T, Ny, n_samples, Ndot, Nddot, N3dot).

    python accept_ft_gate.py --before <ckpt_dir>/eval_by_root_val.csv \
        --after <ft_dir>/eval_by_root_val.csv --tol-rel 0.10 [--order Nddot]

Verdict tiers (Sanaa calibration 2026-07-18 -- binary FAIL was too extreme):
  PASS             no row beyond tolerance, median within tolerance
  PASS-conditional median within tolerance AND every over-tol row is
                   past-wall (dT > member ΔT★, the modified-equation
                   convergence radius -- known-hard rows by theory)
  REGRESSED        anything else
Exit codes: 0 PASS / 2 PASS-conditional / 3 REGRESSED (FINALIZE watchers
that test rc==0 treat conditional as needs-a-human, by design).
ΔT★ defaults: documented radii (Re25k .066, combo .139, kf4 .199);
override/extend with --dtstar-csv (columns member,dtstar). Members with
unknown ΔT★ are conservatively treated as NOT past-wall.
Prints the full per-row table with over-tol rows flagged and past-wall
rows tagged.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load(path, order):
    rows = {}
    with open(path) as f:
        r = csv.DictReader(f)
        if order not in r.fieldnames:
            raise SystemExit(f"{path}: no column {order!r} (have {r.fieldnames})")
        for row in r:
            key = (row['member'], float(row['Delta_T']))
            rows[key] = float(row[order])
    if not rows:
        raise SystemExit(f"{path}: empty")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', type=Path, required=True)
    ap.add_argument('--after', type=Path, required=True)
    ap.add_argument('--order', default='Nddot',
                    help='column to gate on (default Nddot = the rollout floor)')
    ap.add_argument('--tol-rel', type=float, default=0.10,
                    help='max allowed relative degradation per row (default 10%%)')
    ap.add_argument('--dtstar-csv', type=Path, default=None,
                    help='optional CSV (member,dtstar) overriding/extending the '
                         'built-in convergence radii for the past-wall tag')
    args = ap.parse_args()

    # Documented ΔT★ (modified-equation convergence radii, C=2.08 τ_eddy fit).
    dtstar = {'FRC-Re25k': 0.066, 'FRC-combo': 0.139, 'FRC-kf4': 0.199}
    if args.dtstar_csv is not None:
        with open(args.dtstar_csv) as f:
            for row in csv.DictReader(f):
                dtstar[row['member']] = float(row['dtstar'])

    b = load(args.before, args.order)
    a = load(args.after, args.order)
    common = sorted(set(b) & set(a))
    if not common:
        raise SystemExit('no common (member, Delta_T) rows')
    missing = sorted(set(b) - set(a))
    if missing:
        print(f"[gate] WARNING: {len(missing)} rows in before but not after: "
              f"{missing[:5]}")

    fails = []          # (key, ratio, past_wall)
    n_improved = 0
    print(f"{'member':<12}{'dT':>8}  {args.order}: before -> after   (ratio)")
    for key in common:
        ratio = a[key] / max(b[key], 1e-30)
        past_wall = key[0] in dtstar and key[1] > dtstar[key[0]]
        if ratio < 1.0:
            n_improved += 1
        flag = '  [>dT*]' if past_wall else ''
        if ratio > 1.0 + args.tol_rel:
            flag = '  << over-tol' + (' (past-wall)' if past_wall else '')
            fails.append((key, ratio, past_wall))
        print(f"{key[0]:<12}{key[1]:>8.4g}  {b[key]:.4f} -> {a[key]:.4f}"
              f"   ({ratio:.2f}x){flag}")

    med_b = sorted(b[k] for k in common)[len(common) // 2]
    med_a = sorted(a[k] for k in common)[len(common) // 2]
    known = ', '.join(f"{m} {d:g}" for m, d in sorted(dtstar.items()))
    print(f"\n[gate] median {args.order}: {med_b:.4f} -> {med_a:.4f} "
          f"({med_a / max(med_b, 1e-30):.2f}x)   tol +{args.tol_rel:.0%}/row")
    print(f"[gate] rows improved: {n_improved}/{len(common)}; "
          f"over-tol: {len(fails)}   (dT* used: {known})")

    med_ok = med_a <= med_b * (1.0 + args.tol_rel)
    if not fails and med_ok:
        print("[gate] VERDICT: PASS (accuracy held within tolerance on every row)")
        sys.exit(0)
    if med_ok and all(pw for _, _, pw in fails):
        print(f"[gate] VERDICT: PASS-conditional ({len(fails)} over-tol row(s), "
              f"ALL past-wall dT>dT*; median within tolerance)")
        sys.exit(2)
    n_core = sum(1 for _, _, pw in fails if not pw)
    print(f"[gate] VERDICT: REGRESSED ({len(fails)} row(s) beyond tolerance, "
          f"{n_core} in-validity; median {med_a / max(med_b, 1e-30):.2f}x)")
    sys.exit(3)


if __name__ == '__main__':
    main()
