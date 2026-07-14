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

Verdict: PASS if after <= before * (1 + tol_rel) for EVERY (member, dt) row
(and overall median). Exit code 0 PASS / 3 FAIL (scriptable in FINALIZE
monitors). Prints the full per-row table with the worst offenders flagged.
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
    args = ap.parse_args()

    b = load(args.before, args.order)
    a = load(args.after, args.order)
    common = sorted(set(b) & set(a))
    if not common:
        raise SystemExit('no common (member, Delta_T) rows')
    missing = sorted(set(b) - set(a))
    if missing:
        print(f"[gate] WARNING: {len(missing)} rows in before but not after: "
              f"{missing[:5]}")

    fails = []
    print(f"{'member':<12}{'dT':>8}  {args.order}: before -> after   (ratio)")
    for key in common:
        ratio = a[key] / max(b[key], 1e-30)
        flag = ''
        if ratio > 1.0 + args.tol_rel:
            flag = '  << FAIL'
            fails.append((key, b[key], a[key], ratio))
        print(f"{key[0]:<12}{key[1]:>8.4g}  {b[key]:.4f} -> {a[key]:.4f}"
              f"   ({ratio:.2f}x){flag}")

    med_b = sorted(b[k] for k in common)[len(common) // 2]
    med_a = sorted(a[k] for k in common)[len(common) // 2]
    print(f"\n[gate] median {args.order}: {med_b:.4f} -> {med_a:.4f} "
          f"({med_a / max(med_b, 1e-30):.2f}x)   tol +{args.tol_rel:.0%}/row")

    if fails or med_a > med_b * (1.0 + args.tol_rel):
        print(f"[gate] VERDICT: FAIL ({len(fails)} row(s) beyond tolerance)")
        sys.exit(3)
    print("[gate] VERDICT: PASS (accuracy held within tolerance on every row)")


if __name__ == '__main__':
    main()
