#!/usr/bin/env python
"""
verify_target_consistency.py
============================

Closure formula:
    e_total = -coef * (f_anal + f_NN_target)  +  O(DT^4)

with:
    coef        = DT^3 * (1 - 1/K^2)
    f_anal      = (1/12) * [L^3 omega + L^2 N]
    f_NN_target = (1/12) * [L*N_dot - 5*N_ddot]

So define a RECONSTRUCTED empirical bracket directly from saved fields:
    f_recon = -e_total/coef - f_anal

and compare against the saved analytical f_NN_target.

If they agree (rel diff ~ DT), the build is correct.
If they don't, we know exactly what's wrong by inspecting which side has
unphysical magnitudes.

Run:
    python verify_target_consistency.py \\
        --sample /gdata/.../decaying_turbulence_dT_1em3_fixD_v2/samples/sample_000000.npz \\
        --Delta-T 1e-3 --K 100
"""
import argparse
from pathlib import Path
import numpy as np


def rms(arr):
    a = np.asarray(arr).astype(np.float64).ravel()
    return float(np.sqrt(np.mean(a ** 2)))


def stats(arr):
    a = np.asarray(arr).astype(np.float64)
    return f"RMS={rms(a):.4e}  |.|max={np.abs(a).max():.4e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=Path, required=True)
    p.add_argument('--Delta-T', dest='dT', type=float, default=1e-3)
    p.add_argument('--K', type=int, default=100)
    args = p.parse_args()

    d = np.load(args.sample)
    print(f"=== Sample: {args.sample.name} ===")
    print(f"DT = {args.dT:.0e}, K = {args.K}")
    coef = (args.dT ** 3) * (1.0 - 1.0 / (args.K ** 2))
    print(f"coef = DT^3 * (1 - 1/K^2) = {coef:.6e}")
    print()

    # Saved fields, all promoted to float64 for the comparison
    e_total     = d['e_total'].astype(np.float64)
    e_anal_incr = d['e_anal_incr'].astype(np.float64)
    e_NN_incr   = d['e_NN_incr'].astype(np.float64)
    f_anal      = d['f_anal'].astype(np.float64)
    f_NN_target = d['f_NN_target'].astype(np.float64)
    f_NN_target_from_e = d['f_NN_target_from_e'].astype(np.float64)

    print("Saved fields:")
    print(f"  e_total            : {stats(e_total)}")
    print(f"  e_anal_incr        : {stats(e_anal_incr)}")
    print(f"  e_NN_incr          : {stats(e_NN_incr)}")
    print(f"  f_anal             : {stats(f_anal)}")
    print(f"  f_NN_target  (anal): {stats(f_NN_target)}")
    print(f"  f_NN_target_from_e : {stats(f_NN_target_from_e)}")
    print()

    # ------------------------------------------------------------------
    # CHECK 1: Does e_anal_incr satisfy e_anal_incr = -coef * f_anal?
    # ------------------------------------------------------------------
    print("CHECK 1:  e_anal_incr  =?=  -coef * f_anal")
    e_anal_predicted = -coef * f_anal
    diff1 = e_anal_incr - e_anal_predicted
    rel1 = rms(diff1) / max(rms(e_anal_incr), 1e-30)
    print(f"  RMS(e_anal_incr)              = {rms(e_anal_incr):.4e}")
    print(f"  RMS(-coef * f_anal)           = {rms(e_anal_predicted):.4e}")
    print(f"  RMS(diff)                     = {rms(diff1):.4e}")
    print(f"  rel diff                      = {rel1:.4e}")
    print(f"  -> {'CONSISTENT' if rel1 < 1e-3 else 'INCONSISTENT (e_anal_incr != -coef * f_anal)'}")
    print()

    # ------------------------------------------------------------------
    # CHECK 2: Does e_NN_incr = e_total - e_anal_incr (saved)?
    # ------------------------------------------------------------------
    print("CHECK 2:  e_NN_incr  =?=  e_total - e_anal_incr")
    e_NN_predicted = e_total - e_anal_incr
    diff2 = e_NN_incr - e_NN_predicted
    rel2 = rms(diff2) / max(rms(e_NN_incr), 1e-30)
    print(f"  RMS(e_NN_incr saved)          = {rms(e_NN_incr):.4e}")
    print(f"  RMS(e_total - e_anal_incr)    = {rms(e_NN_predicted):.4e}")
    print(f"  RMS(diff)                     = {rms(diff2):.4e}")
    print(f"  rel diff                      = {rel2:.4e}")
    print(f"  -> {'CONSISTENT' if rel2 < 1e-3 else 'INCONSISTENT'}")
    print()

    # ------------------------------------------------------------------
    # CHECK 3: Does f_NN_target_from_e = -e_NN_incr / coef (saved)?
    # ------------------------------------------------------------------
    print("CHECK 3:  f_NN_target_from_e  =?=  -e_NN_incr / coef")
    f_pred = -e_NN_incr / coef
    diff3 = f_NN_target_from_e - f_pred
    rel3 = rms(diff3) / max(rms(f_NN_target_from_e), 1e-30)
    print(f"  RMS(f_NN_target_from_e)       = {rms(f_NN_target_from_e):.4e}")
    print(f"  RMS(-e_NN_incr / coef)        = {rms(f_pred):.4e}")
    print(f"  RMS(diff)                     = {rms(diff3):.4e}")
    print(f"  rel diff                      = {rel3:.4e}")
    print(f"  -> {'CONSISTENT' if rel3 < 1e-3 else 'INCONSISTENT'}")
    print()

    # ------------------------------------------------------------------
    # MAIN CHECK: f_NN_target  =?=  f_NN_target_from_e (modulo O(DT))
    # ------------------------------------------------------------------
    print("MAIN CHECK:  f_NN_target  =?=  f_NN_target_from_e")
    diff_main = f_NN_target_from_e - f_NN_target
    rel_main = rms(diff_main) / max(rms(f_NN_target), 1e-30)
    print(f"  RMS(f_NN_target  ANALYTICAL)  = {rms(f_NN_target):.4e}")
    print(f"  RMS(f_NN_target_from_e EMPIR) = {rms(f_NN_target_from_e):.4e}")
    print(f"  RMS(diff)                     = {rms(diff_main):.4e}")
    print(f"  rel diff                      = {rel_main:.4e}")
    print(f"  expected ~ DT                 = {args.dT:.0e}")
    print()

    if rel_main < 0.1:
        print(">> VERDICT: empirical and analytical brackets agree.")
        print("            Build is correct. Train on f_NN_target safely.")
    elif rel_main > 100:
        print(">> VERDICT: empirical and analytical brackets DISAGREE by huge factor.")
        print("            Need to find out which one is wrong.")
        print()
        print("Diagnostic: check whether e_total alone makes sense at this DT.")
        print(f"  Theory: |e_total|_RMS ~ coef * |f_anal + f_NN_target|_RMS")
        rhs = coef * rms(f_anal + f_NN_target)
        print(f"          predicted ~ {coef:.3e} * {rms(f_anal+f_NN_target):.4e} = {rhs:.4e}")
        print(f"  Actual  |e_total|_RMS = {rms(e_total):.4e}")
        print(f"  ratio actual / predicted = {rms(e_total)/max(rhs,1e-30):.3e}")
        print()
        if rms(e_total) / max(rhs, 1e-30) > 100:
            print("    -> e_total is much larger than physics predicts.")
            print("       Likely cause: float32 rounding noise.")
            print("       e_total ~ 1e-9 in magnitude is below float32 precision,")
            print("       so the saved e_total field is ~1e-7 random noise instead.")
            print()
            print("    SUGGESTED FIX: change `cpu32` -> `cpu64` for these fields:")
            print("       e_total, e_anal_incr, e_NN_incr, f_NN_target_from_e")
            print("       (f_NN_target itself is fine -- it's O(1).)")
        else:
            print("    -> e_total is at the right magnitude, but f_NN_target_from_e")
            print("       is still inflated. Look at coef computation or")
            print("       e_anal_incr sign convention.")
    else:
        print(f">> VERDICT: rel diff {rel_main:.2e} -- larger than expected O(DT).")
        print("            Investigate.")


if __name__ == '__main__':
    main()
