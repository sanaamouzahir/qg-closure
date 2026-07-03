#!/usr/bin/env python
r"""
diagnose_sliced_inputs.py -- localize the corruption: sliced data vs eval plumbing.

Two checks per sliced sweep dir:

A. SMOOTHNESS of the sliced 7-stacks, separately for the omega block (ch 0..6)
   and the psi block (ch 7..13). Same Delta^k logic as the deep probe. If deep
   marks are clean but sliced stacks are rough -> slicer corrupted the data.
   If omega clean but psi rough -> the psi block is the culprit.

B. BYTE-COMPARE sliced rows against the deep marks. For each tested sample,
   search the deep inputs for an exact (or near) match of sliced channel 0 and
   verify all 7 omega channels equal deep marks at a constant stride j.
   exact-match => slicer is a pure strided copy (data innocent, blame eval).
   mismatch    => slicer transformed/recomputed something (data guilty).

Usage (from $QG_DIR/training):
    python diagnose_sliced_inputs.py \
        --sliced data/ensemble_N5_7lag/FRC-256/sweep_dT_5em3 \
        --deep   data/ensemble_N5_7lag/FRC-256/forced_turbulence_dT_5em3 \
        --n-samples 4
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def dk_table(stack: np.ndarray, base: float, kmax: int = 6):
    out = []
    d = stack.astype(np.float64)
    for k in range(1, kmax + 1):
        d = np.diff(d, axis=0)
        out.append(np.linalg.norm(d, axis=(-2, -1)).mean() / base)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sliced', type=Path, required=True)
    ap.add_argument('--deep', type=Path, required=True)
    ap.add_argument('--n-samples', type=int, default=4)
    args = ap.parse_args()

    sman = json.loads((args.sliced / 'manifest.json').read_text())
    dman = json.loads((args.deep / 'manifest.json').read_text())
    S = int(sman['n_snapshots_per_sample'])
    M = int(dman['n_snapshots_per_sample'])
    dt_t = float(sman['Delta_T']); dt_f = float(dman['Delta_T'])
    j = int(round(dt_t / dt_f))
    na = int(sman.get('n_anchors', 0))
    sl = np.load(args.sliced / 'packed' / 'inputs.npy', mmap_mode='r')
    dp = np.load(args.deep / 'packed' / 'inputs.npy', mmap_mode='r')
    print(f"[sliced] {sl.shape} {sl.dtype}  S={S} dt={dt_t}  n_anchors={na}")
    print(f"[deep]   {dp.shape} {dp.dtype}  M={M} dt_fine={dt_f}  stride j={j}")

    N = sl.shape[0]
    picks = np.linspace(1 if na else 0, N - 1, args.n_samples, dtype=int)

    print("\n== A. smoothness of sliced stacks (Delta^k / ||field||) ==")
    hdr = '  '.join(f"k{k}" for k in range(1, 7))
    for s in picks:
        om = np.asarray(sl[s, :S], dtype=np.float64)
        ps = np.asarray(sl[s, S:2 * S], dtype=np.float64)
        b_om = np.linalg.norm(om[0]); b_ps = np.linalg.norm(ps[0])
        t_om = dk_table(om, b_om); t_ps = dk_table(ps, b_ps)
        print(f"  s={s:5d} omega: " + '  '.join(f"{v:.2e}" for v in t_om))
        print(f"          psi  : " + '  '.join(f"{v:.2e}" for v in t_ps))

    print("\n== B. byte-compare sliced vs deep (omega block) ==")
    for s in picks:
        om0 = np.asarray(sl[s, 0])
        w = s // na if na else None
        found = None
        # try the expected window first, then neighbors, then scan
        cand = ([w - 1, w, w + 1] if w is not None else []) + \
               list(range(0, dp.shape[0], max(1, dp.shape[0] // 50)))
        for wc in cand:
            if wc is None or wc < 0 or wc >= dp.shape[0]:
                continue
            for a0 in range(M - (S - 1) * j):
                if np.array_equal(om0, dp[wc, a0]):
                    found = (wc, a0); break
            if found:
                break
        if not found:
            # near-match search on the expected window
            diffs = []
            if w is not None and w < dp.shape[0]:
                for a0 in range(M - (S - 1) * j):
                    diffs.append((np.abs(np.asarray(dp[w, a0], np.float64)
                                         - om0.astype(np.float64)).max(), a0))
                dmin, a0 = min(diffs)
                print(f"  s={s:5d}: NO exact match. Closest deep (win {w}, mark {a0}) "
                      f"maxdiff={dmin:.3e}  -> SLICER TRANSFORMED THE DATA")
            else:
                print(f"  s={s:5d}: no exact match found in scan")
            continue
        wc, a0 = found
        ok = all(np.array_equal(np.asarray(sl[s, m]), dp[wc, a0 + m * j])
                 for m in range(S))
        # psi block: deep psi channels are M..2M-1
        okp = all(np.array_equal(np.asarray(sl[s, S + m]), dp[wc, M + a0 + m * j])
                  for m in range(S))
        print(f"  s={s:5d}: deep win={wc} mark0={a0} stride={j}  "
              f"omega exact-copy={ok}  psi exact-copy={okp}"
              + ("" if (ok and okp) else "  -> SLICER TRANSFORMED THE DATA"))

    print("\nInterpretation:")
    print("  A rough + B mismatch  -> slicer corrupted the inputs (data bug).")
    print("  A clean + B exact     -> data innocent; the error is in eval/model "
          "plumbing (next probe: hand-FD one sample vs its f64 target).")
    print("  omega clean, psi rough-> psi block is the culprit.")


if __name__ == '__main__':
    main()
