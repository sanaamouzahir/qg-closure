#!/usr/bin/env python
r"""
resplit_by_window.py  --  rewrite split.npz of sliced sweep_dT_* dirs to split BY
DEEP WINDOW instead of per-sample. Fixes within-window temporal leakage WITHOUT
re-slicing (reads only the existing packed inputs shape + manifest).

The problem it fixes
--------------------
slice_deriv_from_deep.py lays samples out window-major, anchor-minor:
    row = window * n_anchors + anchor,   anchors ~ dt_fine (5e-3) apart in time.
Its write_split does rng.permutation(N) -- a per-SAMPLE shuffle. That is NOT
time-contiguous (good), but it scatters adjacent anchors of the SAME window (near-
duplicates, ~5e-3 apart, overlapping lag stacks) across train/val/test -> val
leakage, over-optimistic val.

The fix
-------
Assign each WHOLE window (all its anchors) to one split; shuffle WINDOWS. Result is
shuffled (not contiguous) AND leak-free (train/val are different deep windows,
~0.14 apart in time ~ a few eddy turnovers -> decorrelated). Anchors of a window
never straddle the split, so no near-duplicate leaks across train/val.

Run from $QG_DIR/training AFTER slicing completes. Idempotent; backs up the old
split to split_persample.npz once.

Usage:
    python resplit_by_window.py \
        --sweeps data/ensemble_N5_7lag_sliced/FRC-*/sweep_dT_* \
        --train-frac 0.70 --val-frac 0.15 --seed 0
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def resplit(sweep: Path, train_frac: float, val_frac: float, seed: int):
    man = json.loads((sweep / 'manifest.json').read_text())
    n_anchors = int(man.get('n_anchors', 0))
    inp = np.load(sweep / 'packed' / 'inputs.npy', mmap_mode='r')
    N = inp.shape[0]
    if n_anchors <= 0 or N % n_anchors != 0:
        # fall back: can't recover window structure -> refuse rather than mis-split
        print(f"  [{sweep.parent.name}/{sweep.name}] SKIP: n_anchors={n_anchors} "
              f"does not divide N={N} (can't recover windows); leave split as-is.")
        return
    Nwin = N // n_anchors

    rng = np.random.default_rng(seed)
    win_perm = rng.permutation(Nwin)                    # shuffle WINDOWS
    n_tr_w = int(round(train_frac * Nwin))
    n_va_w = int(round(val_frac * Nwin))
    tr_w = win_perm[:n_tr_w]
    va_w = win_perm[n_tr_w:n_tr_w + n_va_w]
    te_w = win_perm[n_tr_w + n_va_w:]

    def rows(win_ids):
        # all anchor rows of each window: row = w*n_anchors + a
        return np.concatenate([w * n_anchors + np.arange(n_anchors)
                               for w in win_ids]).astype(np.int32) if len(win_ids) \
               else np.array([], np.int32)

    train_idx, val_idx, test_idx = rows(tr_w), rows(va_w), rows(te_w)
    assert len(train_idx) + len(val_idx) + len(test_idx) == N
    # no window appears in two splits -> no anchor leakage
    assert len(set(tr_w.tolist()) & set(va_w.tolist())) == 0

    # back up the per-sample split once, then overwrite
    old = sweep / 'split.npz'
    bak = sweep / 'split_persample.npz'
    if old.exists() and not bak.exists():
        np.savez(bak, **{k: np.load(old)[k] for k in np.load(old).files})
    np.savez(old, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"  [{sweep.parent.name}/{sweep.name}] Nwin={Nwin} n_anchors={n_anchors} "
          f"N={N} -> windows train/val/test={len(tr_w)}/{len(va_w)}/{len(te_w)}  "
          f"rows={len(train_idx)}/{len(val_idx)}/{len(test_idx)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweeps', nargs='+', required=True, type=Path,
                   help='sliced sweep_dT_* dirs (glob-expanded by the shell)')
    p.add_argument('--train-frac', type=float, default=0.70)
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    sweeps = [s for s in args.sweeps
              if (s / 'manifest.json').exists() and (s / 'packed' / 'inputs.npy').exists()]
    if not sweeps:
        raise SystemExit("no valid sliced sweep dirs (need manifest.json + packed/inputs.npy)")
    print(f"[resplit] by-window split over {len(sweeps)} sweep dir(s), "
          f"seed={args.seed}, fracs {args.train_frac}/{args.val_frac}/"
          f"{round(1-args.train_frac-args.val_frac,3)}")
    for s in sweeps:
        resplit(s, args.train_frac, args.val_frac, args.seed)
    print("[resplit] done. (old per-sample split saved as split_persample.npz)")


if __name__ == '__main__':
    main()
