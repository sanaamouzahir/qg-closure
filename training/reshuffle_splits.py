#!/usr/bin/env python
r"""
reshuffle_splits.py
===================
Rewrite split.npz in every sliced member dir with a LEAKAGE-FREE BLOCK-SHUFFLE,
replacing the slicer's chronological 70/15/15 split (which put early-time frames
in train and mid/late-time frames in val/test -- a covariate shift along the
chaotic trajectory that makes val rise while train falls).

Per anchor, frames are grouped into contiguous BLOCKS of size >= n_snapshots
(so no train sample shares an input frame with a val/test sample), the blocks are
permuted with a fixed seed, then partitioned 70/15/15. Same distribution in all
three splits; no adjacent-frame leakage. Reads Nsrc and the anchor count from
manifest.json; does NOT touch the delta mmaps. Re-running is idempotent (fixed
seed) so train/val/test stay disjoint and reproducible.

Index layout matches the slicer exactly: sample (anchor ai, frame f) -> ai*Nsrc+f.
Writes both <member>/split.npz and <member>/packed/split.npz (as the slicer does).

Usage:
  python reshuffle_splits.py ROOT [--frac 0.70 0.15] [--seed 0] [--block N] [--dry-run]
    ROOT : ensemble root (recurses to every dir containing manifest.json+split.npz)
           OR a single member dir.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def member_dirs(root: Path):
    if (root / 'manifest.json').exists() and (root / 'split.npz').exists():
        return [root]
    return sorted({m.parent for m in root.rglob('manifest.json')
                   if (m.parent / 'split.npz').exists()})


def block_split(Nsrc, n_anchors, block, ftrain, fval, seed):
    """Per-anchor block-shuffled 3-way split over global indices ai*Nsrc + f."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    nb = int(np.ceil(Nsrc / block))                 # blocks of contiguous frames
    ntr = int(round(nb * ftrain)); nva = int(round(nb * fval))
    for ai in range(n_anchors):
        order = rng.permutation(nb)                  # shuffle THIS anchor's blocks
        groups = {'tr': order[:ntr], 'va': order[ntr:ntr + nva], 'te': order[ntr + nva:]}
        for key, blocks in groups.items():
            for b in blocks:
                lo = b * block; hi = min(lo + block, Nsrc)
                idx = ai * Nsrc + np.arange(lo, hi)
                (tr if key == 'tr' else va if key == 'va' else te).append(idx)
    cat = lambda L: (np.concatenate(L) if L else np.empty(0, int)).astype(np.int32)
    return cat(tr), cat(va), cat(te)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path)
    ap.add_argument('--frac', type=float, nargs=2, default=(0.70, 0.15),
                    metavar=('TRAIN', 'VAL'), help='train/val fractions (test = rest)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--block', type=int, default=8,
                    help='contiguous-frame block size (>= n_snapshots to avoid '
                         'input-frame leakage between splits). Default 8.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    ftrain, fval = args.frac

    members = member_dirs(args.root)
    if not members:
        raise SystemExit(f"no member dirs (manifest.json + split.npz) under {args.root}")
    print(f"[reshuffle] {len(members)} member(s); block={args.block} seed={args.seed} "
          f"frac=({ftrain},{fval},{round(1-ftrain-fval,3)})\n")

    for md in members:
        man = json.loads((md / 'manifest.json').read_text())
        n_total = int(man['n_total'])
        n_anchors = len(man.get('anchors', [])) or 1
        S = int(man.get('n_snapshots_per_sample', man.get('sliced_S', 1)))
        if n_total % n_anchors:
            print(f"  ! {md.name}: n_total={n_total} not divisible by "
                  f"n_anchors={n_anchors}; SKIP"); continue
        Nsrc = n_total // n_anchors
        if args.block < S:
            print(f"  ! {md.name}: block {args.block} < n_snapshots {S} "
                  f"(possible input-frame leakage); proceeding anyway")
        tr, va, te = block_split(Nsrc, n_anchors, args.block, ftrain, fval, args.seed)

        # sanity: disjoint + covering
        allidx = np.concatenate([tr, va, te])
        assert allidx.size == n_total, (allidx.size, n_total)
        assert np.unique(allidx).size == n_total, "overlap/duplicate indices!"

        old = np.load(md / 'split.npz')
        print(f"  {md.name:24s} Nsrc={Nsrc} x{n_anchors}anch  "
              f"old(tr/va/te)={old['train_idx'].size}/{old['val_idx'].size}/{old['test_idx'].size}"
              f"  ->  new={tr.size}/{va.size}/{te.size}")
        if args.dry_run:
            continue
        for tgt in (md / 'split.npz', md / 'packed' / 'split.npz'):
            if tgt.parent.exists():
                np.savez(tgt, train_idx=tr, val_idx=va, test_idx=te)

    print(f"\n[reshuffle] {'DRY-RUN, nothing written' if args.dry_run else 'done'}.")


if __name__ == '__main__':
    main()
