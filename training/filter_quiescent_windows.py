#!/usr/bin/env python
r"""
filter_quiescent_windows.py -- drop spin-up / quasi-zonal windows from the splits.

Those windows have near-zero true Jacobian (J(psi,omega)~0 for zonal states), so
their N-derivative targets are ~1e4-1e5x smaller than developed-flow windows,
and the per-sample relative loss/metric explodes on them (and poisons training:
the optimizer's cheapest move is shrinking all predictions toward zero).

Criterion (per WINDOW, anchors pooled): drop the window if
    median_over_anchors ||N3dot_target||  <  frac * member_median
(default frac=1e-2; the observed separation is ~1e-4, so this is generous).
Also drops any window whose stack is frozen: mean ||Delta^2 omega||/||omega||
below --rough-min (default 1e-5; healthy ~1e-3, frozen ~7e-7).

Rewrites split.npz (backing up to split_prefilter.npz once). Keeps the
by-window structure intact -- it only REMOVES whole windows from each split.

Usage (from $QG_DIR/training):
    python filter_quiescent_windows.py \
        --sweeps data/ensemble_N5_7lag/FRC-*/sweep_dT_* \
        --frac 1e-2 --rough-min 1e-5
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def process(sweep: Path, frac: float, rough_min: float, dry: bool):
    man = json.loads((sweep / 'manifest.json').read_text())
    S = int(man['n_snapshots_per_sample'])
    na = int(man.get('n_anchors', 1))
    tgt = np.load(sweep / 'packed' / 'deriv_anal_f64.npy', mmap_mode='r')
    inp = np.load(sweep / 'packed' / 'inputs.npy', mmap_mode='r')
    N = tgt.shape[0]
    if N % na:
        print(f"  [{sweep.parent.name}/{sweep.name}] N={N} not divisible by "
              f"n_anchors={na}; SKIP")
        return
    W = N // na

    # per-window stats (median over anchors): N3dot target norm + stack roughness
    t3 = np.empty(W); rg = np.empty(W)
    for w in range(W):
        rows = slice(w * na, (w + 1) * na)
        t3[w] = np.median([np.linalg.norm(tgt[r, 2]) for r in range(w * na, (w + 1) * na)])
        om = np.asarray(inp[w * na, :S], np.float64)      # roughness from one anchor
        d2 = np.diff(om, n=2, axis=0)
        rg[w] = np.linalg.norm(d2, axis=(1, 2)).mean() / np.linalg.norm(om[0])

    med = np.median(t3)
    bad = (t3 < frac * med) | (rg < rough_min)
    bad_w = np.where(bad)[0]
    print(f"  [{sweep.parent.name}/{sweep.name}] windows={W}  member-median "
          f"||tN3||={med:.3e}  dropping {bad.sum()} windows "
          f"({100*bad.mean():.1f}%): {bad_w[:20].tolist()}"
          + (' ...' if len(bad_w) > 20 else ''))
    if dry:
        return

    bad_rows = set()
    for w in bad_w:
        bad_rows.update(range(w * na, (w + 1) * na))

    sp_path = sweep / 'split.npz'
    bak = sweep / 'split_prefilter.npz'
    sp = np.load(sp_path)
    if not bak.exists():
        np.savez(bak, **{k: sp[k] for k in sp.files})
    out = {}
    for k in sp.files:
        arr = sp[k]
        out[k] = np.array([i for i in arr if int(i) not in bad_rows], arr.dtype)
        print(f"      {k}: {len(arr)} -> {len(out[k])}")
    np.savez(sp_path, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweeps', nargs='+', type=Path, required=True)
    ap.add_argument('--frac', type=float, default=1e-2,
                    help='drop window if median ||tN3|| < frac * member median')
    ap.add_argument('--rough-min', type=float, default=1e-5,
                    help='drop window if stack Delta^2 roughness below this')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    sweeps = [s for s in args.sweeps if (s / 'manifest.json').exists()
              and (s / 'packed' / 'deriv_anal_f64.npy').exists()]
    print(f"[filter] {len(sweeps)} sweep dir(s), frac={args.frac}, "
          f"rough_min={args.rough_min}, dry={args.dry_run}")
    for s in sweeps:
        process(s, args.frac, args.rough_min, args.dry_run)
    if not args.dry_run:
        print("[filter] done. Old splits backed up to split_prefilter.npz")


if __name__ == '__main__':
    main()
