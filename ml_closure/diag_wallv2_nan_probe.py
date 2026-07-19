#!/usr/bin/env python
"""diag_wallv2_nan_probe.py -- locate the FIRST non-finite statistic in the
wallv2 feature/stat pipeline (post-mortem of the 2026-07-19 100-ep NaN burn;
Sanaa never-again policy). CPU, read-only, no training.

Replicates train_piff's cold-start pre-loop exactly: build_runs ->
conditioning_stats -> target_stats, then walks every per-run array the
wall gate touches (gradmag, lapmag, sdf, Pi targets) reporting count/min/
max/nonfinite BEFORE the aggregate stats can hide the culprit.

Usage: python diag_wallv2_nan_probe.py --config conf_piff_fpc_gjs_wallv2.yaml
"""
import argparse
import json

import numpy as np

from dataset_piff import (load_conf, build_runs, conditioning_stats,
                          target_stats)
from pathlib import Path

HERE = Path(__file__).resolve().parent


def arr_report(name, a):
    a = np.asarray(a)
    nf = int((~np.isfinite(a)).sum())
    msg = (f"  {name:<22} shape={tuple(a.shape)} nonfinite={nf}"
           f" min={np.nanmin(a):.3e} max={np.nanmax(a):.3e}"
           f" absmed={np.nanmedian(np.abs(a)):.3e}")
    print(msg, flush=True)
    return nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    conf = load_conf(HERE / args.config)
    print(f"[probe] config {args.config}  use_wall_gate="
          f"{conf.get('model', {}).get('use_wall_gate')}", flush=True)
    runs = build_runs(conf)
    bad = 0
    for r in runs:
        print(f"[run] {r.name}", flush=True)
        for attr in ('gradmag', 'lapmag', 'sdf', 'sdf_star'):
            if hasattr(r, attr) and getattr(r, attr) is not None:
                bad += arr_report(attr, getattr(r, attr))
        # per-run scale precursors the aggregate stats reduce over
        for attr in ('pi', 'y', 'targets'):
            if hasattr(r, attr) and getattr(r, attr) is not None:
                bad += arr_report(attr, getattr(r, attr))
    print("[probe] conditioning_stats(train):", flush=True)
    try:
        cs = conditioning_stats(runs, 'train', conf)
        print(json.dumps({k: (float(v) if np.isscalar(v) else str(v))
                          for k, v in cs.items()}, indent=1), flush=True)
        for k, v in cs.items():
            if np.isscalar(v) and not np.isfinite(float(v)):
                print(f"  << NON-FINITE STAT: {k}", flush=True)
                bad += 1
            if np.isscalar(v) and float(v) == 0.0:
                print(f"  << ZERO SCALE (division risk): {k}", flush=True)
    except Exception as e:
        print(f"  conditioning_stats RAISED: {e!r}", flush=True)
        bad += 1
    print("[probe] target_stats(train):", flush=True)
    try:
        ys = target_stats(runs, 'train', conf)
        print({k: (float(v) if np.isscalar(v) else v) for k, v in ys.items()},
              flush=True)
        for k in ('mean', 'var'):
            if not np.isfinite(float(ys[k])):
                print(f"  << NON-FINITE TARGET STAT: {k}", flush=True)
                bad += 1
    except Exception as e:
        print(f"  target_stats RAISED: {e!r}", flush=True)
        bad += 1
    print(f"[probe] VERDICT: {'CLEAN' if bad == 0 else f'{bad} bad item(s)'}",
          flush=True)


if __name__ == '__main__':
    main()
