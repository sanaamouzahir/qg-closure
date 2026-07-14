#!/usr/bin/env python3
"""bench_grad31_cost.py -- inference/training cost of grad_kernel 31 vs 15 vs
spectral gradients (Sanaa order 2026-07-14: "check the 31 width vs spectral
gradient costs").

Measures, on GPU, float64, median over --reps timed reps (cuda-synchronized):
  1. model FORWARD walltime: cond_local k15 / cond_local k31 / cond_deriv
     (spectral instrument), at 256^2 and 512^2, batch 1 (inference-like)
     and batch 4 (training-like);
  2. FORWARD+BACKWARD walltime k15 vs k31 (training throughput of the retrain);
  3. FULL closure-step walltime via the validated rollout stepper (bare vs
     closure k15 vs closure k31) on the FRC-256 deep root -- the number that
     enters the runtime pitch (~3x bare + NN forward);
  4. peak CUDA memory per arm.
Writes diagnostics/Results/bench_w31_20260714/bench_w31.csv + .json.
Run FROM training/ (flat sibling imports; the stepper needs the deep root).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from model_deriv_closure import build_model                      # noqa: E402

OUT = Path(__file__).resolve().parent / 'Results' / 'bench_w31_20260714'
DEEP = Path('data/ensemble_N5_7lag/FRC-256/forced_turbulence_dT_5em3')
REF = dict(dt=1.0e-3, dx=0.0245436933, dy=0.0245436933)


def timed(fn, reps, dev):
    for _ in range(5):                                   # warmup
        fn()
    if dev == 'cuda':
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if dev == 'cuda':
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3                    # ms


def peak_mem(fn, dev):
    if dev != 'cuda':
        return float('nan')
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20     # MiB


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    reps = 50 if dev == 'cuda' else 5
    rows = []

    models = {}
    for name, kind, gk in [('cond_local_k15', 'cond_local', 15),
                           ('cond_local_k31', 'cond_local', 31),
                           ('cond_deriv_spec', 'cond_deriv', 15)]:
        m = build_model(kind, in_channels=14, out_orders=3, grad_kernel=gk,
                        physics_init=True, learnable_stencils=True,
                        **REF).to(dev).double()
        models[name] = m
        n = sum(q.numel() for q in m.parameters() if q.requires_grad)
        print(f"[bench] {name}: {n:,} trainable params")

    torch.manual_seed(0)
    for N in (256, 512):
        for B in (1, 4):
            x = torch.randn(B, 14, N, N, dtype=torch.float64, device=dev)
            dt = torch.full((B,), 5.0e-3, dtype=torch.float64, device=dev)
            dx = torch.full((B,), REF['dx'], dtype=torch.float64, device=dev)
            for name, m in models.items():
                def fwd(m=m):
                    with torch.no_grad():
                        m(x, dt=dt, dx=dx, dy=dx)
                ms = timed(fwd, reps, dev)
                mem = peak_mem(fwd, dev)
                rows.append(dict(bench=f'forward_{N}_b{B}', arm=name,
                                 ms=ms, peak_MiB=mem))
                print(f"  forward {N}^2 b{B} {name:>16}: {ms:8.2f} ms  "
                      f"peak {mem:7.1f} MiB")
            # training step (fwd+bwd) for the two local arms only
            if B == 4:
                for name in ('cond_local_k15', 'cond_local_k31'):
                    m = models[name]

                    def step(m=m):
                        m.zero_grad(set_to_none=True)
                        y = m(x, dt=dt, dx=dx, dy=dx)
                        y.square().mean().backward()
                    ms = timed(step, reps, dev)
                    mem = peak_mem(step, dev)
                    rows.append(dict(bench=f'fwdbwd_{N}_b4', arm=name,
                                     ms=ms, peak_MiB=mem))
                    print(f"  fwd+bwd {N}^2 b4 {name:>16}: {ms:8.2f} ms  "
                          f"peak {mem:7.1f} MiB")

    # full closure step via the validated stepper (bare / k15 / k31):
    # one_step(qc, qm, Nc, Nm, om_list, ps_list) -> (qh, Nh, om, ps)
    if DEEP.exists():
        from train_deriv_rollout import (RootCtx, make_stepper, set_globals,
                                         init_state)
        rc = RootCtx(DEEP, dev, roughness_min=1e-4, seed=0)
        set_globals(rc)
        omega_stack, _ = rc.window_tensors(rc.train_idx[0], 1, 1, dev)
        arms = [('bare', None), ('closure_k15', models['cond_local_k15']),
                ('closure_k31', models['cond_local_k31'])]
        for aname, m in arms:
            try:
                stp = make_stepper(rc, 1, m, dev,
                                   arm=('bare' if m is None else 'closure'),
                                   nn_grad=False)
                qc, qm, Nc, Nm, om, ps = init_state(rc, omega_stack, dev)

                def one(stp=stp, qc=qc, qm=qm, Nc=Nc, Nm=Nm, om=om, ps=ps):
                    with torch.no_grad():
                        stp(qc, qm, Nc, Nm, list(om), list(ps))
                ms = timed(one, reps, dev)
                rows.append(dict(bench='closure_step_256', arm=aname,
                                 ms=ms, peak_MiB=float('nan')))
                print(f"  closure step 256^2 {aname:>12}: {ms:8.2f} ms")
            except Exception as e:
                print(f"  closure step {aname}: SKIP ({e})")
    else:
        print(f"[bench] deep root {DEEP} absent -- stepper bench skipped")

    OUT.mkdir(parents=True, exist_ok=True)
    import csv
    with open(OUT / 'bench_w31.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['bench', 'arm', 'ms', 'peak_MiB'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (OUT / 'bench_w31.json').write_text(json.dumps(
        dict(device=dev, reps=reps, rows=rows), indent=2))
    print(f"[bench] wrote {OUT}/bench_w31.csv")


if __name__ == '__main__':
    main()
