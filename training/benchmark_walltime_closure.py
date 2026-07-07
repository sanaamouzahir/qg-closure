"""
benchmark_walltime_closure.py -- 3b: comp-time extension of benchmark_walltime.py.

Measures REAL per-step walltime and walltime-to-horizon for the a-posteriori
configurations, on the same grid/physics as a given sweep root:

  fine_ref   : AB2CN2 at h = Delta_T/K       (the reference the closure replaces;
               time-to-horizon = ms/step * M * K)
  bare       : AB2CN2 at Delta_T
  closure:<ckpt-name>
             : AB2CN2 + trained closure (inference decomposition, from
               rollout_aposteriori.run_arm -- the production loop). Pass
               --ckpt several times to compare models; passing a cheap_deriv
               AND a cond_local checkpoint measures the REAL 2-FFT
               conditioning overhead (contract: cond_local = control + 2 FFTs).
  r3_analytic: AB2CN2 + FULL analytic R3 (chain-rule Ndot/Nddot each step --
               the no-NN alternative; ~3 extra Jacobian evaluations)

Outputs (--out-dir): benchmark_closure_<tag>.csv/.json and
benchmark_closure_<tag>.png -- ms/step bars + cost-vs-accuracy scatter
(walltime-to-horizon vs final rel-L2 pulled from rollout_apost json files
passed via --accuracy-json, matched by arm/model name). The K-fold headline:
closure usefulness = accuracy of ~fine at ~bare-cost x O(3-4), vs the K-fold
cost of actually running fine.

Run on qrsh GPU:
    python benchmark_walltime_closure.py \
        --root-dir data/ensemble_N5_7lag/FRC-b2/sweep_dT_5em3 \
        --ckpt data/.../training_runs/deriv7_cond_local/best.pt \
        --ckpt data/.../training_runs/deriv7_filtered_lr5e-5/best.pt \
        --K 100 --bench-steps 50 --horizon-steps 640 \
        --accuracy-json diagnostics/Results/apost_b2_5em3/rollout_apost_*.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402


def _find_training_dir():
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        if (ancestor / 'dataset.py').exists():
            return ancestor
    return here


sys.path.insert(0, str(_find_training_dir()))

import rollout_aposteriori as ra                                    # noqa: E402
from rollout_timed_pareto import N_spectral, _dealias_mul, build_L_hat, \
    build_forcing, psi_from_omega, _sync                            # noqa: E402
from rollout_perfect_closure import analytic_n_derivs_hat           # noqa: E402


def time_r3_analytic(omega_stack, Delta_T, K, n_steps, derivative, L_hat,
                     F_hat, device):
    """Coarse AB2CN2 + full analytic R3 (implicit L^3 fold + explicit L^2 N +
    chain-rule L Ndot - 5 Nddot). The no-NN full-R3 arm."""
    from qg.solver.opt.basis import to_spectral
    coef = (Delta_T ** 3) * (1.0 - 1.0 / (K ** 2))
    c12 = coef / 12.0
    denom = 1.0 - 0.5 * Delta_T * L_hat + c12 * (L_hat ** 3)
    L2 = L_hat ** 2
    qh_n = to_spectral(omega_stack[0])
    qh_m1 = to_spectral(omega_stack[1])
    Nh_n = N_spectral(qh_n, derivative, F_hat)
    Nh_m1 = N_spectral(qh_m1, derivative, F_hat)

    def one(qh_curr, Nh_curr, Nh_minus):
        Nd = analytic_n_derivs_hat(qh_curr, derivative, L_hat, F_hat, max_order=2)
        f_anal = (1.0 / 12.0) * (L_hat * Nd[1] - 5.0 * Nd[2])
        f_anal = _dealias_mul(f_anal, derivative)
        rhs = (qh_curr + Delta_T * (0.5 * L_hat * qh_curr
                                    + (1.5 * Nh_curr - 0.5 * Nh_minus))
               - c12 * (L2 * Nh_curr) - coef * f_anal)
        return rhs / denom

    for _ in range(3):                                   # warmup, untimed
        qh_new = one(qh_n, Nh_n, Nh_m1)
        Nh_m1, Nh_n, qh_n = Nh_n, N_spectral(qh_new, derivative, F_hat), qh_new
    _sync(device); t0 = time.perf_counter()
    for _ in range(n_steps):
        qh_new = one(qh_n, Nh_n, Nh_m1)
        Nh_m1, Nh_n, qh_n = Nh_n, N_spectral(qh_new, derivative, F_hat), qh_new
    _sync(device)
    return (time.perf_counter() - t0) / n_steps


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-dir', type=Path, required=True)
    p.add_argument('--ckpt', type=Path, action='append', default=[],
                   help='repeatable: each checkpoint is timed as its own '
                        'closure configuration')
    p.add_argument('--ic-index', type=int, default=0)
    p.add_argument('--Delta-T', type=float, default=None)
    p.add_argument('--K', type=int, default=100)
    p.add_argument('--bench-steps', type=int, default=50,
                   help='timed steps per configuration (after 3 warmup steps)')
    p.add_argument('--fine-bench-steps', type=int, default=200,
                   help='timed FINE steps for the reference (extrapolated)')
    p.add_argument('--horizon-steps', type=int, default=640,
                   help='M coarse steps for the walltime-to-horizon numbers')
    p.add_argument('--accuracy-json', type=Path, nargs='*', default=[],
                   help='rollout_apost_*.json files supplying final rel-L2 '
                        'for the cost-vs-accuracy panel')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--tag', type=str, default='bench')
    p.add_argument('--out-dir', type=Path, default=Path('.'))
    args = p.parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    dtype = torch.float64
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.root_dir / 'manifest.json').read_text())
    Nx, Ny = int(manifest['Nx']), int(manifest['Ny'])
    Lx, Ly = float(manifest['Lx']), float(manifest['Ly'])
    nu = float(manifest['nu']); mu = float(manifest.get('mu', 0.0))
    beta = float(manifest.get('beta', 0.0))
    Delta_T = args.Delta_T or float(manifest['Delta_T'])
    K, M = int(args.K), int(args.horizon_steps)
    ra._DX, ra._DY = Lx / Nx, Ly / Ny

    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device,
                         precision='float64')
    derivative = Derivative(grid)
    for attr in ('dx', 'dy', 'laplacian', 'inv_laplacian', 'alias_mask'):
        if hasattr(derivative, attr):
            setattr(derivative, attr, getattr(derivative, attr).to(device))
    L_hat = build_L_hat(derivative, nu, mu, beta).to(device)
    fc = manifest.get('forcing') if manifest.get('has_forcing') else None
    F_phys = build_forcing(grid, fc, device, dtype)
    F_hat = to_spectral(F_phys) if F_phys is not None else None
    print(f'[3b] grid {Ny}x{Nx}  Delta_T={Delta_T} K={K}  horizon M={M} '
          f'coarse steps ({M*Delta_T:.3f} t.u.)  device={device}')

    inp = np.load(args.root_dir / 'packed' / 'inputs.npy', mmap_mode='r')
    fidx = {f: c for c, f in enumerate(manifest['input_fields'])}
    n_all = int(manifest['n_snapshots_per_sample'])

    def stack_for(n_snap):
        om = [torch.tensor(np.asarray(
                inp[args.ic_index, fidx['omega_0' if k == 0 else f'omega_m{k}']],
                dtype=np.float64), dtype=dtype, device=device)[None]
              for k in range(n_snap)]
        ps = [psi_from_omega(o, derivative) for o in om]
        return om, ps

    results = {}

    # ---- fine reference ---- #
    om2, _ = stack_for(2)
    print(f'[3b] fine_ref: {args.fine_bench_steps} timed fine steps ...')
    _, t_warm = ra.run_truth(om2[0], om2[1], Delta_T / K, 3, [],
                             derivative, L_hat, F_hat, device)
    _, t_fine = ra.run_truth(om2[0], om2[1], Delta_T / K,
                             args.fine_bench_steps, [],
                             derivative, L_hat, F_hat, device)
    ms_fine = 1e3 * t_fine / args.fine_bench_steps
    results['fine_ref'] = dict(ms_per_step=ms_fine,
                               steps_to_horizon=M * K,
                               s_to_horizon=ms_fine * M * K / 1e3)

    # ---- bare + r3_analytic ---- #
    om7, ps7 = stack_for(min(7, n_all))
    for arm in ('bare',):
        ra.run_arm(arm, om7, ps7, Delta_T, K, 3, [], derivative, L_hat,
                   F_hat, device, scalars_every=10 ** 9)          # warmup
        r = ra.run_arm(arm, om7, ps7, Delta_T, K, args.bench_steps, [],
                       derivative, L_hat, F_hat, device,
                       scalars_every=10 ** 9)
        ms = 1e3 * r['walltime'] / args.bench_steps
        results[arm] = dict(ms_per_step=ms, steps_to_horizon=M,
                            s_to_horizon=ms * M / 1e3)
        print(f'[3b] {arm}: {ms:.3f} ms/step')

    ms_r3 = 1e3 * time_r3_analytic(om7, Delta_T, K, args.bench_steps,
                                   derivative, L_hat, F_hat, device)
    results['r3_analytic'] = dict(ms_per_step=ms_r3, steps_to_horizon=M,
                                  s_to_horizon=ms_r3 * M / 1e3)
    print(f'[3b] r3_analytic: {ms_r3:.3f} ms/step')

    # ---- closures (one per ckpt) ---- #
    for ck in args.ckpt:
        model, name, n_snap = ra.load_deriv_model(ck, manifest, Delta_T,
                                                  device, nn_float64=True)
        input_fields = (['omega_0'] + [f'omega_m{k}' for k in range(1, n_snap)]
                        + ['psi_0'] + [f'psi_m{k}' for k in range(1, n_snap)])
        omS, psS = stack_for(n_snap)
        label = f'closure:{ck.parent.name}({name})'
        ra.run_arm('closure', omS, psS, Delta_T, K, 3, [], derivative, L_hat,
                   F_hat, device, model=model, input_fields=input_fields,
                   scalars_every=10 ** 9)                          # warmup
        r = ra.run_arm('closure', omS, psS, Delta_T, K, args.bench_steps, [],
                       derivative, L_hat, F_hat, device, model=model,
                       input_fields=input_fields, scalars_every=10 ** 9)
        ms = 1e3 * r['walltime'] / args.bench_steps
        results[label] = dict(ms_per_step=ms, steps_to_horizon=M,
                              s_to_horizon=ms * M / 1e3, model=name,
                              ckpt=str(ck))
        print(f'[3b] {label}: {ms:.3f} ms/step')

    # ---- cost table + K-fold headline ---- #
    ms_bare = results['bare']['ms_per_step']
    print(f'\n[3b] {"config":<42}{"ms/step":>10}{"x bare":>9}{"s->horizon":>12}')
    for name, r in results.items():
        print(f'[3b] {name:<42}{r["ms_per_step"]:>10.3f}'
              f'{r["ms_per_step"]/ms_bare:>9.2f}{r["s_to_horizon"]:>12.2f}')
        r['x_bare'] = r['ms_per_step'] / ms_bare
    print(f'[3b] K-fold headline: fine costs {results["fine_ref"]["s_to_horizon"]/results["bare"]["s_to_horizon"]:.0f}x '
          f'bare to the same horizon; the closure costs '
          + ', '.join(f"{r['x_bare']:.2f}x" for n, r in results.items()
                      if n.startswith('closure:')) + ' bare.')

    # ---- accuracy join ---- #
    acc = {}
    for jp in args.accuracy_json:
        try:
            j = json.loads(Path(jp).read_text())
        except Exception as e:                                     # noqa: BLE001
            print(f'[3b] skip {jp}: {e}')
            continue
        fin = j.get('final_relL2', {})
        mdl = j.get('model', '?')
        for arm, v in fin.items():
            if v is None:
                continue
            key = f'closure({mdl})' if arm == 'closure' else arm
            acc[key] = float(v)
    if acc:
        print(f'[3b] accuracy joined from {len(args.accuracy_json)} json(s): {acc}')

    with open(args.out_dir / f'benchmark_closure_{args.tag}.json', 'w') as f:
        json.dump(dict(config={k: str(v) for k, v in vars(args).items()},
                       Delta_T=Delta_T, K=K, M=M, results=results,
                       accuracy=acc), f, indent=2)
    with open(args.out_dir / f'benchmark_closure_{args.tag}.csv', 'w',
              newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'ms_per_step', 'x_bare', 's_to_horizon'])
        for name, r in results.items():
            w.writerow([name, f'{r["ms_per_step"]:.4f}', f'{r["x_bare"]:.3f}',
                        f'{r["s_to_horizon"]:.3f}'])

    # ---- figure: ms/step bars + cost-vs-accuracy ---- #
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    names = list(results.keys())
    ax[0].barh(range(len(names)), [results[n]['ms_per_step'] for n in names],
               color=['0.4' if n == 'fine_ref' else 'C0' if n == 'bare'
                      else 'C1' if n == 'r3_analytic' else 'C3'
                      for n in names])
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names, fontsize=8)
    ax[0].set_xlabel('ms / step')
    ax[0].set_title(f'per-step cost ({Ny}x{Nx}, {device})')
    ax[0].grid(alpha=0.3, axis='x')
    for i, n in enumerate(names):
        ax[0].text(results[n]['ms_per_step'], i,
                   f"  {results[n]['x_bare']:.2f}x bare", va='center',
                   fontsize=8)

    if acc:
        for key, err in acc.items():
            wt = None
            if key == 'bare':
                wt = results['bare']['s_to_horizon']
            elif key == 'r3only':
                wt = results['r3_analytic']['s_to_horizon']
            elif key.startswith('closure'):
                cands = [r for n, r in results.items()
                         if n.startswith('closure:')]
                wt = cands[0]['s_to_horizon'] if cands else None
            if wt is not None:
                ax[1].loglog([wt], [err], 'o', ms=9, label=key)
        ax[1].axvline(results['fine_ref']['s_to_horizon'], color='0.4',
                      ls='--', lw=1.2,
                      label=f'fine ref cost (K={K})')
        ax[1].set_xlabel('walltime to horizon (s)')
        ax[1].set_ylabel('final rel-L2 vs truth')
        ax[1].set_title('cost vs accuracy (the K-fold headline)')
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which='both')
    else:
        ax[1].text(0.5, 0.5, 'no --accuracy-json given\n(run 3a first, '
                   'then re-run with the json paths)', ha='center',
                   va='center', transform=ax[1].transAxes)
        ax[1].axis('off')
    fig.tight_layout()
    fig.savefig(args.out_dir / f'benchmark_closure_{args.tag}.png', dpi=140)
    print(f'[3b] wrote benchmark_closure_{args.tag}.png/.csv/.json '
          f'in {args.out_dir}')


if __name__ == '__main__':
    main()
