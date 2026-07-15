"""diagnose_head_sign.py -- STEP-1b discriminator: SIGNED correlation per NN head.

rel-L2 alone cannot distinguish 'sign-flipped head' (rel ~ 2.0, corr ~ -1) from
'noisy but right-signed head' (rel ~ 0.2, corr ~ +0.98). This prints, per head
[Ndot, Nddot, N3dot] and per sample:

    corr = <y_nn * y_target> / (|y_nn| |y_target|)     (domain mean, signed)
    rel  = |y_nn - y_target| / |y_target|              (rel-L2, cross-check)

Verdict line: SIGN-OK (all corr > +0.5) / SIGN-FLIP (any corr < -0.5) / MIXED.

Run on a COMPUTE node (qlogin/qrsh; guard blocks diagnostics on the head node):
    cd <worktree>/training
    python ../diagnostics/diagnose_head_sign.py \
        --root-dir data/ensemble_N5_7lag/FRC-kf4/sweep_dT_1p5em2 \
        --ckpt data/ensemble_N5_7lag/training_runs/deriv7_filtered_lr5e-5/best.pt \
        --samples 837 900 1000 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _find_training_dir():
    here = Path(__file__).resolve().parent
    for anc in [here.parent / 'training', here, *here.parents]:
        if (anc / 'dataset.py').exists():
            return anc
    return here


sys.path.insert(0, str(_find_training_dir()))

from rollout_aposteriori import load_deriv_model                     # noqa: E402
from rollout_timed_pareto import assemble_inputs                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root-dir', type=Path, required=True)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--samples', type=int, nargs='+', required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    manifest = json.loads((args.root_dir / 'manifest.json').read_text())
    dt = float(manifest['Delta_T'])
    Nx, Ny = int(manifest['Nx']), int(manifest['Ny'])
    dx, dy = float(manifest['Lx']) / Nx, float(manifest['Ly']) / Ny
    model, name, n_snap = load_deriv_model(args.ckpt, manifest, dt, args.device)
    fields = (['omega_0'] + [f'omega_m{k}' for k in range(1, n_snap)]
              + ['psi_0'] + [f'psi_m{k}' for k in range(1, n_snap)])
    # NB: the sliced manifest's input_fields is the DEEP build's 56-entry list;
    # the sliced pack is (N, 2S, Ny, Nx) = [omega_0..S-1, psi_0..S-1]
    # (slice_deriv_from_deep.py docstring). Index by construction, not manifest.
    fidx = {f'omega_m{k}' if k else 'omega_0': k for k in range(n_snap)}
    fidx.update({f'psi_m{k}' if k else 'psi_0': n_snap + k
                 for k in range(n_snap)})
    inp = np.load(args.root_dir / 'packed' / 'inputs.npy', mmap_mode='r')
    assert inp.shape[1] == 2 * n_snap, (inp.shape, n_snap)
    tgt = np.load(args.root_dir / 'packed' / 'deriv_anal_f64.npy', mmap_mode='r')
    heads = ['Ndot', 'Nddot', 'N3dot']
    dt_v = torch.full((1,), dt, dtype=torch.float64, device=args.device)
    dx_v = torch.full((1,), dx, dtype=torch.float64, device=args.device)
    dy_v = torch.full((1,), dy, dtype=torch.float64, device=args.device)

    print(f'[head-sign] {args.root_dir.name}  dt={dt}  ckpt={args.ckpt.parent.name}'
          f'  samples={args.samples}')
    all_corr = []
    for s in args.samples:
        stack = {f: torch.tensor(np.asarray(inp[s, fidx[f]], dtype=np.float64),
                                 dtype=torch.float64, device=args.device)[None]
                 for f in fields}
        om = [stack['omega_0']] + [stack[f'omega_m{k}'] for k in range(1, n_snap)]
        ps = [stack['psi_0']] + [stack[f'psi_m{k}'] for k in range(1, n_snap)]
        x = assemble_inputs(fields, om, ps, torch.float64, args.device)
        with torch.no_grad():
            y = model(x, dt=dt_v, dx=dx_v, dy=dy_v).to(torch.float64)[0]
        for h, hname in enumerate(heads):
            a = y[h]
            b = torch.tensor(np.asarray(tgt[s, h], dtype=np.float64),
                             device=args.device)
            corr = float((a * b).mean()
                         / max(float(a.pow(2).mean().sqrt()
                                     * b.pow(2).mean().sqrt()), 1e-30))
            rel = float((a - b).pow(2).mean().sqrt()
                        / max(float(b.pow(2).mean().sqrt()), 1e-30))
            all_corr.append(corr)
            print(f'  sample {s:>5}  {hname:<6} corr={corr:+.4f}  rel-L2={rel:.4f}')
    worst = min(all_corr)
    if all(c > 0.5 for c in all_corr):
        v = 'SIGN-OK'
    elif any(c < -0.5 for c in all_corr):
        v = 'SIGN-FLIP'
    else:
        v = 'MIXED'
    print(f'[head-sign] VERDICT: {v} (worst corr {worst:+.4f})')


if __name__ == '__main__':
    main()
