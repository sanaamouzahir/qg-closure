"""
diagnose_condlocal_init.py -- cond_local acceptance gate (init behavior).

Two checks, both REQUIRED before deriv7_cond_local is submitted:

  A. ZERO-INIT EXACTNESS: with the conditioning head zero-initialised and all
     shared weights copied from the control, cond_local outputs must equal the
     unconditioned cheap_deriv to float64 round-off on REAL samples across
     members. (By construction the modulation path contributes exact IEEE
     zeros, so the expected max|diff| is exactly 0.0 -- anything > 1e-14
     relative means the two pipelines diverged somewhere else: FAIL.)

  B. PHYSICS-INIT EVAL: per-sample RAW rel-L2 medians per order over val
     samples of the (filtered) sweep roots, dealias-projected exactly as the
     trainer does. cond_local's medians must equal the control's (same model
     at init -- hard gate), and both are reported against the healthy-tier
     reference medians Ndot~0.19, Nddot~0.26, N3dot~0.33 (width-15 spatial
     gap, RESULTS_2026-07-03) for context. Pooled S=7 numbers include
     near-wall dt samples and sit higher; the context comparison is
     informational, the equality is the gate.

Run on qlogin/qrsh (GPU) from anywhere:
    python diagnose_condlocal_init.py \
        --sweep-roots ../training/data/ensemble_N5_7lag/FRC-*/sweep_dT_* \
        --n-snapshots 7 --grad-kernel 15 --max-val-samples 96 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'training'))

from model_deriv_closure import build_model                     # noqa: E402
from deriv_dataset import make_deriv_loaders                    # noqa: E402


def make_projectors(subsets, device):
    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral, to_physical
    rep = {}
    for s in subsets:
        rep.setdefault((int(s.man['Ny']), int(s.man['Nx'])), s.man)
    proj = {}
    for (Ny, Nx), m in rep.items():
        g = CartesianGrid(Nx=Nx, Ny=Ny, Lx=float(m['Lx']), Ly=float(m['Ly']),
                          device=device, precision='float64')
        keep = (~Derivative(g).alias_mask).to(device)

        def _p(p, keep=keep):
            return to_physical(to_spectral(p) * keep.to(device=p.device,
                                                        dtype=p.dtype))
        proj[(Ny, Nx)] = _p
    return proj


def rel_l2_per_sample(pred, tgt):
    """(B, C) raw per-sample per-channel rel-L2 (no floor)."""
    num = torch.linalg.vector_norm(pred - tgt, dim=(-2, -1))
    den = torch.linalg.vector_norm(tgt, dim=(-2, -1)).clamp_min(1e-300)
    return num / den


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sweep-roots', type=Path, nargs='+', required=True)
    p.add_argument('--n-snapshots', type=int, default=7)
    p.add_argument('--out-orders', type=int, default=3)
    p.add_argument('--grad-kernel', type=int, default=15)
    p.add_argument('--max-val-samples', type=int, default=96)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    roots = [r for r in args.sweep_roots
             if (r / 'manifest.json').exists()
             and (r / 'packed' / 'inputs.npy').exists()]
    if not roots:
        sys.exit('no valid sweep roots')
    print(f'[condlocal-init] roots={len(roots)}')

    _, vl, _, _, val_ds, _ = make_deriv_loaders(
        roots, batch_size=4, num_workers=0,
        n_snapshots=args.n_snapshots, compute_dtype='float64', seed=0)

    # reference grid = most common full grid (mirror the trainer)
    ident = {}
    for s in val_ds.subsets:
        ident.setdefault((int(s.man['Ny']), int(s.man['Nx']),
                          round(float(s.man['Lx']), 6),
                          round(float(s.man['Ly']), 6)), 0)
        ident[(int(s.man['Ny']), int(s.man['Nx']),
               round(float(s.man['Lx']), 6), round(float(s.man['Ly']), 6))] += 1
    Ny0, Nx0, Lx0, Ly0 = max(ident, key=ident.get)
    dx0, dy0 = Lx0 / Nx0, Ly0 / Ny0
    dt0 = float(val_ds.subsets[0].man['Delta_T'])

    in_ch = 2 * args.n_snapshots
    kw = dict(in_channels=in_ch, out_orders=args.out_orders,
              grad_kernel=args.grad_kernel, dt=dt0, dx=dx0, dy=dy0,
              physics_init=True, learnable_stencils=True)
    ctrl = build_model('cheap_deriv', **kw).to(args.device).to(torch.float64)
    cond = build_model('cond_local', **kw).to(args.device).to(torch.float64)
    missing, unexpected = cond.load_state_dict(ctrl.state_dict(), strict=False)
    assert not unexpected, f'unexpected keys: {unexpected}'
    n_new = sum(pp.numel() for pp in cond.cond.parameters())
    print(f'[condlocal-init] shared weights copied; conditioning params={n_new}')

    proj = make_projectors(val_ds.subsets, args.device)
    ctrl.eval(); cond.eval()

    # ---------------- A. zero-init exactness on 4 cross-member samples ----- #
    print('\n=== A. zero-init exactness (4 samples, distinct subsets) ===')
    picks, seen = [], set()
    for gi in range(len(val_ds)):
        s, j = val_ds._locate(gi) if hasattr(val_ds, '_locate') else (None, None)
        if s is None:
            break
        if s not in seen:
            seen.add(s); picks.append(gi)
        if len(picks) == 4:
            break
    if len(picks) < 4:                      # fallback: spread over the index range
        picks = list(np.linspace(0, len(val_ds) - 1, 4).astype(int))
    worst = 0.0
    for gi in picks:
        x, y, regime = val_ds[int(gi)]
        x = x[None].to(args.device)
        r = regime[None].to(args.device)
        dT, dxb, dyb = r[:, 0], r[:, 4], r[:, 5]
        with torch.no_grad():
            y0 = ctrl(x, dt=dT, dx=dxb, dy=dyb)
            y1 = cond(x, dt=dT, dx=dxb, dy=dyb)
        ad = float((y0 - y1).abs().max())
        rd = ad / max(float(y0.abs().max()), 1e-300)
        worst = max(worst, rd)
        print(f'  sample {gi:5d}  (shape {tuple(x.shape[-2:])}, dT={float(dT[0]):.4g})'
              f'  max|diff|={ad:.3e}  rel={rd:.3e}')
    a_pass = worst < 1e-14
    print(f'  A verdict: {"PASS" if a_pass else "FAIL"} (worst rel {worst:.3e}, '
          f'gate 1e-14; expected exactly 0.0)')

    # ---------------- B. physics-init val medians -------------------------- #
    print(f'\n=== B. physics-init val rel-L2 medians (<= {args.max_val_samples} '
          f'samples) ===')
    per_ctrl, per_cond = [], []
    n_done = 0
    with torch.no_grad():
        for x, y, regime in vl:
            x = x.to(args.device); y = y.to(args.device).to(torch.float64)
            r = regime.to(args.device)
            dT, dxb, dyb = r[:, 0].to(x.dtype), r[:, 4].to(x.dtype), r[:, 5].to(x.dtype)
            pj = proj[(x.shape[-2], x.shape[-1])]
            per_ctrl.append(rel_l2_per_sample(pj(ctrl(x, dt=dT, dx=dxb, dy=dyb)), y).cpu())
            per_cond.append(rel_l2_per_sample(pj(cond(x, dt=dT, dx=dxb, dy=dyb)), y).cpu())
            n_done += x.shape[0]
            if n_done >= args.max_val_samples:
                break
    pc = torch.cat(per_ctrl).numpy()
    pd = torch.cat(per_cond).numpy()
    names = ['Ndot', 'Nddot', 'N3dot'][:args.out_orders]
    ref = [0.19, 0.26, 0.33]
    print(f'  {"order":8s} {"ctrl med":>10s} {"cond med":>10s} {"|delta|":>10s} '
          f'{"healthy ref":>12s}')
    b_pass = True
    for c, nm in enumerate(names):
        m0, m1 = float(np.median(pc[:, c])), float(np.median(pd[:, c]))
        d = abs(m0 - m1)
        b_pass &= d < 1e-12
        print(f'  {nm:8s} {m0:10.4f} {m1:10.4f} {d:10.3e} {ref[c]:12.2f}')
    print(f'  B verdict: {"PASS" if b_pass else "FAIL"} (cond==ctrl medians to '
          f'1e-12; healthy-tier refs are context -- pooled S=7 val sits higher '
          f'by design)')

    ok = a_pass and b_pass
    print(f'\n[condlocal-init] OVERALL {"PASS" if ok else "FAIL"}  '
          f'(n_val={n_done}, conditioning params={n_new})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
