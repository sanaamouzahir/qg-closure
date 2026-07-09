"""spectral_error_profile.py -- per-|k|-shell relative error profile eps(k)
of the predicted N-derivatives (Ndot, Nddot, N3dot) for one or more
checkpoints, on VAL samples across (member, dt). Sanaa mandate 2026-07-09
Part 1: where is the error knee; how bad is the 170.7-241.4 aliasing annulus
vs k<60; how much did conditioning bend the profile.

Per sample s, order m, integer mode-radius shell kappa:
    eps_m(kappa) = sqrt( sum_shell w |F(pred-targ)|^2 / sum_shell w |F(targ)|^2 )
with rfft2 Hermitian double-count weights w, prediction end-projected with the
SOLVER's dealias mask before comparison (= the training loss convention).
Aggregation: MEDIAN over samples (rule 16). Additionally per-band
energy-weighted errors for bands k<60 (low), 60-170.7 (mid), 170.7-241.4
(annulus, 512^2 convention).

512^2 members only (one shell convention). One output npz, no per-sample
litter.

Usage (from training/, flat sibling imports):
  python ../diagnostics/spectral_error_profile.py \
      --ckpt cond=<path> --ckpt control=<path> [...] \
      --roots data/ensemble_N5_7lag/FRC-{b1,b2,kf4,combo}/sweep_dT_* \
      --n-samples 6 --device cuda --out <dir>/spectral_error_profile.npz
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
    for cand in [here.parent / 'training', here]:
        if (cand / 'dataset.py').exists() or (cand / 'train_deriv.py').exists():
            return cand
    return here


sys.path.insert(0, str(_find_training_dir()))

from rollout_aposteriori import load_deriv_model            # noqa: E402

ORDERS = ['Ndot', 'Nddot', 'N3dot']
BANDS = {'low_k<60': (0.0, 60.0), 'mid_60-170': (60.0, 170.67),
         'annulus_171-241': (170.67, 241.4)}


def shell_index(Ny, Nx, device):
    iy = torch.fft.fftfreq(Ny, d=1.0 / Ny, device=device)
    ix = torch.arange(Nx // 2 + 1, dtype=torch.float64, device=device)
    kmag = torch.sqrt(iy[:, None] ** 2 + ix[None, :] ** 2)
    sh = torch.round(kmag).to(torch.int64)
    w = torch.full((Nx // 2 + 1,), 2.0, dtype=torch.float64, device=device)
    w[0] = 1.0
    if Nx % 2 == 0:
        w[-1] = 1.0
    return sh, kmag, w[None, :].expand(Ny, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', action='append', required=True,
                    metavar='LABEL=PATH')
    ap.add_argument('--roots', nargs='+', type=Path, required=True)
    ap.add_argument('--n-samples', type=int, default=6,
                    help='val samples per root (evenly spaced in val_idx)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    device = args.device if (args.device == 'cpu'
                             or torch.cuda.is_available()) else 'cpu'
    ckpts = dict(kv.split('=', 1) for kv in args.ckpt)

    from qg.solver.grid.cartesian import CartesianGrid
    from qg.solver.opt.derivative import Derivative
    from qg.solver.opt.basis import to_spectral, to_physical

    models = {}
    man0 = json.loads((args.roots[0] / 'manifest.json').read_text())
    for lab, pth in ckpts.items():
        m, name, ns = load_deriv_model(Path(pth), man0,
                                       float(man0['Delta_T']), device)
        models[lab] = (m, name, ns)

    payload = {}
    summary = []
    for root in args.roots:
        man = json.loads((root / 'manifest.json').read_text())
        Ny, Nx = int(man['Ny']), int(man['Nx'])
        if Ny != 512:
            print(f'[spec-eps] skip {root} (Ny={Ny}; 512^2-only run)')
            continue
        dt = float(man['Delta_T'])
        dx = float(man['Lx']) / Nx
        dy = float(man['Ly']) / Ny
        sp = np.load(root / 'split.npz')
        val = np.sort(sp['val_idx'])
        take = val[np.linspace(0, len(val) - 1, args.n_samples).astype(int)]
        inp = np.load(root / 'packed' / 'inputs.npy', mmap_mode='r')
        tgt = np.load(root / 'packed' / 'deriv_anal_f64.npy', mmap_mode='r')
        g = CartesianGrid(Nx=Nx, Ny=Ny, Lx=float(man['Lx']),
                          Ly=float(man['Ly']), device=device,
                          precision='float64')
        der = Derivative(g)
        keep = (~der.alias_mask).to(device=device, dtype=torch.float64)
        sh, kmag, w = shell_index(Ny, Nx, device)
        n_sh = int(sh.max()) + 1
        shf = sh.reshape(-1)
        dt_v = torch.full((1,), dt, device=device, dtype=torch.float64)
        dx_v = torch.full((1,), dx, device=device, dtype=torch.float64)
        dy_v = torch.full((1,), dy, device=device, dtype=torch.float64)
        rtag = f"{root.parent.name}_{root.name.replace('sweep_dT_', '')}"
        # per (label, order): list over samples of eps(kappa)
        acc = {lab: [[] for _ in ORDERS] for lab in models}
        band_acc = {lab: {m: {b: [] for b in BANDS} for m in ORDERS}
                    for lab in models}
        for row in take:
            x = torch.tensor(np.asarray(inp[row], dtype=np.float64),
                             dtype=torch.float64, device=device)[None]
            y = torch.tensor(np.asarray(tgt[row], dtype=np.float64),
                             dtype=torch.float64, device=device)
            for lab, (mdl, name, ns) in models.items():
                with torch.no_grad():
                    p = mdl(x, dt=dt_v, dx=dx_v, dy=dy_v)[0].to(torch.float64)
                for mo in range(3):
                    ph = to_spectral(p[mo]) * keep       # training projection
                    th = to_spectral(y[mo])
                    e = ((ph - th).real ** 2 + (ph - th).imag ** 2) * w
                    t2 = (th.real ** 2 + th.imag ** 2) * w
                    E = torch.zeros(n_sh, dtype=torch.float64, device=device)
                    T = torch.zeros_like(E)
                    E.scatter_add_(0, shf, e.reshape(-1))
                    T.scatter_add_(0, shf, t2.reshape(-1))
                    eps = torch.sqrt(E / T.clamp_min(1e-300)).cpu().numpy()
                    eps[T.cpu().numpy() <= 0] = np.nan
                    acc[lab][mo].append(eps)
                    for b, (lo, hi) in BANDS.items():
                        m_ = (kmag.reshape(-1) >= lo) & (kmag.reshape(-1) < hi)
                        Eb = float(e.reshape(-1)[m_].sum())
                        Tb = float(t2.reshape(-1)[m_].sum())
                        band_acc[lab][ORDERS[mo]][b].append(
                            np.sqrt(Eb / max(Tb, 1e-300)))
        for lab in models:
            for mo, oname in enumerate(ORDERS):
                med = np.nanmedian(np.stack(acc[lab][mo]), axis=0)
                payload[f'eps_{lab}_{oname}_{rtag}'] = med
            row = dict(root=rtag, ckpt=lab, dt=dt)
            for oname in ORDERS:
                for b in BANDS:
                    row[f'{oname}_{b}'] = float(np.median(
                        band_acc[lab][oname][b]))
            summary.append(row)
            print(f"[spec-eps] {rtag:22s} {lab:8s} Nddot: "
                  + '  '.join(f"{b}={row[f'Nddot_{b}']:.4f}" for b in BANDS))
        payload[f'target_shell_energy_{rtag}'] = (
            T.cpu().numpy())     # last order's target spectrum (N3dot)
    payload['summary_json'] = np.str_(json.dumps(summary))
    payload['bands_json'] = np.str_(json.dumps(
        {b: list(v) for b, v in BANDS.items()}))
    payload['ckpts_json'] = np.str_(json.dumps(ckpts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print(f'[spec-eps] wrote {args.out} ({len(summary)} (root,ckpt) rows)')


if __name__ == '__main__':
    main()
