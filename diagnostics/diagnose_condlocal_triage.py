#!/usr/bin/env python
"""
diagnose_condlocal_triage.py -- D1/D2/D3 triage for the deriv7_cond_local
incident (job 1827034, ep0-5: val_Ndot > 1 oscillating while val_Nddot ~ 0.22).

D1: per-(member,dt) per-order MEDIAN vs MEAN rel-L2 on the val split, RAW and
    FLOORED (denominator = max(||t||, 0.1 * member-median), the training
    convention), for (a) the incident best.pt and (b) a fresh zero-init
    cond_local (== physics-init control per the I8 init gate). Worst-5 samples
    per root by raw Ndot with their per-order ||target||.
D2: target<->input alignment on the TASK-0c salvaged members: recompute
    [Ndot, Nddot, N3dot] analytically from inputs ch0 (the anchor omega_0,
    spectral recursion of add_deriv_targets.py) and compare to the stored
    packed/deriv_anal_f64.npy row; repeat from ch1 (off-by-one probe); plus
    psi_0 == inv_laplacian(omega_0) internal-stack consistency.
D3: zero-init medians grouped by full grid identity (256^2 vs 512^2, L) --
    a grid split at init implicates the per-sample dx rescale.

Also prints the floor audit: stored member-median (regime[6:9], estimated
from <=128 rows of the FULL packed array, pre-filter) vs the val-split
median -- a large ratio means the floor is ineffective for that member.

Usage (from the worktree training/ dir, GPU):
    python ../diagnostics/diagnose_condlocal_triage.py \
        --roots data/ensemble_N5_7lag/*/sweep_dT_* \
        --ckpt data/ensemble_N5_7lag/training_runs/deriv7_cond_local_INCIDENT1827034_snapshot/best.pt \
        --max-per-root 48 --d2
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'training'))

from deriv_dataset import DerivMemberDataset            # noqa: E402
from model_deriv_closure import build_model             # noqa: E402
from add_deriv_targets import setup_solver, compute_n_3dot  # noqa: E402
from build_training_data_fixD_v2 import (compute_n_dot_analytical,   # noqa: E402
                                         compute_n_ddot_analytical)
from qg.solver.grid.cartesian import CartesianGrid      # noqa: E402
from qg.solver.opt.derivative import Derivative         # noqa: E402
from qg.solver.opt.basis import to_spectral, to_physical  # noqa: E402

ONAMES = ['Ndot', 'Nddot', 'N3dot']


def build_projections(mans, device):
    proj = {}
    for m in mans:
        key = (int(m['Ny']), int(m['Nx']))
        if key in proj:
            continue
        g = CartesianGrid(Nx=key[1], Ny=key[0], Lx=float(m['Lx']),
                          Ly=float(m['Ly']), device=device, precision='float64')
        keep = (~Derivative(g).alias_mask).to(device)

        def _p(p, keep=keep):
            return to_physical(to_spectral(p) * keep.to(device=p.device,
                                                        dtype=p.dtype))
        proj[key] = _p
    return proj


@torch.no_grad()
def eval_root(model, ds, project, device, max_n, bs=4):
    n = min(len(ds), max_n)
    pick = np.linspace(0, len(ds) - 1, n, dtype=int)
    raw = np.zeros((n, 3)); flo = np.zeros((n, 3)); den_all = np.zeros((n, 3))
    reg = ds.regime_vec
    floor = 0.1 * reg[6:9].double().to(device)          # (3,)
    for s0 in range(0, n, bs):
        ii = pick[s0:s0 + bs]
        xs, ys = [], []
        for i in ii:
            x, y, _ = ds[int(i)]
            xs.append(x); ys.append(y)
        x = torch.stack(xs).to(device); y = torch.stack(ys).to(device)
        B = x.shape[0]
        dT = torch.full((B,), float(reg[0]), device=device, dtype=x.dtype)
        dx = torch.full((B,), float(reg[4]), device=device, dtype=x.dtype)
        dy = torch.full((B,), float(reg[5]), device=device, dtype=x.dtype)
        p = model(x, dt=dT, dx=dx, dy=dy)
        p = project(p)
        num = (p - y).flatten(2).norm(dim=2)             # (B,3)
        den = y.flatten(2).norm(dim=2)                   # (B,3)
        raw[s0:s0 + B] = (num / den).cpu().numpy()
        flo[s0:s0 + B] = (num / torch.maximum(den, floor)).cpu().numpy()
        den_all[s0:s0 + B] = den.cpu().numpy()
    return raw, flo, den_all, pick


def d2_check(root: Path, device, n_samp=2):
    man = json.loads((root / 'manifest.json').read_text())
    der, L_hat, F = setup_solver(man, device)
    inp = np.load(root / 'packed' / 'inputs.npy', mmap_mode='r')
    tgt = np.load(root / 'packed' / 'deriv_anal_f64.npy', mmap_mode='r')
    S = int(man['n_snapshots_per_sample'])
    sp = np.load(root / 'split.npz')
    vi = sp['val_idx']
    rows = [int(vi[0]), int(vi[len(vi) // 2])][:n_samp]
    out = []
    for g in rows:
        t = torch.as_tensor(np.asarray(tgt[g]), dtype=torch.float64,
                            device=device)              # (3,Ny,Nx)
        errs = {}
        for tag, ch in (('ch0', 0), ('ch1', 1)):
            om = torch.as_tensor(np.asarray(inp[g, ch]), dtype=torch.float64,
                                 device=device)[None]
            r = torch.stack([compute_n_dot_analytical(om, der, L_hat, F)[0],
                             compute_n_ddot_analytical(om, der, L_hat, F)[0],
                             compute_n_3dot(om, der, L_hat, F)[0]])
            errs[tag] = ((r - t).flatten(1).norm(dim=1)
                         / t.flatten(1).norm(dim=1)).cpu().numpy()
        om0 = torch.as_tensor(np.asarray(inp[g, 0]), dtype=torch.float64,
                              device=device)[None]
        ps0 = torch.as_tensor(np.asarray(inp[g, S]), dtype=torch.float64,
                              device=device)[None]
        psr = to_physical(der.inv_laplacian * to_spectral(om0))
        psi_err = float((psr - ps0).norm() / ps0.norm())
        out.append((g, errs['ch0'], errs['ch1'], psi_err))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', type=Path, nargs='+', required=True)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--last', type=Path, default=None,
                    help='optional last.pt for epoch/val recovery printout')
    ap.add_argument('--max-per-root', type=int, default=48)
    ap.add_argument('--n-snapshots', type=int, default=7)
    ap.add_argument('--grad-kernel', type=int, default=15)
    ap.add_argument('--d2', action='store_true')
    ap.add_argument('--d2-members', default='DEC-512,DEC-base,DEC-hiRe,DEC-loRe,'
                    'FRC-b0,FRC-b05,FRC-b075,FRC-b1,FRC-kf4,FRC-b2')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'cpu')
    ap.add_argument('--out-csv', type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    roots = [r for r in args.roots if (r / 'manifest.json').exists()
             and (r / 'packed' / 'deriv_anal_f64.npy').exists()]
    mans = [json.loads((r / 'manifest.json').read_text()) for r in roots]

    # trainer-parity reference grid: most common full identity
    ident = {}
    for r, m in zip(roots, mans):
        ident.setdefault((int(m['Ny']), int(m['Nx']), round(float(m['Lx']), 6),
                          round(float(m['Ly']), 6)), []).append(r)
    ref = max(ident, key=lambda k: len(ident[k]))
    dx0 = ref[2] / ref[1]; dy0 = ref[3] / ref[0]
    dt0 = float(mans[0]['Delta_T'])
    print(f"[triage] roots={len(roots)}  ref={ref}  dx0={dx0:.6e}  dt0={dt0}")

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    print(f"[triage] incident ckpt: epoch={ck.get('epoch')}  "
          f"val={ck.get('val')}")
    if args.last and args.last.exists():
        cl = torch.load(args.last, map_location=dev, weights_only=False)
        print(f"[triage] incident last: epoch={cl.get('epoch')}  "
              f"val={cl.get('val')}")

    def fresh():
        m = build_model('cond_local', in_channels=2 * args.n_snapshots,
                        out_orders=3, grad_kernel=args.grad_kernel, dt=dt0,
                        dx=dx0, dy=dy0, physics_init=True,
                        learnable_stencils=True).to(dev).to(torch.float64)
        m.eval()
        return m

    # strict=False: pre-dt_ref_cond ckpts (e.g. incident 1827034) lack the
    # buffer added 2026-07-08; anything else missing is a real error.
    m_best = fresh()
    _miss, _unexp = m_best.load_state_dict(ck['model'], strict=False)
    assert not _unexp and set(_miss) <= {'dt_ref_cond'}, (_miss, _unexp)
    m_best.eval()
    m_zero = fresh()

    proj = build_projections(mans, dev)

    hdr = (f"{'member':>9s} {'dt':>7s} {'grid':>10s} {'n':>4s} | "
           f"{'model':>5s} | " + ' | '.join(
               f"{o}: medR meanR medF meanF" for o in ONAMES))
    print('\n===== D1: per-root val rel-L2 (R=raw, F=floored 0.1*member-med) =====')
    print(hdr)
    rows_csv = []
    agg = {}   # (model, gridident) -> list of raw arrays
    for r, m in zip(roots, mans):
        ds = DerivMemberDataset(r, split='val', n_snapshots=args.n_snapshots,
                                compute_dtype='float64')
        key = (int(m['Ny']), int(m['Nx']))
        gid = (int(m['Ny']), int(m['Nx']), round(float(m['Lx']), 6))
        member, tag = r.parent.name, r.name.replace('sweep_dT_', '')
        # floor audit: stored member-median (pre-filter, 128-row estimate)
        # vs val-split target medians
        for name, mdl in (('best', m_best), ('zero', m_zero)):
            raw, flo, den, pick = eval_root(mdl, ds, proj[key], dev,
                                            args.max_per_root)
            agg.setdefault((name, gid), []).append(raw)
            cells = []
            for c in range(3):
                cells.append(f"{np.median(raw[:, c]):8.3f} {raw[:, c].mean():8.3f} "
                             f"{np.median(flo[:, c]):8.3f} {flo[:, c].mean():8.3f}")
            print(f"{member:>9s} {tag:>7s} {str(key):>10s} {len(raw):>4d} | "
                  f"{name:>5s} | " + ' | '.join(cells))
            for c in range(3):
                rows_csv.append(dict(member=member, dt=tag, model=name,
                                     order=ONAMES[c],
                                     med_raw=float(np.median(raw[:, c])),
                                     mean_raw=float(raw[:, c].mean()),
                                     med_flo=float(np.median(flo[:, c])),
                                     mean_flo=float(flo[:, c].mean())))
            if name == 'best':
                w = np.argsort(raw[:, 0])[-5:][::-1]
                stored_med = ds.regime_vec[6:9].numpy()
                val_med = np.median(den, axis=0)
                print(f"          floor-audit: stored_med={stored_med}  "
                      f"val_med(||t||)={val_med}  "
                      f"ratio={val_med / np.maximum(stored_med, 1e-300)}")
                for j in w:
                    print(f"          worst[raw Ndot] ds_i={pick[j]:4d} "
                          f"relNdot={raw[j, 0]:9.3f} "
                          f"||t||=({den[j, 0]:.3e},{den[j, 1]:.3e},{den[j, 2]:.3e})")

    print('\n===== D3: zero-init RAW medians by full-grid identity =====')
    for (name, gid), lst in sorted(agg.items()):
        if name != 'zero':
            continue
        a = np.concatenate(lst, 0)
        med = np.median(a, 0)
        print(f"  grid={gid}  n={len(a)}  med raw relL2 = "
              + '  '.join(f"{o}={v:.4f}" for o, v in zip(ONAMES, med)))

    if args.d2:
        print('\n===== D2: target<->input alignment (recompute from ch0/ch1) =====')
        members = args.d2_members.split(',')
        for mem in members:
            r = None
            for cand in roots:
                if cand.parent.name == mem and cand.name == 'sweep_dT_1em2':
                    r = cand; break
            if r is None:
                print(f"  {mem}: no sweep_dT_1em2 root, skip"); continue
            for g, e0, e1, pe in d2_check(r, dev):
                print(f"  {mem:>9s} row={g:4d}  "
                      f"relerr(ch0)=({e0[0]:.2e},{e0[1]:.2e},{e0[2]:.2e})  "
                      f"relerr(ch1)=({e1[0]:.2e},{e1[1]:.2e},{e1[2]:.2e})  "
                      f"psi0-consistency={pe:.2e}")

    if args.out_csv:
        import csv
        with open(args.out_csv, 'w', newline='') as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            wcsv.writeheader()
            for row in rows_csv:
                wcsv.writerow(row)
        print(f"\n[triage] wrote {args.out_csv}")


if __name__ == '__main__':
    main()
