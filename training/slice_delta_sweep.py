"""
slice_delta_sweep.py

Slice a delta sweep out of a packed M-mark fine trajectory -- NO new RK4. The
saved snapshots ARE the RK4-ultrafine flow at dt_base (= manifest Delta_T) marks,
so for a coarse step the fine reference is a *later saved snapshot*:

    ascending-time marks w[0..M-1]  (w[M-1]=omega_0 latest, w[0]=seed earliest)
    Delta_T_c = j*dt_base, anchor a:
        delta = w[a+j] - AB2CN2(w[a], w[a-j]; Delta_T_c)        # one cheap AB2 step
        input stencil (S snaps): w[a], w[a-j], ..., w[a-(S-1)j]

Works on BOTH:
  * the existing 7-mark ensemble (M=7) -> free sweep {1e-3,2e-3,3e-3} (j=1,2,3, S=1)
  * a deep build (--Delta-T 5e-3 --n-marks 25) -> {5e-3,1e-2,1.5e-2}, S up to 7.

Anchor window: max((S-1)*j, j) <= a <= M-1-j.  S=1 (default) maximizes anchors and
is enough for delta-R (delta is a spatial fn of omega_a; condition on Delta_T via
the regime vector). S>1 only fits a deep enough M.

Output per Delta_T_c: <member>/sweep_dT_<tag>/packed/{inputs.npy(float32,2S ch),
delta_f64.npy(float64,1 ch)} + manifest.json + split.npz. Consumed unchanged by
split_ensemble.py / concat_dataset.py.

Usage:
    python slice_delta_sweep.py <ENSEMBLE_ROOT> --js 1,2,3                 # free sweep
    python slice_delta_sweep.py <DEEP_ROOT>     --js 1,2,3 --n-snapshots 1 # unstable
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from qg.solver.grid.cartesian import CartesianGrid
from qg.solver.opt.derivative import Derivative
from qg.solver.opt.basis import to_spectral, to_physical
from build_training_data_mmap import (ab2cn2_step_spectral, rk4_step,
                                       build_L_hat, make_input_fields)


def build_F_phys(forcing, Lx, Ly, Nx, Ny, device, dtype):
    if not forcing or not isinstance(forcing, dict):
        return None
    A = float(forcing.get('A', 0.0)); B = float(forcing.get('B', 0.0))
    D = float(forcing.get('D', 0.0)); E = float(forcing.get('E', 0.0))
    x = torch.linspace(0, Lx, Nx, device=device, dtype=dtype)
    y = torch.linspace(0, Ly, Ny, device=device, dtype=dtype)
    return (A * torch.cos(B * x[None, :]) + D * torch.cos(E * y[:, None]))[None]


def setup_solver(man, device):
    Nx, Ny = int(man['Nx']), int(man['Ny']); Lx, Ly = float(man['Lx']), float(man['Ly'])
    grid = CartesianGrid(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, device=device, precision='float64')
    der = Derivative(grid)
    for a in ('dx', 'dy', 'laplacian', 'inv_laplacian'):
        setattr(der, a, getattr(der, a).to(device))
    L_hat = build_L_hat(der, nu=float(man['nu']), mu=float(man['mu']),
                        B=float(man['beta'])).to(device)
    F = build_F_phys(man.get('forcing'), Lx, Ly, Nx, Ny, device, torch.float64)
    return der, L_hat, F


def dt_tag(dt):
    """Filename-safe, round-trip-unambiguous tag: 5e-3->'5em3', 1e-2->'1em2',
    1.5e-2->'1p5em2'. (A plain '.0e' format would round 1.5e-2 to '2e-02'.)"""
    mant, exp = f'{dt:.10e}'.split('e')
    mant = mant.rstrip('0').rstrip('.')
    return mant.replace('.', 'p') + 'em' + str(-int(exp))


def slice_member(root: Path, js, S: int, n_anchors, chunk: int, device: str):
    man = json.loads((root / 'manifest.json').read_text())
    der, L_hat, F = setup_solver(man, device)
    dt_base = float(man['Delta_T'])
    M = int(man.get('n_snapshots_per_sample', 7))
    Ny, Nx = int(man['Ny']), int(man['Nx'])
    om_ch = lambda k: (M - 1) - k                 # mark k (ascending) -> omega channel
    ps_ch = lambda k: (2 * M - 1) - k             # mark k             -> psi channel
    inp = np.load(root / 'packed' / 'inputs.npy', mmap_mode='r')   # (Nsrc, 2M, Ny, Nx)
    Nsrc = inp.shape[0]
    label = (Path(man.get('source_omega_path', '')).parts[-2]
             if man.get('source_omega_path') else root.parent.name)
    out_fields = make_input_fields(S)              # [omega_0..m{S-1}, psi_0..m{S-1}]

    for j in js:
        a_lo = max((S - 1) * j, j); a_hi = M - 1 - j
        anchors = list(range(a_lo, a_hi + 1))
        if not anchors:
            print(f"  [{label}] j={j} S={S}: no anchors in M={M} (need a in "
                  f"[{a_lo},{a_hi}]), skip"); continue
        if n_anchors and n_anchors < len(anchors):     # evenly-spaced subsample
            sel = np.linspace(0, len(anchors) - 1, n_anchors).round().astype(int)
            anchors = [anchors[i] for i in sorted(set(sel.tolist()))]
        dt_c = j * dt_base
        Nout = Nsrc * len(anchors)
        out_dir = root.parent / f'sweep_dT_{dt_tag(dt_c)}'
        pdir = out_dir / 'packed'; pdir.mkdir(parents=True, exist_ok=True)
        x_mm = open_memmap(pdir / 'inputs.npy', mode='w+', dtype=inp.dtype,
                           shape=(Nout, 2 * S, Ny, Nx))   # preserve build's input dtype
        de_mm = open_memmap(pdir / 'delta_exact_f64.npy', mode='w+', dtype=np.float64,
                            shape=(Nout, 1, Ny, Nx))   # ref = fine RK4 = w[a+j] ~ exact
        dr_mm = open_memmap(pdir / 'delta_rk4_f64.npy', mode='w+', dtype=np.float64,
                            shape=(Nout, 1, Ny, Nx))    # ref = coarse RK4 at dt_c
        rms_e = rms_r = 0.0; w = 0
        for ai, a in enumerate(anchors):
            for s0 in range(0, Nsrc, chunk):
                e = min(s0 + chunk, Nsrc)
                wa = torch.as_tensor(np.ascontiguousarray(inp[s0:e, om_ch(a)]),
                                     dtype=torch.float64, device=device)
                wp = torch.as_tensor(np.ascontiguousarray(inp[s0:e, om_ch(a - j)]),
                                     dtype=torch.float64, device=device)
                wref = torch.as_tensor(np.ascontiguousarray(inp[s0:e, om_ch(a + j)]),
                                       dtype=torch.float64, device=device)
                om_coarse = to_physical(ab2cn2_step_spectral(
                    to_spectral(wa), to_spectral(wp), dt_c, der, L_hat, F))   # AB2 @ dt_c
                om_rk4 = rk4_step(wa, dt_c, der, L_hat, F)                     # 1 RK4 @ dt_c
                delta_exact = wref - om_coarse        # tau_AB2 (ref ~ exact): R3,R4,R5,...
                delta_rk4 = om_rk4 - om_coarse        # tau_AB2 - tau_RK4   : R3,R4,R5-T5,...
                rows = slice(ai * Nsrc + s0, ai * Nsrc + e)
                for s in range(S):                       # stencil at dt_c spacing
                    x_mm[rows, s] = inp[s0:e, om_ch(a - s * j)]
                    x_mm[rows, S + s] = inp[s0:e, ps_ch(a - s * j)]
                de_mm[rows, 0] = delta_exact.detach().cpu().numpy()
                dr_mm[rows, 0] = delta_rk4.detach().cpu().numpy()
                rms_e += float((delta_exact ** 2).sum().item())
                rms_r += float((delta_rk4 ** 2).sum().item()); w += (e - s0) * Ny * Nx
        x_mm.flush(); de_mm.flush(); dr_mm.flush()
        ntr = int(round(Nsrc * 0.70)); nva = int(round(Nsrc * 0.15))
        rep = lambda lo, hi: np.concatenate(
            [np.arange(ai * Nsrc + lo, ai * Nsrc + hi) for ai in range(len(anchors))])
        tr, va, te = rep(0, ntr), rep(ntr, ntr + nva), rep(ntr + nva, Nsrc)
        for tgt in (out_dir / 'split.npz', pdir / 'split.npz'):
            np.savez(tgt, train_idx=tr.astype(np.int32),
                     val_idx=va.astype(np.int32), test_idx=te.astype(np.int32))
        rms_e = float(np.sqrt(rms_e / w)); rms_r = float(np.sqrt(rms_r / w))
        m = dict(man)
        m.update(Delta_T=dt_c, n_snapshots_per_sample=S, input_fields=out_fields,
                 target_fields=['delta'], n_total=int(Nout),
                 n_train=int(tr.size), n_val=int(va.size), n_test=int(te.size),
                 sliced_from=str(root), sliced_j=j, sliced_S=S, anchors=anchors,
                 dt_base=dt_base,
                 extra_targets={
                     'exact': dict(file='packed/delta_exact_f64.npy', dtype='float64',
                                   shape=[Nout, 1, Ny, Nx], rms=rms_e,
                                   definition='w[a+j] - AB2CN2(w[a],w[a-j];dt_c)  '
                                              '[ref=fine RK4~exact; R3,R4,R5,...]'),
                     'rk4':   dict(file='packed/delta_rk4_f64.npy', dtype='float64',
                                   shape=[Nout, 1, Ny, Nx], rms=rms_r,
                                   definition='RK4(w[a];dt_c) - AB2CN2(w[a],w[a-j];dt_c)  '
                                              '[ref=coarse RK4; R3,R4,R5-T5,...]')})
        (out_dir / 'manifest.json').write_text(json.dumps(m, indent=2))
        print(f"  [{label}] dT={dt_c:g} (j={j},S={S}): {len(anchors)} anchors x {Nsrc} "
              f"= {Nout}  |d_exact|={rms_e:.3e} |d_rk4|={rms_r:.3e}  -> {out_dir.name}")


def find_members(ensemble_root: Path):
    return sorted({m.parent for m in ensemble_root.rglob('manifest.json')
                   if (m.parent / 'packed' / 'inputs.npy').exists()
                   and 'sweep_dT_' not in str(m.parent)})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path, help='ensemble root OR a single member dir')
    ap.add_argument('--js', type=str, default='1,2,3')
    ap.add_argument('--n-snapshots', type=int, default=1,
                    help='stencil depth S (default 1; works on M=7 free + deep builds)')
    ap.add_argument('--n-anchors', type=int, default=None,
                    help='cap anchors/dT to N evenly-spaced (default: all). Controls the '
                         'data-multiplication vs storage tradeoff; anchors are correlated.')
    ap.add_argument('--members', type=str, default=None)
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()
    js = [int(x) for x in args.js.split(',')]
    want = set(s.strip() for s in args.members.split(',')) if args.members else None
    roots = ([args.root] if (args.root / 'packed' / 'inputs.npy').exists()
             else find_members(args.root))
    print(f"[slice] members={len(roots)} js={js} S={args.n_snapshots}")
    for root in roots:
        man = json.loads((root / 'manifest.json').read_text()) if (root / 'manifest.json').exists() else {}
        lab = (Path(man.get('source_omega_path', '')).parts[-2]
               if man.get('source_omega_path') else root.parent.name)
        if want and lab not in want:
            continue
        slice_member(root, js, args.n_snapshots, args.n_anchors, args.chunk, args.device)
    print("[slice] done.")


if __name__ == '__main__':
    main()